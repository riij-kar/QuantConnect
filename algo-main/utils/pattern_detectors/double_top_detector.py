from collections import deque
from typing import Optional, Dict, Any
from QuantConnect.Data.Market import TradeBar


class _DoubleTopState:
    """Container for tracking zigzag pivots while scanning for double tops."""

    def __init__(
        self,
        pivot_capacity: int = 12,
        reversal_multiplier: float = 2.0,
        match_multiplier: float = 0.5,
        depth_multiplier: float = 1.0,
    ):
        self.trend = "UP"
        self.last_high = float("-inf")
        self.last_low = float("inf")
        self.last_high_time = None
        self.last_low_time = None
        self.last_high_volume = 0.0
        self.last_high_rsi = 0.0
        self.pivots = deque(maxlen=pivot_capacity)
        self.last_processed_time = None
        self.k_reversal = reversal_multiplier
        self.k_match = match_multiplier
        self.k_depth = depth_multiplier
        self.potential_pattern = None
        self.confirmed_pattern = None


class DoubleTopDetector:
    """ATR-driven zigzag detector that emits structured double-top patterns."""

    def __init__(
        self,
        atr_indicator=None,
        rsi_indicator=None,
        pivot_capacity: int = 12,
        reversal_multiplier: float = 2.0,
        match_multiplier: float = 0.5,
        depth_multiplier: float = 1.0,
    ):
        self.atr_indicator = atr_indicator
        self.rsi_indicator = rsi_indicator
        self.state = _DoubleTopState(
            pivot_capacity=pivot_capacity,
            reversal_multiplier=reversal_multiplier,
            match_multiplier=match_multiplier,
            depth_multiplier=depth_multiplier,
        )

    def update(self, bar: TradeBar) -> Optional[Dict[str, Any]]:
        """Process the latest TradeBar and emit a pattern dictionary when found."""
        atr_value = self._get_indicator_value(self.atr_indicator)
        rsi_value = self._get_indicator_value(self.rsi_indicator)
        
        if atr_value is None or atr_value <= 0:
            return None
        bar_time = self._extract_bar_time(bar)
        state = self.state
        if state.last_processed_time is not None and bar_time is not None:
            if bar_time <= state.last_processed_time:
                return None
        pattern = self._update_state(bar, atr_value, rsi_value, state, bar_time)
        if bar_time is not None:
            state.last_processed_time = bar_time
            
        return pattern

    @staticmethod
    def _extract_bar_time(bar: TradeBar):
        return getattr(bar, "EndTime", None) or getattr(bar, "Time", None)

    @staticmethod
    def _get_indicator_value(indicator) -> Optional[float]:
        if indicator is None:
            return None
        if hasattr(indicator, "IsReady") and not indicator.IsReady:
            return None
        current = getattr(indicator, "Current", None)
        if current is None:
            return None
        value = getattr(current, "Value", None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _update_state(
        self,
        bar: TradeBar,
        atr_value: float,
        rsi_value: Optional[float],
        state: _DoubleTopState,
        bar_time,
    ) -> Optional[Dict[str, Any]]:
        reversal_zone = atr_value * state.k_reversal
        high = float(getattr(bar, "High", getattr(bar, "Close", 0.0)))
        low = float(getattr(bar, "Low", getattr(bar, "Close", 0.0)))
        close = float(getattr(bar, "Close", getattr(bar, "High", 0.0)))
        volume = float(getattr(bar, "Volume", 0.0))
        symbol = getattr(bar, "Symbol", None)

        # Check for Retest if a confirmed pattern exists
        if state.confirmed_pattern is not None:
            retested_pattern = self._is_retested_neckline(state, high, close, bar_time)
            if retested_pattern:
                return retested_pattern
            
            # Invalidate if price goes above Peak 2
            if close > state.confirmed_pattern["peak_2_price"]:
                state.confirmed_pattern = None

        # Check for Neckline Break if a potential pattern exists
        if state.potential_pattern is not None:
            confirmed_pattern = self._is_neckline_broken(state, close, volume, bar_time)
            if confirmed_pattern:
                return confirmed_pattern
            
            # Invalidate if price goes above Peak 2 (or Peak 1) significantly? 
            # For now, let's keep it simple. If it makes a new high, the zigzag logic will handle it.

        if state.trend == "UP":
            if high > state.last_high:
                state.last_high = high
                state.last_high_time = bar_time
                state.last_high_volume = volume
                if rsi_value is not None:
                    state.last_high_rsi = rsi_value
            elif close < (state.last_high - reversal_zone):
                pivot_time = state.last_high_time or bar_time
                state.pivots.append(
                    {
                        "price": state.last_high,
                        "type": "PEAK",
                        "time": pivot_time,
                        "symbol": symbol,
                        "volume": state.last_high_volume,
                        "rsi": state.last_high_rsi
                    }
                )
                state.trend = "DOWN"
                state.last_low = low
                state.last_low_time = bar_time
                state.last_high = float("-inf")
                state.last_high_time = None
                
                # Check for potential pattern formation
                potential = self._evaluate_pattern(state, atr_value)
                if potential:
                    state.potential_pattern = potential
                    # Don't return yet, wait for confirmation
                
                return None
        else:
            if low < state.last_low:
                state.last_low = low
                state.last_low_time = bar_time
            elif close > (state.last_low + reversal_zone):
                pivot_time = state.last_low_time or bar_time
                state.pivots.append(
                    {
                        "price": state.last_low,
                        "type": "TROUGH",
                        "time": pivot_time,
                        "symbol": symbol,
                    }
                )
                state.trend = "UP"
                state.last_high = high
                state.last_high_time = bar_time
                state.last_low = float("inf")
                state.last_low_time = None
        return None

    def _is_neckline_broken(
        self,
        state: _DoubleTopState,
        close: float,
        volume: float,
        bar_time,
    ) -> Optional[Dict[str, Any]]:
        """Check if the price has broken below the neckline (trough price)."""
        trough_price = state.potential_pattern["trough_price"]
        if close < trough_price:
            # Neckline Broken - Confirm Pattern
            pattern = state.potential_pattern.copy()
            pattern["pattern_confirmed"] = True
            pattern["breakdown_time"] = bar_time
            pattern["breakdown_price"] = close
            pattern["breakdown_volume"] = volume

            # Check Divergences
            peak1_vol = pattern.get("peak_1_volume", 0)
            peak2_vol = pattern.get("peak_2_volume", 0)
            peak1_rsi = pattern.get("peak_1_rsi", 0)
            peak2_rsi = pattern.get("peak_2_rsi", 0)

            pattern["volume_divergence"] = (peak2_vol < peak1_vol) and (volume > peak2_vol)
            pattern["rsi_divergence"] = (peak2_rsi < peak1_rsi) if (peak1_rsi and peak2_rsi) else False
            #current close decisively broken below neckline with huge volume
            pattern['strong_breakdown_volume_gt_peak_1_2'] = volume > peak2_vol > peak1_vol
            state.potential_pattern = None  # Reset
            state.confirmed_pattern = pattern # Save for retest check
            return pattern
        return None

    def _is_retested_neckline(
        self,
        state: _DoubleTopState,
        high: float,
        close: float,
        bar_time,
    ) -> Optional[Dict[str, Any]]:
        """Check if the price has retested the neckline after a break."""
        pattern = state.confirmed_pattern
        trough_price = pattern["trough_price"]
        
        # Retest logic: Price touches the neckline from below.
        # We use High >= Trough Price as the trigger.
        # We also ensure it hasn't invalidated (Close > Peak 2) which is checked in update loop.
        
        if high >= trough_price:
             # Retest Confirmed
             pattern = pattern.copy()
             pattern["pattern_retested"] = True
             pattern["retest_time"] = bar_time
             pattern["retest_price"] = high
             
             state.confirmed_pattern = None # Consumed
             return pattern
             
        return None

    def _evaluate_pattern(self, state: _DoubleTopState, atr_value: float) -> Optional[Dict[str, Any]]:
        if len(state.pivots) < 3:
            return None
        peak2 = state.pivots[-1]
        trough = state.pivots[-2]
        peak1 = state.pivots[-3]
        if not (
            peak1["type"] == "PEAK"
            and trough["type"] == "TROUGH"
            and peak2["type"] == "PEAK"
        ):
            return None

        # Ensure all pivots occurred on the same calendar day
        t1 = peak1.get("time")
        t_trough = trough.get("time")
        t2 = peak2.get("time")

        if t1 is not None and t_trough is not None and t2 is not None:
            if not (t1.date() == t_trough.date() == t2.date()):
                return None

        diff = abs(peak1["price"] - peak2["price"])
        drop = peak1["price"] - trough["price"]
        if diff > atr_value * state.k_match:
            return None
        if drop < atr_value * state.k_depth:
            return None

        symbol_obj = peak2.get("symbol") or peak1.get("symbol") or trough.get("symbol")
        symbol = None
        if symbol_obj is not None:
            value_attr = getattr(symbol_obj, "Value", None)
            symbol = value_attr if value_attr is not None else str(symbol_obj)
        return {
            "pattern_name": "Double_Top",
            "peak_1_price": peak1["price"],
            "peak_2_price": peak2["price"],
            "trough_price": trough["price"],
            "peak_1_time": peak1.get("time"),
            "peak_2_time": peak2.get("time"),
            "trough_time": trough.get("time"),
            "peak_difference": diff,
            "trough_depth": drop,
            "diff_atr_ratio": diff / atr_value if atr_value else None,
            "depth_atr_ratio": drop / atr_value if atr_value else None,
            "k_reversal": state.k_reversal,
            "k_match": state.k_match,
            "k_depth": state.k_depth,
            "symbol": symbol,
            "peak_1_volume": peak1.get("volume"),
            "peak_2_volume": peak2.get("volume"),
            "peak_1_rsi": peak1.get("rsi"),
            "peak_2_rsi": peak2.get("rsi"),
        }
