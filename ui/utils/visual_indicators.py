from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import talib

__all__ = ["IndicatorBundle", "compute_visual_indicators"]


MA_TYPE_MAP = {
    "SMA": talib.MA_Type.SMA,
    "EMA": talib.MA_Type.EMA,
    "WMA": talib.MA_Type.WMA,
    "DEMA": talib.MA_Type.DEMA,
    "TEMA": talib.MA_Type.TEMA,
    "TRIMA": talib.MA_Type.TRIMA,
    "KAMA": talib.MA_Type.KAMA,
    "MAMA": talib.MA_Type.MAMA,
}


@dataclass
class IndicatorBundle:
    """Bundle of indicator overlays, oscillators, and informational messages."""
    overlays: Dict[str, Any]
    oscillators: Dict[str, Dict[str, Any]]
    messages: List[str]

    @classmethod
    def empty(cls) -> "IndicatorBundle":
        """Return an empty bundle used when indicator computation is skipped."""
        return cls({}, {}, [])


def _as_series(values: np.ndarray, index: pd.Index, name: str) -> pd.Series:
    """Create a pandas Series with the supplied index and name."""
    return pd.Series(values, index=index, name=name)


def _to_float_array(series: pd.Series) -> np.ndarray:
    """Coerce a pandas Series into a float64 numpy array, preserving NaNs."""
    numeric = pd.to_numeric(series, errors="coerce")
    return np.asarray(numeric.to_numpy(dtype="float64"), dtype="float64")


def _resolve_ma_type(name: Optional[str]) -> int:
    """Translate a moving-average name into the TALib MA type enum."""
    if not name:
        return talib.MA_Type.SMA
    return MA_TYPE_MAP.get(str(name).upper(), talib.MA_Type.SMA)


def _compute_supertrend(df: pd.DataFrame, period: int, multiplier: float) -> Dict[str, Any]:
    """Compute Supertrend bands and trend series for the supplied OHLC data."""
    if any(col not in df.columns for col in ("high", "low", "close")):
        return {}
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    high_arr = _to_float_array(high)
    low_arr = _to_float_array(low)
    close_arr = _to_float_array(close)
    atr = _as_series(
        talib.ATR(high_arr, low_arr, close_arr, timeperiod=period),
        df.index,
        "ATR"
    )
    if atr.isna().all():
        return {}
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    for i in range(1, len(df)):
        prev_upper = final_upper.iloc[i - 1]
        prev_lower = final_lower.iloc[i - 1]
        upper_candidate = basic_upper.iloc[i]
        lower_candidate = basic_lower.iloc[i]

        if not pd.isna(upper_candidate) and (pd.isna(prev_upper) or upper_candidate < prev_upper or close.iloc[i - 1] > prev_upper):
            final_upper.iloc[i] = upper_candidate
        else:
            final_upper.iloc[i] = prev_upper

        if not pd.isna(lower_candidate) and (pd.isna(prev_lower) or lower_candidate > prev_lower or close.iloc[i - 1] < prev_lower):
            final_lower.iloc[i] = lower_candidate
        else:
            final_lower.iloc[i] = prev_lower

    final_upper = final_upper.ffill()
    final_lower = final_lower.ffill()
    start_idx = final_upper.first_valid_index()
    if start_idx is None:
        return {}
    start_pos = final_upper.index.get_loc(start_idx)

    supertrend = pd.Series(np.nan, index=df.index, dtype=float)
    initial_upper = final_upper.iloc[start_pos]
    initial_lower = final_lower.iloc[start_pos]
    supertrend.iloc[start_pos] = initial_upper if close.iloc[start_pos] <= initial_upper else initial_lower
    for i in range(start_pos + 1, len(df)):
        prev = supertrend.iloc[i - 1]
        prev_upper = final_upper.iloc[i - 1]
        prev_lower = final_lower.iloc[i - 1]
        current_upper = final_upper.iloc[i]
        current_lower = final_lower.iloc[i]

        if prev == prev_upper:
            supertrend.iloc[i] = current_upper if close.iloc[i] <= current_upper else current_lower
        else:
            supertrend.iloc[i] = current_lower if close.iloc[i] >= current_lower else current_upper

    return {
        f"Supertrend ({period}, {multiplier})": {
            "type": "supertrend",
            "period": period,
            "multiplier": multiplier,
            "upper": final_upper,
            "lower": final_lower,
            "trend": supertrend,
        }
    }


def _compute_vwap(df: pd.DataFrame, period: int, source: str) -> Optional[pd.Series]:
    """Calculate a rolling VWAP series using the requested price source."""
    if "volume" not in df.columns:
        return None
    volume = df["volume"].astype(float)
    if source.lower() in {"hlc3", "typical"}:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    elif source.lower() == "close":
        typical_price = df["close"]
    else:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0

    tp_vol = typical_price * volume
    rolling_tp_vol = tp_vol.rolling(period).sum()
    rolling_vol = volume.rolling(period).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = rolling_tp_vol / rolling_vol
    return vwap.rename(f"VWAP ({period})")


def compute_visual_indicators(price_df: Optional[pd.DataFrame], indicator_config: Optional[Dict[str, Any]]) -> IndicatorBundle:
    """Build indicator overlays/oscillators based on the dashboard configuration."""
    if price_df is None or price_df.empty or not indicator_config:
        return IndicatorBundle.empty()

    overlays: Dict[str, pd.Series] = {}
    oscillators: Dict[str, Dict[str, Any]] = {}
    messages: List[str] = []

    close = price_df.get("close")
    if close is None:
        messages.append("Price data missing 'close' column; indicators skipped.")
        return IndicatorBundle(overlays, oscillators, messages)
    close = pd.to_numeric(close, errors="coerce")
    close_arr = _to_float_array(close)

    high = price_df.get("high")
    low = price_df.get("low")
    high_series = pd.to_numeric(high, errors="coerce") if high is not None else None
    low_series = pd.to_numeric(low, errors="coerce") if low is not None else None
    high_arr = _to_float_array(high_series) if high_series is not None else None
    low_arr = _to_float_array(low_series) if low_series is not None else None

    # Simple Moving Averages
    ma_periods = indicator_config.get("moving-average", []) or []
    for period in ma_periods:
        try:
            period_int = int(period)
        except (TypeError, ValueError):
            continue
        values = talib.SMA(close_arr, timeperiod=period_int)
        overlays[f"SMA ({period_int})"] = _as_series(values, price_df.index, f"SMA ({period_int})")

    # Exponential Moving Averages
    ema_periods = indicator_config.get("exponential-moving-average", []) or []
    for period in ema_periods:
        try:
            period_int = int(period)
        except (TypeError, ValueError):
            continue
        values = talib.EMA(close_arr, timeperiod=period_int)
        overlays[f"EMA ({period_int})"] = _as_series(values, price_df.index, f"EMA ({period_int})")

    # Bollinger Bands
    bb_cfg = indicator_config.get("bollinger-bands") or {}
    if bb_cfg:
        period = int(bb_cfg.get("period", 20))
        std_dev = float(bb_cfg.get("std_dev", 2))
        ma_type = _resolve_ma_type(bb_cfg.get("ma_type"))
        upper, middle, lower = talib.BBANDS(
            close_arr,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=ma_type
        )
        if bb_cfg.get("upper_band", True):
            overlays[f"BB Upper ({period})"] = _as_series(upper, price_df.index, "BB Upper")
        if bb_cfg.get("middle_band", True):
            overlays[f"BB Middle ({period})"] = _as_series(middle, price_df.index, "BB Middle")
        if bb_cfg.get("lower_band", True):
            overlays[f"BB Lower ({period})"] = _as_series(lower, price_df.index, "BB Lower")

    # VWAP
    vwap_cfg = indicator_config.get("vwap") or {}
    if vwap_cfg:
        period = int(vwap_cfg.get("period", 30))
        source = vwap_cfg.get("source", "hlc3")
        vwap_series = _compute_vwap(price_df, period, source)
        if vwap_series is None:
            messages.append("VWAP skipped: 'volume' column missing.")
        else:
            overlays[vwap_series.name] = vwap_series

    # Supertrend
    supertrend_cfg = indicator_config.get("supertrend") or {}
    if supertrend_cfg:
        if high_series is None or low_series is None:
            messages.append("Supertrend skipped: high/low columns missing.")
        else:
            period = int(supertrend_cfg.get("period", 10))
            multiplier = float(supertrend_cfg.get("multiplier", 3.0))
            supertrend_input = price_df.copy()
            supertrend_input["high"] = high_series
            supertrend_input["low"] = low_series
            supertrend_input["close"] = close
            supertrend_series = _compute_supertrend(supertrend_input, period, multiplier)
            overlays.update(supertrend_series)

    # RSI
    rsi_cfg = indicator_config.get("rsi") or {}
    if rsi_cfg:
        period = int(rsi_cfg.get("period", 14))
        rsi_values = talib.RSI(close_arr, timeperiod=period)
        levels: List[tuple[str, float]] = []
        overbought = rsi_cfg.get("overbought")
        oversold = rsi_cfg.get("oversold")
        try:
            if overbought is not None:
                levels.append(("Overbought", float(overbought)))
        except (TypeError, ValueError):
            messages.append("RSI overbought level invalid; skipping line.")
        try:
            if oversold is not None:
                levels.append(("Oversold", float(oversold)))
        except (TypeError, ValueError):
            messages.append("RSI oversold level invalid; skipping line.")
        oscillators[f"RSI ({period})"] = {
            "type": "line",
            "series": _as_series(rsi_values, price_df.index, "RSI"),
            "levels": levels,
        }

    # MACD
    macd_cfg = indicator_config.get("macd") or {}
    if macd_cfg:
        fast = int(macd_cfg.get("fast_period", 12))
        slow = int(macd_cfg.get("slow_period", 26))
        signal = int(macd_cfg.get("signal_period", 9))
        macd, signal_line, hist = talib.MACD(
            close_arr,
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal
        )
        name = f"MACD ({fast},{slow},{signal})"
        oscillators[name] = {
            "type": "macd",
            "macd": _as_series(macd, price_df.index, "MACD"),
            "signal": _as_series(signal_line, price_df.index, "Signal"),
            "hist": _as_series(hist, price_df.index, "Histogram")
        }

    # ATR
    atr_cfg = indicator_config.get("atr") or {}
    if atr_cfg:
        if high_series is None or low_series is None or high_arr is None or low_arr is None:
            messages.append("ATR skipped: high/low columns missing.")
        else:
            period = int(atr_cfg.get("period", 14))
            atr_values = talib.ATR(high_arr, low_arr, close_arr, timeperiod=period)
            oscillators[f"ATR ({period})"] = {
                "type": "line",
                "series": _as_series(atr_values, price_df.index, "ATR")
            }

    return IndicatorBundle(overlays, oscillators, messages)
