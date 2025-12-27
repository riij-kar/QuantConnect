from collections import deque
from typing import List, Optional, Dict, Any
import os
import json
from datetime import datetime
from QuantConnect.Data.Market import TradeBar

from utils.pattern_detectors.double_top_detector import DoubleTopDetector
from utils.shared import resolve_log_dir

class PriceActionPatterns:
    """
    Quantitative framework for identifying geometric price action patterns.
    Uses a segmented approach to validate patterns against specific definitions.
    """

    def __init__(self, 
                 algorithm,
                 lookback_period: int = 100, 
                 window: Optional[Any] = None, 
                 indicators=None,
                 algo_dir: str = None):
        """
        Initialization for price action pattern recognition.
        
        Parameters
        ----------
        algorithm : QCAlgorithm
            The algorithm instance for logging and context.
        lookback_period : int
            Size of the rolling window (N) for identifying local extrema.
        window : Optional[Iterable[TradeBar]]
            Optional external RollingWindow shared with the alpha. When provided,
            the pattern engine will read history directly from this buffer.
        indicators : Optional[Dict[str, Any]]
            Optional dictionary of indicators used for volatility or trend analysis.
        algo_dir : str
            Root directory of the algorithm to resolve backtest logs.
        """
        self.algorithm = algorithm
        self.indicators = indicators or {}
        atr_indicator = self.indicators.get("atr") if isinstance(self.indicators, dict) else None
        rsi_indicator = self.indicators.get("rsi") if isinstance(self.indicators, dict) else None
        self.lookback_period = lookback_period
        
        # Simplify window management: always use self.window
        if window is None:
            self.window = deque(maxlen=self.lookback_period)
        else:
            self.window = window

        # Stateful containers for individual pattern detectors
        pivot_capacity = max(6, min(50, lookback_period // 4 or 6))
        self._double_top_detector = DoubleTopDetector(
            atr_indicator=atr_indicator,
            rsi_indicator=rsi_indicator,
            pivot_capacity=pivot_capacity,
        )
        
        # Logging setup
        self.log_entries = []
        
        # Resolve log directory similar to CandlestickPatternManager
        if algo_dir:
            self.log_dir = resolve_log_dir(algo_dir, "price_actions")
            self.log_path = os.path.join(self.log_dir, "price_actions.json")
        else:
            self.log_dir = None
            self.log_path = None

    def flush(self) -> None:
        """Write collected pattern logs to disk."""
        if not self.log_entries or not self.log_path:
            return
        
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Prepare for JSON serialization (handle datetime)
            serializable_entries = []
            for entry in self.log_entries:
                serializable_entry = {}
                for k, v in entry.items():
                    if hasattr(v, 'isoformat'):
                        serializable_entry[k] = v.isoformat()
                    else:
                        serializable_entry[k] = v
                serializable_entries.append(serializable_entry)

            with open(self.log_path, "w", encoding="utf-8") as handle:
                json.dump(serializable_entries, handle, indent=4)
                
            if self.algorithm:
                self.algorithm.Debug(f"PriceActionPatterns: Pattern log written to {self.log_path}")
        except Exception as e:
            if self.algorithm:
                self.algorithm.Debug(f"PriceActionPatterns: Error flushing logs: {e}")

    def _log_pattern(self, pattern: Dict[str, Any]):
        """Internal method to buffer a detected pattern."""
        if pattern:
            self.log_entries.append(pattern)
    
    def add_trade_bar(self, trade_bar: TradeBar):
        """
        Adds a TradeBar to the internal history for pattern recognition.
        Only adds if using internal deque storage.
        """
        if isinstance(self.window, deque):
            self.window.append(trade_bar)

    def _get_history_sequence(self) -> List[TradeBar]:
        """Return TradeBars in chronological order regardless of storage backend."""
        if self.window is None:
            return []
            
        if isinstance(self.window, deque):
            return list(self.window)

        # QuantConnect RollingWindow yields newest -> oldest; reverse for chronological order
        try:
            count = getattr(self.window, "Count", None)
            if count is None:
                sequence = list(self.window)
            else:
                sequence = [self.window[i] for i in range(count)]
        except Exception:
            sequence = list(self.window)
        return list(reversed(sequence))


    def is_ready(self) -> bool:
        """
        Checks if sufficient data exists for pattern recognition.
        """
        if self.window is None:
            return False
            
        if isinstance(self.window, deque):
            return len(self.window) == self.window.maxlen
            
        # RollingWindow reports readiness via IsReady property
        return getattr(self.window, "IsReady", False)
        # double top and double bottom pattern
    
    def calculate_double_tops(self) -> List[Dict[str, Any]]:
        """
        Identify Double Tops (M-shape) and expose the pattern states.
        
        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries indicating the specific state of the pattern.
        Core Logic:
            1.  Use ATR Reversal (ZigZag Logic)
        """
        if not self.is_ready():
            return []
        if self._double_top_detector is None:
            return []
        history_sequence = self._get_history_sequence()
        if not history_sequence:
            return []

        latest_bar = history_sequence[-1]
        pattern = self._double_top_detector.update(latest_bar)
        if pattern:
            self._log_pattern(pattern)
        return [pattern] if pattern else []
    
    def calculate_double_bottoms(self) -> List[str]:
        """
        Identify Double Bottoms (W-shape).
        
        Returns
        -------
        List[str]
            A list of detected patterns (e.g., ['Double_Top']).
        """
        if not self.is_ready():
            return []
            
        patterns_found = []
        
        # 1. Convert history to a workable list/series
        history = self._get_history_sequence()
        
        # 2. Implement Segmentation Logic Here
        # Example: Find local maxima/minima to define peaks and troughs
        
        # 3. Geometric checks for M-shape or W-shape
        
        return patterns_found
    
    def calculate_continuation_pattern(self, data: List[TradeBar]):
        """
        Identify continuation patterns indicating the trend is likely to resume.
        """
        pass

    def calculate_price_channel(self, data: List[TradeBar]):
        """
        Identify parallel resistance and support lines using linear regression 
        or local extrema analysis on the TradeBar series.
        """
        pass

    def calculate_triple_tops_bottoms(self, data: List[TradeBar]):
        """
        Identify Triple Tops and Triple Bottoms based on three localized extrema testing a support/resistance level.
        """
        pass

    def calculate_head_and_shoulders(self, data: List[TradeBar]):
        """
        Identify Head and Shoulders (Top and Inverse).
        Requires detection of three peaks with the central peak (Head) being the highest/lowest.
        """
        pass

    def calculate_flag(self, data: List[TradeBar]):
        """
        Identify Flag pattern: a sharp counter-trend consolidation channel following a strong directional move.
        """
        pass

    def calculate_pennant(self, data: List[TradeBar]):
        """
        Identify Pennant pattern: similar to flags but with converging trend lines (triangle-like consolidation).
        """
        pass

    def calculate_triangle(self, data: List[TradeBar]):
        """
        Identify Triangle patterns (Ascending, Descending, Symmetrical) by calculating the slope convergence of pivot points.
        """
        pass

    def calculate_wedge(self, data: List[TradeBar]):
        """
        Identify Wedge patterns (Rising, Falling) where trend lines converge in the same direction.
        """
        pass
    
    def calculate_cup_and_handle(self, data: List[TradeBar]):
        """
        Identify Cup and Handle pattern: a "U" shape recovery followed by a slight drift downward (handle).
        """
        pass