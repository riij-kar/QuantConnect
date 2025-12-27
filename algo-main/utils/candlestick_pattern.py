import os
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from QuantConnect.Data.Consolidators import TradeBarConsolidator
from QuantConnect.Data.Market import TradeBar
from QuantConnect.Indicators.CandlestickPatterns import *  # noqa: F401,F403
from collections import deque
from utils.shared import resolve_log_dir

RAW_PATTERN_NAMES: List[str] = [
    "Abandoned Baby",
    "Advance Block",
    "Belt Hold",
    "Breakaway",
    "Closing Marubozu",
    "Concealed Baby Swallow",
    "Counterattack",
    "Dark Cloud Cover",
    "Doji",
    "Doji Star",
    "Dragonfly Doji",
    "Engulfing",
    "Evening Doji Star",
    "Evening Star",
    "Gap Side By Side White",
    "Gravestone Doji",
    "Hammer",
    "Hanging Man",
    "Harami",
    "Harami Cross",
    "High Wave Candle",
    "Hikkake",
    "Hikkake Modified",
    "Homing Pigeon",
    "Identical Three Crows",
    "In Neck",
    "Inverted Hammer",
    "Kicking",
    "Kicking By Length",
    "Ladder Bottom",
    "Long Legged Doji",
    "Long Line Candle",
    "Marubozu",
    "Mat Hold",
    "Matching Low",
    "Morning Doji Star",
    "Morning Star",
    "On Neck",
    "Piercing",
    "Rickshaw Man",
    "Rise Fall Three Methods",
    "Separating Lines",
    "Shooting Star",
    "Short Line Candle",
    "Spinning Top",
    "Stalled Pattern",
    "Stick Sandwich",
    "Takuri",
    "Tasuki Gap",
    "Three Black Crows",
    "Three Inside",
    "Three Line Strike",
    "Three Outside",
    "Three Stars In South",
    "Three White Soldiers",
    "Thrusting",
    "Tristar",
    "Two Crows",
    "Unique Three River",
    "Up Down Gap Three Methods",
    "Upside Gap Two Crows",
]


def _sanitize(name: str) -> str:
    """Return a condensed version of the pattern name for indicator wiring.

    Parameters
    ----------
    name : str
        Human-readable candlestick pattern label.

    Returns
    -------
    str
        Sanitized identifier that matches the QuantConnect indicator class
        naming scheme.
    """
    return name.replace(" ", "").replace("-", "").replace("/", "").replace("'", "")


PATTERN_CLASSES: Dict[str, type] = {}
for _display_name in RAW_PATTERN_NAMES:
    _cls_name = _sanitize(_display_name)
    _cls = globals().get(_cls_name)
    if _cls is not None:
        PATTERN_CLASSES[_display_name] = _cls

PATTERN_LOOKUP: Dict[str, str] = {}
for _name in PATTERN_CLASSES.keys():
    PATTERN_LOOKUP[_name.lower()] = _name
    PATTERN_LOOKUP[_sanitize(_name).lower()] = _name

SIGNAL_HISTORY_MAXLEN = 400

def resolve_pattern_name(name: Optional[str]) -> Optional[str]:
    """Normalize user-supplied names to the canonical display label.

    Parameters
    ----------
    name : str or None
        Raw pattern identifier coming from configuration or runtime checks.

    Returns
    -------
    str or None
        Canonical display name understood by the indicator registry, or
        ``None`` when the pattern is unknown.
    """
    if not name:
        return None
    key = _sanitize(str(name)).lower()
    direct = PATTERN_LOOKUP.get(key)
    if direct:
        return direct
    key2 = str(name).strip().lower()
    return PATTERN_LOOKUP.get(key2)

class CandlestickPatternManager:
    """Manage candlestick indicators, consolidate data, and log detections."""

    DEFAULT_MINUTE_FRAMES = (5, 15)

    def __init__(
        self,
        algorithm,
        symbol,
        algo_directory: str,
        pattern_names: Optional[List[str]] = None,
    ) -> None:
        """Instantiate detectors and ensure default timeframes are subscribed.

        Parameters
        ----------
        algorithm : QCAlgorithm
            Host algorithm that owns the consolidators.
        symbol : Symbol
            Asset for which candlestick patterns should be tracked.
        algo_directory : str
            Root folder of the algorithm; used to locate the ``backtests``
            directory where pattern logs are stored.
        pattern_names : List[str], optional
            Subset of pattern names to monitor. Defaults to all supported
            QuantConnect candlestick indicators.

        Returns
        -------
        None
            The constructor ensures default timeframes are active and resolves
            a log directory for persisting pattern snapshots.
        """
        self.algorithm = algorithm
        self.symbol = symbol
        self.pattern_names = pattern_names or list(PATTERN_CLASSES.keys())
        self.pattern_names = [p for p in self.pattern_names if p in PATTERN_CLASSES]
        self.indicators: Dict[str, Dict[str, object]] = {}
        self.rolling_windows: Dict[str, deque] = {}
        self.detectors: Dict[str, Callable[["CandlestickPatternManager", str, datetime, float], None]] = {}
        self.log_entries: List[str] = []
        self.signal_history: Dict[str, deque] = {}
        self.timeframe_meta: Dict[str, Dict[str, int]] = {}
        self._consolidators: Dict[str, TradeBarConsolidator] = {}
        
        self.log_dir = resolve_log_dir(algo_directory, "patterns")
        self.log_path = os.path.join(self.log_dir, "patterns_generated.log")
        
        self._build_detectors()
        for frame in self.DEFAULT_MINUTE_FRAMES:
            self.ensure_timeframe(frame)
        # optional filter configuration (moved from alpha models)
        self.filter_enabled: bool = False
        self.filter_name: Optional[str] = None
        self.filter_minutes: Optional[int] = None
        self.filter_within: Optional[timedelta] = None
        self.filter_direction: Optional[str] = None

    def _build_detectors(self) -> None:
        """Bind indicator callbacks that record signals into local history."""
        for name in self.pattern_names:
            detector = globals().get(f"detect{_sanitize(name)}")
            if detector is None:
                detector = _create_detector(name)
            self.detectors[name] = detector

    def ensure_timeframe(self, minutes: int) -> str:
        """Attach a consolidator for the requested timeframe if needed.

        Parameters
        ----------
        minutes : int
            Length of the consolidation window expressed in minutes.

        Returns
        -------
        str
            Timeframe key (for example ``"5m"``) that identifies indicator
            collections and signal deques.
        """
        try:
            minutes_int = int(minutes)
        except (TypeError, ValueError):
            minutes_int = 0
        minutes_int = max(1, minutes_int)
        key = f"{minutes_int}m"
        if key in self.indicators:
            return key
        span = timedelta(minutes=minutes_int)
        consolidator = TradeBarConsolidator(span)
        consolidator.DataConsolidated += lambda sender, bar, k=key: self._on_consolidated(k, bar)
        self.algorithm.SubscriptionManager.AddConsolidator(self.symbol, consolidator)
        self._consolidators[key] = consolidator
        self.rolling_windows[key] = deque(maxlen=max(1, minutes_int))
        indicator_map: Dict[str, object] = {}
        for pattern_name in self.pattern_names:
            cls = PATTERN_CLASSES[pattern_name]
            indicator_map[pattern_name] = cls(f"{self.symbol.Value}-{_sanitize(pattern_name)}-{key}")
        self.indicators[key] = indicator_map
        self.signal_history[key] = deque(maxlen=SIGNAL_HISTORY_MAXLEN)
        self.timeframe_meta[key] = {"minutes": minutes_int}
        return key

    def _on_consolidated(self, timeframe_key: str, bar: TradeBar) -> None:
        """Process each consolidated bar and record signals when detected.

        Parameters
        ----------
        timeframe_key : str
            Key returned by ``ensure_timeframe`` that maps to indicator sets.
        bar : TradeBar
            Consolidated bar emitted by the matching consolidator.
        """
        indicator_map = self.indicators.get(timeframe_key, {})
        if not indicator_map:
            return
        for pattern_name, indicator in indicator_map.items():
            indicator.Update(bar)
            if not getattr(indicator, "IsReady", False):
                continue
            value = float(indicator.Current.Value)
            if value == 0.0:
                continue
            detector = self.detectors.get(pattern_name)
            if detector:
                detector(self, timeframe_key, bar.EndTime, value)
            else:
                self._record_signal(timeframe_key, pattern_name, bar.EndTime, value)

    @staticmethod
    def _direction_label(direction_value: float) -> str:
        """Translate indicator polarity into a human-readable label.

        Parameters
        ----------
        direction_value : float
            Raw indicator output where positive denotes bullish pressure and
            negative denotes bearish pressure.

        Returns
        -------
        str
            ``"bullish"``, ``"bearish"`` or ``"neutral"`` depending on the
            sign of ``direction_value``.
        """
        if direction_value > 0:
            return "bullish"
        if direction_value < 0:
            return "bearish"
        return "neutral"

    def _record_signal(self, timeframe_key: str, pattern_name: str, event_time: datetime, direction_value: float) -> None:
        """Persist a detected pattern to the in-memory ledger and log buffer.

        Parameters
        ----------
        timeframe_key : str
            Identifier of the timeframe on which the pattern fired.
        pattern_name : str
            Human-readable pattern label.
        event_time : datetime
            Timestamp of the consolidated bar for which the indicator crossed.
        direction_value : float
            Raw indicator output used to derive the bullish/bearish label.
        """
        direction_label = self._direction_label(direction_value)
        timestamp = event_time.strftime("%Y-%m-%d %H:%M:%S")
        message = (
            f"{self.symbol} -> {pattern_name} detected at {timestamp}, timeframe {timeframe_key} ({direction_label})"
        )
        self.log_entries.append(message)
        history = self.signal_history.get(timeframe_key)
        if history is not None:
            history.append(
                {
                    "pattern": pattern_name,
                    "time": event_time,
                    "timeframe": timeframe_key,
                    "direction": direction_value,
                    "direction_label": direction_label,
                }
            )

    def get_recent_signals(
        self,
        pattern_name: Optional[str] = None,
        timeframe: Optional[str] = None,
        *,
        within: Optional[timedelta] = None,
        direction: Optional[str] = None,
    ) -> List[dict]:
        """Return cached pattern events filtered by name, timeframe, and age.

        Parameters
        ----------
        pattern_name : str, optional
            Only include signals that match this pattern label.
        timeframe : str, optional
            Specific timeframe key (for example ``"5m"``) to inspect.
        within : timedelta, optional
            Maximum age of the signal relative to the algorithm clock.
        direction : str, optional
            Restrict results to ``"bullish"`` or ``"bearish"`` signals.

        Returns
        -------
        List[dict]
            Ordered list of signal dictionaries describing historical pattern
            hits that meet the supplied filters.
        """
        canonical = resolve_pattern_name(pattern_name) if pattern_name else None
        direction_key = direction.lower() if isinstance(direction, str) else None
        frames = [timeframe] if timeframe else list(self.signal_history.keys())
        results: List[dict] = []
        now = self.algorithm.Time if within is not None else None
        for frame in frames:
            history = self.signal_history.get(frame)
            if not history:
                continue
            for record in history:
                if canonical and record["pattern"] != canonical:
                    continue
                if direction_key == "bullish" and record["direction"] <= 0:
                    continue
                if direction_key == "bearish" and record["direction"] >= 0:
                    continue
                if within is not None and now is not None and now - record["time"] > within:
                    continue
                results.append(record)
        return results

    # Convenience wrapper: resolves and checks whether a pattern exists in the
    # given timeframe / lookback. This mirrors the small helper previously in
    # the alpha model so callers can keep their code compact.
    def isCandleStickPattern(
        self,
        pattern_name: str,
        timeframe_minutes: int,
        direction: Optional[str] = None,
        within: Optional[timedelta] = None,
    ) -> bool:
        """Compatibility wrapper that mirrors the legacy alpha helper."""
        return self.has_pattern(pattern_name, timeframe_minutes, within=within, direction=direction)

    def apply_filter_config(self, cfg: Optional[dict]) -> None:
        """Apply a small candlestick_filter config dict.

        Expected keys (all optional):
         - pattern / name: pattern label
         - enabled: bool
         - timeframe_minutes: int
         - within_minutes / lookback_minutes: int
         - direction: 'bullish'|'bearish'

        Returns
        -------
        None
            The manager's filter attributes are mutated in place; invalid
            values are ignored with a debug breadcrumb.
        """
        # reset defaults
        self.filter_enabled = False
        self.filter_name = None
        self.filter_minutes = None
        self.filter_within = None
        self.filter_direction = None
        if not isinstance(cfg, dict):
            return
        pattern_label = cfg.get("pattern") or cfg.get("name")
        canonical = resolve_pattern_name(pattern_label) if pattern_label else None
        if canonical:
            self.filter_name = canonical
            self.filter_enabled = bool(cfg.get("enabled", True))
        else:
            if pattern_label and hasattr(self, "algorithm") and self.algorithm:
                key = str(pattern_label).strip().lower()
                self.algorithm.Debug(f"CandlestickPatternManager: unknown candlestick pattern '{pattern_label}'; skipping filter.")
        minutes = None
        timeframe_override = cfg.get("timeframe_minutes")
        try:
            if timeframe_override is not None:
                minutes_override = int(timeframe_override)
                if minutes_override > 0:
                    minutes = minutes_override
        except (TypeError, ValueError):
            if hasattr(self, "algorithm") and self.algorithm:
                self.algorithm.Debug("CandlestickPatternManager: invalid candlestick timeframe in config; ignoring.")
        if minutes is None:
            minutes = self.timeframe_meta.get(next(iter(self.timeframe_meta)), {}).get("minutes", 5)
        self.filter_minutes = max(1, int(minutes))
        # lookback/within
        within_override = cfg.get("within_minutes") or cfg.get("lookback_minutes")
        try:
            if within_override is not None:
                within_minutes = int(within_override)
                if within_minutes > 0:
                    self.filter_within = timedelta(minutes=within_minutes)
        except (TypeError, ValueError):
            if hasattr(self, "algorithm") and self.algorithm:
                self.algorithm.Debug("CandlestickPatternManager: invalid candlestick lookback in config; ignoring.")
        if self.filter_within is None:
            self.filter_within = timedelta(minutes=self.filter_minutes)
        # direction
        direction_cfg = cfg.get("direction")
        if isinstance(direction_cfg, str) and direction_cfg.lower() in ("bullish", "bearish"):
            self.filter_direction = direction_cfg.lower()
        else:
            self.filter_direction = None
        # ensure consolidator
        try:
            self.ensure_timeframe(self.filter_minutes)
        except Exception:
            pass

    def filter_passes(self) -> bool:
        """Return True if either no filter is configured or the configured
        candlestick filter has been observed within its lookback window.
        Also emits a single debug message per day when the filter blocks entries.

        Returns
        -------
        bool
            ``True`` when trading is allowed, ``False`` when the filter forbids
            entries because the desired pattern has not appeared recently.
        """
        if not getattr(self, "filter_enabled", False) or not self.filter_name:
            return True
        ok = self.isCandleStickPattern(
            self.filter_name,
            self.filter_minutes or 1,
            direction=self.filter_direction,
            within=self.filter_within,
        )
        now = getattr(self, "algorithm", None) and getattr(self.algorithm, "Time", None)
        if not ok:
            if hasattr(self, "algorithm") and self.algorithm:
                pass
                # self.algorithm.Debug(f"CandlestickPatternManager: candlestick filter '{self.filter_name}' blocked entry at {now}.")
        # no daily suppression: always return blocking result
        return ok

    def has_pattern(
        self,
        pattern_name: str,
        minutes: int,
        *,
        within: Optional[timedelta] = None,
        direction: Optional[str] = None,
    ) -> bool:
        """Determine whether a pattern recently occurred within the window.

        Parameters
        ----------
        pattern_name : str
            Candidate pattern to search for.
        minutes : int
            Timeframe to monitor, expressed in minutes.
        within : timedelta, optional
            Override for the lookback window; defaults to ``minutes``.
        direction : str, optional
            Constrain the search to bullish or bearish hits.

        Returns
        -------
        bool
            ``True`` if the pattern exists within the time window, otherwise
            ``False``.
        """
        canonical = resolve_pattern_name(pattern_name)
        if canonical is None:
            return False
        timeframe_key = self.ensure_timeframe(minutes)
        history = self.signal_history.get(timeframe_key)
        if not history:
            return False
        direction_key = direction.lower() if isinstance(direction, str) else None
        within_delta = within if within is not None else timedelta(minutes=max(1, int(minutes)))
        now = self.algorithm.Time
        for record in reversed(history):
            if record["pattern"] != canonical:
                continue
            if direction_key == "bullish" and record["direction"] <= 0:
                continue
            if direction_key == "bearish" and record["direction"] >= 0:
                continue
            if within_delta is not None and now is not None and now - record["time"] > within_delta:
                return False
            return True
        return False

    def flush(self) -> None:
        """Write collected log entries to disk inside the resolved run folder.

        Returns
        -------
        None
            Creates the ``patterns_generated.log`` file when there are pending
            entries; otherwise the method is a no-op.
        """
        if not self.log_entries:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self.log_entries))
        self.algorithm.Debug(f"CandlestickPatternManager: Candlestick pattern log written to {self.log_path}")


def _create_detector(pattern_name: str) -> Callable[[CandlestickPatternManager, str, datetime, float], None]:
    """Generate a fallback detector that records signals using the manager.

    Parameters
    ----------
    pattern_name : str
        Display name used to label the recorded signals.

    Returns
    -------
    Callable
        Detector function compatible with the QuantConnect indicator callback
        signature. The detector appends signals to the manager history.
    """
    func_name = f"detect{_sanitize(pattern_name)}"

    def _detector(manager: CandlestickPatternManager, timeframe_key: str, event_time: datetime, direction: float) -> None:
        manager._record_signal(timeframe_key, pattern_name, event_time, direction)

    _detector.__name__ = func_name
    globals()[func_name] = _detector
    return _detector


for _pattern in PATTERN_CLASSES.keys():
    _create_detector(_pattern)
