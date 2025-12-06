from datetime import timedelta

from QuantConnect.Algorithm.Framework.Alphas import AlphaModel, InsightDirection, Insight
from QuantConnect.Algorithm.Framework.Execution import ImmediateExecutionModel
from QuantConnect.Algorithm.Framework.Portfolio import (
    EqualWeightingPortfolioConstructionModel,
    PortfolioTarget,
)
from QuantConnect.Algorithm.Framework.Risk import RiskManagementModel

class Strategy:
    def initialize(self, algo, indicators, config=None):
        """Wire strategy-level references to the algorithm context.

        Parameters
        ----------
        algo : QCAlgorithm
            Active Lean algorithm instance that owns the strategy.
        indicators : dict
            Dictionary of pre-built indicators returned by ``build_indicators``.
        config : dict, optional
            Strategy-specific configuration read from ``config.json``.

        Returns
        -------
        None
            The method stores references and emits a debug breadcrumb so the
            runtime log captures the strategy bootstrapping step.
        """
        self.algo = algo
        self.indicators = indicators
        self.config = config or {}
        self.algo.Debug("Mean Reversion Strategy initialized under Algorithm Framework.")


class MeanReversionAlphaModel(AlphaModel):
    """Alpha model emitting long insights when short-term momentum aligns.

    Candlestick confirmation now uses a preset filter so the strategy controls
    the pattern logic without relying on external configuration files.
    """

    CANDLESTICK_FILTER_CONFIG = {
        "enabled": True,
        "pattern": "Marubozu",
        "timeframe_minutes": 5,
        "within_minutes": 30,
        "direction": "bullish",
    }

    def __init__(self):
        """Prepare mutable state that supports the mean-reversion signals."""
        super().__init__()
        self.algorithm = None
        self.symbol = None
        self.indicators = None
        # signal_horizon is the time span you expect the insight to stay valid. Passing it into Insight.Price(...) sets the insight’s expiry—Lean treats the prediction as applicable until that horizon elapses, which influences portfolio sizing, execution timing, and when the system clears the signal if no new one arrives.
        self.signal_horizon = timedelta(minutes=5)
        self.current_direction = InsightDirection.Flat
        self.last_signal_time = None
        self.trade_limit = None
        self.trades_today = 0
        self.last_trade_day = None
        self._limit_notified_day = None
        self.pattern_manager = None

    def configure(self, algorithm, symbol, indicators, config):
        """Bind runtime services and align the candlestick filter parameters.

        Parameters
        ----------
        algorithm : QCAlgorithm
            Host algorithm instance used to access securities and logging.
        symbol : Symbol
            Primary traded symbol for which the insight stream is produced.
        indicators : dict
            Indicator map built during algorithm initialization.
        config : dict
            Strategy configuration dictionary, commonly deserialized from
            ``config.json``.

        Returns
        -------
        None
            Internal state is refreshed and the candlestick filter is pushed
            into the shared ``CandlestickPatternManager`` so the alpha can gate
            trades on fresh pattern observations.
        """
        self.algorithm = algorithm
        self.symbol = symbol
        self.indicators = indicators

        trade_settings_cfg = (config or {}).get("trade_settings", {}) if isinstance(config, dict) else {}
        resolution_cfg = (config or {}).get("resolution", {}) if isinstance(config, dict) else {}
        period_value = resolution_cfg.get("period", 5)
        try:
            forward_minutes = int(period_value or 5)
        except (TypeError, ValueError):
            forward_minutes = 5
        self.signal_horizon = timedelta(minutes=max(1, forward_minutes))
        self.trade_limit = None
        self.trades_today = 0
        self.last_trade_day = None
        self._limit_notified_day = None
        self.pattern_manager = getattr(algorithm, "pattern_tracker", None)
        # Delegate pattern config to the manager using the strategy constant
        # but align the timeframe with the runtime horizon so pattern checks
        # stay in sync with generated insights.
        if self.pattern_manager and hasattr(self.pattern_manager, "apply_filter_config"):
            try:
                dynamic_minutes = max(1, forward_minutes)
                filter_cfg = dict(self.CANDLESTICK_FILTER_CONFIG)
                filter_cfg["timeframe_minutes"] = dynamic_minutes
                if "within_minutes" not in filter_cfg and "lookback_minutes" not in filter_cfg:
                    filter_cfg["within_minutes"] = dynamic_minutes
                self.pattern_manager.apply_filter_config(filter_cfg)
            except Exception:
                if self.algorithm:
                    self.algorithm.Debug(
                        "MeanReversionAlphaModel: failed to apply dynamic candlestick filter to pattern manager."
                    )
        limit_value = trade_settings_cfg.get("trade_limit") or None
        try:
            if limit_value is not None:
                limit_int = int(limit_value)
                if limit_int > 0:
                    self.trade_limit = limit_int
        except (TypeError, ValueError):
            if self.algorithm:
                self.algorithm.Debug("MeanReversionAlphaModel: invalid trade_limit; ignoring.")
        # Candlestick filter is now fully owned by the strategy constant and the
        # manager; no additional parsing from runtime config is needed here.

    # pattern helpers moved into CandlestickPatternManager to keep this
    # alpha small and focused on entry/exit logic.

    def Update(self, algorithm, data):
        """Generate bullish insights when price momentum aligns with filters.

        This method respects trade limits, verifies candlestick confirmation,
        and throttles repeated signals based on ``signal_horizon``.

        Parameters
        ----------
        algorithm : QCAlgorithm
            Algorithm instance passed by Lean during framework updates.
        data : Slice or TradeBar
            Fresh market data payload received for the configured consolidator.

        Returns
        -------
        List[Insight]
            A list containing zero or more newly generated insights. The
            current implementation emits at most one long insight per call.
        """
        if not self.symbol or not self.indicators:
            return []

        if not all(indicator.IsReady for indicator in self.indicators.values()):
            return []

        bar = data if hasattr(data, "Close") else None
        if bar is None and hasattr(data, "Bars"):
            bar = data.Bars[self.symbol] if self.symbol in data.Bars else None
        if bar is None:
            return []

        security = algorithm.Securities.get(self.symbol) if hasattr(algorithm, "Securities") else None
        holding_qty = security.Holdings.Quantity if security and security.Holdings else 0
        if holding_qty <= 0:
            if self.current_direction != InsightDirection.Flat:
                self.current_direction = InsightDirection.Flat
                self.last_signal_time = None

        ema9 = self.indicators["ema_9"].Current.Value
        ema21 = self.indicators["ema_21"].Current.Value
        vwap = self.indicators["vwap"].Current.Value

        insights = []

        current_time = getattr(bar, "EndTime", None)
        #trade limit 5 per day
        if current_time is None:
            current_time = getattr(algorithm, "Time", None)
        current_day = current_time.date() if current_time else None
        if current_day is not None and current_day != self.last_trade_day:
            self.last_trade_day = current_day
            self.trades_today = 0
            self._limit_notified_day = None

        if (
            self.trade_limit is not None
            and current_day is not None
            and self.trades_today >= self.trade_limit
        ):
            if self.algorithm and self._limit_notified_day != current_day:
                self.algorithm.Debug(
                    f"MeanReversionAlphaModel: trade limit {self.trade_limit} reached for {current_day}; skipping new insights."
                )
                self._limit_notified_day = current_day
            return []
        #trade limit 5 per day
        if (
            self.current_direction == InsightDirection.Up
            and self.last_signal_time is not None
            and current_time is not None
        ):
            if current_time - self.last_signal_time >= self.signal_horizon:
                self.current_direction = InsightDirection.Flat
        #chart pattern filter
        pattern_ok = True
        if self.pattern_manager:
            pattern_ok = self.pattern_manager.filter_passes()
            if self.algorithm and pattern_ok:
                self.algorithm.Debug(
                    f"MeanReversionAlphaModel: pattern_ok={pattern_ok} at {getattr(bar, 'EndTime', 'N/A')}"
                )
            
        if self.current_direction != InsightDirection.Up:
            if (
                bar.Close > ema9
                and bar.Close > ema21
                and bar.Close > vwap
                and pattern_ok
            ):
                insights.append(
                    Insight.Price(self.symbol, self.signal_horizon, InsightDirection.Up)
                )
                self.current_direction = InsightDirection.Up
                self.last_signal_time = current_time
                if current_day is not None:
                    self.trades_today += 1
                if self.algorithm:
                    self.algorithm.Debug(
                        f"MeanReversionAlphaModel: emitting Up insight at {bar.EndTime}"
                    )
                detail = "; ".join(str(insight) for insight in insights)
                if self.algorithm:
                    self.algorithm.Debug(
                        f"MeanReversionAlphaModel: insights at {bar.EndTime}: {detail}"
                    )
                return insights

        if self.algorithm and insights:
            detail = "; ".join(str(insight) for insight in insights)
            self.algorithm.Debug(
                f"MeanReversionAlphaModel: insights at {bar.EndTime}: {detail}"
            )
        return insights
class MeanReversionRiskManagementModel(RiskManagementModel):
    """Risk module enforcing fixed stop-loss and take-profit offsets."""

    def __init__(self, stop_loss_offset: float = 10.0, take_profit_offset: float = 20.0):
        """Capture risk thresholds that will be applied to open positions.

        Parameters
        ----------
        stop_loss_offset : float, optional
            Maximum tolerated loss (in price units) before the position is
            liquidated.
        take_profit_offset : float, optional
            Gain threshold (in price units) that locks in profit when reached.
        """
        self.algorithm = None
        self.symbol = None
        self.stop_loss_offset = float(stop_loss_offset) if stop_loss_offset is not None else None
        self.take_profit_offset = float(take_profit_offset) if take_profit_offset is not None else None
        self._last_exit_reason = None

    def configure(self, algorithm, symbol, config):
        """Read risk limits from configuration and attach to the algorithm.

        Parameters
        ----------
        algorithm : QCAlgorithm
            Host algorithm instance, used mainly for logging.
        symbol : Symbol
            Asset whose exposure should be managed by this model.
        config : dict
            Configuration values that may override the default offsets.
        """
        trade_settings_cfg = (config or {}).get("trade_settings", {}) if isinstance(config, dict) else {}
        self.algorithm = algorithm
        self.symbol = symbol
        if not isinstance(config, dict):
            return
        stop_loss = trade_settings_cfg.get("stop_loss_offset", self.stop_loss_offset)
        take_profit = trade_settings_cfg.get("take_profit_offset", self.take_profit_offset)
        try:
            if stop_loss is None:
                self.stop_loss_offset = None
            else:
                self.stop_loss_offset = float(stop_loss)
                if self.stop_loss_offset <= 0:
                    self.stop_loss_offset = None
        except (TypeError, ValueError):
            if self.algorithm:
                self.algorithm.Debug("MeanReversionRiskModel: invalid stop-loss; disabling stop trigger.")
            self.stop_loss_offset = None
        try:
            if take_profit is None:
                self.take_profit_offset = None
            else:
                self.take_profit_offset = float(take_profit)
                if self.take_profit_offset <= 0:
                    self.take_profit_offset = None
        except (TypeError, ValueError):
            if self.algorithm:
                self.algorithm.Debug("MeanReversionRiskModel: invalid take-profit; disabling target trigger.")
            self.take_profit_offset = None

    def ManageRisk(self, algorithm, targets):
        """Exit positions when price breaches the configured offsets.

        Parameters
        ----------
        algorithm : QCAlgorithm
            Host algorithm providing access to holdings and prices.
        targets : List[PortfolioTarget]
            Framework-provided targets (unused here) that represent desired
            portfolio state before risk adjustments.

        Returns
        -------
        List[PortfolioTarget]
            An empty list when no action is required, otherwise a single
            ``PortfolioTarget`` instructing Lean to liquidate the symbol.
        """
        if self.symbol is None:
            return []
        security = algorithm.Securities.get(self.symbol)
        if security is None or not security.Invested:
            self._last_exit_reason = None
            return []
        holding = security.Holdings
        if holding.Quantity <= 0:
            self._last_exit_reason = None
            return []

        entry_price = holding.AveragePrice
        current_price = security.Price
        if entry_price is None or entry_price == 0 or current_price is None or current_price == 0:
            return []

        trigger_reason = None
        if (
            self.stop_loss_offset is not None
            and current_price <= entry_price - self.stop_loss_offset
        ):
            trigger_reason = "stop-loss"
        elif (
            self.take_profit_offset is not None
            and current_price >= entry_price + self.take_profit_offset
        ):
            trigger_reason = "take-profit"

        if trigger_reason is None:
            self._last_exit_reason = None
            return []

        if trigger_reason != self._last_exit_reason and self.algorithm:
            self.algorithm.Debug(
                f"MeanReversionRiskModel: {trigger_reason} triggered at {current_price}"
            )
        self._last_exit_reason = trigger_reason
        return [PortfolioTarget(self.symbol, 0)]

AlphaModel = MeanReversionAlphaModel
PortfolioModel = EqualWeightingPortfolioConstructionModel
ExecutionModel = ImmediateExecutionModel
RiskModel = MeanReversionRiskManagementModel