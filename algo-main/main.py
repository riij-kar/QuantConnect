# region imports
from datetime import datetime, timedelta
import inspect
from AlgorithmImports import *
from utils.indicator_factory import build_indicators
from utils.strategy_loader import load_strategy
from utils.candlestick_pattern import CandlestickPatternManager
import json
import os
# endregion

# Directory structure
# algo-first/
# ├── config.json
# ├── main.py
# ├── research.ipynb
# ├── backtests/
# ├── strategies/
# │   └── mean_reversion.py
# ├── utils/
# │   ├── indicator_factory.py
# │   └── strategy_loader.py
# └── cli/
#     └── data_converter.py


class Algomain(QCAlgorithm):
    """Lean algorithm entry point wiring strategy, indicators, and logging."""

    def Initialize(self):
        """Configure cash, data subscriptions, strategy bundle, and managers.

        The routine reads ``config.json`` from the algorithm directory to load
        trading dates, cash settings, and strategy metadata. It wires up
        indicators, the candlestick pattern tracker, and the strategy bundle
        (alpha, execution, portfolio, and risk models) before subscribing to a
        consolidator that drives the alpha workflow.
        """
        # Read config from algorithm directory
        algo_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(algo_dir, "config.json")
        
        with open(config_path) as f:
            self.cfg = json.load(f)
        
        self.Debug(f"Config loaded: {self.cfg}")
        self.SetAccountCurrency("INR")  # Your account value is tracked in INR
        self.SetCash("INR", self.cfg.get("default-cash"))
        # Read start/end dates from config.json (format: YYYY-MM-DD). Fall back to defaults if missing or invalid.
        try:
            start_str = self.cfg.get("startDate")
            end_str = self.cfg.get("endDate")
            if start_str:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                self.SetStartDate(start_dt.year, start_dt.month, start_dt.day)
            else:
                self.SetStartDate(2025, 8, 5)

            if end_str:
                end_dt = datetime.strptime(end_str, "%Y-%m-%d")
                self.SetEndDate(end_dt.year, end_dt.month, end_dt.day)
            else:
                self.SetEndDate(2025, 8, 22)
        except Exception as e:
            # If parsing fails, use sensible defaults and log the error
            self.Debug(f"Invalid start/end date in config.json: {e}; using defaults.")
            self.SetStartDate(2025, 8, 5)
            self.SetEndDate(2025, 8, 22)

        self.SetTimeZone(self.cfg.get("timezone"))

        resolution_cfg = self.cfg.get("resolution", {}) or {}
        resolution_setting = str(resolution_cfg.get("default", "MINUTE")).upper()
        valid_resolutions = {
            "TICK": Resolution.Tick,
            "SECOND": Resolution.Second,
            "MINUTE": Resolution.Minute,
            "HOUR": Resolution.Hour,
            "DAILY": Resolution.Daily,
        }
        if resolution_setting not in valid_resolutions:
            self.Debug(
                f"Unsupported resolution '{resolution_setting}', defaulting to MINUTE."
            )
            resolution_setting = "MINUTE"

        resolution = valid_resolutions[resolution_setting]

        period_value = resolution_cfg.get("period", 1)

        try:
            period_value = int(period_value or 1)
        except (TypeError, ValueError):
            self.Debug(
                f"Invalid period '{period_value}' for resolution {resolution_setting}; defaulting to 1."
            )
            period_value = 1

        period_value = max(1, period_value)

        if resolution_setting == "DAILY":
            consolidation_span = timedelta(days=period_value)
        elif resolution_setting == "HOUR":
            consolidation_span = timedelta(hours=period_value)
        elif resolution_setting in ("SECOND", "TICK"):
            consolidation_span = timedelta(seconds=period_value)
        else:  # MINUTE and fallback
            consolidation_span = timedelta(minutes=period_value)

        self.symbol = self.AddEquity(self.cfg.get("EquityName"), resolution, Market.India).Symbol
        self.indicators = build_indicators(self, self.symbol, self.cfg)
        self.pattern_tracker = CandlestickPatternManager(
            self,
            self.symbol,
            algo_dir,
        )
        self.strategy_bundle = load_strategy(self.cfg.get("strategy"))
        self.strategy = self.strategy_bundle.strategy

        if inspect.isclass(self.strategy_bundle.alpha_model):
            self.strategy_bundle.alpha_model = self.strategy_bundle.alpha_model()
        if inspect.isclass(self.strategy_bundle.portfolio_model):
            self.strategy_bundle.portfolio_model = self.strategy_bundle.portfolio_model()
        if inspect.isclass(self.strategy_bundle.execution_model):
            self.strategy_bundle.execution_model = self.strategy_bundle.execution_model()
        if inspect.isclass(self.strategy_bundle.risk_model):
            self.strategy_bundle.risk_model = self.strategy_bundle.risk_model()

        if hasattr(self.strategy, "initialize"):
            self.strategy.initialize(self, self.indicators, self.cfg)

        if self.strategy_bundle.alpha_model and hasattr(self.strategy_bundle.alpha_model, "configure"):
            self.strategy_bundle.alpha_model.configure(
                algorithm=self,
                symbol=self.symbol,
                indicators=self.indicators,
                config=self.cfg,
            )

        if self.strategy_bundle.alpha_model:
            self.SetAlpha(self.strategy_bundle.alpha_model)
        if self.strategy_bundle.portfolio_model:
            self.SetPortfolioConstruction(self.strategy_bundle.portfolio_model)
        if self.strategy_bundle.execution_model:
            self.SetExecution(self.strategy_bundle.execution_model)
        if self.strategy_bundle.risk_model and hasattr(self.strategy_bundle.risk_model, "configure"):
            self.strategy_bundle.risk_model.configure(
                algorithm=self,
                symbol=self.symbol,
                config=self.cfg,
            )
        if self.strategy_bundle.risk_model:
            self.SetRiskManagement(self.strategy_bundle.risk_model)

        self.signal_log = []
        self.trade_log = []
        warmup_bars = int(self.cfg.get("warmup_bars", 30) or 0)
        if warmup_bars > 0:
            self.SetWarmUp(warmup_bars)

        self.Debug(
            f"Initialized with {resolution_setting.lower()} data; consolidating every {consolidation_span}"
        )

        self.consolidator = TradeBarConsolidator(consolidation_span)
        self.consolidator.DataConsolidated += self.OnDataConsolidated
        self.SubscriptionManager.AddConsolidator(self.symbol, self.consolidator)

    def OnDataConsolidated(self, sender, bar):
        """Update indicators and request new insights from the alpha model.

        Parameters
        ----------
        sender : object
            Source consolidator that produced the aggregated bar.
        bar : TradeBar
            Newly consolidated bar representing the configured timeframe.
        """
        for ind in self.indicators.values():
            ind.Update(bar)
        if not all(i.IsReady for i in self.indicators.values()):
            return

        alpha_model = getattr(self.strategy_bundle, "alpha_model", None)
        if not alpha_model:
            self.Debug("No alpha model configured; skipping insight generation.")
            return

        insights = alpha_model.Update(self, bar)
        if insights:
            self.EmitInsights(*insights)
            self.Debug(
                "Emitted insights: "
                + ", ".join(f"{insight.Symbol} {insight.Direction}" for insight in insights)
            )

    def OnOrderEvent(self, orderEvent: OrderEvent):
        """Track fills and echo concise diagnostics to the algorithm log.

        Parameters
        ----------
        orderEvent : OrderEvent
            Lean-provided order event describing fill status and pricing.
        """
        if orderEvent.Status not in (OrderStatus.Filled, OrderStatus.PartiallyFilled):
            return
        self.trade_log.append(
            {
                "time": str(orderEvent.UtcTime),
                "symbol": str(orderEvent.Symbol),
                "quantity": orderEvent.FillQuantity,
                "price": orderEvent.FillPrice,
                "direction": str(orderEvent.Direction),
            }
        )
        self.Debug(
            f"OrderEvent logged: {orderEvent.Symbol} {orderEvent.Direction} {orderEvent.FillQuantity} @ {orderEvent.FillPrice}"
        )

    def OnEndOfAlgorithm(self):
        """Flush pattern logs to disk when the backtest or live session ends."""
        if hasattr(self, "pattern_tracker"):
            self.pattern_tracker.flush()
        # with open("backtests/trade_log.json", "w") as f:
        #     json.dump(self.trade_log, f, indent=4)

