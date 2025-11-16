# region imports
from datetime import datetime, timedelta
from AlgorithmImports import *
from utils.indicator_factory import build_indicators
from utils.strategy_loader import load_strategy
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
    def Initialize(self):
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

        resolution_setting = str(self.cfg.get("base_resolution", "minute")).lower()
        if resolution_setting not in ("minute", "daily"):
            self.Debug(f"Unsupported base_resolution '{resolution_setting}', defaulting to minute.")
            resolution_setting = "minute"

        if resolution_setting == "daily":
            resolution = Resolution.Daily
            timeframe_value = int(self.cfg.get("timeframe_days", 1) or 1)
            consolidation_span = timedelta(days=max(1, timeframe_value))
        else:
            resolution = Resolution.Minute
            timeframe_value = int(self.cfg.get("timeframe_minutes", 1) or 1)
            consolidation_span = timedelta(minutes=max(1, timeframe_value))

        self.symbol = self.AddEquity(self.cfg.get("EquityName"), resolution, Market.India).Symbol
        self.indicators = build_indicators(self, self.symbol, self.cfg)
        self.strategy = load_strategy(self.cfg.get("strategy"))
        self.strategy.initialize(self, self.indicators)

        self.signal_log = []
        self.trade_log = []
        warmup_bars = int(self.cfg.get("warmup_bars", 30) or 0)
        if warmup_bars > 0:
            self.SetWarmUp(warmup_bars)

        self.Debug(
            f"Initialized with {resolution_setting} data; consolidating every {consolidation_span}"
        )

        self.consolidator = TradeBarConsolidator(consolidation_span)
        self.consolidator.DataConsolidated += self.OnDataConsolidated
        self.SubscriptionManager.AddConsolidator(self.symbol, self.consolidator)

    def OnDataConsolidated(self, sender, bar):
        for ind in self.indicators.values():
            ind.Update(bar)
        if not all(i.IsReady for i in self.indicators.values()):
            return
        signal = self.strategy.generate_signal(bar)
        if signal:
            self.MarketOrder(self.symbol, 1 if signal == "long" else -1)
            self.trade_log.append({"time": str(self.Time), "action": signal, "price": bar.Close})

    def OnEndOfAlgorithm(self):
        pass
        # with open("backtests/trade_log.json", "w") as f:
        #     json.dump(self.trade_log, f, indent=4)

