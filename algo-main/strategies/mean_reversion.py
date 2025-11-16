class Strategy:
    def initialize(self, algo, indicators):
        self.algo = algo
        self.indicators = indicators
        self.algo.Debug("Mean Reversion Strategy initialized.")

    def generate_signal(self, bar):
        self.algo.Debug(f"Generating signal for bar: {bar.Time}")
        ema9 = self.indicators["ema_9"].Current.Value
        rsi = self.indicators["rsi"].Current.Value
        if rsi < 20 and bar.Close < ema9:
            return "long"
        elif rsi > 70 and bar.Close > ema9:
            return "short"
        return None