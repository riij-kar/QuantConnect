# AiFiesta_MultiTimeframe.py

from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data import *
from QuantConnect.Data.Consolidators import *
from QuantConnect.Data.Market import *
from QuantConnect.Indicators import *
from datetime import time, timedelta
import numpy as np
import statistics
import json

class MeanReversionIntradayAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2025, 8, 5)
        self.SetEndDate(2025, 8, 20)
        self.set_account_currency("INR")
        self.SetCash(100000)
        self.SetTimeZone("Asia/Kolkata")

        # --- Re-factored Section: Timeframe Configuration ---
        # User-configurable timeframe. Change the value here.
        # Examples: 5, 15, 45, 60 (for 1hr), 75, etc.
        self.timeframe_minutes = 3 
        self.timeframe = timedelta(minutes=self.timeframe_minutes)
        # --- End Re-factored Section ---

        #IntradayTradeCount
        self.intraday_trade_count = 1

        self.symbols = ["HAL"]
        self.indicators = {}
        self.deviation_history = {}
        self.sigma_avg_history = {}
        self.positions = {}

        # The base resolution for data subscription will always be Minute
        # to feed the consolidator accurately.
        base_resolution = Resolution.Minute

        for symbol in self.symbols:
            equity = self.AddEquity(symbol, base_resolution, Market.INDIA)
            
            # Initialize dictionaries for this symbol
            self.deviation_history[symbol] = []
            self.sigma_avg_history[symbol] = []
            self.positions[symbol] = None

            # Setup indicators but DO NOT register them for auto-updates.
            # We will update them manually with consolidated data.
            self.indicators[symbol] = self._setup_indicators()
            
            # Create and register the consolidator for the custom timeframe
            consolidator = TradeBarConsolidator(self.timeframe)
            consolidator.DataConsolidated += self.OnDataConsolidated
            self.SubscriptionManager.AddConsolidator(equity.Symbol, consolidator)

        # Logs
        self.signal_log = []
        self.trade_log = []

        # --- Re-factored Section: Dynamic Warm-up Period ---
        # Warm-up period must be long enough to prime the indicators on the consolidated timeframe.
        # We calculate the number of minute bars needed.
        # Example: 100 bars of 15-min data = 100 * 15 = 1500 minute bars.
        history_period = 100 
        warmup_period = history_period * self.timeframe_minutes
        self.SetWarmUp(warmup_period, base_resolution)
        # --- End Re-factored Section ---

    def _setup_indicators(self):
        # This function no longer needs a symbol, as indicator instances are created per symbol in Initialize
        return {
            'sma21': SimpleMovingAverage(21),
            'sma55': SimpleMovingAverage(55),
            'vwap': VolumeWeightedAveragePriceIndicator(100),
            'rsi': RelativeStrengthIndex(14, MovingAverageType.Simple),
            'atr': AverageTrueRange(14, MovingAverageType.Simple),
            'vol_ma': SimpleMovingAverage(20) # For volume MA
        }

    def OnData(self, data):
        #when next day starts, reset intraday trade count
        if self.Time.hour == 9 and self.Time.minute == 15:
            self.intraday_trade_count = 1
        # OnData now runs every minute. It is used for high-frequency tasks
        # like risk management and end-of-day checks.
        # Signal generation is moved to OnDataConsolidated.
        if self.IsWarmingUp:
            return

        # Manage risk for existing positions on a minute-by-minute basis
        for symbol in self.symbols:
            if self.positions[symbol] is not None and data.Bars.ContainsKey(symbol):
                self.IntradayRiskManager(symbol, data.Bars[symbol].Close)

        # End of day liquidation logic remains here
        market_close = time(15, 30)
        if self.Time.time() >= market_close:
            for symbol in self.symbols:
                if self.positions[symbol] is not None:
                    self.Liquidate(symbol)
                    self.trade_log.append({
                        "time": self.Time.strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": symbol, "action": "exit", "reason": "EOD liquidation",
                        "price": self.Securities[symbol].Price, "quantity": self.Portfolio[symbol].Quantity
                    })
                    self.positions[symbol] = None

    def OnDataConsolidated(self, sender, consolidated_bar):
        # This event handler is triggered ONLY when a new consolidated bar
        # of the custom timeframe (e.g., 15 minutes) is formed.
        # This is where we generate trading signals.
        if self.IsWarmingUp:
            return

        symbol = consolidated_bar.Symbol.Value
        ind = self.indicators[symbol]

        # --- Corrected Section: Manual Indicator Updates ---
        # Update all indicators with the data from the new consolidated bar.
        ind['sma21'].Update(consolidated_bar.Time, consolidated_bar.Close)
        ind['sma55'].Update(consolidated_bar.Time, consolidated_bar.Close)
        ind['rsi'].Update(consolidated_bar.Time, consolidated_bar.Close)
        ind['vol_ma'].Update(consolidated_bar.Time, consolidated_bar.Volume)
        
        # MODIFICATION: Pass the entire bar to indicators that need more than just a single price.
        ind['vwap'].Update(consolidated_bar) # VWAP needs both price and volume.
        ind['atr'].Update(consolidated_bar)  # ATR needs High, Low, and Close.
        # --- End Corrected Section ---

        # Check if all indicators are ready before proceeding
        if not all(indicator.IsReady for indicator in ind.values()):
            return
        
        #no signal generation if time crossed 3:00pm
        is3PM = time(15, 0)
        if self.Time.time() >= is3PM:
            return

        # Generate signal using the consolidated bar's price
        signal = self.IntradayMeanReversionAlpha(symbol, consolidated_bar.Close, ind)
        if signal:
            self.signal_log.append({
                "time": self.Time.strftime("%Y-%m-%d %H:%M:%S"), "symbol": symbol,
                "type": signal['type'], "reason": signal['reason'], "deviation": signal['avg_dev'],
                "sigma_avg": signal['sigma_avg'], "rsi": signal['rsi']
            })

            # Enter trade if no position exists and no multiple trades per day limit not exceeded
            if self.positions[symbol] is None:
                    # and (self.intraday_trade_count <= 5):  # Limit to 5 trades per day
                self._enter_trade(symbol, signal['type'], consolidated_bar.Close, ind['atr'].Current.Value)
                self.intraday_trade_count += 1


    def IntradayMeanReversionAlpha(self, symbol, price, ind):
        dev_sma21 = price - ind['sma21'].Current.Value
        dev_sma55 = price - ind['sma55'].Current.Value
        dev_vwap = price - ind['vwap'].Current.Value

        self.deviation_history[symbol].append([dev_sma21, dev_sma55, dev_vwap])
        if len(self.deviation_history[symbol]) > 100:
            self.deviation_history[symbol].pop(0)

        if len(self.deviation_history[symbol]) < 2: return None
        
        devs = np.array(self.deviation_history[symbol])
        sigma_sma21 = np.std(devs[:, 0], ddof=1)
        sigma_sma55 = np.std(devs[:, 1], ddof=1)
        sigma_vwap = np.std(devs[:, 2], ddof=1)
        sigma_avg = statistics.mean([sigma_sma21, sigma_sma55, sigma_vwap])
        
        self.sigma_avg_history[symbol].append(sigma_avg)
        if len(self.sigma_avg_history[symbol]) > 100:
            self.sigma_avg_history[symbol].pop(0)

        current_sigma_avg = statistics.mean(self.sigma_avg_history[symbol]) if self.sigma_avg_history[symbol] else 0
        if current_sigma_avg == 0: return None

        avg_dev = statistics.mean([dev_sma21, dev_sma55, dev_vwap])
        is_significant = abs(avg_dev) >= current_sigma_avg or abs(abs(avg_dev) - current_sigma_avg) < 0.1 * current_sigma_avg
        rsi_val = ind['rsi'].Current.Value
        vol = self.Securities[symbol].Volume
        vol_ma = ind['vol_ma'].Current.Value
        is_quality = is_significant and vol > vol_ma

        if is_quality:
            # if rsi_val < 30:
            if avg_dev < -1.5 * current_sigma_avg and rsi_val < 30:
                return {"type": "long", "reason": "oversold deviation", "avg_dev": avg_dev, "sigma_avg": current_sigma_avg, "rsi": rsi_val}
            # elif rsi_val > 70:
            elif avg_dev > 1.5 * current_sigma_avg and rsi_val > 70:
                return {"type": "short", "reason": "overbought deviation", "avg_dev": avg_dev, "sigma_avg": current_sigma_avg, "rsi": rsi_val}
        return None

    def _enter_trade(self, symbol, signal_type, entry_price, atr):
        direction = 1 if signal_type == "long" else -1
        quantity = self.CalculateOrderQuantity(symbol, 0.1) * direction
        self.MarketOrder(symbol, quantity)

        sl_distance = 2.0 * atr
        tp_distance = 2.6 * sl_distance
        self.positions[symbol] = {
            "entry_price": entry_price,
            "sl_price": entry_price - direction * sl_distance,
            "tp_price": entry_price + direction * tp_distance,
            "direction": direction, "quantity": quantity, "trailing": False
        }
        self.trade_log.append({
            "time": self.Time.strftime("%Y-%m-%d %H:%M:%S"), "symbol": symbol,
            "action": "entry", "type": signal_type, "price": entry_price, "quantity": quantity,
            "atr": atr, 
            "sl_price": self.positions[symbol]["sl_price"],
            "tp_price": self.positions[symbol]["tp_price"]
        })

    def IntradayRiskManager(self, symbol, current_price):
        pos = self.positions[symbol]
        if pos is None: return

        direction = pos["direction"]
        entry_price = pos["entry_price"]
        atr = self.indicators[symbol]['atr'].Current.Value

        if not self.indicators[symbol]['atr'].IsReady: return
         # --- TRAILING STOP LOGIC 
        if pos["trailing"]:
            trailing_sl = entry_price + direction * 0.5 * atr
            pos["sl_price"] = max(pos["sl_price"], trailing_sl) if direction == 1 else min(pos["sl_price"], trailing_sl)
        # --- STOP-LOSS CHECK
        sl_price = pos["sl_price"]
        if (direction == 1 and current_price <= sl_price) or \
           (direction == -1 and current_price >= sl_price):
            self.Liquidate(symbol)
            self.trade_log.append({
                "time": self.Time.strftime("%Y-%m-%d %H:%M:%S"), "symbol": symbol, "action": "exit",
                "reason": "stop loss", "price": current_price, "quantity": self.Portfolio[symbol].Quantity
            })
            self.positions[symbol] = None
            return
        # --- MODIFIED TAKE-PROFIT LOGIC ---
        tp_price = pos["tp_price"]
        if not pos["trailing"] and ((direction == 1 and current_price >= tp_price) or \
                                (direction == -1 and current_price <= tp_price)):
            exit_quantity = self.Portfolio[symbol].Quantity * 0.5 * (-1)
            self.MarketOrder(symbol, exit_quantity)
            self.trade_log.append({
                "time": self.Time.strftime("%Y-%m-%d %H:%M:%S"), "symbol": symbol, "action": "partial_exit",
                "reason": "take profit", "price": current_price, "quantity": exit_quantity
            })
            # This flag now acts as a lock to prevent this block from running again.
            pos["trailing"] = True 
    
    def OnEndOfAlgorithm(self):
        self.Debug(f"Ending Algorithm. Signal Log has {len(self.signal_log)} entries. Trade Log has {len(self.trade_log)} entries.")
        try:
            signal_json = json.dumps(self.signal_log, indent=4)
            trade_json = json.dumps(self.trade_log, indent=4)
            self.Debug(f"Signal Log JSON:\n{signal_json}")
            self.Debug(f"Trade Log JSON:\n{trade_json}")
        except Exception as e:
            self.Debug(f"JSON dump failed: {str(e)}")

