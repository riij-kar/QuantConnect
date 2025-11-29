# Algorithm Component Documentation

## cli/data_converter.py

### convert_to_lean_csv
- **Purpose:** Ingest a raw trade history CSV and emit Lean-formatted zipped minute files partitioned by trading date.
- **Inputs:**
  - `filepath` (str): Path to the source CSV file containing the raw history.
  - `symbol` (str): Symbol used to name destination folders and files; lower-cased when written.
  - `resolution` (str): Target Lean resolution. Only the value `"minute"` triggers processing; other values are ignored.
  - `output_dir` (str): Destination directory that mirrors the Lean data folder layout.
- **Outputs:** Writes one zipped CSV per trading day into `output_dir/<symbol>/<YYYYMMDD>_trade.zip`; returns `None`.
- **Side Effects:** Creates directories as needed, leaves a debug-friendly intermediary CSV before compressing, deletes the intermediary file after zipping.

### Script Entry Point
- The `if __name__ == "__main__"` block demonstrates ad-hoc usage by converting a sample HAL dataset into the Lean directory structure.

## strategies/mean_reversion.py

### Strategy.initialize
- **Purpose:** Store references to the hosting algorithm, indicator map, and configuration so downstream models can retrieve them.
- **Inputs:** `algo` (QCAlgorithm), `indicators` (dict), optional `config` (dict).
- **Outputs:** Returns `None` after setting member variables and emitting a debug breadcrumb.

### MeanReversionAlphaModel
- **Class Role:** Implements a simple long-only momentum alpha with candlestick confirmation and daily trade limits.

#### __init__
- Initializes state such as `signal_horizon`, trade-count bookkeeping, and pattern manager pointer.
- Accepts no parameters beyond `self`; returns `None`.

#### configure
- **Purpose:** Bind the algorithm context, align the signal horizon with the configured timeframe, and push a dynamic candlestick filter into the shared pattern manager.
- **Inputs:** `algorithm` (QCAlgorithm), `symbol` (Symbol), `indicators` (dict), `config` (dict).
- **Outputs:** Returns `None`; logs configuration issues when they arise.

#### Update
- **Purpose:** Evaluate the most recent consolidated bar, apply indicator and candlestick filters, honor daily trade limits, and possibly emit a bullish insight.
- **Inputs:** `algorithm` (QCAlgorithm), `data` (TradeBar or Slice).
- **Outputs:** Returns a list of `Insight` instances (empty when no signal is produced).
- **Side Effects:** Updates internal state, increments daily trade counters, and logs insight activity for diagnostics.

### MeanReversionRiskManagementModel
- **Class Role:** Applies fixed-price stop-loss and take-profit exits to the active position.

#### __init__
- **Inputs:** Optional `stop_loss_offset` and `take_profit_offset` floats expressed in absolute price units.
- **Outputs:** Returns `None`; stores offsets for later evaluation.

#### configure
- **Purpose:** Read override values from the algorithm configuration and sanitize them.
- **Inputs:** `algorithm` (QCAlgorithm), `symbol` (Symbol), `config` (dict).
- **Outputs:** Returns `None`; logs warnings for invalid values.

#### ManageRisk
- **Purpose:** Inspect the current holding and trigger liquidation when price breaches either offset.
- **Inputs:** `algorithm` (QCAlgorithm), `targets` (list of PortfolioTarget; unused in logic).
- **Outputs:** Returns a list containing a single flat `PortfolioTarget` when an exit is required, otherwise an empty list.
- **Side Effects:** Emits debug statements when exit conditions fire.

## utils/candlestick_pattern.py

### _sanitize
- Condenses a human-readable pattern name by removing whitespace and punctuation so it matches QuantConnect indicator class names.
- **Input:** `name` (str). **Output:** Sanitized string.

### resolve_pattern_name
- Normalizes user-provided pattern labels to canonical display names using the precomputed lookup table.
- **Input:** `name` (str or None). **Output:** Canonical pattern label or `None` when unresolved.

### CandlestickPatternManager
- **Class Role:** Centralizes pattern indicator setup, consolidator management, runtime filtering, and log generation.

#### __init__
- **Inputs:** `algorithm` (QCAlgorithm), `symbol` (Symbol), `algo_directory` (str), optional `pattern_names` (list of str).
- **Outputs:** Returns `None`; ensures default timeframes exist and resolves the log directory under the latest backtest run.

#### _build_detectors
- Registers detector functions for each tracked pattern so indicator callbacks feed the manager’s recorder.
- **Inputs/Outputs:** No external inputs; returns `None`.

#### ensure_timeframe
- **Purpose:** Attach a consolidator and indicator bundle for a requested minute timeframe.
- **Input:** `minutes` (int). **Output:** Timeframe key string such as `"5m"`.

#### _on_consolidated
- Responds to consolidator events, updating each indicator and recording signals when thresholds trip.
- **Inputs:** `timeframe_key` (str), `bar` (TradeBar). **Output:** Returns `None`.

#### _direction_label
- Converts the raw indicator value into a `"bullish"`, `"bearish"`, or `"neutral"` tag.
- **Input:** `direction_value` (float). **Output:** Direction label (str).

#### _record_signal
- Persists a detected pattern both in memory (for future queries) and in a pending log message list.
- **Inputs:** `timeframe_key` (str), `pattern_name` (str), `event_time` (datetime), `direction_value` (float). **Output:** Returns `None`.

#### get_recent_signals
- Retrieves historical signals matching optional filters such as pattern name, timeframe, direction, and age.
- **Inputs:** Optional `pattern_name`, `timeframe`, `within` (timedelta), `direction` (str).
- **Output:** List of dictionaries describing each qualifying signal.

#### isCandleStickPattern
- Helper retained for backward compatibility; delegates to `has_pattern` while keeping the original signature.
- **Inputs:** `pattern_name` (str), `timeframe_minutes` (int), optional `direction` and `within`.
- **Output:** Boolean indicating whether the pattern was seen recently.

#### apply_filter_config
- Parses a `candlestick_filter` configuration dictionary and primes filter state (pattern, timeframe, lookback, direction).
- **Input:** `cfg` (dict or None). **Output:** Returns `None`; invalid values trigger debug messages.

#### filter_passes
- Evaluates the active filter and reports whether trading is allowed.
- **Input:** None (relies on internal state). **Output:** Boolean gate result.

#### has_pattern
- Searches the cached signal history for the requested pattern within the specified lookback window.
- **Inputs:** `pattern_name` (str), `minutes` (int), optional `within` (timedelta) and `direction` (str).
- **Output:** Boolean flag.

#### flush
- Writes accumulated log entries to `patterns_generated.log` beneath the resolved backtest folder.
- **Input:** None. **Output:** Returns `None`.

#### _resolve_latest_backtest_dir (static)
- Scans the `backtests` directory and selects the most recently named run folder, skipping pattern subfolders.
- **Input:** `backtests_root` (str). **Output:** Path to the newest backtest folder or `None`.

### _create_detector
- Factory that produces lightweight detector callbacks for patterns lacking explicit handlers.
- **Input:** `pattern_name` (str). **Output:** Callable compatible with QuantConnect indicator events.

## utils/indicator_factory.py

### build_indicators
- **Purpose:** Instantiate moving averages, RSI, Bollinger Bands, ATR, and VWAP based on configuration-provided periods.
- **Inputs:** `algo` (QCAlgorithm), `symbol` (Symbol), `config` (dict containing `indicator_params`).
- **Outputs:** Dictionary mapping indicator aliases (for example `"ema_9"`) to QuantConnect indicator objects.
- **Side Effects:** Logs the constructed indicator keys to aid debugging.

## utils/strategy_loader.py

### StrategyBundle (dataclass)
- Aggregates the primary strategy and optional framework components for convenient transport back to the algorithm.
- Fields: `strategy`, optional `alpha_model`, `execution_model`, `portfolio_model`, `risk_model`.

### _try_instantiate
- Attempts to instantiate classes or call callables retrieved from a strategy module; returns the original attribute when instantiation is impossible or unnecessary.
- **Input:** `attr` (Any). **Output:** Instantiated object, original attribute, or `None`.

### load_strategy
- Imports `strategies.<strategy_name>`, extracts the expected framework classes, instantiates them where possible, and bundles everything into a `StrategyBundle`.
- **Input:** `strategy_name` (str). **Output:** `StrategyBundle` instance. **Raises:** `AttributeError` when the module lacks a `Strategy` attribute.

## main.py

### Algomain (QCAlgorithm)
- Central algorithm class that wires configuration, indicators, strategies, and logging for Lean runtime execution.

#### Initialize
- Reads `config.json`, configures cash/timezone/dates, constructs indicators and the candlestick pattern manager, loads the strategy bundle, and sets up data consolidators.
- **Inputs:** None. **Output:** Returns `None`; prepares the algorithm for live data updates.

#### OnDataConsolidated
- Receives each consolidated bar, updates tracked indicators, and delegates insight generation to the configured alpha model.
- **Inputs:** `sender` (object), `bar` (TradeBar). **Output:** Returns `None` while emitting insights via `EmitInsights`.

#### OnOrderEvent
- Captures fills or partial fills, appends them to `trade_log`, and echoes a concise message to the debug log.
- **Input:** `orderEvent` (OrderEvent). **Output:** Returns `None`.

#### OnEndOfAlgorithm
- Flushes candlestick pattern logs at the conclusion of the backtest or live run.
- **Inputs:** None. **Output:** Returns `None`.
