QC Local Dashboard UI (app.py)
=================================

This document explains the purpose of each helper, callback, and utility defined in `app.py`. Use it as a reference when extending or debugging the dashboard.

Module Overview
---------------
`app.py` wires together Dash components, LEAN backtest artifacts, and Plotly charts. It:

* discovers eligible project folders and their backtest runs
* loads and normalizes JSON/CSV data into pandas objects
* constructs the multi-panel visualization (price, indicators, equity, returns, drawdown)
* registers Dash callbacks for project/backtest selection plus trade and order tables

Key Constants
-------------
* `WORKSPACE_ROOT`: absolute path to the repo root; starting point for project discovery.
* `EXCLUDE_FOLDERS`: folders to skip when scanning for projects (e.g., `data`, `storage`, `ui`).

App Initialization
------------------
`app = Dash(...)` creates the Dash application, sets the browser title, and raises the upload limit to 200 MB to accommodate large CSVs.

Helper Functions
----------------
`find_project_paths()`
    Scans `WORKSPACE_ROOT` for subdirectories that contain a `backtests/` folder with at least one child directory. Returns the sorted list of absolute project paths.

`list_backtests(project_path)`
    Enumerates backtest run folders under `<project_path>/backtests/`. Enriches each dropdown label using the first `*-summary.json` it finds (net profit, trade count) and returns a Dash dropdown options list.

`load_backtest_folder(folder)`
    Reads every top-level JSON file inside a backtest run directory. Returns a mapping from filename to parsed JSON (or `None` when parsing fails), providing raw data for charts and statistics.

`extract_series(charts_obj)`
    Converts LEAN chart payloads into pandas Series. Handles both OHLC arrays (`[ts, o, h, l, c]`) and simple `[ts, value]` points, normalizing timestamps to UTC. Keys follow `<Chart>::<Series>::<component>` so later steps can target `Price` or `Strategy Equity` series.

`build_price_from_series(series_map)`
    Combines the canonical `Price::open/high/low/close` series into a pandas OHLC DataFrame. Falls back to any `::close` series if full OHLC data is missing. Returns `None` when no price data can be inferred.

`compute_indicators(price_df)`
    Generates EMA9, EMA21, and RSI14 from the price DataFrame. Returns a `{name: Series}` mapping and gracefully handles empty inputs.

`build_equity_and_drawdown(series_map)`
    Searches for equity-related series, preferring OHLC components. Computes drawdown (`(equity - peak) / peak`) and percent returns (`pct_change × 100`). Returns `(equity, drawdown, returns)` where `equity` may be a DataFrame or Series.

`_to_float(val)`
    Normalizes statistics strings by removing currency symbols, commas, and `%`, then converts to floats. Supports numeric inputs and returns `None` for unparseable values.

`_stat_color(key, value)`
    Applies simple heuristics (profit, drawdown, Sharpe, win rate, fees) to choose a color for statistic text (green/orange/red). Uses `_to_float` to interpret strings.

`_render_stats(stats)`
    Builds a list of Dash `Span` elements for headline statistics, applying spacing and the color hints from `_stat_color`.

`_find_first_key(stats, candidates)`
    Looks for the first stat whose key matches any candidate substring (case-insensitive). Returns `(key, value)` or `None`.

`_kv_item(label, key, value)`
    Creates a styled Dash `Div` showing a label on the left and the colored value right-aligned. Used across the stats panel sections.

`_section(title, items)`
    Wraps related statistics in a collapsible block, using `<details>/<summary>` when the Dash HTML components support them. Falls back to a simple container otherwise.

`_render_stats_panel(stats)`
    Combines the helpers above to produce the sidebar: headline KPIs plus grouped sections (Portfolio, Trade, Runtime, Other) and optional notices about price data issues.

`parse_trades(perf_jsons)`
    Walks performance JSON files to extract `closedTrades`. Normalizes nested symbol data, converts datetime columns, serializes complex fields, and returns a `DataFrame` ready for a Dash `DataTable`.

`build_figure(price_df, indicators, equity, drawdown, returns, trades_df)`
    Creates the main four-row Plotly figure:
    * Row 1 – Price candlestick/line with overlayed indicators and trade entry/exit markers.
    * Row 2 – RSI line.
    * Row 3 – Equity OHLC/line.
    * Row 4 – Return bars (primary y-axis) and Drawdown line (secondary y-axis).
    Enforces shared x-axis bounds, limits point counts via environment variables, and displays date ticks across all panels.

`get_chart_series(series_map, chart_prefix)`
    Filters additional chart data (e.g., Portfolio Margin, Exposure) by prefix. Prefers `::close` components to plot clean single-line traces and feeds the “extra charts” section.

`get_main_layout()`
    Assembles the two-column layout consisting of the sidebar (project/backtest selectors, stats) and main content area (primary chart, extra charts, trades/orders tables).

`route(pathname)`
    A lightweight router callback: `/upload` renders the CSV upload page; anything else shows the main dashboard layout.

`update_backtests(project_path)`
    Dash callback that repopulates the backtest dropdown whenever the user selects a different project by invoking `list_backtests`.

`update_visual(backtest_folder)`
    Central visualization callback. Performs the following pipeline:
    1. Load all JSON artifacts for the chosen backtest.
    2. Merge chart payloads and extract timeseries.
    3. Load OHLCV from CSV based on project configuration.
    4. Compute indicators, equity, drawdown, returns.
    5. Build trade and order tables from order events.
    6. Construct the main figure and extra charts.
    7. Aggregate and render statistics plus notices.
    Returns the Plotly figure, sidebar contents, trades table, extra charts, and orders table components.

Application Startup
-------------------
At the bottom of the file the CSV upload callbacks are registered and the Dash server is started (debug mode with hot reload) when the module is executed directly.

Extending `app.py`
------------------
* Add new indicators inside `compute_indicators` and update `build_figure` if they require dedicated panels.
* Surface additional charts by returning them from `get_chart_series` and appending to `extra_blocks` in `update_visual`.
* Integrate new statistics by extending `_stat_color`, `_render_stats_panel`, or the notice logic in `update_visual`.

Questions or enhancements can be captured in issues/PRs referencing the relevant section of this document.

Command Reference
-----------------
- `docker pull quantconnect/lean:latest`
- `docker pull quantconnect/research:latest`
- `pip install --upgrade lean`
- `lean create-project "MyProjectName"`
- `lean backtest "algo-main"`
- `lean backtest "MyProjectName"`
- `lean research "algo-main" --image quantconnect/research:latest`
- `cd D:\Algos\QuantConnect\ui`
- `.\.venv\Scripts\Activate.ps1`
- `python app.py`
- `pip install TA-Lib`