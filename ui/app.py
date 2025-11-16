import os, glob, json, pandas as pd, numpy as np
from datetime import datetime
from dash import Dash, dcc, html, Input, Output, State, dash_table
from typing import Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.chart_utils import (
    extract_series,
    build_price_from_series,
    compute_indicators,
    build_equity_and_drawdown,
    get_chart_series
)
from utils.trade_mapper import (
    load_all_json,
    parse_order_events,
    enrich_orders,
    reconstruct_trades,
    build_trade_table,
    build_order_table
)
from utils.price_loader import load_ohlcv_from_csv
from utils.consolidated import (
    normalize_frequency,
    format_frequency_label,
    frequency_to_timedelta,
    resample_ohlcv
)

# Root workspace path (adjust if running outside repo root)
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Candidate project folders (contain backtests folders)
EXCLUDE_FOLDERS = {"data", "storage", "__pycache__", "ui"}

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "QC Local Dashboard"
# Allow large CSV uploads up to 200MB (adjust as needed)
app.server.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

# ---------- Helpers ----------

# Scan the workspace root for candidate project directories that contain a
# 'backtests' folder with at least one backtest. Returns absolute paths.
def find_project_paths():
    print('WORKSPACE_ROOT', WORKSPACE_ROOT)
    paths = []
    for entry in os.listdir(WORKSPACE_ROOT):
        full = os.path.join(WORKSPACE_ROOT, entry)
        if os.path.isdir(full) and entry not in EXCLUDE_FOLDERS:
            # must have backtests folder with at least one child
            bt_dir = os.path.join(full, 'backtests')
            if os.path.isdir(bt_dir):
                has_child = any(os.path.isdir(os.path.join(bt_dir, x)) for x in os.listdir(bt_dir))
                if has_child:
                    paths.append(full)
    return sorted(paths)

# Enumerate backtest run folders for a given project and build dropdown options.
# Tries to enrich the label using <timestamp>-summary.json statistics when present.
def list_backtests(project_path: str):
    bt_dir = os.path.join(project_path, 'backtests')
    if not os.path.isdir(bt_dir):
        return []
    folders = [os.path.join(bt_dir, f) for f in os.listdir(bt_dir) if os.path.isdir(os.path.join(bt_dir, f))]
    options = []
    for f in sorted(folders, reverse=True):
        # attempt to read summary json for label enrichment
        summary_json = None
        for cand in glob.glob(os.path.join(f, '*-summary.json')):
            summary_json = cand
            break
        label = os.path.basename(f)
        if summary_json:
            try:
                data = json.load(open(summary_json, 'r'))
                stats = data.get('statistics', {})
                net = stats.get('Net Profit', '') or stats.get('End Equity', '')
                trades = stats.get('Total Trades', '') or stats.get('Total Orders', '')
                label = f"{label} | Net: {net} | Trades: {trades}"
            except Exception:
                pass
        options.append({"label": label, "value": f})
    return options

# Load all top-level JSON files from a backtest folder into a dict keyed by filename.
def load_backtest_folder(folder: str):
    result = {}
    for path in glob.glob(os.path.join(folder, '*.json')):
        name = os.path.basename(path)
        try:
            result[name] = json.load(open(path, 'r'))
        except Exception:
            result[name] = None
    return result

# Convert the LEAN 'charts' object into a map of pandas Series. Supports both simple
# [ts, value] points and OHLC arrays [ts, o, h, l, c]. Timestamps are parsed as UTC.
def extract_series(charts_obj: dict):
    """Parse LEAN chart JSON into individual pandas Series.
    Supports OHLC arrays [ts, o, h, l, c] and simple [ts, value] points.
    Automatically detects epoch second numeric timestamps and converts to UTC.
    Creates distinct series keys for OHLC: <Chart>::<Series>::open/high/low/close.
    """
    series_map: dict[str, pd.Series] = {}
    if not isinstance(charts_obj, dict):
        return series_map

    def to_dt(xs):
        if not xs:
            return pd.DatetimeIndex([])
        # If all numeric -> assume epoch seconds (LEAN uses seconds for daily/period charts)
        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in xs):
            try:
                return pd.to_datetime(xs, unit='s', utc=True)
            except Exception:
                pass
        # Fallback generic parse
        return pd.to_datetime(xs, utc=True, errors='coerce')

    for chart_name, chart in charts_obj.items():
        ser_dict = chart.get('series', {}) if isinstance(chart, dict) else {}
        for ser_name, ser in ser_dict.items():
            values = ser.get('values') or ser.get('points') or []
            if not values:
                continue
            # Detect OHLC style: every element list/tuple len >=5
            is_ohlc = all(isinstance(p, (list, tuple)) and len(p) >= 5 for p in values)
            if is_ohlc:
                xs = [p[0] for p in values]
                opens = [p[1] for p in values]
                highs = [p[2] for p in values]
                lows  = [p[3] for p in values]
                closes= [p[4] for p in values]
                idx = to_dt(xs)
                base = f"{chart_name}::{ser_name}"
                try:
                    series_map[f"{base}::open"]  = pd.Series(opens, index=idx).sort_index()
                    series_map[f"{base}::high"]  = pd.Series(highs, index=idx).sort_index()
                    series_map[f"{base}::low"]   = pd.Series(lows,  index=idx).sort_index()
                    series_map[f"{base}::close"] = pd.Series(closes,index=idx).sort_index()
                except Exception:
                    continue
                continue
            # Simple point series
            xs, ys = [], []
            for p in values:
                if isinstance(p, dict):
                    xs.append(p.get('x'))
                    ys.append(p.get('y'))
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    xs.append(p[0])
                    ys.append(p[1])
            idx = to_dt(xs)
            try:
                series_map[f"{chart_name}::{ser_name}"] = pd.Series(ys, index=idx).sort_index()
            except Exception:
                continue
    return series_map

# Try to build a price DataFrame from the extracted series using the common
# 'Price::open/high/low/close' naming. Falls back to a single 'close' series.
def build_price_from_series(series_map):
    # Attempt to construct OHLC from known naming patterns
    lower = {k.lower(): k for k in series_map.keys()}
    keys = {part: lower.get(part) for part in ['price::open','price::high','price::low','price::close']}
    if all(v for v in keys.values()):
        return pd.DataFrame({
            'open': series_map[keys['price::open']],
            'high': series_map[keys['price::high']],
            'low': series_map[keys['price::low']],
            'close': series_map[keys['price::close']]
        }).dropna()
    # fallback: look for a close series
    for k in series_map:
        if k.endswith('::Close') or k.lower().endswith('::close'):
            return pd.DataFrame({'close': series_map[k]}).dropna()
    return None

# Compute a small set of technical indicators (EMA9, EMA21, RSI14) from a price DataFrame.
def compute_indicators(price_df):
    if price_df is None or price_df.empty:
        return {}
    close = price_df['close'] if 'close' in price_df.columns else price_df.iloc[:, -1]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi14 = 100 - (100 / (1 + rs))
    return {'EMA9': ema9, 'EMA21': ema21, 'RSI14': rsi14}

# Build equity (OHLC when available), drawdown, and percent returns from extracted series.
def build_equity_and_drawdown(series_map):
    """Return equity as OHLC DataFrame when available, else Series; also drawdown and returns%.
    Searches for keys containing 'equity' and '::open/high/low/close'.
    """
    # Collect potential OHLC keys
    lower_keys = {k.lower(): k for k in series_map.keys()}
    def find(name):
        return lower_keys.get(name)
    # Try common pattern from extract_series: "Strategy Equity::Equity::<part>"
    parts = {
        'open': None,
        'high': None,
        'low': None,
        'close': None
    }
    for k in list(series_map.keys()):
        kl = k.lower()
        if 'equity' in kl and '::open' in kl:
            parts['open'] = k
        if 'equity' in kl and '::high' in kl:
            parts['high'] = k
        if 'equity' in kl and '::low' in kl:
            parts['low'] = k
        if 'equity' in kl and '::close' in kl:
            parts['close'] = k
    equity_close = None
    equity = None
    if all(parts.values()):
        equity = pd.DataFrame({
            'open': series_map[parts['open']],
            'high': series_map[parts['high']],
            'low': series_map[parts['low']],
            'close': series_map[parts['close']],
        }).dropna()
        equity_close = equity['close']
    else:
        # Fallback: any series whose name contains 'equity' (single line)
        for k,s in series_map.items():
            if 'equity' in k.lower():
                equity = s
                equity_close = s
                break
    drawdown = None
    if equity_close is not None:
        peak = equity_close.cummax()
        drawdown = (equity_close - peak) / peak
    returns = None
    if equity_close is not None:
        returns = equity_close.pct_change() * 100.0
    return equity, drawdown, returns

# Best-effort numeric parser for QC statistics strings (handles %, currency, commas).
def _to_float(val):
    """Parse numeric-like strings from QC stats (%, currency, commas). Return float or None."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # remove currency symbols and commas
    for ch in ['₹', '$', '€', ',', ' ']:
        s = s.replace(ch, '')
    is_percent = s.endswith('%')
    if is_percent:
        s = s[:-1]
    try:
        num = float(s)
        return num / 100.0 if is_percent else num
    except Exception:
        return None

# Assign a color hint for a given statistics key/value using simple finance rules.
def _stat_color(key: str, value):
    """Decide color for a given statistic key/value with sensible finance rules."""
    v = _to_float(value)
    k = (key or '').lower()
    # Fees/costs always red when non-zero
    if 'fee' in k:
        return 'red' if (v is None or abs(v) > 0) else 'inherit'
    # Drawdown: higher is worse -> red if > 0, green if 0
    if 'drawdown' in k:
        return 'red' if (v is not None and v > 0) else 'inherit'
    # Net Profit, CAGR/Compounding Annual Return
    if 'profit' in k or 'compounding' in k or 'return' in k:
        if v is None:
            return 'inherit'
        return 'green' if v > 0 else ('red' if v < 0 else 'inherit')
    # Sharpe/Sortino
    if 'sharpe' in k or 'sortino' in k:
        if v is None:
            return 'inherit'
        return 'green' if v >= 1 else ('orange' if 0 <= v < 1 else 'red')
    # Win Rate
    if 'win rate' in k:
        if v is None:
            return 'inherit'
        return 'green' if v >= 0.5 else ('orange' if 0.4 <= v < 0.5 else 'red')
    # Profit-Loss Ratio
    if 'profit-loss ratio' in k or 'profit loss ratio' in k:
        if v is None:
            return 'inherit'
        return 'green' if v >= 1 else 'red'
    # Default: color negatives red, positives inherit
    if v is not None and v < 0:
        return 'red'
    return 'inherit'

# Render statistics as inline spans with spacing and heuristic coloring.
def _render_stats(stats: dict):
    items = []
    for k, v in stats.items():
        color = _stat_color(k, v)
        items.append(html.Span(f"{k}: {v}", style={'marginRight':'12px', 'color': color}))
    return items

# --- Stats side-panel rendering helpers ---
def _find_first_key(stats: dict, candidates: list[str]) -> tuple[str, str] | None:
    """Return the first (key, value) pair where key contains any of the candidate substrings (case-insensitive)."""
    if not stats:
        return None
    lower_map = {k.lower(): k for k in stats.keys()}
    for cand in candidates:
        c = cand.lower()
        # exact lower match first
        if c in lower_map:
            k = lower_map[c]
            return k, stats[k]
        # fallback: substring scan
        for lk, orig in lower_map.items():
            if c in lk:
                return orig, stats[orig]
    return None

def _kv_item(label: str, key: str, value):
    """Render a single key/value row with heuristic color on the value."""
    color = _stat_color(key, value)
    return html.Div([
        html.Span(label, style={'color':'#555'}),
        html.Span(str(value), style={'float':'right', 'fontWeight':'600', 'color': color})
    ], style={'fontSize':'13px', 'padding':'2px 0', 'borderBottom':'1px dashed #eee'})

def _section(title: str, items: List[Any]) -> Any:
    """Collapsible section using native <details>/<summary> if available; otherwise a plain block."""
    if hasattr(html, 'Details') and hasattr(html, 'Summary'):
        return html.Details([
            html.Summary(title),
            html.Div(items)
        ], open=False, style={'margin':'6px 0'})
    # Fallback for older Dash versions without Details/Summary
    return html.Div([
        html.Div(title, style={'fontWeight':'600', 'margin':'6px 0'}),
        html.Div(items)
    ], style={'margin':'6px 0'})

def _render_stats_panel(stats: dict) -> html.Div:
    """Render a two-part stats panel: headline KPIs + collapsible groups for Trade/Portfolio/Runtime/Other."""
    headline_defs = [
        ('End Equity', ['End Equity', 'Portfolio endEquity']),
        ('Net Profit', ['Net Profit', 'Runtime Net Profit']),
        ('Sharpe', ['Sharpe Ratio', 'Portfolio sharpeRatio']),
        ('Sortino', ['Sortino Ratio', 'Portfolio sortinoRatio']),
        ('Drawdown', ['Drawdown', 'Portfolio drawdown']),
        ('Win Rate', ['Win Rate', 'Portfolio winRate', 'Trade winRate']),
        ('Profit Factor', ['Profit Factor', 'Trade profitFactor']),
        ('Total Trades', ['Total Trades', 'Trade totalNumberOfTrades'])
    ]
    headline = []
    for label, cands in headline_defs:
        found = _find_first_key(stats, cands)
        if found:
            k, v = found
            headline.append(_kv_item(label, k, v))

    # Group remaining stats
    trade_items, port_items, runtime_items, other_items = [], [], [], []
    for k, v in stats.items():
        item = _kv_item(k, k, v)
        lk = k.lower()
        if lk.startswith('trade '):
            trade_items.append(item)
        elif lk.startswith('portfolio '):
            port_items.append(item)
        elif lk.startswith('runtime '):
            runtime_items.append(item)
        else:
            other_items.append(item)

    return html.Div([
        html.Div(headline, style={'background':'#fff', 'border':'1px solid #eee', 'borderRadius':'6px', 'padding':'8px 10px', 'marginBottom':'8px'}),
        _section('Portfolio', port_items),
        _section('Trade', trade_items),
        _section('Runtime', runtime_items),
        _section('Other', other_items)
    ])

# Extract closed trades from performance JSON files and return as a DataFrame suitable
# for use in a Dash DataTable. Datetime fields are parsed and complex values serialized.
def parse_trades(perf_jsons: dict) -> pd.DataFrame:
    # Look for closedTrades inside any performance JSON
    for name,data in perf_jsons.items():
        if not isinstance(data, dict):
            continue
        tp = data.get('totalPerformance', {}) or {}
        closed = tp.get('closedTrades') or data.get('closedTrades') or []
        if closed:
            df = pd.DataFrame(closed)
            # Flatten nested symbol dicts -> permtick string
            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].apply(lambda v: v.get('permtick') if isinstance(v, dict) and 'permtick' in v else (v.get('value') if isinstance(v, dict) and 'value' in v else v))
            # Convert datetime fields
            for tcol in ['entryTime','exitTime','uTCTime','submissionTime']:
                if tcol in df.columns:
                    try:
                        df[tcol] = pd.to_datetime(df[tcol])
                    except Exception:
                        pass
            # Serialize any remaining dicts/lists to JSON strings for DataTable compatibility
            for col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict,list)) else x)
            return df
    return pd.DataFrame()

# Build the main multi-panel Plotly figure (Price, Equity, Returns, Drawdown, RSI)
# and mark trade entries/exits when available.
def build_figure(price_df, indicators, equity, drawdown, returns, trades_df, expected_interval=None):
    # Limit rendering load to keep scrolling smooth
    # Use all points by default; set MAX_PRICE_POINTS>0 to limit for performance
    MAX_PRICE_POINTS = int(os.environ.get('MAX_PRICE_POINTS', '0'))
    MAX_TRADE_MARKERS = int(os.environ.get('MAX_TRADE_MARKERS', '400'))

    gap_threshold = None
    if expected_interval is not None:
        try:
            gap_threshold = expected_interval * 3
        except Exception:
            gap_threshold = None

    def _with_gap_breaks(series):
        if series is None:
            return [], []
        if gap_threshold is None or len(series) < 2:
            return list(series.index), list(series.values)
        xs = []
        ys = []
        prev_ts = None
        for ts, val in series.items():
            if prev_ts is not None and (ts - prev_ts) > gap_threshold:
                xs.append(None)
                ys.append(None)
            xs.append(ts)
            ys.append(val)
            prev_ts = ts
        return xs, ys

    # Slice price to recent window
    price_plot = None
    if price_df is not None and not price_df.empty:
        if MAX_PRICE_POINTS and MAX_PRICE_POINTS > 0 and len(price_df) > MAX_PRICE_POINTS:
            price_plot = price_df.tail(MAX_PRICE_POINTS)
        else:
            price_plot = price_df
        tmin, tmax = price_plot.index.min(), price_plot.index.max()
    else:
        tmin = tmax = None

    # Slice indicators into same time window
    indicators_plot = {}
    if indicators:
        for k, ser in indicators.items():
            try:
                indicators_plot[k] = ser.loc[tmin:tmax] if tmin is not None else ser
            except Exception:
                indicators_plot[k] = ser

    # Slice equity/returns/drawdown to window
    def _slice(obj):
        if obj is None:
            return None
        try:
            return obj.loc[tmin:tmax] if tmin is not None else obj
        except Exception:
            return obj
    equity_plot = _slice(equity)
    drawdown_plot = _slice(drawdown)
    returns_plot = _slice(returns)

    # Filter trades to window and cap markers
    trades_plot = trades_df
    if trades_df is not None and not trades_df.empty and tmin is not None:
        def _within(ts):
            try:
                return (pd.to_datetime(ts) >= tmin) and (pd.to_datetime(ts) <= tmax)
            except Exception:
                return False
        trades_plot = trades_df[trades_df.apply(lambda r: _within(r.get('entryTime')) or _within(r.get('exitTime')), axis=1)]
        if len(trades_plot) > MAX_TRADE_MARKERS:
            trades_plot = trades_plot.tail(MAX_TRADE_MARKERS)

    # 6 panels: Price & Indicators, Volume, RSI, Equity, Return %, Drawdown
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.32, 0.12, 0.18, 0.2, 0.12, 0.06],
        vertical_spacing=0.03,
        specs=[[{}], [{}], [{}], [{}], [{}], [{}]],
        subplot_titles=(
            'Price & Indicators',
            'Volume',
            'RSI (14)',
            'Equity Curve',
            'Return %',
            'Drawdown'
        )
    )

    legend_seen: set[str] = set()

    def _add_trace(trace, *, row: int, col: int, secondary_y: bool = False):
        name = getattr(trace, 'name', None)
        if name:
            show = name not in legend_seen
            trace.showlegend = show
            if show:
                legend_seen.add(name)
        fig.add_trace(trace, row=row, col=col, secondary_y=secondary_y)

    # Price (row 1)
    if price_plot is not None:
        if all(c in price_plot.columns for c in ['open', 'high', 'low', 'close']):
            hover_text = [
                (
                    f"Time: {idx.strftime('%b %d %H:%M')}<br>"
                    f"Open: {o}<br>"
                    f"High: {h}<br>"
                    f"Low: {l}<br>"
                    f"Close: {c}"
                )
                for idx, o, h, l, c in zip(
                    price_plot.index,
                    price_plot['open'],
                    price_plot['high'],
                    price_plot['low'],
                    price_plot['close']
                )
            ]
            price_trace = go.Candlestick(
                x=price_plot.index,
                open=price_plot['open'],
                high=price_plot['high'],
                low=price_plot['low'],
                close=price_plot['close'],
                name='Price',
                hovertext=hover_text,
                hoverinfo='text'
            )
            _add_trace(price_trace, row=1, col=1)
        else:
            yser = price_plot['close'] if 'close' in price_plot.columns else price_plot.iloc[:, -1]
            trace_cls = go.Scattergl if len(yser) > 2000 else go.Scatter
            px, py = _with_gap_breaks(yser)
            price_line = trace_cls(
                x=px,
                y=py,
                mode='lines',
                name='Price',
                hovertemplate='Time: %{x|%b %d %H:%M}<br>Price: %{y}<extra></extra>',
                connectgaps=False
            )
            _add_trace(price_line, row=1, col=1)

        if 'volume' in price_plot.columns:
            vol = price_plot['volume'].fillna(0)
            if all(col in price_plot.columns for col in ['open', 'close']):
                colors = ['#2ca02c' if c >= o else '#d62728' for o, c in zip(price_plot['open'], price_plot['close'])]
            else:
                colors = ['#2ca02c'] * len(vol)
            volume_bar = go.Bar(
                x=vol.index,
                y=vol.values,
                marker=dict(color=colors, line=dict(width=0)),
                name='Volume',
                opacity=0.7
            )
            _add_trace(volume_bar, row=2, col=1)

    # Indicators (exclude RSI14 from price panel)
    for name, ser in indicators_plot.items():
        if name == 'RSI14':
            continue
        trace_cls = go.Scattergl if len(ser) > 3000 else go.Scatter
        ix, iy = _with_gap_breaks(ser)
        indicator_trace = trace_cls(
            x=ix,
            y=iy,
            mode='lines',
            name=name,
            connectgaps=False
        )
        _add_trace(indicator_trace, row=1, col=1)

    # Trades markers (still on price panel)
    if trades_plot is not None and not trades_plot.empty:
        for _, r in trades_plot.iterrows():
            if 'entryTime' in r and 'entryPrice' in r:
                entry_trace = go.Scatter(
                    x=[r['entryTime']],
                    y=[r.get('entryPrice')],
                    mode='markers',
                    marker_symbol='triangle-up',
                    marker_color='green',
                    marker_size=10,
                    name='Entry'
                )
                _add_trace(entry_trace, row=1, col=1)
            if 'exitTime' in r and 'exitPrice' in r:
                exit_trace = go.Scatter(
                    x=[r['exitTime']],
                    y=[r.get('exitPrice')],
                    mode='markers',
                    marker_symbol='x',
                    marker_color='red',
                    marker_size=9,
                    name='Exit'
                )
                _add_trace(exit_trace, row=1, col=1)

    # RSI panel (row 3)
    if 'RSI14' in indicators_plot:
        rsi = indicators_plot['RSI14']
        trace_cls = go.Scattergl if len(rsi) > 3000 else go.Scatter
        rx, ry = _with_gap_breaks(rsi)
        rsi_trace = trace_cls(x=rx, y=ry, mode='lines', name='RSI14', connectgaps=False)
        _add_trace(rsi_trace, row=3, col=1)

    # Equity panel (row 4)
    if equity_plot is not None:
        if isinstance(equity_plot, pd.DataFrame) and all(c in equity_plot.columns for c in ['open', 'high', 'low', 'close']):
            equity_candle = go.Candlestick(
                x=equity_plot.index,
                open=equity_plot['open'],
                high=equity_plot['high'],
                low=equity_plot['low'],
                close=equity_plot['close'],
                name='Equity OHLC'
            )
            _add_trace(equity_candle, row=4, col=1)
        else:
            ser = equity_plot
            eq_trace = go.Scattergl(x=ser.index, y=getattr(ser, 'values', ser), mode='lines', name='Equity')
            _add_trace(eq_trace, row=4, col=1)

    # Returns panel (row 5)
    if returns_plot is not None:
        ret_values = returns_plot.fillna(0)
        colors = ['#2ca02c' if v >= 0 else '#d62728' for v in ret_values.values]
        returns_bar = go.Bar(
            x=ret_values.index,
            y=ret_values.values,
            marker=dict(color=colors, line=dict(width=0)),
            opacity=0.75,
            name='Return %'
        )
        _add_trace(returns_bar, row=5, col=1)

    # Drawdown panel (row 6)
    if drawdown_plot is not None:
        dd = drawdown_plot.fillna(0)
        drawdown_line = go.Scatter(
            x=dd.index,
            y=dd.values,
            mode='lines',
            name='Drawdown',
            line=dict(color='#9467bd', width=1.6),
            fill='tozeroy',
            fillcolor='rgba(148, 103, 189, 0.25)',
            hovertemplate='Drawdown: %{y:.2%}<extra></extra>'
        )
        _add_trace(drawdown_line, row=6, col=1)

    fig.update_layout(
        height=1550,
        title='Backtest Visualization',
        legend_orientation='h',
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(t=90, l=60, r=40, b=90)
    )
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    fig.update_yaxes(title_text='RSI (14)', row=3, col=1)
    fig.update_yaxes(title_text='Equity', row=4, col=1)
    fig.update_yaxes(title_text='Return %', row=5, col=1)
    fig.update_yaxes(title_text='Drawdown', row=6, col=1, tickformat='.1%')

    # Force the x-axis range to the actual price window so earliest data (e.g., Jul 14)
    # is included, and show tidy tick labels.
    try:
        if price_plot is not None and len(price_plot.index) > 0:
            _xmin = price_plot.index.min()
            _xmax = price_plot.index.max()
            base_axis_kwargs = dict(type='date', range=[_xmin, _xmax], automargin=True)
            for r in (1, 2, 3, 4, 5, 6):
                fig.update_xaxes(row=r, col=1, **base_axis_kwargs)
            fig.update_xaxes(
                row=1,
                col=1,
                showticklabels=True,
                tickformat='%b %d<br>%H:%M',
                tickangle=0,
                ticklabelposition='outside top',
                ticks='inside',
                ticklen=4,
                tickfont=dict(size=10)
            )
            for r in (2, 3, 4, 5):
                fig.update_xaxes(row=r, col=1, showticklabels=False)
            fig.update_xaxes(
                row=6,
                col=1,
                showticklabels=True,
                tickformat='%b %d<br>%H:%M',
                tickangle=0,
                tickfont=dict(size=11)
            )
    except Exception:
        # Non-fatal; fall back to Plotly's autorange
        pass
    return fig

# --- Extras: other charts (margin/turnover/exposure/capacity) ---
# Gather series that belong to a chart prefix (e.g., 'Portfolio Margin') and prefer the
# OHLC 'close' component if present so we draw a clean line per series.
def get_chart_series(series_map: dict, chart_prefix: str) -> dict[str, pd.Series]:
    """Return dict of name->Series for keys matching a chart prefix (case-insensitive).
    Skips OHLC component keys; if OHLC exists, returns the close series only for a clean line.
    """
    result = {}
    prefix = chart_prefix.lower() + '::'
    # collect keys
    keys = [k for k in series_map.keys() if k.lower().startswith(prefix)]
    if not keys:
        return result
    # group by base name before ::open/close
    bases = {}
    for k in keys:
        kl = k.lower()
        if kl.endswith('::open') or kl.endswith('::high') or kl.endswith('::low'):
            continue
        if kl.endswith('::close'):
            base = k.rsplit('::', 1)[0]
            bases[base] = series_map[k]
            continue
        # simple series
        result[k] = series_map[k]
    # prefer close when available
    for base, ser in bases.items():
        name = base.split('::', 1)[-1]  # series name part
        result[name] = ser
    return result

# ---------- Layout ----------
from pages.upload_page import get_upload_layout, register_upload_callbacks  # type: ignore

# Build the main page layout with project/backtest selectors, chart, stats bar,
# extra charts area, and trades table.
def get_main_layout():
    # Two-column layout: left sidebar (selectors + stats), right main content (charts)
    sidebar = html.Div([
        html.H3('QC Dashboard'),
        html.Label('Project'),
        dcc.Dropdown(id='project-dd', options=[{'label': os.path.basename(p), 'value': p} for p in find_project_paths()], placeholder='Select project'),
        html.Label('Backtest', style={'marginTop':'6px'}),
        dcc.Dropdown(id='backtest-dd', options=[], placeholder='Select backtest folder'),
        html.Div([
            html.A('Go to Uploads', href='/upload')
        ], style={'marginTop':'8px'}),
        html.Hr(),
        html.Div(id='stats-panel'),
    ], style={'flex':'0 0 320px', 'padding':'10px 12px', 'borderRight':'1px solid #eee', 'background':'#fafafa', 'height':'100vh', 'overflowY':'auto'})

    main = html.Div([
        html.Div('Backtest Visualization', style={'fontSize':'22px','fontWeight':'600','margin':'10px 0 6px 10px'}),
        html.Div([
            html.Span('Consolidate to', style={'marginRight':'6px', 'color':'#555'}),
            dcc.Input(id='resample-value', type='number', min=1, step=1, value=1, style={'width':'70px'}),
            dcc.Dropdown(
                id='resample-unit',
                options=[
                    {'label': 'Minute(s)', 'value': 'minute'},
                    {'label': 'Hour(s)', 'value': 'hour'},
                    {'label': 'Day(s)', 'value': 'day'},
                    {'label': 'Month(s)', 'value': 'month'}
                ],
                value='minute',
                clearable=False,
                style={'width':'150px', 'height':'38px'}
            ),
            html.Button('Load Chart', id='resample-btn', n_clicks=0, style={'marginLeft':'8px'})
        ], style={'display':'flex', 'alignItems':'center', 'justifyContent':'flex-end', 'gap':'10px', 'margin':'0 10px 12px 10px'}),
        dcc.Loading(dcc.Graph(id='chart'), type='default'),
        html.Div(id='extra-charts', style={'padding':'0 10px'}),
        html.H4('Trades', style={'margin':'10px'}),
        html.Div(id='trades-table', style={'padding':'0 10px 10px 10px'}),
        html.H4('Orders', style={'margin':'10px'}),
        html.Div(id='orders-table', style={'padding':'0 10px 20px 10px'})
    ], style={'flex':'1 1 auto', 'height':'100vh', 'overflowY':'auto'})

    return html.Div([sidebar, main], style={'display':'flex', 'height':'100vh'})

app.layout = html.Div([
    dcc.Location(id='url'),
    html.Div(id='page-content')
])

# Simple router: '/' -> main dashboard, '/upload' -> CSV upload page.
@app.callback(Output('page-content','children'), Input('url','pathname'))
def route(pathname: str):
    if pathname == '/upload':
        return get_upload_layout()
    return get_main_layout()

# ---------- Callbacks ----------
# Populate the backtest dropdown when a project is selected.
@app.callback(Output('backtest-dd','options'), Input('project-dd','value'))
def update_backtests(project_path):
    if not project_path:
        return []
    return list_backtests(project_path)

# Main visualization callback: loads the selected backtest, builds series, figures,
# stats, extra charts, and trades table.
@app.callback(
    Output('chart','figure'),
    Output('stats-panel','children'),
    Output('trades-table','children'),
    Output('extra-charts','children'),
    Output('orders-table','children'),
    Input('backtest-dd','value'),
    Input('resample-btn','n_clicks'),
    State('resample-value','value'),
    State('resample-unit','value')
)
def update_visual(backtest_folder, _resample_clicks, resample_value, resample_unit):
    if not backtest_folder:
        return {}, '', '', [], ''
    jsons = load_backtest_folder(backtest_folder)
    # aggregate charts
    charts = {}
    for name,data in jsons.items():
        if isinstance(data, dict) and 'charts' in data:
            # prefer summary charts; merge others
            for k,v in data.get('charts', {}).items():
                charts.setdefault(k, v)
    series_map = extract_series(charts)
    # Always load OHLCV from CSV using project config EquityName
    project_path = os.path.dirname(os.path.dirname(backtest_folder))
    data_root = os.path.join(WORKSPACE_ROOT, 'data')
    price_df = None
    price_source_note = None
    price_diag = {}
    freq_value, freq_unit = normalize_frequency(resample_value, resample_unit)
    freq_label = format_frequency_label(freq_value, freq_unit)
    expected_interval = frequency_to_timedelta(freq_value, freq_unit)
    try:
        csv_df, _csv_path, price_diag = load_ohlcv_from_csv(project_path, data_root)
        print('[app] Price loader diagnostics:', csv_df.head(10))
    except Exception as e:
        csv_df, _csv_path, price_diag = None, None, {'error': f'Loader exception: {e}'}
    if csv_df is not None and not csv_df.empty:
        price_df = resample_ohlcv(csv_df, freq_value, freq_unit)
        try:
            price_source_note = f"Price source: {os.path.basename(_csv_path)}"
        except Exception:
            price_source_note = "Price loaded from CSV"
    # As a fallback, attempt to build price data from the backtest charts when CSV is missing.
    if (price_df is None or price_df.empty) and series_map:
        built_price = build_price_from_series(series_map)
        if built_price is not None and not built_price.empty:
            price_df = resample_ohlcv(built_price, freq_value, freq_unit)
            price_source_note = "Price source: backtest chart series"
    indicators = compute_indicators(price_df)
    equity, drawdown, returns = build_equity_and_drawdown(series_map)
    events_df = parse_order_events(jsons)
    orders_df = enrich_orders(jsons)
    trades_df = reconstruct_trades(jsons)
    trades_df = build_trade_table(trades_df, orders_df, events_df)
    orders_df = build_order_table(orders_df, events_df)

    fig = build_figure(price_df, indicators, equity, drawdown, returns, trades_df, expected_interval)

    # Build additional charts to render below the main chart
    extra_blocks = []
    for title, prefix in [
        ('Portfolio Margin', 'Portfolio Margin'),
        ('Portfolio Turnover', 'Portfolio Turnover'),
        ('Exposure', 'Exposure'),
        ('Strategy Capacity', 'Capacity')
    ]:
        series = get_chart_series(series_map, title)
        if not series:
            # Try alternative label if different
            series = get_chart_series(series_map, prefix)
        if not series:
            continue
        # Build a small figure per block
        mini = make_subplots(rows=1, cols=1, shared_xaxes=True)
        for name, s in series.items():
            mini.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines', name=name))
        mini.update_layout(height=250, margin=dict(l=40,r=10,t=30,b=30), legend_orientation='h', showlegend=True,
                           title=title)
        # Append this mini figure to the extra blocks
        extra_blocks.append(dcc.Graph(figure=mini))

    # stats summarization (combine multiple stat sources)
    stats = {}
    for name,data in jsons.items():
        if not isinstance(data, dict):
            continue
        if 'statistics' in data:
            stats.update(data['statistics'])
        tp = data.get('totalPerformance', {}) or {}
        if isinstance(tp, dict):
            trade_stats = tp.get('tradeStatistics', {}) or {}
            portfolio_stats = tp.get('portfolioStatistics', {}) or {}
            # Prefix to avoid key collisions
            stats.update({f"Trade {k}": v for k,v in trade_stats.items() if k not in ['startDateTime','endDateTime']})
            stats.update({f"Portfolio {k}": v for k,v in portfolio_stats.items()})
        runtime = data.get('runtimeStatistics', {}) or {}
        if runtime:
            stats.update({f"Runtime {k}": v for k,v in runtime.items()})
    stats_items = _render_stats(stats)
    # Build optional notices
    notices = []
    if price_df is None:
        notices.append(html.Div([
            html.B('Price data not loaded.'), html.Br(),
            html.Span(f"Reason: {price_diag.get('error','unknown')}") ,
            html.Ul([
                html.Li('Ensure a CSV exists inside folder named as EquityName (case-insensitive).'),
                html.Li("Expected path: data/**/<EquityName>/original*.csv")
            ])
        ], style={'color':'#b36b00', 'background':'#fff7e6', 'border':'1px solid #ffe58f', 'padding':'6px 10px', 'margin':'8px 0'}))
    else:
        if price_source_note:
            notice_text = f"{price_source_note} | Resolution: {freq_label}"
        else:
            notice_text = f"Resolution: {freq_label}"
        notices.append(html.Div(notice_text, style={'color':'#135200','background':'#f6ffed','border':'1px solid #b7eb8f','padding':'4px 8px','margin':'8px 0','borderRadius':'4px'}))

    # trades table
    if not trades_df.empty:
        trade_table = dash_table.DataTable(
            columns=[{'name':c,'id':c} for c in trades_df.columns],
            data=trades_df.to_dict('records'),
            page_size=15,
            style_table={'overflowX':'auto'}
        )
    else:
        trade_table = html.Div('No closed trades found.')

    if not orders_df.empty:
        order_table = dash_table.DataTable(
            columns=[{'name':c,'id':c} for c in orders_df.columns],
            data=orders_df.to_dict('records'),
            page_size=15,
            style_table={'overflowX':'auto'}
        )
    else:
        order_table = html.Div('No orders found.')

    stats_component = html.Div([
        _render_stats_panel(stats),
        *notices
    ])
    return fig, stats_component, trade_table, extra_blocks, order_table

# Register upload callbacks (after app defined)
register_upload_callbacks(app)

if __name__ == '__main__':
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=8050,
        dev_tools_hot_reload=True,
        dev_tools_hot_reload_interval=1000,
        dev_tools_hot_reload_max_retry=8
    )
