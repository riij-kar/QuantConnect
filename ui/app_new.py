import os, glob, json, pandas as pd, numpy as np
from copy import deepcopy
from datetime import datetime
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.chart_utils import (
    extract_series,
    build_price_from_series,
    build_equity_and_drawdown,
    get_chart_series,
    build_lightweight_price_payload,
    build_lightweight_overlay_payload,
    build_lightweight_markers,
    build_echarts_indicator_payload
)
try:
    from .tradingview import PriceVolumeChart
except ImportError:
    from tradingview import PriceVolumeChart
try:
    from .echarts import EChartsPanel
except ImportError:
    from echarts import EChartsPanel
from utils.visual_indicators import compute_visual_indicators, IndicatorBundle
from utils.trade_mapper import (
    load_all_json,
    parse_order_events,
    enrich_orders,
    reconstruct_trades,
    build_trade_table,
    build_order_table,
    build_trade_order_table
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

INDICATOR_CHECKLIST_OPTIONS = [
    ("sma", "Simple MA"),
    ("ema", "Exponential MA"),
    ("bbands", "Bollinger Bands"),
    ("supertrend", "Supertrend"),
    ("vwap", "VWAP"),
    ("atr", "ATR"),
    ("rsi", "RSI"),
    ("macd", "MACD"),
]

INDICATOR_CONFIG_KEY_MAP = {
    'moving-average': 'sma',
    'exponential-moving-average': 'ema',
    'bollinger-bands': 'bbands',
    'supertrend': 'supertrend',
    'vwap': 'vwap',
    'atr': 'atr',
    'rsi': 'rsi',
    'macd': 'macd'
}

CHECKLIST_VALUE_ORDER = [value for value, _ in INDICATOR_CHECKLIST_OPTIONS]

DEFAULT_SMA_PERIOD_OPTIONS = [5, 8, 9, 10, 12, 15, 21, 34, 55, 89, 144, 200]
DEFAULT_EMA_PERIOD_OPTIONS = [5, 8, 9, 12, 21, 34, 55, 89, 144, 200]

PLOTLY_PRIMARY = '#636efa'
PLOTLY_SECONDARY = '#ef553b'
PLOTLY_POSITIVE = '#2ca02c'
PLOTLY_NEGATIVE = '#d62728'
PLOTLY_COMPACT_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'responsive': True,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
}
PLOTLY_RSI_COLOR = '#d946ef'
PLOTLY_RSI_SMA_COLOR = '#f97316'
PLOTLY_MACD_LINE = '#0ea5e9'
PLOTLY_MACD_SIGNAL = '#8b5cf6'
PLOTLY_MACD_HIST_POS = '#16a34a'
PLOTLY_MACD_HIST_NEG = '#dc2626'
PLOTLY_ATR_COLOR = '#fb923c'
PLOTLY_INDICATOR_BG = '#060816'
PLOTLY_INDICATOR_GRID = 'rgba(148,163,184,0.28)'
PLOTLY_AXIS_LINE = '#1f2937'
PLOTLY_TICK_COLOR = '#cbd5f5'
PLOTLY_SPIKE_COLOR = 'rgba(226,232,240,0.45)'
RSI_LEVEL_TEXT_COLORS = {
    'OVERBOUGHT': '#f97316',
    'MIDDLE': '#fbbf24',
    'OVERSOLD': '#22d3ee'
}
RSI_ZONE_FILL = {
    'overbought': 'rgba(249,115,22,0.08)',
    'oversold': 'rgba(34,197,94,0.12)'
}

SIDEBAR_BASE_STYLE = {
    'flex': '0 0 320px',
    'padding': '10px 12px',
    'borderRight': '1px solid #eee',
    'background': '#fafafa',
    'height': '100vh',
    'overflowY': 'auto'
}

MAIN_BASE_STYLE = {
    'flex': '1 1 auto',
    'height': '100vh',
    'overflowY': 'auto'
}

def _sidebar_style(collapsed: bool = False) -> Dict[str, Any]:
    style = SIDEBAR_BASE_STYLE.copy()
    if collapsed:
        style.update({'display': 'none'})
    return style


def _main_style(sidebar_collapsed: bool = False) -> Dict[str, Any]:
    style = MAIN_BASE_STYLE.copy()
    if sidebar_collapsed:
        style.update({'flex': '1 1 100%', 'width': '100%'})
    return style


def _build_indicator_legend(entries: List[Dict[str, Any]]) -> List[html.Div]:
    """Render a simple color legend for active overlays."""
    if not entries:
        return []
    chips: List[html.Div] = []
    for entry in entries:
        label = str(entry.get('label', '')).strip()
        if not label:
            continue
        color = entry.get('color') or '#555555'
        chip = html.Div([
            html.Span(className='legend-color-dot', style={
                'display': 'inline-block',
                'width': '10px',
                'height': '10px',
                'borderRadius': '50%',
                'backgroundColor': color,
                'marginRight': '6px'
            }),
            html.Span(label, style={'fontSize': '12px', 'color': '#e2e8f0'})
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'marginRight': '14px',
            'marginBottom': '6px'
        })
        chips.append(chip)
    return chips


def _default_indicator_selection(config: Optional[Dict[str, Any]]) -> List[str]:
    """Return checklist values enabled by the project indicator configuration."""
    if not config:
        return []
    defaults: List[str] = []
    for config_key, checklist_value in INDICATOR_CONFIG_KEY_MAP.items():
        if config.get(config_key):
            defaults.append(checklist_value)
    return defaults


def _ordered_indicator_values(values: Set[str]) -> List[str]:
    """Return checklist selections in canonical display order."""
    if not values:
        return []
    normalized = {str(v) for v in values}
    ordered: List[str] = []
    for candidate in CHECKLIST_VALUE_ORDER:
        if candidate in normalized:
            ordered.append(candidate)
    return ordered


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_plotly_indicator_blocks(
    indicator_bundle: Optional[IndicatorBundle],
    expected_interval: Optional[pd.Timedelta] = None
) -> List[html.Div]:
    oscillators = (indicator_bundle.oscillators if indicator_bundle else {}) or {}
    if not oscillators:
        return []

    container_style = {
        'marginBottom': '18px',
        'padding': '12px',
        'background': '#0b1220',
        'border': '1px solid #1f2937',
        'borderRadius': '8px',
        'boxShadow': '0 10px 25px -18px rgba(15, 23, 42, 0.65)'
    }

    def _normalize_timestamp(value: Any) -> Optional[pd.Timestamp]:
        try:
            ts = pd.Timestamp(value)
        except Exception:
            return None
        if ts is pd.NaT:
            return None
        if ts.tzinfo is not None:
            try:
                ts = ts.tz_convert('UTC')
            except Exception:
                pass
            try:
                ts = ts.tz_localize(None)
            except TypeError:
                pass
        return pd.Timestamp(ts)

    def _collect_axis_values(series: Optional[pd.Series], target: Set[pd.Timestamp]) -> None:
        if series is None:
            return
        for ts in getattr(series, 'index', []):
            norm = _normalize_timestamp(ts)
            if norm is not None:
                target.add(norm)

    valid_entries: List[tuple[str, Dict[str, Any]]] = []
    axis_values: Set[pd.Timestamp] = set()

    for label, spec in oscillators.items():
        if not isinstance(spec, dict):
            continue
        panel_type = spec.get('type')
        if panel_type == 'line':
            series = spec.get('series') if isinstance(spec.get('series'), pd.Series) else None
            if series is None:
                continue
            numeric = pd.to_numeric(series, errors='coerce')
            if numeric.dropna().empty:
                continue
            smoothing_struct = None
            smoothing_spec = spec.get('smoothing') if isinstance(spec.get('smoothing'), dict) else None
            if smoothing_spec:
                smoothing_series = smoothing_spec.get('series') if isinstance(smoothing_spec.get('series'), pd.Series) else None
                if smoothing_series is not None:
                    smoothing_numeric = pd.to_numeric(smoothing_series, errors='coerce')
                    if not smoothing_numeric.dropna().empty:
                        smoothing_struct = {**smoothing_spec, 'series': smoothing_numeric}
                        _collect_axis_values(smoothing_numeric, axis_values)
            spec = {**spec, 'series': numeric}
            if smoothing_struct is not None:
                spec['smoothing'] = smoothing_struct
            elif 'smoothing' in spec:
                spec.pop('smoothing', None)
            valid_entries.append((str(label), spec))
            _collect_axis_values(numeric, axis_values)
        elif panel_type == 'macd':
            macd_ser = spec.get('macd') if isinstance(spec.get('macd'), pd.Series) else None
            signal_ser = spec.get('signal') if isinstance(spec.get('signal'), pd.Series) else None
            hist_ser = spec.get('hist') if isinstance(spec.get('hist'), pd.Series) else None
            any_series = [s for s in (macd_ser, signal_ser, hist_ser) if s is not None]
            if not any_series:
                continue
            numeric_map = {
                key: pd.to_numeric(s, errors='coerce') if s is not None else None
                for key, s in [('macd', macd_ser), ('signal', signal_ser), ('hist', hist_ser)]
            }
            if all(s is None or s.dropna().empty for s in numeric_map.values()):
                continue
            merged_spec = {**spec, **numeric_map}
            valid_entries.append((str(label), merged_spec))
            for series in numeric_map.values():
                _collect_axis_values(series, axis_values)

    if not valid_entries or not axis_values:
        return []

    axis_list = sorted(axis_values)
    axis_lookup = {ts: idx for idx, ts in enumerate(axis_list)}

    def _map_index(idx_like: Iterable[Any]) -> List[Optional[int]]:
        mapped: List[Optional[int]] = []
        for ts in idx_like:
            norm = _normalize_timestamp(ts)
            mapped.append(axis_lookup.get(norm))
        return mapped

    def _format_tick(ts: pd.Timestamp) -> str:
        try:
            if expected_interval is not None and expected_interval >= pd.Timedelta(days=1):
                return ts.strftime('%b %d')
            return ts.strftime('%b %d\n%H:%M')
        except Exception:
            return str(ts)

    def _yref(row_index: int) -> str:
        return 'y' if row_index == 1 else f'y{row_index}'

    subplot_titles = [label for label, _ in valid_entries]
    row_count = len(valid_entries)
    fig = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=tuple(subplot_titles)
    )

    rsi_rows: Set[int] = set()
    rsi_zone_map: Dict[int, Dict[str, Optional[float]]] = {}
    level_annotation_queue: List[Tuple[int, str, float]] = []

    for row_idx, (label, spec) in enumerate(valid_entries, start=1):
        panel_type = spec.get('type')
        upper_label = label.upper()
        if panel_type == 'line':
            series = spec.get('series')
            assert isinstance(series, pd.Series)
            x_positions = _map_index(series.index)
            y_values = [None if pd.isna(val) else float(val) for val in series]
            trace_cls = go.Scattergl if len(series) > 3000 else go.Scatter
            line_color = PLOTLY_PRIMARY
            smoothing_spec = spec.get('smoothing') if isinstance(spec.get('smoothing'), dict) else None
            custom_data = None
            hover_template = '%{y:.2f}<extra></extra>'
            if 'RSI' in upper_label:
                line_color = PLOTLY_RSI_COLOR
                rsi_rows.add(row_idx)
                zone_data = {
                    'overbought': _safe_float(spec.get('overbought')),
                    'oversold': _safe_float(spec.get('oversold')),
                    'middle': _safe_float(spec.get('middle'))
                }
                rsi_zone_map[row_idx] = zone_data
                overbought_val = zone_data.get('overbought')
                oversold_val = zone_data.get('oversold')
                middle_val = zone_data.get('middle')

                def _zone_label(value: float) -> str:
                    if overbought_val is not None and not pd.isna(overbought_val) and value >= overbought_val:
                        return f"Overbought >= {overbought_val:.0f}"
                    if oversold_val is not None and not pd.isna(oversold_val) and value <= oversold_val:
                        return f"Oversold <= {oversold_val:.0f}"
                    if middle_val is not None and not pd.isna(middle_val):
                        return "Above Middle" if value >= middle_val else "Below Middle"
                    return "Neutral"

                custom_data = []
                for val in series:
                    if pd.isna(val):
                        custom_data.append('')
                    else:
                        custom_data.append(_zone_label(float(val)))
                hover_template = '%{y:.2f} | %{customdata}<extra></extra>'
            elif 'ATR' in upper_label:
                line_color = PLOTLY_ATR_COLOR
            trace = trace_cls(
                x=x_positions,
                y=y_values,
                mode='lines',
                name=label,
                line=dict(color=line_color, width=2.0),
                connectgaps=False,
                hovertemplate=hover_template,
                customdata=custom_data
            )
            fig.add_trace(trace, row=row_idx, col=1)
            if smoothing_spec:
                smoothing_series = smoothing_spec.get('series') if isinstance(smoothing_spec.get('series'), pd.Series) else None
                if smoothing_series is not None and not smoothing_series.dropna().empty:
                    x_smooth = _map_index(smoothing_series.index)
                    y_smooth = [None if pd.isna(val) else float(val) for val in smoothing_series]
                    ma_name = smoothing_spec.get('label') or f"{label} MA"
                    smooth_trace = trace_cls(
                        x=x_smooth,
                        y=y_smooth,
                        mode='lines',
                        name=ma_name,
                        line=dict(color=PLOTLY_RSI_SMA_COLOR, width=1.8),
                        connectgaps=False,
                        hovertemplate='%{y:.2f}<extra></extra>'
                    )
                    fig.add_trace(smooth_trace, row=row_idx, col=1)
            for level_label, level_value in spec.get('levels', []) or []:
                numeric_level = _safe_float(level_value)
                if numeric_level is None or pd.isna(numeric_level):
                    continue
                level_key = str(level_label).upper()
                level_color = RSI_LEVEL_TEXT_COLORS.get(level_key, '#94a3b8')
                level_dash = 'dot' if level_key == 'MIDDLE' else 'dash'
                level_trace = go.Scatter(
                    x=x_positions,
                    y=[numeric_level] * len(x_positions),
                    mode='lines',
                    name=f"{label} {level_label}",
                    line=dict(color=level_color, width=1.1, dash=level_dash),
                    hoverinfo='skip'
                )
                level_trace.showlegend = False
                fig.add_trace(level_trace, row=row_idx, col=1)
                level_annotation_queue.append((row_idx, str(level_label), numeric_level))
        elif panel_type == 'macd':
            hist_series = spec.get('hist') if isinstance(spec.get('hist'), pd.Series) else None
            macd_series = spec.get('macd') if isinstance(spec.get('macd'), pd.Series) else None
            signal_series = spec.get('signal') if isinstance(spec.get('signal'), pd.Series) else None
            if hist_series is not None and not hist_series.dropna().empty:
                x_hist = _map_index(hist_series.index)
                y_hist = [None if pd.isna(val) else float(val) for val in hist_series]
                hist_colors = [
                    'rgba(0,0,0,0)' if val is None else (PLOTLY_MACD_HIST_POS if val >= 0 else PLOTLY_MACD_HIST_NEG)
                    for val in y_hist
                ]
                hist_trace = go.Bar(
                    x=x_hist,
                    y=y_hist,
                    marker=dict(color=hist_colors, line=dict(width=0)),
                    opacity=0.65,
                    name=f'{label} Hist'
                )
                hist_trace.showlegend = False
                fig.add_trace(hist_trace, row=row_idx, col=1)
            if macd_series is not None and not macd_series.dropna().empty:
                x_macd = _map_index(macd_series.index)
                y_macd = [None if pd.isna(val) else float(val) for val in macd_series]
                macd_trace = go.Scatter(
                    x=x_macd,
                    y=y_macd,
                    mode='lines',
                    name=f'{label} MACD',
                    line=dict(color=PLOTLY_MACD_LINE, width=2.0),
                    connectgaps=False
                )
                fig.add_trace(macd_trace, row=row_idx, col=1)
            if signal_series is not None and not signal_series.dropna().empty:
                x_signal = _map_index(signal_series.index)
                y_signal = [None if pd.isna(val) else float(val) for val in signal_series]
                signal_trace = go.Scatter(
                    x=x_signal,
                    y=y_signal,
                    mode='lines',
                    name=f'{label} Signal',
                    line=dict(color=PLOTLY_MACD_SIGNAL, width=1.6, dash='dash'),
                    connectgaps=False
                )
                fig.add_trace(signal_trace, row=row_idx, col=1)

    total_points = len(axis_list)
    if total_points == 0:
        return []

    max_ticks = 12
    step = max(1, total_points // max_ticks)
    tick_positions = list(range(0, total_points, step))
    if (total_points - 1) not in tick_positions:
        tick_positions.append(total_points - 1)
    tick_text = [_format_tick(axis_list[idx]) for idx in tick_positions]
    spike_style = dict(
        showspikes=True,
        spikemode='across+toaxis+marker',
        spikesnap='cursor',
        spikecolor=PLOTLY_SPIKE_COLOR,
        spikedash='dot',
        spikethickness=1
    )

    base_axis_kwargs = dict(
        type='linear',
        range=[-0.5, total_points - 0.5],
        tickmode='array',
        tickvals=tick_positions,
        ticktext=tick_text,
        automargin=True,
        **spike_style
    )

    for row_idx in range(1, row_count + 1):
        fig.update_xaxes(row=row_idx, col=1, **base_axis_kwargs)
        show_ticks = (row_idx == row_count)
        fig.update_xaxes(
            row=row_idx,
            col=1,
            showticklabels=show_ticks,
            ticklabelposition='outside bottom' if show_ticks else 'outside top',
            ticks='outside',
            ticklen=4,
            tickfont=dict(size=10, color=PLOTLY_TICK_COLOR),
            showline=True,
            linecolor=PLOTLY_AXIS_LINE,
            rangeslider=dict(visible=False)
        )
        fig.update_yaxes(
            row=row_idx,
            col=1,
            showgrid=True,
            gridcolor=PLOTLY_INDICATOR_GRID,
            zeroline=False,
            tickfont=dict(color=PLOTLY_TICK_COLOR),
            linecolor=PLOTLY_AXIS_LINE
        )

    for row_idx in rsi_rows:
        fig.update_yaxes(row=row_idx, col=1, range=[0, 100])

    iso_meta: List[str] = []
    epoch_meta: List[int] = []
    for ts in axis_list:
        if hasattr(ts, 'isoformat'):
            try:
                iso_meta.append(ts.isoformat())
            except Exception:
                iso_meta.append(str(ts))
        else:
            iso_meta.append(str(ts))
        try:
            epoch_meta.append(int(pd.Timestamp(ts).value // 1_000_000))
        except Exception:
            epoch_meta.append(0)

    fig.update_layout(
        height=max(360, 260 * row_count),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.03, x=0, font=dict(size=10, color=PLOTLY_TICK_COLOR)),
        margin=dict(l=60, r=130, t=60, b=60),
        paper_bgcolor=PLOTLY_INDICATOR_BG,
        plot_bgcolor=PLOTLY_INDICATOR_BG,
        font=dict(color=PLOTLY_TICK_COLOR),
        hoverlabel=dict(bgcolor='#172033', font=dict(color='#f8fafc')),
        title=None,
        meta={'axisTimestamps': iso_meta, 'axisEpochMillis': epoch_meta}
    )

    for row_idx in sorted(rsi_rows):
        yaxis_layout_key = 'yaxis' if row_idx == 1 else f'yaxis{row_idx}'
        axis_obj = getattr(fig.layout, yaxis_layout_key, None)
        domain = getattr(axis_obj, 'domain', None) if axis_obj is not None else None
        if not domain:
            continue
        fig.add_shape(
            type='rect',
            xref='paper',
            yref='paper',
            x0=0,
            x1=1,
            y0=domain[0],
            y1=domain[1],
            fillcolor='rgba(6,8,22,0.9)',
            line=dict(width=0),
            layer='below'
        )
        zone_info = rsi_zone_map.get(row_idx, {})
        oversold_val = zone_info.get('oversold')
        overbought_val = zone_info.get('overbought')
        if oversold_val is not None and not pd.isna(oversold_val):
            oversold_clamped = max(0.0, min(float(oversold_val), 100.0))
            if oversold_clamped > 0:
                fig.add_shape(
                    type='rect',
                    xref='paper',
                    yref=_yref(row_idx),
                    x0=0,
                    x1=1,
                    y0=0,
                    y1=oversold_clamped,
                    fillcolor=RSI_ZONE_FILL['oversold'],
                    line=dict(width=0),
                    layer='below'
                )
        if overbought_val is not None and not pd.isna(overbought_val):
            overbought_clamped = max(0.0, min(float(overbought_val), 100.0))
            if overbought_clamped < 100:
                fig.add_shape(
                    type='rect',
                    xref='paper',
                    yref=_yref(row_idx),
                    x0=0,
                    x1=1,
                    y0=overbought_clamped,
                    y1=100,
                    fillcolor=RSI_ZONE_FILL['overbought'],
                    line=dict(width=0),
                    layer='below'
                )

    seen_level_annotations: Set[Tuple[int, str]] = set()
    for row_idx, level_label, level_value in level_annotation_queue:
        if level_value is None or pd.isna(level_value):
            continue
        key = (row_idx, level_label)
        if key in seen_level_annotations:
            continue
        seen_level_annotations.add(key)
        level_key = str(level_label).upper()
        text_color = RSI_LEVEL_TEXT_COLORS.get(level_key, PLOTLY_TICK_COLOR)
        value_text = f"{level_value:.2f}".rstrip('0').rstrip('.')
        fig.add_annotation(
            x=1.02,
            y=level_value,
            xref='paper',
            yref=_yref(row_idx),
            text=f"{level_label} {value_text}",
            showarrow=False,
            font=dict(color=text_color, size=10),
            align='left',
            yanchor='middle',
            bgcolor='rgba(15,23,42,0.78)',
            bordercolor='rgba(148,163,184,0.6)',
            borderwidth=0.7,
            borderpad=4,
            xanchor='left'
        )

    return [html.Div(dcc.Graph(figure=fig, config=PLOTLY_COMPACT_CONFIG), style=container_style)]

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "QC Local Dashboard"
# Allow large CSV uploads up to 200MB (adjust as needed)
app.server.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

# ---------- Helpers ----------

# Scan the workspace root for candidate project directories that contain a
# 'backtests' folder with at least one backtest. Returns absolute paths.
def find_project_paths():
    """Discover QC projects under ``WORKSPACE_ROOT`` that contain backtests."""
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
    """Return Dash dropdown options for backtests within a project."""
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
    """Load all JSON artifacts in a backtest folder into a dictionary."""
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
    """Construct an OHLC/close DataFrame from the extracted series map."""
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


def _coerce_period_list(raw_values) -> List[int]:
    """Convert user input (dropdown selections or comma strings) into a list of unique ints."""
    values: List[int] = []
    if raw_values is None:
        return values
    if isinstance(raw_values, (int, float)):
        raw_values = [raw_values]
    elif isinstance(raw_values, str):
        raw_values = [raw_values]
    for item in raw_values:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            tokens = item
        else:
            tokens = str(item).replace(',', ' ').split()
        for token in tokens:
            if not token:
                continue
            try:
                values.append(int(float(token)))
            except (TypeError, ValueError):
                continue
    return sorted(set(values))


def _validate_timezone(tz_name: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Return (timezone, error_message)."""
    if not tz_name:
        return None, None
    try:
        tz_normalized = str(tz_name).strip()
        if not tz_normalized:
            return None, None
        # pandas validates the timezone string by attempting to use it
        pd.Timestamp.now(tz=tz_normalized)
        return tz_normalized, None
    except Exception:
        return None, f"Invalid timezone '{tz_name}' in project config; timestamps left unchanged."


def _convert_index_timezone(obj: Any, tz_name: Optional[str], naive_origin: Optional[str]) -> Any:
    """Return a copy of the DataFrame/Series with its index converted to ``tz_name``.

    ``naive_origin`` indicates the timezone to assume when the index is naive (e.g. ``'UTC'`` or target tz).
    """
    if obj is None or tz_name is None or not isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj
    idx = getattr(obj, 'index', None)
    if not isinstance(idx, pd.DatetimeIndex):
        return obj
    try:
        localized = idx
        if localized.tz is None:
            origin = naive_origin or 'UTC'
            localized = localized.tz_localize(origin)
        converted = localized.tz_convert(tz_name)
    except Exception:
        return obj
    clone = obj.copy()
    clone.index = converted
    return clone


def _convert_column_timezone(df: Optional[pd.DataFrame], column: str, tz_name: Optional[str], naive_origin: Optional[str]) -> Optional[pd.DataFrame]:
    """Convert a datetime-like column to the target timezone, returning a copy when modified."""
    if df is None or tz_name is None or column not in df.columns:
        return df
    try:
        converted = pd.to_datetime(df[column], errors='coerce')
    except Exception:
        return df
    if not pd.api.types.is_datetime64_any_dtype(converted):
        return df
    try:
        if converted.dt.tz is None:
            origin = naive_origin or 'UTC'
            converted = converted.dt.tz_localize(origin)
        converted = converted.dt.tz_convert(tz_name)
    except Exception:
        return df
    clone = df.copy()
    clone[column] = converted
    return clone

# Render statistics as inline spans with spacing and heuristic coloring.
def _render_stats(stats: dict):
    """Build styled Dash spans for headline statistics."""
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
    """Extract closed trades from performance JSON artifacts into a DataFrame."""
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
def build_figure(price_df, indicator_bundle: Optional[IndicatorBundle], equity, drawdown, returns, trades_df, expected_interval=None, show_entry_exit: bool = False):
    """Assemble the dashboard's multi-panel Plotly figure.

    Parameters
    ----------
    price_df : pandas.DataFrame or None
        Primary price series (OHLC or close-only) used for the first subplot.
    indicator_bundle : IndicatorBundle, optional
        Visual indicators to overlay on the price chart and oscillators.
    equity : pandas.DataFrame or pandas.Series
        Equity curve data used to render the equity subplot.
    drawdown : pandas.Series
        Drawdown percentage series aligned with equity.
    returns : pandas.Series
        Period-over-period return series (percentage) for the returns subplot.
    trades_df : pandas.DataFrame
        Closed trades used for plotting entry/exit markers and trade tables.
    expected_interval : pandas.Timedelta, optional
        Target bar width for spacing markers (used for entry/exit clustering).
    show_entry_exit : bool, optional
        Toggle to control whether entry/exit markers are displayed.

    Returns
    -------
    plotly.graph_objects.Figure
        Four-row subplot figure containing price, RSI/oscillators, equity, and
        return/drawdown series.
    """
    MAX_PRICE_POINTS = int(os.environ.get('MAX_PRICE_POINTS', '0'))
    MAX_TRADE_MARKERS = int(os.environ.get('MAX_TRADE_MARKERS', '400'))

    bundle = indicator_bundle or IndicatorBundle.empty()
    overlays_raw = bundle.overlays or {}
    oscillators_raw = bundle.oscillators or {}

    def _normalize_timestamp(value):
        try:
            ts = pd.Timestamp(value)
        except Exception:
            return None
        if ts.tzinfo is not None:
            ts = pd.Timestamp(ts.to_pydatetime().replace(tzinfo=None))
        return ts

    price_plot = None
    tmin = tmax = None
    if price_df is not None and not price_df.empty:
        if MAX_PRICE_POINTS and MAX_PRICE_POINTS > 0 and len(price_df) > MAX_PRICE_POINTS:
            price_plot = price_df.tail(MAX_PRICE_POINTS)
        else:
            price_plot = price_df
        tmin, tmax = price_plot.index.min(), price_plot.index.max()

    if tmin is not None:
        tmin = _normalize_timestamp(tmin)
    if tmax is not None:
        tmax = _normalize_timestamp(tmax)

    if tmin is None:
        for series_obj in overlays_raw.values():
            if isinstance(series_obj, pd.Series) and not series_obj.dropna().empty:
                tmin, tmax = series_obj.index.min(), series_obj.index.max()
                break
    if tmin is not None:
        tmin = _normalize_timestamp(tmin)
    if tmax is not None:
        tmax = _normalize_timestamp(tmax)

    if tmin is None:
        for spec in oscillators_raw.values():
            if not isinstance(spec, dict):
                continue
            found_index = None
            for value in spec.values():
                if isinstance(value, pd.Series) and not value.dropna().empty:
                    found_index = value.index
                    break
            if found_index is not None:
                tmin, tmax = found_index.min(), found_index.max()
                break
    if tmin is not None:
        tmin = _normalize_timestamp(tmin)
    if tmax is not None:
        tmax = _normalize_timestamp(tmax)

    def _slice_series(series_obj: Optional[pd.Series]) -> Optional[pd.Series]:
        if series_obj is None:
            return None
        try:
            if tmin is not None and tmax is not None:
                return series_obj.loc[tmin:tmax]
            return series_obj
        except Exception:
            return series_obj

    overlays_plot: Dict[str, pd.Series] = {}
    overlay_structs: List[tuple[str, Dict[str, Any]]] = []
    for name, series_obj in overlays_raw.items():
        if isinstance(series_obj, pd.Series):
            sliced = _slice_series(series_obj)
            if sliced is None or sliced.dropna().empty:
                continue
            overlays_plot[name] = sliced
        elif isinstance(series_obj, dict):
            overlay_structs.append((name, series_obj))

    oscillator_plot = []
    for name, spec in oscillators_raw.items():
        if not isinstance(spec, dict):
            continue
        plot_spec = {}
        for key, value in spec.items():
            if isinstance(value, pd.Series):
                plot_spec[key] = _slice_series(value)
            else:
                plot_spec[key] = value
        has_series = any(
            isinstance(value, pd.Series) and not value.dropna().empty
            for key, value in plot_spec.items()
            if key != 'levels'
        )
        if not has_series:
            continue
        oscillator_plot.append((name, plot_spec))

    def _slice_frame(obj):
        if obj is None:
            return None
        try:
            if tmin is not None and tmax is not None:
                return obj.loc[tmin:tmax]
            return obj
        except Exception:
            return obj

    equity_plot = _slice_frame(equity)
    drawdown_plot = _slice_frame(drawdown)
    returns_plot = _slice_frame(returns)

    trades_plot = trades_df
    if trades_df is not None and not trades_df.empty and tmin is not None and tmax is not None:
        def _within(ts):
            try:
                val = pd.to_datetime(ts)
            except Exception:
                return False
            if val is pd.NaT:
                return False
            norm = _normalize_timestamp(val)
            if norm is None:
                return False
            return (norm >= tmin) and (norm <= tmax)

        trades_plot = trades_df[trades_df.apply(lambda r: _within(r.get('entryTime')) or _within(r.get('exitTime')), axis=1)]
        if len(trades_plot) > MAX_TRADE_MARKERS:
            trades_plot = trades_plot.tail(MAX_TRADE_MARKERS)

    axis_values: set[pd.Timestamp] = set()

    def _extend_axis(idx_like):
        if idx_like is None:
            return
        for ts in idx_like:
            norm = _normalize_timestamp(ts)
            if norm is not None:
                axis_values.add(norm)

    if price_plot is not None:
        _extend_axis(price_plot.index)
    for ser in overlays_plot.values():
        _extend_axis(getattr(ser, 'index', None))
    for _, spec in overlay_structs:
        if not isinstance(spec, dict):
            continue
        if spec.get('type') == 'supertrend':
            for key in ('upper', 'lower', 'trend'):
                series_candidate = spec.get(key)
                if isinstance(series_candidate, pd.Series):
                    _extend_axis(series_candidate.index)
    for _, spec in oscillator_plot:
        for value in spec.values():
            if isinstance(value, pd.Series):
                _extend_axis(value.index)
    _extend_axis(getattr(equity_plot, 'index', None))
    _extend_axis(getattr(drawdown_plot, 'index', None))
    _extend_axis(getattr(returns_plot, 'index', None))
    if trades_plot is not None and not trades_plot.empty:
        if 'entryTime' in trades_plot.columns:
            _extend_axis(pd.to_datetime(trades_plot['entryTime'], errors='coerce'))
        if 'exitTime' in trades_plot.columns:
            _extend_axis(pd.to_datetime(trades_plot['exitTime'], errors='coerce'))

    axis_list = sorted(axis_values)
    axis_lookup = {ts: idx for idx, ts in enumerate(axis_list)}

    def _map_index(idx_like):
        mapped = []
        for ts in idx_like:
            norm = _normalize_timestamp(ts)
            if norm is None:
                mapped.append(None)
            else:
                mapped.append(axis_lookup.get(norm))
        return mapped

    def _format_tick(ts: pd.Timestamp) -> str:
        if expected_interval is not None and expected_interval >= pd.Timedelta(days=1):
            return ts.strftime('%b %d')
        return ts.strftime('%b %d\n%H:%M')

    oscillator_count = len(oscillator_plot)
    row_heights = [0.32, 0.12] + ([0.16] * oscillator_count) + [0.2, 0.12, 0.08]
    subplot_titles = ['Price & Overlays', 'Volume'] + [name for name, _ in oscillator_plot] + ['Equity Curve', 'Return %', 'Drawdown']
    total_rows = len(row_heights)
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
        specs=[[{}] for _ in range(total_rows)],
        subplot_titles=tuple(subplot_titles)
    )

    legend_seen: set[str] = set()

    def _add_trace(trace, *, row: int, col: int, secondary_y: bool = False):
        name = getattr(trace, 'name', None)
        if name and trace.showlegend is not False:
            show = name not in legend_seen
            trace.showlegend = show
            if show:
                legend_seen.add(name)
        fig.add_trace(trace, row=row, col=col, secondary_y=secondary_y)

    price_row = 1
    volume_row = 2
    equity_row = 3 + oscillator_count
    returns_row = equity_row + 1
    drawdown_row = returns_row + 1

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
            price_positions = _map_index(price_plot.index)
            price_trace = go.Candlestick(
                x=price_positions,
                open=price_plot['open'],
                high=price_plot['high'],
                low=price_plot['low'],
                close=price_plot['close'],
                name='Price',
                hovertext=hover_text,
                hoverinfo='text'
            )
            _add_trace(price_trace, row=price_row, col=1)
        else:
            yser = price_plot['close'] if 'close' in price_plot.columns else price_plot.iloc[:, -1]
            trace_cls = go.Scattergl if len(yser) > 2000 else go.Scatter
            px = _map_index(yser.index)
            py = list(yser.values)
            price_line = trace_cls(
                x=px,
                y=py,
                mode='lines',
                name='Price',
                hovertemplate='Time: %{x|%b %d %H:%M}<br>Price: %{y}<extra></extra>',
                connectgaps=False
            )
            _add_trace(price_line, row=price_row, col=1)

        if 'volume' in price_plot.columns:
            vol = price_plot['volume'].fillna(0)
            vol_x = _map_index(vol.index)
            if all(col in price_plot.columns for col in ['open', 'close']):
                colors = ['#2ca02c' if c >= o else '#d62728' for o, c in zip(price_plot['open'], price_plot['close'])]
            else:
                colors = ['#2ca02c'] * len(vol)
            volume_bar = go.Bar(
                x=vol_x,
                y=vol.values,
                marker=dict(color=colors, line=dict(width=0)),
                name='Volume',
                opacity=0.7
            )
            _add_trace(volume_bar, row=volume_row, col=1)

    for name, ser in overlays_plot.items():
        if ser is None or ser.empty:
            continue
        trace_cls = go.Scattergl if len(ser) > 3000 else go.Scatter
        overlay_trace = trace_cls(
            x=_map_index(ser.index),
            y=ser.values,
            mode='lines',
            name=name,
            connectgaps=False
        )
        _add_trace(overlay_trace, row=price_row, col=1)

    supertrend_plot: List[tuple[str, pd.Series, pd.Series, pd.Series]] = []
    for name, spec in overlay_structs:
        if spec.get('type') != 'supertrend':
            continue
        upper_ser = spec.get('upper')
        lower_ser = spec.get('lower')
        trend_ser = spec.get('trend')
        if not all(isinstance(obj, pd.Series) for obj in (upper_ser, lower_ser, trend_ser)):
            continue
        upper_slice = _slice_series(upper_ser)
        lower_slice = _slice_series(lower_ser)
        trend_slice = _slice_series(trend_ser)
        if any(obj is None for obj in (upper_slice, lower_slice, trend_slice)):
            continue
        # Align indices so segment masks line up
        trend_slice = trend_slice.astype(float)
        common_index = trend_slice.index
        upper_slice = upper_slice.reindex(common_index)
        lower_slice = lower_slice.reindex(common_index)
        supertrend_plot.append((name, upper_slice, lower_slice, trend_slice))

    for name, upper_ser, lower_ser, trend_ser in supertrend_plot:
        if upper_ser.dropna().empty or lower_ser.dropna().empty or trend_ser.dropna().empty:
            continue

        valid_mask = ~(upper_ser.isna() | lower_ser.isna() | trend_ser.isna())
        if not valid_mask.any():
            continue

        trend_equals_lower = pd.Series(
            np.isclose(trend_ser.values, lower_ser.values, equal_nan=True),
            index=trend_ser.index
        )
        up_mask = valid_mask & trend_equals_lower
        down_mask = valid_mask & ~trend_equals_lower

        def _plot_supertrend_segment(mask: pd.Series, line_color: str, label_suffix: str):
            if not mask.any():
                return
            masked_trend = trend_ser.where(mask, np.nan)
            if masked_trend.dropna().empty:
                return

            x_vals = _map_index(masked_trend.index)

            trend_line = go.Scatter(
                x=x_vals,
                y=masked_trend.values,
                mode='lines',
                name=f"{name} {label_suffix}",
                line=dict(color=line_color, width=2),
                connectgaps=False,
                legendgroup=name,
                hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>"
            )
            _add_trace(trend_line, row=price_row, col=1)

        _plot_supertrend_segment(up_mask, 'rgba(34, 197, 94, 1)', '(Up)')
        _plot_supertrend_segment(down_mask, 'rgba(239, 68, 68, 1)', '(Down)')

    if show_entry_exit and trades_plot is not None and not trades_plot.empty:
        for _, r in trades_plot.iterrows():
            if 'entryTime' in r and 'entryPrice' in r:
                entry_x = _normalize_timestamp(r['entryTime'])
                if entry_x in axis_lookup:
                    entry_trace = go.Scatter(
                        x=[axis_lookup[entry_x]],
                        y=[r.get('entryPrice')],
                        mode='markers',
                        marker_symbol='triangle-up',
                        marker_color='green',
                        marker_size=10,
                        name='Entry',
                        hovertemplate='Entry Price: %{y}<extra></extra>'
                    )
                    _add_trace(entry_trace, row=price_row, col=1)
            if 'exitTime' in r and 'exitPrice' in r:
                exit_x = _normalize_timestamp(r['exitTime'])
                if exit_x in axis_lookup:
                    exit_trace = go.Scatter(
                        x=[axis_lookup[exit_x]],
                        y=[r.get('exitPrice')],
                        mode='markers',
                        marker_symbol='x',
                        marker_color='red',
                        marker_size=9,
                        name='Exit',
                        hovertemplate='Exit Price: %{y}<extra></extra>'
                    )
                    _add_trace(exit_trace, row=price_row, col=1)

    for idx, (name, spec) in enumerate(oscillator_plot):
        row_index = 3 + idx
        otype = spec.get('type')
        if otype == 'line':
            series = spec.get('series')
            if isinstance(series, pd.Series) and not series.empty:
                trace_cls = go.Scattergl if len(series) > 3000 else go.Scatter
                osc_trace = trace_cls(
                    x=_map_index(series.index),
                    y=series.values,
                    mode='lines',
                    name=name,
                    connectgaps=False
                )
                _add_trace(osc_trace, row=row_index, col=1)
                for level_label, level_value in spec.get('levels', []) or []:
                    if level_value is None:
                        continue
                    level_trace = go.Scatter(
                        x=_map_index(series.index),
                        y=[float(level_value)] * len(series),
                        mode='lines',
                        name=f"{name} {level_label}",
                        line=dict(color='#999999', width=1, dash='dash'),
                        hoverinfo='skip'
                    )
                    level_trace.showlegend = False
                    _add_trace(level_trace, row=row_index, col=1)
        elif otype == 'macd':
            hist = spec.get('hist')
            macd_series = spec.get('macd')
            signal_series = spec.get('signal')
            if isinstance(hist, pd.Series) and not hist.empty:
                colors = ['#2ca02c' if v >= 0 else '#d62728' for v in hist.values]
                hist_trace = go.Bar(
                    x=_map_index(hist.index),
                    y=hist.values,
                    marker=dict(color=colors, line=dict(width=0)),
                    opacity=0.65,
                    name=f"{name} Hist"
                )
                hist_trace.showlegend = False
                _add_trace(hist_trace, row=row_index, col=1)
            if isinstance(macd_series, pd.Series) and not macd_series.empty:
                macd_trace = go.Scatter(
                    x=_map_index(macd_series.index),
                    y=macd_series.values,
                    mode='lines',
                    name=f"{name} MACD",
                    line=dict(width=1.8)
                )
                _add_trace(macd_trace, row=row_index, col=1)
            if isinstance(signal_series, pd.Series) and not signal_series.empty:
                signal_trace = go.Scatter(
                    x=_map_index(signal_series.index),
                    y=signal_series.values,
                    mode='lines',
                    name=f"{name} Signal",
                    line=dict(width=1.4, dash='dash')
                )
                _add_trace(signal_trace, row=row_index, col=1)

    if equity_plot is not None:
        if isinstance(equity_plot, pd.DataFrame) and all(c in equity_plot.columns for c in ['open', 'high', 'low', 'close']):
            equity_positions = _map_index(equity_plot.index)
            equity_candle = go.Candlestick(
                x=equity_positions,
                open=equity_plot['open'],
                high=equity_plot['high'],
                low=equity_plot['low'],
                close=equity_plot['close'],
                name='Equity OHLC'
            )
            _add_trace(equity_candle, row=equity_row, col=1)
        else:
            ser = equity_plot
            eq_trace = go.Scattergl(
                x=_map_index(ser.index),
                y=getattr(ser, 'values', ser),
                mode='lines',
                name='Equity'
            )
            _add_trace(eq_trace, row=equity_row, col=1)

    if returns_plot is not None:
        ret_values = returns_plot.fillna(0)
        colors = ['#2ca02c' if v >= 0 else '#d62728' for v in ret_values.values]
        returns_bar = go.Bar(
            x=_map_index(ret_values.index),
            y=ret_values.values,
            marker=dict(color=colors, line=dict(width=0)),
            opacity=0.75,
            name='Return %'
        )
        _add_trace(returns_bar, row=returns_row, col=1)

    if drawdown_plot is not None:
        dd = drawdown_plot.fillna(0)
        drawdown_line = go.Scatter(
            x=_map_index(dd.index),
            y=dd.values,
            mode='lines',
            name='Drawdown',
            line=dict(color='#9467bd', width=1.6),
            fill='tozeroy',
            fillcolor='rgba(148, 103, 189, 0.25)',
            hovertemplate='Drawdown: %{y:.2%}<extra></extra>'
        )
        _add_trace(drawdown_line, row=drawdown_row, col=1)

    figure_height = max(900, 240 * total_rows + 120)
    fig.update_layout(
        height=figure_height,
        title=dict(text='Backtest Visualization', font=dict(color=PLOTLY_TICK_COLOR, size=22)),
        legend=dict(
            orientation='h',
            y=1.02,
            x=0,
            font=dict(size=11, color=PLOTLY_TICK_COLOR),
            bgcolor='rgba(15,23,42,0.55)',
            bordercolor='rgba(148,163,184,0.45)',
            borderwidth=0.6
        ),
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        hoverdistance=30,
        spikedistance=-1,
        margin=dict(t=90, l=70, r=120, b=90),
        paper_bgcolor=PLOTLY_INDICATOR_BG,
        plot_bgcolor=PLOTLY_INDICATOR_BG,
        font=dict(color=PLOTLY_TICK_COLOR),
        hoverlabel=dict(bgcolor='#172033', font=dict(color='#f8fafc'))
    )
    spike_style = dict(
        showspikes=True,
        spikemode='across+toaxis+marker',
        spikesnap='cursor',
        spikecolor=PLOTLY_SPIKE_COLOR,
        spikedash='dot',
        spikethickness=1,
        hoverformat='.2f'
    )
    fig.update_yaxes(
        row=price_row,
        col=1,
        tickfont=dict(color=PLOTLY_TICK_COLOR)
    )
    fig.update_yaxes(
        title_text='Volume',
        row=volume_row,
        col=1,
        titlefont=dict(color=PLOTLY_TICK_COLOR),
        tickfont=dict(color=PLOTLY_TICK_COLOR)
    )
    for idx, (name, _) in enumerate(oscillator_plot):
        fig.update_yaxes(
            title_text=name,
            row=3 + idx,
            col=1,
            titlefont=dict(color=PLOTLY_TICK_COLOR),
            tickfont=dict(color=PLOTLY_TICK_COLOR)
        )
    fig.update_yaxes(
        title_text='Equity',
        row=equity_row,
        col=1,
        titlefont=dict(color=PLOTLY_TICK_COLOR),
        tickfont=dict(color=PLOTLY_TICK_COLOR)
    )
    fig.update_yaxes(
        title_text='Return %',
        row=returns_row,
        col=1,
        titlefont=dict(color=PLOTLY_TICK_COLOR),
        tickfont=dict(color=PLOTLY_TICK_COLOR)
    )
    fig.update_yaxes(
        title_text='Drawdown',
        row=drawdown_row,
        col=1,
        tickformat='.1%',
        titlefont=dict(color=PLOTLY_TICK_COLOR),
        tickfont=dict(color=PLOTLY_TICK_COLOR)
    )

    for axis_row in range(1, total_rows + 1):
        fig.update_yaxes(
            row=axis_row,
            col=1,
            gridcolor=PLOTLY_INDICATOR_GRID,
            linecolor=PLOTLY_AXIS_LINE,
            zeroline=False,
            tickfont=dict(color=PLOTLY_TICK_COLOR),
            titlefont=dict(color=PLOTLY_TICK_COLOR)
        )

    try:
        total_points = len(axis_list)
        if total_points > 0:
            max_ticks = 12
            step = max(1, total_points // max_ticks)
            tick_positions = list(range(0, total_points, step))
            if (total_points - 1) not in tick_positions:
                tick_positions.append(total_points - 1)
            tick_text = [_format_tick(axis_list[i]) for i in tick_positions]

            base_axis_kwargs = dict(
                type='linear',
                range=[-0.5, total_points - 0.5],
                tickmode='array',
                tickvals=tick_positions,
                ticktext=tick_text,
                automargin=True,
                showgrid=True,
                gridcolor=PLOTLY_INDICATOR_GRID,
                linecolor=PLOTLY_AXIS_LINE,
                zeroline=False,
                tickfont=dict(size=10, color=PLOTLY_TICK_COLOR),
                ticks='outside',
                tickcolor=PLOTLY_TICK_COLOR,
                **spike_style
            )
            for r in range(1, total_rows + 1):
                fig.update_xaxes(row=r, col=1, **base_axis_kwargs)
            fig.update_xaxes(
                row=price_row,
                col=1,
                showticklabels=True,
                ticklabelposition='outside top',
                ticks='outside',
                ticklen=4,
                tickfont=dict(size=10, color=PLOTLY_TICK_COLOR)
            )
            for r in range(1, total_rows + 1):
                if r in (price_row, drawdown_row):
                    continue
                fig.update_xaxes(row=r, col=1, showticklabels=False)
            fig.update_xaxes(
                row=drawdown_row,
                col=1,
                showticklabels=True,
                tickangle=0,
                tickfont=dict(size=11, color=PLOTLY_TICK_COLOR)
            )
        fig.update_yaxes(**spike_style)
    except Exception:
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


def _entry_exit_enabled_for_project(project_path: str) -> bool:
    """Read the project config and return whether entry/exit signals should be shown."""
    config_path = os.path.join(project_path, 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as cfg_file:
            project_config = json.load(cfg_file)
    except Exception:
        return False
    ui_section = project_config.get('ui') or {}
    controls_section = ui_section.get('controls') or {}
    return bool(controls_section.get('entry_exit_signals'))

# ---------- Layout ----------
from pages.upload_page import get_upload_layout, register_upload_callbacks  # type: ignore

# Build the main page layout with project/backtest selectors, chart, stats bar,
# extra charts area, and trades table.
def get_main_layout():
    """Create the root Dash layout containing sidebar controls and charts."""
    # Two-column layout: left sidebar (selectors + stats), right main content (charts)
    sidebar = html.Div([
        html.Div([
            html.H3('QC Dashboard', style={'margin': '0'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
        html.Label('Project'),
        dcc.Dropdown(id='project-dd', options=[{'label': os.path.basename(p), 'value': p} for p in find_project_paths()], placeholder='Select project'),
        html.Label('Backtest', style={'marginTop':'6px'}),
        dcc.Dropdown(id='backtest-dd', options=[], placeholder='Select backtest folder'),
        html.Div([
            html.A('Go to Uploads', href='/upload')
        ], style={'marginTop':'8px'}),
        html.Hr(),
        html.Div(id='stats-panel'),
    ], id='sidebar-panel', style=_sidebar_style(False))

    main = html.Div([
        html.Div([
            html.Div('Backtest Visualization', style={'fontSize': '22px', 'fontWeight': '600'}),
            html.Div([
                html.Button('Entry/Exit Signals (Disabled)', id='toggle-entry-exit', n_clicks=0, disabled=True, style={'marginRight': '8px'}),
                html.Button('Toggle Sidebar', id='toggle-sidebar', n_clicks=0)
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'margin': '10px'}),
        dcc.Store(id='tradingview-price-volume-store'),
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
        ], style={'display':'flex', 'alignItems':'center', 'justifyContent':'flex-end', 'gap':'10px', 'margin':'0 10px 6px 10px'}),
        html.Div([
            html.Div('Indicators', style={'fontWeight':'600', 'color':'#444'}),
            dcc.Checklist(
                id='indicator-checklist',
                options=[{'label': label, 'value': value} for value, label in INDICATOR_CHECKLIST_OPTIONS],
                value=[],
                inputStyle={'marginRight':'4px'},
                labelStyle={'marginRight':'12px', 'marginBottom':'4px'},
                style={'display':'flex', 'flexWrap':'wrap'}
            ),
            html.Div([
                html.Div([
                    html.Label('SMA Periods', style={'color':'#555', 'fontSize':'12px'}),
                    dcc.Dropdown(
                        id='indicator-sma-periods',
                        options=[{'label': str(p), 'value': str(p)} for p in DEFAULT_SMA_PERIOD_OPTIONS],
                        value=[],
                        multi=True,
                        placeholder='Select SMA periods',
                        clearable=True,
                        style={'minWidth':'160px'}
                    )
                ], style={'minWidth':'180px'}),
                html.Div([
                    html.Label('EMA Periods', style={'color':'#555', 'fontSize':'12px'}),
                    dcc.Dropdown(
                        id='indicator-ema-periods',
                        options=[{'label': str(p), 'value': str(p)} for p in DEFAULT_EMA_PERIOD_OPTIONS],
                        value=[],
                        multi=True,
                        placeholder='Select EMA periods',
                        clearable=True,
                        style={'minWidth':'160px'}
                    )
                ], style={'minWidth':'180px'}),
            ], style={'display':'flex', 'flexWrap':'wrap', 'gap':'16px', 'marginTop':'6px'})
        ], style={'margin':'0 10px 16px 10px', 'padding':'10px', 'border':'1px solid #eee', 'borderRadius':'6px', 'background':'#fafafa'}),
        html.Div([
            html.Div('Price and Volume', style={'fontWeight': '600', 'fontSize': '16px', 'color': '#222', 'marginBottom': '6px'}),
            html.Div(id='indicator-legend', style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'center', 'marginBottom': '8px', 'minHeight': '16px'}),
            dcc.Loading(
                PriceVolumeChart(
                    id='tradingview-price-volume',
                    candles=[],
                    volume=[],
                    overlays={'lines': {}, 'supertrend': {}, 'legend': []},
                    markers=[],
                    chart_options={
                        'layout': {
                            'backgroundColor': '#060816',
                            'textColor': '#cbd5f5'
                        },
                        'grid': {
                            'vertLines': {'color': 'rgba(148,163,184,0.18)'},
                            'horzLines': {'color': 'rgba(148,163,184,0.18)'}
                        },
                        'rightPriceScale': {
                            'visible': True,
                            'borderColor': '#1f2937'
                        },
                        'leftPriceScale': {
                            'visible': False
                        },
                        'timeScale': {
                            'timeVisible': True,
                            'secondsVisible': False,
                            'borderColor': '#1f2937'
                        },
                        'crosshair': {
                            'mode': 0,
                            'vertLine': {
                                'color': 'rgba(94,234,212,0.65)',
                                'labelBackgroundColor': '#0f172a'
                            },
                            'horzLine': {
                                'color': 'rgba(94,234,212,0.65)',
                                'labelBackgroundColor': '#0f172a'
                            }
                        },
                        'watermark': {
                            'visible': False
                        }
                    },
                    style={'height': '600px'}
                ),
                type='default'
            ),
            html.Div([
                html.Div('Plotly Oscillators (RSI / MACD / ATR)', style={'fontWeight': '600', 'fontSize': '15px', 'color': '#1f2937', 'marginBottom': '6px'}),
                html.Div(id='extra-charts', style={'marginTop': '8px'})
            ], style={'marginTop': '18px'}),
            EChartsPanel(
                id='indicator-echarts-panel',
                config={'oscillators': [], 'performance': {}, 'analytics': []},
                style={'marginTop': '18px'}
            )
        ], id='chart-fullscreen-container', style={'padding':'0 10px'}),
        html.H4('Trades', style={'margin':'10px'}),
        html.Div(id='trades-table', style={'padding':'0 10px 10px 10px'}),
        html.H4('Trades & Orders', style={'margin':'10px'}),
        html.Div(id='trade-orders-table', style={'padding':'0 10px 10px 10px'}),
        html.H4('Orders', style={'margin':'10px'}),
        html.Div(id='orders-table', style={'padding':'0 10px 20px 10px'})
    ], id='main-panel', className='main-panel', style=_main_style(False))

    return html.Div([sidebar, main], id='layout-container', className='dashboard-root', style={'display':'flex', 'height':'100vh'})

app.layout = html.Div([
    dcc.Location(id='url'),
    html.Div(id='page-content'),
    dcc.Store(id='entry-exit-visible'),
    PriceVolumeChart.script_tag(),
    EChartsPanel.script_tag()
], id='root-container', className='dashboard-root-container')

# Simple router: '/' -> main dashboard, '/upload' -> CSV upload page.
@app.callback(Output('page-content','children'), Input('url','pathname'))
def route(pathname: str):
    """Return the appropriate Dash layout for the requested ``pathname``."""
    if pathname == '/upload':
        return get_upload_layout()
    return get_main_layout()

# ---------- Callbacks ----------
# Populate the backtest dropdown when a project is selected.
@app.callback(Output('backtest-dd','options'), Input('project-dd','value'))
def update_backtests(project_path):
    """Refresh the backtest dropdown options when a project selection changes."""
    if not project_path:
        return []
    return list_backtests(project_path)


# Main visualization callback: loads the selected backtest, builds series, figures,
# stats, extra charts, and trades table.
@app.callback(
    Output('tradingview-price-volume-store', 'data'),
    Output('tradingview-price-volume', 'data-candles'),
    Output('tradingview-price-volume', 'data-volume'),
    Output('tradingview-price-volume', 'data-overlays'),
    Output('tradingview-price-volume', 'data-markers'),
    Output('indicator-legend', 'children'),
    Output('indicator-echarts-panel', 'data-config'),
    Output('indicator-echarts-panel', 'data-last-render'),
    Output('tradingview-price-volume', 'data-last-render'),
    Output('stats-panel','children'),
    Output('trades-table','children'),
    Output('trade-orders-table','children'),
    Output('extra-charts','children'),
    Output('indicator-checklist','value'),
    Output('orders-table','children'),
    Input('backtest-dd','value'),
    Input('resample-btn','n_clicks'),
    Input('entry-exit-visible','data'),
    State('resample-value','value'),
    State('resample-unit','value'),
    State('indicator-checklist','value'),
    State('indicator-sma-periods','value'),
    State('indicator-ema-periods','value')
)
def update_visual(backtest_folder, _resample_clicks, entry_exit_visible, resample_value, resample_unit, indicator_selection, sma_period_values, ema_period_values):
    """Render charts, stats, and trade tables for the active backtest selection."""
    empty_payload = {
        'candles': [],
        'volume': [],
        'markers': [],
        'overlays': {'lines': {}, 'supertrend': {}, 'legend': []},
        'meta': {
            'hasPriceData': False,
            'entryExitEnabled': False,
            'resolution': None,
            'resampleValue': None,
            'resampleUnit': None,
            'priceSource': None,
            'timezone': None
        }
    }
    empty_overlays_json = json.dumps({'lines': {}, 'supertrend': {}, 'legend': []})
    if not backtest_folder:
        empty_json = json.dumps([])
        empty_config = json.dumps({'oscillators': [], 'performance': {}, 'analytics': []})
        return (
            empty_payload,
            empty_json,
            empty_json,
            empty_overlays_json,
            empty_json,
            [],
            empty_config,
            '',
            '',
            '',
            '',
            '',
            [],
            [],
            ''
        )
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
    indicator_config: Dict[str, Any] = {}
    indicator_load_messages: List[str] = []
    ui_timezone: Optional[str] = None
    timezone_message: Optional[str] = None
    entry_exit_config_enabled = False
    config_path = os.path.join(project_path, 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as cfg_file:
            project_config = json.load(cfg_file)
        ui_section = project_config.get('ui') or {}
        indicator_config = (ui_section.get('indicators')) or {}
        tz_candidate = ui_section.get('timezone') or project_config.get('timezone')
        ui_timezone, timezone_message = _validate_timezone(tz_candidate)
        controls_section = ui_section.get('controls') or {}
        entry_exit_config_enabled = bool(controls_section.get('entry_exit_signals'))
    except FileNotFoundError:
        indicator_load_messages.append("Indicator config not found; overlays/oscillators disabled.")
    except Exception as exc:
        indicator_load_messages.append(f"Indicator config load error: {exc}")
        entry_exit_config_enabled = False
    price_df = None
    price_source_note = None
    price_diag = {}
    freq_value, freq_unit = normalize_frequency(resample_value, resample_unit)
    freq_label = format_frequency_label(freq_value, freq_unit)
    expected_interval = frequency_to_timedelta(freq_value, freq_unit)
    try:
        csv_df, _csv_path, price_diag = load_ohlcv_from_csv(project_path, data_root, backtest_folder)
        # print('[app] Price loader diagnostics:', csv_df.head(10))
    except Exception as e:
        csv_df, _csv_path, price_diag = None, None, {'error': f'Loader exception: {e}'}
    if csv_df is not None and not csv_df.empty:
        if ui_timezone:
            csv_before_samples = [str(idx) for idx in csv_df.index[:5]]
            converted_csv = _convert_index_timezone(csv_df, ui_timezone, naive_origin=ui_timezone)
            csv_after_samples = [str(idx) for idx in converted_csv.index[:5]]
            print('[timezone] CSV index before conversion:', csv_before_samples)
            print('[timezone] CSV index after conversion:', csv_after_samples)
            csv_df = converted_csv
        price_df = resample_ohlcv(csv_df, freq_value, freq_unit)
        try:
            price_source_note = f"Price source: {os.path.basename(_csv_path)}"
        except Exception:
            price_source_note = "Price loaded from CSV"
    # As a fallback, attempt to build price data from the backtest charts when CSV is missing.
    if (price_df is None or price_df.empty) and series_map:
        built_price = build_price_from_series(series_map)
        if built_price is not None and not built_price.empty:
            if ui_timezone:
                built_before_samples = [str(idx) for idx in built_price.index[:5]]
                converted_built = _convert_index_timezone(built_price, ui_timezone, naive_origin='UTC')
                built_after_samples = [str(idx) for idx in converted_built.index[:5]]
                print('[timezone] Chart-series index before conversion:', built_before_samples)
                print('[timezone] Chart-series index after conversion:', built_after_samples)
                built_price = converted_built
            price_df = resample_ohlcv(built_price, freq_value, freq_unit)
            price_source_note = "Price source: backtest chart series"
    if price_df is not None and not price_df.empty and ui_timezone:
        price_df = _convert_index_timezone(price_df, ui_timezone, naive_origin=ui_timezone)
    base_indicator_config = indicator_config or {}
    selected_indicator_config: Dict[str, Any] = {}
    user_indicator_messages: List[str] = []
    selected_keys: Set[str] = {str(key) for key in (indicator_selection or []) if key is not None}
    triggered_component = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else None
    should_apply_defaults = (
        not selected_keys and (
            triggered_component in (None, 'backtest-dd') or
            (_resample_clicks or 0) == 0
        )
    )
    if should_apply_defaults:
        selected_keys.clear()
    applied_indicator_values = _ordered_indicator_values(selected_keys)

    source_value = base_indicator_config.get('source')
    if source_value and selected_keys:
        selected_indicator_config['source'] = source_value

    if 'sma' in selected_keys:
        sma_periods = _coerce_period_list(sma_period_values)
        if not sma_periods:
            sma_periods = _coerce_period_list(base_indicator_config.get('moving-average'))
        if sma_periods:
            selected_indicator_config['moving-average'] = sma_periods
        else:
            user_indicator_messages.append('Simple MA selected but no periods provided; skipping.')

    if 'ema' in selected_keys:
        ema_periods = _coerce_period_list(ema_period_values)
        if not ema_periods:
            ema_periods = _coerce_period_list(base_indicator_config.get('exponential-moving-average'))
        if ema_periods:
            selected_indicator_config['exponential-moving-average'] = ema_periods
        else:
            user_indicator_messages.append('Exponential MA selected but no periods provided; skipping.')

    if 'bbands' in selected_keys:
        base_bb = base_indicator_config.get('bollinger-bands')
        if isinstance(base_bb, dict):
            selected_indicator_config['bollinger-bands'] = deepcopy(base_bb)
        else:
            user_indicator_messages.append('Bollinger Bands config missing; skipping.')

    if 'supertrend' in selected_keys:
        base_supertrend = base_indicator_config.get('supertrend')
        if isinstance(base_supertrend, dict):
            selected_indicator_config['supertrend'] = deepcopy(base_supertrend)
        else:
            user_indicator_messages.append('Supertrend config missing; skipping.')

    if 'vwap' in selected_keys:
        base_vwap = base_indicator_config.get('vwap')
        if isinstance(base_vwap, dict):
            selected_indicator_config['vwap'] = deepcopy(base_vwap)
        else:
            user_indicator_messages.append('VWAP config missing; skipping.')

    if 'atr' in selected_keys:
        base_atr = base_indicator_config.get('atr')
        if isinstance(base_atr, dict):
            selected_indicator_config['atr'] = deepcopy(base_atr)
        else:
            user_indicator_messages.append('ATR config missing; skipping.')

    if 'rsi' in selected_keys:
        base_rsi = base_indicator_config.get('rsi')
        if isinstance(base_rsi, dict):
            selected_indicator_config['rsi'] = deepcopy(base_rsi)
        else:
            user_indicator_messages.append('RSI config missing; skipping.')

    if 'macd' in selected_keys:
        base_macd = base_indicator_config.get('macd')
        if isinstance(base_macd, dict):
            selected_indicator_config['macd'] = deepcopy(base_macd)
        else:
            user_indicator_messages.append('MACD config missing; skipping.')

    indicator_bundle = compute_visual_indicators(price_df, selected_indicator_config)
    indicator_messages = indicator_load_messages + user_indicator_messages + indicator_bundle.messages
    if timezone_message:
        indicator_messages = [timezone_message] + indicator_messages
    equity, drawdown, returns = build_equity_and_drawdown(series_map)
    if ui_timezone:
        equity = _convert_index_timezone(equity, ui_timezone, naive_origin='UTC')
        drawdown = _convert_index_timezone(drawdown, ui_timezone, naive_origin='UTC')
        returns = _convert_index_timezone(returns, ui_timezone, naive_origin='UTC')
    events_df = parse_order_events(jsons)
    if ui_timezone and not events_df.empty:
        if 'dt' in events_df.columns:
            events_before_samples = [str(val) for val in events_df['dt'].head(5)]
            converted_events = _convert_column_timezone(events_df, 'dt', ui_timezone, naive_origin='UTC')
            events_after_samples = [str(val) for val in converted_events['dt'].head(5)]
            print('[timezone] Order events dt before conversion:', events_before_samples)
            print('[timezone] Order events dt after conversion:', events_after_samples)
            events_df = converted_events
        else:
            events_df = _convert_column_timezone(events_df, 'dt', ui_timezone, naive_origin='UTC')
    orders_df = enrich_orders(jsons)
    if ui_timezone and not orders_df.empty:
        for col in ['time', 'createdTime', 'lastFillTime']:
            orders_df = _convert_column_timezone(orders_df, col, ui_timezone, naive_origin='UTC')

    trades_raw = reconstruct_trades(jsons)
    if ui_timezone and not trades_raw.empty:
        for col in ['entryTime', 'exitTime']:
            trades_raw = _convert_column_timezone(trades_raw, col, ui_timezone, naive_origin='UTC')
    trades_for_markers = trades_raw.copy()

    trades_df = build_trade_table(trades_raw, orders_df, events_df)
    if ui_timezone and not trades_df.empty:
        for col in ['entryTime', 'exitTime']:
            trades_df = _convert_column_timezone(trades_df, col, ui_timezone, naive_origin='UTC')

    orders_df = build_order_table(orders_df, events_df)
    if ui_timezone and not orders_df.empty:
        orders_df = _convert_column_timezone(orders_df, 'lastEventTime', ui_timezone, naive_origin='UTC')
    trade_orders_df = build_trade_order_table(trades_df, orders_df)

    toggle_value = bool(entry_exit_visible) if entry_exit_visible is not None else False
    show_entry_exit = bool(entry_exit_config_enabled and toggle_value)

    analytics_panels_input: List[Dict[str, Any]] = []
    analytics_catalog = [
        {'title': 'Portfolio Margin', 'alias': 'Portfolio Margin', 'area': False, 'height': 220},
        {'title': 'Portfolio Turnover', 'alias': 'Portfolio Turnover', 'area': False, 'height': 220},
        {'title': 'Exposure', 'alias': 'Exposure', 'area': True, 'height': 220},
        {'title': 'Strategy Capacity', 'alias': 'Capacity', 'area': True, 'height': 220}
    ]
    for entry in analytics_catalog:
        series = get_chart_series(series_map, entry['title'])
        if not series:
            series = get_chart_series(series_map, entry['alias'])
        if not series:
            continue
        analytics_panels_input.append({
            'title': entry['title'],
            'series': series,
            'area': entry.get('area', False),
            'height': entry.get('height')
        })

    candles, volume_payload = build_lightweight_price_payload(price_df, assume_timezone=ui_timezone)
    overlay_payload = build_lightweight_overlay_payload(indicator_bundle.overlays, assume_timezone=ui_timezone)
    markers = build_lightweight_markers(trades_for_markers if show_entry_exit else None, assume_timezone=ui_timezone)
    legend_children = _build_indicator_legend(overlay_payload.get('legend', []))
    echarts_payload = build_echarts_indicator_payload(
        indicator_bundle,
        equity,
        drawdown,
        returns,
        assume_timezone=ui_timezone,
        analytics=analytics_panels_input,
        expected_interval=expected_interval
    )
    echarts_payload['messages'] = indicator_messages
    json_echarts = json.dumps(echarts_payload)

    extra_blocks: List[Any] = _build_plotly_indicator_blocks(indicator_bundle, expected_interval=expected_interval)

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
        symbol_hint = price_diag.get('symbol_hint', 'the detected symbol')
        notices.append(html.Div([
            html.B('Price data not loaded.'), html.Br(),
            html.Span(f"Reason: {price_diag.get('error','unknown')}") ,
            html.Ul([
                html.Li(f"Ensure a CSV exists inside a folder named after {symbol_hint} (case-insensitive)."),
                html.Li("Expected path: data/**/<symbol>/original*.csv")
            ])
        ], style={'color':'#b36b00', 'background':'#fff7e6', 'border':'1px solid #ffe58f', 'padding':'6px 10px', 'margin':'8px 0'}))
    else:
        if price_source_note:
            notice_text = f"{price_source_note} | Resolution: {freq_label}"
        else:
            notice_text = f"Resolution: {freq_label}"
        if ui_timezone:
            notice_text += f" | TZ: {ui_timezone}"
        notices.append(html.Div(notice_text, style={'color':'#135200','background':'#f6ffed','border':'1px solid #b7eb8f','padding':'4px 8px','margin':'8px 0','borderRadius':'4px'}))

    for msg in indicator_messages:
        notices.append(html.Div(
            msg,
            style={'color':'#664d03','background':'#fff9db','border':'1px solid #ffe58f','padding':'4px 8px','margin':'6px 0','borderRadius':'4px'}
        ))

    tv_payload = {
        'candles': candles,
        'volume': volume_payload,
        'markers': markers,
        'overlays': overlay_payload,
        'meta': {
            'resolution': freq_label,
            'resampleValue': freq_value,
            'resampleUnit': freq_unit,
            'priceSource': price_source_note,
            'timezone': ui_timezone,
            'entryExitEnabled': show_entry_exit,
            'hasPriceData': bool(price_df is not None and not price_df.empty)
        }
    }
    render_token = datetime.utcnow().isoformat() + 'Z'
    echarts_render_token = render_token + '-ech'
    json_candles = json.dumps(candles)
    json_volume = json.dumps(volume_payload)
    json_overlays = json.dumps(overlay_payload)
    json_markers = json.dumps(markers)

    # trades table
    if not trades_df.empty:
        trade_columns = [{'name': column, 'id': column} for column in trades_df.columns]
        hidden_trade_columns = [column for column in trades_df.columns if column.startswith('_')]
        trade_table = dash_table.DataTable(
            columns=trade_columns,
            data=trades_df.to_dict('records'),
            page_size=15,
            hidden_columns=hidden_trade_columns,
            style_table={'overflowX':'auto'}
        )
    else:
        trade_table = html.Div('No closed trades found.')

    if not trade_orders_df.empty:
        trade_orders_table = dash_table.DataTable(
            columns=[{'name':c,'id':c} for c in trade_orders_df.columns],
            data=trade_orders_df.to_dict('records'),
            page_size=15,
            style_table={'overflowX':'auto'}
        )
    else:
        trade_orders_table = html.Div('No trade/order relationships found.')

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
    return (
        tv_payload,
        json_candles,
        json_volume,
        json_overlays,
        json_markers,
        legend_children,
        json_echarts,
        echarts_render_token,
        render_token,
        stats_component,
        trade_table,
        trade_orders_table,
        extra_blocks,
        applied_indicator_values,
        order_table
    )


@app.callback(
    Output('entry-exit-visible','data'),
    Output('toggle-entry-exit','children'),
    Output('toggle-entry-exit','disabled'),
    Input('toggle-entry-exit','n_clicks'),
    Input('backtest-dd','value'),
    State('entry-exit-visible','data')
)
def sync_entry_exit_toggle(n_clicks, backtest_folder, current_visible):
    """Keep the entry/exit signal toggle aligned with project configuration state."""
    triggered = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else None
    if not backtest_folder:
        return None, 'Entry/Exit Signals (Disabled)', True
    project_path = os.path.dirname(os.path.dirname(backtest_folder))
    config_enabled = _entry_exit_enabled_for_project(project_path)
    if not config_enabled:
        return None, 'Entry/Exit Signals (Disabled)', True
    if triggered == 'backtest-dd':
        visible = False
    elif triggered == 'toggle-entry-exit':
        visible = not bool(current_visible) if current_visible is not None else True
    else:
        visible = bool(current_visible) if current_visible is not None else False
    label = 'Entry/Exit Signals: On' if visible else 'Entry/Exit Signals: Off'
    return visible, label, False

# Register upload callbacks (after app defined)
register_upload_callbacks(app)

@app.callback(
    Output('sidebar-panel', 'style'),
    Output('main-panel', 'style'),
    Input('toggle-sidebar', 'n_clicks')
)
def toggle_sidebar(n_clicks):
    """Collapse or expand the sidebar based on the toggle button click count."""
    collapsed = bool((n_clicks or 0) % 2)
    return _sidebar_style(collapsed), _main_style(collapsed)

if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=8050,
        dev_tools_hot_reload=True,
        dev_tools_hot_reload_interval=1000,
        dev_tools_hot_reload_max_retry=8
    )
