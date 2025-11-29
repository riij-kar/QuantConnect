from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
#Below transform helpers for lightweight-charts
#build_lightweight_price_payload
#build_lightweight_overlay_payload
#build_lightweight_markers
__all__ = [
    'extract_series',
    'build_price_from_series',
    'build_equity_and_drawdown',
    'get_chart_series',
    'build_lightweight_price_payload',
    'build_lightweight_overlay_payload',
    'build_lightweight_markers'
]

def extract_series(charts_obj: dict):
    """Convert the LEAN ``charts`` payload into a dictionary of Series.

    Parameters
    ----------
    charts_obj : dict
        Parsed JSON object under ``charts`` from Lean backtest results.

    Returns
    -------
    dict[str, pandas.Series]
        Mapping of chart/series identifiers to pandas Series indexed by UTC
        timestamps. OHLC series use the ``<Chart>::<Series>::component`` naming
        convention.
    """
    series_map: dict[str, pd.Series] = {}
    if not isinstance(charts_obj, dict):
        return series_map

    def to_dt(xs):
        if not xs:
            return pd.DatetimeIndex([])
        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in xs):
            try:
                return pd.to_datetime(xs, unit='s', utc=True)
            except Exception:
                pass
        return pd.to_datetime(xs, utc=True, errors='coerce')

    for chart_name, chart in charts_obj.items():
        ser_dict = chart.get('series', {}) if isinstance(chart, dict) else {}
        for ser_name, ser in ser_dict.items():
            values = ser.get('values') or ser.get('points') or []
            if not values:
                continue
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


def build_price_from_series(series_map):
    """Assemble a price DataFrame from extracted chart series.

    This helper feeds the legacy Plotly dashboard pipeline.

    Parameters
    ----------
    series_map : dict[str, pandas.Series]
        Result of ``extract_series``.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing ``open/high/low/close`` columns when available, or
        a close-only frame. ``None`` when no price series can be found.
    """
    lower = {k.lower(): k for k in series_map.keys()}
    keys = {part: lower.get(part) for part in ['price::open','price::high','price::low','price::close']}
    if all(v for v in keys.values()):
        return pd.DataFrame({
            'open': series_map[keys['price::open']],
            'high': series_map[keys['price::high']],
            'low': series_map[keys['price::low']],
            'close': series_map[keys['price::close']]
        }).dropna()
    for k in series_map:
        if k.endswith('::Close') or k.lower().endswith('::close'):
            return pd.DataFrame({'close': series_map[k]}).dropna()
    return None


def build_equity_and_drawdown(series_map):
    """Derive equity curve, drawdown, and returns series from chart data.

    Primarily consumed by the Plotly visualization flow.

    Parameters
    ----------
    series_map : dict[str, pandas.Series]
        Dictionary returned by ``extract_series``.

    Returns
    -------
    tuple
        ``(equity, drawdown, returns)`` where ``equity`` is either an OHLC
        DataFrame or Series, and the remaining values are Series or ``None`` if
        not computable.
    """
    parts = {'open': None,'high': None,'low': None,'close': None}
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


def get_chart_series(series_map: dict, chart_prefix: str) -> dict:
    """Select chart series matching the supplied prefix.

    Intended for additional Plotly subplots (margin, capacity, etc.).

    Parameters
    ----------
    series_map : dict[str, pandas.Series]
        Chart series map produced by ``extract_series``.
    chart_prefix : str
        Prefix of the desired chart (e.g., ``"Portfolio Margin"``).

    Returns
    -------
    dict[str, pandas.Series]
        Mapping of simplified names to Series for the requested chart prefix.
    """
    result = {}
    prefix = chart_prefix.lower() + '::'
    keys = [k for k in series_map.keys() if k.lower().startswith(prefix)]
    if not keys:
        return result
    bases = {}
    for k in keys:
        kl = k.lower()
        if kl.endswith('::open') or kl.endswith('::high') or kl.endswith('::low'):
            continue
        if kl.endswith('::close'):
            base = k.rsplit('::', 1)[0]
            bases[base] = series_map[k]
            continue
        result[k] = series_map[k]
    for base, ser in bases.items():
        name = base.split('::', 1)[-1]
        result[name] = ser
    return result

#transform helpers for lightweight-charts
def _timestamp_to_epoch_seconds(value: Any, assume_timezone: Optional[str] = None) -> Optional[int]:
    """Return Unix epoch seconds for the supplied timestamp.

    Parameters
    ----------
    value : Any
        Original timestamp value from a pandas index or column.
    assume_timezone : str, optional
        Timezone name to assume when ``value`` is naive (no ``tzinfo``).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        timestamp = value
    else:
        try:
            timestamp = pd.to_datetime(value, errors='coerce')
        except Exception:
            return None
    if timestamp is pd.NaT or pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        if assume_timezone:
            try:
                timestamp = timestamp.tz_localize(assume_timezone)
            except Exception:
                timestamp = timestamp.tz_localize('UTC')
        else:
            timestamp = timestamp.tz_localize('UTC')
    elif assume_timezone:
        try:
            timestamp = timestamp.tz_convert(assume_timezone)
        except Exception:
            pass
    timestamp = timestamp.tz_convert('UTC')
    return int(timestamp.timestamp())

#transform helpers for lightweight-charts
def _to_float(value: Any) -> Optional[float]:
    """Safely coerce a scalar to ``float`` while tolerating missing values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

#transform helpers for lightweight-charts
def build_lightweight_price_payload(price_df: Optional[pd.DataFrame], assume_timezone: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Transform ``price_df`` into lightweight-charts candle and volume arrays.

    Used by the TradingView-based dashboard.
    """
    candles: List[Dict[str, Any]] = []
    volume: List[Dict[str, Any]] = []
    if price_df is None or price_df.empty:
        return candles, volume

    has_volume = 'volume' in price_df.columns

    for idx, row in price_df.iterrows():
        epoch_time = _timestamp_to_epoch_seconds(idx, assume_timezone)
        if epoch_time is None:
            continue

        open_val = _to_float(row.get('open', row.get('close')))
        close_val = _to_float(row.get('close', open_val))
        high_val = _to_float(row.get('high', max(open_val or 0.0, close_val or 0.0)))
        low_val = _to_float(row.get('low', min(open_val or 0.0, close_val or 0.0)))

        if open_val is None and close_val is None:
            continue
        if open_val is None:
            open_val = close_val
        if close_val is None:
            close_val = open_val
        if high_val is None:
            high_val = max(open_val, close_val)
        if low_val is None:
            low_val = min(open_val, close_val)

        candles.append({
            'time': epoch_time,
            'open': open_val,
            'high': high_val,
            'low': low_val,
            'close': close_val
        })

        if has_volume:
            vol_val = _to_float(row.get('volume'))
            if vol_val is not None:
                color = '#26a69a'
                if open_val is not None and close_val is not None and close_val < open_val:
                    color = '#ef5350'
                volume.append({
                    'time': epoch_time,
                    'value': vol_val,
                    'color': color
                })

    return candles, volume

#transform helpers for lightweight-charts
def _series_to_lightweight(series: Optional[pd.Series], assume_timezone: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convert a pandas Series into a lightweight-charts line payload."""
    payload: List[Dict[str, Any]] = []
    if series is None:
        return payload
    numeric = pd.to_numeric(series, errors='coerce')
    for idx, value in numeric.dropna().items():
        epoch_time = _timestamp_to_epoch_seconds(idx, assume_timezone)
        scalar = _to_float(value)
        if epoch_time is not None and scalar is not None:
            payload.append({'time': epoch_time, 'value': scalar})
    return payload

#transform helpers for lightweight-charts
COLOR_PALETTE: List[str] = [
    '#089981', '#f97316', '#1d4ed8', '#10b981', '#d946ef',
    '#ef4444', '#3b82f6', '#6366f1', '#22c55e', '#fb7185'
]
SUPER_TREND_COLORS: Dict[str, str] = {
    'upper': '#ef4444',
    'lower': '#22c55e'
}
VWAP_COLOR = '#000000'


def _hash_string(value: str) -> int:
    hash_val = 0
    if not value:
        return hash_val
    for char in value:
        hash_val = ((hash_val << 5) - hash_val) + ord(char)
        hash_val &= 0xFFFFFFFF
    # convert to signed 32-bit
    if hash_val & 0x80000000:
        hash_val -= 0x100000000
    return hash_val


def _color_for_series(name: str) -> str:
    if not COLOR_PALETTE:
        return '#1d4ed8'
    idx = abs(_hash_string(name)) % len(COLOR_PALETTE)
    return COLOR_PALETTE[idx]


def build_lightweight_overlay_payload(overlays: Optional[Dict[str, Any]], assume_timezone: Optional[str] = None) -> Dict[str, Any]:
    """Serialize overlay indicators for consumption by lightweight-charts.

    Used by the TradingView-based dashboard.
    """
    result: Dict[str, Any] = {
        'lines': {},
        'supertrend': {},
        'legend': []
    }
    if not overlays:
        return result

    for name, value in overlays.items():
        if isinstance(value, pd.Series):
            data = _series_to_lightweight(value, assume_timezone)
            if data:
                label = str(name)
                color = VWAP_COLOR if label.upper().startswith('VWAP') else _color_for_series(label)
                result['lines'][label] = {
                    'data': data,
                    'color': color,
                    'lineWidth': 2,
                    'priceLineVisible': False,
                    'lastValueVisible': False
                }
                result['legend'].append({'label': label, 'color': color})
        elif isinstance(value, dict) and value.get('type') == 'supertrend':
            upper = _series_to_lightweight(value.get('upper'), assume_timezone)
            lower = _series_to_lightweight(value.get('lower'), assume_timezone)
            parts: Dict[str, Any] = {}
            label = str(name)
            if upper:
                parts['upper'] = {
                    'data': upper,
                    'color': SUPER_TREND_COLORS['upper'],
                    'lineWidth': 1,
                    'lineStyle': 2,
                    'priceLineVisible': False,
                    'lastValueVisible': False
                }
                result['legend'].append({'label': f"{label} Upper", 'color': SUPER_TREND_COLORS['upper']})
            if lower:
                parts['lower'] = {
                    'data': lower,
                    'color': SUPER_TREND_COLORS['lower'],
                    'lineWidth': 1,
                    'lineStyle': 2,
                    'priceLineVisible': False,
                    'lastValueVisible': False
                }
                result['legend'].append({'label': f"{label} Lower", 'color': SUPER_TREND_COLORS['lower']})
            if parts:
                result['supertrend'][label] = parts
    return result

#transform helpers for lightweight-charts
def build_lightweight_markers(trades_df: Optional[pd.DataFrame], assume_timezone: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generate entry/exit markers compatible with lightweight-charts.

    Used by the TradingView-based dashboard.
    """
    markers: List[Dict[str, Any]] = []
    if trades_df is None or trades_df.empty:
        return markers

    df = trades_df.copy()

    if 'entryPrice' in df.columns:
        df['entryPrice'] = pd.to_numeric(df['entryPrice'], errors='coerce')
    if 'exitPrice' in df.columns:
        df['exitPrice'] = pd.to_numeric(df['exitPrice'], errors='coerce')

    for _, row in df.iterrows():
        entry_time_epoch = _timestamp_to_epoch_seconds(row.get('entryTime'), assume_timezone)
        exit_time_epoch = _timestamp_to_epoch_seconds(row.get('exitTime'), assume_timezone)

        quantity = _to_float(row.get('quantity')) or _to_float(row.get('quantityFilled'))
        symbol = row.get('symbol') or row.get('symbolId')

        if entry_time_epoch is not None:
            text_parts: List[str] = []
            if symbol:
                text_parts.append(str(symbol))
            entry_price = _to_float(row.get('entryPrice'))
            if entry_price is not None:
                text_parts.append(f"Entry {entry_price:.2f}")
            if quantity is not None:
                text_parts.append(f"Qty {quantity:.0f}")
            markers.append({
                'time': entry_time_epoch,
                'position': 'belowBar',
                'color': '#16a34a',
                'shape': 'arrowUp',
                'text': ' | '.join(text_parts) if text_parts else 'Entry'
            })

        if exit_time_epoch is not None:
            text_parts = []
            if symbol:
                text_parts.append(str(symbol))
            exit_price = _to_float(row.get('exitPrice'))
            if exit_price is not None:
                text_parts.append(f"Exit {exit_price:.2f}")
            profit = _to_float(row.get('profit')) or _to_float(row.get('pnl'))
            if profit is not None:
                text_parts.append(f"PnL {profit:.2f}")
            markers.append({
                'time': exit_time_epoch,
                'position': 'aboveBar',
                'color': '#dc2626',
                'shape': 'arrowDown',
                'text': ' | '.join(text_parts) if text_parts else 'Exit'
            })

    return markers
