import pandas as pd
import numpy as np
import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go

__all__ = [
    'extract_series',
    'build_price_from_series',
    'compute_indicators',
    'build_equity_and_drawdown',
    'get_chart_series'
]

def extract_series(charts_obj: dict):
    """Parse LEAN chart JSON into individual pandas Series.
    Supports OHLC arrays [ts, o, h, l, c] and simple [ts, value] points.
    Automatically detects epoch seconds numeric timestamps.
    Keys for OHLC become <Chart>::<Series>::open/high/low/close.
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


def build_equity_and_drawdown(series_map):
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
