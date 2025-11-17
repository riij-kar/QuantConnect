import os
import json
import re
import pandas as pd
from typing import Optional, Tuple, Dict

__all__ = ['load_ohlcv_from_csv']

# Given a backtest context and the workspace data folder,
# locate the latest original*.csv for the detected symbol. Return (df, path).

ORDER_EVENTS_PATTERN = re.compile(r"-order-events\.json$", re.IGNORECASE)
SYMBOL_VALUE_RE = re.compile(r'"symbolValue"\s*:\s*"([^"]+)"')


def _symbol_from_order_events(backtest_folder: Optional[str]) -> Optional[str]:
    if not backtest_folder:
        return None
    try:
        candidates = sorted(
            [
                os.path.join(backtest_folder, name)
                for name in os.listdir(backtest_folder)
                if ORDER_EVENTS_PATTERN.search(name)
            ]
        )
    except OSError:
        return None
    for path in candidates:
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                for line in handle:
                    match = SYMBOL_VALUE_RE.search(line)
                    if match:
                        value = match.group(1).strip()
                        if value:
                            return value
        except Exception:
            continue
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        value = (entry.get('symbolValue')
                                 or entry.get('symbol')
                                 or entry.get('symbolPermtick'))
                        if value:
                            value = str(value).strip()
                            if value:
                                return value
        except Exception:
            continue
    return None


def load_ohlcv_from_csv(project_path: str, data_root: str, backtest_folder: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[str], Dict[str, str]]:
    """
    REQUIRED CSV FORMAT (strict):
    - Header: datetime,open,high,low,close,volume  (all lowercase, comma-separated)
    - datetime values parseable by pandas.to_datetime with utc=True
    - All OHLC columns numeric; volume numeric (can be empty)

    Loader behavior:
        - Derives the symbol from the selected backtest's *-order-events.json
        - Finds a folder named exactly as that symbol (case-insensitive) under data_root
    - Inside that folder, picks the newest file named original*.csv
    - Reads the CSV and validates strict columns; returns (df, path, diag)
      where diag has an 'error' and 'needed_format' on failure.
    """
    symbol = _symbol_from_order_events(backtest_folder)
    diag: Dict[str, str] = {}
    if not symbol:
        diag.setdefault('needed_format', 'datetime,open,high,low,close,volume')
        diag.setdefault('error', 'Unable to determine symbol from order-events JSON')
        return None, None, diag
    diag.setdefault('symbol_hint', symbol)

    # 1) Find a folder under data_root whose basename matches EquityName (case-insensitive)
    target_dir = None
    sym_l = symbol.lower()
    for root, dirs, files in os.walk(data_root):
        base = os.path.basename(root)
        if base.lower() == sym_l:
            target_dir = root
            break
    if target_dir is None:
        diag.setdefault('needed_format', 'datetime,open,high,low,close,volume')
        diag['error'] = f"Folder named '{symbol}' not found under data root (case-insensitive)"
        diag['symbol_hint'] = symbol
        return None, None, diag

    # 2) Inside that folder, locate any 'original*.csv' (case-insensitive) and pick the newest
    try:
        entries = [
            os.path.join(target_dir, name) for name in os.listdir(target_dir)
            if name.lower().startswith('original') and name.lower().endswith('.csv')
        ]
    except Exception as e:
        diag.setdefault('needed_format', 'datetime,open,high,low,close,volume')
        diag['error'] = f'Failed to list files in folder {target_dir}: {e}'
        diag['symbol_hint'] = symbol
        return None, None, diag
    if not entries:
        diag.setdefault('needed_format', 'datetime,open,high,low,close,volume')
        diag['error'] = f"No 'original*.csv' found inside folder '{os.path.basename(target_dir)}'"
        diag['symbol_hint'] = symbol
        return None, None, diag
    # Prefer most recently modified file
    entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    csv_path = entries[0]
    print(f'[price_loader] Loading OHLCV from {csv_path}')
    # 3) Read and normalize CSV into OHLCV, preserving timezone from file
    try:
        raw = pd.read_csv(csv_path, encoding='utf-8-sig')
        if raw is None or raw.empty:
            diag.setdefault('needed_format', 'datetime,open,high,low,close,volume')
            diag['error'] = 'CSV is empty'
            diag['symbol_hint'] = symbol
            return None, csv_path, diag

        # Strict header check
        required = ['datetime','open','high','low','close','volume']
        actual = [str(c).strip().lower() for c in list(raw.columns)]
        if actual != required:
            diag.update({
                'error': 'CSV header does not match required format',
                'found_header': actual,
                'needed_format': 'datetime,open,high,low,close,volume',
                'symbol_hint': symbol
            })
            return None, csv_path, diag

        # Parse datetime (preserve timezone if present; no forced UTC conversion)
        raw['datetime'] = pd.to_datetime(raw['datetime'], errors='coerce')
        if raw['datetime'].isna().all():
            diag.update({
                'error': 'Datetime parsing failed for all rows',
                'needed_format': 'datetime,open,high,low,close,volume',
                'sample_values': raw['datetime'].astype(str).head(5).tolist(),
                'symbol_hint': symbol
            })
            return None, csv_path, diag

        # Convert numeric columns
        for col in ['open','high','low','close','volume']:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')

        # Build frame then set index to avoid label-alignment NaNs
        out = raw[['open','high','low','close','volume']].copy()
        out.index = raw['datetime']
        out = out.loc[~out.index.isna()].sort_index()
        out = out.dropna(subset=['open','high','low','close'])

        if out.empty:
            nan_counts = {c: int(raw[c].isna().sum()) for c in ['open','high','low','close','volume']}
            diag.update({
                'error': 'All OHLC rows invalid/empty after cleaning',
                'needed_format': 'datetime,open,high,low,close,volume',
                'nan_counts': nan_counts,
                'symbol_hint': symbol
            })
            return None, csv_path, diag

        return out, csv_path, {'symbol_hint': symbol}
    except Exception as e:
        print('[price_loader] Exception reading CSV:', e)
        diag.update({'error': f'Failed to read/parse CSV: {e}', 'needed_format': 'datetime,open,high,low,close,volume', 'symbol_hint': symbol})
        return None, csv_path, diag
