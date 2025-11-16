import os
import json
import pandas as pd
from typing import Optional, Tuple, Dict

__all__ = ['load_ohlcv_from_csv']

# Given a project folder (with config.json) and the workspace data folder,
# read EquityName and locate an original*.csv recursively. Return (df, path).

def load_ohlcv_from_csv(project_path: str, data_root: str) -> Tuple[Optional[pd.DataFrame], Optional[str], Dict[str, str]]:
    """
    REQUIRED CSV FORMAT (strict):
    - Header: datetime,open,high,low,close,volume  (all lowercase, comma-separated)
    - datetime values parseable by pandas.to_datetime with utc=True
    - All OHLC columns numeric; volume numeric (can be empty)

    Loader behavior:
    - Finds a folder named exactly as EquityName (case-insensitive) under data_root
    - Inside that folder, picks the newest file named original*.csv
    - Reads the CSV and validates strict columns; returns (df, path, diag)
      where diag has an 'error' and 'needed_format' on failure.
    """
    cfg_path = os.path.join(project_path, 'config.json')
    if not os.path.isfile(cfg_path):
        return None, None, {'error': 'config.json not found', 'needed_format': 'datetime,open,high,low,close,volume'}
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        symbol = (cfg.get('EquityName') or cfg.get('equity') or cfg.get('symbol') or '').strip()
        if not symbol:
            return None, None, {'error': 'EquityName not found in config.json', 'needed_format': 'datetime,open,high,low,close,volume'}
    except Exception as e:
        return None, None, {'error': f'Failed to parse config.json: {e}', 'needed_format': 'datetime,open,high,low,close,volume'}

    # 1) Find a folder under data_root whose basename matches EquityName (case-insensitive)
    target_dir = None
    sym_l = symbol.lower()
    for root, dirs, files in os.walk(data_root):
        base = os.path.basename(root)
        if base.lower() == sym_l:
            target_dir = root
            break
    if target_dir is None:
        return None, None, {'error': f"Folder named '{symbol}' not found under data root (case-insensitive)", 'needed_format': 'datetime,open,high,low,close,volume'}

    # 2) Inside that folder, locate any 'original*.csv' (case-insensitive) and pick the newest
    try:
        entries = [
            os.path.join(target_dir, name) for name in os.listdir(target_dir)
            if name.lower().startswith('original') and name.lower().endswith('.csv')
        ]
    except Exception as e:
        return None, None, {'error': f'Failed to list files in folder {target_dir}: {e}', 'needed_format': 'datetime,open,high,low,close,volume'}
    if not entries:
        return None, None, {'error': f"No 'original*.csv' found inside folder '{os.path.basename(target_dir)}'", 'needed_format': 'datetime,open,high,low,close,volume'}
    # Prefer most recently modified file
    entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    csv_path = entries[0]
    print(f'[price_loader] Loading OHLCV from {csv_path}')
    # 3) Read and normalize CSV into OHLCV, preserving timezone from file
    try:
        raw = pd.read_csv(csv_path, encoding='utf-8-sig')
        if raw is None or raw.empty:
            return None, csv_path, {'error': 'CSV is empty', 'needed_format': 'datetime,open,high,low,close,volume'}

        # Strict header check
        required = ['datetime','open','high','low','close','volume']
        actual = [str(c).strip().lower() for c in list(raw.columns)]
        if actual != required:
            return None, csv_path, {
                'error': 'CSV header does not match required format',
                'found_header': actual,
                'needed_format': 'datetime,open,high,low,close,volume'
            }

        # Parse datetime (preserve timezone if present; no forced UTC conversion)
        raw['datetime'] = pd.to_datetime(raw['datetime'], errors='coerce')
        if raw['datetime'].isna().all():
            return None, csv_path, {
                'error': 'Datetime parsing failed for all rows',
                'needed_format': 'datetime,open,high,low,close,volume',
                'sample_values': raw['datetime'].astype(str).head(5).tolist()
            }

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
            return None, csv_path, {
                'error': 'All OHLC rows invalid/empty after cleaning',
                'needed_format': 'datetime,open,high,low,close,volume',
                'nan_counts': nan_counts
            }

        return out, csv_path, {}
    except Exception as e:
        print('[price_loader] Exception reading CSV:', e)
        return None, csv_path, {'error': f'Failed to read/parse CSV: {e}', 'needed_format': 'datetime,open,high,low,close,volume'}
