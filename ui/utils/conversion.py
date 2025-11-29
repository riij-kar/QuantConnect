import pandas as pd, os, io, zipfile, re
from datetime import datetime

REQUIRED_PRICE_COLS = ['open','high','low','close']
VOLUME_COLS = ['volume','vol']
DATETIME_CANDIDATES = ['datetime','date','time','timestamp']
THOUSANDS_NUMBER_RE = re.compile(r'((?:^|,)\s*)(\d{1,3}(?:,\d{3})+(?:\.\d+)?)', flags=re.MULTILINE)

def _find_col(cols, names):
    """Return the first column whose lowercase name matches any alias."""
    lower = {c.lower(): c for c in cols}
    for n in names:
        if n in lower:
            return lower[n]
    # fuzzy: return first starting with any name
    for n in names:
        for c in cols:
            if c.lower().startswith(n):
                return c
    return None

def _extract_datetime(df):
    """Derive a pandas datetime Series from common datetime column patterns."""
    cols = list(df.columns)
    # try single datetime col
    single = _find_col(cols, DATETIME_CANDIDATES)
    if single:
        try:
            return pd.to_datetime(df[single], errors='coerce')
        except Exception:
            pass
    # attempt date + time separate
    date_col = _find_col(cols, ['date'])
    time_col = _find_col(cols, ['time'])
    if date_col and time_col:
        try:
            return pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), errors='coerce')
        except Exception:
            pass
    raise ValueError('Could not locate datetime column(s). Columns available: ' + ','.join(cols))

def _detect_minute(dt_series: pd.Series) -> bool:
    """Heuristically determine whether data contains intra-day observations."""
    # minute data will have repeated dates (multiple rows per date)
    if dt_series.isna().all():
        return False
    counts = dt_series.dt.date.value_counts()
    return counts.max() > 1

def _ms_since_midnight(dt: datetime) -> int:
    """Return milliseconds elapsed since midnight for the supplied timestamp."""
    return (dt.hour * 3600 + dt.minute * 60 + dt.second) * 1000 + int(dt.microsecond/1000)


def _scale_price(value) -> int:
    """Convert a price into Lean's integer scaling (price * 10,000)."""
    if pd.isna(value):
        raise ValueError('Encountered missing price while converting to Lean format.')
    if isinstance(value, (int, float)):
        return int(round(float(value) * 10000))
    text = str(value).strip()
    if not text:
        raise ValueError('Encountered missing price while converting to Lean format.')
    sanitized = text.replace(',', '')
    match = re.search(r'[-+]?\d*\.?\d+', sanitized)
    if not match:
        raise ValueError(f"Could not parse price value '{value}'.")
    return int(round(float(match.group()) * 10000))


def _scale_volume(value) -> int:
    """Normalize volume strings with optional suffixes into integer units."""
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    text = str(value).strip()
    if not text:
        return 0
    multiplier = 1
    suffix_map = {'k': 1_000, 'm': 1_000_000, 'b': 1_000_000_000}
    last_char = text[-1].lower()
    if last_char in suffix_map:
        multiplier = suffix_map[last_char]
        text = text[:-1]
    sanitized = text.replace(',', '').strip()
    if not sanitized:
        return 0
    match = re.search(r'[-+]?\d*\.?\d+', sanitized)
    if not match:
        return 0
    return int(round(float(match.group()) * multiplier))


def _append_daily_rows(df: pd.DataFrame, col_map: dict, vol_col: str, rows: list[str]):
    """Append Lean daily CSV rows into ``rows`` for the provided dataframe."""
    ordered = df.dropna(subset=['__dt']).sort_values('__dt')
    for _, record in ordered.iterrows():
        day_key = record['__dt'].strftime('%Y%m%d')
        o = _scale_price(record[col_map['open']])
        h = _scale_price(record[col_map['high']])
        l = _scale_price(record[col_map['low']])
        c = _scale_price(record[col_map['close']])
        v = _scale_volume(record[vol_col])
        rows.append(f'{day_key} 00:00,{o},{h},{l},{c},{v}')


def _format_timestamp(value) -> str | None:
    """Format timestamps as ISO8601 strings compatible with Lean uploads."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        ts = value.to_pydatetime()
    else:
        try:
            ts = pd.to_datetime(value, errors='coerce')
        except Exception:
            return None
        if pd.isna(ts):
            return None
        ts = ts.to_pydatetime()
    ts = ts.replace(microsecond=0)
    text = ts.isoformat()
    if 'T' not in text:
        text = text.replace(' ', 'T')
    if text.endswith('+00:00'):
        text = text[:-6] + 'Z'
    return text


def _clean_price(value) -> float:
    """Return a float price derived from Lean's scaled integer representation."""
    try:
        return _scale_price(value) / 10000
    except Exception:
        return float('nan')


def sanitize_to_strict_csv(raw_bytes: bytes) -> str:
    """Normalize arbitrary CSV input into the strict datetime,open,high,low,close,volume schema.

    Follows the rules documented in data/CSV_FORMAT_README.md. Any missing or unparsable
    OHLC row is dropped; volume falls back to 0 when absent.
    """
    if not raw_bytes:
        raise ValueError('Empty upload received; aborting.')

    try:
        decoded = raw_bytes.decode('utf-8-sig')
    except UnicodeDecodeError:
        decoded = raw_bytes.decode('utf-8', errors='ignore')

    decoded = THOUSANDS_NUMBER_RE.sub(lambda m: m.group(1) + m.group(2).replace(',', ''), decoded)
    df = pd.read_csv(io.StringIO(decoded), skipinitialspace=True)
    dt_series = _extract_datetime(df)
    df['__dt'] = dt_series

    col_map = {}
    for col in REQUIRED_PRICE_COLS:
        origin = _find_col(df.columns, [col])
        if not origin:
            raise ValueError(f"Missing required column for '{col}'.")
        col_map[col] = origin
    vol_col = _find_col(df.columns, VOLUME_COLS)
    if not vol_col:
        raise ValueError('Missing volume column. Expected one of volume/vol variants.')

    sanitized = pd.DataFrame()
    sanitized['datetime'] = df['__dt'].apply(_format_timestamp)
    for col in REQUIRED_PRICE_COLS:
        sanitized[col] = df[col_map[col]].apply(_clean_price)
    sanitized['volume'] = df[vol_col].apply(_scale_volume)

    sanitized = sanitized.dropna(subset=['datetime', 'open', 'high', 'low', 'close'])
    if sanitized.empty:
        raise ValueError('No valid rows remained after sanitizing OHLC values.')

    sanitized['volume'] = sanitized['volume'].fillna(0).astype(int)
    output = io.StringIO()
    sanitized[['datetime', 'open', 'high', 'low', 'close', 'volume']].to_csv(
        output,
        index=False,
        float_format='%.6f'
    )
    return output.getvalue()

def convert_equity_minute_to_lean(raw_csv: str, symbol: str):
    """Convert minute-level equity CSV text into Lean zip payloads.

    Parameters
    ----------
    raw_csv : str
        Uploaded CSV content already decoded to text.
    symbol : str
        Symbol ticker used to name the emitted files.

    Returns
    -------
    dict[str, bytes]
        Mapping of ``{date}_trade.zip`` -> zip file bytes. Empty when the input
        is daily rather than minute-granularity.
    """
    df = pd.read_csv(io.StringIO(raw_csv))
    dt = _extract_datetime(df)
    df['__dt'] = dt
    df = df.dropna(subset=['__dt'])
    # normalize needed columns
    col_map = {}
    for col in REQUIRED_PRICE_COLS:
        c = _find_col(df.columns, [col])
        if not c:
            raise ValueError(f'Missing price column: {col}')
        col_map[col] = c
    vol_col = _find_col(df.columns, VOLUME_COLS)
    if not vol_col:
        raise ValueError('Missing volume column')
    if not _detect_minute(df['__dt']):
        # treat as daily -> ignore
        return {}
    outputs = {}
    for date, g in df.groupby(df['__dt'].dt.date):
        day_str = date.strftime('%Y%m%d')
        # Build lean CSV content
        rows = []
        g_sorted = g.sort_values('__dt')
        for _, r in g_sorted.iterrows():
            t = _ms_since_midnight(r['__dt'])
            o = int(float(r[col_map['open']]) * 10000)
            h = int(float(r[col_map['high']]) * 10000)
            l = int(float(r[col_map['low']]) * 10000)
            c = int(float(r[col_map['close']]) * 10000)
            v = int(float(r[vol_col])) if pd.notna(r[vol_col]) else 0
            rows.append(f'{t},{o},{h},{l},{c},{v}')
        csv_name = f'{day_str}_{symbol.lower()}_minute_trade.csv'
        csv_bytes = '\n'.join(rows).encode('utf-8')
        zip_name = f'{day_str}_trade.zip'
        # create zip in memory
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(csv_name, csv_bytes)
        outputs[zip_name] = bio.getvalue()
    return outputs

def convert_equity_minute_to_lean_stream(csv_path: str, symbol: str, chunk_rows: int = 250000):
    """Stream large CSV uploads and emit Lean-formatted archives.

    Parameters
    ----------
    csv_path : str
        Path to the uploaded CSV file on disk.
    symbol : str
        Symbol used when naming produced files.
    chunk_rows : int, optional
        Chunk size used when iterating through the CSV to limit memory usage.

    Returns
    -------
    dict
        Structure describing the output mode (``minute`` or ``daily``) and a
        list of generated files with their metadata and bytes.
    """
    result = {"mode": "minute", "files": []}
    try:
        chunk_iter = pd.read_csv(csv_path, chunksize=chunk_rows)
    except Exception as e:
        raise ValueError(f'Unable to read CSV in chunks: {e}')
    try:
        first = next(chunk_iter)
    except StopIteration:
        return result

    dt_first = _extract_datetime(first)
    first['__dt'] = dt_first
    first = first.dropna(subset=['__dt'])
    if first.empty:
        return result

    col_map = {}
    for col in REQUIRED_PRICE_COLS:
        c = _find_col(first.columns, [col])
        if not c:
            raise ValueError(f'Missing price column: {col}')
        col_map[col] = c
    vol_col = _find_col(first.columns, VOLUME_COLS)
    if not vol_col:
        raise ValueError('Missing volume column')

    # Daily data -> create folder bundle zip
    if first['__dt'].dt.date.nunique() == first.shape[0]:
        rows: list[str] = []
        _append_daily_rows(first, col_map, vol_col, rows)
        for chunk in chunk_iter:
            dt_chunk = _extract_datetime(chunk)
            chunk['__dt'] = dt_chunk
            _append_daily_rows(chunk, col_map, vol_col, rows)
        if not rows:
            return {"mode": "daily", "files": []}

        folder_name = (symbol or 'daily').strip()
        safe_name = folder_name.replace(' ', '_').lower()
        base_dir = os.path.dirname(csv_path)
        lean_csv_path = os.path.join(base_dir, f'{safe_name}.csv')
        with open(lean_csv_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(rows))

        zip_path = os.path.join(base_dir, f'{safe_name}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(lean_csv_path, arcname=f'{safe_name}.csv')

        # Remove the intermediate Lean CSV so the folder only keeps the original and the zip bundle
        try:
            os.remove(lean_csv_path)
        except OSError:
            pass

        return {"mode": "daily", "files": [os.path.basename(zip_path)]}

    # Minute path retained
    written: list[str] = []

    def flush_day(day_date, rows_accum):
        if not rows_accum:
            return
        day_str = day_date.strftime('%Y%m%d')
        csv_internal = f'{day_str}_{symbol.lower()}_minute_trade.csv'
        zip_name = f'{day_str}_trade.zip'
        csv_bytes = '\n'.join(rows_accum).encode('utf-8')
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(csv_internal, csv_bytes)
        target = os.path.join(os.path.dirname(csv_path), zip_name)
        with open(target, 'wb') as f:
            f.write(bio.getvalue())
        written.append(zip_name)

    current_day = None
    day_rows: list[str] = []

    def process_chunk(chunk: pd.DataFrame):
        nonlocal current_day, day_rows
        dt_chunk = _extract_datetime(chunk)
        chunk['__dt'] = dt_chunk
        chunk = chunk.dropna(subset=['__dt']).sort_values('__dt')
        for _, r in chunk.iterrows():
            d = r['__dt'].date()
            if current_day is not None and d != current_day:
                flush_day(current_day, day_rows)
                day_rows = []
            current_day = d
            ms = _ms_since_midnight(r['__dt'])
            o = _scale_price(r[col_map['open']])
            h = _scale_price(r[col_map['high']])
            l = _scale_price(r[col_map['low']])
            c = _scale_price(r[col_map['close']])
            v = _scale_volume(r[vol_col])
            day_rows.append(f'{ms},{o},{h},{l},{c},{v}')

    process_chunk(first)
    for chunk in chunk_iter:
        process_chunk(chunk)
    flush_day(current_day, day_rows)
    return {"mode": "minute", "files": written}
