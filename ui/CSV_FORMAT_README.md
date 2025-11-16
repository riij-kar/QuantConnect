# Strict Price CSV Format
For backtesting UI
This project expects each price CSV used for charting to follow a single strict schema.
If any column deviates, the loader refuses the file and returns a diagnostic with the
exact header found and the required format.

## Required Header (exact order, lowercase)
```
datetime,open,high,low,close,volume
```

## Column Rules
- datetime: ISO8601 or pandas-recognizable timestamp. Timezone offsets allowed and preserved (e.g. 2025-07-14T09:15:00+05:30). The loader does not force UTC.
- open, high, low, close: Numeric (float or int). No currency symbols, commas, or stray characters.
- volume: Numeric (integer or float). Can be 0 or blank. Must be present as a column.

Rows with any missing OHLC value are dropped. Volume may be missing/blank and is not used to drop rows.

## Sample Rows
```
datetime,open,high,low,close,volume
2025-07-14T09:15:00+05:30,406.75,407.7,405.85,406.5,271777
2025-07-14T09:16:00+05:30,406.7,407.6,405.15,407.3,325234
2025-07-14T09:17:00+05:30,407.75,407.8,406.65,407.25,169511
2025-07-14T09:18:00+05:30,407.4,407.9,407.3,407.35,156502
```

## Loader Behavior (summary)
- Reads EquityName from the project’s config.json.
- Recursively finds a folder whose name matches EquityName (case-insensitive) under data/.
- Picks the newest file inside that folder named original*.csv.
- Parses datetime (preserving timezone if present). Converts OHLC/volume to numeric.
- Sets the DataFrame index to the parsed datetime, sorts, and drops rows missing any OHLC.

## Common Errors & Fixes
| Diagnostic Error | Cause | Fix |
|---|---|---|
| config.json not found | Project config missing | Create config.json with {"EquityName": "SYMBOL"} |
| Folder named 'SYMBOL' not found | Symbol folder absent under data root | Create directory matching EquityName (case-insensitive) |
| No 'original*.csv' found | CSV not present or wrong name | Place a file starting with original and ending .csv in symbol folder |
| CSV header does not match required format | Header mismatch (wrong order/case/name) | Rename header exactly to required list |
| CSV is empty | File has only header or no rows | Add data rows |
| Datetime parsing failed for all rows | All datetime strings invalid | Ensure ISO8601 format and valid timezone offsets |
| All OHLC rows invalid/empty after cleaning | OHLC columns non-numeric or missing | Remove thousand separators/characters; ensure numerics |

## Validation Checklist
1. Filename: starts with original and ends with .csv (e.g. original_NSE_BEL_1.csv).
2. Located in: data/**/<EquityName>/ folder (any depth is fine).
3. Header matches exactly.
4. First data row parses: pandas.to_datetime(value) succeeds.
5. No thousands separators: use 4067.5 not 4,067.5.

## Quick Conversion Script
```python
import pandas as pd

def normalize_csv(path, out_path=None):
    df = pd.read_csv(path)
    # Rename columns if needed
    rename_map = {c.lower(): c for c in df.columns}
    required = ['datetime','open','high','low','close','volume']
    # Attempt minimal normalization
    cols_lower = [c.lower() for c in df.columns]
    if set(cols_lower) >= set(required):
        df = df[[rename_map[c] for c in required]]
        df.columns = required
    else:
        raise ValueError('Source CSV missing required columns')
    # Write out
    out_path = out_path or path
    df.to_csv(out_path, index=False)
    return out_path
```

## Troubleshooting Datetime Parsing
- T or space separator both acceptable.
- Timezone offsets (+05:30) are preserved; if missing, timestamps are naive (no timezone).
- Milliseconds (e.g. 2025-07-14T09:15:00.123+05:30) are acceptable.
- If you prefer UTC, pre-convert externally or modify the loader to use utc=True and tz_convert('UTC').

## Rationale
A single strict format eliminates ambiguity, simplifies parsing, and ensures consistent indicator calculations (EMA, RSI) and candlestick rendering.

## Future Extension (Optional)
If later you want flexibility (alternate headers or Lean zip fallback), extend the loader but keep this README as the canonical specification.

---
Generated automatically. Keep this file versioned with any future format changes.
