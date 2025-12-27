"""Helpers to build consolidated (resampled) OHLC data frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from pandas.tseries.frequencies import to_offset

__all__ = [
    "RESAMPLE_UNITS",
    "normalize_frequency",
    "format_frequency_label",
    "frequency_to_timedelta",
    "resample_ohlcv"
]


@dataclass(frozen=True)
class _UnitConfig:
    code: str  # pandas offset alias (T, H, D, M)
    label: str  # human readable label in singular form


RESAMPLE_UNITS: dict[str, _UnitConfig] = {
    "minute": _UnitConfig(code="min", label="minute"),
    "hour": _UnitConfig(code="h", label="hour"),
    "day": _UnitConfig(code="D", label="day"),
    "month": _UnitConfig(code="M", label="month"),
}


def normalize_frequency(value: Optional[int | float | str], unit: Optional[str]) -> tuple[int, str]:
    """Return a sanitized ``(value, unit)`` pair for resampling.

    Fallbacks to ``(1, "minute")`` when input is invalid.
    """
    try:
        num = int(float(value)) if value is not None else 1
    except (TypeError, ValueError):
        num = 1
    if num <= 0:
        num = 1
    key = (unit or "minute").strip().lower()
    if key not in RESAMPLE_UNITS:
        key = "minute"
    return num, key


def format_frequency_label(value: int, unit: str) -> str:
    """Return a friendly label such as ``"5 minutes"``."""
    label = RESAMPLE_UNITS.get(unit, RESAMPLE_UNITS["minute"]).label
    suffix = "s" if value != 1 else ""
    return f"{value} {label}{suffix}"


def frequency_to_timedelta(value: int, unit: str) -> Optional[pd.Timedelta]:
    """Return a pandas Timedelta for the requested consolidation interval."""
    try:
        config = RESAMPLE_UNITS[unit]
    except KeyError:
        return None

    if config.code in {"min", "h", "D"}:
        try:
            return pd.to_timedelta(value, unit=config.code)
        except Exception:
            return None

    try:
        offset = to_offset(f"{value}{config.code}")
    except Exception:
        return None
    try:
        delta = offset.delta  # type: ignore[attr-defined]
    except (AttributeError, ValueError, TypeError):
        delta = None
    if delta is not None:
        return delta
    try:
        nanos = offset.nanos  # type: ignore[attr-defined]
    except (AttributeError, ValueError, TypeError):
        nanos = None
    if nanos is not None:
        try:
            return pd.to_timedelta(nanos, unit="ns")
        except Exception:
            return None
    return None


def _infer_session_offset(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    """Infer the most common intraday offset of the first print per session."""
    if index is None or index.empty:
        return None
    try:
        normalized = index.normalize()
        first_by_day = pd.Series(index, index=normalized).groupby(level=0).min()
    except Exception:
        return None
    if first_by_day.empty:
        return None
    deltas = (first_by_day - first_by_day.index).dropna()
    if deltas.empty:
        return None
    try:
        mode = deltas.mode()
        if not mode.empty:
            return pd.to_timedelta(mode.iloc[0])
    except Exception:
        mode = None
    try:
        return pd.to_timedelta(deltas.median())
    except Exception:
        return None



def resample_ohlcv(df: Optional[pd.DataFrame], value: Optional[int | float | str], unit: Optional[str]) -> Optional[pd.DataFrame]:
    """Resample an OHLC(V) DataFrame to a new frequency.

    The dataframe must have a ``DatetimeIndex`` and the standard OHLC column names.
    ``volume`` is optional and will be aggregated via sum when present.
    """
    if df is None or df.empty:
        return df

    freq_value, freq_unit = normalize_frequency(value, unit)
    config = RESAMPLE_UNITS[freq_unit]

    # Skip resampling when it would not change the data.
    if freq_value == 1 and freq_unit == "minute":
        return df

    freq_code = f"{freq_value}{config.code}"
    tz_info = df.index.tz
    working = df.copy()
    working['__bucket_first_ts'] = working.index

    session_offset = _infer_session_offset(working.index)
    alignable_units = {"minute", "hour", "day"}
    freq_delta = frequency_to_timedelta(freq_value, freq_unit)
    use_groupby_grid = (
        session_offset is not None
        and freq_unit in alignable_units
        and freq_delta is not None
    )

    # Determine aggregation map dynamically so we can handle close-only data.
    columns = {c.lower(): c for c in working.columns}
    has = lambda name: name in columns

    agg: dict[str, str] = {}
    if has("open"):
        agg[columns["open"]] = "first"
    if has("high"):
        agg[columns["high"]] = "max"
    if has("low"):
        agg[columns["low"]] = "min"
    if has("close"):
        agg[columns["close"]] = "last"
    if has("volume"):
        agg[columns["volume"]] = "sum"
    agg['__bucket_first_ts'] = 'first'

    if not agg:
        return df

    try:
        if use_groupby_grid:
            bucket_labels = (working.index - session_offset).floor(freq_code) + session_offset
            resampled = working.groupby(bucket_labels).agg(agg)
        else:
            resampled = working.resample(freq_code, label="right", closed="right").agg(agg)
    except Exception:
        return df

    bucket_first = resampled.pop('__bucket_first_ts') if '__bucket_first_ts' in resampled.columns else None

    # Insert NaNs for periods with no data to avoid drawing flat lines across breaks.
    resampled = resampled.where(~resampled.isna(), other=resampled)

    # Drop any bars where we failed to compute a closing value to avoid flat segments.
    close_col = columns.get("close")
    if close_col in resampled.columns:
        valid_mask = resampled[close_col].notna()
    else:
        valid_mask = ~resampled.isna().all(axis=1)

    if bucket_first is not None:
        valid_mask = valid_mask & bucket_first.notna()

    resampled = resampled.loc[valid_mask]
    if bucket_first is not None:
        bucket_first = bucket_first.loc[valid_mask]

    bucket_times: Optional[pd.Series] = None
    if (not use_groupby_grid) and bucket_first is not None and not bucket_first.empty:
        try:
            if is_datetime64_any_dtype(bucket_first):
                bucket_times = bucket_first
                if tz_info is not None and not is_datetime64tz_dtype(bucket_first):
                    bucket_times = bucket_times.dt.tz_localize(tz_info)
            else:
                bucket_times = pd.to_datetime(bucket_first)
                if tz_info is not None and not is_datetime64tz_dtype(bucket_times):
                    bucket_times = bucket_times.dt.tz_localize(tz_info)
        except Exception:
            bucket_times = None

    if bucket_times is not None and not bucket_times.empty:
        try:
            resampled.index = bucket_times
        except Exception:
            pass

    resampled.index.name = df.index.name

    return resampled
