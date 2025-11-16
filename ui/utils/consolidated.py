"""Helpers to build consolidated (resampled) OHLC data frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
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
    "minute": _UnitConfig(code="T", label="minute"),
    "hour": _UnitConfig(code="H", label="hour"),
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
    try:
        offset = to_offset(f"{value}{config.code}")
    except Exception:
        return None
    if hasattr(offset, "delta") and offset.delta is not None:
        return offset.delta
    if hasattr(offset, "nanos"):
        try:
            return pd.to_timedelta(offset.nanos, unit="ns")
        except Exception:
            return None
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
    # Determine aggregation map dynamically so we can handle close-only data.
    columns = {c.lower(): c for c in df.columns}
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

    if not agg:
        return df

    try:
        resampled = df.resample(freq_code, label="right", closed="right").agg(agg)
    except Exception:
        return df

    # Insert NaNs for periods with no data to avoid drawing flat lines across breaks.
    resampled = resampled.where(~resampled.isna(), other=resampled)

    # Drop any bars where we failed to compute a closing value to avoid flat segments.
    close_col = columns.get("close")
    if close_col in resampled.columns:
        resampled = resampled.dropna(subset=[close_col])
    else:
        resampled = resampled.dropna(how="all")

    return resampled
