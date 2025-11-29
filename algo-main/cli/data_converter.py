import pandas as pd, os, zipfile
from datetime import datetime

def convert_to_lean_csv(filepath, symbol, resolution, output_dir):
    """Convert a raw trade history CSV into Lean-compatible zipped files.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file that contains the raw trade
        history. The file is expected to expose at least the columns
        ``time``, ``open``, ``high``, ``low``, ``close`` and ``volume``.
    symbol : str
        Ticker symbol that should back the generated Lean file names. The
        function lower-cases this value when building destination folders.
    resolution : str
        Target Lean resolution. Only ``"minute"`` is currently supported.
    output_dir : str
        Destination directory that mirrors QuantConnect's Lean data layout.

    Returns
    -------
    None
        The function writes one zipped CSV per day to ``output_dir`` and
        removes the intermediate plain-text CSV after archiving. Resolutions
        other than ``"minute"`` are ignored. Any file-system errors are
        propagated to the caller.
    """
    df = pd.read_csv(filepath)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    if resolution == "minute":
        for date, gdf in df.groupby(df["time"].dt.date):
            day = date.strftime("%Y%m%d")
            fname = f"{output_dir}/{symbol.lower()}/{day}_trade.csv"
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            gdf[["time", "open", "high", "low", "close", "volume"]].to_csv(fname, index=False)
            with zipfile.ZipFile(fname.replace(".csv", ".zip"), 'w', zipfile.ZIP_DEFLATED) as z:
                z.write(fname, arcname=f"{symbol.lower()}/{day}_trade.csv")
            os.remove(fname)

if __name__ == "__main__":
    # Simple manual invocation for ad-hoc conversions; adjust paths as needed.
    convert_to_lean_csv("../20250805_hal_minute_trade.csv", "HAL", "minute", "../data/equity/india/minute")