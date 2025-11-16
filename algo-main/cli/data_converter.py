import pandas as pd, os, zipfile
from datetime import datetime

def convert_to_lean_csv(filepath, symbol, resolution, output_dir):
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
    convert_to_lean_csv("../20250805_hal_minute_trade.csv", "HAL", "minute", "../data/equity/india/minute")