import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Paths and directories
# --------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
LARGE_DATA_DIR = os.path.join(DATA_DIR, "large_data")

HEATMAP_DIR = os.path.join(BASE_DIR, "../../images/02_heatmap_correlation")
TREND_DIR = os.path.join(BASE_DIR, "../../images/03_features_plotting")

os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(TREND_DIR, exist_ok=True)

# Pattern for all OHLCV csvs created by the retriever
all_files = sorted(glob.glob(os.path.join(LARGE_DATA_DIR, "01_*pairs_OHLCV_*.csv")))

def process_file(csv_path: str) -> None:
    """Create correlation heatmap (and optional trend plot) for a single csv file."""
    filename = os.path.basename(csv_path)  # e.g. 01_BTCpairs_OHLCV_240Mins.csv
    name_no_ext = filename.replace(".csv", "")
    parts = name_no_ext.split("_")

    # 01_BTCpairs_OHLCV_240Mins -> group = BTCpairs, tf_label = 240Mins
    group = parts[1]          # 'BTCpairs' or 'USDTPairs'
    tf_label = parts[-1]      # '15Mins', '1Day', '240Mins', ...

    print(f"Processing {filename}  (group={group}, timeframe={tf_label})")

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Pivot: timestamp x symbol -> close prices
    df_close = df.pivot(index="timestamp", columns="symbol", values="close")
    df_close = df_close.dropna()

    # Log returns
    returns = np.log(df_close / df_close.shift(1)).dropna()

    # Correlation
    corr = returns.corr()

    # ---------------- Heatmap ----------------
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title(f"{group} Log-Return Correlation ({tf_label})")
    plt.tight_layout()

    heatmap_path = os.path.join(HEATMAP_DIR, f"{group}_{tf_label}_Correlation.png")
    plt.savefig(heatmap_path)
    plt.close()

    # ---------------- Trend plot (only for BTC/USDT 1Day) ----------------
    if group == "USDTPairs" and tf_label == "1Day" and "BTC/USDT" in df_close.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df_close.index, df_close["BTC/USDT"], label="BTC/USDT")
        plt.title("BTC/USDT Price Trend (1 Day timeframe)")
        plt.xlabel("Time")
        plt.ylabel("Close Price")
        plt.legend()
        plt.tight_layout()

        trend_path = os.path.join(TREND_DIR, "BTC_USDT_1Day_Trend.png")
        plt.savefig(trend_path)
        plt.close()
        print(f"Saved trend plot → {trend_path}")


def main():
    if not all_files:
        print("No OHLCV csv files found. Did you run OHLCV_retriever.py?")
        return

    for path in all_files:
        process_file(path)

    print("All data understanding plots generated.")


if __name__ == "__main__":
    main()
