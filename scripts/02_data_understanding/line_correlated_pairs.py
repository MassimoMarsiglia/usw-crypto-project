import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
LARGE_DATA_DIR = os.path.join(DATA_DIR, "large_data")
OUT_DIR = os.path.join(BASE_DIR, "../../images/03_features_plotting")
os.makedirs(OUT_DIR, exist_ok=True)

def load_usdt_1day():
    # try to find 1-day USDT file automatically
    candidates = glob.glob(os.path.join(LARGE_DATA_DIR, "01_USDTpairs_OHLCV_1Day*.csv"))
    if not candidates:
        raise FileNotFoundError("No 1Day USDT OHLCV file found in large_data.")
    csv_path = candidates[0]
    print(f"Using USDT 1Day file: {csv_path}")

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # pivot to wide: timestamp x symbol -> close
    df_close = df.pivot(index="timestamp", columns="symbol", values="close")
    df_close = df_close.dropna()

    return df_close

def find_correlated_pairs(df_close: pd.DataFrame):
    # log returns
    returns = np.log(df_close / df_close.shift(1)).dropna()
    corr = returns.corr()

    # flatten upper triangle (without diagonal)
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    stacked = corr.where(mask).stack()  # index: (sym1, sym2)

    # two strongest positive correlations
    strongest_pos = stacked.sort_values(ascending=False).head(2)

    # most negative correlation, or weakest positive if none negative
    negative = stacked[stacked < 0]
    if not negative.empty:
        weakest_pair = negative.idxmin()
        weakest_value = negative.min()
    else:
        weakest_pair = stacked.idxmin()
        weakest_value = stacked.min()

    print("\nTop positive correlations:")
    for (s1, s2), v in strongest_pos.items():
        print(f"{s1} – {s2}: {v:.3f}")

    print("\nMost negative (or least positive) correlation:")
    print(f"{weakest_pair[0]} – {weakest_pair[1]}: {weakest_value:.3f}")

    return strongest_pos, (weakest_pair, weakest_value)

def plot_pair(df_close: pd.DataFrame, sym1: str, sym2: str, corr_value: float, tag: str):
    # normalize to start at 1.0 so shapes are comparable
    sub = df_close[[sym1, sym2]].copy()
    sub_norm = sub / sub.iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(sub_norm.index, sub_norm[sym1], label=sym1)
    plt.plot(sub_norm.index, sub_norm[sym2], label=sym2)
    plt.title(f"{sym1} vs {sym2} – 1D normalized price (corr = {corr_value:.2f})")
    plt.xlabel("Time")
    plt.ylabel("Normalized Close Price")
    plt.legend()
    plt.tight_layout()

    fname = f"02_line_{tag}_{sym1.replace('/', '-')}_{sym2.replace('/', '-')}.png"
    out_path = os.path.join(OUT_DIR, fname)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved line chart → {out_path}")

def main():
    df_close = load_usdt_1day()
    strongest_pos, (weakest_pair, weakest_value) = find_correlated_pairs(df_close)

    # positive pair 1 & 2
    for i, ((s1, s2), v) in enumerate(strongest_pos.items(), start=1):
        plot_pair(df_close, s1, s2, v, tag=f"strong_pos{i}")

    # negative / weakest pair
    s1, s2 = weakest_pair
    plot_pair(df_close, s1, s2, weakest_value, tag="weak_or_neg")

if __name__ == "__main__":
    main()
