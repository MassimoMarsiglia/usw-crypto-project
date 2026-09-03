import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LARGE_DATA_DIR = os.path.join(BASE_DIR, "../../data/large_data")
OUT_DIR = os.path.join(BASE_DIR, "../../images/03_features_plotting")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES_FILE = "03_crypto_ml_dataset.csv"

def main():
    # Load full feature dataset
    csv_path = os.path.join(LARGE_DATA_DIR, FEATURES_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Features file not found: {csv_path}")

    print(f"Using features file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Settings according to Hana:
    #  - timeframe: 15m
    #  - 4 USDT pairs
    #  - indicators: EMA9, MACD, RSI14
    pairs = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT"]
    indicators = ["EMA9", "MACD", "RSI14"]
    timeframe = "15m"

    # Build expected column names, e.g. BTCUSDT_EMA9_15m
    wanted_cols = [
        f"{pair}_{ind}_{timeframe}"
        for pair in pairs
        for ind in indicators
    ]

    # Keep only those that actually exist in the dataframe
    existing_cols = [c for c in wanted_cols if c in df.columns]

    if not existing_cols:
        raise ValueError("No matching technical indicator columns found. "
                         "Check pairs/indicators/timeframe names.")

    print("Using these technical indicator columns:")
    for c in existing_cols:
        print("  -", c)

    corr = df[existing_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Technical Indicator Correlation (4 USDT pairs, 15m)")
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "USDT_4pairs_15m_technical_correlation.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved clean technical correlation heatmap → {out_path}")

if __name__ == "__main__":
    main()
