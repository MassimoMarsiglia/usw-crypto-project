import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LARGE_DATA_DIR = os.path.join(BASE_DIR, "../../data/large_data")
OUT_DIR = os.path.join(BASE_DIR, "../../images/03_features_plotting")
os.makedirs(OUT_DIR, exist_ok=True)

# HIER ggf. den Dateinamen anpassen, falls er anders heißt:
FEATURES_FILE = "03_crypto_ml_dataset.csv"

def main():
    csv_path = os.path.join(LARGE_DATA_DIR, FEATURES_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Features file not found: {csv_path}")

    print(f"Using features file: {csv_path}")
    df = pd.read_csv(csv_path)

    tech_keywords = ["ema", "rsi", "macd"]
    tech_cols = [c for c in df.columns if any(k in c.lower() for k in tech_keywords)]

    if not tech_cols:
        raise ValueError("No technical indicator columns (ema/rsi/macd) found in features dataset.")

    print(f"Selected {len(tech_cols)} technical columns:")
    for c in tech_cols:
        print("  -", c)

    corr = df[tech_cols].corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Technical Indicator Correlation")
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "03_technical_indicators_correlation.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved technical indicator correlation heatmap → {out_path}")

if __name__ == "__main__":
    main()
