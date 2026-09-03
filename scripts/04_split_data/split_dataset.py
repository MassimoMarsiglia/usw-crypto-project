import os
import pandas as pd
import numpy as np
import json

# config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
LARGE_DATA_DIR = os.path.join(BASE_DIR, "../../data/large_data")
SPLIT_DATA_DIR = os.path.join(BASE_DIR, "../../data/split_data")
IMAGE_DIR = os.path.join(BASE_DIR, "../../images/03_features_plotting")
IMAGE_DIR_ORG = os.path.join(BASE_DIR, "../../images")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR_ORG, exist_ok=True)
DATASET_FILE = os.path.join(LARGE_DATA_DIR, "03_crypto_ml_dataset.parquet")

TRAIN_FILE = "train.parquet"
VAL_FILE   = "val.parquet"
TEST_FILE  = "test.parquet"

# Ratios for splits
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15 

# -------------------------------
# 1) Load dataset
# -------------------------------
df = pd.read_parquet(DATASET_FILE)

print(f"Original dataset size: {len(df)} rows")

# -------------------------------
# 2) Identify features & targets (BEFORE splitting)
# -------------------------------
IGNORE_COLS = []
FEATURE_COLUMNS = [col for col in df.columns if "target" not in col]
TARGET_COLUMNS  = [col for col in df.columns if "target" in col]

print(f"Features: {len(FEATURE_COLUMNS)}, Targets: {len(TARGET_COLUMNS)}")

# -------------------------------
# 3) Split data by timestamp FIRST (STRICT temporal ordering)
# -------------------------------
# CRITICAL: Split BEFORE normalization to prevent leakage
total_len = len(df)
train_end = int(total_len * TRAIN_RATIO)
val_end   = int(total_len * (TRAIN_RATIO + VAL_RATIO))

# Split the data
train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()

print(f"\nSplit sizes (before normalization) - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Date ranges:")
print(f"  Train: {train_df.index.min()} to {train_df.index.max()}")
print(f"  Val:   {val_df.index.min()} to {val_df.index.max()}")
print(f"  Test:  {test_df.index.min()} to {test_df.index.max()}")

# No normalization - log returns are already stationary
# Model will use layer normalization instead

# Clean up inf/nan values in each split
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
val_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN
train_df.dropna(subset=FEATURE_COLUMNS, inplace=True)
val_df.dropna(subset=FEATURE_COLUMNS, inplace=True)
test_df.dropna(subset=FEATURE_COLUMNS, inplace=True)

print(f"\nFinal split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Final date ranges:")
print(f"  Train: {train_df.index.min()} to {train_df.index.max()}")
print(f"  Val:   {val_df.index.min()} to {val_df.index.max()}")
print(f"  Test:  {test_df.index.min()} to {test_df.index.max()}")

# -------------------------------
# 5) Save splits
# -------------------------------
os.makedirs(SPLIT_DATA_DIR, exist_ok=True)

train_df.to_parquet(os.path.join(SPLIT_DATA_DIR, TRAIN_FILE))
val_df.to_parquet(os.path.join(SPLIT_DATA_DIR, VAL_FILE))
test_df.to_parquet(os.path.join(SPLIT_DATA_DIR, TEST_FILE))

# Save split metadata (date ranges) for backtesting
split_metadata = {
    "train": {
        "start": str(train_df.index.min()),
        "end": str(train_df.index.max()),
        "rows": len(train_df)
    },
    "val": {
        "start": str(val_df.index.min()),
        "end": str(val_df.index.max()),
        "rows": len(val_df)
    },
    "test": {
        "start": str(test_df.index.min()),
        "end": str(test_df.index.max()),
        "rows": len(test_df)
    },
    "split_ratios": {
        "train": TRAIN_RATIO,
        "val": VAL_RATIO,
        "test": TEST_RATIO
    }
}

metadata_file = os.path.join(SPLIT_DATA_DIR, "split_metadata.json")
with open(metadata_file, 'w') as f:
    json.dump(split_metadata, f, indent=2)

print(f"\n{'='*60}")
print(f"SPLITS SAVED SUCCESSFULLY - NO NORMALIZATION")
print(f"{'='*60}")
print(f"Location: {SPLIT_DATA_DIR}")
print(f"Final sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Total features: {len(FEATURE_COLUMNS)}")
print(f"Total targets: {len(TARGET_COLUMNS)}")
print(f"\nSplit metadata saved to: {metadata_file}")
print(f"\nNote: Features are NOT normalized (log returns are stationary)")
print(f"      Use layer normalization in your neural network instead")
print(f"{'='*60}")