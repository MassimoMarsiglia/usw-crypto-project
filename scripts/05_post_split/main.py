import pandas as pd
import os
import numpy as np
import torch

#config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
LARGE_DATA_DIR = os.path.join(BASE_DIR, "../../data/large_data")
SPLIT_DATA_DIR = os.path.join(BASE_DIR, "../../data/split_data")
IMAGE_DIR = os.path.join(BASE_DIR, "../../images/03_features_plotting")
IMAGE_DIR_ORG = os.path.join(BASE_DIR, "../../images")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR_ORG, exist_ok=True)
DATASET_FILE = os.path.join(LARGE_DATA_DIR, "03_crypto_ml_dataset.parquet")
POST_SPLIT_DIR = os.path.join(BASE_DIR, "../../data/post_split_data")
os.makedirs(POST_SPLIT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(SPLIT_DATA_DIR, "train.parquet")
VAL_FILE   = os.path.join(SPLIT_DATA_DIR, "val.parquet")
TEST_FILE  = os.path.join(SPLIT_DATA_DIR, "test.parquet")

# Read parquet files
train_df = pd.read_parquet(TRAIN_FILE)
val_df = pd.read_parquet(VAL_FILE)
test_df = pd.read_parquet(TEST_FILE)

# Define X and y columns
IGNORE_COLS = ["SMA200", "regime", "regime_block"]
FEATURE_COLUMNS = [col for col in train_df.columns if "target" not in col and col not in IGNORE_COLS]
TARGET_COLUMNS  = [col for col in train_df.columns if "target" in col and col not in IGNORE_COLS]

# save features columns
FEATURE_COLUMNS_FILE = os.path.join(POST_SPLIT_DIR, "feature_columns.pt")
torch.save(FEATURE_COLUMNS, FEATURE_COLUMNS_FILE)
print(f"[INFO] Saved feature columns ({len(FEATURE_COLUMNS)}) to {FEATURE_COLUMNS_FILE}")

# Drop missing values
train_df = train_df.dropna(subset=FEATURE_COLUMNS)
val_df   = val_df.dropna(subset=FEATURE_COLUMNS)
test_df  = test_df.dropna(subset=FEATURE_COLUMNS)

# Save post-split data as parquet
train_df[FEATURE_COLUMNS].to_parquet(os.path.join(POST_SPLIT_DIR, "X_train.parquet"))
train_df[TARGET_COLUMNS].to_parquet(os.path.join(POST_SPLIT_DIR, "y_train.parquet"))

val_df[FEATURE_COLUMNS].to_parquet(os.path.join(POST_SPLIT_DIR, "X_val.parquet"))
val_df[TARGET_COLUMNS].to_parquet(os.path.join(POST_SPLIT_DIR, "y_val.parquet"))

test_df[FEATURE_COLUMNS].to_parquet(os.path.join(POST_SPLIT_DIR, "X_test.parquet"))
test_df[TARGET_COLUMNS].to_parquet(os.path.join(POST_SPLIT_DIR, "y_test.parquet"))

print("Post-split datasets saved successfully!")
print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
