"""
split_dataset_backtest.py

Create raw OHLCV splits for backtesting that match the feature-engineered dataset splits.
This allows the backtester to use real market data with proper temporal alignment.
"""

import os
import pandas as pd
import numpy as np
import json

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LARGE_DATA_DIR = os.path.join(BASE_DIR, "../../data/large_data")
SPLIT_DATA_DIR = os.path.join(BASE_DIR, "../../data/split_data")
os.makedirs(SPLIT_DATA_DIR, exist_ok=True)

# Load split metadata
METADATA_FILE = os.path.join(SPLIT_DATA_DIR, "split_metadata.json")

def load_split_metadata():
    """Load the date ranges from the main dataset split."""
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            f"Split metadata not found at {METADATA_FILE}. "
            "Please run split_dataset.py first to generate the splits."
        )
    
    with open(METADATA_FILE, 'r') as f:
        return json.load(f)

def load_base_ohlcv(base_pairs=None, base_timeframe="15m"):
    """
    Load OHLCV data for multiple pairs used for backtesting.
    
    Args:
        base_pairs: List of trading pairs (e.g., ["BTC/USDT", "ETH/USDT"])
        base_timeframe: Timeframe for the data
    
    Returns:
        Dictionary of DataFrames with OHLCV data, keyed by pair
    """
    if base_pairs is None:
        base_pairs = ["BTC/USDT"]
    
    # Map timeframe to filename
    timeframe_labels = {
        "15m": "15Mins",
        "1h": "60Mins",
        "4h": "240Mins",
        "1d": "1Day"
    }
    
    label = timeframe_labels[base_timeframe]
    
    # Load data for all pairs
    all_data = {}
    
    for pair in base_pairs:
        pair_type = "USDTpairs" if "USDT" in pair else "BTCpairs"
        filename = f"01_{pair_type}_OHLCV_{label}.csv"
        filepath = os.path.join(LARGE_DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: OHLCV data not found at {filepath}, skipping {pair}")
            continue
        
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        
        # CRITICAL: Filter for the specific pair
        if 'symbol' in df.columns:
            df = df[df['symbol'] == pair].copy()
            print(f"  Filtered for {pair}: {len(df)} rows")
        
        df.set_index('timestamp', inplace=True)
        
        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='first')].sort_index()
        
        # Keep required columns for backtesting
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
        df = df[required_cols]
        
        # Add pair identifier to column names
        pair_clean = pair.replace("/", "")
        df.columns = [f"{pair_clean}_{col}" for col in df.columns]
        
        all_data[pair] = df
    
    return all_data

def split_backtest_data(data_dict, metadata):
    """
    Split the raw OHLCV data for multiple pairs according to date ranges.
    
    Args:
        data_dict: Dictionary of DataFrames (one per pair)
        metadata: Split metadata with date ranges
    
    Returns:
        train_dict, val_dict, test_dict - dictionaries of split DataFrames per pair
    """
    # Parse dates from metadata
    train_start = pd.to_datetime(metadata['train']['start'])
    train_end = pd.to_datetime(metadata['train']['end'])
    val_start = pd.to_datetime(metadata['val']['start'])
    val_end = pd.to_datetime(metadata['val']['end'])
    test_start = pd.to_datetime(metadata['test']['start'])
    test_end = pd.to_datetime(metadata['test']['end'])
    
    train_dict = {}
    val_dict = {}
    test_dict = {}
    
    for pair, df in data_dict.items():
        # Split data by date ranges
        train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
        val_df = df[(df.index >= val_start) & (df.index <= val_end)].copy()
        test_df = df[(df.index >= test_start) & (df.index <= test_end)].copy()
        
        # Clean up inf/nan values
        for split_df in [train_df, val_df, test_df]:
            split_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            split_df.dropna(inplace=True)
        
        train_dict[pair] = train_df
        val_dict[pair] = val_df
        test_dict[pair] = test_df
    
    return train_dict, val_dict, test_dict

def main():
    """Main execution."""
    print("="*70)
    print("CREATING MULTI-PAIR BACKTEST DATASET FROM RAW OHLCV DATA")
    print("="*70)
    
    # Define pairs to include (must match training: 3 BTC pairs + 4 USDT pairs = 7 total)
    pairs = ["ETH/BTC", "LTC/BTC", "SOL/BTC", "BTC/USDT", "ETH/USDT", "LTC/USDT", "SOL/USDT"]
    
    # Load split metadata
    print("\nLoading split metadata...")
    metadata = load_split_metadata()
    print(f"Train: {metadata['train']['start']} to {metadata['train']['end']}")
    print(f"Val:   {metadata['val']['start']} to {metadata['val']['end']}")
    print(f"Test:  {metadata['test']['start']} to {metadata['test']['end']}")
    
    # Load OHLCV data for all pairs
    print(f"\nLoading raw OHLCV data for {len(pairs)} pairs (15m timeframe)...")
    data_dict = load_base_ohlcv(base_pairs=pairs, base_timeframe="15m")
    
    # Merge all pairs into single DataFrame with aligned timestamps
    print("\nAligning timestamps across all pairs...")
    # Use outer join to preserve USDT pair data even if BTC pairs have gaps
    merged_df = pd.concat(data_dict.values(), axis=1, join='outer')
    
    # Forward-fill BTC pairs (they're less liquid, may have gaps)
    btc_cols = [c for c in merged_df.columns if 'BTC_' in c and 'USDT' not in c]
    if btc_cols:
        print(f"Forward-filling {len(btc_cols)} BTC pair columns (limit=96 periods = 1 day)...")
        merged_df[btc_cols] = merged_df[btc_cols].ffill(limit=96)
    
    # Drop rows where USDT pairs are missing (they're the primary trading assets)
    usdt_cols = [c for c in merged_df.columns if 'USDT' in c]
    merged_df = merged_df.dropna(subset=usdt_cols, how='any')
    
    print(f"Merged data: {len(merged_df)} rows from {merged_df.index.min()} to {merged_df.index.max()}")
    
    # Split data
    print("\nSplitting data according to metadata dates...")
    train_df = merged_df[(merged_df.index >= pd.to_datetime(metadata['train']['start'])) & 
                         (merged_df.index <= pd.to_datetime(metadata['train']['end']))].copy()
    val_df = merged_df[(merged_df.index >= pd.to_datetime(metadata['val']['start'])) & 
                       (merged_df.index <= pd.to_datetime(metadata['val']['end']))].copy()
    test_df = merged_df[(merged_df.index >= pd.to_datetime(metadata['test']['start'])) & 
                        (merged_df.index <= pd.to_datetime(metadata['test']['end']))].copy()
    
    # Clean up
    for df in [train_df, val_df, test_df]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"  Val:   {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")
    print(f"  Test:  {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")
    
    # Save splits
    print("\nSaving backtest splits...")
    train_df.to_csv(os.path.join(SPLIT_DATA_DIR, "train_backtest_multi.csv"))
    val_df.to_csv(os.path.join(SPLIT_DATA_DIR, "val_backtest_multi.csv"))
    test_df.to_csv(os.path.join(SPLIT_DATA_DIR, "test_backtest_multi.csv"))
    
    # Also save as parquet for faster loading
    train_df.to_parquet(os.path.join(SPLIT_DATA_DIR, "train_backtest_multi.parquet"))
    val_df.to_parquet(os.path.join(SPLIT_DATA_DIR, "val_backtest_multi.parquet"))
    test_df.to_parquet(os.path.join(SPLIT_DATA_DIR, "test_backtest_multi.parquet"))
    
    print(f"\n{'='*70}")
    print("MULTI-PAIR BACKTEST DATASET CREATED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Location: {SPLIT_DATA_DIR}")
    print(f"Pairs included: {', '.join(pairs)}")
    print(f"Files created:")
    print(f"  - train_backtest_multi.parquet ({len(train_df)} rows)")
    print(f"  - val_backtest_multi.parquet ({len(val_df)} rows)")
    print(f"  - test_backtest_multi.parquet ({len(test_df)} rows)")
    print(f"\nColumns per pair: open, high, low, close, volume, vwap, trade_count")
    print(f"Use with multi-asset trading simulation for diversification.")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
