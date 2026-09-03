import pandas as pd
import numpy as np
import os
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
LARGE_DATA_DIR = os.path.join(BASE_DIR, "../../data/large_data")
IMAGE_DIR = os.path.join(BASE_DIR, "../../images/03_features_plotting")
IMAGE_DIR_ORG = os.path.join(BASE_DIR, "../../images")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR_ORG, exist_ok=True)

# Pairs with enough data 
BTC_PAIRS = ["ETH/BTC", "LTC/BTC", "SOL/BTC"]
USDT_PAIRS = ["BTC/USDT", "ETH/USDT", "LTC/USDT", "SOL/USDT"]

# Base timeframes
TIMEFRAMES = ["15m", "1h", "4h", "1d"] #extended as needed

TIMEFRAME_LABELS = {
    "15m": "15Mins",
    "1h": "60Mins",
    "4h": "240Mins",
    "1d": "1Day"
}

# Mapping timeframe strings to pandas offset aliases
TIMEFRAME_OFFSETS = {
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D"
}

# Pandas Dataframe from fetched data (csv file). Load function
def load_ohlcv(pair, timeframe):
    """
    Load OHLCV CSV from large_data directory
    """
    pair_type = "BTCpairs" if "BTC" in pair else "USDTpairs"
    pair_type = "BTCpairs" if "BTC" in pair else "USDTpairs"
    label = TIMEFRAME_LABELS[timeframe]
    filename = f"01_{pair_type}_OHLCV_{label}.csv"
    filepath = os.path.join(LARGE_DATA_DIR, filename)

    if not os.path.exists(filepath):
        print(f"Warning: {filepath} does not exist. Skipping {pair} {timeframe}")
        return None
    
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Remove duplicate timestamps (keep first occurrence)
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort by timestamp to ensure temporal order
    df = df.sort_index()

    # filter column 
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
    df = df[required_cols]

    return df

# Compute indicators for a single pair and timeframe
def compute_features(df, pair, timeframe):
    """
    Compute ML-ready features for a single crypto pair and timeframe.
    All features are stationary or relative to current price/previous value.
    """
    features = pd.DataFrame(index=df.index)


    # -----------------------------
    # 1. Log returns for OHLC prices
    # -----------------------------
    for col in ["open", "high", "low", "close"]:
        features[f"{pair}_{col}_logret_{timeframe}"] = np.log(df[col] / df[col].shift(1))

    # -----------------------------
    # 2. Volume & VWAP features
    # -----------------------------
    features[f"{pair}_volume_pct_{timeframe}"] = df["volume"].pct_change().fillna(0)
    features[f"{pair}_vwap_diff_{timeframe}"] = (df["vwap"] - df["close"]) / df["close"]
    features[f"{pair}_trade_count_pct_{timeframe}"] = df["trade_count"].pct_change().fillna(0)

    # -----------------------------
    # 3. Trend indicators (EMA, MACD)
    # -----------------------------
    ema_windows = [9, 21, 50]
    for w in ema_windows:
        ema = EMAIndicator(df["close"], window=w).ema_indicator()
        features[f"{pair}_EMA{w}_{timeframe}"] = ema / df["close"] - 1  # relative to price

    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    features[f"{pair}_MACD_{timeframe}"] = macd.macd() / df["close"]
    features[f"{pair}_MACD_signal_{timeframe}"] = macd.macd_signal() / df["close"]

    # -----------------------------
    # 4. Momentum indicators
    # -----------------------------
    features[f"{pair}_RSI14_{timeframe}"] = RSIIndicator(df["close"], window=14).rsi() / 100  # 0-1

    # -----------------------------
    # 5. Volatility indicators
    # -----------------------------
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    features[f"{pair}_ATR14_{timeframe}"] = atr / df["close"]  # relative

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    features[f"{pair}_BB_high_{timeframe}"] = bb.bollinger_hband() / df["close"] - 1
    features[f"{pair}_BB_low_{timeframe}"] = bb.bollinger_lband() / df["close"] - 1

    # -----------------------------
    # 6. Volume oscillator - OBV
    # -----------------------------
    obv = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    features[f"{pair}_OBV_pct_{timeframe}"] = obv.pct_change().fillna(0)

    # -----------------------------
    # 7. Multi-period log returns
    # -----------------------------
    features[f"{pair}_logret1_{timeframe}"] = np.log(df["close"] / df["close"].shift(1))
    features[f"{pair}_logret5_{timeframe}"] = np.log(df["close"] / df["close"].shift(5))

    # -----------------------------
    # 8. Optional: further normalized EMA (by local volatility)
    # -----------------------------
    # This can be uncommented if you want EMA relative to recent volatility
    # features[f"{pair}_EMA50_volnorm_{timeframe}"] = (EMAIndicator(df["close"], window=50).ema_indicator() - df["close"]) / df["close"].rolling(20).std()

    return features

# build ML-ready dataset from multiple pairs and timeframes
def build_dataset(pairs, timeframes, base_timeframe="15m"):
    """
    Build ML-ready dataset from multiple pairs and multiple timeframes
    """
    # Start with empty dataframe
    master_df = pd.DataFrame()
    
    for tf in timeframes:
        tf_dfs = []
        for pair in pairs:
            try:
                df = load_ohlcv(pair, tf)
                if df is None:
                    continue
                feat_df = compute_features(df, pair.replace("/", ""), tf)

                # Resample/align to base timeframe
                # CRITICAL: Use only PAST completed values to avoid look-ahead bias
                if tf == base_timeframe:
                    # Same timeframe - no resampling needed
                    feat_df = feat_df[~feat_df.index.duplicated(keep='first')].sort_index()
                    tf_dfs.append(feat_df)
                else:
                    # Higher timeframe - align to base frequency using forward-fill
                    # This propagates the last completed candle forward until the next one
                    feat_df_resampled = feat_df.resample(TIMEFRAME_OFFSETS[base_timeframe]).asfreq()
                    feat_df_resampled = feat_df_resampled.ffill()
                    feat_df_resampled = feat_df_resampled[~feat_df_resampled.index.duplicated(keep='first')]
                    tf_dfs.append(feat_df_resampled)
            except Exception as e:
                print(f"Skipping {pair} {tf} due to error: {e}")
                continue
        
        # Merge all features for this timeframe
        if tf_dfs:
            tf_features = pd.concat(tf_dfs, axis=1, join='outer')
            if master_df.empty:
                master_df = tf_features
            else:
                master_df = master_df.join(tf_features, how="outer")
    
    # Target column: predict future returns of crypto pairs
    # CRITICAL: Create targets FIRST before any NaN handling to maintain temporal integrity
    target_coins = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT"]

    for coin in target_coins:
        # Target: CUMULATIVE return over next 4 hours (16 periods ahead)
        # Use the close_logret feature to derive the cumulative return
        # We need to sum the log returns from now to 16 periods ahead
        logret_col = f"{coin}_close_logret_15m"
        
        # Cumulative return = sum of log returns
        master_df[f"{coin}_target_logret_4h"] = master_df[logret_col].rolling(16).sum().shift(-16)
        
        # Target: CUMULATIVE return over next 1 day (96 periods ahead)
        master_df[f"{coin}_target_logret_1d"] = master_df[logret_col].rolling(96).sum().shift(-96)
    
    # Now handle remaining NaN values
    # After resampling with ffill, we mainly need to handle:
    # 1. Initial indicator warm-up periods (EMA50, RSI14, BB20, etc.)
    # 2. Any small gaps from data quality issues
    feature_cols = [col for col in master_df.columns if "target" not in col]
    
    # Use a conservative forward-fill limit
    # Since indicators need warm-up (EMA50~50 periods, RSI14~14 periods, BB20~20 periods)
    # But we don't want to propagate stale data too far
    # limit=10 allows reasonable warm-up filling without excessive staleness
    for col in feature_cols:
        master_df[col] = master_df[col].ffill(limit=10)
    
    # Drop rows with ANY remaining NaN in features OR targets
    initial_rows = len(master_df)
    master_df = master_df.dropna()
    print(f"\nDropped {initial_rows - len(master_df)} rows with NaN values")
    print(f"Final dataset shape: {master_df.shape}")
    print(f"Date range: {master_df.index.min()} to {master_df.index.max()}")
    
    # Sanity check: print target statistics
    target_cols = [col for col in master_df.columns if "target" in col]
    print(f"\nTarget statistics:")
    for col in target_cols:
        print(f"  {col}: mean={master_df[col].mean():.6f}, std={master_df[col].std():.6f}")
    
    return master_df

# ==============================
# Main Execution
# ==============================

if __name__ == "__main__":
    # Combine BTC & USDT pairs
    all_pairs = BTC_PAIRS + USDT_PAIRS
    
    # Build the dataset
    dataset = build_dataset(all_pairs, TIMEFRAMES, base_timeframe="15m")
    
    # Optionally, save dataset
    dataset.to_parquet(os.path.join(LARGE_DATA_DIR, "03_crypto_ml_dataset.parquet"))
    
    print("Dataset ready!")
    print(dataset.head())

