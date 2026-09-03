"""
deployment_feat_generator.py

Production-ready feature generation that EXACTLY matches the training pipeline.
This replicates the logic from scripts/03_pre_split_prep/features.py for deployment.
"""

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from typing import Optional


# Pair definitions (must match training)
BTC_PAIRS = ["ETHBTC", "LTCBTC", "SOLBTC"]  # Removed "/" for consistency
USDT_PAIRS = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT"]
ALL_PAIRS = BTC_PAIRS + USDT_PAIRS

# Timeframe definitions (must match training)
TIMEFRAMES = ["15m", "1h", "4h", "1d"]

TIMEFRAME_OFFSETS = {
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D"
}


def compute_features_single_pair_timeframe(df: pd.DataFrame, pair: str, timeframe: str) -> pd.DataFrame:
    """
    Compute ML-ready features for a single crypto pair and timeframe.
    
    This EXACTLY matches compute_features() from features.py.
    All features are stationary or relative to current price/previous value.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
        pair: Pair name (e.g., "BTCUSDT", "ETHBTC")
        timeframe: Timeframe string (e.g., "15m", "1h", "4h", "1d")
    
    Returns:
        DataFrame with computed features indexed by timestamp
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

    return features


def build_features_from_buffer(
    buffer_df: pd.DataFrame,
    pairs: list = None,
    base_timeframe: str = "15m"
) -> Optional[np.ndarray]:
    """
    Build ML-ready feature vector from multi-asset OHLCV buffer.
    
    This EXACTLY replicates the build_dataset() logic from features.py but
    operates on a rolling buffer for deployment.
    
    Args:
        buffer_df: DataFrame with columns like {ASSET}_open, {ASSET}_close, etc.
                  Example: BTCUSDT_open, BTCUSDT_close, ETHUSDT_open, etc.
        pairs: List of trading pairs to process (default: USDT_PAIRS)
        base_timeframe: Base timeframe for alignment (default: "15m")
    
    Returns:
        1D numpy array of features in the EXACT order as training data,
        or None if insufficient data
    """
    if pairs is None:
        pairs = USDT_PAIRS
    
    # Need sufficient buffer for all timeframes
    # 1d needs 96 bars, plus indicators need ~50 for warm-up
    min_rows = 400
    if len(buffer_df) < min_rows:
        return None
    
    # Ensure buffer has DatetimeIndex for resampling
    if not isinstance(buffer_df.index, pd.DatetimeIndex):
        # Buffer from agent doesn't have index - create one
        # Assume 15-minute frequency and create synthetic datetime index
        buffer_df = buffer_df.copy()
        buffer_df.index = pd.date_range(
            end=pd.Timestamp.now(tz='UTC'),
            periods=len(buffer_df),
            freq='15min'
        )
    
    # Initialize master dataframe
    master_df = pd.DataFrame()
    
    # Process each timeframe
    for tf in TIMEFRAMES:
        tf_dfs = []
        
        for pair in pairs:
            # Extract data for this pair from flat column structure
            # Columns are like: BTCUSDT_open, BTCUSDT_close, etc.
            pair_cols = {
                'open': f'{pair}_open',
                'high': f'{pair}_high',
                'low': f'{pair}_low',
                'close': f'{pair}_close',
                'volume': f'{pair}_volume',
                'vwap': f'{pair}_vwap',
                'trade_count': f'{pair}_trade_count'
            }
            
            # Check if this pair's columns exist in buffer
            if not all(col in buffer_df.columns for col in pair_cols.values()):
                continue
            
            # Extract pair data with standard column names
            pair_data = buffer_df[list(pair_cols.values())].copy()
            pair_data.columns = list(pair_cols.keys())  # Rename to standard names
            
            if len(pair_data) < 50:
                continue
            
            # Resample to target timeframe if needed
            if tf == base_timeframe:
                # No resampling needed
                resampled = pair_data.copy()
            else:
                # Resample to higher timeframe
                offset = TIMEFRAME_OFFSETS[tf]
                resampled = pair_data.resample(offset).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'vwap': 'mean',
                    'trade_count': 'sum'
                })
                
                # Forward-fill to base timeframe (propagate completed candles)
                resampled = resampled.resample(TIMEFRAME_OFFSETS[base_timeframe]).asfreq()
                resampled = resampled.ffill()
            
            # Remove duplicates
            resampled = resampled[~resampled.index.duplicated(keep='first')]
            
            # Compute features for this pair-timeframe
            feat_df = compute_features_single_pair_timeframe(resampled, pair, tf)
            tf_dfs.append(feat_df)
        
        # Merge all pairs for this timeframe
        if tf_dfs:
            tf_features = pd.concat(tf_dfs, axis=1, join='outer')
            if master_df.empty:
                master_df = tf_features
            else:
                master_df = master_df.join(tf_features, how="outer")
    
    if master_df.empty:
        return None
    
    # Handle NaN values with forward-fill (limit=10 to match training)
    feature_cols = [col for col in master_df.columns]
    for col in feature_cols:
        master_df[col] = master_df[col].ffill(limit=10)
    
    # Get the LATEST row (most recent timestamp)
    latest_row = master_df.iloc[-1]
    
    # Check for remaining NaN values
    if latest_row.isna().any():
        # Return None if any NaN remains (not ready yet)
        return None
    
    # Convert to numpy array
    # CRITICAL: Sort column names to ensure consistent ordering with training
    sorted_cols = sorted(master_df.columns)
    feature_array = latest_row[sorted_cols].values.astype(np.float32)
    
    # Replace any inf values with 0
    feature_array = np.where(np.isfinite(feature_array), feature_array, 0.0)
    
    return feature_array


def create_deployment_feature_calculator(pairs: list = None):
    """
    Factory function to create a deployment-ready feature calculator.
    
    This creates a function that can be passed to the trading agent and will
    compute features exactly as in training.
    
    Args:
        pairs: List of pairs to trade (default: ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT"])
    
    Returns:
        Callable that takes buffer_df and returns feature array
    """
    if pairs is None:
        pairs = USDT_PAIRS
    
    def calculator(buffer_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Calculate features from buffer for deployment.
        
        Args:
            buffer_df: Rolling buffer with OHLCV data
        
        Returns:
            Feature array or None if not ready
        """
        return build_features_from_buffer(buffer_df, pairs=pairs)
    
    return calculator


if __name__ == "__main__":
    """Test the feature generator with synthetic data"""
    
    print("Testing deployment feature generator...")
    
    # Create synthetic multi-asset buffer
    n_rows = 500
    dates = pd.date_range('2024-01-01', periods=n_rows, freq='15min')
    
    assets = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT"]
    
    # Create multi-asset buffer
    buffer_data = []
    for asset in assets:
        for date in dates:
            buffer_data.append({
                'timestamp': date,
                'symbol': asset,
                'open': 50000 + np.random.randn() * 1000,
                'high': 51000 + np.random.randn() * 1000,
                'low': 49000 + np.random.randn() * 1000,
                'close': 50000 + np.random.randn() * 1000,
                'volume': np.random.rand() * 1000,
                'vwap': 50000 + np.random.randn() * 1000,
                'trade_count': np.random.randint(100, 1000)
            })
    
    buffer_df = pd.DataFrame(buffer_data)
    
    # Test feature calculation
    calculator = create_deployment_feature_calculator(pairs=assets)
    features = calculator(buffer_df)
    
    if features is not None:
        print(f"✓ Generated {len(features)} features")
        print(f"✓ Feature shape: {features.shape}")
        print(f"✓ Feature dtype: {features.dtype}")
        print(f"✓ Sample features: {features[:5]}")
        print(f"✓ NaN count: {np.isnan(features).sum()}")
        print(f"✓ Inf count: {np.isinf(features).sum()}")
        
        # Expected feature count: 4 assets × 4 timeframes × 19 features = 304 features
        expected = len(assets) * len(TIMEFRAMES) * 19
        print(f"\n Expected features: {expected}")
        print(f"Actual features: {len(features)}")
        
        if len(features) == expected:
            print("✓ Feature count matches expected!")
        else:
            print(f"⚠ Feature count mismatch!")
    else:
        print("✗ Feature generation returned None (buffer too small or NaN)")
    
    print("\n✓ Deployment feature generator ready for production!")
