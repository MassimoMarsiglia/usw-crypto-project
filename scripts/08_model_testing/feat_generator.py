"""
feat_generator.py

Feature generation for live trading simulation based on the features.py logic.
This module provides real-time feature calculation from a rolling buffer of OHLCV data.
"""

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


def compute_features_single_timeframe(df: pd.DataFrame, pair: str, timeframe: str) -> dict:
    """
    Compute features for a single pair at a single timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        pair: Trading pair name
        timeframe: Timeframe label (e.g., "15m", "1h", "4h", "1d")
    
    Returns:
        Dictionary of features
    """
    features = {}
    
    # Need minimum data for indicators
    if len(df) < 50:
        return None
    
    # -----------------------------
    # 1. Log returns for OHLC prices
    # -----------------------------
    for col in ["open", "high", "low", "close"]:
        logret = np.log(df[col] / df[col].shift(1))
        features[f"{pair}_{col}_logret_{timeframe}"] = logret.iloc[-1]
    
    # -----------------------------
    # 2. Volume & VWAP features
    # -----------------------------
    features[f"{pair}_volume_pct_{timeframe}"] = df["volume"].pct_change().iloc[-1]
    features[f"{pair}_vwap_diff_{timeframe}"] = ((df["vwap"].iloc[-1] - df["close"].iloc[-1]) / df["close"].iloc[-1])
    features[f"{pair}_trade_count_pct_{timeframe}"] = df["trade_count"].pct_change().iloc[-1]
    
    # -----------------------------
    # 3. Trend indicators (EMA, MACD)
    # -----------------------------
    ema_windows = [9, 21, 50]
    for w in ema_windows:
        ema = EMAIndicator(df["close"], window=w).ema_indicator()
        features[f"{pair}_EMA{w}_{timeframe}"] = ema.iloc[-1] / df["close"].iloc[-1] - 1
    
    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    features[f"{pair}_MACD_{timeframe}"] = macd.macd().iloc[-1] / df["close"].iloc[-1]
    features[f"{pair}_MACD_signal_{timeframe}"] = macd.macd_signal().iloc[-1] / df["close"].iloc[-1]
    
    # -----------------------------
    # 4. Momentum indicators
    # -----------------------------
    features[f"{pair}_RSI14_{timeframe}"] = RSIIndicator(df["close"], window=14).rsi().iloc[-1] / 100
    
    # -----------------------------
    # 5. Volatility indicators
    # -----------------------------
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    features[f"{pair}_ATR14_{timeframe}"] = atr.iloc[-1] / df["close"].iloc[-1]
    
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    features[f"{pair}_BB_high_{timeframe}"] = bb.bollinger_hband().iloc[-1] / df["close"].iloc[-1] - 1
    features[f"{pair}_BB_low_{timeframe}"] = bb.bollinger_lband().iloc[-1] / df["close"].iloc[-1] - 1
    
    # -----------------------------
    # 6. Volume oscillator - OBV
    # -----------------------------
    obv = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    features[f"{pair}_OBV_pct_{timeframe}"] = obv.pct_change().iloc[-1]
    
    # -----------------------------
    # 7. Multi-period log returns
    # -----------------------------
    features[f"{pair}_logret1_{timeframe}"] = np.log(df["close"].iloc[-1] / df["close"].iloc[-2]) if len(df) > 1 else 0.0
    features[f"{pair}_logret5_{timeframe}"] = np.log(df["close"].iloc[-1] / df["close"].iloc[-6]) if len(df) > 5 else 0.0
    
    # Replace NaN/inf with 0
    for k in features:
        if not np.isfinite(features[k]):
            features[k] = 0.0
    
    return features


def compute_features_from_buffer(buffer_df: pd.DataFrame, pairs: list, timeframes_config: dict) -> np.ndarray:
    """
    Compute multi-asset, multi-timeframe features from a rolling buffer.
    
    Args:
        buffer_df: Multi-index DataFrame with (timestamp, symbol) index and OHLCV columns
        pairs: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT"])
        timeframes_config: Dict mapping timeframe labels to resample rules
                          e.g., {"15m": "15T", "1h": "1H", "4h": "4H", "1d": "1D"}
    
    Returns:
        1D numpy array of all features in consistent order, or None if insufficient data
    """
    # Need sufficient data for all timeframes
    min_rows_needed = {
        "15m": 400,  # ~100 hours of 15m data
        "1h": 100,   # Need at least this for 1h resampling
        "4h": 50,    # For 4h indicators
        "1d": 50     # For 1d indicators
    }
    
    if len(buffer_df) < min_rows_needed.get("15m", 400):
        return None
    
    all_features = {}
    
    # For each pair
    for pair in pairs:
        # Extract data for this pair
        pair_data = buffer_df[buffer_df.index.get_level_values('symbol') == pair].copy()
        pair_data = pair_data.droplevel('symbol')  # Remove symbol level from index
        
        if len(pair_data) < 50:
            continue
        
        # For each timeframe
        for tf_label, resample_rule in timeframes_config.items():
            # Resample to target timeframe
            if tf_label == "15m":
                # No resampling needed for base timeframe
                resampled = pair_data
            else:
                # Resample OHLCV data
                resampled = pair_data.resample(resample_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'vwap': 'mean',  # Average VWAP
                    'trade_count': 'sum'
                }).dropna()
            
            if len(resampled) < 50:
                continue
            
            # Compute features for this pair-timeframe combination
            feat_dict = compute_features_single_timeframe(resampled, pair, tf_label)
            
            if feat_dict:
                all_features.update(feat_dict)
    
    # Return None if no features computed
    if not all_features:
        return None
    
    # Convert to ordered array (must match training order)
    # Sort by feature name to ensure consistent ordering
    sorted_features = sorted(all_features.items())
    feature_array = np.array([v for k, v in sorted_features], dtype=np.float32)
    
    return feature_array


def create_feature_calculator(pairs: list = None, timeframes: dict = None):
    """
    Factory function to create a multi-asset, multi-timeframe feature calculator.
    
    Args:
        pairs: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT"])
        timeframes: Dict mapping labels to pandas resample rules
                   e.g., {"15m": "15T", "1h": "1H", "4h": "4H", "1d": "1D"}
    
    Returns:
        Function that takes a buffer_df and returns feature array
    """
    if pairs is None:
        pairs = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT"]
    
    if timeframes is None:
        timeframes = {
            "15m": "15T",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D"
        }
    
    def calculator(buffer_df: pd.DataFrame) -> np.ndarray:
        """Calculate multi-timeframe features from buffer for all pairs."""
        return compute_features_from_buffer(buffer_df, pairs, timeframes)
    
    return calculator


if __name__ == "__main__":
    # Test feature generation
    np.random.seed(42)
    
    # Create dummy OHLCV data
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='15min')
    
    test_df = pd.DataFrame({
        'open': np.random.randn(n).cumsum() + 50000,
        'high': np.random.randn(n).cumsum() + 50100,
        'low': np.random.randn(n).cumsum() + 49900,
        'close': np.random.randn(n).cumsum() + 50000,
        'volume': np.random.rand(n) * 1000,
        'vwap': np.random.randn(n).cumsum() + 50000,
        'trade_count': np.random.randint(100, 1000, n)
    }, index=dates)
    
    # Test single pair
    features = compute_features_from_buffer(test_df, pair="BTCUSDT", timeframe="15m")
    print(f"Generated {len(features)} features")
    print(f"Feature sample: {features[:5]}")
    
    # Test multi-pair calculator
    calculator = create_feature_calculator(pairs=["BTCUSDT", "ETHUSDT"])
    multi_features = calculator(test_df)
    if multi_features is not None:
        print(f"\nMulti-pair features: {len(multi_features)} features")
