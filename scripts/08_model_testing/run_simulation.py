"""
run_simulation.py

Example script to run the realistic trading simulation with the new architecture:
- TradingAgent (with model inside)
- SimulationEnv (gym-like environment)
- CryptoBacktester (runs agent in env, logs to TensorBoard)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crypto_backtester import CryptoBacktester
from trading_agent import TradingAgent, CryptoMLP
from simulation_env import TradingSimulationEnv
from feat_generator import compute_features_from_buffer
from trading_rules import (
    get_conservative_rules,
    get_aggressive_rules,
    get_balanced_rules,
    ThresholdRule,
    RiskManagementRule,
    CompositeRule
)
from simulation_env import ALPACA_FEE_TIERS

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../models/07_01_feed_forward.pt")
POST_SPLIT_DIR = os.path.join(BASE_DIR, "../../data/post_split_data")
SPLIT_DATA_DIR = os.path.join(BASE_DIR, "../../data/split_data")
X_test_path = os.path.join(POST_SPLIT_DIR, "X_test.parquet")
y_test_path = os.path.join(POST_SPLIT_DIR, "y_test.parquet")
TEST_FILE = os.path.join(SPLIT_DATA_DIR, "test.csv")
TEST_BACKTEST_FILE = os.path.join(SPLIT_DATA_DIR, "test_backtest.parquet")


def load_raw_test_data():
    """
    Load raw OHLCV data for the test set.
    
    This should be the original market data (not normalized features).
    First tries to load test_backtest.parquet (preferred), then falls back to test.csv.
    """
    # Try backtest file first (has proper OHLCV data)
    if os.path.exists(TEST_BACKTEST_FILE):
        df = pd.read_parquet(TEST_BACKTEST_FILE)
        print(f"Loaded raw test data from test_backtest.parquet with {len(df)} rows")
        return df
    
    # Fallback to test.csv
    if os.path.exists(TEST_FILE):
        df = pd.read_csv(TEST_FILE, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(TEST_FILE, nrows=1).columns else None)
        
        # Check if we have OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if all(col in df.columns for col in required_cols):
            print(f"Loaded raw test data from test.csv with {len(df)} rows")
            return df
        else:
            print("Warning: test.csv doesn't have OHLCV columns.")
            print("Please run scripts/04_split_data/split_dataset_backtest.py to create proper backtest data.")
            print("Using synthetic data for now...")
            return None
    else:
        print("No test data found. Please run split_dataset_backtest.py")
        print("Using synthetic data for now...")
        return None


def create_synthetic_data(n_rows):
    """Create synthetic OHLCV data for testing."""
    dates = pd.date_range('2024-01-01', periods=n_rows, freq='15min')
    
    # Generate realistic-looking price data
    close = 50000 + np.random.randn(n_rows).cumsum() * 100
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close + np.random.randn(n_rows) * 50,
        'high': close + np.abs(np.random.randn(n_rows)) * 100,
        'low': close - np.abs(np.random.randn(n_rows)) * 100,
        'close': close,
        'volume': np.abs(np.random.randn(n_rows)) * 1000 + 500,
        'vwap': close + np.random.randn(n_rows) * 30,
        'trade_count': np.random.randint(100, 1000, n_rows)
    })
    
    return df


def main():
    """Run the trading simulation."""
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Load test features and targets
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print("Error: Test data not found. Please run feature generation first.")
        return
    
    X_test = pd.read_parquet(X_test_path).to_numpy(dtype=np.float32)
    y_test = pd.read_parquet(y_test_path).to_numpy(dtype=np.float32)
    print(f"Loaded test data: X shape={X_test.shape}, y shape={y_test.shape}")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = CryptoMLP(
        input_dim=X_test.shape[1],
        output_dim=y_test.shape[1],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")
    
    # Load or create raw market data
    raw_data = load_raw_test_data()
    if raw_data is None:
        raw_data = create_synthetic_data(len(X_test))
        print(f"Created synthetic data with {len(raw_data)} rows")
    
    # Feature calculator (using precomputed for demo)
    # We need to track global step separately since buffer is fixed size
    feature_step = {'count': 0}
    
    def feature_calculator(buffer_df):
        """
        Calculate features from buffer.
        
        In production, this should use compute_features_from_buffer
        with your actual feature engineering pipeline.
        """
        idx = feature_step['count']
        feature_step['count'] += 1
        
        if idx < len(X_test):
            return X_test[idx]
        return X_test[-1]
    
    # Calculate fee-aware trading thresholds
    # Get taker fee for typical trading volume (assume tier 1: 0-100k volume)
    taker_fee = ALPACA_FEE_TIERS[0][3]  # 0.0025 (0.25%)
    round_trip_fee = taker_fee * 2       # 0.005 (0.5%) - buy + sell
    
    # Set thresholds relative to break-even
    fee_multiplier = 1.2  # Require 20% margin above break-even
    buy_sell_threshold = round_trip_fee * fee_multiplier  # 0.6%
    close_threshold = round_trip_fee * 0.5  # Exit at half of break-even
    
    print(f"\nFee-aware trading thresholds:")
    print(f"  Taker fee: {taker_fee*100:.2f}% per side")
    print(f"  Round-trip: {round_trip_fee*100:.2f}%")
    print(f"  Buy/Sell threshold: {buy_sell_threshold*100:.2f}% ({fee_multiplier}x break-even)")
    print(f"  Close threshold: {close_threshold*100:.2f}%")
    
    # Choose trading rules (uncomment one)
    # Option 1: Conservative (high thresholds, tight risk management)
    # trading_rules = get_conservative_rules()
    
    # Option 2: Aggressive (low thresholds, wider stops)
    # trading_rules = get_aggressive_rules()
    
    # Option 3: Balanced (recommended for testing)
    # trading_rules = get_balanced_rules()
    
    # Option 4: Custom rules tuned for fee economics (RECOMMENDED)
    trading_rules = CompositeRule([
        ThresholdRule(
            buy_threshold=buy_sell_threshold,
            sell_threshold=buy_sell_threshold,
            close_threshold=close_threshold
        ),
        RiskManagementRule(
            stop_loss_pct=-0.02,     # -2% stop loss
            take_profit_pct=0.02     # +2% take profit
        )
    ], logic="AND")
    
    print(f"Using trading rules: {trading_rules.name}")
    
    # Create trading agent with model inside
    agent = TradingAgent(
        model=model,
        feature_calculator=feature_calculator,
        buffer_size=100,  # Match your feature window
        trading_rules=trading_rules,  # Pass trading rules
        target_idx=0,  # Trade first prediction target
        device=DEVICE,
        price_col='close'
    )
    
    print(f"Created TradingAgent with buffer_size={agent.buffer_size}")
    
    # Create simulation environment
    env = TradingSimulationEnv(
        data=raw_data,
        starting_balance=10000.0,  # Start with $10k
        use_taker_fee=True,  # Use taker fees (conservative)
        price_col='close'
    )
    
    print(f"Created SimulationEnv with {len(env.data)} timesteps")
    
    # Create backtester
    backtester = CryptoBacktester(
        agent=agent,
        env=env,
        tensorboard_dir=os.path.join(BASE_DIR, "../../runs/simulation"),
        log_every=100
    )
    
    # Run simulation
    print("\n" + "="*70)
    print("Running Trading Simulation with Alpaca Fee Structure")
    print("="*70)
    
    results = backtester.run(verbose=True)
    
    # Print detailed summary
    backtester.print_summary()
    
    # Save results
    results_dir = os.path.join(BASE_DIR, "../../results/simulation")
    os.makedirs(results_dir, exist_ok=True)
    backtester.save_results(os.path.join(results_dir, "simulation_results.parquet"))
    
    backtester.close()
    
    print(f"\n✓ Simulation complete!")
    print(f"✓ View results in TensorBoard:")
    print(f"  tensorboard --logdir={os.path.join(BASE_DIR, '../../runs')}")
    
    # Print key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print(f"Initial Capital:     ${results['starting_balance']:,.2f}")
    print(f"Final Equity:        ${results['final_equity']:,.2f}")
    print(f"Total Return:        {results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:        {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:        {results['max_drawdown']*100:.2f}%")
    print(f"Total Trades:        {results['num_trades']}")
    print(f"Total Fees:          ${results['total_fees']:,.2f}")
    print(f"Fee Impact:          {results['fees_pct']:.2f}% of starting capital")
    print(f"Avg Fee per Trade:   ${results['avg_fee_per_trade']:.2f}")
    
    # Profitability check
    if results['total_return_pct'] > 0:
        print(f"\n✓ Strategy is PROFITABLE after fees!")
    else:
        print(f"\n✗ Strategy is UNPROFITABLE - fees may be eating returns")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
