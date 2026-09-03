"""
run_simulation_multi_lstm.py

Run multi-asset trading simulation with LSTM model using sequence data.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crypto_backtester import CryptoBacktester
from multi_asset_agent import MultiAssetTradingAgent
from multi_asset_env import MultiAssetSimulationEnv
from trading_rules import (
    ThresholdRule,
    RiskManagementRule,
    CompositeRule,
    DiversificationRule
)
from simulation_env import ALPACA_FEE_TIERS
from deployment_feat_generator import create_deployment_feature_calculator

# Define LSTM model class
class CryptoLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Layer normalization for input features
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Layer normalization after LSTM
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # Normalize input features
        x = self.input_norm(x)
        
        # LSTM initializes hidden states to zero by default if not provided
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]  # take last timestep
        out = self.output_norm(out)  # normalize before final layer
        out = self.fc(out)
        return out


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../models/07_02_LSTM.pt")
POST_SPLIT_DIR = os.path.join(BASE_DIR, "../../data/post_split_data")
SPLIT_DATA_DIR = os.path.join(BASE_DIR, "../../data/split_data")
X_test_path = os.path.join(POST_SPLIT_DIR, "X_test.parquet")
y_test_path = os.path.join(POST_SPLIT_DIR, "y_test.parquet")
TEST_BACKTEST_MULTI_FILE = os.path.join(SPLIT_DATA_DIR, "test_backtest_multi.parquet")


def main():
    """Run the multi-asset LSTM trading simulation."""
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Define assets to trade
    ASSETS = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT"]
    
    # Map assets to model output indices (4h and 1d)
    TARGET_INDICES = {
        "BTCUSDT": [0, 1],  # BTC 4h and 1d targets
        "ETHUSDT": [2, 3],  # ETH 4h and 1d targets
        "LTCUSDT": [4, 5],  # LTC 4h and 1d targets
        "SOLUSDT": [6, 7],  # SOL 4h and 1d targets
    }
    
    # LSTM sequence length
    SEQ_LEN = 16  # 16 x 15min = 4 hours
    
    # Calculate expected dimensions based on feature structure
    # Each asset has 19 features per timeframe (4 timeframes)
    # With 7 total assets (3 BTC pairs + 4 USDT pairs) in training
    # BUT we're only trading 4 USDT pairs, so model expects all 7 × 4 × 19
    num_training_pairs = 7  # 3 BTC pairs + 4 USDT pairs used in training
    num_timeframes = 4  # 15m, 1h, 4h, 1d
    features_per_pair_tf = 19  # 19 features per pair-timeframe combination
    input_dim = num_training_pairs * num_timeframes * features_per_pair_tf  # 7 × 4 × 19 = 532
    
    # Output: 4 USDT pairs × 2 horizons (4h, 1d)
    output_dim = len(ASSETS) * 2  # 4 × 2 = 8
    
    print(f"Model architecture:")
    print(f"  Input dimension: {input_dim} features")
    print(f"  Output dimension: {output_dim} targets (4 assets × 2 horizons)")
    print(f"  Sequence length: {SEQ_LEN}")
    
    # Load LSTM model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: LSTM model not found at {MODEL_PATH}")
        return
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    model = CryptoLSTM(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=output_dim,
        num_layers=2,
        dropout=0.2
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded LSTM model from {MODEL_PATH}")
    print(f"Model dimensions: input_dim={input_dim}, output_dim={output_dim}, seq_len={SEQ_LEN}")
    
    # Load multi-asset raw market data
    if not os.path.exists(TEST_BACKTEST_MULTI_FILE):
        print(f"Error: Multi-asset backtest data not found at {TEST_BACKTEST_MULTI_FILE}")
        print("Please run scripts/04_split_data/split_dataset_backtest.py first")
        return
    
    raw_data = pd.read_parquet(TEST_BACKTEST_MULTI_FILE)
    print(f"Loaded multi-asset data: {raw_data.shape}")
    print(f"Assets: {ASSETS}")
    print(f"Date range: {raw_data.index.min()} to {raw_data.index.max()}")
    
    # Create deployment-ready feature calculator
    # This EXACTLY matches the training data preparation pipeline
    base_feature_calculator = create_deployment_feature_calculator(pairs=ASSETS)
    
    # Sequence buffer for LSTM
    sequence_buffer = []
    
    def feature_calculator(buffer_df):
        """
        Calculate features on-the-fly and maintain sequence for LSTM.
        
        For LSTM, we compute features from raw buffer using the EXACT
        training pipeline, then build a sequence of SEQ_LEN feature vectors.
        """
        # Compute current features from raw OHLCV buffer
        current_features = base_feature_calculator(buffer_df)
        
        if current_features is None:
            return None
        
        # Add to sequence buffer
        sequence_buffer.append(current_features)
        
        # Keep only last SEQ_LEN features
        if len(sequence_buffer) > SEQ_LEN:
            sequence_buffer.pop(0)
        
        # Return sequence when ready
        if len(sequence_buffer) == SEQ_LEN:
            return np.array(sequence_buffer, dtype=np.float32)  # Shape: (SEQ_LEN, num_features)
        
        return None
    
    print(f"\nDeployment feature calculator initialized:")
    print(f"  Pairs: {ASSETS}")
    print(f"  Timeframes: 15m, 1h, 4h, 1d (resampled from 15m buffer)")
    print(f"  Features per asset-timeframe: 19")
    print(f"  Total features per timestep: {len(ASSETS)} × 4 × 19 = {len(ASSETS) * 4 * 19}")
    print(f"  LSTM sequence length: {SEQ_LEN} timesteps")
    print(f"  Computing on-the-fly from raw OHLCV buffer")
    print(f"  ⚠️  This will be slower but matches training pipeline exactly!")
    
    # Calculate fee-aware trading thresholds
    taker_fee = ALPACA_FEE_TIERS[0][3]  # 0.0025 (0.25%)
    round_trip_fee = taker_fee * 2       # 0.005 (0.5%)
    
    # Set thresholds relative to break-even
    fee_multiplier = 1.2  # Require 20% margin above break-even
    buy_sell_threshold = round_trip_fee * fee_multiplier
    close_threshold = round_trip_fee * 0.5
    
    print(f"\nFee-aware trading thresholds:")
    print(f"  Taker fee: {taker_fee*100:.2f}% per side")
    print(f"  Round-trip: {round_trip_fee*100:.2f}%")
    print(f"  Buy/Sell threshold: {buy_sell_threshold*100:.2f}% ({fee_multiplier}x break-even)")
    print(f"  Close threshold: {close_threshold*100:.2f}%")
    
    # Create trading rules with diversification
    trading_rules = CompositeRule([
        ThresholdRule(
            buy_threshold=buy_sell_threshold,
            sell_threshold=buy_sell_threshold,
            close_threshold=close_threshold
        ),
        RiskManagementRule(
            stop_loss_pct=-0.02,
            take_profit_pct=0.02
        ),
        DiversificationRule(
            max_position_pct=0.4,  # Max 40% per asset
            prediction_threshold=buy_sell_threshold
        )
    ], logic="AND")
    
    print(f"Using trading rules: {trading_rules.name}")
    print(f"Max position per asset: 40% (diversification enforced)")
    
    # Create multi-asset trading agent
    # Buffer size must be large enough for multi-timeframe resampling:
    # - 1 day = 96 x 15-minute bars
    # - Need extra for indicators (EMA50, etc.)
    # - 400 bars = ~4 days of data
    agent = MultiAssetTradingAgent(
        model=model,
        feature_calculator=feature_calculator,
        assets=ASSETS,
        target_indices=TARGET_INDICES,
        buffer_size=400,
        trading_rules=trading_rules,
        device=DEVICE
    )
    
    print(f"Created MultiAssetTradingAgent with {len(ASSETS)} assets")
    print(f"LSTM sequence length: {SEQ_LEN} timesteps")
    
    # Create multi-asset simulation environment
    env = MultiAssetSimulationEnv(
        data=raw_data,
        assets=ASSETS,
        starting_balance=10000.0,
        use_taker_fee=True,
        max_position_pct=0.4,  # Max 40% per asset
    )
    
    print(f"Created MultiAssetSimulationEnv with {len(raw_data)} timesteps")
    
    # Run backtest
    print(f"\n{'='*70}")
    print("Running Multi-Asset LSTM Trading Simulation with Alpaca Fee Structure")
    print(f"{'='*70}")
    
    # Setup TensorBoard
    tensorboard_dir = os.path.join(BASE_DIR, "../../runs/eval_lstm_multi")
    writer = SummaryWriter(tensorboard_dir)
    print(f"TensorBoard logging to: {tensorboard_dir}")
    
    # Manual backtesting loop for multi-asset
    obs = env.reset()
    agent.reset()
    
    step_count = 0
    log_every = 100
    
    # For rolling metrics
    equity_returns = []
    btc_returns = []
    last_equity = env.starting_balance
    last_btc_price = None
    
    # Open trade log file
    trade_log_file = os.path.join(BASE_DIR, "../../results/lstm/detailed_trade_log.txt")
    os.makedirs(os.path.dirname(trade_log_file), exist_ok=True)
    trade_log = open(trade_log_file, 'w')
    trade_log.write("="*100 + "\n")
    trade_log.write("DETAILED TRADE LOG - Multi-Asset LSTM Simulation\n")
    trade_log.write("="*100 + "\n\n")
    trade_log.write(f"Starting Balance: ${env.starting_balance:,.2f}\n")
    trade_log.write(f"Max Position per Asset: {env.max_position_pct*100:.0f}% = ${env.starting_balance * env.max_position_pct:,.2f}\n")
    trade_log.write(f"LSTM Sequence Length: {SEQ_LEN} timesteps (4 hours)\n\n")
    
    # Track individual trade numbers (unique sequential ID)
    trade_counter = 0
    
    print(f"Starting backtest with {len(env.data)} timesteps...")
    print(f"Starting balance: ${env.starting_balance:,.2f}")
    print(f"Note: First {SEQ_LEN} steps will build sequence buffer (no trades)")
    print(f"Logging detailed trades to: {trade_log_file}")
    
    while not env.done:
        if obs is not None:
            agent.add_data(obs)
        
        if agent.is_ready():
            # Get portfolio state
            portfolio_value = env.get_portfolio_value()
            position_values = env.get_position_values()
            
            # Decide actions for all assets
            actions = agent.decide_actions(portfolio_value, position_values)
            
            # Execute in environment
            obs, done, info = env.step(actions)
            
            # Log any trades that executed
            if 'trades_executed' in info:
                for asset, trade_info in info['trades_executed'].items():
                    if trade_info.get('executed', False):
                        trade_counter += 1
                        trade_log.write(f"\n{'='*100}\n")
                        trade_log.write(f"TRADE #{trade_counter} | Step {step_count} | {asset}\n")
                        trade_log.write(f"{'='*100}\n")
                        trade_log.write(f"Price: ${trade_info['price']:,.2f}\n")
                        trade_log.write(f"Position: {trade_info['old_position']:+.0f} → {trade_info['new_position']:+.0f}\n\n")
                        
                        # CLOSING existing position
                        if 'close_pnl' in trade_info:
                            trade_log.write(f"CLOSING POSITION:\n")
                            trade_log.write(f"  Shares held: {env.positions[asset]['shares']:.6f}\n")
                            trade_log.write(f"  Entry price: ${trade_info.get('entry_price', 0):,.2f}\n")
                            trade_log.write(f"  Exit price:  ${trade_info['price']:,.2f}\n")
                            trade_log.write(f"  Position value: ${trade_info.get('close_proceeds', 0) + trade_info.get('close_fee', 0):,.2f}\n")
                            trade_log.write(f"  Close fee: ${trade_info.get('close_fee', 0):.2f}\n")
                            trade_log.write(f"  Proceeds (after fee): ${trade_info.get('close_proceeds', 0):,.2f}\n")
                            trade_log.write(f"  Realized P&L: ${trade_info.get('close_pnl', 0):,.2f}\n\n")
                        
                        # OPENING new position
                        if 'open_value' in trade_info:
                            trade_log.write(f"OPENING POSITION:\n")
                            trade_log.write(f"  Available cash: ${info['balance'] + trade_info.get('open_value', 0) + trade_info.get('open_fee', 0):,.2f}\n")
                            trade_log.write(f"  Position size: ${trade_info.get('open_value', 0) + trade_info.get('open_fee', 0):,.2f}\n")
                            trade_log.write(f"  Open fee: ${trade_info.get('open_fee', 0):.2f}\n")
                            trade_log.write(f"  Net invested: ${trade_info.get('open_value', 0):,.2f}\n")
                            trade_log.write(f"  Shares acquired: {trade_info.get('shares', 0):.6f}\n")
                            trade_log.write(f"  Entry price: ${trade_info['price']:,.2f}\n\n")
                        
                        # BALANCE UPDATE
                        trade_log.write(f"ACCOUNT STATE:\n")
                        trade_log.write(f"  Balance: ${trade_info['balance']:,.2f}\n")
                        trade_log.write(f"  Equity: ${trade_info['equity']:,.2f}\n")
                        trade_log.write(f"  Total fees paid: ${info['total_fees']:,.2f}\n")
                        
                        # VALIDATION
                        portfolio_value = info['equity']
                        total_position_value = sum(info['position_values'].values())
                        balance = info['balance']
                        calculated_equity = balance + total_position_value
                        
                        trade_log.write(f"\nVALIDATION:\n")
                        trade_log.write(f"  Cash: ${balance:,.2f}\n")
                        trade_log.write(f"  Total positions: ${total_position_value:,.2f}\n")
                        trade_log.write(f"  Calculated equity: ${calculated_equity:,.2f}\n")
                        trade_log.write(f"  Reported equity: ${portfolio_value:,.2f}\n")
                        trade_log.write(f"  Difference: ${abs(calculated_equity - portfolio_value):.6f}\n")
                        
                        if abs(calculated_equity - portfolio_value) > 0.01:
                            trade_log.write(f"  ⚠️  WARNING: Equity mismatch > $0.01!\n")
                        
                        trade_log.flush()
            
            # Update agent's positions
            agent.update_positions(info['positions'])
            
            # Calculate returns for rolling metrics
            current_equity = info['equity']
            current_btc_price = info['prices'].get('BTCUSDT', 0)
            
            if last_equity > 0 and current_equity != last_equity:
                equity_return = (current_equity - last_equity) / last_equity
                equity_returns.append(equity_return)
                last_equity = current_equity
            
            if last_btc_price is not None and last_btc_price > 0 and current_btc_price > 0 and current_btc_price != last_btc_price:
                btc_return = (current_btc_price - last_btc_price) / last_btc_price
                btc_returns.append(btc_return)
            
            if current_btc_price > 0:
                last_btc_price = current_btc_price
            
            # Log to TensorBoard
            if step_count % log_every == 0:
                # Portfolio-level metrics
                writer.add_scalar('Simulation/Equity', info['equity'], step_count)
                writer.add_scalar('Simulation/Balance', info['balance'], step_count)
                writer.add_scalar('Simulation/EquityChange', info['equity_change'], step_count)
                writer.add_scalar('Simulation/TotalFees', info['total_fees'], step_count)
                writer.add_scalar('Simulation/NumTrades', info['total_trades'], step_count)
                
                # Per-asset positions
                for asset in ASSETS:
                    position = info['positions'].get(asset, 0)
                    position_value = info['position_values'].get(asset, 0)
                    weight = info['portfolio_weights'].get(asset, 0)
                    price = info['prices'].get(asset, 0)
                    
                    # Get actual shares held
                    shares = env.positions[asset]['shares']
                    
                    writer.add_scalar(f'Positions/{asset}', position, step_count)
                    writer.add_scalar(f'PositionValues/{asset}', position_value, step_count)
                    writer.add_scalar(f'Weights/{asset}', weight * 100, step_count)  # in %
                    writer.add_scalar(f'PositionStakes/{asset}', abs(shares), step_count)  # Absolute shares held
                    if price > 0:
                        writer.add_scalar(f'Prices/{asset}', price, step_count)
                
                # Log cash (unallocated capital)
                cash_weight = (info['balance'] / info['equity']) * 100 if info['equity'] > 0 else 0
                writer.add_scalar('Weights/CASH', cash_weight, step_count)
                writer.add_scalar('PositionStakes/CASH', info['balance'], step_count)  # Cash balance in USD
                
                # Rolling Sharpe Ratio
                window_size = min(5000, len(equity_returns))
                if len(equity_returns) >= 100:
                    window_returns = equity_returns[-window_size:]
                    if len(window_returns) > 1:
                        mean_return = np.mean(window_returns)
                        std_return = np.std(window_returns, ddof=1)
                        if std_return > 1e-8:
                            avg_periods_per_year = 252 * 96
                            rolling_sharpe = (mean_return / std_return) * np.sqrt(avg_periods_per_year)
                            rolling_sharpe = np.clip(rolling_sharpe, -20, 20)
                            writer.add_scalar('RollingMetrics/SharpeRatio', rolling_sharpe, step_count)
                            writer.add_scalar('RollingMetrics/MeanReturn', mean_return * 100, step_count)
                            writer.add_scalar('RollingMetrics/StdReturn', std_return * 100, step_count)
                
                # Information Ratio vs BTC
                if len(equity_returns) >= 100 and len(btc_returns) >= 100:
                    window_size_ir = min(5000, len(equity_returns), len(btc_returns))
                    portfolio_rets = equity_returns[-window_size_ir:]
                    benchmark_rets = btc_returns[-window_size_ir:]
                    
                    min_len = min(len(portfolio_rets), len(benchmark_rets))
                    portfolio_rets = portfolio_rets[-min_len:]
                    benchmark_rets = benchmark_rets[-min_len:]
                    
                    if len(portfolio_rets) > 1:
                        excess_returns = [p - b for p, b in zip(portfolio_rets, benchmark_rets)]
                        mean_excess = np.mean(excess_returns)
                        tracking_error = np.std(excess_returns, ddof=1)
                        if tracking_error > 1e-8:
                            avg_periods_per_year = 252 * 96
                            information_ratio = (mean_excess / tracking_error) * np.sqrt(avg_periods_per_year)
                            information_ratio = np.clip(information_ratio, -20, 20)
                            writer.add_scalar('RollingMetrics/InformationRatio', information_ratio, step_count)
                            writer.add_scalar('RollingMetrics/ExcessReturn', mean_excess * 100, step_count)
                            writer.add_scalar('RollingMetrics/TrackingError', tracking_error * np.sqrt(avg_periods_per_year) * 100, step_count)
            
            if step_count % 1000 == 0:
                # Enhanced debug output with shares
                shares_info = " | ".join([f"{asset}: {abs(env.positions[asset]['shares']):.4f}" 
                                         for asset in ASSETS if env.positions[asset]['shares'] != 0])
                print(f"Step {step_count} | {env.render()}")
                if shares_info:
                    print(f"  STAKES | {shares_info}")
                
                # Show total position exposure (includes unrealized gains)
                total_long_value = sum(env.positions[asset]['shares'] * info['prices'].get(asset, 0) 
                                      for asset in ASSETS if env.positions[asset]['shares'] > 0)
                if total_long_value > 0:
                    exposure_pct = (total_long_value / env.starting_balance) * 100
                    if step_count % 5000 == 0:  # Log every 5000 steps
                        print(f"  📊 Total Long Exposure: ${total_long_value:,.2f} ({exposure_pct:.0f}% of starting capital)")
            
            step_count += 1
        else:
            # Buffer not ready (building sequence), just step
            actions = {asset: 0.0 for asset in ASSETS}
            obs, done, info = env.step(actions)
    
    # Get final metrics
    metrics = env.get_metrics()
    
    # Log final metrics to TensorBoard
    writer.add_scalar('Final/TotalReturn', metrics['total_return_pct'], 0)
    writer.add_scalar('Final/SharpeRatio', metrics['sharpe_ratio'], 0)
    writer.add_scalar('Final/MaxDrawdown', metrics['max_drawdown'] * 100, 0)
    writer.add_scalar('Final/WinRate', metrics['win_rate'] * 100, 0)
    writer.add_scalar('Final/NumTrades', metrics['num_trades'], 0)
    writer.add_scalar('Final/TotalFees', metrics['total_fees'], 0)
    writer.add_scalar('Final/FeesPct', metrics['fees_pct'], 0)
    
    # Create equity curve plot in TensorBoard
    equity_curve = np.array(env.equity_history)
    for i, equity in enumerate(equity_curve):
        writer.add_scalar('EquityCurve/Value', equity, i)
    
    # Create portfolio weights over time
    if env.portfolio_weights_history:
        for i, weights_dict in enumerate(env.portfolio_weights_history):
            for asset, weight in weights_dict.items():
                writer.add_scalar(f'WeightsCurve/{asset}', weight * 100, i)
    
    writer.flush()
    writer.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Backtest Complete!")
    print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print("MULTI-ASSET LSTM TRADING SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Model Type:          LSTM (seq_len={SEQ_LEN})")
    print(f"Assets Traded:       {', '.join(ASSETS)}")
    print(f"Starting Balance:    ${metrics.get('starting_balance', env.starting_balance):,.2f}")
    print(f"Final Equity:        ${metrics['final_equity']:,.2f}")
    print(f"Total Return:        {metrics['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate:            {metrics['win_rate']*100:.2f}%")
    
    print(f"\nTrading Activity:")
    print(f"  Total Steps:       {step_count}")
    print(f"  Total Trades:      {metrics['num_trades']}")
    print(f"  Total Fees:        ${metrics['total_fees']:,.2f} ({metrics['fees_pct']:.2f}% of capital)")
    print(f"  Avg Fee/Trade:     ${metrics['avg_fee_per_trade']:.2f}")
    
    # Profitability assessment
    print(f"\n{'='*60}")
    if metrics['total_return_pct'] > 0:
        print("✓ Strategy is PROFITABLE after fees")
    else:
        print("✗ Strategy is UNPROFITABLE - consider adjusting parameters")
    print(f"{'='*60}")
    
    # Save results
    results_dir = os.path.join(BASE_DIR, "../../results/lstm")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save trade log
    import json
    trade_log_path = os.path.join(results_dir, "simulation_results_multi_trades.json")
    with open(trade_log_path, 'w') as f:
        json.dump(env.trade_history, f, indent=2)
    
    print(f"\nSaved trade log to {trade_log_path}")
    
    # Save equity curve
    equity_df = pd.DataFrame({
        'equity': env.equity_history,
    })
    equity_df.to_parquet(os.path.join(results_dir, "simulation_results_multi_equity.parquet"))
    
    print(f"Saved equity curve to {results_dir}")
    
    # Close trade log
    trade_log.write(f"\n\n{'='*100}\n")
    trade_log.write("SIMULATION COMPLETE\n")
    trade_log.write(f"{'='*100}\n")
    trade_log.write(f"Final Equity: ${metrics['final_equity']:,.2f}\n")
    trade_log.write(f"Total Return: {metrics['total_return_pct']:+.2f}%\n")
    trade_log.write(f"Total Trades: {metrics['num_trades']}\n")
    trade_log.write(f"Total Fees: ${metrics['total_fees']:,.2f}\n")
    trade_log.close()
    
    print(f"\n✓ Multi-asset LSTM simulation complete!")
    print(f"✓ Detailed trade log saved to: {trade_log_file}")
    print(f"✓ TensorBoard logs saved to: {tensorboard_dir}")
    print(f"  View with: tensorboard --logdir={tensorboard_dir}")


if __name__ == "__main__":
    main()
