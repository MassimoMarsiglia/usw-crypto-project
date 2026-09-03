"""
crypto_backtester.py

Simplified backtesting system using TradingAgent and SimulationEnv.
All visualization through TensorBoard only.

Usage:
    from crypto_backtester import CryptoBacktester
    
    backtester = CryptoBacktester(
        agent=agent,
        env=env,
        tensorboard_dir="runs/backtest"
    )
    
    results = backtester.run()
    backtester.print_summary()
"""

import os
import json
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional

from trading_agent import TradingAgent
from simulation_env import TradingSimulationEnv


class CryptoBacktester:
    """
    Backtesting engine that runs TradingAgent in SimulationEnv.
    
    Logs all metrics to TensorBoard for visualization.
    """
    
    def __init__(
        self,
        agent: TradingAgent,
        env: TradingSimulationEnv,
        tensorboard_dir: str = None,
        log_every: int = 100,
    ):
        """
        Initialize backtester.
        
        Args:
            agent: TradingAgent instance
            env: TradingSimulationEnv instance
            tensorboard_dir: Directory for TensorBoard logs
            log_every: Log metrics every N steps
        """
        self.agent = agent
        self.env = env
        self.log_every = log_every
        
        # TensorBoard writer
        if tensorboard_dir is None:
            tensorboard_dir = "../../runs/backtest"
        self.writer = SummaryWriter(tensorboard_dir)
        
        # Results
        self.results: Optional[Dict[str, Any]] = None
        
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run backtest simulation.
        
        Args:
            verbose: Print progress updates
            
        Returns:
            Dictionary with comprehensive results
        """
        # Reset environment and agent
        obs = self.env.reset()
        self.agent.reset()
        
        step_count = 0
        predictions = []
        actions_taken = []
        
        # For rolling Sharpe and Information Ratio calculations
        equity_returns = []
        btc_returns = []
        last_equity = self.env.starting_balance
        last_btc_price = None
        
        if verbose:
            print(f"Starting backtest with {len(self.env.data)} timesteps...")
            print(f"Starting balance: ${self.env.starting_balance:,.2f}")
        
        # Run simulation loop
        while not self.env.done:
            # Agent needs to build buffer first
            if obs is not None:
                self.agent.add_data(obs)
            
            # Decide action
            action_info = self.agent.decide_action()
            
            if action_info['ready']:
                predictions.append(action_info['prediction'])
                actions_taken.append(action_info['action'])
                
                # Execute in environment
                action = {'target_position': action_info['target_position']}
                obs, done, info = self.env.step(action)
                
                # Update agent's position
                self.agent.update_position(info['position'])
                
                # Calculate returns for Sharpe and Information Ratio
                # Only calculate when equity/price actually changes (not on every step)
                current_equity = info['equity']
                current_price = info['price']
                
                if last_equity > 0 and current_equity != last_equity:
                    equity_return = (current_equity - last_equity) / last_equity
                    equity_returns.append(equity_return)
                    last_equity = current_equity
                
                if last_btc_price is not None and last_btc_price > 0 and current_price > 0 and current_price != last_btc_price:
                    btc_return = (current_price - last_btc_price) / last_btc_price
                    btc_returns.append(btc_return)
                
                if current_price > 0:
                    last_btc_price = current_price
                
                # Log to TensorBoard
                if step_count % self.log_every == 0:
                    self.writer.add_scalar('Simulation/Equity', info['equity'], step_count)
                    self.writer.add_scalar('Simulation/Position', info['position'], step_count)
                    self.writer.add_scalar('Simulation/Prediction', action_info['prediction'], step_count)
                    self.writer.add_scalar('Simulation/EquityChange', info['equity_change'], step_count)
                    self.writer.add_scalar('Simulation/TotalFees', info['total_fees'], step_count)
                    self.writer.add_scalar('Simulation/NumTrades', info['total_trades'], step_count)
                    
                    if info['price'] > 0:
                        self.writer.add_scalar('Simulation/Price', info['price'], step_count)
                    
                    # Log P&L if in position
                    if 'pnl_pct' in action_info:
                        self.writer.add_scalar('Simulation/PnL_Pct', action_info['pnl_pct'], step_count)
                    
                    # Calculate and log rolling Sharpe Ratio
                    # Use rolling window of last 5000 periods (~3.6 days at 15min intervals)
                    window_size = min(5000, len(equity_returns))
                    if len(equity_returns) >= 100:  # Need minimum data
                        window_returns = equity_returns[-window_size:]
                        if len(window_returns) > 1:
                            mean_return = np.mean(window_returns)
                            std_return = np.std(window_returns, ddof=1)  # Use sample std
                            if std_return > 1e-8:  # Avoid division by tiny numbers
                                # Annualize assuming 96 periods per day, 252 trading days
                                # But since we're using actual returns (not daily), just use sqrt(N returns per year)
                                # If we have returns every 15min, that's 96*252 = 24,192 per year
                                # But we only record when equity changes, so scale appropriately
                                avg_periods_per_year = 252 * 96  # Theoretical max
                                rolling_sharpe = (mean_return / std_return) * np.sqrt(avg_periods_per_year)
                                # Clip to reasonable range
                                rolling_sharpe = np.clip(rolling_sharpe, -20, 20)
                                self.writer.add_scalar('RollingMetrics/SharpeRatio', rolling_sharpe, step_count)
                                self.writer.add_scalar('RollingMetrics/MeanReturn', mean_return * 100, step_count)  # in %
                                self.writer.add_scalar('RollingMetrics/StdReturn', std_return * 100, step_count)  # in %
                    
                    # Calculate and log Information Ratio (excess return vs BTC benchmark)
                    if len(equity_returns) >= 100 and len(btc_returns) >= 100:
                        # Use same window size for both
                        window_size_ir = min(5000, len(equity_returns), len(btc_returns))
                        portfolio_rets = equity_returns[-window_size_ir:]
                        benchmark_rets = btc_returns[-window_size_ir:]
                        
                        # Align the arrays (may have different lengths due to price changes)
                        min_len = min(len(portfolio_rets), len(benchmark_rets))
                        portfolio_rets = portfolio_rets[-min_len:]
                        benchmark_rets = benchmark_rets[-min_len:]
                        
                        if len(portfolio_rets) > 1:
                            excess_returns = [p - b for p, b in zip(portfolio_rets, benchmark_rets)]
                            mean_excess = np.mean(excess_returns)
                            tracking_error = np.std(excess_returns, ddof=1)
                            if tracking_error > 1e-8:
                                # Annualize
                                avg_periods_per_year = 252 * 96
                                information_ratio = (mean_excess / tracking_error) * np.sqrt(avg_periods_per_year)
                                information_ratio = np.clip(information_ratio, -20, 20)
                                self.writer.add_scalar('RollingMetrics/InformationRatio', information_ratio, step_count)
                                self.writer.add_scalar('RollingMetrics/ExcessReturn', mean_excess * 100, step_count)  # in %
                                self.writer.add_scalar('RollingMetrics/TrackingError', tracking_error * np.sqrt(avg_periods_per_year) * 100, step_count)
                
                if verbose and step_count % 1000 == 0:
                    print(f"Step {step_count} | {self.env.render()}")
                
                step_count += 1
            else:
                # Buffer not ready, just step environment
                action = {'target_position': 0.0}
                obs, done, info = self.env.step(action)
        
        # Get final metrics
        metrics = self.env.get_metrics()
        
        # Log final metrics to TensorBoard
        self.writer.add_scalar('Final/TotalReturn', metrics['total_return_pct'], 0)
        self.writer.add_scalar('Final/SharpeRatio', metrics['sharpe_ratio'], 0)
        self.writer.add_scalar('Final/MaxDrawdown', metrics['max_drawdown'], 0)
        self.writer.add_scalar('Final/WinRate', metrics['win_rate'], 0)
        self.writer.add_scalar('Final/NumTrades', metrics['num_trades'], 0)
        self.writer.add_scalar('Final/TotalFees', metrics['total_fees'], 0)
        
        # Create equity curve plot in TensorBoard
        equity_curve = np.array(self.env.equity_history)
        for i, equity in enumerate(equity_curve):
            self.writer.add_scalar('EquityCurve/Value', equity, i)
        
        # Create position plot in TensorBoard
        positions = np.array(self.env.position_history)
        for i, pos in enumerate(positions):
            self.writer.add_scalar('Position/Value', pos, i)
        
        # Compile results
        self.results = {
            **metrics,
            'starting_balance': self.env.starting_balance,
            'predictions': np.array(predictions),
            'actions': actions_taken,
            'equity_curve': equity_curve,
            'position_history': positions,
            'trade_history': self.env.trade_history,
            'steps': step_count,
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print("Backtest Complete!")
            print(f"{'='*60}")
        
        self.writer.flush()
        return self.results
    
    def print_summary(self):
        """Print comprehensive summary of backtest results."""
        if self.results is None:
            print("No results yet. Call run() first.")
            return
        
        r = self.results
        
        print("\n" + "="*60)
        print("TRADING SIMULATION RESULTS")
        print("="*60)
        print(f"Starting Balance:    ${r['starting_balance']:,.2f}")
        print(f"Final Equity:        ${r['final_equity']:,.2f}")
        print(f"Total Return:        {r['total_return_pct']:+.2f}%")
        print(f"Sharpe Ratio:        {r['sharpe_ratio']:.3f}")
        print(f"Max Drawdown:        {r['max_drawdown']*100:.2f}%")
        print(f"Win Rate:            {r['win_rate']*100:.2f}%")
        print(f"\nTrading Activity:")
        print(f"  Total Steps:       {r['steps']}")
        print(f"  Total Trades:      {r['num_trades']}")
        print(f"  Total Fees:        ${r['total_fees']:,.2f} ({r['fees_pct']:.2f}% of capital)")
        print(f"  Avg Fee/Trade:     ${r['avg_fee_per_trade']:.2f}")
        
        # Profitability assessment
        print(f"\n{'='*60}")
        if r['total_return_pct'] > 0:
            print("✓ Strategy is PROFITABLE after fees")
        else:
            print("✗ Strategy is UNPROFITABLE - consider adjusting parameters")
        print(f"{'='*60}\n")
    
    def save_results(self, path: str):
        """
        Save results to disk.
        
        Args:
            path: Path to save results (parquet file)
        """
        if self.results is None:
            raise RuntimeError("No results to save. Run backtest first.")
        
        out_dir = os.path.dirname(path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        r = self.results
        
        # Create DataFrame
        df = pd.DataFrame({
            'equity': r['equity_curve'],
            'position': r['position_history'],
        })
        
        # Add predictions if available
        if len(r['predictions']) > 0:
            # Align predictions with equity curve
            pred_series = pd.Series(r['predictions'])
            df['prediction'] = pred_series.reindex(df.index, fill_value=np.nan)
        
        df.to_parquet(path)
        
        # Save metadata
        meta = {
            'starting_balance': float(r['starting_balance']),
            'final_equity': float(r['final_equity']),
            'total_return_pct': float(r['total_return_pct']),
            'sharpe_ratio': float(r['sharpe_ratio']),
            'max_drawdown': float(r['max_drawdown']),
            'win_rate': float(r['win_rate']),
            'num_trades': int(r['num_trades']),
            'total_fees': float(r['total_fees']),
            'fees_pct': float(r['fees_pct']),
            'avg_fee_per_trade': float(r['avg_fee_per_trade']),
            'steps': int(r['steps']),
        }
        
        meta_path = path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        # Save trade log
        if r['trade_history']:
            trade_log_path = path.replace('.parquet', '_trades.json')
            with open(trade_log_path, 'w') as f:
                json.dump(r['trade_history'], f, indent=2)
            print(f"Saved trade log to {trade_log_path}")
        
        print(f"Saved results to {path} and {meta_path}")
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


if __name__ == "__main__":
    """
    Example usage of the backtesting system.
    """
    import torch
    from trading_agent import CryptoMLP, TradingAgent
    from feat_generator import compute_features_from_buffer
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "../../models/07_01_feed_forward.pt")
    POST_SPLIT_DIR = os.path.join(BASE_DIR, "../../data/post_split_data")
    X_test_path = os.path.join(POST_SPLIT_DIR, "X_test.parquet")
    y_test_path = os.path.join(POST_SPLIT_DIR, "y_test.parquet")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Please train a model first.")
        exit(1)
    
    # Load test data
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print("Test data not found. Please run data preparation first.")
        exit(1)
    
    X_test = pd.read_parquet(X_test_path).to_numpy(dtype=np.float32)
    y_test = pd.read_parquet(y_test_path).to_numpy(dtype=np.float32)
    
    print(f"Loaded test data: X={X_test.shape}, y={y_test.shape}")
    
    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = CryptoMLP(
        input_dim=X_test.shape[1],
        output_dim=y_test.shape[1],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    
    print(f"Loaded model from {MODEL_PATH}")
    
    # Create synthetic market data for demonstration
    n = len(X_test)
    market_data = pd.DataFrame({
        'open': np.random.randn(n).cumsum() + 50000,
        'high': np.random.randn(n).cumsum() + 50100,
        'low': np.random.randn(n).cumsum() + 49900,
        'close': np.random.randn(n).cumsum() + 50000,
        'volume': np.random.rand(n) * 1000,
        'vwap': np.random.randn(n).cumsum() + 50000,
        'trade_count': np.random.randint(100, 1000, n)
    })
    
    # Feature calculator (using precomputed features for demo)
    def feature_calculator(buffer_df):
        idx = len(buffer_df) - 1
        if idx < len(X_test):
            return X_test[idx]
        return X_test[-1]
    
    # Create trading agent
    agent = TradingAgent(
        model=model,
        feature_calculator=feature_calculator,
        buffer_size=100,
        prediction_threshold=0.0,
        target_idx=0,
        device=DEVICE
    )
    
    # Create simulation environment
    env = TradingSimulationEnv(
        data=market_data,
        starting_balance=10000.0,
        use_taker_fee=True,
        price_col='close'
    )
    
    # Create backtester
    backtester = CryptoBacktester(
        agent=agent,
        env=env,
        tensorboard_dir=os.path.join(BASE_DIR, "../../runs/backtest"),
        log_every=100
    )
    
    # Run backtest
    print("\n" + "="*60)
    print("STARTING BACKTEST")
    print("="*60)
    
    results = backtester.run(verbose=True)
    backtester.print_summary()
    
    # Save results
    results_dir = os.path.join(BASE_DIR, "../../results/backtest")
    os.makedirs(results_dir, exist_ok=True)
    backtester.save_results(os.path.join(results_dir, "backtest_results.parquet"))
    
    backtester.close()
    
    print(f"\n✓ View results in TensorBoard:")
    print(f"  tensorboard --logdir={os.path.join(BASE_DIR, '../../runs')}")
