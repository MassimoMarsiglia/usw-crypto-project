"""
multi_asset_env.py

Multi-asset portfolio simulation environment.
Manages positions across multiple cryptocurrencies with portfolio-level risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict

from simulation_env import ALPACA_FEE_TIERS


class MultiAssetSimulationEnv:
    """
    Multi-asset portfolio simulation environment.
    
    Manages:
    - Multiple positions across different assets
    - Portfolio-level balance and equity
    - Position sizing with diversification
    - Fees per asset
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],  # e.g., ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT"]
        starting_balance: float = 10000.0,
        use_taker_fee: bool = True,
        max_position_pct: float = 0.4,  # Max 40% per asset
    ):
        """
        Initialize multi-asset environment.
        
        Args:
            data: DataFrame with OHLCV columns for each asset (e.g., BTCUSDT_close, ETHUSDT_close)
            assets: List of asset names
            starting_balance: Initial capital
            use_taker_fee: Use taker fees (True) or maker fees (False)
            max_position_pct: Maximum position size per asset as % of portfolio
        """
        self.data = data.reset_index(drop=True)
        self.assets = assets
        self.starting_balance = starting_balance
        self.use_taker_fee = use_taker_fee
        self.max_position_pct = max_position_pct
        
        # State variables
        self.current_step = 0
        self.balance = starting_balance  # Cash balance
        
        # Positions per asset: {asset: {'position': -1/0/1, 'shares': float, 'entry_price': float}}
        self.positions = {asset: {'position': 0.0, 'shares': 0.0, 'entry_price': 0.0} for asset in assets}
        
        # Tracking
        self.total_trades = 0
        self.total_fees = 0.0
        self.trade_history = []
        self.equity_history = [starting_balance]
        self.portfolio_weights_history = []
        
        # Episode tracking
        self.done = False
        
    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = self.starting_balance
        self.positions = {asset: {'position': 0.0, 'shares': 0.0, 'entry_price': 0.0} for asset in self.assets}
        self.total_trades = 0
        self.total_fees = 0.0
        self.trade_history = []
        self.equity_history = [self.starting_balance]
        self.portfolio_weights_history = []
        self.done = False
        
        return self._get_obs()
    
    def _get_obs(self) -> Optional[Dict[str, Any]]:
        """Get current observation for all assets."""
        if self.current_step >= len(self.data):
            return None
        
        obs = {}
        for asset in self.assets:
            # Get price data for this asset
            for col in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']:
                col_name = f"{asset}_{col}"
                if col_name in self.data.columns:
                    obs[col_name] = self.data.iloc[self.current_step][col_name]
        
        return obs
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current price for each asset."""
        if self.current_step >= len(self.data):
            return {asset: 0.0 for asset in self.assets}
        
        prices = {}
        for asset in self.assets:
            col_name = f"{asset}_close"
            if col_name in self.data.columns:
                prices[asset] = self.data.iloc[self.current_step][col_name]
            else:
                prices[asset] = 0.0
        
        return prices
    
    def _get_alpaca_fee(self, volume_30d: float) -> float:
        """Get appropriate Alpaca fee based on 30-day volume."""
        for min_vol, max_vol, maker_fee, taker_fee in ALPACA_FEE_TIERS:
            if min_vol <= volume_30d < max_vol:
                return taker_fee if self.use_taker_fee else maker_fee
        return 0.0010  # Default to highest tier taker fee
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + all positions)."""
        current_prices = self._get_current_prices()
        
        total_position_value = 0.0
        for asset in self.assets:
            shares = self.positions[asset]['shares']
            price = current_prices.get(asset, 0.0)
            total_position_value += shares * price
        
        return self.balance + total_position_value
    
    def get_position_values(self) -> Dict[str, float]:
        """Get current value of each position."""
        current_prices = self._get_current_prices()
        
        position_values = {}
        for asset in self.assets:
            shares = self.positions[asset]['shares']
            price = current_prices.get(asset, 0.0)
            position_values[asset] = shares * price
        
        return position_values
    
    def step(self, actions: Dict[str, float]) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """
        Execute one step with actions for multiple assets.
        
        Args:
            actions: Dictionary mapping asset -> target_position (-1, 0, or 1)
            
        Returns:
            observation, done, info
        """
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")
        
        portfolio_value_before = self.get_portfolio_value()
        
        # Execute trades for each asset
        trades_executed = {}
        for asset, target_position in actions.items():
            if asset in self.assets:
                trade_info = self._execute_trade(asset, target_position)
                trades_executed[asset] = trade_info
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.data):
            self.done = True
            # Close all open positions
            for asset in self.assets:
                if self.positions[asset]['position'] != 0:
                    self._execute_trade(asset, 0.0)
        
        # Track equity and weights
        new_equity = self.get_portfolio_value()
        self.equity_history.append(new_equity)
        
        # Calculate portfolio weights
        position_values = self.get_position_values()
        weights = {asset: abs(val) / new_equity if new_equity > 0 else 0.0 
                  for asset, val in position_values.items()}
        self.portfolio_weights_history.append(weights)
        
        equity_change = new_equity - portfolio_value_before
        
        # Get next observation
        obs = self._get_obs()
        
        # Info dictionary
        info = {
            'equity': new_equity,
            'equity_change': equity_change,
            'balance': self.balance,
            'positions': {asset: pos['position'] for asset, pos in self.positions.items()},
            'position_values': position_values,
            'portfolio_weights': weights,
            'prices': self._get_current_prices(),
            'total_trades': self.total_trades,
            'total_fees': self.total_fees,
            'trades_executed': trades_executed,
        }
        
        return obs, self.done, info
    
    def _execute_trade(self, asset: str, target_position: float) -> Dict[str, Any]:
        """
        Execute trade for a specific asset.
        
        Args:
            asset: Asset name (e.g., "BTCUSDT")
            target_position: Target position (-1, 0, or 1)
            
        Returns:
            Trade execution details
        """
        current_prices = self._get_current_prices()
        current_price = current_prices.get(asset, 0.0)
        
        if current_price <= 0:
            return {"executed": False, "reason": "invalid_price", "asset": asset}
        
        current_position = self.positions[asset]['position']
        
        # Check if position change needed (directional only - no rebalancing)
        # This is a signal-based strategy, not a weight-maintenance strategy
        if target_position == current_position:
            return {"executed": False, "reason": "no_change", "asset": asset}
        
        # Calculate fee
        actual_fee = self._get_alpaca_fee(self.total_fees * 30)
        
        trade_info = {
            "step": self.current_step,
            "asset": asset,
            "price": current_price,
            "old_position": current_position,
            "new_position": target_position,
            "executed": True,
        }
        
        # Close existing position first
        if current_position != 0:
            shares = self.positions[asset]['shares']
            entry_price = self.positions[asset]['entry_price']
            
            # Sell the position and receive proceeds (minus fees)
            close_value = abs(shares * current_price)
            close_fee = close_value * actual_fee
            proceeds = close_value - close_fee
            
            # Calculate realized P&L for tracking
            pnl = shares * (current_price - entry_price)
            
            # Add proceeds to balance
            self.balance += proceeds
            self.total_fees += close_fee
            self.total_trades += 1
            
            trade_info["close_pnl"] = pnl
            trade_info["close_fee"] = close_fee
            trade_info["close_proceeds"] = proceeds
            
            # Reset position
            self.positions[asset]['shares'] = 0.0
            self.positions[asset]['entry_price'] = 0.0
        
        # Open new position
        if target_position != 0:
            # Check available balance first
            available_cash = self.balance * 0.99
            
            # Check minimum trade size
            min_trade_size = 100.0
            if available_cash < min_trade_size:
                trade_info["executed"] = False
                trade_info["reason"] = f"insufficient_balance (${available_cash:.2f} < ${min_trade_size})"
                return trade_info
            
            # Calculate maximum position size with proper constraints:
            # 1. Can't exceed max_position_pct of STARTING balance (prevents individual asset over-concentration)
            # 2. Can't exceed available cash (prevents overdraft)
            # 3. Total positions across all assets should not exceed starting balance × reasonable multiplier
            #
            # With max_position_pct=0.4 and 4 assets, theoretical max is 160% of starting balance,
            # but cash constraint naturally limits this to ~100% at opening.
            # As positions gain value, total exposure can drift higher - this is acceptable as it
            # represents unrealized gains, not additional capital deployment.
            max_position_value = self.starting_balance * self.max_position_pct
            available = min(available_cash, max_position_value)
            
            # Log under-allocation for monitoring (not an error, just FYI)
            if available < max_position_value * 0.8 and available_cash >= min_trade_size:
                trade_info["note"] = f"reduced_size: ${available:.2f} (target ${max_position_value:.2f}, limited by cash)"
            
            # Calculate trade value and fee
            trade_value = available
            open_fee = trade_value * actual_fee / (1 + actual_fee)
            
            # Net value after fee goes into position
            net_value = available - open_fee
            shares = (net_value / current_price) * target_position
            
            # Update balance based on position type
            if target_position > 0:
                # LONG: Pay cash to buy shares (deduct from balance)
                self.balance -= available
            else:
                # SHORT: Receive cash from selling shares (add to balance)
                self.balance += net_value  # Add proceeds after fee
            
            self.positions[asset]['entry_price'] = current_price
            self.positions[asset]['shares'] = shares
            self.total_fees += open_fee
            self.total_trades += 1
            
            open_value = abs(net_value)
            
            trade_info["open_value"] = open_value
            trade_info["open_fee"] = open_fee
            trade_info["shares"] = shares
        
        self.positions[asset]['position'] = target_position
        trade_info["balance"] = self.balance
        trade_info["equity"] = self.get_portfolio_value()
        
        self.trade_history.append(trade_info)
        return trade_info
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        equity_curve = np.array(self.equity_history)
        
        # Returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[np.isfinite(returns)]
        
        # Total return
        total_return = (equity_curve[-1] - self.starting_balance) / self.starting_balance
        
        # Sharpe ratio (annualized, 96 periods per day * 252 trading days)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(96 * 252)
        else:
            sharpe = 0.0
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = [t for t in self.trade_history 
                         if t.get('close_pnl', 0) > 0]
        total_closing_trades = [t for t in self.trade_history 
                               if 'close_pnl' in t]
        win_rate = len(winning_trades) / len(total_closing_trades) if total_closing_trades else 0.0
        
        # Fee analysis
        fees_pct = (self.total_fees / self.starting_balance) * 100
        avg_fee = self.total_fees / self.total_trades if self.total_trades > 0 else 0.0
        
        return {
            'final_equity': equity_curve[-1],
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': self.total_trades,
            'total_fees': self.total_fees,
            'fees_pct': fees_pct,
            'avg_fee_per_trade': avg_fee,
        }
    
    def render(self) -> str:
        """Render current state."""
        equity = self.get_portfolio_value()
        weights = self.portfolio_weights_history[-1] if self.portfolio_weights_history else {}
        
        position_str = ", ".join([f"{asset}: {pos['position']:+.0f}" 
                                 for asset, pos in self.positions.items()])
        weights_str = ", ".join([f"{asset}: {w*100:.1f}%" 
                                for asset, w in weights.items() if w > 0.01])
        
        # Add shares held for non-zero positions
        shares_str = ", ".join([f"{asset}: {abs(pos['shares']):.4f}" 
                               for asset, pos in self.positions.items() if pos['shares'] != 0])
        shares_display = f" | Stakes: [{shares_str}]" if shares_str else ""
        
        return (f"Step: {self.current_step} | Equity: ${equity:,.2f} | "
                f"Positions: [{position_str}] | Weights: [{weights_str}]{shares_display} | "
                f"Trades: {self.total_trades} | Fees: ${self.total_fees:.2f}")
