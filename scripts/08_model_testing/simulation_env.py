"""
simulation_env.py

Gym-like simulation environment for backtesting trading strategies.
Manages portfolio state, executes trades with fees, and tracks performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from collections import deque


# Alpaca fee structure (maker/taker)
ALPACA_FEE_TIERS = [
    (0, 100_000, 0.0015, 0.0025),           # 0-100k
    (100_000, 500_000, 0.0012, 0.0022),     # 100k-500k
    (500_000, 1_000_000, 0.0010, 0.0020),   # 500k-1M
    (1_000_000, 10_000_000, 0.0008, 0.0018), # 1M-10M
    (10_000_000, 25_000_000, 0.0005, 0.0015), # 10M-25M
    (25_000_000, 50_000_000, 0.0002, 0.0013), # 25M-50M
    (50_000_000, 100_000_000, 0.0002, 0.0012), # 50M-100M
    (100_000_000, float('inf'), 0.0, 0.0010)  # 100M+
]


class TradingSimulationEnv:
    """
    Gym-like environment for trading simulation.
    
    Manages:
    - Portfolio state (balance, position, equity)
    - Trade execution with Alpaca fees
    - Performance tracking
    - Reward calculation
    
    Similar to OpenAI Gym interface but specialized for trading.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        starting_balance: float = 10000.0,
        use_taker_fee: bool = True,
        price_col: str = 'close',
    ):
        """
        Initialize trading environment.
        
        Args:
            data: DataFrame with OHLCV data
            starting_balance: Initial capital
            use_taker_fee: Use taker fees (True) or maker fees (False)
            price_col: Column name for price
        """
        self.data = data.reset_index(drop=True)
        self.starting_balance = starting_balance
        self.use_taker_fee = use_taker_fee
        self.price_col = price_col
        
        # State variables
        self.current_step = 0
        self.balance = starting_balance
        self.position = 0.0  # -1, 0, or 1
        self.shares = 0.0
        self.entry_price = 0.0
        
        # Tracking
        self.total_trades = 0
        self.total_fees = 0.0
        self.trade_history = []
        self.equity_history = [starting_balance]
        self.position_history = [0.0]
        
        # Episode tracking
        self.done = False
        
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_step = 0
        self.balance = self.starting_balance
        self.position = 0.0
        self.shares = 0.0
        self.entry_price = 0.0
        self.total_trades = 0
        self.total_fees = 0.0
        self.trade_history = []
        self.equity_history = [self.starting_balance]
        self.position_history = [0.0]
        self.done = False
        
        return self._get_obs()
    
    def _get_obs(self) -> Dict[str, Any]:
        """Get current observation."""
        if self.current_step >= len(self.data):
            return None
        
        return self.data.iloc[self.current_step].to_dict()
    
    def _get_current_price(self) -> float:
        """Get current market price."""
        if self.current_step >= len(self.data):
            return 0.0
        return self.data.iloc[self.current_step][self.price_col]
    
    def _get_alpaca_fee(self, volume_30d: float) -> float:
        """Get appropriate Alpaca fee based on 30-day volume."""
        for min_vol, max_vol, maker_fee, taker_fee in ALPACA_FEE_TIERS:
            if min_vol <= volume_30d < max_vol:
                return taker_fee if self.use_taker_fee else maker_fee
        return 0.0010  # Default to highest tier taker fee
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + position value)."""
        current_price = self._get_current_price()
        position_value = self.shares * current_price
        return self.balance + position_value
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Dictionary with 'target_position' (-1, 0, or 1)
        
        Returns:
            observation: Next state observation
            done: Whether episode is finished
            info: Additional information (includes equity_change for tracking)
        """
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")
        
        target_position = action.get('target_position', self.position)
        
        # Execute trade if position changes
        trade_info = self._execute_trade(target_position)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.data):
            self.done = True
            # Close any open position
            if self.position != 0:
                self._execute_trade(0.0)
        
        # Track equity change (not for RL, just for monitoring)
        new_equity = self.get_portfolio_value()
        self.equity_history.append(new_equity)
        self.position_history.append(self.position)
        
        if len(self.equity_history) > 1:
            equity_change = new_equity - self.equity_history[-2]
        else:
            equity_change = 0.0
        
        # Get next observation
        obs = self._get_obs()
        
        # Info dictionary
        info = {
            'equity': new_equity,
            'equity_change': equity_change,  # For tracking, not RL reward
            'balance': self.balance,
            'position': self.position,
            'price': self._get_current_price() if obs else 0.0,
            'total_trades': self.total_trades,
            'total_fees': self.total_fees,
            'trade_executed': trade_info.get('executed', False),
        }
        
        return obs, self.done, info
    
    def _execute_trade(self, target_position: float) -> Dict[str, Any]:
        """
        Execute trade to reach target position.
        
        Args:
            target_position: Target position (-1, 0, or 1)
            
        Returns:
            Trade execution details
        """
        current_price = self._get_current_price()
        
        if current_price <= 0:
            return {"executed": False, "reason": "invalid_price"}
        
        # Check if position change needed
        if target_position == self.position:
            return {"executed": False, "reason": "no_change"}
        
        # Calculate fee based on 30-day volume
        actual_fee = self._get_alpaca_fee(self.total_fees * 30)
        
        trade_info = {
            "step": self.current_step,
            "price": current_price,
            "old_position": self.position,
            "new_position": target_position,
            "executed": True,
        }
        
        # Close existing position first
        if self.position != 0:
            # Sell the position and receive proceeds (minus fees)
            close_value = abs(self.shares * current_price)
            close_fee = close_value * actual_fee
            proceeds = close_value - close_fee
            
            # Calculate realized P&L for tracking
            pnl = self.shares * (current_price - self.entry_price)
            
            # Add proceeds to balance (this is the full value, not just PnL)
            self.balance += proceeds
            self.total_fees += close_fee
            self.total_trades += 1
            
            trade_info["close_pnl"] = pnl
            trade_info["close_fee"] = close_fee
            trade_info["close_proceeds"] = proceeds
            
            # Reset position
            self.shares = 0.0
            self.entry_price = 0.0
        
        # Open new position
        if target_position != 0:
            # Check if we have enough balance to trade
            min_trade_size = 100.0  # Minimum $100 to open position
            if self.balance < min_trade_size:
                trade_info["executed"] = False
                trade_info["reason"] = f"insufficient_balance (${self.balance:.2f} < ${min_trade_size})"
                return trade_info
            
            # Use all available balance for the trade (minus fees)
            available = self.balance * 0.99  # Keep 1% buffer
            
            # Calculate trade value and fee
            trade_value = available
            open_fee = trade_value * actual_fee / (1 + actual_fee)  # Fee is part of available
            
            # Net value after fee goes into position
            net_value = available - open_fee
            self.shares = (net_value / current_price) * target_position
            
            # Deduct from balance (the full available amount including fee)
            self.balance -= available
            self.entry_price = current_price
            self.total_fees += open_fee
            self.total_trades += 1
            
            open_value = abs(net_value)
            
            trade_info["open_value"] = open_value
            trade_info["open_fee"] = open_fee
            trade_info["shares"] = self.shares
        
        self.position = target_position
        trade_info["balance"] = self.balance
        trade_info["equity"] = self.get_portfolio_value()
        
        self.trade_history.append(trade_info)
        return trade_info
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        equity_curve = np.array(self.equity_history)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[np.isfinite(returns)]
        
        # Total return
        total_return = (equity_curve[-1] - self.starting_balance) / self.starting_balance
        
        # Sharpe ratio (annualized for 15-min bars)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 4)
        else:
            sharpe = 0.0
        
        # Max drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (cummax - equity_curve) / cummax
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Win rate
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0.0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': self.total_trades,
            'total_fees': self.total_fees,
            'fees_pct': (self.total_fees / self.starting_balance) * 100,
            'final_equity': equity_curve[-1],
            'avg_fee_per_trade': self.total_fees / self.total_trades if self.total_trades > 0 else 0.0,
        }
    
    def render(self) -> str:
        """Render current state as string."""
        equity = self.get_portfolio_value()
        return (f"Step: {self.current_step} | "
                f"Equity: ${equity:.2f} | "
                f"Position: {self.position:+.0f} | "
                f"Trades: {self.total_trades} | "
                f"Fees: ${self.total_fees:.2f}")
