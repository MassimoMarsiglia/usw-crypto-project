"""
multi_asset_agent.py

Multi-asset trading agent that makes decisions for a portfolio of cryptocurrencies.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from collections import deque

from trading_rules import TradingRule


class MultiAssetTradingAgent:
    """
    Multi-asset trading agent with portfolio-level decision making.
    
    Makes trading decisions for multiple assets simultaneously,
    considering portfolio constraints and diversification.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        feature_calculator: Callable,
        assets: List[str],  # e.g., ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT"]
        target_indices: Dict[str, int],  # Maps asset to model output index
        buffer_size: int = 100,
        trading_rules: Optional[TradingRule] = None,
        device: str = 'cpu',
    ):
        """
        Initialize multi-asset agent.
        
        Args:
            model: Neural network for predictions
            feature_calculator: Function to compute features from buffer
            assets: List of asset names to trade
            target_indices: Mapping from asset to model output index
            buffer_size: Size of ring buffer for historical data
            trading_rules: TradingRule instance for decision making
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.feature_calculator = feature_calculator
        self.assets = assets
        self.target_indices = target_indices
        self.buffer_size = buffer_size
        self.trading_rules = trading_rules
        
        # Ring buffer for historical data (shared across all assets)
        self.buffer: deque = deque(maxlen=buffer_size)
        
        # Trading state per asset
        self.positions = {asset: 0.0 for asset in assets}
        self.entry_prices = {asset: 0.0 for asset in assets}
        
        # Timestep counter
        self.step_count = 0
        
        # Last predictions
        self.last_predictions = {asset: None for asset in assets}
        
    def reset(self):
        """Reset agent state."""
        self.buffer.clear()
        self.positions = {asset: 0.0 for asset in self.assets}
        self.entry_prices = {asset: 0.0 for asset in self.assets}
        self.step_count = 0
        self.last_predictions = {asset: None for asset in self.assets}
        
    def add_data(self, data: Dict[str, Any]):
        """Add new market data to ring buffer."""
        self.buffer.append(data)
        self.step_count += 1
        
    def is_ready(self) -> bool:
        """Check if buffer has enough data."""
        return len(self.buffer) >= self.buffer_size
    
    def get_features(self) -> Optional[np.ndarray]:
        """Calculate features from current buffer state."""
        if not self.is_ready():
            return None
        
        buffer_df = pd.DataFrame(list(self.buffer))
        return self.feature_calculator(buffer_df)
    
    def predict(self) -> Optional[Dict[str, float]]:
        """
        Make predictions for all assets.
        
        Returns:
            Dictionary mapping asset -> predicted return
        """
        features = self.get_features()
        if features is None:
            return None
        
        with torch.no_grad():
            x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred = self.model(x_tensor).cpu().numpy()[0]
        
        # Map predictions to assets (combine 4h and 1d predictions)
        predictions = {}
        for asset in self.assets:
            target_indices = self.target_indices.get(asset, [0])
            
            # Handle both single index and list of indices
            if isinstance(target_indices, list):
                # Average multiple time horizon predictions (4h and 1d)
                valid_preds = []
                for idx in target_indices:
                    if idx < len(pred):
                        valid_preds.append(float(pred[idx]))
                
                if valid_preds:
                    # Equal weight average of 4h and 1d predictions
                    predictions[asset] = sum(valid_preds) / len(valid_preds)
                else:
                    predictions[asset] = 0.0
            else:
                # Single index (backward compatibility)
                if target_indices < len(pred):
                    predictions[asset] = float(pred[target_indices])
                else:
                    predictions[asset] = 0.0
        
        self.last_predictions = predictions
        return predictions
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current market prices from buffer."""
        if len(self.buffer) == 0:
            return {asset: 0.0 for asset in self.assets}
        
        prices = {}
        for asset in self.assets:
            col_name = f"{asset}_close"
            prices[asset] = self.buffer[-1].get(col_name, 0.0)
        
        return prices
    
    def decide_actions(
        self, 
        portfolio_value: float,
        position_values: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Decide trading actions for all assets.
        
        Args:
            portfolio_value: Current total portfolio value
            position_values: Current value of each position
            
        Returns:
            Dictionary mapping asset -> target_position (-1, 0, or 1)
        """
        predictions = self.predict()
        
        if predictions is None:
            return {asset: self.positions[asset] for asset in self.assets}
        
        current_prices = self.get_current_prices()
        actions = {}
        
        # Decide for each asset
        for asset in self.assets:
            prediction = predictions[asset]
            current_position = self.positions[asset]
            entry_price = self.entry_prices[asset]
            current_price = current_prices[asset]
            position_value = position_values.get(asset, 0.0)
            
            # Calculate P&L percentage if in position
            if current_position != 0 and entry_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price * current_position
            else:
                pnl_pct = 0.0
            
            # Prepare state for trading rules
            state = {
                'prediction': prediction,
                'position': current_position,
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl_pct': pnl_pct,
                'step': self.step_count,
                'portfolio_value': portfolio_value,
                'position_value': position_value,
            }
            
            # Use trading rules if provided
            if self.trading_rules is not None:
                # Debug logging every 1000 steps for first asset only (shares logged in run_simulation_multi.py)
                if self.step_count % 1000 == 0 and asset == self.assets[0]:
                    print(f"  DEBUG | Step {self.step_count} | {asset} | Pred: {prediction:.6f} | Pos: {current_position:.0f} | PnL: {pnl_pct:.4f}")
                
                # Check if should close position first
                if current_position != 0 and self.trading_rules.should_close(state):
                    target_position = 0.0
                # Check for new positions (only if not already in that position)
                elif current_position != 1.0 and self.trading_rules.should_buy(state):
                    target_position = 1.0
                    if self.step_count % 1000 == 0:
                        print(f"  SIGNAL | {asset} | Step {self.step_count} | BUY signal | Pred: {prediction:.6f}")
                elif current_position != -1.0 and self.trading_rules.should_sell(state):
                    target_position = -1.0
                    if self.step_count % 1000 == 0:
                        print(f"  SIGNAL | {asset} | Step {self.step_count} | SELL signal | Pred: {prediction:.6f}")
                else:
                    target_position = current_position  # Hold current position
            else:
                # Default: simple prediction-based trading
                if prediction > 0:
                    target_position = 1.0
                elif prediction < 0:
                    target_position = -1.0
                else:
                    target_position = 0.0
            
            actions[asset] = target_position
        
        return actions
    
    def update_positions(self, positions: Dict[str, float]):
        """Update agent's internal position state."""
        self.positions = positions.copy()
