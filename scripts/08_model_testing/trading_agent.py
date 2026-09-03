"""
trading_agent.py

Autonomous trading agent that can operate in both simulation and live trading environments.
Maintains a ring buffer for feature calculation and manages the model internally.

Features:
- Ring buffer for historical data
- Feature calculation on-the-fly
- Model inference
- Position management
- Works with both simulation and websocket (deployment)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from collections import deque


class CryptoMLP(nn.Module):
    """Default model architecture - can be replaced with custom model."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.input_norm(x)
        return self.net(x)


class TradingAgent:
    """
    Autonomous trading agent with internal model and ring buffer.
    
    Can operate in:
    - Simulation mode: Reads from DataFrame
    - Deployment mode: Receives data from websocket
    
    The agent maintains its own state and makes autonomous trading decisions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_calculator: Callable,
        buffer_size: int = 100,
        trading_rules: Optional['TradingRule'] = None,
        target_idx: int = 0,
        device: str = "cpu",
        price_col: str = 'close',
    ):
        """
        Initialize trading agent.
        
        Args:
            model: PyTorch model for predictions
            feature_calculator: Function that takes buffer DataFrame and returns features
            buffer_size: Size of ring buffer for historical data
            trading_rules: TradingRule instance for decision making (optional)
            target_idx: Which model output to use for trading decisions
            device: 'cpu' or 'cuda'
            price_col: Column name for price data
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.feature_calculator = feature_calculator
        self.buffer_size = buffer_size
        self.trading_rules = trading_rules
        self.target_idx = target_idx
        self.price_col = price_col
        
        # Ring buffer for historical data (FIFO queue)
        self.buffer: deque = deque(maxlen=buffer_size)
        
        # Trading state
        self.position = 0.0  # -1 (short), 0 (neutral), +1 (long)
        self.entry_price = 0.0
        
        # Timestep counter
        self.step_count = 0
        
        # Last prediction
        self.last_prediction = None
        
    def reset(self):
        """Reset agent state."""
        self.buffer.clear()
        self.position = 0.0
        self.entry_price = 0.0
        self.step_count = 0
        self.last_prediction = None
        
    def add_data(self, data: Dict[str, Any]):
        """
        Add new market data to ring buffer.
        
        Args:
            data: Dictionary with OHLCV data
        """
        self.buffer.append(data)
        self.step_count += 1
        
    def is_ready(self) -> bool:
        """Check if buffer has enough data for feature calculation."""
        return len(self.buffer) >= self.buffer_size
    
    def get_features(self) -> Optional[np.ndarray]:
        """
        Calculate features from current buffer state.
        
        Returns:
            Feature vector as numpy array, or None if buffer not ready
        """
        if not self.is_ready():
            return None
        
        buffer_df = pd.DataFrame(list(self.buffer))
        return self.feature_calculator(buffer_df)
    
    def predict(self) -> Optional[float]:
        """
        Make prediction using current buffer state.
        
        Returns:
            Predicted return for target_idx, or None if not ready
        """
        features = self.get_features()
        if features is None:
            return None
        
        with torch.no_grad():
            x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred = self.model(x_tensor).cpu().numpy()[0]
        
        # Get prediction for target
        if self.target_idx >= len(pred):
            self.target_idx = 0
        
        self.last_prediction = float(pred[self.target_idx])
        return self.last_prediction
    
    def get_current_price(self) -> float:
        """Get current market price from buffer."""
        if len(self.buffer) == 0:
            return 0.0
        return self.buffer[-1].get(self.price_col, 0.0)
    
    def decide_action(self) -> Dict[str, Any]:
        """
        Decide trading action based on prediction and trading rules.
        
        Returns:
            Dictionary with action details:
            - action: 'buy', 'sell', 'hold'
            - target_position: -1, 0, or 1
            - prediction: model prediction value
            - current_position: current position
        """
        prediction = self.predict()
        
        if prediction is None:
            return {
                'action': 'hold',
                'target_position': self.position,
                'prediction': None,
                'current_position': self.position,
                'ready': False
            }
        
        # Calculate P&L percentage if in position
        current_price = self.get_current_price()
        if self.position != 0 and self.entry_price > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price * self.position
        else:
            pnl_pct = 0.0
        
        # Prepare state for trading rules
        state = {
            'prediction': prediction,
            'position': self.position,
            'entry_price': self.entry_price,
            'current_price': current_price,
            'pnl_pct': pnl_pct,
            'step': self.step_count,
        }
        
        # Use trading rules if provided, otherwise use simple threshold
        if self.trading_rules is not None:
            # Debug logging every 1000 steps
            if self.step_count % 1000 == 0:
                print(f"  DEBUG | Step {self.step_count} | Pred: {prediction:.6f} | Pos: {self.position:.0f} | PnL: {pnl_pct:.4f} | Price: {current_price:.2f}")
            
            # Check if should close position first
            if self.position != 0 and self.trading_rules.should_close(state):
                target_position = 0.0
            # Check for new positions (only if not already in that position)
            elif self.position != 1.0 and self.trading_rules.should_buy(state):
                target_position = 1.0
                print(f"  SIGNAL | Step {self.step_count} | BUY signal | Pred: {prediction:.6f}")
            elif self.position != -1.0 and self.trading_rules.should_sell(state):
                target_position = -1.0
                print(f"  SIGNAL | Step {self.step_count} | SELL signal | Pred: {prediction:.6f}")
            else:
                target_position = self.position  # Hold current position
        else:
            # Default: simple prediction-based trading (backward compatible)
            if prediction > 0:
                target_position = 1.0
            elif prediction < 0:
                target_position = -1.0
            else:
                target_position = 0.0
        
        # Determine action
        if target_position == self.position:
            action = 'hold'
        elif target_position > self.position:
            action = 'buy'
        elif target_position < self.position:
            action = 'sell'
        else:
            action = 'hold'
        
        return {
            'action': action,
            'target_position': target_position,
            'prediction': prediction,
            'current_position': self.position,
            'price': current_price,
            'pnl_pct': pnl_pct,
            'ready': True,
            'step': self.step_count,
            'rule_used': self.trading_rules.name if self.trading_rules else 'default'
        }
    
    def update_position(self, new_position: float, entry_price: float = None):
        """
        Update agent's position after trade execution.
        
        Args:
            new_position: New position value (-1, 0, or 1)
            entry_price: Entry price for new position (optional)
        """
        self.position = new_position
        if entry_price is not None:
            self.entry_price = entry_price
        elif new_position != 0:
            self.entry_price = self.get_current_price()
        else:
            self.entry_price = 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current agent state.
        
        Returns:
            Dictionary with agent state information
        """
        return {
            'position': self.position,
            'entry_price': self.entry_price,
            'current_price': self.get_current_price(),
            'buffer_size': len(self.buffer),
            'ready': self.is_ready(),
            'step_count': self.step_count,
            'last_prediction': self.last_prediction
        }
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'buffer_size': self.buffer_size,
            'prediction_threshold': self.prediction_threshold,
            'target_idx': self.target_idx,
        }, path)
    
    def load_model(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()


# WebSocket integration for live trading
class WebSocketTradingAgent(TradingAgent):
    """
    Extended agent for live trading with websocket data feed.
    
    Usage:
        agent = WebSocketTradingAgent(...)
        
        # In your websocket handler:
        async def on_message(msg):
            action = agent.process_message(msg)
            if action['action'] != 'hold':
                await execute_trade(action)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_message_time = None
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming websocket message and decide action.
        
        Args:
            message: Message from websocket with OHLCV data
            
        Returns:
            Action dictionary from decide_action()
        """
        # Add timestamp tracking
        self.last_message_time = pd.Timestamp.now()
        
        # Extract relevant data from message
        # This should be customized based on your websocket format
        data = self._extract_data(message)
        
        # Add to buffer
        self.add_data(data)
        
        # Decide action
        return self.decide_action()
    
    def _extract_data(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract OHLCV data from websocket message.
        
        Override this method to match your websocket message format.
        """
        # Default implementation - customize based on your format
        return {
            'open': message.get('o', message.get('open')),
            'high': message.get('h', message.get('high')),
            'low': message.get('l', message.get('low')),
            'close': message.get('c', message.get('close')),
            'volume': message.get('v', message.get('volume')),
            'vwap': message.get('vwap', message.get('close')),
            'trade_count': message.get('n', message.get('trade_count', 0))
        }
