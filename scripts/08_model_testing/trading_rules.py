"""
trading_rules.py

Flexible trading rules system for making trading decisions based on predictions
and market conditions.

Rules can be combined, customized, and passed to the TradingAgent.
"""

import numpy as np
from typing import Dict, Any, Callable, List, Optional


class TradingRule:
    """Base class for trading rules."""
    
    def __init__(self, name: str):
        self.name = name
    
    def should_buy(self, state: Dict[str, Any]) -> bool:
        """Return True if should enter long position."""
        raise NotImplementedError
    
    def should_sell(self, state: Dict[str, Any]) -> bool:
        """Return True if should enter short position."""
        raise NotImplementedError
    
    def should_close(self, state: Dict[str, Any]) -> bool:
        """Return True if should close current position."""
        raise NotImplementedError


class ThresholdRule(TradingRule):
    """
    Simple threshold-based rule.
    
    Buy if prediction > buy_threshold
    Sell if prediction < -sell_threshold
    Close if |prediction| < close_threshold
    """
    
    def __init__(
        self,
        buy_threshold: float = 0.001,
        sell_threshold: float = 0.001,
        close_threshold: float = 0.0005,
    ):
        super().__init__("ThresholdRule")
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.close_threshold = close_threshold
    
    def should_buy(self, state: Dict[str, Any]) -> bool:
        prediction = state.get('prediction', 0.0)
        return prediction > self.buy_threshold
    
    def should_sell(self, state: Dict[str, Any]) -> bool:
        prediction = state.get('prediction', 0.0)
        return prediction < -self.sell_threshold
    
    def should_close(self, state: Dict[str, Any]) -> bool:
        prediction = state.get('prediction', 0.0)
        return abs(prediction) < self.close_threshold


class TrendFollowingRule(TradingRule):
    """
    Trend following: only trade in direction of prediction.
    Requires strong signal and won't trade against trend.
    """
    
    def __init__(
        self,
        min_prediction: float = 0.002,
        stop_loss: float = -0.02,
    ):
        super().__init__("TrendFollowingRule")
        self.min_prediction = min_prediction
        self.stop_loss = stop_loss
    
    def should_buy(self, state: Dict[str, Any]) -> bool:
        prediction = state.get('prediction', 0.0)
        return prediction > self.min_prediction
    
    def should_sell(self, state: Dict[str, Any]) -> bool:
        prediction = state.get('prediction', 0.0)
        return prediction < -self.min_prediction
    
    def should_close(self, state: Dict[str, Any]) -> bool:
        # Close if losing too much
        pnl_pct = state.get('pnl_pct', 0.0)
        return pnl_pct < self.stop_loss


class MeanReversionRule(TradingRule):
    """
    Mean reversion: trade against extreme moves.
    Buy when price drops significantly, sell when it rises significantly.
    """
    
    def __init__(
        self,
        buy_on_drop: float = -0.03,  # Buy if predicted drop > 3%
        sell_on_rise: float = 0.03,   # Sell if predicted rise > 3%
        take_profit: float = 0.01,    # Close at 1% profit
    ):
        super().__init__("MeanReversionRule")
        self.buy_on_drop = buy_on_drop
        self.sell_on_rise = sell_on_rise
        self.take_profit = take_profit
    
    def should_buy(self, state: Dict[str, Any]) -> bool:
        # Buy after big predicted drop (expecting rebound)
        prediction = state.get('prediction', 0.0)
        return prediction < self.buy_on_drop
    
    def should_sell(self, state: Dict[str, Any]) -> bool:
        # Sell after big predicted rise (expecting pullback)
        prediction = state.get('prediction', 0.0)
        return prediction > self.sell_on_rise
    
    def should_close(self, state: Dict[str, Any]) -> bool:
        # Close at take profit
        pnl_pct = state.get('pnl_pct', 0.0)
        return pnl_pct > self.take_profit


class VolatilityFilterRule(TradingRule):
    """
    Only trade when volatility is within acceptable range.
    Avoids trading in extremely volatile or dead markets.
    """
    
    def __init__(
        self,
        min_volatility: float = 0.0005,
        max_volatility: float = 0.05,
        prediction_threshold: float = 0.001,
    ):
        super().__init__("VolatilityFilterRule")
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.prediction_threshold = prediction_threshold
    
    def _check_volatility(self, state: Dict[str, Any]) -> bool:
        """Check if volatility is in acceptable range."""
        volatility = state.get('volatility', 0.01)  # Default to 1%
        return self.min_volatility < volatility < self.max_volatility
    
    def should_buy(self, state: Dict[str, Any]) -> bool:
        if not self._check_volatility(state):
            return False
        prediction = state.get('prediction', 0.0)
        return prediction > self.prediction_threshold
    
    def should_sell(self, state: Dict[str, Any]) -> bool:
        if not self._check_volatility(state):
            return False
        prediction = state.get('prediction', 0.0)
        return prediction < -self.prediction_threshold
    
    def should_close(self, state: Dict[str, Any]) -> bool:
        # Close if volatility becomes too extreme
        return not self._check_volatility(state)


class CompositeRule(TradingRule):
    """
    Combine multiple rules with AND or OR logic.
    """
    
    def __init__(
        self,
        rules: List[TradingRule],
        logic: str = "AND",  # "AND" or "OR"
    ):
        super().__init__(f"CompositeRule({logic})")
        self.rules = rules
        self.logic = logic.upper()
    
    def should_buy(self, state: Dict[str, Any]) -> bool:
        results = [rule.should_buy(state) for rule in self.rules]
        return all(results) if self.logic == "AND" else any(results)
    
    def should_sell(self, state: Dict[str, Any]) -> bool:
        results = [rule.should_sell(state) for rule in self.rules]
        return all(results) if self.logic == "AND" else any(results)
    
    def should_close(self, state: Dict[str, Any]) -> bool:
        results = [rule.should_close(state) for rule in self.rules]
        return all(results) if self.logic == "AND" else any(results)


class RiskManagementRule(TradingRule):
    """
    Risk management: stop loss, take profit, max positions.
    """
    
    def __init__(
        self,
        stop_loss_pct: float = -0.05,      # Stop loss at -5%
        take_profit_pct: float = 0.05,     # Take profit at +5%
        max_loss_per_day: float = -0.10,   # Max -10% per day
        prediction_threshold: float = 0.0,
    ):
        super().__init__("RiskManagementRule")
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_loss_per_day = max_loss_per_day
        self.prediction_threshold = prediction_threshold
        self.daily_pnl = 0.0
    
    def should_buy(self, state: Dict[str, Any]) -> bool:
        # Don't buy if hit daily loss limit
        if self.daily_pnl < self.max_loss_per_day:
            return False
        prediction = state.get('prediction', 0.0)
        return prediction > self.prediction_threshold
    
    def should_sell(self, state: Dict[str, Any]) -> bool:
        # Don't sell if hit daily loss limit
        if self.daily_pnl < self.max_loss_per_day:
            return False
        prediction = state.get('prediction', 0.0)
        return prediction < -self.prediction_threshold
    
    def should_close(self, state: Dict[str, Any]) -> bool:
        # Close on stop loss or take profit
        pnl_pct = state.get('pnl_pct', 0.0)
        return (pnl_pct <= self.stop_loss_pct or 
                pnl_pct >= self.take_profit_pct or
                self.daily_pnl < self.max_loss_per_day)


# Pre-defined rule sets
def get_conservative_rules() -> TradingRule:
    """Conservative trading: high thresholds, risk management."""
    return CompositeRule([
        ThresholdRule(
            buy_threshold=0.005,   # 0.5% min predicted return
            sell_threshold=0.005,
            close_threshold=0.002
        ),
        RiskManagementRule(
            stop_loss_pct=-0.03,   # Stop loss at -3%
            take_profit_pct=0.03,  # Take profit at +3%
        )
    ], logic="AND")


def get_aggressive_rules() -> TradingRule:
    """Aggressive trading: low thresholds, trend following."""
    return TrendFollowingRule(
        min_prediction=0.0005,  # 0.05% min
        stop_loss=-0.10         # -10% stop loss
    )


def get_balanced_rules() -> TradingRule:
    """Balanced approach: moderate thresholds with risk management."""
    return CompositeRule([
        ThresholdRule(
            buy_threshold=0.002,
            sell_threshold=0.002,
            close_threshold=0.001
        ),
        RiskManagementRule(
            stop_loss_pct=-0.05,
            take_profit_pct=0.05,
        )
    ], logic="AND")


def get_volatility_filtered_rules() -> TradingRule:
    """Trade only in normal volatility conditions."""
    return VolatilityFilterRule(
        min_volatility=0.001,
        max_volatility=0.03,
        prediction_threshold=0.002
    )


class DiversificationRule(TradingRule):
    """
    Limits position size per asset to enforce diversification.
    Prevents over-concentration in a single asset.
    """
    
    def __init__(
        self,
        max_position_pct: float = 0.5,  # Max 50% of capital in one asset
        prediction_threshold: float = 0.005,
    ):
        super().__init__("DiversificationRule")
        self.max_position_pct = max_position_pct
        self.prediction_threshold = prediction_threshold
    
    def should_buy(self, state: Dict[str, Any]) -> bool:
        prediction = state.get('prediction', 0.0)
        portfolio_value = state.get('portfolio_value', 0.0)
        position_value = state.get('position_value', 0.0)
        
        # Check prediction threshold
        if prediction <= self.prediction_threshold:
            return False
        
        # Check if adding to this position would exceed diversification limit
        if portfolio_value > 0:
            current_weight = abs(position_value) / portfolio_value
            if current_weight >= self.max_position_pct:
                return False
        
        return True
    
    def should_sell(self, state: Dict[str, Any]) -> bool:
        prediction = state.get('prediction', 0.0)
        portfolio_value = state.get('portfolio_value', 0.0)
        position_value = state.get('position_value', 0.0)
        
        # Check prediction threshold
        if prediction >= -self.prediction_threshold:
            return False
        
        # Check if adding to this position would exceed diversification limit
        if portfolio_value > 0:
            current_weight = abs(position_value) / portfolio_value
            if current_weight >= self.max_position_pct:
                return False
        
        return True
    
    def should_close(self, state: Dict[str, Any]) -> bool:
        prediction = state.get('prediction', 0.0)
        return abs(prediction) < self.prediction_threshold / 2


# Example: Custom rule
def custom_rule_example() -> TradingRule:
    """
    Example of creating a custom composite rule.
    """
    # Combine trend following with volatility filter
    return CompositeRule([
        TrendFollowingRule(min_prediction=0.003),
        VolatilityFilterRule(max_volatility=0.02),
    ], logic="AND")
