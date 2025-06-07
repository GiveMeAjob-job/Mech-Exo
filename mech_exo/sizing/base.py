"""
Base classes for position sizing
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methods"""
    FIXED_PERCENT = "fixed_percent"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"


class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"


class BaseSizer(ABC):
    """Base class for position sizers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nav = config.get("nav", 100000.0)  # Default NAV
        self.max_position_size = config.get("max_position_size", 0.10)  # 10% max
        self.min_position_value = config.get("min_position_value", 1000.0)  # $1000 min
        
    @abstractmethod
    def calculate_size(self, symbol: str, price: float, **kwargs) -> int:
        """Calculate position size in shares"""
        pass
    
    def validate_size(self, symbol: str, price: float, shares: int) -> bool:
        """Validate calculated position size"""
        position_value = abs(shares * price)
        
        # Check minimum position value
        if position_value < self.min_position_value:
            logger.warning(f"Position value ${position_value:.2f} below minimum ${self.min_position_value}")
            return False
            
        # Check maximum position size as % of NAV
        position_pct = position_value / self.nav
        if position_pct > self.max_position_size:
            logger.warning(f"Position {position_pct:.2%} exceeds maximum {self.max_position_size:.2%}")
            return False
            
        return True
    
    def adjust_for_liquidity(self, symbol: str, calculated_shares: int, 
                           avg_daily_volume: Optional[float] = None) -> int:
        """Adjust position size for liquidity constraints"""
        if avg_daily_volume is None:
            return calculated_shares
            
        # Don't exceed 10% of average daily volume
        max_liquid_shares = int(avg_daily_volume * 0.10)
        
        if abs(calculated_shares) > max_liquid_shares:
            logger.info(f"Reducing {symbol} size from {calculated_shares} to {max_liquid_shares} for liquidity")
            return max_liquid_shares if calculated_shares > 0 else -max_liquid_shares
            
        return calculated_shares


class SizingError(Exception):
    """Raised when position sizing fails"""
    pass


class InsufficientCapitalError(SizingError):
    """Raised when insufficient capital for position"""
    pass


class LiquidityConstraintError(SizingError):
    """Raised when position too large for liquidity"""
    pass