"""
Base classes for risk management
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StopType(Enum):
    """Types of stop losses"""
    HARD_STOP = "hard_stop"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    PROFIT_TARGET = "profit_target"
    VOLATILITY_STOP = "volatility_stop"


class RiskStatus(Enum):
    """Risk status levels"""
    OK = "ok"
    WARNING = "warning"
    BREACH = "breach"
    CRITICAL = "critical"


class BaseRiskChecker(ABC):
    """Base class for risk checkers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def check(self, **kwargs) -> Dict[str, Any]:
        """Perform risk check and return status"""
        pass


class BaseStopEngine(ABC):
    """Base class for stop loss engines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def generate_stops(self, entry_price: float, position_type: str, **kwargs) -> Dict[str, float]:
        """Generate stop loss levels"""
        pass


class RiskViolationError(Exception):
    """Raised when risk limits are violated"""
    pass


class StopCalculationError(Exception):
    """Raised when stop calculation fails"""
    pass