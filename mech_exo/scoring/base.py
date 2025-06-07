"""
Base classes for scoring and factor models
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseFactor(ABC):
    """Base class for factor calculations"""
    
    def __init__(self, name: str, weight: float, direction: str = "higher_better"):
        self.name = name
        self.weight = weight
        self.direction = direction  # "higher_better", "lower_better", "mean_revert"
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate factor values for given data"""
        pass
    
    def normalize(self, values: pd.Series, method: str = "zscore") -> pd.Series:
        """Normalize factor values"""
        if method == "zscore":
            return (values - values.mean()) / values.std()
        elif method == "rank":
            return values.rank(pct=True)
        elif method == "minmax":
            return (values - values.min()) / (values.max() - values.min())
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def apply_direction(self, normalized_values: pd.Series) -> pd.Series:
        """Apply direction preference to normalized values"""
        if self.direction == "lower_better":
            return -normalized_values
        elif self.direction == "mean_revert":
            return -np.abs(normalized_values)
        else:  # higher_better
            return normalized_values


class BaseScorer(ABC):
    """Base class for scoring models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.factors = {}
        self._initialize_factors()
        
    @abstractmethod
    def _initialize_factors(self):
        """Initialize factor models"""
        pass
    
    @abstractmethod
    def score(self, symbols: List[str], **kwargs) -> pd.DataFrame:
        """Score symbols and return ranked DataFrame"""
        pass
    
    def validate_input_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate input data has required columns"""
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True


class FactorCalculationError(Exception):
    """Raised when factor calculation fails"""
    pass


class ScoringError(Exception):
    """Raised when scoring fails"""
    pass