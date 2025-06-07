"""
Base classes for data fetching
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime


class BaseDataFetcher(ABC):
    """Base class for all data fetchers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rate_limit_delay = config.get("rate_limit_delay", 0.1)
        
    @abstractmethod
    def fetch(self, symbols: List[str], **kwargs) -> pd.DataFrame:
        """Fetch data for given symbols"""
        pass
        
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate fetched data quality"""
        pass


class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded"""
    pass