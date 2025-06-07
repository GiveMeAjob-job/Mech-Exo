"""
OHLC data fetcher using yfinance and other sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
import logging
from .base import BaseDataFetcher, DataValidationError, RateLimitError

logger = logging.getLogger(__name__)


class OHLCDownloader(BaseDataFetcher):
    """Downloads OHLC data from Yahoo Finance and other sources"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        
    def fetch(self, symbols: List[str], period: str = "1y", 
              interval: str = "1d", **kwargs) -> pd.DataFrame:
        """
        Fetch OHLC data for given symbols
        
        Args:
            symbols: List of ticker symbols
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLC data
        """
        all_data = []
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching OHLC data for {symbol}")
                data = self._fetch_single_symbol(symbol, period, interval)
                
                if data is not None and not data.empty:
                    data['symbol'] = symbol
                    all_data.append(data)
                    
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue
                
        if not all_data:
            raise DataValidationError("No data fetched for any symbols")
            
        combined_data = pd.concat(all_data, ignore_index=True)
        
        if not self.validate_data(combined_data):
            raise DataValidationError("Data validation failed")
            
        return combined_data
    
    def _fetch_single_symbol(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol with retry logic"""
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if data.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return None
                    
                # Reset index to get datetime as column
                data = data.reset_index()
                
                # Standardize column names
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                
                # Ensure we have required columns
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in data.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in data.columns]
                    logger.error(f"Missing columns for {symbol}: {missing}")
                    return None
                
                # Calculate additional metrics
                data['returns'] = data['close'].pct_change()
                data['volatility'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
                data['atr'] = self._calculate_atr(data)
                
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All attempts failed for {symbol}")
                    return None
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate OHLC data quality"""
        if data.empty:
            logger.error("Data is empty")
            return False
            
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (data[col] <= 0).any():
                logger.error(f"Found non-positive prices in {col}")
                return False
                
        # Check high >= low
        if (data['high'] < data['low']).any():
            logger.error("Found high < low")
            return False
            
        # Check for excessive missing data
        missing_pct = data.isnull().sum() / len(data)
        if (missing_pct > 0.1).any():  # More than 10% missing
            logger.warning(f"High missing data percentage: {missing_pct.max():.2%}")
            
        logger.info(f"Data validation passed for {len(data)} rows")
        return True
    
    def fetch_latest(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch latest available data (last 5 days)"""
        return self.fetch(symbols, period="5d", interval="1d")
    
    def fetch_intraday(self, symbols: List[str], interval: str = "1h") -> pd.DataFrame:
        """Fetch intraday data (last 7 days)"""
        return self.fetch(symbols, period="7d", interval=interval)