"""
Fundamental data fetcher using yfinance and Finnhub
"""

import yfinance as yf
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import time
import logging
from .base import BaseDataFetcher, DataValidationError, RateLimitError

logger = logging.getLogger(__name__)


class FundamentalFetcher(BaseDataFetcher):
    """Fetches fundamental data from various sources"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.finnhub_api_key = config.get("finnhub", {}).get("api_key")
        self.finnhub_base_url = config.get("finnhub", {}).get("base_url", "https://finnhub.io/api/v1")
        self.max_retries = config.get("max_retries", 3)
        
    def fetch(self, symbols: List[str], **kwargs) -> pd.DataFrame:
        """Fetch fundamental data for given symbols"""
        all_data = []
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching fundamental data for {symbol}")
                
                # Get data from yfinance (free)
                yf_data = self._fetch_yfinance_fundamentals(symbol)
                
                # Get additional data from Finnhub if API key available
                if self.finnhub_api_key:
                    finnhub_data = self._fetch_finnhub_fundamentals(symbol)
                    yf_data.update(finnhub_data)
                
                if yf_data:
                    yf_data['symbol'] = symbol
                    yf_data['fetch_date'] = pd.Timestamp.now()
                    all_data.append(yf_data)
                    
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Failed to fetch fundamentals for {symbol}: {e}")
                continue
                
        if not all_data:
            raise DataValidationError("No fundamental data fetched")
            
        df = pd.DataFrame(all_data)
        
        if not self.validate_data(df):
            raise DataValidationError("Fundamental data validation failed")
            
        return df
    
    def _fetch_yfinance_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key fundamental metrics
            fundamentals = {
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'enterprise_to_revenue': info.get('enterpriseToRevenue'),
                'enterprise_to_ebitda': info.get('enterpriseToEbitda'),
                'debt_to_equity': info.get('debtToEquity'),
                'return_on_equity': info.get('returnOnEquity'),
                'return_on_assets': info.get('returnOnAssets'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'gross_margins': info.get('grossMargins'),
                'operating_margins': info.get('operatingMargins'),
                'profit_margins': info.get('profitMargins'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'total_cash': info.get('totalCash'),
                'total_debt': info.get('totalDebt'),
                'total_revenue': info.get('totalRevenue'),
                'ebitda': info.get('ebitda'),
                'free_cashflow': info.get('freeCashflow'),
                'beta': info.get('beta'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'dividend_yield': info.get('dividendYield'),
                'ex_dividend_date': info.get('exDividendDate'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'currency': info.get('currency'),
                'current_price': info.get('currentPrice'),
                'target_high_price': info.get('targetHighPrice'),
                'target_low_price': info.get('targetLowPrice'),
                'target_mean_price': info.get('targetMeanPrice'),
                'recommendation_key': info.get('recommendationKey'),
                'number_of_analyst_opinions': info.get('numberOfAnalystOpinions')
            }
            
            # Clean None values
            fundamentals = {k: v for k, v in fundamentals.items() if v is not None}
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {symbol}: {e}")
            return {}
    
    def _fetch_finnhub_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch additional fundamental data from Finnhub"""
        if not self.finnhub_api_key:
            return {}
            
        try:
            # Get basic financials
            url = f"{self.finnhub_base_url}/stock/metric"
            params = {
                'symbol': symbol,
                'metric': 'all',
                'token': self.finnhub_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            metric_data = data.get('metric', {})
            
            # Extract useful metrics
            finnhub_fundamentals = {
                'eps_ttm': metric_data.get('epsInclExtraItemsTTM'),
                'sales_per_share_ttm': metric_data.get('salesPerShareTTM'),
                'book_value_per_share': metric_data.get('bookValuePerShareQuarterly'),
                'cash_per_share': metric_data.get('cashPerShareQuarterly'),
                'dividend_per_share_ttm': metric_data.get('dividendPerShareTTM'),
                'revenue_ttm': metric_data.get('revenueTTM'),
                'gross_profit_ttm': metric_data.get('grossProfitTTM'),
                'operating_income_ttm': metric_data.get('operatingIncomeTTM'),
                'net_income_ttm': metric_data.get('netIncomeTTM'),
                'ebit_ttm': metric_data.get('ebitTTM'),
                'ebitda_ttm': metric_data.get('ebitdaTTM'),
                'total_assets': metric_data.get('totalAssetsQuarterly'),
                'total_liabilities': metric_data.get('totalLiabilitiesQuarterly'),
                'total_equity': metric_data.get('totalEquityQuarterly'),
                'working_capital': metric_data.get('workingCapitalQuarterly'),
                'inventory_turnover_ttm': metric_data.get('inventoryTurnoverTTM'),
                'asset_turnover_ttm': metric_data.get('assetTurnoverTTM'),
                'roe_ttm': metric_data.get('roeTTM'),
                'roa_ttm': metric_data.get('roaTTM'),
                'roic_ttm': metric_data.get('roicTTM')
            }
            
            # Clean None values
            finnhub_fundamentals = {k: v for k, v in finnhub_fundamentals.items() if v is not None}
            
            return finnhub_fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub data for {symbol}: {e}")
            return {}
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate fundamental data quality"""
        if data.empty:
            logger.error("Fundamental data is empty")
            return False
            
        required_columns = ['symbol', 'fetch_date']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Check if we have at least some fundamental metrics
        fundamental_cols = [col for col in data.columns 
                          if col not in ['symbol', 'fetch_date']]
        
        if len(fundamental_cols) == 0:
            logger.error("No fundamental metrics found")
            return False
            
        # Check for reasonable values in key ratios
        if 'pe_ratio' in data.columns:
            extreme_pe = (data['pe_ratio'] < 0) | (data['pe_ratio'] > 1000)
            if extreme_pe.any():
                logger.warning(f"Found extreme P/E ratios: {data.loc[extreme_pe, 'pe_ratio'].tolist()}")
        
        logger.info(f"Fundamental data validation passed for {len(data)} symbols")
        return True
    
    def fetch_earnings_calendar(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch earnings calendar data"""
        if not self.finnhub_api_key:
            logger.warning("Finnhub API key required for earnings calendar")
            return pd.DataFrame()
            
        all_earnings = []
        
        for symbol in symbols:
            try:
                url = f"{self.finnhub_base_url}/calendar/earnings"
                params = {
                    'symbol': symbol,
                    'token': self.finnhub_api_key
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                earnings_data = data.get('earningsCalendar', [])
                
                for earning in earnings_data:
                    earning['symbol'] = symbol
                    all_earnings.append(earning)
                    
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error fetching earnings calendar for {symbol}: {e}")
                continue
                
        return pd.DataFrame(all_earnings) if all_earnings else pd.DataFrame()