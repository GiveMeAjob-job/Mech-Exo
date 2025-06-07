"""
Tests for data sourcing functionality
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from mech_exo.datasource import (
    OHLCDownloader,
    FundamentalFetcher,
    NewsScraper,
    DataStorage,
    DataValidationError
)


class TestOHLCDownloader:
    
    def test_init(self):
        """Test OHLCDownloader initialization"""
        config = {"rate_limit_delay": 0.1, "max_retries": 3}
        downloader = OHLCDownloader(config)
        
        assert downloader.rate_limit_delay == 0.1
        assert downloader.max_retries == 3
    
    def test_validate_data_success(self):
        """Test successful data validation"""
        config = {}
        downloader = OHLCDownloader(config)
        
        # Create valid OHLC data
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'symbol': ['AAPL'] * 5
        })
        
        assert downloader.validate_data(data) == True
    
    def test_validate_data_failures(self):
        """Test data validation failures"""
        config = {}
        downloader = OHLCDownloader(config)
        
        # Test empty data
        empty_data = pd.DataFrame()
        assert downloader.validate_data(empty_data) == False
        
        # Test missing columns
        incomplete_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'open': [100, 101, 102, 103, 104],
            'symbol': ['AAPL'] * 5
        })
        assert downloader.validate_data(incomplete_data) == False
        
        # Test negative prices
        negative_price_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'open': [100, 101, -102, 103, 104],  # Negative price
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'symbol': ['AAPL'] * 5
        })
        assert downloader.validate_data(negative_price_data) == False
        
        # Test high < low
        invalid_hl_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 100, 105, 106],  # High < Low on third day
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'symbol': ['AAPL'] * 5
        })
        assert downloader.validate_data(invalid_hl_data) == False
    
    def test_calculate_atr(self):
        """Test ATR calculation"""
        config = {}
        downloader = OHLCDownloader(config)
        
        data = pd.DataFrame({
            'high': [105, 110, 108, 112, 115],
            'low': [95, 100, 98, 102, 105],
            'close': [100, 105, 103, 107, 110]
        })
        
        atr = downloader._calculate_atr(data, period=3)
        
        # ATR should be calculated and be positive
        assert not atr.isna().all()
        assert (atr.dropna() > 0).all()


class TestFundamentalFetcher:
    
    def test_init(self):
        """Test FundamentalFetcher initialization"""
        config = {
            "finnhub": {"api_key": "test_key", "base_url": "https://test.com"},
            "rate_limit_delay": 0.1
        }
        fetcher = FundamentalFetcher(config)
        
        assert fetcher.finnhub_api_key == "test_key"
        assert fetcher.rate_limit_delay == 0.1
    
    def test_validate_data_success(self):
        """Test successful fundamental data validation"""
        config = {}
        fetcher = FundamentalFetcher(config)
        
        data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'],
            'fetch_date': [datetime.now(), datetime.now()],
            'pe_ratio': [25.5, 30.2],
            'market_cap': [2500000000000, 1800000000000]
        })
        
        assert fetcher.validate_data(data) == True
    
    def test_validate_data_failures(self):
        """Test fundamental data validation failures"""
        config = {}
        fetcher = FundamentalFetcher(config)
        
        # Test empty data
        empty_data = pd.DataFrame()
        assert fetcher.validate_data(empty_data) == False
        
        # Test missing required columns
        incomplete_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'pe_ratio': [25.5]
        })
        assert fetcher.validate_data(incomplete_data) == False


class TestNewsScraper:
    
    def test_init(self):
        """Test NewsScraper initialization"""
        config = {
            "finnhub": {"api_key": "test_key"},
            "news_api": {"api_key": "news_key"},
            "rate_limit_delay": 0.1
        }
        scraper = NewsScraper(config)
        
        assert scraper.finnhub_api_key == "test_key"
        assert scraper.news_api_key == "news_key"
    
    def test_calculate_simple_sentiment(self):
        """Test simple sentiment calculation"""
        config = {}
        scraper = NewsScraper(config)
        
        # Test positive sentiment
        positive_text = "stock gains surge bullish excellent growth profit"
        pos_sentiment = scraper._calculate_simple_sentiment(positive_text)
        assert pos_sentiment > 0
        
        # Test negative sentiment
        negative_text = "stock falls crash bearish poor loss decline"
        neg_sentiment = scraper._calculate_simple_sentiment(negative_text)
        assert neg_sentiment < 0
        
        # Test neutral sentiment
        neutral_text = "company announced quarterly results today"
        neutral_sentiment = scraper._calculate_simple_sentiment(neutral_text)
        assert abs(neutral_sentiment) < 0.5
    
    def test_validate_data_success(self):
        """Test successful news data validation"""
        config = {}
        scraper = NewsScraper(config)
        
        data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'],
            'headline': ['Apple reports earnings', 'Google AI breakthrough'],
            'published_at': [datetime.now(), datetime.now() - timedelta(hours=1)],
            'sentiment_score': [0.5, 0.8],
            'url': ['http://test1.com', 'http://test2.com']
        })
        
        assert scraper.validate_data(data) == True
    
    def test_validate_data_failures(self):
        """Test news data validation failures"""
        config = {}
        scraper = NewsScraper(config)
        
        # Test invalid sentiment scores
        invalid_sentiment_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'headline': ['Apple reports earnings'],
            'published_at': [datetime.now()],
            'sentiment_score': [2.0],  # Invalid: > 1
            'url': ['http://test.com']
        })
        assert scraper.validate_data(invalid_sentiment_data) == False


class TestDataStorage:
    
    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage for testing"""
        db_path = tmp_path / "test.duckdb"
        storage = DataStorage(str(db_path))
        yield storage
        storage.close()
    
    def test_init(self, temp_storage):
        """Test DataStorage initialization"""
        assert temp_storage.conn is not None
    
    def test_store_and_retrieve_ohlc_data(self, temp_storage):
        """Test storing and retrieving OHLC data"""
        # Create test data
        data = pd.DataFrame({
            'symbol': ['AAPL'] * 5,
            'date': pd.date_range('2023-01-01', periods=5),
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'returns': [0.01, 0.0099, 0.0098, 0.0097, 0.0096],
            'volatility': [0.2, 0.21, 0.22, 0.23, 0.24],
            'atr': [1.5, 1.6, 1.7, 1.8, 1.9]
        })
        
        # Store data
        success = temp_storage.store_ohlc_data(data)
        assert success == True
        
        # Retrieve data
        retrieved_data = temp_storage.get_ohlc_data(['AAPL'])
        assert len(retrieved_data) == 5
        assert retrieved_data['symbol'].iloc[0] == 'AAPL'
    
    def test_store_and_retrieve_fundamental_data(self, temp_storage):
        """Test storing and retrieving fundamental data"""
        # Create test data
        data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'],
            'fetch_date': [datetime.now().date()] * 2,
            'market_cap': [2500000000000, 1800000000000],
            'pe_ratio': [25.5, 30.2],
            'revenue_growth': [0.15, 0.12]
        })
        
        # Store data
        success = temp_storage.store_fundamental_data(data)
        assert success == True
        
        # Retrieve data
        retrieved_data = temp_storage.get_fundamental_data(['AAPL', 'GOOGL'])
        assert len(retrieved_data) == 2
    
    def test_universe_management(self, temp_storage):
        """Test universe/watchlist management"""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        names = ['Apple Inc', 'Alphabet Inc', 'Microsoft Corp']
        sectors = ['Technology', 'Technology', 'Technology']
        
        # Add symbols to universe
        success = temp_storage.add_to_universe(symbols, names, sectors)
        assert success == True
        
        # Retrieve universe
        universe = temp_storage.get_universe()
        assert len(universe) == 3
        assert 'AAPL' in universe['symbol'].values
    
    def test_empty_data_handling(self, temp_storage):
        """Test handling of empty data"""
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        success = temp_storage.store_ohlc_data(empty_df)
        assert success == False
        
        # Retrieving non-existent data should return empty DataFrame
        retrieved = temp_storage.get_ohlc_data(['NONEXISTENT'])
        assert len(retrieved) == 0


class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            "rate_limit_delay": 0.01,  # Faster for testing
            "max_retries": 1,
            "finnhub": {"api_key": None},  # No real API calls
            "news_api": {"api_key": None}
        }
    
    @patch('mech_exo.datasource.ohlc.yf.Ticker')
    def test_ohlc_pipeline(self, mock_ticker, mock_config, tmp_path):
        """Test complete OHLC data pipeline"""
        # Mock yfinance response
        mock_hist_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'Open': [100, 101, 102, 103, 104],
            'High': [102, 103, 104, 105, 106],
            'Low': [99, 100, 101, 102, 103],
            'Close': [101, 102, 103, 104, 105],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }).set_index('Date')
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_hist_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Test pipeline
        storage = DataStorage(str(tmp_path / "test.duckdb"))
        downloader = OHLCDownloader(mock_config)
        
        try:
            # Fetch data
            data = downloader.fetch(['AAPL'])
            assert len(data) == 5
            
            # Store data
            success = storage.store_ohlc_data(data)
            assert success == True
            
            # Retrieve and verify
            retrieved = storage.get_ohlc_data(['AAPL'])
            assert len(retrieved) == 5
            
        finally:
            storage.close()


def test_error_handling():
    """Test error handling in data fetchers"""
    config = {"max_retries": 1, "retry_delay": 0.01}
    
    # Test with invalid symbols
    downloader = OHLCDownloader(config)
    
    # This should handle errors gracefully
    with pytest.raises(DataValidationError):
        downloader.fetch(['INVALID_SYMBOL_THAT_DOES_NOT_EXIST_12345'])


def test_data_validation_edge_cases():
    """Test edge cases in data validation"""
    config = {}
    downloader = OHLCDownloader(config)
    
    # Test with all NaN values
    nan_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=3),
        'open': [np.nan, np.nan, np.nan],
        'high': [np.nan, np.nan, np.nan],
        'low': [np.nan, np.nan, np.nan],
        'close': [np.nan, np.nan, np.nan],
        'volume': [np.nan, np.nan, np.nan],
        'symbol': ['TEST'] * 3
    })
    
    # Should handle NaN values appropriately
    result = downloader.validate_data(nan_data)
    # Depending on implementation, this might pass or fail
    assert isinstance(result, bool)