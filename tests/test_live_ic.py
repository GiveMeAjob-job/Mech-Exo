"""
Unit tests for live Information Coefficient calculation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

from mech_exo.reporting.query import get_live_ic


class TestLiveIC:
    """Test cases for live IC calculation functionality"""

    def test_get_live_ic_with_mock_data(self):
        """Test live IC calculation with mocked database data"""
        
        # Create mock data with known correlation
        mock_data = pd.DataFrame({
            'symbol': ['AAPL'] * 20 + ['MSFT'] * 20,
            'ml_score': np.concatenate([
                np.linspace(0.1, 0.9, 20),  # AAPL scores ascending
                np.linspace(0.2, 0.8, 20)   # MSFT scores ascending
            ]),
            'next_day_return': np.concatenate([
                np.linspace(-0.02, 0.03, 20),  # AAPL returns ascending (positive correlation)
                np.linspace(-0.01, 0.02, 20)   # MSFT returns ascending (positive correlation)
            ]),
            'prediction_date': [date.today() - timedelta(days=i//2) for i in range(40)]
        })
        
        with patch('mech_exo.reporting.query.DataStorage') as mock_storage_class:
            # Mock the storage instance and connection
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            # Mock the SQL query to return our test data
            with patch('pandas.read_sql_query') as mock_read_sql:
                mock_read_sql.return_value = mock_data
                
                # Call the function
                ic = get_live_ic(lookback_days=30)
                
                # Assert the result
                assert isinstance(ic, float)
                assert ic > 0.0  # Should be positive due to positive correlation in test data
                assert ic <= 1.0  # IC should not exceed 1.0
                
                # Verify the storage was closed
                mock_storage.close.assert_called_once()

    def test_get_live_ic_insufficient_data(self):
        """Test live IC calculation with insufficient data"""
        
        # Create minimal mock data (less than 10 observations)
        mock_data = pd.DataFrame({
            'symbol': ['AAPL'] * 5,
            'ml_score': [0.1, 0.3, 0.5, 0.7, 0.9],
            'next_day_return': [0.01, -0.01, 0.02, -0.01, 0.01],
            'prediction_date': [date.today() - timedelta(days=i) for i in range(5)]
        })
        
        with patch('mech_exo.reporting.query.DataStorage') as mock_storage_class:
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            with patch('pandas.read_sql_query') as mock_read_sql:
                mock_read_sql.return_value = mock_data
                
                # Call the function
                ic = get_live_ic(lookback_days=30)
                
                # Should return 0.0 due to insufficient data
                assert ic == 0.0

    def test_get_live_ic_empty_data(self):
        """Test live IC calculation with empty data"""
        
        # Empty DataFrame
        mock_data = pd.DataFrame(columns=['symbol', 'ml_score', 'next_day_return', 'prediction_date'])
        
        with patch('mech_exo.reporting.query.DataStorage') as mock_storage_class:
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            with patch('pandas.read_sql_query') as mock_read_sql:
                mock_read_sql.return_value = mock_data
                
                # Call the function
                ic = get_live_ic(lookback_days=30)
                
                # Should return 0.0 due to empty data
                assert ic == 0.0

    def test_get_live_ic_database_error(self):
        """Test live IC calculation with database error"""
        
        with patch('mech_exo.reporting.query.DataStorage') as mock_storage_class:
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            with patch('pandas.read_sql_query') as mock_read_sql:
                # Simulate database error
                mock_read_sql.side_effect = Exception("no such table: ml_scores")
                
                # Call the function
                ic = get_live_ic(lookback_days=30)
                
                # Should return 0.0 due to database error
                assert ic == 0.0

    def test_get_live_ic_nan_handling(self):
        """Test live IC calculation with NaN values in data"""
        
        # Create mock data with some NaN values
        mock_data = pd.DataFrame({
            'symbol': ['AAPL'] * 15,
            'ml_score': [0.1, 0.2, np.nan, 0.4, 0.5, 0.6, np.nan, 0.8, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6],
            'next_day_return': [0.01, np.nan, 0.02, -0.01, 0.01, 0.03, -0.02, np.nan, 0.01, -0.01, 0.02, 0.01, -0.01, 0.02, 0.01],
            'prediction_date': [date.today() - timedelta(days=i) for i in range(15)]
        })
        
        with patch('mech_exo.reporting.query.DataStorage') as mock_storage_class:
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            with patch('pandas.read_sql_query') as mock_read_sql:
                mock_read_sql.return_value = mock_data
                
                # Call the function
                ic = get_live_ic(lookback_days=30)
                
                # Should handle NaN values and return a valid IC
                assert isinstance(ic, float)
                assert not pd.isna(ic)

    def test_get_live_ic_default_lookback(self):
        """Test live IC calculation with default lookback period"""
        
        # Create sufficient mock data
        mock_data = pd.DataFrame({
            'symbol': ['AAPL'] * 15,
            'ml_score': np.random.uniform(0.1, 0.9, 15),
            'next_day_return': np.random.normal(0, 0.02, 15),
            'prediction_date': [date.today() - timedelta(days=i) for i in range(15)]
        })
        
        with patch('mech_exo.reporting.query.DataStorage') as mock_storage_class:
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            
            with patch('pandas.read_sql_query') as mock_read_sql:
                mock_read_sql.return_value = mock_data
                
                # Call the function with default parameters
                ic = get_live_ic()  # Should use 30 days default
                
                # Should return a valid IC
                assert isinstance(ic, float)
                assert -1.0 <= ic <= 1.0  # IC should be in valid range


if __name__ == "__main__":
    pytest.main([__file__])