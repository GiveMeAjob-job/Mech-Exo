"""
Tests for backtesting engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from mech_exo.backtest import Backtester, BacktestResults, create_simple_signals


class TestBacktester:
    """Test backtesting engine functionality"""
    
    def test_backtester_initialization(self):
        """Test backtester can be initialized"""
        start_date = "2020-01-01"
        end_date = "2020-12-31"
        cash = 100000
        
        backtester = Backtester(start=start_date, end=end_date, cash=cash)
        
        assert backtester.start_date == start_date
        assert backtester.end_date == end_date
        assert backtester.initial_cash == cash
        assert backtester.commission == 0.005  # default
        assert backtester.slippage == 0.001    # default
    
    def test_create_simple_signals(self):
        """Test simple signal creation"""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        start = "2020-01-01"
        end = "2020-01-31"
        
        signals = create_simple_signals(symbols, start, end, frequency='weekly')
        
        # Check structure
        assert isinstance(signals, pd.DataFrame)
        assert list(signals.columns) == symbols
        assert signals.index[0] >= pd.to_datetime(start)
        assert signals.index[-1] <= pd.to_datetime(end)
        assert signals.dtypes.all() == bool
        
        # Check weekly frequency (Mondays should be True)
        monday_signals = signals.loc[signals.index.dayofweek == 0]
        assert monday_signals.all().all()  # All Monday signals should be True
    
    def test_simple_signals_monthly(self):
        """Test monthly signal frequency"""
        symbols = ["SPY"]
        start = "2020-01-01"
        end = "2020-12-31"
        
        signals = create_simple_signals(symbols, start, end, frequency='monthly')
        
        # Should have signals on first trading day of each month
        assert signals.any().any()  # At least some signals
        
        # Count monthly signals (should be ~12 for a year)
        monthly_count = signals.sum().sum()
        assert 10 <= monthly_count <= 15  # Allow some flexibility
    
    def test_backtester_with_empty_data(self):
        """Test backtester handles empty data gracefully"""
        # Use dates where we likely have no data
        backtester = Backtester(start="1900-01-01", end="1900-01-31", cash=10000)
        
        # Should initialize but have empty prices
        assert backtester.prices.empty
        
        # Running backtest should raise error
        signals = create_simple_signals(["SPY"], "1900-01-01", "1900-01-31")
        
        with pytest.raises(ValueError, match="No price data available"):
            backtester.run(signals)
    
    def test_backtest_results_structure(self):
        """Test BacktestResults structure"""
        # Create mock results
        mock_metrics = {
            'total_return': 0.15,
            'cagr': 0.12,
            'sharpe': 1.5,
            'max_drawdown': -0.05,
            'win_rate': 0.6,
            'total_trades': 10,
            'start_date': '2020-01-01',
            'end_date': '2020-12-31'
        }
        
        mock_prices = pd.DataFrame({'SPY': [100, 101, 102]}, 
                                  index=pd.date_range('2020-01-01', periods=3))
        mock_signals = pd.DataFrame({'SPY': [True, False, True]}, 
                                   index=pd.date_range('2020-01-01', periods=3))
        
        results = BacktestResults(
            portfolio=None,  # Mock
            metrics=mock_metrics,
            prices=mock_prices,
            signals=mock_signals,
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Test summary generation
        summary = results.summary()
        assert "Backtest Summary" in summary
        assert "15.00%" in summary  # Total return
        assert "12.00%" in summary  # CAGR
        assert "1.50" in summary     # Sharpe
        
        # Test dict export
        result_dict = results.to_dict()
        assert 'metrics' in result_dict
        assert result_dict['metrics']['total_return'] == 0.15


if __name__ == "__main__":
    pytest.main([__file__])