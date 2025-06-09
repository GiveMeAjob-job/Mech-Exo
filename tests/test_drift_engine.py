"""
Unit tests for drift metric engine
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

from mech_exo.reporting.drift import DriftMetricEngine, calculate_daily_drift, get_drift_status


class TestDriftMetricEngine:
    """Test cases for DriftMetricEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = DriftMetricEngine(initial_cash=100000.0)
        
    def test_init(self):
        """Test engine initialization"""
        assert self.engine.initial_cash == 100000.0
        assert self.engine.storage is not None
        assert self.engine.fill_store is not None
        
    def test_empty_fills_returns_flat_nav(self):
        """Test that empty fills data returns flat NAV series"""
        with patch.object(self.engine.fill_store, 'get_fills_df') as mock_fills:
            # Mock empty DataFrame
            mock_fills.return_value = pd.DataFrame()
            
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 10)
            
            nav_series = self.engine.build_live_nav_series(start_date, end_date)
            
            # Should return flat series at initial cash level
            assert len(nav_series) == 10  # 10 days
            assert all(nav_series == 100000.0)
            assert nav_series.name == 'live_nav'
            
    def test_build_live_nav_with_sample_fills(self):
        """Test NAV calculation with sample fills data"""
        # Create sample fills DataFrame
        fills_data = pd.DataFrame({
            'filled_at': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 2, 11, 0),
                datetime(2024, 1, 3, 14, 0)
            ],
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'quantity': [100, -50, 200],  # Buy 100 AAPL, sell 50 MSFT, buy 200 GOOGL
            'price': [150.0, 300.0, 2500.0],
            'commission': [1.0, 1.5, 5.0]
        })
        
        with patch.object(self.engine.fill_store, 'get_fills_df') as mock_fills:
            mock_fills.return_value = fills_data
            
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 5)
            
            nav_series = self.engine.build_live_nav_series(start_date, end_date)
            
            # Should have 5 days of data
            assert len(nav_series) == 5
            
            # Day 1: Buy 100 AAPL at $150 + $1 commission = -$15,001
            # NAV = $100,000 - $15,001 = $84,999
            assert abs(nav_series.iloc[0] - 84999.0) < 0.01
            
            # Day 2: Sell 50 MSFT at $300 - $1.5 commission = +$14,998.5  
            # NAV = $84,999 + $14,998.5 = $99,997.5
            assert abs(nav_series.iloc[1] - 99997.5) < 0.01
            
            # Day 3: Buy 200 GOOGL at $2500 + $5 commission = -$500,005
            # NAV = $99,997.5 - $500,005 = -$400,007.5
            assert abs(nav_series.iloc[2] - (-400007.5)) < 0.01
            
    def test_no_recent_backtest_returns_none(self):
        """Test that no recent backtest returns None"""
        with patch.object(self.engine.storage.conn, 'execute') as mock_execute:
            # Mock no results
            mock_result = Mock()
            mock_result.fetchone.return_value = None
            mock_execute.return_value = mock_result
            
            backtest_nav = self.engine.get_latest_backtest_nav(lookback_days=30)
            assert backtest_nav is None
            
    def test_backtest_nav_generation(self):
        """Test backtest NAV series generation from stored metrics"""
        # Mock backtest result
        backtest_result = (
            datetime(2024, 1, 1),  # backtest_date
            date(2023, 12, 1),     # period_start
            date(2024, 1, 1),      # period_end
            100000.0,              # initial_cash
            0.15,                  # total_return_net (15%)
            0.12                   # cagr_net (12%)
        )
        
        with patch.object(self.engine.storage.conn, 'execute') as mock_execute:
            mock_result = Mock()
            mock_result.fetchone.return_value = backtest_result
            mock_execute.return_value = mock_result
            
            backtest_nav = self.engine.get_latest_backtest_nav(lookback_days=30)
            
            assert backtest_nav is not None
            assert backtest_nav.name == 'backtest_nav'
            assert len(backtest_nav) == 32  # Dec 1 to Jan 1 inclusive
            assert backtest_nav.iloc[0] == 100000.0  # Starts at initial cash
            assert backtest_nav.iloc[-1] > 100000.0  # Should grow with positive CAGR
            
    def test_calculate_drift_metrics_no_backtest(self):
        """Test drift calculation when no backtest is available"""
        with patch.object(self.engine, 'get_latest_backtest_nav') as mock_backtest:
            mock_backtest.return_value = None
            
            with patch.object(self.engine, 'build_live_nav_series') as mock_live:
                mock_live.return_value = pd.Series([100000, 101000, 102000])
                
                start_date = date(2024, 1, 1)
                end_date = date(2024, 1, 3)
                
                metrics = self.engine.calculate_drift_metrics(start_date, end_date)
                
                assert metrics['drift_pct'] == 0.0
                assert metrics['information_ratio'] == 0.0
                assert metrics['data_quality'] == 'no_backtest'
                assert metrics['days_analyzed'] == 0
                
    def test_calculate_drift_metrics_with_data(self):
        """Test drift calculation with live and backtest data"""
        # Create sample live NAV (growing faster than backtest)
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        live_nav = pd.Series([100000 * (1.001 ** i) for i in range(10)], index=dates, name='live_nav')
        backtest_nav = pd.Series([100000 * (1.0005 ** i) for i in range(10)], index=dates, name='backtest_nav')
        
        with patch.object(self.engine, 'build_live_nav_series') as mock_live:
            mock_live.return_value = live_nav
            
            with patch.object(self.engine, 'get_latest_backtest_nav') as mock_backtest:
                mock_backtest.return_value = backtest_nav
                
                start_date = date(2024, 1, 1) 
                end_date = date(2024, 1, 10)
                
                metrics = self.engine.calculate_drift_metrics(start_date, end_date)
                
                # Live should outperform backtest
                assert metrics['live_cagr'] > metrics['backtest_cagr']
                assert metrics['drift_pct'] > 0  # Positive drift (outperformance)
                assert metrics['data_quality'] in ['good', 'fair', 'poor']
                assert metrics['days_analyzed'] > 0
                assert isinstance(metrics['information_ratio'], float)


class TestDriftUtilityFunctions:
    """Test utility functions for drift analysis"""
    
    def test_calculate_daily_drift_default_date(self):
        """Test daily drift calculation with default date"""
        with patch('mech_exo.reporting.drift.DriftMetricEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine.calculate_drift_metrics.return_value = {
                'date': '2024-01-15',
                'drift_pct': 5.0,
                'information_ratio': 1.2
            }
            mock_engine_class.return_value = mock_engine
            
            result = calculate_daily_drift()
            
            assert result['drift_pct'] == 5.0
            assert result['information_ratio'] == 1.2
            
    def test_get_drift_status_ok(self):
        """Test drift status determination - OK case"""
        status = get_drift_status(drift_pct=5.0, information_ratio=1.5)
        assert status == 'OK'
        
    def test_get_drift_status_warn(self):
        """Test drift status determination - WARN case"""
        # High drift
        status = get_drift_status(drift_pct=15.0, information_ratio=1.0)
        assert status == 'WARN'
        
        # Low IR
        status = get_drift_status(drift_pct=5.0, information_ratio=0.1)
        assert status == 'WARN'
        
    def test_get_drift_status_breach(self):
        """Test drift status determination - BREACH case"""
        # Very high drift
        status = get_drift_status(drift_pct=25.0, information_ratio=1.0)
        assert status == 'BREACH'
        
        # Negative IR
        status = get_drift_status(drift_pct=5.0, information_ratio=-0.5)
        assert status == 'BREACH'
        
    def test_drift_status_absolute_value(self):
        """Test that drift status uses absolute value of drift percentage"""
        # Negative drift should be treated same as positive
        status = get_drift_status(drift_pct=-15.0, information_ratio=1.0)
        assert status == 'WARN'
        
        status = get_drift_status(drift_pct=-25.0, information_ratio=1.0)
        assert status == 'BREACH'


@pytest.fixture
def sample_engine():
    """Create a sample drift engine for testing"""
    return DriftMetricEngine(initial_cash=100000.0)


def test_engine_creation(sample_engine):
    """Test that engine can be created successfully"""
    assert sample_engine.initial_cash == 100000.0
    assert hasattr(sample_engine, 'storage')
    assert hasattr(sample_engine, 'fill_store')


def test_edge_cases():
    """Test edge cases and error handling"""
    engine = DriftMetricEngine()
    
    # Test with same start and end date
    start_date = end_date = date(2024, 1, 1)
    
    with patch.object(engine.fill_store, 'get_fills_df') as mock_fills:
        mock_fills.return_value = pd.DataFrame()
        
        nav_series = engine.build_live_nav_series(start_date, end_date)
        assert len(nav_series) == 1
        assert nav_series.iloc[0] == engine.initial_cash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])