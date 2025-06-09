"""
Tests for walk-forward analysis functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from mech_exo.backtest.walk_forward import make_walk_windows, WalkForwardAnalyzer, WalkForwardResults


class TestWalkForwardWindows:
    """Test walk-forward window generation"""
    
    def test_basic_window_generation(self):
        """Test basic window generation logic"""
        windows = make_walk_windows("2020-01-01", "2023-01-01", "12M", "6M")
        
        # Should have multiple windows
        assert len(windows) > 0
        
        # Check window structure
        for window in windows:
            assert len(window) == 4  # train_start, train_end, test_start, test_end
            train_start, train_end, test_start, test_end = window
            
            # Dates should be properly formatted
            assert len(train_start) == 10  # YYYY-MM-DD format
            assert len(test_start) == 10
            
            # Test period should follow training period
            assert pd.to_datetime(test_start) == pd.to_datetime(train_end)
            
            # Test should be 6 months after training ends
            assert pd.to_datetime(test_end) > pd.to_datetime(test_start)
    
    def test_insufficient_data_period(self):
        """Test with insufficient data for any windows"""
        # Very short period that can't accommodate 36M training + 12M test
        windows = make_walk_windows("2020-01-01", "2020-06-01", "36M", "12M")
        
        # Should return empty list
        assert len(windows) == 0
    
    def test_edge_case_exact_fit(self):
        """Test edge case where period exactly fits one window"""
        # 48 months total = 36 months train + 12 months test
        windows = make_walk_windows("2020-01-01", "2024-01-01", "36M", "12M")
        
        # Should have at least one window (might have more due to rolling)
        assert len(windows) >= 1
        
        train_start, train_end, test_start, test_end = windows[0]
        assert train_start == "2020-01-01"
        assert test_end <= "2024-01-01"
    
    def test_different_time_units(self):
        """Test with different time units (days, months, years)"""
        # Test with days
        windows_days = make_walk_windows("2022-01-01", "2022-07-01", "90D", "30D")
        assert len(windows_days) > 0
        
        # Test with months  
        windows_months = make_walk_windows("2020-01-01", "2022-01-01", "12M", "6M")
        assert len(windows_months) > 0


class TestWalkForwardAnalyzer:
    """Test walk-forward analyzer functionality"""
    
    def setup_method(self):
        """Setup test data"""
        # Create mock signal data
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        self.mock_signals = pd.DataFrame({
            'AAPL': [True, False] * (len(dates) // 2) + [True] * (len(dates) % 2),
            'MSFT': [False, True] * (len(dates) // 2) + [False] * (len(dates) % 2)
        }, index=dates)
        
        # Create mock rankings
        np.random.seed(42)
        self.mock_rankings = pd.DataFrame({
            'AAPL': np.random.randn(len(dates)),
            'MSFT': np.random.randn(len(dates)),
            'GOOGL': np.random.randn(len(dates))
        }, index=dates)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        analyzer = WalkForwardAnalyzer(train_period="24M", test_period="6M")
        
        assert analyzer.train_period == "24M"
        assert analyzer.test_period == "6M"
        assert analyzer.results == []
        assert analyzer.combined_equity is None
    
    def test_window_logic_validation(self):
        """Test window generation within analyzer"""
        analyzer = WalkForwardAnalyzer("12M", "6M")
        
        # Test window generation
        windows = make_walk_windows("2020-01-01", "2022-01-01", "12M", "6M")
        
        # Should have at least one window
        assert len(windows) >= 1
        
        # Check that each window has proper structure
        for window in windows:
            train_start, train_end, test_start, test_end = window
            
            # Training period should be ~12 months
            train_duration = pd.to_datetime(train_end) - pd.to_datetime(train_start)
            assert 330 <= train_duration.days <= 400  # ~12 months ± some tolerance
            
            # Test period should be ~6 months
            test_duration = pd.to_datetime(test_end) - pd.to_datetime(test_start)
            assert 150 <= test_duration.days <= 200  # ~6 months ± some tolerance
    
    def test_equity_curve_stitching(self):
        """Test equity curve stitching logic"""
        analyzer = WalkForwardAnalyzer()
        
        # Create mock equity curves for different segments
        dates1 = pd.date_range('2020-01-01', '2020-06-30', freq='D')
        dates2 = pd.date_range('2020-07-01', '2020-12-31', freq='D')
        
        curve1 = pd.Series(100 + np.cumsum(np.random.randn(len(dates1)) * 0.1), index=dates1)
        curve2 = pd.Series(100 + np.cumsum(np.random.randn(len(dates2)) * 0.1), index=dates2)
        
        curves = [curve1, curve2]
        
        # Test stitching
        combined = analyzer._stitch_equity_curves(curves, 100000)
        
        # Should have data
        assert not combined.empty
        
        # Should be continuous (no gaps)
        assert len(combined) > len(curve1)
        
        # Should start at initial cash level
        assert abs(combined.iloc[0] - 100000) < 1000  # Allow small tolerance
    
    def test_metrics_aggregation(self):
        """Test aggregate metrics calculation"""
        analyzer = WalkForwardAnalyzer()
        
        # Mock segment results
        segment_results = [
            {
                'window_id': 1, 'cagr_net': 0.12, 'sharpe_net': 1.5, 'max_drawdown': -0.08,
                'total_trades': 24, 'win_rate': 0.65, 'total_fees': 1200, 'cost_drag_annual': 0.02
            },
            {
                'window_id': 2, 'cagr_net': 0.15, 'sharpe_net': 1.8, 'max_drawdown': -0.06,
                'total_trades': 18, 'win_rate': 0.72, 'total_fees': 900, 'cost_drag_annual': 0.015
            }
        ]
        
        # Mock combined equity
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        combined_equity = pd.Series(100000 + np.cumsum(np.random.randn(len(dates)) * 100), index=dates)
        
        # Test aggregation
        agg_metrics = analyzer._calculate_aggregate_metrics(segment_results, combined_equity)
        
        # Check key aggregate metrics
        assert agg_metrics['total_segments'] == 2
        assert agg_metrics['mean_cagr_net'] == pytest.approx(0.135, rel=1e-3)  # (0.12 + 0.15) / 2
        assert agg_metrics['mean_sharpe_net'] == pytest.approx(1.65, rel=1e-3)  # (1.5 + 1.8) / 2
        assert agg_metrics['total_trades'] == 42  # 24 + 18
        assert agg_metrics['total_fees'] == 2100  # 1200 + 900
        
        # Should have combined metrics
        assert 'combined_total_return' in agg_metrics
        assert 'combined_cagr' in agg_metrics
        assert 'combined_sharpe' in agg_metrics


class TestWalkForwardResults:
    """Test walk-forward results container"""
    
    def test_results_initialization(self):
        """Test results object initialization"""
        segment_results = [{'window_id': 1, 'cagr_net': 0.12}]
        combined_equity = pd.Series([100000, 101000, 102000])
        aggregate_metrics = {'total_segments': 1}
        windows = [('2020-01-01', '2020-12-31', '2021-01-01', '2021-06-30')]
        
        results = WalkForwardResults(
            segment_results=segment_results,
            combined_equity=combined_equity,
            aggregate_metrics=aggregate_metrics,
            windows=windows,
            start_date='2020-01-01',
            end_date='2021-06-30'
        )
        
        assert len(results.segment_results) == 1
        assert len(results.combined_equity) == 3
        assert results.aggregate_metrics['total_segments'] == 1
        assert results.start_date == '2020-01-01'
    
    def test_summary_table_fallback(self):
        """Test summary table with fallback formatting"""
        # Test with no segment results
        results = WalkForwardResults([], pd.Series(), {}, [], '2020-01-01', '2020-12-31')
        
        summary = results._simple_summary_table()
        assert "No walk-forward results available" in summary
        
        # Test with segment results
        segment_results = [
            {'window_id': 1, 'test_start': '2020-01-01', 'test_end': '2020-06-30',
             'cagr_net': 0.12, 'sharpe_net': 1.5, 'max_drawdown': -0.08, 'total_trades': 24}
        ]
        
        results = WalkForwardResults(
            segment_results, pd.Series(), {'total_segments': 1}, [], '2020-01-01', '2020-12-31'
        )
        
        summary = results._simple_summary_table()
        assert "Walk-Forward Analysis Results" in summary
        assert "12.00%" in summary  # CAGR formatting
        assert "1.50" in summary    # Sharpe formatting
    
    def test_chart_data_preparation(self):
        """Test chart data preparation for HTML export"""
        # Create mock data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='M')
        combined_equity = pd.Series(range(100000, 100000 + len(dates) * 1000), index=dates)
        
        segment_results = [
            {'window_id': 1, 'cagr_net': 0.12, 'sharpe_net': 1.5, 'max_drawdown': -0.08},
            {'window_id': 2, 'cagr_net': 0.15, 'sharpe_net': 1.8, 'max_drawdown': -0.06}
        ]
        
        results = WalkForwardResults(
            segment_results, combined_equity, {}, [], '2020-01-01', '2020-12-31'
        )
        
        # Test chart data preparation
        chart_data = results._prepare_walkforward_charts()
        
        # Should have proper structure
        assert 'combined_equity' in chart_data
        assert 'segment_performance' in chart_data
        
        # Equity data should have dates and values
        equity_data = chart_data['combined_equity']
        assert 'dates' in equity_data
        assert 'values' in equity_data
        assert len(equity_data['dates']) == len(combined_equity)
        
        # Performance data should have segment metrics
        perf_data = chart_data['segment_performance']
        assert 'windows' in perf_data
        assert 'cagr' in perf_data
        assert 'sharpe' in perf_data
        assert len(perf_data['windows']) == 2


if __name__ == "__main__":
    pytest.main([__file__])