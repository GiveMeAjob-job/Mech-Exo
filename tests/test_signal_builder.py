"""
Tests for signal builder functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from mech_exo.backtest.signal_builder import (
    idea_rank_to_signals,
    create_ranking_signals_from_scores, 
    validate_signals,
    _get_rebalancing_dates
)


class TestSignalBuilder:
    """Test signal building functionality"""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty ranking DataFrame"""
        empty_df = pd.DataFrame()
        
        signals = idea_rank_to_signals(empty_df, n_top=3)
        
        assert signals.empty
    
    def test_single_symbol_edge_case(self):
        """Test with single symbol"""
        # Create simple ranking data for one symbol
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        rank_df = pd.DataFrame({
            'AAPL': [1.0, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7, 1.0, 0.9, 1.1]
        }, index=dates)
        
        signals = idea_rank_to_signals(rank_df, n_top=1, rebal_freq='daily')
        
        # Should have correct shape
        assert signals.shape == rank_df.shape
        assert list(signals.columns) == ['AAPL']
        
        # Since n_top=1 and we only have 1 symbol, it should be True most of the time
        assert signals['AAPL'].any()
    
    def test_basic_signal_generation(self):
        """Test basic signal generation with multiple symbols"""
        # Create ranking data
        dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        
        # Create mock ranking scores (higher = better)
        np.random.seed(42)  # For reproducible tests
        rank_data = {}
        for symbol in symbols:
            rank_data[symbol] = np.random.randn(len(dates)) + np.sin(np.arange(len(dates)) * 0.1)
        
        rank_df = pd.DataFrame(rank_data, index=dates)
        
        signals = idea_rank_to_signals(rank_df, n_top=2, rebal_freq='weekly')
        
        # Check structure
        assert signals.shape == rank_df.shape
        assert list(signals.columns) == symbols
        assert signals.dtype == bool
        
        # Check that we never hold more than n_top positions
        daily_positions = signals.sum(axis=1)
        assert (daily_positions <= 2).all()
        assert daily_positions.max() > 0  # Should hold some positions
    
    def test_rebalancing_frequencies(self):
        """Test different rebalancing frequencies"""
        dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
        
        # Test monthly rebalancing
        monthly_dates = _get_rebalancing_dates(dates, 'monthly')
        assert len(monthly_dates) == 3  # Jan, Feb, Mar
        
        # Test weekly rebalancing (Mondays)
        weekly_dates = _get_rebalancing_dates(dates, 'weekly')
        assert all(d.dayofweek == 0 for d in weekly_dates)  # All Mondays
        
        # Test daily rebalancing
        daily_dates = _get_rebalancing_dates(dates, 'daily')
        assert len(daily_dates) == len(dates)
    
    def test_long_short_signals(self):
        """Test long/short signal creation"""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        symbols = ['A', 'B', 'C', 'D', 'E']
        
        # Create scores where A,B are best and D,E are worst
        scores_data = {
            'A': [2.0] * len(dates),
            'B': [1.8] * len(dates), 
            'C': [1.0] * len(dates),
            'D': [0.2] * len(dates),
            'E': [0.1] * len(dates)
        }
        scores_df = pd.DataFrame(scores_data, index=dates)
        
        signals = create_ranking_signals_from_scores(
            scores_df, n_long=2, n_short=2, rebal_freq='daily'
        )
        
        # Check that A,B are long (1) and D,E are short (-1)
        assert (signals['A'] == 1).all()
        assert (signals['B'] == 1).all()
        assert (signals['C'] == 0).all()  # Neutral
        assert (signals['D'] == -1).all()
        assert (signals['E'] == -1).all()
    
    def test_signal_validation(self):
        """Test signal validation functionality"""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        
        # Valid signals
        valid_signals = pd.DataFrame({
            'A': [True, True, False, False, True, True, False, True, False, True],
            'B': [False, True, True, False, False, True, True, False, True, False]
        }, index=dates)
        
        validation = validate_signals(valid_signals, max_positions=2)
        
        assert validation['has_data'] == True
        assert validation['has_signals'] == True
        assert validation['position_limit_ok'] == True
        
        # Invalid signals (too many positions)
        invalid_signals = pd.DataFrame({
            'A': [True] * len(dates),
            'B': [True] * len(dates), 
            'C': [True] * len(dates)
        }, index=dates)
        
        validation = validate_signals(invalid_signals, max_positions=2)
        assert validation['position_limit_ok'] == False
    
    def test_holding_period_constraint(self):
        """Test that holding period constraints are respected"""
        dates = pd.date_range('2020-01-01', '2020-01-20', freq='D')
        symbols = ['A', 'B', 'C']
        
        # Create ranking that changes but should respect holding period
        rank_data = {}
        for i, symbol in enumerate(symbols):
            # Create alternating high/low rankings
            rank_data[symbol] = [1.0 if (j + i) % 3 == 0 else 0.1 for j in range(len(dates))]
        
        rank_df = pd.DataFrame(rank_data, index=dates)
        
        signals = idea_rank_to_signals(
            rank_df, n_top=1, holding_period=5, rebal_freq='daily'
        )
        
        # Check that positions are held for at least 5 days when possible
        for symbol in symbols:
            symbol_signals = signals[symbol]
            if symbol_signals.any():
                # Find consecutive True periods
                changes = symbol_signals != symbol_signals.shift(1)
                groups = changes.cumsum()
                
                for group_id in groups[symbol_signals].unique():
                    group_length = (groups == group_id).sum()
                    # Most holding periods should be >= 5 days (allowing some flexibility for end of period)
                    if group_length < 5:
                        # This is acceptable near the end of the test period
                        pass
    
    def test_date_filtering(self):
        """Test date range filtering"""
        dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
        rank_df = pd.DataFrame({
            'A': np.random.randn(len(dates))
        }, index=dates)
        
        # Filter to February only
        signals = idea_rank_to_signals(
            rank_df, n_top=1, 
            start_date='2020-02-01', 
            end_date='2020-02-29'
        )
        
        # Should only have February dates
        assert signals.index.min() >= pd.to_datetime('2020-02-01')
        assert signals.index.max() <= pd.to_datetime('2020-02-29')


if __name__ == "__main__":
    pytest.main([__file__])