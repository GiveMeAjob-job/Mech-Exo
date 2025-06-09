"""
Tests for Alpha Decay Engine

Tests the alpha decay monitoring functionality for factor health tracking.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date

from mech_exo.research.alpha_decay import (
    AlphaDecayEngine,
    calc_half_life,
    generate_synthetic_factor_data,
    _spearman_correlation
)


def test_spearman_correlation():
    """Test Spearman correlation calculation"""
    # Test perfect positive correlation
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    corr = _spearman_correlation(x, y)
    assert abs(corr - 1.0) < 0.01
    
    # Test perfect negative correlation
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([10, 8, 6, 4, 2])
    corr = _spearman_correlation(x, y)
    assert abs(corr + 1.0) < 0.01
    
    # Test no correlation
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([3, 1, 4, 2, 5])
    corr = _spearman_correlation(x, y)
    assert abs(corr) < 0.5  # Should be weak correlation


def test_alpha_decay_engine_init():
    """Test AlphaDecayEngine initialization"""
    engine = AlphaDecayEngine()
    assert engine.window == 252
    assert engine.min_periods == 60
    
    # Test with custom parameters
    engine_custom = AlphaDecayEngine(window=100, min_periods=30)
    assert engine_custom.window == 100
    assert engine_custom.min_periods == 30


def test_information_coefficient_calculation():
    """Test IC calculation with synthetic data"""
    engine = AlphaDecayEngine(window=50, min_periods=20)
    
    # Create synthetic data with known correlation
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='B')
    
    factor_values = np.random.normal(0, 1, 100)
    # Returns correlated with factor
    returns_values = factor_values * 0.5 + np.random.normal(0, 0.1, 100)
    
    factor_series = pd.Series(factor_values, index=dates)
    returns_series = pd.Series(returns_values, index=dates)
    
    ic_series = engine.calc_information_coefficient(factor_series, returns_series)
    
    assert len(ic_series) > 0
    assert not ic_series.isna().all()
    # Should show positive correlation
    assert ic_series.mean() > 0


def test_half_life_calculation():
    """Test half-life calculation with synthetic data"""
    np.random.seed(42)
    
    # Generate data with known half-life
    known_half_life = 25.0
    factor_series, returns_series = generate_synthetic_factor_data(
        n_days=400, 
        half_life=known_half_life,
        noise_level=0.05
    )
    
    engine = AlphaDecayEngine(window=252, min_periods=30)
    results = engine.calc_half_life(factor_series, returns_series)
    
    assert results['status'] == 'success'
    assert not np.isnan(results['half_life'])
    assert not np.isnan(results['latest_ic'])
    assert results['ic_observations'] > 0
    
    # Half-life should be reasonably close to known value
    error_pct = abs(results['half_life'] - known_half_life) / known_half_life * 100
    assert error_pct < 100  # Within 100% (conservative for synthetic data)


def test_multiple_factors():
    """Test multiple factor calculation"""
    engine = AlphaDecayEngine(window=150, min_periods=30)
    
    # Create multiple factors with different characteristics
    dates = pd.date_range('2023-01-01', periods=300, freq='B')
    
    factor_data = pd.DataFrame(index=dates)
    
    # Factor 1: Strong initial correlation, slow decay
    np.random.seed(42)
    factor1, returns1 = generate_synthetic_factor_data(n_days=300, half_life=60, noise_level=0.03)
    factor_data['momentum'] = factor1
    
    # Factor 2: Moderate correlation, fast decay
    np.random.seed(43)
    factor2, returns2 = generate_synthetic_factor_data(n_days=300, half_life=15, noise_level=0.05)
    factor_data['mean_reversion'] = factor2
    
    # Use combined returns
    combined_returns = (returns1 + returns2) / 2
    
    results_df = engine.calc_multiple_factors(factor_data, combined_returns)
    
    assert len(results_df) == 2
    assert 'momentum' in results_df['factor_name'].values
    assert 'mean_reversion' in results_df['factor_name'].values
    
    # Check that results contain expected columns
    expected_columns = ['half_life', 'latest_ic', 'factor_name', 'calculation_date']
    for col in expected_columns:
        assert col in results_df.columns


def test_convenience_function():
    """Test the convenience calc_half_life function"""
    np.random.seed(42)
    
    factor_series, returns_series = generate_synthetic_factor_data(
        n_days=200, half_life=20, noise_level=0.08
    )
    
    results = calc_half_life(factor_series, returns_series, window=100)
    
    assert isinstance(results, dict)
    assert 'half_life' in results
    assert 'latest_ic' in results


def test_insufficient_data_handling():
    """Test handling of insufficient data"""
    engine = AlphaDecayEngine(window=100, min_periods=50)
    
    # Create very small dataset
    dates = pd.date_range('2024-01-01', periods=10, freq='B')
    factor_series = pd.Series(np.random.normal(0, 1, 10), index=dates)
    returns_series = pd.Series(np.random.normal(0, 0.01, 10), index=dates)
    
    results = engine.calc_half_life(factor_series, returns_series)
    
    assert results['status'] == 'insufficient_data'
    assert np.isnan(results['half_life'])
    assert np.isnan(results['latest_ic'])


def test_synthetic_data_generation():
    """Test synthetic data generation"""
    np.random.seed(42)
    
    factor_series, returns_series = generate_synthetic_factor_data(
        n_days=100, 
        half_life=30,
        noise_level=0.1
    )
    
    assert len(factor_series) == 100
    assert len(returns_series) == 100
    assert factor_series.index.equals(returns_series.index)
    
    # Check that returns are somewhat correlated with factor initially
    initial_corr = factor_series.head(50).corr(returns_series.head(50))
    assert abs(initial_corr) > 0.1  # Should have some correlation


def test_edge_cases():
    """Test edge cases and error handling"""
    engine = AlphaDecayEngine()
    
    # Test with empty series
    empty_factor = pd.Series(dtype=float)
    empty_returns = pd.Series(dtype=float)
    
    results = engine.calc_half_life(empty_factor, empty_returns)
    assert results['status'] != 'success'
    
    # Test with constant values
    dates = pd.date_range('2024-01-01', periods=50, freq='B')
    constant_factor = pd.Series(1.0, index=dates)
    constant_returns = pd.Series(0.01, index=dates)
    
    results = engine.calc_half_life(constant_factor, constant_returns)
    # Should handle gracefully without crashing
    assert isinstance(results, dict)


if __name__ == "__main__":
    # Run tests manually
    print("🧪 Testing Alpha Decay Engine...")
    
    try:
        test_spearman_correlation()
        print("✅ Spearman correlation test passed")
        
        test_alpha_decay_engine_init()
        print("✅ Engine initialization test passed")
        
        test_information_coefficient_calculation()
        print("✅ IC calculation test passed")
        
        test_half_life_calculation()
        print("✅ Half-life calculation test passed")
        
        test_multiple_factors()
        print("✅ Multiple factors test passed")
        
        test_convenience_function()
        print("✅ Convenience function test passed")
        
        test_insufficient_data_handling()
        print("✅ Insufficient data handling test passed")
        
        test_synthetic_data_generation()
        print("✅ Synthetic data generation test passed")
        
        test_edge_cases()
        print("✅ Edge cases test passed")
        
        print("🎉 All alpha decay tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()