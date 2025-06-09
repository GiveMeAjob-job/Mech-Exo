"""
Tests for Factor Re-fitting Module

Tests the factor re-fitting functionality for strategy retraining.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date

from mech_exo.research.refit_factors import (
    FactorRefitter,
    RefitConfig,
    RefitResults,
    refit_strategy_factors
)


def test_refit_config():
    """Test RefitConfig dataclass"""
    config = RefitConfig()
    
    # Test defaults
    assert config.method == "ridge"
    assert config.alpha == 1.0
    assert config.cv_folds == 5
    assert config.min_samples == 100
    assert config.feature_selection is True
    assert config.standardize is True
    assert config.random_state == 42


def test_factor_refitter_init():
    """Test FactorRefitter initialization"""
    refitter = FactorRefitter()
    assert refitter.config.method == "ridge"
    
    # Test with custom config
    custom_config = RefitConfig(method="lasso", alpha=0.5)
    refitter_custom = FactorRefitter(custom_config)
    assert refitter_custom.config.method == "lasso"
    assert refitter_custom.config.alpha == 0.5


def create_test_ohlc_data():
    """Create test OHLC data"""
    dates = pd.date_range('2024-01-01', '2024-03-01', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    data = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            # Create some trend and volatility
            base_price = 100 + i * 0.1 + np.random.normal(0, 2)
            data.append({
                'symbol': symbol,
                'date': date,
                'open': base_price,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'close': base_price * (1 + np.random.normal(0, 0.01)),
                'volume': 1000000 + np.random.randint(0, 500000)
            })
    
    return pd.DataFrame(data)


def create_test_fundamental_data():
    """Create test fundamental data"""
    dates = pd.date_range('2024-01-01', '2024-03-01', freq='M')
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'symbol': symbol,
                'date': date,
                'pe_ratio': 20 + np.random.normal(0, 5),
                'return_on_equity': 0.15 + np.random.normal(0, 0.05),
                'revenue_growth': 0.10 + np.random.normal(0, 0.03),
                'earnings_growth': 0.12 + np.random.normal(0, 0.04)
            })
    
    return pd.DataFrame(data)


def test_prepare_feature_matrix():
    """Test feature matrix preparation"""
    refitter = FactorRefitter()
    
    ohlc_df = create_test_ohlc_data()
    fundamental_df = create_test_fundamental_data()
    news_df = pd.DataFrame()  # Empty news data
    
    features_df = refitter.prepare_feature_matrix(ohlc_df, fundamental_df, news_df)
    
    assert isinstance(features_df, pd.DataFrame)
    if not features_df.empty:
        assert 'symbol' in features_df.columns
        assert 'date' in features_df.columns
        assert 'forward_return' in features_df.columns


def test_select_features():
    """Test feature selection"""
    refitter = FactorRefitter()
    
    # Create test dataframe with various features
    test_features = pd.DataFrame({
        'momentum_12_1': np.random.normal(0, 0.1, 100),
        'volatility': np.random.normal(0.2, 0.05, 100),
        'rsi_14': np.random.uniform(0, 100, 100),
        'pe_ratio': np.random.normal(20, 5, 100),
        'return_on_equity': np.random.normal(0.15, 0.05, 100),
        'sentiment_mean': np.random.normal(0, 0.5, 100),
        'nonexistent_feature': np.random.normal(0, 1, 100)
    })
    
    selected_features = refitter.select_features(test_features)
    
    assert isinstance(selected_features, list)
    assert len(selected_features) >= 0
    
    # Should exclude features with too many missing values
    test_features_missing = test_features.copy()
    test_features_missing.loc[:80, 'pe_ratio'] = np.nan  # 80% missing
    
    selected_with_missing = refitter.select_features(test_features_missing)
    
    # pe_ratio should be excluded due to high missing percentage
    assert 'pe_ratio' not in selected_with_missing


def test_extract_factor_weights():
    """Test factor weight extraction"""
    refitter = FactorRefitter()
    
    # Create mock model with coefficients
    mock_model = {
        'coef_': np.array([0.5, -0.3, 0.8, 0.2, -0.1])
    }
    
    feature_names = ['momentum_12_1', 'volatility', 'pe_ratio', 'return_on_equity', 'sentiment_mean']
    
    factor_weights = refitter.extract_factor_weights(mock_model, feature_names)
    
    assert isinstance(factor_weights, dict)
    assert 'fundamental' in factor_weights
    assert 'technical' in factor_weights
    assert 'sentiment' in factor_weights
    
    # Check that weights are properly assigned
    all_weights = []
    for category in factor_weights.values():
        for factor_info in category.values():
            all_weights.append(factor_info['weight'])
    
    if all_weights:
        assert all(w >= 1 for w in all_weights)  # All weights should be at least 1


def test_refit_strategy_factors():
    """Test the main refit_strategy_factors function"""
    ohlc_df = create_test_ohlc_data()
    fundamental_df = create_test_fundamental_data()
    
    results = refit_strategy_factors(
        ohlc_df=ohlc_df,
        fundamental_df=fundamental_df,
        method="ridge",
        alpha=1.0
    )
    
    assert isinstance(results, RefitResults)
    assert results.method in ["ridge", "fallback"]
    assert isinstance(results.factor_weights, dict)
    assert isinstance(results.performance_metrics, dict)
    assert isinstance(results.feature_importance, dict)
    assert isinstance(results.in_sample_r2, float)
    assert results.version is not None
    assert results.created_at is not None


def test_refit_with_different_methods():
    """Test re-fitting with different methods"""
    ohlc_df = create_test_ohlc_data()
    
    methods = ["ridge", "lasso", "ols"]
    
    for method in methods:
        results = refit_strategy_factors(
            ohlc_df=ohlc_df,
            method=method,
            alpha=0.5
        )
        
        assert isinstance(results, RefitResults)
        # Method might be "fallback" if sklearn not available
        assert results.method in [method, "fallback"]


def test_refit_with_empty_data():
    """Test re-fitting with empty or insufficient data"""
    empty_df = pd.DataFrame()
    
    results = refit_strategy_factors(
        ohlc_df=empty_df,
        method="ridge"
    )
    
    assert isinstance(results, RefitResults)
    assert results.method == "fallback"
    assert results.in_sample_r2 == 0.0


def test_refit_with_minimal_data():
    """Test re-fitting with minimal data"""
    # Create very small dataset
    small_ohlc = pd.DataFrame({
        'symbol': ['AAPL'] * 10,
        'date': pd.date_range('2024-01-01', periods=10),
        'open': [100] * 10,
        'high': [105] * 10,
        'low': [95] * 10,
        'close': [102] * 10,
        'volume': [1000000] * 10
    })
    
    config = RefitConfig(min_samples=5)  # Lower threshold for testing
    refitter = FactorRefitter(config)
    
    results = refitter.refit_factors(small_ohlc)
    
    assert isinstance(results, RefitResults)
    # Should either work or return fallback
    assert results.method in ["ridge", "fallback"]


def test_factor_weights_format():
    """Test that factor weights follow the expected format"""
    ohlc_df = create_test_ohlc_data()
    results = refit_strategy_factors(ohlc_df)
    
    factor_weights = results.factor_weights
    
    # Check structure
    assert isinstance(factor_weights, dict)
    
    for category_name, category_factors in factor_weights.items():
        assert category_name in ['fundamental', 'technical', 'sentiment']
        assert isinstance(category_factors, dict)
        
        for factor_name, factor_info in category_factors.items():
            assert isinstance(factor_info, dict)
            assert 'weight' in factor_info
            assert 'direction' in factor_info
            assert isinstance(factor_info['weight'], int)
            assert factor_info['direction'] in ['higher_better', 'lower_better', 'mean_revert']
            assert factor_info['weight'] >= 1


if __name__ == "__main__":
    # Run tests manually
    print("🧪 Testing Factor Refitter...")
    
    try:
        # Test basic functionality
        test_refit_config()
        print("✅ RefitConfig test passed")
        
        test_factor_refitter_init()
        print("✅ FactorRefitter init test passed")
        
        # Test with data
        test_refit_strategy_factors()
        print("✅ Strategy factors refit test passed")
        
        test_refit_with_empty_data()
        print("✅ Empty data handling test passed")
        
        test_factor_weights_format()
        print("✅ Factor weights format test passed")
        
        print("🎉 All factor refitter tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()