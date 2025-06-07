"""
Tests for scoring functionality
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from mech_exo.scoring import (
    IdeaScorer,
    FactorFactory,
    BaseFactor,
    FactorCalculationError,
    ScoringError
)
from mech_exo.scoring.factors import (
    PERatioFactor,
    ROEFactor,
    RSIFactor,
    MomentumFactor
)


class TestBaseFactor:
    
    def test_normalize_zscore(self):
        """Test z-score normalization"""
        factor = PERatioFactor({'weight': 10})
        values = pd.Series([10, 20, 30, 40, 50])
        
        normalized = factor.normalize(values, method="zscore")
        
        # Z-score should have mean ~0 and std ~1
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1) < 0.1
    
    def test_normalize_rank(self):
        """Test rank normalization"""
        factor = PERatioFactor({'weight': 10})
        values = pd.Series([10, 20, 30, 40, 50])
        
        normalized = factor.normalize(values, method="rank")
        
        # Rank normalization should be between 0 and 1
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_apply_direction(self):
        """Test direction application"""
        # Test higher_better
        factor = ROEFactor({'weight': 10, 'direction': 'higher_better'})
        values = pd.Series([0.1, 0.2, 0.3])
        result = factor.apply_direction(values)
        assert (result == values).all()
        
        # Test lower_better
        factor = PERatioFactor({'weight': 10, 'direction': 'lower_better'})
        result = factor.apply_direction(values)
        assert (result == -values).all()
        
        # Test mean_revert
        factor = RSIFactor({'weight': 10, 'direction': 'mean_revert'})
        result = factor.apply_direction(values)
        assert (result == -np.abs(values)).all()


class TestPERatioFactor:
    
    def test_calculate_success(self):
        """Test successful P/E ratio calculation"""
        factor = PERatioFactor({'weight': 15})
        
        data = pd.DataFrame({
            'pe_ratio': [15.5, 25.0, 30.2, 12.8, 18.9]
        })
        
        result = factor.calculate(data)
        assert len(result) == 5
        assert result.min() >= 0
        assert result.max() < 100  # Should filter extreme values
    
    def test_calculate_missing_column(self):
        """Test calculation with missing column"""
        factor = PERatioFactor({'weight': 15})
        
        data = pd.DataFrame({
            'other_column': [1, 2, 3, 4, 5]
        })
        
        with pytest.raises(FactorCalculationError):
            factor.calculate(data)
    
    def test_extreme_value_filtering(self):
        """Test filtering of extreme P/E values"""
        factor = PERatioFactor({'weight': 15})
        
        data = pd.DataFrame({
            'pe_ratio': [15.5, 150.0, -5.0, 12.8, 200.0]  # Extreme values
        })
        
        result = factor.calculate(data)
        
        # Should filter out negative and very high values
        assert result.min() >= 0
        assert result.max() < 100


class TestROEFactor:
    
    def test_calculate_percentage_conversion(self):
        """Test ROE calculation with percentage conversion"""
        factor = ROEFactor({'weight': 18})
        
        # Test decimal format (should be converted to percentage)
        data = pd.DataFrame({
            'return_on_equity': [0.15, 0.20, 0.12, 0.25, 0.08]
        })
        
        result = factor.calculate(data)
        
        # Should be converted to percentage
        assert result.min() > 1  # Should be > 1 after conversion
        assert result.max() < 100


class TestRSIFactor:
    
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        factor = RSIFactor({'weight': 8})
        
        # Create sample price data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = 100 + np.random.randn(50).cumsum() * 0.5
        
        data = pd.DataFrame({
            'symbol': ['AAPL'] * 50,
            'date': dates,
            'close': prices
        })
        
        result = factor.calculate(data)
        
        # RSI should be between 0 and 100
        if not result.empty:
            assert result.min() >= 0
            assert result.max() <= 100
    
    def test_rsi_calculation_formula(self):
        """Test RSI calculation formula directly"""
        factor = RSIFactor({'weight': 8})
        
        # Create test prices with known pattern
        prices = pd.Series([100, 101, 102, 101, 100, 99, 100, 101, 102, 103,
                           104, 103, 102, 103, 104, 105, 104, 103, 104, 105])
        
        rsi = factor._calculate_rsi(prices)
        
        # RSI should be calculated for periods after the lookback
        assert len(rsi.dropna()) >= 1
        assert rsi.dropna().min() >= 0
        assert rsi.dropna().max() <= 100


class TestMomentumFactor:
    
    def test_calculate_momentum(self):
        """Test momentum calculation"""
        factor = MomentumFactor({'weight': 12})
        
        # Create sample data with enough history
        dates = pd.date_range('2022-01-01', periods=300, freq='D')
        base_price = 100
        prices = [base_price]
        
        # Generate trending price series
        for i in range(299):
            change = np.random.normal(0.001, 0.02)  # Small upward drift
            prices.append(prices[-1] * (1 + change))
        
        data = pd.DataFrame({
            'symbol': ['AAPL'] * 300,
            'date': dates,
            'close': prices
        })
        
        result = factor.calculate(data)
        
        # Should calculate momentum
        if not result.empty:
            assert isinstance(result.iloc[0], (int, float, np.number))


class TestFactorFactory:
    
    def test_create_factor(self):
        """Test factor creation"""
        config = {'weight': 15, 'direction': 'lower_better'}
        
        factor = FactorFactory.create_factor('pe_ratio', config)
        
        assert isinstance(factor, PERatioFactor)
        assert factor.weight == 15
        assert factor.direction == 'lower_better'
    
    def test_create_unknown_factor(self):
        """Test creation of unknown factor"""
        config = {'weight': 10}
        
        with pytest.raises(ValueError):
            FactorFactory.create_factor('unknown_factor', config)
    
    def test_create_all_factors(self):
        """Test creating all factors from config"""
        factor_config = {
            'pe_ratio': {'weight': 15, 'direction': 'lower_better'},
            'return_on_equity': {'weight': 18, 'direction': 'higher_better'},
            'unknown_factor': {'weight': 10}  # Should be skipped
        }
        
        factors = FactorFactory.create_all_factors(factor_config)
        
        assert len(factors) == 2  # Should skip unknown factor
        assert 'pe_ratio' in factors
        assert 'return_on_equity' in factors
        assert 'unknown_factor' not in factors


class TestIdeaScorer:
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'fundamental': {
                'pe_ratio': {'weight': 15, 'direction': 'lower_better'},
                'return_on_equity': {'weight': 18, 'direction': 'higher_better'}
            },
            'technical': {
                'rsi_14': {'weight': 8, 'direction': 'mean_revert'}
            },
            'sector_adjustments': {
                'Technology': 1.1,
                'Healthcare': 1.0
            }
        }
    
    @patch('mech_exo.scoring.scorer.ConfigManager')
    @patch('mech_exo.scoring.scorer.DataStorage')
    def test_initialization(self, mock_storage, mock_config_manager, mock_config):
        """Test scorer initialization"""
        # Mock config manager
        mock_config_instance = Mock()
        mock_config_instance.get_factor_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        # Mock storage
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        scorer = IdeaScorer()
        
        assert len(scorer.factors) >= 2  # Should have at least pe_ratio and roe
        assert 'pe_ratio' in scorer.factors
        assert 'return_on_equity' in scorer.factors
    
    @patch('mech_exo.scoring.scorer.ConfigManager')
    @patch('mech_exo.scoring.scorer.DataStorage')
    def test_prepare_data(self, mock_storage, mock_config_manager, mock_config):
        """Test data preparation"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.get_factor_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        mock_storage_instance = Mock()
        
        # Mock data returns
        fundamental_data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'],
            'pe_ratio': [25.5, 30.2],
            'return_on_equity': [0.15, 0.20],
            'fetch_date': [datetime.now(), datetime.now()]
        })
        
        ohlc_data = pd.DataFrame({
            'symbol': ['AAPL'] * 5 + ['GOOGL'] * 5,
            'date': pd.date_range('2023-01-01', periods=5).tolist() * 2,
            'close': [100, 101, 102, 103, 104, 200, 201, 202, 203, 204],
            'returns': [0.01] * 10
        })
        
        mock_storage_instance.get_fundamental_data.return_value = fundamental_data
        mock_storage_instance.get_ohlc_data.return_value = ohlc_data
        mock_storage_instance.get_news_data.return_value = pd.DataFrame()
        
        mock_storage.return_value = mock_storage_instance
        
        scorer = IdeaScorer()
        data = scorer._prepare_data(['AAPL', 'GOOGL'])
        
        assert not data.empty
        assert 'symbol' in data.columns
        assert len(data) >= 2
    
    def test_combine_factor_scores(self):
        """Test factor score combination"""
        # Create mock scorer with simplified factors
        scorer = Mock()
        scorer.factors = {
            'factor1': Mock(weight=30),
            'factor2': Mock(weight=70)
        }
        
        # Create test scores
        factor_scores = {
            'factor1': pd.Series([0.5, 0.8, 0.2], index=[0, 1, 2]),
            'factor2': pd.Series([0.3, 0.6, 0.9], index=[0, 1, 2])
        }
        
        # Use actual method
        from mech_exo.scoring.scorer import IdeaScorer
        composite = IdeaScorer._combine_factor_scores(scorer, factor_scores)
        
        assert len(composite) == 3
        assert isinstance(composite, pd.Series)
        # Should be weighted average
        expected_0 = (0.5 * 0.3 + 0.3 * 0.7) / 1.0  # Normalized weights
        assert abs(composite.iloc[0] - expected_0) < 0.01
    
    def test_apply_adjustments(self):
        """Test sector and market adjustments"""
        scorer = Mock()
        scorer.sector_adjustments = {'Technology': 1.1}
        scorer.market_regime_config = {'bull_market_multiplier': 1.2}
        
        scores = pd.Series([1.0, 2.0, 3.0])
        data = pd.DataFrame({
            'sector': ['Technology', 'Healthcare', 'Technology']
        })
        
        from mech_exo.scoring.scorer import IdeaScorer
        adjusted = IdeaScorer._apply_adjustments(scorer, scores, data)
        
        # Should apply sector adjustment to Technology stocks
        assert adjusted.iloc[0] > scores.iloc[0]  # Technology adjustment
        assert adjusted.iloc[2] > scores.iloc[2]  # Technology adjustment
        
        # All should have market regime adjustment
        assert all(adjusted > scores)  # Bull market multiplier


class TestScoringIntegration:
    """Integration tests for complete scoring pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for integration testing"""
        return {
            'fundamental': pd.DataFrame({
                'symbol': ['AAPL', 'GOOGL', 'MSFT'],
                'pe_ratio': [25.5, 30.2, 28.1],
                'return_on_equity': [0.15, 0.20, 0.18],
                'revenue_growth': [0.08, 0.12, 0.10],
                'sector': ['Technology', 'Technology', 'Technology'],
                'fetch_date': [datetime.now()] * 3
            }),
            'ohlc': pd.DataFrame({
                'symbol': ['AAPL'] * 20 + ['GOOGL'] * 20 + ['MSFT'] * 20,
                'date': pd.date_range('2023-01-01', periods=20).tolist() * 3,
                'close': (100 + np.random.randn(20).cumsum()).tolist() + 
                        (200 + np.random.randn(20).cumsum()).tolist() + 
                        (300 + np.random.randn(20).cumsum()).tolist(),
                'returns': np.random.normal(0.001, 0.02, 60).tolist()
            })
        }
    
    @patch('mech_exo.scoring.scorer.DataStorage')
    @patch('mech_exo.scoring.scorer.ConfigManager')
    def test_end_to_end_scoring(self, mock_config_manager, mock_storage, sample_data):
        """Test complete scoring pipeline"""
        # Setup configuration
        factor_config = {
            'fundamental': {
                'pe_ratio': {'weight': 30, 'direction': 'lower_better'},
                'return_on_equity': {'weight': 70, 'direction': 'higher_better'}
            },
            'sector_adjustments': {'Technology': 1.1}
        }
        
        mock_config_instance = Mock()
        mock_config_instance.get_factor_config.return_value = factor_config
        mock_config_manager.return_value = mock_config_instance
        
        # Setup storage
        mock_storage_instance = Mock()
        mock_storage_instance.get_fundamental_data.return_value = sample_data['fundamental']
        mock_storage_instance.get_ohlc_data.return_value = sample_data['ohlc']
        mock_storage_instance.get_news_data.return_value = pd.DataFrame()
        mock_storage.return_value = mock_storage_instance
        
        # Test scoring
        scorer = IdeaScorer()
        ranking = scorer.score(['AAPL', 'GOOGL', 'MSFT'])
        
        # Verify results
        assert not ranking.empty
        assert len(ranking) == 3
        assert 'rank' in ranking.columns
        assert 'composite_score' in ranking.columns
        assert 'symbol' in ranking.columns
        
        # Verify ranking order
        assert ranking['rank'].min() == 1
        assert ranking['rank'].max() == 3
        
        # Verify scores are reasonable
        assert ranking['composite_score'].min() >= -3  # Should be reasonable range
        assert ranking['composite_score'].max() <= 3


def test_scoring_with_missing_data():
    """Test scoring behavior with missing data"""
    factor = PERatioFactor({'weight': 15})
    
    # Test with some missing values
    data = pd.DataFrame({
        'pe_ratio': [15.5, np.nan, 30.2, np.nan, 18.9]
    })
    
    result = factor.calculate(data)
    
    # Should handle missing values appropriately
    assert not result.empty
    assert result.notna().sum() >= 1  # At least some valid values


def test_error_handling():
    """Test error handling in scoring components"""
    # Test with completely invalid data
    factor = PERatioFactor({'weight': 15})
    
    with pytest.raises(FactorCalculationError):
        factor.calculate(pd.DataFrame({'wrong_column': [1, 2, 3]}))
    
    # Test factory with invalid factor
    with pytest.raises(ValueError):
        FactorFactory.create_factor('nonexistent_factor', {'weight': 10})