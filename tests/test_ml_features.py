"""
Unit tests for ML feature engineering.
"""

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from mech_exo.ml.features import FeatureBuilder


class TestFeatureBuilder(unittest.TestCase):
    """Test cases for FeatureBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "features"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock storage
        self.mock_storage = Mock()
        self.builder = FeatureBuilder(storage=self.mock_storage, output_dir=str(self.output_dir))
        
    def test_feature_builder_init(self):
        """Test FeatureBuilder initialization."""
        self.assertEqual(self.builder.output_dir, self.output_dir)
        self.assertEqual(self.builder.price_lags, [1, 5, 20])
        self.assertEqual(self.builder.volatility_windows, [5, 20])
        self.assertEqual(self.builder.z_score_clip, 5.0)
        
    def test_build_price_features(self):
        """Test price feature engineering."""
        # Create sample OHLC data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        symbols = ['AAPL', 'MSFT']
        
        ohlc_data = []
        for symbol in symbols:
            for i, date in enumerate(dates):
                base_price = 150 if symbol == 'AAPL' else 300
                price = base_price + i * 0.5 + np.random.normal(0, 1)
                ohlc_data.append({
                    'symbol': symbol,
                    'date': date,
                    'close': price,
                    'volume': 1000000 + np.random.randint(-100000, 100000),
                    'atr_20': price * 0.02
                })
                
        ohlc_df = pd.DataFrame(ohlc_data)
        target_date = dates[-1]
        
        # Test price feature building
        price_features = self.builder._build_price_features(ohlc_df, target_date)
        
        # Verify output structure
        self.assertEqual(len(price_features), 2)  # Two symbols
        self.assertTrue('symbol' in price_features.columns)
        self.assertTrue('price' in price_features.columns)
        self.assertTrue('volume' in price_features.columns)
        
        # Verify lag features
        for lag in [1, 5, 20]:
            self.assertTrue(f'return_{lag}d' in price_features.columns)
            
        # Verify volatility features
        for window in [5, 20]:
            self.assertTrue(f'volatility_{window}d' in price_features.columns)
            
        # Verify RSI feature
        self.assertTrue('rsi_14' in price_features.columns)
        
    def test_build_fundamental_features(self):
        """Test fundamental feature engineering."""
        # Create sample fundamental data
        fund_data = [
            {
                'symbol': 'AAPL',
                'date': pd.Timestamp('2024-01-15'),
                'pe_ratio': 25.5,
                'return_on_equity': 0.35,
                'revenue_growth': 0.08,
                'earnings_growth': 0.12
            },
            {
                'symbol': 'MSFT',
                'date': pd.Timestamp('2024-01-15'),
                'pe_ratio': 30.2,
                'return_on_equity': 0.42,
                'revenue_growth': 0.15,
                'earnings_growth': 0.18
            }
        ]
        
        fund_df = pd.DataFrame(fund_data)
        target_date = pd.Timestamp('2024-01-20')
        
        # Test fundamental feature building
        fund_features = self.builder._build_fundamental_features(fund_df, target_date)
        
        # Verify output structure
        self.assertEqual(len(fund_features), 2)  # Two symbols
        self.assertTrue('symbol' in fund_features.columns)
        self.assertTrue('fund_pe_ratio' in fund_features.columns)
        self.assertTrue('fund_return_on_equity' in fund_features.columns)
        
    def test_build_sentiment_features(self):
        """Test sentiment feature engineering."""
        # Create sample news data
        base_date = pd.Timestamp('2024-01-15')
        news_data = []
        
        for i in range(5):  # 5 days of data
            for symbol in ['AAPL', 'MSFT']:
                news_data.append({
                    'symbol': symbol,
                    'date': base_date + pd.Timedelta(days=i),
                    'sentiment_score': np.random.normal(0.5, 0.2),
                    'article_count': np.random.randint(5, 20)
                })
                
        news_df = pd.DataFrame(news_data)
        target_date = base_date + pd.Timedelta(days=5)
        
        # Test sentiment feature building
        sentiment_features = self.builder._build_sentiment_features(news_df, target_date)
        
        # Verify output structure
        self.assertEqual(len(sentiment_features), 2)  # Two symbols
        self.assertTrue('symbol' in sentiment_features.columns)
        self.assertTrue('sent_score_mean' in sentiment_features.columns)
        self.assertTrue('sent_score_std' in sentiment_features.columns)
        self.assertTrue('total_articles' in sentiment_features.columns)
        
    def test_clean_features(self):
        """Test feature cleaning and validation."""
        # Create test data with extreme values
        test_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'feature_1': [1.0, 100.0, 2.0],  # Extreme value
            'feature_2': [0.5, 0.6, np.nan],  # Missing value
            'feature_3': [10.0, 20.0, 15.0]   # Normal values
        })
        
        # Test cleaning
        cleaned_data = self.builder._clean_features(test_data)
        
        # Verify no NaN values remain
        self.assertFalse(cleaned_data.isnull().any().any())
        
        # Verify structure preserved
        self.assertEqual(len(cleaned_data), 3)
        self.assertTrue('symbol' in cleaned_data.columns)
        
    def test_get_feature_summary(self):
        """Test feature summary generation."""
        # Create sample feature data
        feature_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'price': [150.0, 300.0],
            'return_1d': [0.01, -0.005],
            'volatility_20d': [0.25, 0.30],
            'feature_date': ['2024-01-20', '2024-01-20'],
            'feature_count': [10, 10]
        })
        
        # Test summary generation
        summary = self.builder.get_feature_summary(feature_data)
        
        # Verify summary structure
        self.assertEqual(summary['total_symbols'], 2)
        self.assertGreater(summary['total_features'], 0)
        self.assertIsInstance(summary['feature_names'], list)
        self.assertIsInstance(summary['missing_data_pct'], dict)
        self.assertEqual(summary['feature_date'], '2024-01-20')
        
    @patch('mech_exo.ml.features.DataStorage')
    def test_build_features_integration(self, mock_storage_class):
        """Test end-to-end feature building."""
        # Mock storage queries
        mock_storage = mock_storage_class.return_value
        
        # Mock OHLC data
        ohlc_data = pd.DataFrame([
            {'symbol': 'AAPL', 'date': '2024-01-15', 'close': 150.0, 'volume': 1000000, 'atr_20': 3.0},
            {'symbol': 'AAPL', 'date': '2024-01-16', 'close': 151.0, 'volume': 1100000, 'atr_20': 3.1},
            {'symbol': 'MSFT', 'date': '2024-01-15', 'close': 300.0, 'volume': 800000, 'atr_20': 6.0},
            {'symbol': 'MSFT', 'date': '2024-01-16', 'close': 299.0, 'volume': 850000, 'atr_20': 5.9}
        ])
        
        # Mock fundamental data
        fund_data = pd.DataFrame([
            {'symbol': 'AAPL', 'date': '2024-01-10', 'pe_ratio': 25.5, 'return_on_equity': 0.35},
            {'symbol': 'MSFT', 'date': '2024-01-10', 'pe_ratio': 30.2, 'return_on_equity': 0.42}
        ])
        
        # Mock news data
        news_data = pd.DataFrame([
            {'symbol': 'AAPL', 'date': '2024-01-15', 'sentiment_score': 0.6, 'article_count': 10},
            {'symbol': 'MSFT', 'date': '2024-01-15', 'sentiment_score': 0.4, 'article_count': 8}
        ])
        
        mock_storage.query.side_effect = [ohlc_data, fund_data, news_data]
        
        # Create builder with mocked storage
        builder = FeatureBuilder(storage=mock_storage, output_dir=str(self.output_dir))
        
        # Test feature building
        feature_files = builder.build_features('2024-01-16', '2024-01-16', symbols=['AAPL', 'MSFT'])
        
        # Verify files were created
        self.assertTrue(len(feature_files) > 0)
        
        # Verify file exists and contains data
        sample_file = list(feature_files.values())[0]
        self.assertTrue(sample_file.exists())
        
        # Load and verify feature data
        feature_df = pd.read_parquet(sample_file)
        self.assertGreater(len(feature_df), 0)
        self.assertTrue('symbol' in feature_df.columns)
        self.assertTrue('feature_date' in feature_df.columns)


if __name__ == '__main__':
    unittest.main()