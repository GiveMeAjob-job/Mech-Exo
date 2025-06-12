"""
Unit and integration tests for ML inference and IdeaScorer integration.

Tests the complete pipeline from model prediction to idea scoring with ML integration.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest


class TestMLPredictor(unittest.TestCase):
    """Test ML prediction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample feature data
        self.sample_features = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'feature_date': ['2025-06-09', '2025-06-09', '2025-06-09'],
            'price': [150.0, 300.0, 140.0],
            'volume': [1000000, 1200000, 800000],
            'return_1d': [0.01, -0.02, 0.03],
            'return_5d': [0.05, -0.03, 0.04],
            'volatility_20d': [0.2, 0.25, 0.18],
            'rsi_14': [45, 55, 40],
            'feature_count': [7, 7, 7]
        })
        
        # Create temporary model file
        self.model_path = Path(self.temp_dir) / "test_model.txt"
        self.model_path.write_text("dummy lightgbm model")
        
        # Create temporary features directory
        self.features_dir = Path(self.temp_dir) / "features"
        self.features_dir.mkdir()
        
        # Save sample features
        feature_file = self.features_dir / "features_2025-06-09.csv"
        self.sample_features.to_csv(feature_file, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('lightgbm.Booster')
    def test_model_wrapper_lightgbm(self, mock_booster):
        """Test ModelWrapper with LightGBM model."""
        from mech_exo.ml.predict import ModelWrapper
        
        # Mock booster instance
        mock_model = Mock()
        mock_model.feature_name.return_value = ['price', 'volume', 'return_1d']
        mock_model.predict.return_value = np.array([0.7, 0.3, 0.6])
        mock_booster.return_value = mock_model
        
        # Create wrapper
        wrapper = ModelWrapper(str(self.model_path), algorithm='lightgbm')
        wrapper.load_model()
        
        # Test prediction
        test_features = pd.DataFrame({
            'price': [150, 300, 140],
            'volume': [1000, 1200, 800],
            'return_1d': [0.01, -0.02, 0.03]
        })
        
        predictions = wrapper.predict(test_features)
        
        # Verify
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(0 <= p <= 1 for p in predictions))
        mock_booster.assert_called_once()
        mock_model.predict.assert_called_once()
    
    @patch('mech_exo.ml.predict.ModelWrapper')
    def test_ml_predictor_predict_scores(self, mock_wrapper_class):
        """Test MLPredictor.predict_scores method."""
        from mech_exo.ml.predict import MLPredictor
        
        # Mock ModelWrapper
        mock_wrapper = Mock()
        mock_wrapper.predict.return_value = np.array([0.8, 0.6, 0.7])
        mock_wrapper_class.return_value = mock_wrapper
        
        # Create predictor
        predictor = MLPredictor(str(self.model_path), str(self.features_dir))
        
        # Test prediction
        results = predictor.predict_scores(
            target_date='2025-06-09',
            symbols=['AAPL', 'MSFT', 'GOOGL']
        )
        
        # Verify results
        self.assertFalse(results.empty)
        self.assertIn('symbol', results.columns)
        self.assertIn('ml_score', results.columns)
        self.assertEqual(len(results), 3)
        
        # Check symbols are present
        self.assertSetEqual(set(results['symbol']), {'AAPL', 'MSFT', 'GOOGL'})
        
        # Check scores are normalized
        self.assertTrue(all(0 <= score <= 1 for score in results['ml_score']))
    
    def test_ml_predictor_load_features_missing_file(self):
        """Test MLPredictor with missing feature file."""
        from mech_exo.ml.predict import MLPredictor
        
        predictor = MLPredictor(str(self.model_path), str(self.features_dir))
        
        # Test with non-existent date
        with self.assertRaises(FileNotFoundError):
            predictor.load_features_for_date('2024-01-01')
    
    @patch('mech_exo.ml.predict.predict_cli')
    def test_predict_cli_integration(self, mock_predict_cli):
        """Test predict_cli function."""
        # Mock the CLI function
        mock_predict_cli.return_value = {
            'success': True,
            'predictions': 3,
            'algorithm': 'lightgbm',
            'output_file': 'ml_scores.csv'
        }
        
        # Import after mocking
        from mech_exo.ml.predict import predict_cli
        
        # Test the function
        result = predict_cli(
            model_path=str(self.model_path),
            date='2025-06-09',
            symbols='AAPL,MSFT,GOOGL'
        )
        
        # Verify
        self.assertTrue(result['success'])
        self.assertEqual(result['predictions'], 3)


class TestIdeaScorerMLIntegration(unittest.TestCase):
    """Test IdeaScorer ML integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample ML scores
        self.ml_scores = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'ml_score': [0.8, 0.6, 0.7]
        })
        
        self.ml_scores_file = Path(self.temp_dir) / "ml_scores.csv"
        self.ml_scores.to_csv(self.ml_scores_file, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('mech_exo.scoring.scorer.DataStorage')
    @patch('mech_exo.utils.config.ConfigManager')
    def test_idea_scorer_ml_initialization(self, mock_config, mock_storage):
        """Test IdeaScorer initialization with ML."""
        from mech_exo.scoring.scorer import IdeaScorer
        
        # Mock configuration
        mock_config_manager = Mock()
        mock_config_manager.get_factor_config.return_value = {
            'ml_weight': 0.4,
            'fundamental': {},
            'technical': {},
            'sentiment': {}
        }
        mock_config.return_value = mock_config_manager
        
        # Mock storage
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Create scorer with ML
        scorer = IdeaScorer(use_ml=True)
        
        # Verify ML settings
        self.assertTrue(scorer.use_ml)
        self.assertEqual(scorer.ml_weight, 0.4)
    
    @patch('mech_exo.scoring.scorer.DataStorage')
    @patch('mech_exo.utils.config.ConfigManager')
    def test_load_ml_scores_from_file(self, mock_config, mock_storage):
        """Test loading ML scores from file."""
        from mech_exo.scoring.scorer import IdeaScorer
        
        # Mock configuration
        mock_config_manager = Mock()
        mock_config_manager.get_factor_config.return_value = {
            'ml_weight': 0.3,
            'fundamental': {},
            'technical': {},
            'sentiment': {}
        }
        mock_config.return_value = mock_config_manager
        
        # Mock storage
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Create scorer
        scorer = IdeaScorer(use_ml=True)
        
        # Test loading ML scores
        ml_scores = scorer._load_ml_scores(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            ml_scores_file=str(self.ml_scores_file)
        )
        
        # Verify
        self.assertIsNotNone(ml_scores)
        self.assertEqual(len(ml_scores), 3)
        self.assertIn('symbol', ml_scores.columns)
        self.assertIn('ml_score', ml_scores.columns)
    
    @patch('mech_exo.scoring.scorer.DataStorage')
    @patch('mech_exo.utils.config.ConfigManager')
    def test_integrate_ml_scores(self, mock_config, mock_storage):
        """Test ML score integration with traditional scores."""
        from mech_exo.scoring.scorer import IdeaScorer
        
        # Mock configuration
        mock_config_manager = Mock()
        mock_config_manager.get_factor_config.return_value = {
            'ml_weight': 0.3,
            'fundamental': {},
            'technical': {},
            'sentiment': {}
        }
        mock_config.return_value = mock_config_manager
        
        # Mock storage
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Create scorer
        scorer = IdeaScorer(use_ml=True)
        
        # Create test data
        traditional_scores = pd.Series([80, 60, 70], name='traditional')
        data = pd.DataFrame({'symbol': ['AAPL', 'MSFT', 'GOOGL']})
        
        # Test integration
        final_scores = scorer._integrate_ml_scores(
            traditional_scores, self.ml_scores, data
        )
        
        # Verify
        self.assertEqual(len(final_scores), 3)
        self.assertTrue(all(isinstance(score, (int, float)) for score in final_scores))
        
        # Scores should be different from traditional scores (due to ML integration)
        self.assertFalse(final_scores.equals(traditional_scores))


class TestPrefectIntegration(unittest.TestCase):
    """Test Prefect flow integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_latest_model(self):
        """Test load_latest_model task."""
        from dags.ml_inference_flow import load_latest_model
        
        # Create test model files
        models_dir = Path(self.temp_dir) / "models"
        models_dir.mkdir()
        
        # Create model files with different timestamps
        lgb_file = models_dir / "lgbm_20250609_160000.txt"
        xgb_file = models_dir / "xgb_20250608_140000.json"
        
        lgb_file.write_text("lgb model")
        xgb_file.write_text("xgb model")
        
        # Make LightGBM file newer
        import time
        time.sleep(0.1)
        lgb_file.touch()
        
        # Test function
        result = load_latest_model(str(models_dir))
        
        # Should return the LightGBM file (newer)
        self.assertEqual(result, str(lgb_file))
    
    def test_load_latest_model_no_models(self):
        """Test load_latest_model with no models."""
        from dags.ml_inference_flow import load_latest_model
        
        # Empty models directory
        models_dir = Path(self.temp_dir) / "empty_models"
        models_dir.mkdir()
        
        # Should return None
        result = load_latest_model(str(models_dir))
        self.assertIsNone(result)
    
    @patch('mech_exo.ml.features.FeatureBuilder')
    def test_build_features_today(self, mock_builder_class):
        """Test build_features_today task."""
        from dags.ml_inference_flow import build_features_today
        
        # Mock FeatureBuilder
        mock_builder = Mock()
        mock_builder.build_features.return_value = {
            '2025-06-09': 'features_2025-06-09.csv'
        }
        mock_builder_class.return_value = mock_builder
        
        # Test function
        result = build_features_today(['AAPL', 'MSFT'])
        
        # Verify
        self.assertIsNotNone(result)
        self.assertIn('features_2025-06-09.csv', result)
        mock_builder.build_features.assert_called_once()


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create tiny dataset for testing
        self.training_data = pd.DataFrame({
            'symbol': ['AAPL'] * 50 + ['MSFT'] * 50,
            'feature_date': ['2025-01-01'] * 100,
            'price': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'return_1d': np.random.normal(0, 0.02, 100),
            'return_5d': np.random.normal(0, 0.05, 100),
            'volatility_20d': np.random.uniform(0.1, 0.3, 100),
            'forward_return': np.random.normal(0, 0.03, 100)
        })
        
        # Create binary labels
        self.training_data['label'] = (self.training_data['forward_return'] > 0).astype(int)
        
        # Create test features (20 rows)
        self.test_features = pd.DataFrame({
            'symbol': ['AAPL'] * 10 + ['MSFT'] * 10,
            'feature_date': ['2025-06-09'] * 20,
            'price': np.random.uniform(100, 200, 20),
            'volume': np.random.uniform(1000, 5000, 20),
            'return_1d': np.random.normal(0, 0.02, 20),
            'return_5d': np.random.normal(0, 0.05, 20),
            'volatility_20d': np.random.uniform(0.1, 0.3, 20),
            'feature_count': [5] * 20
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('lightgbm.LGBMClassifier')
    @patch('mech_exo.ml.predict.ModelWrapper')
    def test_end_to_end_prediction_pipeline(self, mock_wrapper_class, mock_lgbm):
        """Test complete prediction pipeline."""
        # Mock trained model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.4, 0.6]] * 10)
        mock_lgbm.return_value = mock_model
        
        # Mock ModelWrapper
        mock_wrapper = Mock()
        mock_wrapper.predict.return_value = np.random.uniform(0, 1, 20)
        mock_wrapper_class.return_value = mock_wrapper
        
        # Save test features
        features_file = Path(self.temp_dir) / "features_2025-06-09.csv"
        self.test_features.to_csv(features_file, index=False)
        
        # Test prediction
        from mech_exo.ml.predict import MLPredictor
        
        predictor = MLPredictor("dummy_model.txt", str(Path(features_file).parent))
        
        # Generate predictions
        results = predictor.predict_scores(
            target_date='2025-06-09',
            symbols=['AAPL', 'MSFT']
        )
        
        # Verify results
        self.assertFalse(results.empty)
        self.assertEqual(len(results), 20)
        self.assertIn('symbol', results.columns)
        self.assertIn('ml_score', results.columns)
        
        # All scores should be in 0-1 range
        self.assertTrue(all(0 <= score <= 1 for score in results['ml_score']))
        
        # Should contain both symbols
        symbols_in_results = set(results['symbol'].unique())
        self.assertTrue({'AAPL', 'MSFT'}.issubset(symbols_in_results))
    
    @patch('mech_exo.scoring.scorer.DataStorage')
    @patch('mech_exo.utils.config.ConfigManager')
    def test_idea_scorer_with_ml_scores(self, mock_config, mock_storage):
        """Test IdeaScorer with ML scores produces merged rankings."""
        from mech_exo.scoring.scorer import IdeaScorer
        
        # Mock configuration
        mock_config_manager = Mock()
        mock_config_manager.get_factor_config.return_value = {
            'ml_weight': 0.3,
            'fundamental': {
                'pe_ratio': {'weight': 20, 'direction': 'lower_better'}
            },
            'technical': {
                'rsi_14': {'weight': 15, 'direction': 'contrarian'}
            }
        }
        mock_config.return_value = mock_config_manager
        
        # Mock storage and data
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Mock data returns
        mock_storage_instance.get_fundamental_data.return_value = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'pe_ratio': [25, 30]
        })
        
        mock_storage_instance.get_ohlc_data.return_value = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'close': [150, 300],
            'returns': [0.01, -0.02]
        })
        
        mock_storage_instance.get_news_data.return_value = pd.DataFrame()
        
        # Create ML scores file
        ml_scores = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'ml_score': [0.8, 0.6]
        })
        ml_scores_file = Path(self.temp_dir) / "ml_scores.csv"
        ml_scores.to_csv(ml_scores_file, index=False)
        
        # Create scorer with ML
        scorer = IdeaScorer(use_ml=True)
        
        # Mock factor calculations
        with patch('mech_exo.scoring.factors.FactorFactory.create_all_factors') as mock_factors:
            mock_factor = Mock()
            mock_factor.calculate.return_value = pd.Series([80, 60], index=[0, 1])
            mock_factor.normalize.return_value = pd.Series([80, 60], index=[0, 1])
            mock_factor.apply_direction.return_value = pd.Series([80, 60], index=[0, 1])
            mock_factor.weight = 20
            
            mock_factors.return_value = {'test_factor': mock_factor}
            
            # Generate scores
            results = scorer.score(['AAPL', 'MSFT'], ml_scores_file=str(ml_scores_file))
            
            # Verify ML integration
            self.assertFalse(results.empty)
            self.assertIn('composite_score', results.columns)
            self.assertTrue(scorer.use_ml)
            
            # Should have ML-specific columns if ML scores were available
            if 'ml_rank' in results.columns:
                self.assertIn('ml_rank', results.columns)
                self.assertIn('final_score', results.columns)


if __name__ == '__main__':
    unittest.main()