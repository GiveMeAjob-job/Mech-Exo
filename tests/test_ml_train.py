"""
Unit tests for ML model training pipeline.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from mech_exo.ml.train_ml import MLTrainer


class TestMLTrainer(unittest.TestCase):
    """Test cases for MLTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def test_trainer_init_lightgbm(self):
        """Test MLTrainer initialization with LightGBM."""
        try:
            trainer = MLTrainer(algorithm="lightgbm", random_state=42)
            self.assertEqual(trainer.algorithm, "lightgbm")
            self.assertEqual(trainer.random_state, 42)
            self.assertIsNone(trainer.model)
            self.assertIsNone(trainer.best_params)
        except ImportError:
            self.skipTest("LightGBM not installed")
    
    def test_trainer_init_xgboost(self):
        """Test MLTrainer initialization with XGBoost."""
        try:
            trainer = MLTrainer(algorithm="xgboost", random_state=42)
            self.assertEqual(trainer.algorithm, "xgboost")
            self.assertEqual(trainer.random_state, 42)
        except ImportError:
            self.skipTest("XGBoost not installed")
    
    def test_trainer_init_invalid_algorithm(self):
        """Test MLTrainer initialization with invalid algorithm."""
        with self.assertRaises(ValueError):
            MLTrainer(algorithm="invalid_algo")
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        try:
            trainer = MLTrainer(algorithm="lightgbm", random_state=42)
        except ImportError:
            self.skipTest("LightGBM not installed")
        
        # Create sample features
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        symbols = ['AAPL', 'MSFT']
        
        features_data = []
        for symbol in symbols:
            base_price = 150 if symbol == 'AAPL' else 300
            
            for i, date in enumerate(dates):
                price = base_price + i * 0.5 + np.random.normal(0, 1)
                features_data.append({
                    'symbol': symbol,
                    'feature_date': date.strftime('%Y-%m-%d'),
                    'price': price,
                    'volume': 1000000,
                    'return_1d': np.random.normal(0, 0.02),
                    'return_5d': np.random.normal(0, 0.05),
                    'volatility_20d': np.random.uniform(0.15, 0.35),
                    'rsi_14': np.random.uniform(30, 70),
                    'fund_pe_ratio': np.random.uniform(20, 35),
                    'sent_score_mean': np.random.uniform(0.3, 0.7)
                })
        
        features_df = pd.DataFrame(features_data)
        
        # Test data preparation
        X, y = trainer.prepare_training_data(features_df, forward_days=5)
        
        # Verify output structure
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X), 0)
        
        # Verify feature columns
        expected_features = [
            'price', 'volume', 'return_1d', 'return_5d', 'volatility_20d',
            'rsi_14', 'fund_pe_ratio', 'sent_score_mean'
        ]
        for feature in expected_features:
            self.assertIn(feature, X.columns)
        
        # Verify labels are binary
        self.assertTrue(set(y.unique()).issubset({0, 1}))
        
        # Verify no NaN values
        self.assertFalse(X.isnull().any().any())
        self.assertFalse(y.isnull().any())
    
    def test_hyperparameter_grids(self):
        """Test hyperparameter grid generation."""
        
        # Test LightGBM grid
        try:
            trainer = MLTrainer(algorithm="lightgbm")
            lgb_grid = trainer.get_hyperparameter_grid()
            
            expected_lgb_params = [
                'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda'
            ]
            for param in expected_lgb_params:
                self.assertIn(param, lgb_grid)
                self.assertIsInstance(lgb_grid[param], list)
                self.assertGreater(len(lgb_grid[param]), 0)
        except ImportError:
            pass  # Skip if LightGBM not available
        
        # Test XGBoost grid
        try:
            trainer = MLTrainer(algorithm="xgboost")
            xgb_grid = trainer.get_hyperparameter_grid()
            
            expected_xgb_params = [
                'max_depth', 'learning_rate', 'n_estimators', 'subsample',
                'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda'
            ]
            for param in expected_xgb_params:
                self.assertIn(param, xgb_grid)
                self.assertIsInstance(xgb_grid[param], list)
                self.assertGreater(len(xgb_grid[param]), 0)
        except ImportError:
            pass  # Skip if XGBoost not available
    
    def test_create_model(self):
        """Test model creation with parameters."""
        try:
            trainer = MLTrainer(algorithm="lightgbm", random_state=42)
            
            # Test model creation with custom parameters
            model = trainer.create_model(num_leaves=31, learning_rate=0.1)
            
            # Verify model type and parameters
            self.assertIsNotNone(model)
            self.assertEqual(model.random_state, 42)
            self.assertEqual(model.num_leaves, 31)
            self.assertEqual(model.learning_rate, 0.1)
            
        except ImportError:
            self.skipTest("LightGBM not installed")
    
    @patch('mech_exo.ml.train_ml.RandomizedSearchCV')
    def test_train_with_cv_mock(self, mock_search):
        """Test training with mocked RandomizedSearchCV."""
        try:
            trainer = MLTrainer(algorithm="lightgbm", random_state=42)
        except ImportError:
            self.skipTest("LightGBM not installed")
        
        # Mock search results
        mock_estimator = Mock()
        mock_search_instance = Mock()
        mock_search_instance.best_estimator_ = mock_estimator
        mock_search_instance.best_params_ = {'num_leaves': 31, 'learning_rate': 0.1}
        mock_search_instance.best_score_ = 0.75
        mock_search_instance.cv_results_ = {'mean_test_score': [0.75, 0.70, 0.65]}
        mock_search.return_value = mock_search_instance
        
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        # Test training
        results = trainer.train_with_cv(X, y, cv_folds=2, n_iter=5)
        
        # Verify results structure
        self.assertIn('best_auc', results)
        self.assertIn('best_params', results)
        self.assertIn('metrics', results)
        self.assertIn('algorithm', results)
        self.assertIn('training_samples', results)
        
        # Verify values
        self.assertEqual(results['best_auc'], 0.75)
        self.assertEqual(results['algorithm'], 'lightgbm')
        self.assertEqual(results['training_samples'], 100)
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        try:
            trainer = MLTrainer(algorithm="lightgbm", random_state=42)
        except ImportError:
            self.skipTest("LightGBM not installed")
        
        # Set mock best parameters
        trainer.best_params = {'num_leaves': 31, 'learning_rate': 0.1}
        
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'feature3': np.random.randn(50)
        })
        y = pd.Series(np.random.choice([0, 1], 50))
        
        # Test metrics calculation
        metrics = trainer._calculate_metrics(X, y, cv_folds=2)
        
        # Verify metrics structure
        expected_metrics = [
            'mean_auc', 'std_auc', 'mean_ic', 'std_ic',
            'mean_accuracy', 'std_accuracy', 'fold_aucs', 'fold_ics'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Verify value ranges
        self.assertGreaterEqual(metrics['mean_auc'], 0)
        self.assertLessEqual(metrics['mean_auc'], 1)
        self.assertGreaterEqual(metrics['mean_accuracy'], 0)
        self.assertLessEqual(metrics['mean_accuracy'], 1)
    
    def test_save_metrics(self):
        """Test metrics saving."""
        try:
            trainer = MLTrainer(algorithm="lightgbm", random_state=42)
        except ImportError:
            self.skipTest("LightGBM not installed")
        
        # Create sample metrics
        metrics = {
            'best_auc': 0.75,
            'best_params': {'num_leaves': 31},
            'metrics': {'mean_auc': 0.73, 'std_auc': 0.02},
            'algorithm': 'lightgbm'
        }
        
        # Test saving
        metrics_file = trainer.save_metrics(metrics, str(self.models_dir))
        
        # Verify file exists
        self.assertTrue(Path(metrics_file).exists())
        
        # Verify file content
        with open(metrics_file, 'r') as f:
            saved_metrics = json.load(f)
        
        self.assertEqual(saved_metrics['algorithm'], 'lightgbm')
        self.assertEqual(saved_metrics['best_auc'], 0.75)
        self.assertIn('timestamp', saved_metrics)


class TestMLTrainingIntegration(unittest.TestCase):
    """Integration tests for ML training pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.features_dir = Path(self.temp_dir) / "features"
        self.models_dir = Path(self.temp_dir) / "models"
        
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample feature files
        self._create_sample_feature_files()
    
    def _create_sample_feature_files(self):
        """Create sample feature files for testing."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for i, date in enumerate(dates[:10]):  # Create 10 feature files
            date_str = date.strftime('%Y-%m-%d')
            features_data = []
            
            for symbol in symbols:
                base_price = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 140}[symbol]
                price = base_price + i * 0.5 + np.random.normal(0, 2)
                
                features_data.append({
                    'symbol': symbol,
                    'feature_date': date_str,
                    'price': price,
                    'volume': 1000000 + np.random.randint(-100000, 100000),
                    'return_1d': np.random.normal(0, 0.02),
                    'return_5d': np.random.normal(0, 0.05),
                    'return_20d': np.random.normal(0, 0.08),
                    'volatility_5d': np.random.uniform(0.15, 0.25),
                    'volatility_20d': np.random.uniform(0.20, 0.35),
                    'rsi_14': np.random.uniform(30, 70),
                    'fund_pe_ratio': np.random.uniform(20, 35),
                    'fund_return_on_equity': np.random.uniform(0.15, 0.45),
                    'sent_score_mean': np.random.uniform(0.3, 0.7),
                    'sent_score_std': np.random.uniform(0.1, 0.3),
                    'feature_count': 12
                })
            
            # Save as CSV
            features_df = pd.DataFrame(features_data)
            file_path = self.features_dir / f"features_{date_str}.csv"
            features_df.to_csv(file_path, index=False)
    
    def test_training_pipeline_integration(self):
        """Test end-to-end training pipeline."""
        try:
            from mech_exo.ml.train_ml import train_ml_cli
        except ImportError:
            self.skipTest("ML dependencies not installed")
        
        try:
            # Run training with minimal settings
            results = train_ml_cli(
                algorithm="lightgbm",
                lookback="30d",
                cv_folds=2,
                n_iter=3,  # Minimal iterations for speed
                seed=42,
                features_dir=str(self.features_dir),
                models_dir=str(self.models_dir)
            )
            
            # Verify results structure
            self.assertIn('best_auc', results)
            self.assertIn('model_file', results)
            self.assertIn('metrics_file', results)
            
            # Verify files were created
            self.assertTrue(Path(results['model_file']).exists())
            self.assertTrue(Path(results['metrics_file']).exists())
            
            # Verify AUC is reasonable (>0.4 for random data)
            self.assertGreater(results['best_auc'], 0.4)
            
            # Verify model file size is reasonable
            model_size = Path(results['model_file']).stat().st_size
            self.assertGreater(model_size, 100)  # At least 100 bytes
            self.assertLess(model_size, 5 * 1024 * 1024)  # Less than 5MB
            
            print(f"✅ Integration test passed: AUC={results['best_auc']:.4f}")
            
        except ImportError as e:
            self.skipTest(f"Required ML library not installed: {e}")


if __name__ == '__main__':
    unittest.main()