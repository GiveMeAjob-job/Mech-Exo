"""
Unit tests for ML SHAP feature importance reporting.

Tests SHAPReporter class and SHAP report generation functionality.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest


class TestSHAPReporter(unittest.TestCase):
    """Test SHAPReporter class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample feature data
        self.sample_features = pd.DataFrame({
            'price': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'return_1d': [0.01, -0.02, 0.03, -0.01, 0.02],
            'return_5d': [0.05, -0.03, 0.04, 0.02, -0.01],
            'volatility_20d': [0.2, 0.25, 0.18, 0.22, 0.19],
            'rsi_14': [45, 55, 40, 60, 50],
            'sent_score_mean': [0.6, 0.4, 0.7, 0.5, 0.8]
        })
        
        # Create temporary model file
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model.txt"
        self.model_path.write_text("dummy model content")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_algorithm_detection(self):
        """Test algorithm detection from file extension."""
        from mech_exo.ml.report_ml import SHAPReporter
        
        # Test LightGBM detection
        lgb_path = Path(self.temp_dir) / "model.txt"
        lgb_path.write_text("lgb model")
        
        with patch('mech_exo.ml.report_ml.shap'):
            with patch('mech_exo.ml.report_ml.plt'):
                reporter = SHAPReporter(str(lgb_path))
                self.assertEqual(reporter.algorithm, 'lightgbm')
        
        # Test XGBoost detection
        xgb_path = Path(self.temp_dir) / "model.json"
        xgb_path.write_text("xgb model")
        
        with patch('mech_exo.ml.report_ml.shap'):
            with patch('mech_exo.ml.report_ml.plt'):
                reporter = SHAPReporter(str(xgb_path))
                self.assertEqual(reporter.algorithm, 'xgboost')
        
        # Test filename-based detection
        lgb_name_path = Path(self.temp_dir) / "lgbm_model.pkl"
        lgb_name_path.write_text("lgb model")
        
        with patch('mech_exo.ml.report_ml.shap'):
            with patch('mech_exo.ml.report_ml.plt'):
                reporter = SHAPReporter(str(lgb_name_path))
                self.assertEqual(reporter.algorithm, 'lightgbm')
    
    def test_file_not_found_error(self):
        """Test FileNotFoundError for non-existent model."""
        from mech_exo.ml.report_ml import SHAPReporter
        
        with patch('mech_exo.ml.report_ml.shap'):
            with patch('mech_exo.ml.report_ml.plt'):
                with self.assertRaises(FileNotFoundError):
                    SHAPReporter("/nonexistent/model.txt")
    
    @patch('mech_exo.ml.report_ml.shap')
    @patch('mech_exo.ml.report_ml.plt')
    def test_import_error_handling(self, mock_plt, mock_shap):
        """Test handling of missing SHAP dependencies."""
        from mech_exo.ml.report_ml import SHAPReporter
        
        # Simulate ImportError
        mock_shap.side_effect = ImportError("shap not found")
        
        with self.assertRaises(ImportError) as cm:
            SHAPReporter(str(self.model_path))
        
        self.assertIn("SHAP dependencies not installed", str(cm.exception))
    
    @patch('mech_exo.ml.report_ml.shap')
    @patch('mech_exo.ml.report_ml.plt')
    @patch('lightgbm.Booster')
    def test_lightgbm_model_loading(self, mock_booster, mock_plt, mock_shap):
        """Test LightGBM model loading."""
        from mech_exo.ml.report_ml import SHAPReporter
        
        # Mock SHAP and matplotlib
        mock_shap.TreeExplainer = Mock()
        mock_plt.switch_backend = Mock()
        
        # Create reporter
        reporter = SHAPReporter(str(self.model_path), algorithm='lightgbm')
        
        # Mock the booster instance
        mock_model = Mock()
        mock_booster.return_value = mock_model
        
        # Test model loading
        reporter.load_model()
        
        # Verify LightGBM booster was called
        mock_booster.assert_called_once_with(model_file=str(self.model_path))
        self.assertEqual(reporter.model, mock_model)
    
    @patch('mech_exo.ml.report_ml.shap')
    @patch('mech_exo.ml.report_ml.plt')
    def test_feature_importance_ranking(self, mock_plt, mock_shap):
        """Test feature importance ranking calculation."""
        from mech_exo.ml.report_ml import SHAPReporter
        
        # Mock SHAP and matplotlib
        mock_shap.TreeExplainer = Mock()
        mock_plt.switch_backend = Mock()
        
        # Create reporter
        reporter = SHAPReporter(str(self.model_path), algorithm='lightgbm')
        
        # Create mock SHAP values
        shap_values = np.array([
            [0.1, -0.2, 0.3, -0.1, 0.2],  # Sample 1
            [-0.1, 0.3, -0.2, 0.2, -0.1], # Sample 2
            [0.2, -0.1, 0.1, -0.3, 0.1]   # Sample 3
        ])
        
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
        # Get importance ranking
        importance_df = reporter.get_feature_importance_ranking(
            shap_values, feature_names, top_k=3
        )
        
        # Verify structure
        self.assertEqual(len(importance_df), 3)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)
        self.assertIn('rank', importance_df.columns)
        
        # Verify sorting (highest importance first)
        self.assertTrue(importance_df['importance'].is_monotonic_decreasing)
        self.assertListEqual(list(importance_df['rank']), [1, 2, 3])


class TestSHAPReportGeneration(unittest.TestCase):
    """Test SHAP report generation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample feature matrix
        self.feature_matrix = pd.DataFrame({
            'price': np.random.uniform(100, 200, 50),
            'volume': np.random.uniform(1000, 5000, 50),
            'return_1d': np.random.normal(0, 0.02, 50),
            'volatility_20d': np.random.uniform(0.1, 0.3, 50),
            'rsi_14': np.random.uniform(30, 70, 50)
        })
        
        # Create temporary model file
        self.model_path = Path(self.temp_dir) / "test_model.txt"
        self.model_path.write_text("dummy lightgbm model")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('mech_exo.ml.report_ml.SHAPReporter')
    def test_make_shap_report(self, mock_reporter_class):
        """Test make_shap_report function."""
        from mech_exo.ml.report_ml import make_shap_report
        
        # Mock SHAPReporter instance
        mock_reporter = Mock()
        mock_reporter_class.return_value = mock_reporter
        mock_reporter.algorithm = 'lightgbm'
        
        # Mock SHAP values and sample data
        mock_shap_values = np.random.random((10, 5))
        mock_sample_data = self.feature_matrix.head(10)
        mock_reporter.calculate_shap_values.return_value = (mock_shap_values, mock_sample_data)
        
        # Mock file paths
        png_path = str(Path(self.temp_dir) / "test.png")
        html_path = str(Path(self.temp_dir) / "test.html")
        
        mock_reporter.generate_summary_plot.return_value = png_path
        mock_reporter.generate_force_plot_html.return_value = html_path
        
        # Mock importance ranking
        mock_importance_df = pd.DataFrame({
            'feature': ['feature_1', 'feature_2'],
            'importance': [0.5, 0.3]
        })
        mock_reporter.get_feature_importance_ranking.return_value = mock_importance_df
        
        # Create mock files for size calculation
        Path(png_path).write_text("fake png")
        Path(html_path).write_text("fake html")
        
        # Call function
        metadata = make_shap_report(
            model_path=str(self.model_path),
            feature_matrix=self.feature_matrix,
            out_html=html_path,
            out_png=png_path,
            top_k=20,
            algorithm='lightgbm'
        )
        
        # Verify metadata structure
        expected_keys = [
            'timestamp', 'model_path', 'algorithm', 'samples_analyzed',
            'features_count', 'top_k', 'png_path', 'html_path',
            'png_size_mb', 'html_size_mb', 'top_features', 'feature_importance_scores'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metadata)
        
        # Verify values
        self.assertEqual(metadata['algorithm'], 'lightgbm')
        self.assertEqual(metadata['model_path'], str(self.model_path))
        self.assertEqual(metadata['png_path'], png_path)
        self.assertEqual(metadata['html_path'], html_path)
    
    @patch('mech_exo.ml.report_ml.make_shap_report')
    def test_shap_report_cli(self, mock_make_shap_report):
        """Test shap_report_cli function."""
        from mech_exo.ml.report_ml import shap_report_cli
        
        # Create test feature files
        features_dir = Path(self.temp_dir) / "features"
        features_dir.mkdir()
        
        # Create feature file for today
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        feature_file = features_dir / f"features_{today_str}.csv"
        
        test_features = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'feature_date': [today_str, today_str],
            'price': [150, 300],
            'return_1d': [0.01, -0.02],
            'feature_count': [2, 2]
        })
        test_features.to_csv(feature_file, index=False)
        
        # Mock the make_shap_report function
        mock_metadata = {
            'timestamp': '2025-06-09T10:00:00',
            'algorithm': 'lightgbm',
            'samples_analyzed': 2,
            'features_count': 2,
            'top_features': ['price', 'return_1d']
        }
        mock_make_shap_report.return_value = mock_metadata
        
        # Call CLI function
        result = shap_report_cli(
            model_path=str(self.model_path),
            date="today",
            features_dir=str(features_dir),
            output_dir=str(Path(self.temp_dir) / "reports"),
            top_k=20
        )
        
        # Verify result
        self.assertIn('algorithm', result)
        self.assertIn('date_requested', result)
        self.assertIn('target_date', result)
        self.assertIn('feature_files_used', result)
        
        # Verify make_shap_report was called
        mock_make_shap_report.assert_called_once()
    
    def test_shap_report_cli_no_features(self):
        """Test shap_report_cli when no feature files are found."""
        from mech_exo.ml.report_ml import shap_report_cli
        
        # Empty features directory
        features_dir = Path(self.temp_dir) / "empty_features"
        features_dir.mkdir()
        
        # Should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError) as cm:
            shap_report_cli(
                model_path=str(self.model_path),
                date="today",
                features_dir=str(features_dir)
            )
        
        self.assertIn("No feature files found", str(cm.exception))


class TestPrefectIntegration(unittest.TestCase):
    """Test Prefect flow integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_find_latest_models(self):
        """Test find_latest_models task."""
        from dags.ml_report_flow import find_latest_models
        
        # Create test model files
        models_dir = Path(self.temp_dir) / "models"
        models_dir.mkdir()
        
        # Create model files with different timestamps
        lgb_file1 = models_dir / "lgbm_20250601_120000.txt"
        lgb_file2 = models_dir / "lgbm_20250609_160000.txt"  # Latest
        xgb_file = models_dir / "xgb_20250608_140000.json"
        
        lgb_file1.write_text("old lgb model")
        lgb_file2.write_text("new lgb model")
        xgb_file.write_text("xgb model")
        
        # Test function
        models = find_latest_models(str(models_dir))
        
        # Verify results
        self.assertEqual(len(models), 2)
        self.assertIn('lightgbm', models)
        self.assertIn('xgboost', models)
        
        # Verify latest files are selected
        self.assertEqual(models['lightgbm'], str(lgb_file2))
        self.assertEqual(models['xgboost'], str(xgb_file))
    
    def test_find_latest_models_empty_dir(self):
        """Test find_latest_models with empty directory."""
        from dags.ml_report_flow import find_latest_models
        
        # Empty models directory
        models_dir = Path(self.temp_dir) / "empty_models"
        models_dir.mkdir()
        
        # Should return empty dict
        models = find_latest_models(str(models_dir))
        self.assertEqual(models, {})
    
    def test_load_recent_features(self):
        """Test load_recent_features task."""
        from dags.ml_report_flow import load_recent_features
        
        # Create test feature files
        features_dir = Path(self.temp_dir) / "features"
        features_dir.mkdir()
        
        # Create feature files for recent dates
        today = pd.Timestamp.now().date()
        yesterday = today - pd.Timedelta(days=1)
        
        today_file = features_dir / f"features_{today.strftime('%Y-%m-%d')}.csv"
        yesterday_file = features_dir / f"features_{yesterday.strftime('%Y-%m-%d')}.parquet"
        
        # Create dummy feature data
        dummy_features = pd.DataFrame({
            'symbol': ['AAPL'],
            'price': [150],
            'feature_date': [today.strftime('%Y-%m-%d')]
        })
        
        dummy_features.to_csv(today_file, index=False)
        # For parquet, just create empty file for this test
        yesterday_file.write_text("dummy parquet")
        
        # Test function
        result = load_recent_features(str(features_dir), lookback_days=5)
        
        # Should return a file path
        self.assertIsNotNone(result)
        self.assertTrue(Path(result).exists())
    
    def test_load_recent_features_no_data(self):
        """Test load_recent_features with no data."""
        from dags.ml_report_flow import load_recent_features
        
        # Empty features directory
        features_dir = Path(self.temp_dir) / "empty_features"
        features_dir.mkdir()
        
        # Should return None
        result = load_recent_features(str(features_dir))
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()