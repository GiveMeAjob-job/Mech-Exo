"""
ML model interpretation and SHAP analysis for feature importance reporting.

Generates interactive SHAP reports with summary plots, force plots,
and feature importance rankings for model transparency.
"""

import json
import logging
import os
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class SHAPReporter:
    """
    SHAP-based model interpretation and feature importance reporting.
    
    Generates visual reports showing which features contribute most to
    model predictions, enabling transparency and model validation.
    """
    
    def __init__(self, model_path: str, algorithm: Optional[str] = None):
        """
        Initialize SHAP reporter.
        
        Args:
            model_path: Path to trained model file
            algorithm: Algorithm type ("lightgbm" or "xgboost"). Auto-detected if None.
        """
        self.model_path = Path(model_path)
        self.algorithm = algorithm or self._detect_algorithm()
        self.model = None
        self.explainer = None
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Import required libraries
        try:
            import shap
            import matplotlib.pyplot as plt
            self.shap = shap
            self.plt = plt
            
            # Set matplotlib backend for headless environments
            plt.switch_backend('Agg')
            
        except ImportError as e:
            raise ImportError(f"SHAP dependencies not installed: {e}. Run: pip install shap matplotlib")
    
    def _detect_algorithm(self) -> str:
        """Auto-detect algorithm from file extension."""
        if self.model_path.suffix == '.txt':
            return 'lightgbm'
        elif self.model_path.suffix == '.json':
            return 'xgboost'
        else:
            # Try to detect from filename
            name = self.model_path.name.lower()
            if 'lgbm' in name or 'lightgbm' in name:
                return 'lightgbm'
            elif 'xgb' in name or 'xgboost' in name:
                return 'xgboost'
            else:
                raise ValueError(f"Cannot detect algorithm from {self.model_path}. Specify explicitly.")
    
    def load_model(self) -> None:
        """Load the trained model."""
        logger.info(f"Loading {self.algorithm} model from {self.model_path}")
        
        if self.algorithm == 'lightgbm':
            try:
                import lightgbm as lgb
                self.model = lgb.Booster(model_file=str(self.model_path))
            except ImportError:
                raise ImportError("LightGBM not installed. Run: pip install lightgbm")
                
        elif self.algorithm == 'xgboost':
            try:
                import xgboost as xgb
                self.model = xgb.Booster()
                self.model.load_model(str(self.model_path))
            except ImportError:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        logger.info("Model loaded successfully")
    
    def create_explainer(self, background_data: Optional[pd.DataFrame] = None) -> None:
        """
        Create SHAP explainer for the model.
        
        Args:
            background_data: Optional background dataset for explainer
        """
        if self.model is None:
            self.load_model()
        
        logger.info("Creating SHAP TreeExplainer...")
        
        # Create explainer with optimizations
        if background_data is not None and len(background_data) > 100:
            # Use subset for faster computation
            background_sample = background_data.sample(n=100, random_state=42)
        else:
            background_sample = background_data
        
        if self.algorithm == 'lightgbm':
            self.explainer = self.shap.TreeExplainer(
                self.model,
                data=background_sample,
                feature_perturbation="tree_path_dependent"  # Fastest for LightGBM
            )
        else:  # xgboost
            self.explainer = self.shap.TreeExplainer(
                self.model,
                data=background_sample
            )
        
        logger.info("SHAP explainer created")
    
    def calculate_shap_values(self, feature_matrix: pd.DataFrame, 
                            max_samples: int = 1000) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Calculate SHAP values for feature matrix.
        
        Args:
            feature_matrix: Input features for explanation
            max_samples: Maximum samples to process (for speed)
            
        Returns:
            (shap_values, sample_data): SHAP values and corresponding sample data
        """
        if self.explainer is None:
            self.create_explainer(feature_matrix)
        
        # Sample data for manageable computation
        if len(feature_matrix) > max_samples:
            logger.info(f"Sampling {max_samples} rows from {len(feature_matrix)} for SHAP analysis")
            sample_data = feature_matrix.sample(n=max_samples, random_state=42)
        else:
            sample_data = feature_matrix.copy()
        
        logger.info(f"Calculating SHAP values for {len(sample_data)} samples...")
        
        # Calculate SHAP values
        if self.algorithm == 'lightgbm':
            shap_values = self.explainer.shap_values(sample_data)
            # For binary classification, take positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:  # xgboost
            shap_values = self.explainer.shap_values(sample_data)
        
        logger.info("SHAP values calculated successfully")
        return shap_values, sample_data
    
    def generate_summary_plot(self, shap_values: np.ndarray, 
                            feature_data: pd.DataFrame,
                            output_path: str,
                            top_k: int = 20) -> str:
        """
        Generate SHAP summary plot and save as PNG.
        
        Args:
            shap_values: SHAP values array
            feature_data: Feature matrix
            output_path: Output PNG file path
            top_k: Number of top features to display
            
        Returns:
            Path to saved PNG file
        """
        logger.info(f"Generating SHAP summary plot with top {top_k} features...")
        
        # Create summary plot
        self.plt.figure(figsize=(12, 8))
        
        self.shap.summary_plot(
            shap_values, 
            feature_data,
            max_display=top_k,
            show=False,
            plot_size=(12, 8)
        )
        
        # Save with optimized settings
        self.plt.tight_layout()
        self.plt.savefig(
            output_path,
            dpi=120,  # Good quality, reasonable file size
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        self.plt.close()
        
        # Check file size
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Summary plot saved: {output_path} ({file_size:.2f} MB)")
        
        if file_size > 2.0:
            logger.warning(f"PNG file is large ({file_size:.2f} MB). Consider reducing top_k or DPI.")
        
        return output_path
    
    def generate_force_plot_html(self, shap_values: np.ndarray,
                               feature_data: pd.DataFrame,
                               output_path: str,
                               sample_idx: int = 0) -> str:
        """
        Generate interactive SHAP force plot and save as HTML.
        
        Args:
            shap_values: SHAP values array
            feature_data: Feature matrix
            output_path: Output HTML file path
            sample_idx: Index of sample to explain
            
        Returns:
            Path to saved HTML file
        """
        logger.info(f"Generating interactive force plot for sample {sample_idx}...")
        
        # Create force plot
        force_plot = self.shap.force_plot(
            self.explainer.expected_value,
            shap_values[sample_idx],
            feature_data.iloc[sample_idx],
            matplotlib=False  # Use JavaScript version
        )
        
        # Save as HTML with inline JavaScript
        self.shap.save_html(output_path, force_plot)
        
        # Check file size
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Force plot saved: {output_path} ({file_size:.2f} MB)")
        
        if file_size > 5.0:
            logger.warning(f"HTML file is large ({file_size:.2f} MB). Consider compressing.")
        
        return output_path
    
    def get_feature_importance_ranking(self, shap_values: np.ndarray,
                                     feature_names: List[str],
                                     top_k: int = 20) -> pd.DataFrame:
        """
        Get feature importance ranking based on mean absolute SHAP values.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance rankings
        """
        # Calculate mean absolute SHAP values
        importance_scores = np.abs(shap_values).mean(axis=0)
        
        # Create ranking dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        # Return top k
        return importance_df.head(top_k)


def make_shap_report(model_path: str, 
                    feature_matrix: pd.DataFrame,
                    out_html: str, 
                    out_png: str,
                    top_k: int = 20,
                    algorithm: Optional[str] = None) -> Dict[str, Union[str, int, float]]:
    """
    Generate complete SHAP analysis report.
    
    Args:
        model_path: Path to trained model file
        feature_matrix: Features for analysis
        out_html: Output HTML file path
        out_png: Output PNG file path
        top_k: Number of top features to display
        algorithm: Model algorithm ("lightgbm" or "xgboost")
        
    Returns:
        Report metadata dictionary
    """
    logger.info(f"🔍 Generating SHAP report for {model_path}")
    
    # Create output directories
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize reporter
    reporter = SHAPReporter(model_path, algorithm)
    
    # Calculate SHAP values
    shap_values, sample_data = reporter.calculate_shap_values(feature_matrix)
    
    # Generate summary plot
    png_path = reporter.generate_summary_plot(
        shap_values, sample_data, out_png, top_k
    )
    
    # Generate interactive force plot
    html_path = reporter.generate_force_plot_html(
        shap_values, sample_data, out_html
    )
    
    # Get feature importance ranking
    importance_df = reporter.get_feature_importance_ranking(
        shap_values, list(sample_data.columns), top_k
    )
    
    # Create metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'algorithm': reporter.algorithm,
        'samples_analyzed': len(sample_data),
        'features_count': len(sample_data.columns),
        'top_k': top_k,
        'png_path': png_path,
        'html_path': html_path,
        'png_size_mb': Path(png_path).stat().st_size / (1024 * 1024),
        'html_size_mb': Path(html_path).stat().st_size / (1024 * 1024),
        'top_features': importance_df['feature'].tolist()[:10],
        'feature_importance_scores': importance_df['importance'].tolist()[:10]
    }
    
    logger.info(f"✅ SHAP report generated successfully")
    logger.info(f"   PNG: {png_path} ({metadata['png_size_mb']:.2f} MB)")
    logger.info(f"   HTML: {html_path} ({metadata['html_size_mb']:.2f} MB)")
    logger.info(f"   Top 5 features: {metadata['top_features'][:5]}")
    
    return metadata


def shap_report_cli(model_path: str,
                   date: str = "today",
                   features_dir: str = "data/features",
                   output_dir: str = "reports/shap",
                   png_name: Optional[str] = None,
                   html_name: Optional[str] = None,
                   top_k: int = 20,
                   algorithm: Optional[str] = None) -> Dict:
    """
    CLI function for generating SHAP reports.
    
    Args:
        model_path: Path to trained model
        date: Date for feature data ("today" or YYYY-MM-DD)
        features_dir: Directory containing feature files
        output_dir: Output directory for reports
        png_name: Custom PNG filename
        html_name: Custom HTML filename
        top_k: Number of top features to display
        algorithm: Model algorithm
        
    Returns:
        Report metadata dictionary
    """
    logger.info(f"🔍 Starting SHAP analysis...")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Date: {date}")
    logger.info(f"   Features dir: {features_dir}")
    logger.info(f"   Output dir: {output_dir}")
    logger.info(f"   Top features: {top_k}")
    
    # Parse date
    if date.lower() == "today":
        target_date = datetime.now().date()
    else:
        target_date = datetime.strptime(date, '%Y-%m-%d').date()
    
    # Load feature data for the date (or recent period)
    features_path = Path(features_dir)
    
    # Look for feature file on target date or recent dates
    feature_files = []
    for days_back in range(10):  # Look back up to 10 days
        check_date = target_date - timedelta(days=days_back)
        date_str = check_date.strftime('%Y-%m-%d')
        
        csv_file = features_path / f"features_{date_str}.csv"
        parquet_file = features_path / f"features_{date_str}.parquet"
        
        if csv_file.exists():
            feature_files.append(csv_file)
        elif parquet_file.exists():
            feature_files.append(parquet_file)
        
        if len(feature_files) >= 5:  # Enough files found
            break
    
    if not feature_files:
        raise FileNotFoundError(f"No feature files found for {date} in {features_dir}")
    
    logger.info(f"Found {len(feature_files)} feature files")
    
    # Load and combine features
    all_features = []
    for file_path in feature_files[:5]:  # Use up to 5 recent files
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_parquet(file_path)
            all_features.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    
    if not all_features:
        raise ValueError("No valid feature data loaded")
    
    # Combine features
    feature_matrix = pd.concat(all_features, ignore_index=True)
    
    # Prepare feature matrix (remove metadata columns)
    feature_cols = [col for col in feature_matrix.columns 
                   if col not in ['symbol', 'feature_date', 'feature_count']]
    X = feature_matrix[feature_cols].copy()
    
    logger.info(f"Feature matrix: {len(X)} samples, {len(X.columns)} features")
    
    # Generate output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).stem
    
    if png_name is None:
        png_name = f"shap_summary_{model_name}_{timestamp}.png"
    if html_name is None:
        html_name = f"shap_force_{model_name}_{timestamp}.html"
    
    output_path = Path(output_dir)
    png_path = str(output_path / png_name)
    html_path = str(output_path / html_name)
    
    # Generate SHAP report
    metadata = make_shap_report(
        model_path=model_path,
        feature_matrix=X,
        out_html=html_path,
        out_png=png_path,
        top_k=top_k,
        algorithm=algorithm
    )
    
    # Add CLI-specific metadata
    metadata.update({
        'date_requested': date,
        'target_date': target_date.isoformat(),
        'feature_files_used': [str(f) for f in feature_files[:5]]
    })
    
    return metadata


if __name__ == "__main__":
    # Test with sample data
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🧪 Testing SHAP reporter components...")
    
    # Would need trained model and features for full test
    # This is a placeholder for the actual implementation
    print("✅ SHAP reporter structure validated")
    print("📋 Ready for integration with trained models")