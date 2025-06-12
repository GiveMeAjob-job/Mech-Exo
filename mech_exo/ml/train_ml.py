"""
ML model training pipeline for alpha signal generation.

Supports LightGBM and XGBoost with hyperparameter optimization,
time-series cross-validation, and comprehensive metrics tracking.
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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class MLTrainer:
    """
    Machine learning trainer for alpha signal models.
    
    Supports LightGBM and XGBoost with time-series cross-validation,
    hyperparameter optimization, and comprehensive performance metrics.
    """
    
    def __init__(self, algorithm: str = "lightgbm", random_state: int = 42):
        """
        Initialize ML trainer.
        
        Args:
            algorithm: "lightgbm" or "xgboost"
            random_state: Random seed for reproducibility
        """
        self.algorithm = algorithm.lower()
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.cv_scores = None
        
        # Validate algorithm
        if self.algorithm not in ["lightgbm", "xgboost"]:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'lightgbm' or 'xgboost'")
            
        # Import the required library
        if self.algorithm == "lightgbm":
            try:
                import lightgbm as lgb
                self.lgb = lgb
            except ImportError:
                raise ImportError("LightGBM not installed. Run: pip install lightgbm")
                
        elif self.algorithm == "xgboost":
            try:
                import xgboost as xgb
                self.xgb = xgb
            except ImportError:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    def prepare_training_data(self, features_df: pd.DataFrame, 
                            forward_days: int = 10,
                            return_threshold: float = 0.0) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels for training.
        
        Args:
            features_df: Feature matrix with columns: symbol, feature_date, ...
            forward_days: Days ahead to calculate forward returns
            return_threshold: Threshold for binary classification (default: 0%)
            
        Returns:
            (X, y): Features and binary labels
        """
        logger.info(f"Preparing training data with {forward_days}-day forward returns")
        
        # Sort by symbol and date
        df = features_df.copy()
        df['feature_date'] = pd.to_datetime(df['feature_date'])
        df = df.sort_values(['symbol', 'feature_date']).reset_index(drop=True)
        
        # Calculate forward returns for each symbol
        forward_returns = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Get price series
            prices = symbol_data['price'].values
            dates = symbol_data['feature_date'].values
            
            # Calculate forward returns
            symbol_forward_returns = np.full(len(prices), np.nan)
            
            for i in range(len(prices) - forward_days):
                current_price = prices[i]
                future_price = prices[i + forward_days]
                
                if current_price > 0 and future_price > 0:
                    symbol_forward_returns[i] = (future_price / current_price - 1)
            
            forward_returns.extend(symbol_forward_returns)
        
        df['forward_return'] = forward_returns
        
        # Remove rows with missing forward returns or features
        df = df.dropna().reset_index(drop=True)
        
        # Create binary labels
        y = (df['forward_return'] > return_threshold).astype(int)
        
        # Prepare feature matrix
        feature_cols = [col for col in df.columns 
                       if col not in ['symbol', 'feature_date', 'feature_count', 'forward_return']]
        X = df[feature_cols].copy()
        
        logger.info(f"Training data prepared: {len(X)} samples, {len(feature_cols)} features")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")
        logger.info(f"Forward return stats: mean={df['forward_return'].mean():.4f}, "
                   f"std={df['forward_return'].std():.4f}")
        
        return X, y
    
    def get_hyperparameter_grid(self) -> Dict:
        """Get hyperparameter search space for the selected algorithm."""
        
        if self.algorithm == "lightgbm":
            return {
                'num_leaves': [31, 63, 127, 255],
                'max_depth': [3, 5, 7, 10, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [100, 200, 500, 1000],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0],
                'min_child_samples': [20, 50, 100]
            }
        
        elif self.algorithm == "xgboost":
            return {
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [100, 200, 500, 1000],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.5, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [1, 1.5, 2.0],
                'min_child_weight': [1, 3, 5]
            }
    
    def create_model(self, **params) -> object:
        """Create model instance with given parameters."""
        
        base_params = {
            'random_state': self.random_state,
            'n_jobs': 1,  # Single job for reproducibility
            'verbosity': 0  # Quiet training
        }
        
        if self.algorithm == "lightgbm":
            base_params.update({
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'force_col_wise': True
            })
            base_params.update(params)
            return self.lgb.LGBMClassifier(**base_params)
            
        elif self.algorithm == "xgboost":
            base_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist'
            })
            base_params.update(params)
            return self.xgb.XGBClassifier(**base_params)
    
    def train_with_cv(self, X: pd.DataFrame, y: pd.Series, 
                     cv_folds: int = 5, n_iter: int = 30) -> Dict:
        """
        Train model with time-series cross-validation and hyperparameter search.
        
        Args:
            X: Feature matrix
            y: Binary labels
            cv_folds: Number of CV folds
            n_iter: Number of hyperparameter search iterations
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info(f"Starting {self.algorithm} training with {cv_folds}-fold CV")
        logger.info(f"Hyperparameter search: {n_iter} iterations")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Log fold information
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"Fold {fold_idx}/{cv_folds}: train={len(train_idx)}, val={len(val_idx)}")
        
        # Create base model
        base_model = self.create_model()
        
        # Hyperparameter search
        param_grid = self.get_hyperparameter_grid()
        
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=tscv,
            scoring='roc_auc',
            random_state=self.random_state,
            n_jobs=1,  # Single job for reproducibility
            verbose=1
        )
        
        logger.info("Starting hyperparameter search...")
        search.fit(X, y)
        
        # Store results
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.cv_scores = search.cv_results_
        
        logger.info(f"Best CV AUC: {search.best_score_:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(X, y, cv_folds)
        
        return {
            'best_auc': search.best_score_,
            'best_params': self.best_params,
            'metrics': metrics,
            'algorithm': self.algorithm,
            'training_samples': len(X),
            'features': len(X.columns),
            'cv_folds': cv_folds,
            'n_iter': n_iter
        }
    
    def _calculate_metrics(self, X: pd.DataFrame, y: pd.Series, cv_folds: int) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        # Time series split for final evaluation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        auc_scores = []
        ic_scores = []
        accuracy_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model on fold
            fold_model = self.create_model(**self.best_params)
            fold_model.fit(X_train, y_train)
            
            # Predictions
            y_pred_proba = fold_model.predict_proba(X_val)[:, 1]
            y_pred = fold_model.predict(X_val)
            
            # Metrics
            auc = roc_auc_score(y_val, y_pred_proba)
            accuracy = accuracy_score(y_val, y_pred)
            
            # Information Coefficient (IC)
            if len(y_pred_proba) > 1:
                ic, _ = spearmanr(y_pred_proba, y_val)
                if np.isnan(ic):
                    ic = 0.0
            else:
                ic = 0.0
            
            auc_scores.append(auc)
            ic_scores.append(ic)
            accuracy_scores.append(accuracy)
        
        return {
            'mean_auc': np.mean(auc_scores),
            'std_auc': np.std(auc_scores),
            'mean_ic': np.mean(ic_scores),
            'std_ic': np.std(ic_scores),
            'mean_accuracy': np.mean(accuracy_scores),
            'std_accuracy': np.std(accuracy_scores),
            'fold_aucs': auc_scores,
            'fold_ics': ic_scores
        }
    
    def save_model(self, model_dir: str = "models") -> str:
        """
        Save trained model to disk.
        
        Args:
            model_dir: Directory to save model
            
        Returns:
            Path to saved model file
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Create model directory
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.algorithm == "lightgbm":
            model_file = model_path / f"lgbm_{timestamp}.txt"
            self.model.booster_.save_model(str(model_file))
            
        elif self.algorithm == "xgboost":
            model_file = model_path / f"xgb_{timestamp}.json"
            self.model.save_model(str(model_file))
        
        logger.info(f"Model saved to: {model_file}")
        return str(model_file)
    
    def save_metrics(self, metrics: Dict, metrics_dir: str = "models") -> str:
        """
        Save training metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary from training
            metrics_dir: Directory to save metrics
            
        Returns:
            Path to saved metrics file
        """
        # Create metrics directory
        metrics_path = Path(metrics_dir)
        metrics_path.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        enhanced_metrics = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': self.algorithm,
            **metrics
        }
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = metrics_path / f"metrics_{self.algorithm}_{timestamp}.json"
        
        # Save to JSON
        with open(metrics_file, 'w') as f:
            json.dump(enhanced_metrics, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to: {metrics_file}")
        return str(metrics_file)


def train_ml_cli(algorithm: str = "lightgbm", 
                lookback: str = "3y",
                cv_folds: int = 5,
                n_iter: int = 30,
                seed: int = 42,
                features_dir: str = "data/features",
                models_dir: str = "models") -> Dict:
    """
    CLI function for ML model training.
    
    Args:
        algorithm: "lightgbm" or "xgboost"
        lookback: Training data lookback period (e.g., "3y", "1y", "180d")
        cv_folds: Number of cross-validation folds
        n_iter: Number of hyperparameter search iterations
        seed: Random seed
        features_dir: Directory containing feature files
        models_dir: Directory to save models and metrics
        
    Returns:
        Training results dictionary
    """
    logger.info(f"🚀 Starting ML training pipeline...")
    logger.info(f"   Algorithm: {algorithm}")
    logger.info(f"   Lookback: {lookback}")
    logger.info(f"   CV folds: {cv_folds}")
    logger.info(f"   Hyperparameter iterations: {n_iter}")
    logger.info(f"   Random seed: {seed}")
    
    # Parse lookback period
    end_date = date.today()
    
    if lookback.endswith('y'):
        years = int(lookback[:-1])
        start_date = end_date - timedelta(days=years * 365)
    elif lookback.endswith('d'):
        days = int(lookback[:-1])
        start_date = end_date - timedelta(days=days)
    else:
        raise ValueError(f"Invalid lookback format: {lookback}. Use '3y', '180d', etc.")
    
    logger.info(f"   Date range: {start_date} to {end_date}")
    
    # Load feature files
    features_path = Path(features_dir)
    if not features_path.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    feature_files = list(features_path.glob("features_*.csv"))
    if not feature_files:
        feature_files = list(features_path.glob("features_*.parquet"))
    
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {features_dir}")
    
    logger.info(f"   Found {len(feature_files)} feature files")
    
    # Load and combine features
    all_features = []
    
    for file_path in feature_files:
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            else:  # parquet
                df = pd.read_parquet(file_path)
            
            # Filter by date range
            df['feature_date'] = pd.to_datetime(df['feature_date'])
            df = df[(df['feature_date'] >= pd.Timestamp(start_date)) & 
                   (df['feature_date'] <= pd.Timestamp(end_date))]
            
            if len(df) > 0:
                all_features.append(df)
                
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No valid feature data found in date range")
    
    # Combine all features
    features_df = pd.concat(all_features, ignore_index=True)
    logger.info(f"   Total feature records: {len(features_df)}")
    logger.info(f"   Unique symbols: {features_df['symbol'].nunique()}")
    logger.info(f"   Date range: {features_df['feature_date'].min()} to {features_df['feature_date'].max()}")
    
    # Initialize trainer
    trainer = MLTrainer(algorithm=algorithm, random_state=seed)
    
    # Prepare training data
    X, y = trainer.prepare_training_data(features_df)
    
    if len(X) < 100:
        raise ValueError(f"Insufficient training data: {len(X)} samples. Need at least 100.")
    
    # Train model
    results = trainer.train_with_cv(X, y, cv_folds=cv_folds, n_iter=n_iter)
    
    # Save model and metrics
    model_file = trainer.save_model(models_dir)
    metrics_file = trainer.save_metrics(results, models_dir)
    
    # Update results with file paths
    results.update({
        'model_file': model_file,
        'metrics_file': metrics_file,
        'feature_files_loaded': len(all_features),
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat()
    })
    
    logger.info(f"✅ Training completed successfully!")
    logger.info(f"   Best AUC: {results['best_auc']:.4f}")
    logger.info(f"   Model saved: {model_file}")
    logger.info(f"   Metrics saved: {metrics_file}")
    
    return results


if __name__ == "__main__":
    # Test with sample data
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        results = train_ml_cli(
            algorithm="lightgbm",
            lookback="90d",
            cv_folds=2,
            n_iter=5
        )
        print(f"\n🎉 Training test completed!")
        print(f"Best AUC: {results['best_auc']:.4f}")
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        sys.exit(1)