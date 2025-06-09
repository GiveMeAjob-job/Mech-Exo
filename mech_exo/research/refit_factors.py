"""
Factor Re-fitting Module

Implements factor weight re-fitting using various regression techniques
for strategy retraining and adaptation to changing market conditions.
"""

import logging
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict

try:
    from sklearn.linear_model import Ridge, Lasso, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RefitConfig:
    """Configuration for factor re-fitting"""
    method: str = "ridge"  # ridge, lasso, ols
    alpha: float = 1.0  # Regularization strength
    cv_folds: int = 5  # Cross-validation folds
    min_samples: int = 100  # Minimum samples required
    feature_selection: bool = True  # Whether to perform feature selection
    standardize: bool = True  # Whether to standardize features
    random_state: int = 42


@dataclass
class RefitResults:
    """Results from factor re-fitting"""
    method: str
    factor_weights: Dict[str, Any]
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    cross_validation_scores: List[float]
    in_sample_r2: float
    out_of_sample_r2: Optional[float]
    version: str
    created_at: str


class FactorRefitter:
    """
    Factor weight re-fitting using various regression techniques
    
    Supports Ridge, Lasso, and OLS regression with cross-validation
    and feature importance analysis.
    """
    
    def __init__(self, config: RefitConfig = None):
        """
        Initialize the factor refitter
        
        Args:
            config: Configuration for re-fitting
        """
        self.config = config or RefitConfig()
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, using fallback implementation")
        
        logger.info(f"FactorRefitter initialized with method: {self.config.method}")
    
    def prepare_feature_matrix(self, ohlc_df: pd.DataFrame, 
                             fundamental_df: pd.DataFrame,
                             news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature matrix from raw data
        
        Args:
            ohlc_df: OHLC price data
            fundamental_df: Fundamental data
            news_df: News/sentiment data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Preparing feature matrix for factor re-fitting")
        
        try:
            if ohlc_df.empty:
                logger.warning("No OHLC data available")
                return pd.DataFrame()
            
            # Start with OHLC data
            features_df = ohlc_df.copy()
            
            # Calculate technical indicators
            features_df = self._add_technical_features(features_df)
            
            # Add fundamental features if available
            if not fundamental_df.empty:
                features_df = self._add_fundamental_features(features_df, fundamental_df)
            
            # Add sentiment features if available
            if not news_df.empty:
                features_df = self._add_sentiment_features(features_df, news_df)
            
            # Calculate forward returns (target variable)
            features_df = self._add_forward_returns(features_df)
            
            # Remove rows with missing target
            features_df = features_df.dropna(subset=['forward_return'])
            
            logger.info(f"Feature matrix prepared: {features_df.shape} with {features_df.columns.tolist()}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to prepare feature matrix: {e}")
            return pd.DataFrame()
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        try:
            # Sort by symbol and date for proper calculations
            df = df.sort_values(['symbol', 'date'])
            
            # RSI (simplified)
            for symbol in df['symbol'].unique():
                mask = df['symbol'] == symbol
                symbol_data = df[mask].copy()
                
                # Price changes
                symbol_data['price_change'] = symbol_data['close'].pct_change()
                
                # Simple momentum (12-month, 1-month)
                if len(symbol_data) >= 252:  # At least 1 year of data
                    symbol_data['momentum_12_1'] = (
                        symbol_data['close'] / symbol_data['close'].shift(252) - 1 -
                        symbol_data['close'] / symbol_data['close'].shift(21) - 1
                    )
                
                # Volatility (20-day rolling)
                if len(symbol_data) >= 20:
                    symbol_data['volatility'] = symbol_data['price_change'].rolling(20).std()
                
                # Simple RSI approximation
                if len(symbol_data) >= 14:
                    gains = symbol_data['price_change'].where(symbol_data['price_change'] > 0, 0)
                    losses = -symbol_data['price_change'].where(symbol_data['price_change'] < 0, 0)
                    
                    avg_gains = gains.rolling(14).mean()
                    avg_losses = losses.rolling(14).mean()
                    
                    rs = avg_gains / (avg_losses + 1e-8)  # Avoid division by zero
                    symbol_data['rsi_14'] = 100 - (100 / (1 + rs))
                
                # Update main dataframe
                df.loc[mask, symbol_data.columns] = symbol_data
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to add technical features: {e}")
            return df
    
    def _add_fundamental_features(self, df: pd.DataFrame, 
                                fundamental_df: pd.DataFrame) -> pd.DataFrame:
        """Add fundamental features"""
        try:
            # Merge fundamental data (forward fill for missing dates)
            fundamental_df = fundamental_df.sort_values(['symbol', 'date'])
            
            # Basic merge on symbol and approximate date matching
            merged_df = pd.merge_asof(
                df.sort_values(['symbol', 'date']),
                fundamental_df.sort_values(['symbol', 'date']),
                on='date',
                by='symbol',
                direction='backward',
                suffixes=('', '_fund')
            )
            
            logger.info(f"Added fundamental features: {set(fundamental_df.columns) - {'symbol', 'date'}}")
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Failed to add fundamental features: {e}")
            return df
    
    def _add_sentiment_features(self, df: pd.DataFrame, 
                              news_df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment features"""
        try:
            # Aggregate news sentiment by symbol and date
            if 'sentiment' in news_df.columns:
                sentiment_agg = news_df.groupby(['symbol', 'date']).agg({
                    'sentiment': ['mean', 'std', 'count']
                }).reset_index()
                
                # Flatten column names
                sentiment_agg.columns = ['symbol', 'date', 'sentiment_mean', 'sentiment_std', 'news_count']
                
                # Merge with main dataframe
                df = pd.merge(df, sentiment_agg, on=['symbol', 'date'], how='left')
                
                # Fill missing sentiment with neutral
                df['sentiment_mean'] = df['sentiment_mean'].fillna(0.0)
                df['sentiment_std'] = df['sentiment_std'].fillna(0.1)
                df['news_count'] = df['news_count'].fillna(0)
                
                logger.info("Added sentiment features: sentiment_mean, sentiment_std, news_count")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to add sentiment features: {e}")
            return df
    
    def _add_forward_returns(self, df: pd.DataFrame, 
                           forward_days: int = 21) -> pd.DataFrame:
        """Add forward returns as target variable"""
        try:
            # Sort by symbol and date
            df = df.sort_values(['symbol', 'date'])
            
            # Calculate forward returns
            for symbol in df['symbol'].unique():
                mask = df['symbol'] == symbol
                symbol_data = df[mask].copy()
                
                # Forward return calculation
                symbol_data['forward_return'] = (
                    symbol_data['close'].shift(-forward_days) / symbol_data['close'] - 1
                )
                
                # Update main dataframe
                df.loc[mask, 'forward_return'] = symbol_data['forward_return']
            
            logger.info(f"Added forward returns with {forward_days} day horizon")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to add forward returns: {e}")
            return df
    
    def select_features(self, features_df: pd.DataFrame) -> List[str]:
        """
        Select relevant features for factor modeling
        
        Args:
            features_df: DataFrame with all features
            
        Returns:
            List of selected feature names
        """
        # Define core feature categories
        technical_features = [
            'momentum_12_1', 'volatility', 'rsi_14', 'price_change'
        ]
        
        fundamental_features = [
            'pe_ratio', 'return_on_equity', 'revenue_growth', 'earnings_growth',
            'debt_to_equity', 'current_ratio', 'gross_margin'
        ]
        
        sentiment_features = [
            'sentiment_mean', 'sentiment_std', 'news_count'
        ]
        
        # Select features that exist in the dataframe
        available_features = []
        
        for feature_list in [technical_features, fundamental_features, sentiment_features]:
            for feature in feature_list:
                if feature in features_df.columns:
                    available_features.append(feature)
        
        # Filter out features with too many missing values
        selected_features = []
        for feature in available_features:
            missing_pct = features_df[feature].isnull().sum() / len(features_df)
            if missing_pct < 0.5:  # Less than 50% missing
                selected_features.append(feature)
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        return selected_features
    
    def fit_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Dict[str, float]]:
        """
        Fit regression model with specified method
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (fitted_model, performance_metrics)
        """
        if not SKLEARN_AVAILABLE:
            return self._fit_simple_model(X, y)
        
        try:
            # Choose model based on method
            if self.config.method == "ridge":
                model = Ridge(alpha=self.config.alpha, random_state=self.config.random_state)
            elif self.config.method == "lasso":
                model = Lasso(alpha=self.config.alpha, random_state=self.config.random_state)
            elif self.config.method == "ols":
                model = LinearRegression()
            else:
                logger.warning(f"Unknown method {self.config.method}, using Ridge")
                model = Ridge(alpha=self.config.alpha, random_state=self.config.random_state)
            
            # Fit model
            model.fit(X, y)
            
            # Calculate performance metrics
            y_pred = model.predict(X)
            
            # Cross-validation
            cv_scores = []
            if len(X) >= self.config.cv_folds * 20:  # Enough data for CV
                tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
            
            performance_metrics = {
                'in_sample_r2': r2_score(y, y_pred),
                'in_sample_mse': mean_squared_error(y, y_pred),
                'mean_cv_score': np.mean(cv_scores) if cv_scores else 0.0,
                'std_cv_score': np.std(cv_scores) if cv_scores else 0.0,
                'n_features': X.shape[1],
                'n_samples': X.shape[0]
            }
            
            logger.info(f"Model fitted: R² = {performance_metrics['in_sample_r2']:.3f}, CV = {performance_metrics['mean_cv_score']:.3f}")
            
            return model, performance_metrics
            
        except Exception as e:
            logger.error(f"Failed to fit model: {e}")
            return None, {}
    
    def _fit_simple_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Dict[str, float]]:
        """Simple fallback model when sklearn not available"""
        try:
            # Simple OLS using numpy
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            
            # Predictions
            y_pred = X_with_intercept @ coefficients
            
            # Simple R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Create simple model object
            simple_model = {
                'coefficients': coefficients,
                'intercept': coefficients[0],
                'coef_': coefficients[1:],
                'method': 'simple_ols'
            }
            
            performance_metrics = {
                'in_sample_r2': r2,
                'in_sample_mse': np.mean((y - y_pred) ** 2),
                'mean_cv_score': 0.0,
                'std_cv_score': 0.0,
                'n_features': X.shape[1],
                'n_samples': X.shape[0]
            }
            
            logger.info(f"Simple model fitted: R² = {r2:.3f}")
            
            return simple_model, performance_metrics
            
        except Exception as e:
            logger.error(f"Failed to fit simple model: {e}")
            return None, {}
    
    def extract_factor_weights(self, model: Any, feature_names: List[str]) -> Dict[str, Any]:
        """
        Extract factor weights from fitted model and map to factor configuration
        
        Args:
            model: Fitted regression model
            feature_names: List of feature names used in model
            
        Returns:
            Dictionary with factor weights in config format
        """
        try:
            # Get coefficients
            if hasattr(model, 'coef_'):
                coefficients = model.coef_
            elif isinstance(model, dict) and 'coef_' in model:
                coefficients = model['coef_']
            else:
                logger.error("Cannot extract coefficients from model")
                return self._get_default_factors()
            
            # Create feature importance mapping
            feature_importance = dict(zip(feature_names, np.abs(coefficients)))
            
            # Map to factor categories
            factor_weights = {
                'fundamental': {},
                'technical': {},
                'sentiment': {}
            }
            
            # Define mapping from features to factors
            feature_to_factor = {
                # Technical factors
                'momentum_12_1': ('technical', 'momentum_12_1', 'higher_better'),
                'volatility': ('technical', 'volatility_ratio', 'lower_better'),
                'rsi_14': ('technical', 'rsi_14', 'mean_revert'),
                'price_change': ('technical', 'momentum_1m', 'higher_better'),
                
                # Fundamental factors  
                'pe_ratio': ('fundamental', 'pe_ratio', 'lower_better'),
                'return_on_equity': ('fundamental', 'return_on_equity', 'higher_better'),
                'revenue_growth': ('fundamental', 'revenue_growth', 'higher_better'),
                'earnings_growth': ('fundamental', 'earnings_growth', 'higher_better'),
                'debt_to_equity': ('fundamental', 'debt_to_equity', 'lower_better'),
                'current_ratio': ('fundamental', 'current_ratio', 'higher_better'),
                'gross_margin': ('fundamental', 'gross_margin', 'higher_better'),
                
                # Sentiment factors
                'sentiment_mean': ('sentiment', 'news_sentiment', 'higher_better'),
                'sentiment_std': ('sentiment', 'sentiment_volatility', 'lower_better'),
                'news_count': ('sentiment', 'news_volume', 'higher_better')
            }
            
            # Scale weights to sum to 100
            total_importance = sum(feature_importance.values())
            if total_importance == 0:
                logger.warning("Total feature importance is zero, using default weights")
                return self._get_default_factors()
            
            # Map features to factors with scaled weights
            for feature_name, importance in feature_importance.items():
                if feature_name in feature_to_factor:
                    category, factor_name, direction = feature_to_factor[feature_name]
                    weight = int(importance / total_importance * 100)
                    
                    if weight > 0:  # Only include factors with positive weight
                        factor_weights[category][factor_name] = {
                            'weight': max(weight, 1),  # Minimum weight of 1
                            'direction': direction
                        }
            
            # Ensure we have at least some factors
            if not any(factor_weights.values()):
                logger.warning("No factors extracted, using default weights")
                return self._get_default_factors()
            
            logger.info(f"Extracted factor weights: {factor_weights}")
            
            return factor_weights
            
        except Exception as e:
            logger.error(f"Failed to extract factor weights: {e}")
            return self._get_default_factors()
    
    def _get_default_factors(self) -> Dict[str, Any]:
        """Get default factor weights as fallback"""
        return {
            'fundamental': {
                'pe_ratio': {'weight': 15, 'direction': 'lower_better'},
                'return_on_equity': {'weight': 18, 'direction': 'higher_better'},
                'revenue_growth': {'weight': 15, 'direction': 'higher_better'},
                'earnings_growth': {'weight': 20, 'direction': 'higher_better'}
            },
            'technical': {
                'rsi_14': {'weight': 8, 'direction': 'mean_revert'},
                'momentum_12_1': {'weight': 12, 'direction': 'higher_better'},
                'volatility_ratio': {'weight': 6, 'direction': 'lower_better'}
            },
            'sentiment': {
                'news_sentiment': {'weight': 6, 'direction': 'higher_better'}
            }
        }
    
    def refit_factors(self, ohlc_df: pd.DataFrame,
                     fundamental_df: pd.DataFrame = None,
                     news_df: pd.DataFrame = None) -> RefitResults:
        """
        Main method to refit factor weights from data
        
        Args:
            ohlc_df: OHLC price data
            fundamental_df: Fundamental data (optional)
            news_df: News/sentiment data (optional)
            
        Returns:
            RefitResults with new factor weights and metrics
        """
        logger.info(f"Starting factor re-fitting with {self.config.method} method")
        
        # Handle missing dataframes
        fundamental_df = fundamental_df if fundamental_df is not None else pd.DataFrame()
        news_df = news_df if news_df is not None else pd.DataFrame()
        
        try:
            # Prepare feature matrix
            features_df = self.prepare_feature_matrix(ohlc_df, fundamental_df, news_df)
            
            if features_df.empty or len(features_df) < self.config.min_samples:
                logger.warning(f"Insufficient data for re-fitting: {len(features_df)} samples")
                return self._create_fallback_results()
            
            # Select features
            feature_names = self.select_features(features_df)
            
            if not feature_names:
                logger.warning("No features selected for re-fitting")
                return self._create_fallback_results()
            
            # Prepare X and y
            X = features_df[feature_names].fillna(0).values
            y = features_df['forward_return'].fillna(0).values
            
            # Standardize features if requested
            if self.config.standardize and SKLEARN_AVAILABLE:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            
            # Fit model
            model, performance_metrics = self.fit_model(X, y)
            
            if model is None:
                logger.error("Model fitting failed")
                return self._create_fallback_results()
            
            # Extract factor weights
            factor_weights = self.extract_factor_weights(model, feature_names)
            
            # Create feature importance dictionary
            if hasattr(model, 'coef_'):
                coefficients = model.coef_
            elif isinstance(model, dict) and 'coef_' in model:
                coefficients = model['coef_']
            else:
                coefficients = np.zeros(len(feature_names))
            
            feature_importance = dict(zip(feature_names, np.abs(coefficients)))
            
            # Create results
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            results = RefitResults(
                method=self.config.method,
                factor_weights=factor_weights,
                performance_metrics=performance_metrics,
                feature_importance=feature_importance,
                cross_validation_scores=performance_metrics.get('cv_scores', []),
                in_sample_r2=performance_metrics.get('in_sample_r2', 0.0),
                out_of_sample_r2=None,  # Will be calculated in validation
                version=version,
                created_at=datetime.now().isoformat()
            )
            
            logger.info(f"Factor re-fitting completed successfully: R² = {results.in_sample_r2:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Factor re-fitting failed: {e}")
            return self._create_fallback_results()
    
    def _create_fallback_results(self) -> RefitResults:
        """Create fallback results when re-fitting fails"""
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return RefitResults(
            method="fallback",
            factor_weights=self._get_default_factors(),
            performance_metrics={'in_sample_r2': 0.0, 'n_samples': 0},
            feature_importance={},
            cross_validation_scores=[],
            in_sample_r2=0.0,
            out_of_sample_r2=None,
            version=version,
            created_at=datetime.now().isoformat()
        )


def refit_strategy_factors(ohlc_df: pd.DataFrame,
                          fundamental_df: pd.DataFrame = None,
                          news_df: pd.DataFrame = None,
                          method: str = "ridge",
                          alpha: float = 1.0) -> RefitResults:
    """
    Convenience function to refit strategy factors
    
    Args:
        ohlc_df: OHLC price data
        fundamental_df: Fundamental data (optional)
        news_df: News/sentiment data (optional)
        method: Regression method ('ridge', 'lasso', 'ols')
        alpha: Regularization strength
        
    Returns:
        RefitResults with new factor weights and metrics
    """
    config = RefitConfig(method=method, alpha=alpha)
    refitter = FactorRefitter(config)
    
    return refitter.refit_factors(ohlc_df, fundamental_df, news_df)


if __name__ == "__main__":
    # Test the factor refitter
    print("🔄 Testing Factor Refitter...")
    
    try:
        # Create some test data
        dates = pd.date_range('2024-01-01', '2024-06-01', freq='D')
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Generate synthetic OHLC data
        ohlc_data = []
        for symbol in symbols:
            for date in dates:
                price = 100 + np.random.normal(0, 5)
                ohlc_data.append({
                    'symbol': symbol,
                    'date': date,
                    'open': price,
                    'high': price * 1.02,
                    'low': price * 0.98,
                    'close': price * (1 + np.random.normal(0, 0.01)),
                    'volume': 1000000
                })
        
        ohlc_df = pd.DataFrame(ohlc_data)
        
        # Test re-fitting
        results = refit_strategy_factors(ohlc_df, method="ridge", alpha=0.5)
        
        print(f"✅ Factor re-fitting test completed:")
        print(f"   • Method: {results.method}")
        print(f"   • Version: {results.version}")
        print(f"   • In-sample R²: {results.in_sample_r2:.3f}")
        print(f"   • Features: {len(results.feature_importance)}")
        print(f"   • Factor categories: {list(results.factor_weights.keys())}")
        
        # Show sample factors
        for category, factors in results.factor_weights.items():
            if factors:
                print(f"   • {category}: {list(factors.keys())}")
        
        print("🎉 Factor refitter test completed successfully!")
        
    except Exception as e:
        print(f"❌ Factor refitter test failed: {e}")
        import traceback
        traceback.print_exc()