"""
Alpha-Decay Engine for Factor Health Monitoring

Measures how quickly each factor's predictive power is fading using rolling
information coefficient analysis and half-life calculations.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings

# Try to import scipy, fall back to built-in implementation
try:
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Spearman rank correlation coefficient
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Spearman correlation coefficient
    """
    if SCIPY_AVAILABLE:
        corr, _ = spearmanr(x, y)
        return corr if not np.isnan(corr) else 0.0
    else:
        # Built-in implementation using rank correlation
        try:
            # Convert to pandas Series for ranking
            x_series = pd.Series(x)
            y_series = pd.Series(y)
            
            # Calculate ranks
            x_ranks = x_series.rank()
            y_ranks = y_series.rank()
            
            # Calculate Pearson correlation of ranks
            correlation = x_ranks.corr(y_ranks)
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.warning(f"Spearman correlation calculation failed: {e}")
            return 0.0


class AlphaDecayEngine:
    """
    Calculates alpha decay metrics for trading factors
    
    Measures the half-life of factor predictive power using rolling information
    coefficients (IC) and exponential decay modeling.
    """
    
    def __init__(self, window: int = 252, min_periods: int = 60):
        """
        Initialize alpha decay engine
        
        Args:
            window: Rolling window for calculations (default: 252 trading days)
            min_periods: Minimum periods required for calculation
        """
        self.window = window
        self.min_periods = min_periods
        logger.info(f"AlphaDecayEngine initialized: window={window}, min_periods={min_periods}")
    
    def calc_information_coefficient(self, factor_series: pd.Series, 
                                   returns_series: pd.Series) -> pd.Series:
        """
        Calculate rolling information coefficient (IC) between factor and forward returns
        
        Args:
            factor_series: Factor values indexed by date
            returns_series: Forward returns indexed by date
            
        Returns:
            Series of rolling IC values
        """
        try:
            # Align series by date
            aligned_df = pd.DataFrame({
                'factor': factor_series,
                'returns': returns_series
            }).dropna()
            
            if len(aligned_df) < self.min_periods:
                logger.warning(f"Insufficient data for IC calculation: {len(aligned_df)} < {self.min_periods}")
                return pd.Series(dtype=float)
            
            # Calculate rolling Spearman correlation
            ic_series = []
            dates = []
            
            for i in range(self.window, len(aligned_df) + 1):
                window_data = aligned_df.iloc[i-self.window:i]
                
                if len(window_data) >= self.min_periods:
                    # Use Spearman rank correlation for IC
                    ic_value = _spearman_correlation(
                        window_data['factor'].values, 
                        window_data['returns'].values
                    )
                    
                    ic_series.append(ic_value)
                    dates.append(window_data.index[-1])
            
            return pd.Series(ic_series, index=dates, name='ic')
            
        except Exception as e:
            logger.error(f"Failed to calculate IC: {e}")
            return pd.Series(dtype=float)
    
    def calc_half_life(self, factor_series: pd.Series, returns_series: pd.Series,
                      window: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate half-life of factor alpha decay
        
        Args:
            factor_series: Factor values indexed by date
            returns_series: Forward returns indexed by date
            window: Override default window size
            
        Returns:
            Dictionary with half_life (days), latest_ic, and metadata
        """
        if window is not None:
            original_window = self.window
            self.window = window
        
        try:
            # Calculate IC time series
            ic_series = self.calc_information_coefficient(factor_series, returns_series)
            
            if len(ic_series) < 10:  # Need minimum observations for decay calc
                logger.warning(f"Insufficient IC observations for half-life: {len(ic_series)}")
                return {
                    'half_life': np.nan,
                    'latest_ic': np.nan,
                    'ic_observations': len(ic_series),
                    'status': 'insufficient_data'
                }
            
            # Calculate half-life using exponential decay model
            half_life = self._estimate_half_life(ic_series)
            
            # Get latest IC value
            latest_ic = ic_series.iloc[-1] if len(ic_series) > 0 else np.nan
            
            # Calculate additional metrics
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ic_trend = self._calculate_trend(ic_series)
            
            result = {
                'half_life': half_life,
                'latest_ic': latest_ic,
                'ic_observations': len(ic_series),
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ic_trend': ic_trend,
                'status': 'success'
            }
            
            logger.debug(f"Half-life calculation completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate half-life: {e}")
            return {
                'half_life': np.nan,
                'latest_ic': np.nan,
                'ic_observations': 0,
                'status': f'error: {e}'
            }
        
        finally:
            # Restore original window if modified
            if window is not None:
                self.window = original_window
    
    def _estimate_half_life(self, ic_series: pd.Series) -> float:
        """
        Estimate half-life of IC decay using exponential model
        
        Uses multiple methods to estimate decay rate:
        1. Linear regression on log(abs(IC)) vs time
        2. Autocorrelation-based method
        3. Ratio-based method as fallback
        
        Args:
            ic_series: Time series of IC values
            
        Returns:
            Half-life in days (clipped to max 90 days)
        """
        try:
            if len(ic_series) < 5:
                return 90.0  # Max value for insufficient data
            
            # Method 1: Linear regression on log(abs(IC))
            try:
                # Use absolute values to handle negative IC
                abs_ic = np.abs(ic_series.values)
                abs_ic = np.maximum(abs_ic, 1e-6)  # Avoid log(0)
                
                log_ic = np.log(abs_ic)
                x = np.arange(len(log_ic))
                
                # Linear regression: log(IC) = a + b*t
                slope = np.polyfit(x, log_ic, 1)[0]
                
                if slope < 0:  # Decaying
                    # half_life = ln(0.5) / slope
                    half_life = -np.log(0.5) / abs(slope)
                    half_life = np.clip(half_life, 1.0, 90.0)
                    return half_life
                    
            except Exception:
                pass
            
            # Method 2: Autocorrelation approach
            try:
                # Calculate lag-1 autocorrelation
                ic_shifted = ic_series.shift(1)
                autocorr = ic_series.corr(ic_shifted)
                
                if not np.isnan(autocorr) and 0 < autocorr < 1:
                    # half_life = -ln(2) / ln(autocorr)
                    half_life = -np.log(0.5) / np.log(autocorr)
                    half_life = np.clip(half_life, 1.0, 90.0)
                    return half_life
                    
            except Exception:
                pass
            
            # Method 3: Ratio-based fallback
            return self._estimate_half_life_from_ratios(ic_series)
            
        except Exception as e:
            logger.warning(f"Half-life estimation failed: {e}")
            return 90.0  # Conservative default
    
    def _estimate_half_life_from_ratios(self, ic_series: pd.Series) -> float:
        """
        Alternative half-life estimation using IC value ratios
        
        Args:
            ic_series: Time series of IC values
            
        Returns:
            Half-life in days
        """
        try:
            # Calculate rolling ratios of IC values
            ratios = []
            for i in range(1, min(len(ic_series), 10)):  # Use last 10 observations
                if ic_series.iloc[-i-1] != 0:
                    ratio = abs(ic_series.iloc[-i] / ic_series.iloc[-i-1])
                    if 0.1 <= ratio <= 10:  # Filter out extreme ratios
                        ratios.append(ratio)
            
            if not ratios:
                return 90.0
            
            # Average ratio gives decay per period
            avg_ratio = np.mean(ratios)
            
            # Convert to half-life: half_life = ln(0.5) / ln(avg_ratio)
            if avg_ratio > 0 and avg_ratio != 1:
                half_life = np.log(0.5) / np.log(avg_ratio)
                return np.clip(abs(half_life), 1.0, 90.0)
            else:
                return 90.0
                
        except Exception as e:
            logger.warning(f"Ratio-based half-life estimation failed: {e}")
            return 90.0
    
    def _calculate_trend(self, ic_series: pd.Series) -> float:
        """
        Calculate trend in IC values (slope of linear regression)
        
        Args:
            ic_series: Time series of IC values
            
        Returns:
            Slope coefficient (positive = improving, negative = degrading)
        """
        try:
            if len(ic_series) < 3:
                return 0.0
            
            # Use last 20 observations for trend
            recent_ic = ic_series.tail(20)
            x = np.arange(len(recent_ic))
            y = recent_ic.values
            
            # Simple linear regression
            slope = np.polyfit(x, y, 1)[0]
            return slope
            
        except Exception as e:
            logger.warning(f"Trend calculation failed: {e}")
            return 0.0
    
    def calc_multiple_factors(self, factor_data: pd.DataFrame, 
                            returns_series: pd.Series) -> pd.DataFrame:
        """
        Calculate alpha decay for multiple factors
        
        Args:
            factor_data: DataFrame with factors as columns, dates as index
            returns_series: Forward returns series
            
        Returns:
            DataFrame with decay metrics for each factor
        """
        results = []
        
        for factor_name in factor_data.columns:
            logger.info(f"Calculating alpha decay for factor: {factor_name}")
            
            factor_series = factor_data[factor_name]
            decay_metrics = self.calc_half_life(factor_series, returns_series)
            
            # Add factor name to results
            decay_metrics['factor_name'] = factor_name
            decay_metrics['calculation_date'] = datetime.now()
            
            results.append(decay_metrics)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by half-life (shortest first - most concerning)
        if 'half_life' in results_df.columns:
            results_df = results_df.sort_values('half_life')
        
        logger.info(f"Alpha decay calculated for {len(results)} factors")
        return results_df


def calc_half_life(factor_series: pd.Series, returns_series: pd.Series, 
                  window: int = 252) -> Dict[str, float]:
    """
    Convenience function to calculate factor half-life
    
    Args:
        factor_series: Factor values indexed by date
        returns_series: Forward returns indexed by date
        window: Rolling window for IC calculation (default: 252 days)
        
    Returns:
        Dictionary with half_life and latest_ic
    """
    engine = AlphaDecayEngine(window=window)
    return engine.calc_half_life(factor_series, returns_series)


def generate_synthetic_factor_data(n_days: int = 500, half_life: float = 30.0, 
                                 noise_level: float = 0.1) -> Tuple[pd.Series, pd.Series]:
    """
    Generate synthetic factor data with known half-life for testing
    
    Args:
        n_days: Number of trading days to generate
        half_life: Known half-life in days for validation
        noise_level: Amount of random noise to add
        
    Returns:
        Tuple of (factor_series, returns_series)
    """
    # Note: Random seed should be set externally for reproducible tests
    
    # Generate dates
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
    
    # Generate factor with exponential decay in correlation strength
    decay_rate = np.log(0.5) / half_life  # Note: positive decay_rate for decay
    correlation_strength = np.exp(decay_rate * np.arange(n_days))
    
    # Generate factor values (standardized)
    factor_values = np.random.normal(0, 1, n_days)
    
    # Generate returns that decay in correlation with factor
    base_noise = np.random.normal(0, 0.02, n_days)
    correlated_component = factor_values * 0.01 * correlation_strength
    returns_values = correlated_component + base_noise
    
    # Add factor noise
    factor_values += np.random.normal(0, noise_level, n_days)
    
    # Create series
    factor_series = pd.Series(factor_values, index=dates, name='test_factor')
    returns_series = pd.Series(returns_values, index=dates, name='forward_returns')
    
    return factor_series, returns_series


if __name__ == "__main__":
    # Test the alpha decay engine
    print("🧪 Testing Alpha Decay Engine...")
    
    try:
        # Generate synthetic data with known half-life
        known_half_life = 30.0
        np.random.seed(42)  # Set seed for reproducible test
        factor_series, returns_series = generate_synthetic_factor_data(
            n_days=300, 
            half_life=known_half_life,
            noise_level=0.05
        )
        
        print(f"Generated synthetic data: {len(factor_series)} observations")
        print(f"Known half-life: {known_half_life} days")
        
        # Calculate half-life
        engine = AlphaDecayEngine(window=252, min_periods=30)
        results = engine.calc_half_life(factor_series, returns_series)
        
        print(f"\n📊 Alpha Decay Results:")
        print(f"   • Estimated half-life: {results['half_life']:.1f} days")
        print(f"   • Latest IC: {results['latest_ic']:.3f}")
        print(f"   • IC observations: {results['ic_observations']}")
        print(f"   • IC mean: {results['ic_mean']:.3f}")
        print(f"   • IC trend: {results['ic_trend']:.6f}")
        print(f"   • Status: {results['status']}")
        
        # Validate against known value
        error_pct = abs(results['half_life'] - known_half_life) / known_half_life * 100
        print(f"\n✅ Validation:")
        print(f"   • Error vs known: {error_pct:.1f}%")
        print(f"   • Test result: {'PASS' if error_pct < 50 else 'FAIL'}")
        
        # Test with multiple factors
        print(f"\n🔄 Testing multiple factors...")
        
        # Create multiple synthetic factors
        factor_data = pd.DataFrame()
        returns_data = []
        
        for i, hl in enumerate([15, 30, 60]):  # Different half-lives
            # Use different seeds for different factors
            np.random.seed(42 + i)
            factor, returns = generate_synthetic_factor_data(
                n_days=400, half_life=hl, noise_level=0.05
            )
            factor_data[f'factor_{i+1}'] = factor
            returns_data.append(returns)
        
        # Use average returns
        avg_returns = pd.concat(returns_data, axis=1).mean(axis=1)
        
        # Calculate decay for all factors
        multi_results = engine.calc_multiple_factors(factor_data, avg_returns)
        
        print(f"✅ Multiple factor results:")
        for _, row in multi_results.iterrows():
            print(f"   • {row['factor_name']}: half-life={row['half_life']:.1f}d, IC={row['latest_ic']:.3f}")
        
        print(f"\n🎉 Alpha decay engine test completed successfully!")
        
    except Exception as e:
        print(f"❌ Alpha decay engine test failed: {e}")
        import traceback
        traceback.print_exc()