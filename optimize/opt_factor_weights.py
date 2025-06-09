"""
Optuna-based Factor Weight Optimization

Uses Optuna to search optimal factor weights and thresholds to maximize
6-month rolling backtest Sharpe ratio with risk constraints.
"""

import logging
import sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import pandas as pd
import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TPESampler = None
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


class FactorWeightOptimizer:
    """
    Optuna-based factor weight optimization for systematic trading strategy
    """
    
    def __init__(self, study_file: str = "studies/factor_opt.db"):
        """
        Initialize the optimizer
        
        Args:
            study_file: Path to SQLite study database
        """
        # Check if optuna is available
        try:
            import optuna
            global optuna
            global OPTUNA_AVAILABLE
            OPTUNA_AVAILABLE = True
        except ImportError:
            raise ImportError("Optuna not installed. Run: pip install optuna optuna-dashboard")
        
        self.study_file = Path(study_file)
        self.study_file.parent.mkdir(exist_ok=True)
        
        # Optimization parameters
        self.lookback_years = 3  # Years of historical data
        self.rolling_months = 6  # Rolling backtest window
        self.max_dd_threshold = 0.12  # Maximum drawdown threshold (12%)
        
        # Factor configuration
        self.factor_categories = {
            'fundamental': ['pe_ratio', 'return_on_equity', 'revenue_growth', 'earnings_growth'],
            'technical': ['rsi_14', 'momentum_12_1', 'volatility_ratio'],
            'sentiment': ['news_sentiment', 'analyst_revisions']
        }
        
        logger.info(f"FactorWeightOptimizer initialized with study file: {self.study_file}")
    
    def create_study(self, study_name: str = "factor_weight_optimization"):
        """
        Create or load Optuna study
        
        Args:
            study_name: Name of the optimization study
            
        Returns:
            Optuna study object
        """
        # Create storage URL for SQLite
        storage_url = f"sqlite:///{self.study_file}"
        
        # Create study with TPE sampler
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="maximize",  # Maximize Sharpe ratio
            sampler=TPESampler(n_startup_trials=10, seed=42)
        )
        
        logger.info(f"Study created/loaded: {study_name}")
        logger.info(f"Storage: {storage_url}")
        
        return study
    
    def create_enhanced_study(self, study_name: str = "factor_weight_optimization"):
        """
        Create enhanced Optuna study with TPESampler and MedianPruner
        
        Args:
            study_name: Name of the optimization study
            
        Returns:
            Enhanced Optuna study object
        """
        # Create storage URL for SQLite
        storage_url = f"sqlite:///{self.study_file}"
        
        # Create enhanced sampler with multivariate TPE
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=10,  # Random trials before TPE kicks in
            multivariate=True,    # Consider parameter interactions
            seed=42               # Reproducible results
        )
        
        # Create pruner to stop unpromising trials early
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,   # Don't prune first 5 trials
            n_warmup_steps=3,     # Wait 3 steps before pruning
            interval_steps=1      # Prune every step
        )
        
        # Create enhanced study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="maximize",  # Maximize Sharpe ratio
            sampler=sampler,
            pruner=pruner
        )
        
        logger.info(f"Enhanced study created: {study_name}")
        logger.info(f"Sampler: {sampler.__class__.__name__}")
        logger.info(f"Pruner: {pruner.__class__.__name__}")
        
        return study
    
    def load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load historical factor scores and returns for optimization
        
        Returns:
            Dictionary with factor_scores and returns DataFrames
        """
        try:
            from mech_exo.datasource.storage import DataStorage
            from mech_exo.scoring.scorer import IdeaScorer
            
            logger.info(f"Loading {self.lookback_years} years of historical data...")
            
            # Calculate date range
            end_date = date.today()
            start_date = end_date - timedelta(days=self.lookback_years * 365)
            
            # Initialize data storage
            storage = DataStorage()
            
            # Load OHLC data
            ohlc_df = storage.get_ohlc_data(start_date=start_date, end_date=end_date)
            
            if ohlc_df.empty:
                logger.warning("No OHLC data available, generating synthetic data for testing...")
                # Generate synthetic data for testing
                return self._generate_synthetic_data()
            
            # Calculate returns
            ohlc_df['forward_return'] = ohlc_df.groupby('symbol')['close'].pct_change().shift(-1)
            
            # Load fundamental data  
            fund_df = storage.get_fundamental_data(start_date=start_date, end_date=end_date)
            
            # Generate factor scores using IdeaScorer
            scorer = IdeaScorer()
            
            # Create factor scores DataFrame
            factor_scores = self._calculate_factor_scores(ohlc_df, fund_df)
            
            # Create returns DataFrame aligned with factor scores
            returns_df = ohlc_df[['date', 'symbol', 'forward_return']].dropna()
            
            storage.close()
            
            logger.info(f"Loaded factor scores: {factor_scores.shape}")
            logger.info(f"Loaded returns: {returns_df.shape}")
            
            return {
                'factor_scores': factor_scores,
                'returns': returns_df
            }
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            logger.info("Falling back to synthetic data for testing...")
            return self._generate_synthetic_data()
    
    def _calculate_factor_scores(self, ohlc_df: pd.DataFrame, 
                                fund_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate factor scores from OHLC and fundamental data
        
        Args:
            ohlc_df: OHLC price data
            fund_df: Fundamental data
            
        Returns:
            DataFrame with factor scores
        """
        factor_data = []
        
        # Calculate technical factors from OHLC data
        for symbol_group in ohlc_df.groupby('symbol'):
            symbol = symbol_group[0]
            symbol_data = symbol_group[1].set_index('date').sort_index()
            
            if len(symbol_data) >= 20:  # Minimum data requirement
                # RSI-like factor
                returns = symbol_data['close'].pct_change()
                rsi_factor = -returns.rolling(14).std()  # Negative volatility
                
                # Momentum factor
                momentum = symbol_data['close'].pct_change(12) - symbol_data['close'].pct_change(1)
                
                # Volatility ratio
                vol_short = returns.rolling(5).std()
                vol_long = returns.rolling(20).std()
                vol_ratio = vol_short / vol_long
                
                # Add to factor data
                for date_idx in symbol_data.index:
                    if pd.notna(rsi_factor.get(date_idx)):
                        factor_data.append({
                            'date': date_idx,
                            'symbol': symbol,
                            'rsi_14': rsi_factor.get(date_idx, 0),
                            'momentum_12_1': momentum.get(date_idx, 0),
                            'volatility_ratio': vol_ratio.get(date_idx, 1)
                        })
        
        # Add fundamental factors if available
        if not fund_df.empty:
            for _, row in fund_df.iterrows():
                # Find matching factor data entry
                matching_entries = [f for f in factor_data 
                                  if f['date'] == row['date'] and f['symbol'] == row['symbol']]
                
                for entry in matching_entries:
                    entry['pe_ratio'] = 1 / (row.get('pe_ratio', 20) + 1e-6)  # Inverted P/E
                    entry['return_on_equity'] = row.get('return_on_equity', 0.1)
                    entry['revenue_growth'] = row.get('revenue_growth', 0.05)
                    entry['earnings_growth'] = row.get('earnings_growth', 0.05)
        
        # Fill missing fundamental factors with defaults
        for entry in factor_data:
            entry.setdefault('pe_ratio', 0.05)  # Default inverted P/E
            entry.setdefault('return_on_equity', 0.1)
            entry.setdefault('revenue_growth', 0.05)
            entry.setdefault('earnings_growth', 0.05)
            entry.setdefault('news_sentiment', 0.0)  # Neutral sentiment
            entry.setdefault('analyst_revisions', 0.0)
        
        factor_df = pd.DataFrame(factor_data)
        return factor_df
    
    def _generate_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic factor scores and returns for testing
        
        Returns:
            Dictionary with synthetic factor_scores and returns
        """
        logger.info("Generating synthetic data for optimization testing...")
        
        # Generate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=self.lookback_years * 365)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        
        np.random.seed(42)  # Reproducible results
        
        factor_data = []
        returns_data = []
        
        for symbol in symbols:
            for date_idx in dates:
                # Generate correlated factor scores
                base_factor = np.random.normal(0, 1)
                
                factor_scores = {
                    'date': date_idx,
                    'symbol': symbol,
                    'pe_ratio': base_factor * 0.3 + np.random.normal(0, 0.2),
                    'return_on_equity': base_factor * 0.4 + np.random.normal(0, 0.3),
                    'revenue_growth': base_factor * 0.2 + np.random.normal(0, 0.25),
                    'earnings_growth': base_factor * 0.35 + np.random.normal(0, 0.3),
                    'rsi_14': -base_factor * 0.15 + np.random.normal(0, 0.4),  # Mean reversion
                    'momentum_12_1': base_factor * 0.25 + np.random.normal(0, 0.3),
                    'volatility_ratio': -base_factor * 0.1 + np.random.normal(0.5, 0.2),
                    'news_sentiment': base_factor * 0.1 + np.random.normal(0, 0.3),
                    'analyst_revisions': base_factor * 0.2 + np.random.normal(0, 0.25)
                }
                
                factor_data.append(factor_scores)
                
                # Generate forward returns with some correlation to factors
                return_component = (
                    base_factor * 0.003 +  # Base correlation
                    np.random.normal(0, 0.02)  # Random noise
                )
                
                returns_data.append({
                    'date': date_idx,
                    'symbol': symbol,
                    'forward_return': return_component
                })
        
        factor_df = pd.DataFrame(factor_data)
        returns_df = pd.DataFrame(returns_data)
        
        logger.info(f"Generated synthetic factor scores: {factor_df.shape}")
        logger.info(f"Generated synthetic returns: {returns_df.shape}")
        
        return {
            'factor_scores': factor_df,
            'returns': returns_df
        }
    
    def objective_function(self, trial, 
                          factor_scores: pd.DataFrame, 
                          returns: pd.DataFrame) -> float:
        """
        Enhanced Optuna objective function with constraints
        
        Args:
            trial: Optuna trial object
            factor_scores: Historical factor scores
            returns: Historical returns
            
        Returns:
            6-month rolling Sharpe ratio to maximize (with constraints)
        """
        try:
            from research.obj_utils import ObjectiveUtils, BacktestEngine
            
            logger.info(f"Running trial {trial.number}...")
            
            # Initialize objective utilities
            obj_utils = ObjectiveUtils()
            backtest_engine = BacktestEngine()
            
            # Sample factor weights
            factor_weights = {}
            
            # Fundamental factors
            for factor in self.factor_categories['fundamental']:
                factor_weights[factor] = trial.suggest_float(f"weight_{factor}", -1.0, 1.0)
            
            # Technical factors  
            for factor in self.factor_categories['technical']:
                factor_weights[factor] = trial.suggest_float(f"weight_{factor}", -1.0, 1.0)
            
            # Sentiment factors
            for factor in self.factor_categories['sentiment']:
                factor_weights[factor] = trial.suggest_float(f"weight_{factor}", -1.0, 1.0)
            
            # Additional hyperparameters
            cash_pct = trial.suggest_float("cash_pct", 0.0, 0.3)  # 0-30% cash
            stop_loss_pct = trial.suggest_float("stop_loss_pct", 0.05, 0.25)  # 5-25% stop loss
            position_size_pct = trial.suggest_float("position_size_pct", 0.05, 0.15)  # 5-15% per position
            
            # Validate factor weights
            is_valid, error_msg = obj_utils.validate_factor_weights(factor_weights)
            if not is_valid:
                logger.warning(f"Trial {trial.number}: Invalid weights - {error_msg}")
                return -500.0  # Penalty for invalid weights
            
            hyperparameters = {
                'cash_pct': cash_pct,
                'stop_loss_pct': stop_loss_pct,
                'position_size_pct': position_size_pct
            }
            
            # Run enhanced backtest with transaction costs and risk management
            daily_returns = self._run_enhanced_backtest(
                factor_scores, returns, factor_weights,
                cash_pct, stop_loss_pct, position_size_pct,
                backtest_engine
            )
            
            if len(daily_returns) == 0:
                logger.warning(f"Trial {trial.number}: No backtest data generated")
                return -999.0
            
            # Create comprehensive performance summary
            performance = obj_utils.create_performance_summary(
                daily_returns, factor_weights, hyperparameters
            )
            
            # Calculate 6-month rolling Sharpe ratio
            rolling_sharpe = obj_utils.calculate_rolling_sharpe(daily_returns, window=126)
            
            # Use rolling Sharpe as primary objective (already includes penalties)
            final_score = performance['final_score']
            
            logger.info(f"Trial {trial.number}: Rolling Sharpe={rolling_sharpe:.3f}, "
                       f"Max DD={performance['max_drawdown']:.3f}, "
                       f"Violations={performance['constraint_violations']}, "
                       f"Final Score={final_score:.3f}")
            
            # Store comprehensive metrics as trial attributes
            trial.set_user_attr("max_drawdown", performance['max_drawdown'])
            trial.set_user_attr("total_return", performance['total_return'])
            trial.set_user_attr("volatility", performance['volatility'])
            trial.set_user_attr("rolling_sharpe", rolling_sharpe)
            trial.set_user_attr("constraints_satisfied", performance['constraints_satisfied'])
            trial.set_user_attr("constraint_violations", performance['constraint_violations'])
            trial.set_user_attr("factor_weights", factor_weights)
            trial.set_user_attr("hyperparameters", hyperparameters)
            
            # Log trial metrics to database for tracking
            try:
                study_db = str(self.study_file)
                obj_utils.log_trial_metrics(
                    trial.number, 
                    trial.study.study_name,
                    {
                        'trial_id': trial.number,
                        'sharpe_ratio': performance['sharpe_ratio'],
                        'max_drawdown': performance['max_drawdown'],
                        'total_return': performance['total_return'],
                        'volatility': performance['volatility'],
                        'status': 'completed',
                        'factor_weights': factor_weights,
                        'hyperparameters': hyperparameters,
                        'constraint_violations': performance['constraint_violations']
                    },
                    study_db
                )
            except Exception as e:
                logger.warning(f"Failed to log trial metrics: {e}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return -999.0  # Return very low score for failed trials
    
    def _run_enhanced_backtest(self, factor_scores: pd.DataFrame,
                              returns: pd.DataFrame,
                              factor_weights: Dict[str, float],
                              cash_pct: float,
                              stop_loss_pct: float,
                              position_size_pct: float,
                              backtest_engine) -> pd.Series:
        """
        Run enhanced backtest using BacktestEngine
        
        Args:
            factor_scores: Factor score data
            returns: Return data
            factor_weights: Factor weights to test
            cash_pct: Cash percentage
            stop_loss_pct: Stop loss percentage
            position_size_pct: Position size percentage
            backtest_engine: BacktestEngine instance
            
        Returns:
            Daily returns series
        """
        try:
            # Calculate composite scores
            composite_scores = self._calculate_composite_scores(factor_scores, factor_weights)
            
            # Generate trading signals
            signals = self._generate_signals(composite_scores, returns)
            
            # Run enhanced backtest with transaction costs
            daily_returns = backtest_engine.run_backtest(
                signals, returns, cash_pct, stop_loss_pct, position_size_pct
            )
            
            return daily_returns
            
        except Exception as e:
            logger.error(f"Enhanced backtest failed: {e}")
            return pd.Series(dtype=float)
    
    def _run_rolling_backtest(self, factor_scores: pd.DataFrame, 
                             returns: pd.DataFrame,
                             factor_weights: Dict[str, float],
                             cash_pct: float,
                             stop_loss_pct: float, 
                             position_size_pct: float) -> tuple:
        """
        Run 6-month rolling backtest with given parameters (legacy method)
        
        Args:
            factor_scores: Factor score data
            returns: Return data
            factor_weights: Factor weights to test
            cash_pct: Cash percentage
            stop_loss_pct: Stop loss percentage
            position_size_pct: Position size percentage
            
        Returns:
            Tuple of (sharpe_ratio, max_drawdown)
        """
        try:
            # Calculate composite scores
            composite_scores = self._calculate_composite_scores(factor_scores, factor_weights)
            
            # Generate trading signals
            signals = self._generate_signals(composite_scores, returns)
            
            # Run backtest simulation
            performance_metrics = self._simulate_backtest(
                signals, returns, cash_pct, stop_loss_pct, position_size_pct
            )
            
            return performance_metrics['sharpe_ratio'], performance_metrics['max_drawdown']
            
        except Exception as e:
            logger.error(f"Rolling backtest failed: {e}")
            return -1.0, 1.0  # Poor performance for failed backtests
    
    def _calculate_composite_scores(self, factor_scores: pd.DataFrame, 
                                   factor_weights: Dict[str, float]) -> pd.DataFrame:
        """Calculate composite scores using factor weights"""
        composite_data = []
        
        for _, row in factor_scores.iterrows():
            score = 0.0
            for factor, weight in factor_weights.items():
                if factor in row:
                    score += weight * row[factor]
            
            composite_data.append({
                'date': row['date'],
                'symbol': row['symbol'],
                'composite_score': score
            })
        
        return pd.DataFrame(composite_data)
    
    def _generate_signals(self, composite_scores: pd.DataFrame,
                         returns: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from composite scores"""
        # Merge scores with returns
        signals_df = composite_scores.merge(
            returns, on=['date', 'symbol'], how='inner'
        )
        
        # Check if forward_return column exists, if not add it
        if 'forward_return' not in signals_df.columns:
            # Use the return column and rename it
            if 'return' in returns.columns:
                returns_copy = returns.copy()
                returns_copy = returns_copy.rename(columns={'return': 'forward_return'})
                signals_df = composite_scores.merge(
                    returns_copy, on=['date', 'symbol'], how='inner'
                )
        
        # Rank symbols by score each day and take top 5
        signals_df['rank'] = signals_df.groupby('date')['composite_score'].rank(
            method='dense', ascending=False
        )
        
        # Generate boolean signals for top-ranked symbols
        signals_df['signal'] = signals_df['rank'] <= 5
        
        return signals_df
    
    def _simulate_backtest(self, signals: pd.DataFrame,
                          returns: pd.DataFrame,
                          cash_pct: float,
                          stop_loss_pct: float,
                          position_size_pct: float) -> Dict[str, float]:
        """
        Simulate backtest performance
        
        Returns:
            Dictionary with performance metrics
        """
        # Simple simulation - in practice would use vectorbt
        daily_returns = []
        equity_curve = [100000]  # Starting capital
        peak = 100000
        max_dd = 0.0
        
        # Group by date and simulate daily performance
        for date_group in signals.groupby('date'):
            date_signals = date_group[1]
            active_signals = date_signals[date_signals['signal']]
            
            if len(active_signals) > 0:
                # Equal weight positions
                weight_per_position = (1 - cash_pct) / len(active_signals)
                weight_per_position = min(weight_per_position, position_size_pct)
                
                # Calculate daily return
                daily_return = (active_signals['forward_return'] * weight_per_position).sum()
                
                # Apply stop loss (simplified)
                if daily_return < -stop_loss_pct:
                    daily_return = -stop_loss_pct
                
                daily_returns.append(daily_return)
                
                # Update equity curve
                new_equity = equity_curve[-1] * (1 + daily_return)
                equity_curve.append(new_equity)
                
                # Track drawdown
                if new_equity > peak:
                    peak = new_equity
                else:
                    drawdown = (peak - new_equity) / peak
                    max_dd = max(max_dd, drawdown)
            else:
                daily_returns.append(0.0)
                equity_curve.append(equity_curve[-1])
        
        if len(daily_returns) == 0:
            return {'sharpe_ratio': -1.0, 'max_drawdown': 1.0}
        
        # Calculate Sharpe ratio
        daily_returns = np.array(daily_returns)
        if np.std(daily_returns) == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'total_return': (equity_curve[-1] / equity_curve[0]) - 1
        }


def create_study_db(study_file: str = "studies/factor_opt.db") -> bool:
    """
    Create Optuna study database
    
    Args:
        study_file: Path to study database file
        
    Returns:
        True if successful
    """
    try:
        # Check if optuna is available
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna not installed. Run: pip install optuna optuna-dashboard")
        
        optimizer = FactorWeightOptimizer(study_file)
        study = optimizer.create_study()
        
        logger.info(f"Study database created: {study_file}")
        logger.info(f"Study name: {study.study_name}")
        logger.info(f"Direction: {study.direction}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create study database: {e}")
        return False


if __name__ == "__main__":
    # Test the optimizer
    print("🧪 Testing FactorWeightOptimizer...")
    
    # Create study
    success = create_study_db()
    if success:
        print("✅ Study database created successfully")
        
        # Test data loading
        optimizer = FactorWeightOptimizer()
        data = optimizer.load_historical_data()
        print(f"✅ Data loaded: {len(data['factor_scores'])} factor records")
        
        print("🎯 Ready for optimization!")
    else:
        print("❌ Failed to create study database")