"""
Objective function utilities for Optuna factor weight optimization

Contains utilities for backtesting, constraint handling, and metrics calculation
used in the optimization objective function.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class ObjectiveUtils:
    """
    Utilities for Optuna objective function evaluation
    """
    
    def __init__(self, db_file: str = "data/mech_exo.duckdb"):
        """
        Initialize objective utilities
        
        Args:
            db_file: Path to DuckDB database file
        """
        self.db_file = db_file
        self.max_dd_threshold = 0.12  # 12% max drawdown constraint
        self.min_sharpe_threshold = 0.5  # Minimum acceptable Sharpe ratio
        
    def log_trial_metrics(self, trial_number: int, study_name: str, 
                         metrics: Dict, study_db: str) -> bool:
        """
        Log trial metrics to DuckDB table for tracking
        
        Args:
            trial_number: Trial number
            study_name: Name of optimization study
            metrics: Dictionary of trial metrics
            study_db: Path to study database
            
        Returns:
            True if successful
        """
        try:
            import duckdb
            
            # Connect to main database
            conn = duckdb.connect(self.db_file)
            
            # Create optuna_trials table if it doesn't exist
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS optuna_trials (
                trial_id INTEGER,
                study_name VARCHAR,
                trial_number INTEGER,
                sharpe_ratio DOUBLE,
                max_drawdown DOUBLE,
                total_return DOUBLE,
                volatility DOUBLE,
                trial_status VARCHAR,
                factor_weights JSON,
                hyperparameters JSON,
                calculation_date TIMESTAMP,
                constraint_violations INTEGER,
                PRIMARY KEY (study_name, trial_number)
            )
            """
            
            conn.execute(create_table_sql)
            
            # Insert trial metrics
            insert_sql = """
            INSERT OR REPLACE INTO optuna_trials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            trial_data = (
                metrics.get('trial_id', trial_number),
                study_name,
                trial_number,
                metrics.get('sharpe_ratio', 0.0),
                metrics.get('max_drawdown', 0.0),
                metrics.get('total_return', 0.0),
                metrics.get('volatility', 0.0),
                metrics.get('status', 'completed'),
                str(metrics.get('factor_weights', {})),  # JSON as string
                str(metrics.get('hyperparameters', {})),  # JSON as string
                datetime.now(),
                metrics.get('constraint_violations', 0)
            )
            
            conn.execute(insert_sql, trial_data)
            conn.close()
            
            logger.info(f"Logged trial {trial_number} metrics to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log trial metrics: {e}")
            return False
    
    def evaluate_constraints(self, metrics: Dict[str, float]) -> Tuple[bool, int, float]:
        """
        Evaluate constraint violations and apply penalties
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Tuple of (constraints_satisfied, violation_count, penalty_score)
        """
        violations = 0
        penalty = 0.0
        
        max_dd = metrics.get('max_drawdown', 0.0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
        
        # Max drawdown constraint (hard constraint with penalty)
        if max_dd > self.max_dd_threshold:
            violations += 1
            # Heavy penalty for exceeding max drawdown
            penalty += (max_dd - self.max_dd_threshold) * 10
            logger.info(f"Max DD constraint violated: {max_dd:.3f} > {self.max_dd_threshold}")
        
        # Minimum Sharpe ratio (soft constraint)
        if sharpe_ratio < self.min_sharpe_threshold:
            violations += 1
            # Light penalty for low Sharpe ratio
            penalty += (self.min_sharpe_threshold - sharpe_ratio) * 0.5
            logger.info(f"Low Sharpe ratio: {sharpe_ratio:.3f} < {self.min_sharpe_threshold}")
        
        # Additional constraint: excessive volatility
        volatility = metrics.get('volatility', 0.0)
        if volatility > 0.25:  # 25% annual volatility threshold
            violations += 1
            penalty += (volatility - 0.25) * 2
            logger.info(f"High volatility: {volatility:.3f} > 0.25")
        
        constraints_satisfied = violations == 0
        
        return constraints_satisfied, violations, penalty
    
    def calculate_rolling_sharpe(self, returns: pd.Series, window: int = 126) -> float:
        """
        Calculate rolling Sharpe ratio over specified window
        
        Args:
            returns: Daily returns series
            window: Rolling window in days (default: 6 months = 126 business days)
            
        Returns:
            Average rolling Sharpe ratio
        """
        try:
            if len(returns) < window:
                logger.warning(f"Insufficient data for rolling Sharpe: {len(returns)} < {window}")
                return 0.0
            
            # Calculate rolling Sharpe ratios
            rolling_sharpe = []
            
            for i in range(window, len(returns) + 1):
                window_returns = returns.iloc[i-window:i]
                
                if len(window_returns) > 0 and window_returns.std() > 0:
                    sharpe = window_returns.mean() / window_returns.std() * np.sqrt(252)
                    rolling_sharpe.append(sharpe)
            
            if len(rolling_sharpe) == 0:
                return 0.0
            
            # Return average rolling Sharpe ratio
            avg_rolling_sharpe = np.mean(rolling_sharpe)
            
            logger.info(f"Rolling Sharpe calculation: {len(rolling_sharpe)} windows, avg: {avg_rolling_sharpe:.3f}")
            
            return avg_rolling_sharpe
            
        except Exception as e:
            logger.error(f"Failed to calculate rolling Sharpe: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown from equity curve
        
        Args:
            equity_curve: Equity curve series
            
        Returns:
            Maximum drawdown as a percentage
        """
        try:
            if len(equity_curve) < 2:
                return 0.0
            
            # Calculate running maximum (peak)
            peak = equity_curve.expanding().max()
            
            # Calculate drawdown at each point
            drawdown = (equity_curve - peak) / peak
            
            # Maximum drawdown (most negative value)
            max_dd = abs(drawdown.min())
            
            logger.info(f"Max drawdown calculation: {max_dd:.4f} ({max_dd:.1%})")
            
            return max_dd
            
        except Exception as e:
            logger.error(f"Failed to calculate max drawdown: {e}")
            return 1.0  # Return worst case
    
    def validate_factor_weights(self, factor_weights: Dict[str, float]) -> Tuple[bool, str]:
        """
        Validate factor weights for reasonableness
        
        Args:
            factor_weights: Dictionary of factor weights
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for reasonable weight ranges
            for factor, weight in factor_weights.items():
                if abs(weight) > 2.0:  # Weights should be reasonable
                    return False, f"Extreme weight for {factor}: {weight}"
            
            # Check total absolute weight (diversification)
            total_abs_weight = sum(abs(w) for w in factor_weights.values())
            if total_abs_weight > 5.0:  # Prevent over-concentration
                return False, f"Total absolute weight too high: {total_abs_weight}"
            
            # Check for zero weights (all factors should contribute)
            zero_weights = sum(1 for w in factor_weights.values() if abs(w) < 0.01)
            if zero_weights > len(factor_weights) * 0.5:  # More than 50% zero weights
                return False, f"Too many zero weights: {zero_weights}/{len(factor_weights)}"
            
            return True, "Valid"
            
        except Exception as e:
            logger.error(f"Failed to validate factor weights: {e}")
            return False, f"Validation error: {e}"
    
    def calculate_information_ratio(self, returns: pd.Series, 
                                  benchmark_returns: pd.Series = None) -> float:
        """
        Calculate Information Ratio (excess return / tracking error)
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns (default: zero)
            
        Returns:
            Information ratio
        """
        try:
            if benchmark_returns is None:
                # Use zero benchmark (cash)
                excess_returns = returns
            else:
                # Calculate excess returns
                aligned_data = pd.DataFrame({
                    'strategy': returns,
                    'benchmark': benchmark_returns
                }).dropna()
                
                if len(aligned_data) == 0:
                    return 0.0
                
                excess_returns = aligned_data['strategy'] - aligned_data['benchmark']
            
            if len(excess_returns) == 0 or excess_returns.std() == 0:
                return 0.0
            
            # Information ratio = mean excess return / std of excess returns
            ir = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            
            logger.info(f"Information ratio: {ir:.3f}")
            
            return ir
            
        except Exception as e:
            logger.error(f"Failed to calculate information ratio: {e}")
            return 0.0
    
    def create_performance_summary(self, returns: pd.Series, 
                                 factor_weights: Dict[str, float],
                                 hyperparameters: Dict[str, float]) -> Dict[str, float]:
        """
        Create comprehensive performance summary
        
        Args:
            returns: Daily returns series
            factor_weights: Factor weights used
            hyperparameters: Hyperparameters used
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            if len(returns) == 0:
                return self._empty_performance_summary()
            
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Create equity curve
            equity_curve = (1 + returns).cumprod() * 100000  # Starting with $100k
            max_drawdown = self.calculate_max_drawdown(equity_curve)
            
            # Rolling metrics
            rolling_sharpe = self.calculate_rolling_sharpe(returns)
            information_ratio = self.calculate_information_ratio(returns)
            
            # Win rate
            win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
            
            # Risk-adjusted metrics
            calmar_ratio = (total_return / max_drawdown) if max_drawdown > 0 else 0
            sortino_ratio = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
            
            # Validate constraints
            metrics = {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
            constraints_satisfied, violations, penalty = self.evaluate_constraints(metrics)
            
            summary = {
                'total_return': total_return,
                'annualized_return': total_return * (252 / len(returns)) if len(returns) > 0 else 0,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'rolling_sharpe': rolling_sharpe,
                'max_drawdown': max_drawdown,
                'information_ratio': information_ratio,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio,
                'win_rate': win_rate,
                'constraints_satisfied': constraints_satisfied,
                'constraint_violations': violations,
                'penalty_score': penalty,
                'final_score': max(0, sharpe_ratio - penalty),  # Penalized score
                'factor_weights': factor_weights,
                'hyperparameters': hyperparameters,
                'observations': len(returns)
            }
            
            logger.info(f"Performance summary: Sharpe={sharpe_ratio:.3f}, Max DD={max_drawdown:.3f}, Penalty={penalty:.3f}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create performance summary: {e}")
            return self._empty_performance_summary()
    
    def _empty_performance_summary(self) -> Dict[str, float]:
        """Return empty performance summary for failed calculations"""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': -1.0,
            'rolling_sharpe': -1.0,
            'max_drawdown': 1.0,
            'information_ratio': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'constraints_satisfied': False,
            'constraint_violations': 3,
            'penalty_score': 10.0,
            'final_score': -11.0,
            'factor_weights': {},
            'hyperparameters': {},
            'observations': 0
        }


class BacktestEngine:
    """
    Enhanced backtest engine for objective function evaluation
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize backtest engine
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.transaction_cost = 0.001  # 10 bps transaction cost
        
    def run_backtest(self, signals: pd.DataFrame, returns: pd.DataFrame,
                    cash_pct: float, stop_loss_pct: float, 
                    position_size_pct: float) -> pd.Series:
        """
        Run enhanced backtest with transaction costs and risk management
        
        Args:
            signals: Trading signals DataFrame
            returns: Returns DataFrame
            cash_pct: Cash percentage to hold
            stop_loss_pct: Stop loss percentage
            position_size_pct: Maximum position size percentage
            
        Returns:
            Daily returns series
        """
        try:
            logger.info(f"Running backtest: cash={cash_pct:.1%}, stop_loss={stop_loss_pct:.1%}, pos_size={position_size_pct:.1%}")
            logger.info(f"Signals shape: {signals.shape}, columns: {signals.columns.tolist()}")
            logger.info(f"Returns shape: {returns.shape}, columns: {returns.columns.tolist()}")
            
            # Check if signals already has forward_return (from _generate_signals)
            if 'forward_return' in signals.columns:
                backtest_data = signals.copy()
            else:
                # Merge signals with returns
                backtest_data = signals.merge(returns, on=['date', 'symbol'], how='inner')
            
            logger.info(f"Backtest data shape: {backtest_data.shape}, columns: {backtest_data.columns.tolist()}")
            
            if len(backtest_data) == 0:
                logger.warning("No data for backtest")
                return pd.Series(dtype=float)
            
            # Ensure forward_return column exists
            if 'forward_return' not in backtest_data.columns:
                logger.error("Missing forward_return column in backtest data")
                return pd.Series(dtype=float)
            
            # Group by date and simulate daily performance
            daily_returns = []
            equity = self.initial_capital
            position_history = {}  # Track position entry points for stop loss
            
            for date, date_group in backtest_data.groupby('date'):
                active_signals = date_group[date_group['signal'] == True]
                
                if len(active_signals) == 0:
                    daily_returns.append(0.0)
                    continue
                
                # Calculate position sizes
                max_positions = min(len(active_signals), 10)  # Limit to 10 positions
                available_capital = equity * (1 - cash_pct)
                position_value = min(available_capital / max_positions, 
                                   equity * position_size_pct)
                
                # Calculate daily P&L
                day_pnl = 0.0
                
                for _, signal_row in active_signals.head(max_positions).iterrows():
                    symbol = signal_row['symbol']
                    forward_return = signal_row['forward_return']
                    
                    # Apply transaction costs
                    gross_return = forward_return
                    net_return = gross_return - self.transaction_cost
                    
                    # Apply stop loss if position exists
                    if symbol in position_history:
                        # Check if stop loss is triggered
                        if net_return < -stop_loss_pct:
                            net_return = -stop_loss_pct
                            # Remove from position history (stopped out)
                            del position_history[symbol]
                        else:
                            # Update position history
                            position_history[symbol] = date
                    else:
                        # New position
                        if net_return >= -stop_loss_pct:
                            position_history[symbol] = date
                        else:
                            net_return = -stop_loss_pct
                    
                    # Add to daily P&L
                    position_pnl = net_return * position_value
                    day_pnl += position_pnl
                
                # Calculate daily return
                daily_return = day_pnl / equity if equity > 0 else 0
                daily_returns.append(daily_return)
                
                # Update equity
                equity += day_pnl
                
                # Risk management: if equity drops too much, reduce exposure
                if equity < self.initial_capital * 0.7:  # 30% total loss limit
                    cash_pct = min(cash_pct + 0.1, 0.5)  # Increase cash allocation
            
            returns_series = pd.Series(daily_returns, 
                                     index=pd.to_datetime(backtest_data['date'].unique()))
            
            logger.info(f"Backtest completed: {len(returns_series)} days, final equity: ${equity:,.0f}")
            
            return returns_series
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return pd.Series(dtype=float)


def test_objective_utils():
    """Test the ObjectiveUtils class"""
    print("🧪 Testing ObjectiveUtils...")
    
    # Create test instance
    utils = ObjectiveUtils()
    
    # Test rolling Sharpe calculation
    test_returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 1 year of daily returns
    rolling_sharpe = utils.calculate_rolling_sharpe(test_returns)
    print(f"✅ Rolling Sharpe: {rolling_sharpe:.3f}")
    
    # Test constraint evaluation
    test_metrics = {
        'max_drawdown': 0.15,  # Exceeds 12% threshold
        'sharpe_ratio': 0.3,   # Below 0.5 threshold
        'volatility': 0.20     # Within limits
    }
    
    satisfied, violations, penalty = utils.evaluate_constraints(test_metrics)
    print(f"✅ Constraints: satisfied={satisfied}, violations={violations}, penalty={penalty:.2f}")
    
    # Test performance summary
    summary = utils.create_performance_summary(
        test_returns, 
        {'pe_ratio': 0.5, 'momentum': 0.3}, 
        {'cash_pct': 0.1, 'stop_loss': 0.05}
    )
    print(f"✅ Performance summary: {len(summary)} metrics calculated")
    print(f"   Final score: {summary['final_score']:.3f}")
    
    print("🎯 ObjectiveUtils test completed!")


if __name__ == "__main__":
    test_objective_utils()