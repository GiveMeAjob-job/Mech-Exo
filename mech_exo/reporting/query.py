"""
Database query layer for dashboard data access
Abstracts all SQL queries from dashboard components
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..execution.fill_store import FillStore
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)


class DashboardDataProvider:
    """
    Centralized data provider for dashboard queries
    Handles all database access and data transformations
    """
    
    def __init__(self):
        self.fill_store = FillStore()
        self.config_manager = ConfigManager()
        
    def close(self):
        """Clean up database connections"""
        if self.fill_store:
            self.fill_store.close()
    
    def get_nav_series(self, days: int = 730) -> pd.DataFrame:
        """
        Get NAV (Net Asset Value) time series for equity curve
        
        Args:
            days: Number of days to look back (default 2 years)
            
        Returns:
            DataFrame with columns: date, cumulative_pnl, daily_pnl
        """
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Query daily P&L from fills
            query = """
            SELECT 
                DATE(filled_at) as date,
                SUM(CASE WHEN quantity > 0 THEN -1 * (quantity * price + total_fees) 
                         ELSE quantity * price - total_fees END) as daily_pnl
            FROM fills 
            WHERE filled_at >= ? AND filled_at <= ?
            GROUP BY DATE(filled_at)
            ORDER BY date
            """
            
            df = pd.read_sql_query(
                query, 
                self.fill_store.conn,
                params=[start_date, end_date]
            )
            
            if df.empty:
                # Return empty dataframe with correct structure
                return pd.DataFrame(columns=['date', 'daily_pnl', 'cumulative_pnl'])
            
            # Calculate cumulative P&L
            df['cumulative_pnl'] = df['daily_pnl'].cumsum()
            df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Retrieved {len(df)} days of NAV data")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get NAV series: {e}")
            return pd.DataFrame(columns=['date', 'daily_pnl', 'cumulative_pnl'])
    
    def get_current_positions(self) -> pd.DataFrame:
        """
        Get current positions with unrealized P&L
        
        Returns:
            DataFrame with columns: symbol, quantity, avg_price, market_value, 
                                   unrealized_pnl, last_price, pnl_pct
        """
        try:
            # Query to calculate positions from fills
            query = """
            SELECT 
                symbol,
                SUM(quantity) as quantity,
                SUM(quantity * price) / SUM(ABS(quantity)) as avg_price,
                SUM(quantity * price + total_fees) as cost_basis,
                MAX(filled_at) as last_trade_date
            FROM fills 
            WHERE symbol IS NOT NULL
            GROUP BY symbol
            HAVING SUM(quantity) != 0
            ORDER BY ABS(SUM(quantity * price)) DESC
            """
            
            df = pd.read_sql_query(query, self.fill_store.conn)
            
            if df.empty:
                return pd.DataFrame(columns=[
                    'symbol', 'quantity', 'avg_price', 'market_value', 
                    'unrealized_pnl', 'last_price', 'pnl_pct'
                ])
            
            # For MVP, use avg_price as current price (in production would fetch live prices)
            df['last_price'] = df['avg_price'] * (1 + pd.Series([0.001, -0.002, 0.005] * len(df))[:len(df)])  # Simulate price movement
            df['market_value'] = df['quantity'] * df['last_price']
            df['unrealized_pnl'] = df['market_value'] + df['cost_basis']  # cost_basis is negative for buys
            df['pnl_pct'] = (df['unrealized_pnl'] / abs(df['cost_basis'])) * 100
            
            # Round for display
            df['avg_price'] = df['avg_price'].round(2)
            df['last_price'] = df['last_price'].round(2)
            df['market_value'] = df['market_value'].round(2)
            df['unrealized_pnl'] = df['unrealized_pnl'].round(2)
            df['pnl_pct'] = df['pnl_pct'].round(2)
            
            logger.info(f"Retrieved {len(df)} current positions")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get current positions: {e}")
            return pd.DataFrame(columns=[
                'symbol', 'quantity', 'avg_price', 'market_value', 
                'unrealized_pnl', 'last_price', 'pnl_pct'
            ])
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get current risk metrics for heat-map
        
        Returns:
            Dictionary with risk metrics and their utilization percentages
        """
        try:
            # Get portfolio summary
            positions_df = self.get_current_positions()
            
            if positions_df.empty:
                return {
                    'portfolio_delta': 0.0,
                    'portfolio_theta': 0.0, 
                    'gross_leverage': 0.0,
                    'net_leverage': 0.0,
                    'var_95': 0.0,
                    'max_position_pct': 0.0,
                    'sector_concentration': 0.0
                }
            
            # Calculate basic risk metrics
            total_gross_exposure = positions_df['market_value'].abs().sum()
            total_net_exposure = positions_df['market_value'].sum()
            nav = 100000  # Default NAV for MVP
            
            # Simulate risk metrics (in production would use actual risk calculations)
            portfolio_delta = total_net_exposure / nav * 100  # Net exposure as % of NAV
            gross_leverage = total_gross_exposure / nav * 100  # Gross leverage
            max_position_pct = (positions_df['market_value'].abs().max() / nav * 100) if not positions_df.empty else 0
            
            # Simulate other metrics
            portfolio_theta = abs(portfolio_delta) * 0.1  # Simplified theta
            var_95 = abs(total_net_exposure) * 0.02  # 2% VaR
            sector_concentration = 30.0  # Placeholder
            
            risk_metrics = {
                'portfolio_delta': portfolio_delta,
                'portfolio_theta': portfolio_theta,
                'gross_leverage': gross_leverage, 
                'net_leverage': abs(portfolio_delta),
                'var_95': var_95,
                'max_position_pct': max_position_pct,
                'sector_concentration': sector_concentration
            }
            
            logger.info("Retrieved risk metrics")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Failed to get risk metrics: {e}")
            return {
                'portfolio_delta': 0.0,
                'portfolio_theta': 0.0,
                'gross_leverage': 0.0,
                'net_leverage': 0.0, 
                'var_95': 0.0,
                'max_position_pct': 0.0,
                'sector_concentration': 0.0
            }
    
    def get_risk_limits(self) -> Dict[str, float]:
        """
        Get risk limits for heat-map comparison
        
        Returns:
            Dictionary with risk limits
        """
        try:
            # Load from config or use defaults
            return {
                'portfolio_delta': 100.0,  # Max 100% net exposure
                'portfolio_theta': 50.0,   # Max theta exposure
                'gross_leverage': 200.0,   # Max 2x gross leverage
                'net_leverage': 100.0,     # Max 100% net leverage
                'var_95': 5000.0,          # Max $5k VaR
                'max_position_pct': 20.0,  # Max 20% per position
                'sector_concentration': 40.0  # Max 40% in one sector
            }
            
        except Exception as e:
            logger.error(f"Failed to get risk limits: {e}")
            return {}
    
    def get_system_health(self) -> Dict[str, any]:
        """
        Get system health metrics for dashboard status
        
        Returns:
            Dictionary with system status indicators
        """
        try:
            # Check last fill timestamp
            last_fill_ts = self.fill_store.last_fill_ts()
            
            # Check database connectivity
            try:
                self.fill_store.conn.execute("SELECT 1").fetchone()
                db_status = "healthy"
            except Exception:
                db_status = "error"
            
            # Get fill count for today
            today = datetime.now(timezone.utc).date()
            today_fills = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM fills WHERE DATE(filled_at) = ?",
                self.fill_store.conn,
                params=[today]
            ).iloc[0]['count']
            
            return {
                'database_status': db_status,
                'last_fill_time': last_fill_ts.isoformat() if last_fill_ts else None,
                'fills_today': int(today_fills),
                'system_status': 'operational' if db_status == 'healthy' else 'degraded',
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                'database_status': 'error',
                'last_fill_time': None,
                'fills_today': 0,
                'system_status': 'error',
                'last_updated': datetime.now(timezone.utc).isoformat()
            }


# Convenience functions for dashboard callbacks
def get_nav_data(days: int = 730) -> pd.DataFrame:
    """Get NAV series data"""
    provider = DashboardDataProvider()
    try:
        return provider.get_nav_series(days)
    finally:
        provider.close()


def get_positions_data() -> pd.DataFrame:
    """Get current positions data"""
    provider = DashboardDataProvider()
    try:
        return provider.get_current_positions()
    finally:
        provider.close()


def get_risk_data() -> Tuple[Dict[str, float], Dict[str, float]]:
    """Get risk metrics and limits"""
    provider = DashboardDataProvider()
    try:
        metrics = provider.get_risk_metrics()
        limits = provider.get_risk_limits()
        return metrics, limits
    finally:
        provider.close()


def get_health_data() -> Dict[str, any]:
    """Get system health data"""
    provider = DashboardDataProvider()
    try:
        health = provider.get_system_health()
        # Add risk status to health data
        risk_metrics, risk_limits = get_risk_data()
        risk_ok = all(
            risk_metrics.get(metric, 0) <= risk_limits.get(metric, 100) 
            for metric in risk_limits.keys()
        )
        health['risk_ok'] = risk_ok
        
        # Add backtest metrics to health data
        backtest_metrics = get_latest_backtest_metrics()
        health.update(backtest_metrics)
        
        # Add drift metrics to health data
        drift_metrics = get_latest_drift_metrics()
        health.update(drift_metrics)
        
        return health
    finally:
        provider.close()


def get_latest_drift_metrics() -> Dict[str, any]:
    """
    Get latest drift metrics from DuckDB for health endpoint
    
    Returns:
        Dictionary with latest drift metrics or None values if unavailable
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        # Query latest drift metrics
        query = """
        SELECT 
            drift_date,
            calculated_at,
            drift_pct,
            information_ratio,
            drift_status,
            data_quality,
            days_analyzed
        FROM drift_metrics 
        WHERE drift_date = (
            SELECT MAX(drift_date) FROM drift_metrics
        )
        LIMIT 1
        """
        
        try:
            result = pd.read_sql_query(query, storage.conn)
            storage.close()
            
            if result.empty:
                logger.info("No drift metrics found in database")
                return {
                    'drift_pct': None,
                    'drift_status': 'unknown',
                    'drift_ir': None,
                    'drift_date': None,
                    'drift_quality': None
                }
            
            # Extract latest metrics
            latest = result.iloc[0]
            
            drift_metrics = {
                'drift_pct': float(latest['drift_pct']) if pd.notna(latest['drift_pct']) else 0.0,
                'drift_status': latest['drift_status'] if pd.notna(latest['drift_status']) else 'unknown',
                'drift_ir': float(latest['information_ratio']) if pd.notna(latest['information_ratio']) else 0.0,
                'drift_date': latest['drift_date'] if pd.notna(latest['drift_date']) else None,
                'drift_quality': latest['data_quality'] if pd.notna(latest['data_quality']) else 'unknown',
                'drift_days_analyzed': int(latest['days_analyzed']) if pd.notna(latest['days_analyzed']) else 0
            }
            
            logger.info(f"Retrieved latest drift metrics: {drift_metrics['drift_pct']:.1f}% ({drift_metrics['drift_status']})")
            return drift_metrics
            
        except Exception as e:
            # Table might not exist yet
            if "no such table" in str(e).lower() or "does not exist" in str(e).lower():
                logger.info("Drift metrics table does not exist yet")
            else:
                logger.error(f"Failed to query drift metrics: {e}")
            
            return {
                'drift_pct': None,
                'drift_status': 'unknown',
                'drift_ir': None,
                'drift_date': None,
                'drift_quality': None
            }
        
    except Exception as e:
        logger.error(f"Failed to get drift metrics: {e}")
        return {
            'drift_pct': None,
            'drift_status': 'unknown',
            'drift_ir': None,
            'drift_date': None,
            'drift_quality': None
        }


def get_latest_backtest_metrics() -> Dict[str, any]:
    """
    Get latest backtest metrics from DuckDB for health endpoint
    
    Returns:
        Dictionary with latest backtest metrics or None values if unavailable
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        # Query latest backtest metrics
        query = """
        SELECT 
            backtest_date,
            period_start,
            period_end,
            cagr_net,
            sharpe_net,
            max_drawdown,
            total_trades,
            win_rate
        FROM backtest_metrics 
        WHERE backtest_date = (
            SELECT MAX(backtest_date) FROM backtest_metrics
        )
        LIMIT 1
        """
        
        try:
            result = pd.read_sql_query(query, storage.conn)
            storage.close()
            
            if result.empty:
                logger.info("No backtest metrics found in database")
                return {
                    'backtest_sharpe': None,
                    'backtest_cagr': None,
                    'backtest_max_dd': None,
                    'backtest_date': None,
                    'backtest_trades': None
                }
            
            # Extract latest metrics
            latest = result.iloc[0]
            
            backtest_metrics = {
                'backtest_sharpe': float(latest['sharpe_net']) if pd.notna(latest['sharpe_net']) else None,
                'backtest_cagr': float(latest['cagr_net']) if pd.notna(latest['cagr_net']) else None,
                'backtest_max_dd': float(latest['max_drawdown']) if pd.notna(latest['max_drawdown']) else None,
                'backtest_date': latest['backtest_date'] if pd.notna(latest['backtest_date']) else None,
                'backtest_trades': int(latest['total_trades']) if pd.notna(latest['total_trades']) else None,
                'backtest_period': f"{latest['period_start']} to {latest['period_end']}" if pd.notna(latest['period_start']) else None
            }
            
            logger.info(f"Retrieved latest backtest metrics: Sharpe={backtest_metrics['backtest_sharpe']}")
            return backtest_metrics
            
        except Exception as e:
            # Table might not exist yet
            if "no such table" in str(e).lower():
                logger.info("Backtest metrics table does not exist yet")
            else:
                logger.error(f"Failed to query backtest metrics: {e}")
            
            return {
                'backtest_sharpe': None,
                'backtest_cagr': None,
                'backtest_max_dd': None,
                'backtest_date': None,
                'backtest_trades': None
            }
        
    except Exception as e:
        logger.error(f"Failed to get backtest metrics: {e}")
        return {
            'backtest_sharpe': None,
            'backtest_cagr': None, 
            'backtest_max_dd': None,
            'backtest_date': None,
            'backtest_trades': None
        }


def get_live_risk() -> pd.DataFrame:
    """Get live risk metrics as DataFrame for heat-map"""
    provider = DashboardDataProvider()
    try:
        metrics, limits = provider.get_risk_metrics(), provider.get_risk_limits()
        
        # Create DataFrame with metric, value, limit, utilization
        risk_data = []
        for metric, value in metrics.items():
            limit = limits.get(metric, 100.0)
            utilization = (value / limit) if limit > 0 else 0.0
            
            # Map internal names to display names
            display_names = {
                'portfolio_delta': 'Portfolio Delta',
                'portfolio_theta': 'Portfolio Theta', 
                'gross_leverage': 'Gross Leverage',
                'net_leverage': 'Net Leverage',
                'var_95': 'VaR (95%)',
                'max_position_pct': 'Max Position %',
                'sector_concentration': 'Sector Concentration'
            }
            
            risk_data.append({
                'metric': display_names.get(metric, metric),
                'value': value,
                'limit': limit,
                'utilization': utilization,
                'status': 'OK' if utilization < 0.7 else 'WARNING' if utilization < 0.9 else 'BREACH'
            })
        
        return pd.DataFrame(risk_data)
        
    except Exception as e:
        logger.error(f"Failed to get live risk data: {e}")
        return pd.DataFrame(columns=['metric', 'value', 'limit', 'utilization', 'status'])
    finally:
        provider.close()


def get_factor_decay_latest() -> pd.DataFrame:
    """
    Get latest factor decay metrics from DuckDB for Factor Health dashboard
    
    Returns:
        DataFrame with columns: factor, half_life, latest_ic, status, etc.
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        # Query latest factor decay metrics
        query = """
        SELECT 
            factor,
            half_life,
            latest_ic,
            ic_observations,
            ic_mean,
            ic_std,
            ic_trend,
            data_points,
            calculation_timestamp,
            status
        FROM factor_decay 
        WHERE date = (
            SELECT MAX(date) FROM factor_decay
        )
        ORDER BY half_life ASC
        """
        
        try:
            result = pd.read_sql_query(query, storage.conn)
            storage.close()
            
            if result.empty:
                logger.info("No factor decay metrics found in database")
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=[
                    'factor', 'half_life', 'latest_ic', 'ic_observations', 
                    'ic_mean', 'ic_std', 'ic_trend', 'data_points', 
                    'calculation_timestamp', 'status'
                ])
            
            # Add color coding based on half-life thresholds
            def get_color_status(half_life):
                if pd.isna(half_life):
                    return 'unknown'
                elif half_life > 30:
                    return 'green'
                elif half_life >= 10:
                    return 'yellow'
                else:
                    return 'red'
            
            result['color_status'] = result['half_life'].apply(get_color_status)
            
            # Format numeric columns
            for col in ['half_life', 'latest_ic', 'ic_mean', 'ic_std']:
                if col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors='coerce')
            
            logger.info(f"Retrieved {len(result)} factor decay metrics")
            return result
            
        except Exception as e:
            # Table might not exist yet
            if "no such table" in str(e).lower() or "does not exist" in str(e).lower():
                logger.info("Factor decay table does not exist yet")
            else:
                logger.error(f"Failed to query factor decay metrics: {e}")
            
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'factor', 'half_life', 'latest_ic', 'ic_observations', 
                'ic_mean', 'ic_std', 'ic_trend', 'data_points', 
                'calculation_timestamp', 'status', 'color_status'
            ])
        
    except Exception as e:
        logger.error(f"Failed to get factor decay data: {e}")
        return pd.DataFrame(columns=[
            'factor', 'half_life', 'latest_ic', 'ic_observations', 
            'ic_mean', 'ic_std', 'ic_trend', 'data_points', 
            'calculation_timestamp', 'status', 'color_status'
        ])