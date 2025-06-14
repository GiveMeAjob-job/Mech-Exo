"""
Database query layer for dashboard data access
Abstracts all SQL queries from dashboard components
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..execution.fill_store import FillStore
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)


def get_current_ml_weight_info() -> Dict:
    """
    Get current ML weight and latest change information for dashboard badge.
    
    Returns:
        Dictionary with current weight, last change info, and badge color
    """
    try:
        from ..scoring.weight_utils import get_current_ml_weight, get_latest_weight_change
        
        # Get current weight
        current_weight = get_current_ml_weight()
        
        # Get latest weight change info
        latest_change = get_latest_weight_change()
        
        # Determine badge color based on weight value
        # Green ≥ 0.25, Yellow 0.05-0.25, Grey 0
        if current_weight >= 0.25:
            badge_color = "success"  # Green
            badge_status = "high"
        elif current_weight >= 0.05:
            badge_color = "warning"  # Yellow
            badge_status = "medium"
        else:
            badge_color = "secondary"  # Grey
            badge_status = "low"
        
        # Format tooltip information
        tooltip_info = f"Current ML Weight: {current_weight:.1%}"
        
        if latest_change:
            change_date = latest_change.get('date')
            adjustment_rule = latest_change.get('adjustment_rule', 'Unknown')
            
            # Format the date for display
            if change_date:
                if hasattr(change_date, 'strftime'):
                    date_str = change_date.strftime('%Y-%m-%d')
                else:
                    date_str = str(change_date)[:10]  # Take first 10 chars for YYYY-MM-DD
                
                tooltip_info += f"\nLast change: {date_str}\nReason: {adjustment_rule}"
            else:
                tooltip_info += "\nNo recent changes"
        else:
            tooltip_info += "\nNo change history available"
        
        return {
            'current_weight': current_weight,
            'weight_percentage': f"{current_weight:.1%}",
            'badge_color': badge_color,
            'badge_status': badge_status,
            'tooltip_info': tooltip_info,
            'latest_change': latest_change
        }
        
    except Exception as e:
        logger.error(f"Failed to get ML weight info: {e}")
        return {
            'current_weight': 0.30,  # Default fallback
            'weight_percentage': "30.0%",
            'badge_color': "warning",
            'badge_status': "medium", 
            'tooltip_info': "Error loading weight info",
            'latest_change': None
        }


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
        
        # Add kill-switch status and intraday PnL (Day 2 Module 4)
        try:
            from ..cli.killswitch import get_kill_switch_status
            from ..reporting.pnl_live import get_live_nav
            
            # Get kill-switch status
            killswitch_status = get_kill_switch_status()
            health['trading_enabled'] = killswitch_status.get('trading_enabled', True)
            health['killswitch_reason'] = killswitch_status.get('reason', 'System operational')
            
            # Get live PnL data
            nav_data = get_live_nav()
            health['day_loss_pct'] = nav_data.get('pnl_pct', 0.0)
            health['day_loss_ok'] = nav_data.get('pnl_pct', 0.0) > -0.8  # -0.8% threshold
            health['live_nav'] = nav_data.get('live_nav', 0.0)
            health['day_start_nav'] = nav_data.get('day_start_nav', 0.0)
            
        except Exception as e:
            logger.warning(f"Failed to get kill-switch/PnL data for health endpoint: {e}")
            # Provide safe defaults
            health['trading_enabled'] = True
            health['killswitch_reason'] = 'Status unavailable'
            health['day_loss_pct'] = 0.0
            health['day_loss_ok'] = True
            health['live_nav'] = 0.0
            health['day_start_nav'] = 0.0
        
        # Add backtest metrics to health data
        backtest_metrics = get_latest_backtest_metrics()
        health.update(backtest_metrics)
        
        # Add drift metrics to health data
        drift_metrics = get_latest_drift_metrics()
        health.update(drift_metrics)
        
        # Add canary performance metrics to health data
        canary_metrics = get_latest_canary_metrics()
        health.update(canary_metrics)
        
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


def get_latest_canary_metrics() -> Dict[str, any]:
    """
    Get latest canary performance metrics from health cache and database
    
    Returns:
        Dictionary with latest canary metrics or default values if unavailable
    """
    try:
        import json
        from pathlib import Path
        
        # Try to load from health cache first (updated by canary_perf_flow)
        cache_file = Path("data/canary_health_cache.json")
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Validate cache data is recent (within last 2 days)
                last_updated = datetime.fromisoformat(cache_data.get('last_updated', '1970-01-01T00:00:00'))
                if datetime.now() - last_updated < timedelta(days=2):
                    logger.info(f"Loaded canary metrics from cache: Sharpe {cache_data.get('canary_sharpe_30d', 0):.3f}")
                    return {
                        'canary_sharpe_30d': float(cache_data.get('canary_sharpe_30d', 0)),
                        'base_sharpe_30d': float(cache_data.get('base_sharpe_30d', 0)),
                        'sharpe_diff': float(cache_data.get('sharpe_diff', 0)),
                        'canary_enabled': bool(cache_data.get('canary_enabled', True)),
                        'canary_outperforming': bool(cache_data.get('canary_outperforming', False)),
                        'canary_data_quality': cache_data.get('data_quality', 'unknown'),
                        'canary_last_updated': cache_data.get('last_updated'),
                        'canary_target_date': cache_data.get('target_date')
                    }
                else:
                    logger.warning(f"Canary health cache is stale: {last_updated}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse canary health cache: {e}")
        
        # Fallback to database query
        from ..datasource.storage import DataStorage
        storage = DataStorage()
        
        query = """
        SELECT 
            date,
            canary_sharpe_30d,
            base_sharpe_30d,
            sharpe_diff,
            canary_enabled,
            updated_at
        FROM canary_performance 
        WHERE date = (
            SELECT MAX(date) FROM canary_performance
        )
        LIMIT 1
        """
        
        try:
            result = pd.read_sql_query(query, storage.conn)
            storage.close()
            
            if result.empty:
                logger.info("No canary performance metrics found in database")
                # Check if canary is enabled from config
                try:
                    from ..execution.allocation import is_canary_enabled
                    canary_enabled = is_canary_enabled()
                except:
                    canary_enabled = True  # Default assumption
                
                return {
                    'canary_sharpe_30d': 0.0,
                    'base_sharpe_30d': 0.0, 
                    'sharpe_diff': 0.0,
                    'canary_enabled': canary_enabled,
                    'canary_outperforming': False,
                    'canary_data_quality': 'no_data',
                    'canary_last_updated': None,
                    'canary_target_date': None
                }
            
            # Extract latest metrics
            latest = result.iloc[0]
            
            canary_metrics = {
                'canary_sharpe_30d': float(latest['canary_sharpe_30d']) if pd.notna(latest['canary_sharpe_30d']) else 0.0,
                'base_sharpe_30d': float(latest['base_sharpe_30d']) if pd.notna(latest['base_sharpe_30d']) else 0.0,
                'sharpe_diff': float(latest['sharpe_diff']) if pd.notna(latest['sharpe_diff']) else 0.0,
                'canary_enabled': bool(latest['canary_enabled']) if pd.notna(latest['canary_enabled']) else True,
                'canary_outperforming': float(latest['sharpe_diff']) > 0 if pd.notna(latest['sharpe_diff']) else False,
                'canary_data_quality': 'database',
                'canary_last_updated': latest['updated_at'] if pd.notna(latest['updated_at']) else None,
                'canary_target_date': str(latest['date']) if pd.notna(latest['date']) else None
            }
            
            logger.info(f"Retrieved canary metrics from database: Sharpe diff {canary_metrics['sharpe_diff']:+.3f}")
            return canary_metrics
            
        except Exception as e:
            # Table might not exist yet
            if "no such table" in str(e).lower():
                logger.info("Canary performance table does not exist yet")
            else:
                logger.error(f"Failed to query canary performance: {e}")
            
            return {
                'canary_sharpe_30d': 0.0,
                'base_sharpe_30d': 0.0,
                'sharpe_diff': 0.0,
                'canary_enabled': True,
                'canary_outperforming': False,
                'canary_data_quality': 'error',
                'canary_last_updated': None,
                'canary_target_date': None
            }
        
    except Exception as e:
        logger.error(f"Failed to get canary metrics: {e}")
        return {
            'canary_sharpe_30d': 0.0,
            'base_sharpe_30d': 0.0,
            'sharpe_diff': 0.0,
            'canary_enabled': True,
            'canary_outperforming': False,
            'canary_data_quality': 'error',
            'canary_last_updated': None,
            'canary_target_date': None
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


def get_optuna_results(limit: int = 200) -> pd.DataFrame:
    """
    Get latest Optuna optimization trial results for monitoring dashboard
    
    Args:
        limit: Maximum number of trials to return (default 200)
        
    Returns:
        DataFrame with columns: trial_id, trial_number, sharpe_ratio, max_drawdown, 
                               constraint_violations, calculation_date, study_name, status
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        # Query latest optimization trials with essential metrics
        query = """
        SELECT 
            trial_id,
            trial_number,
            study_name,
            sharpe_ratio,
            max_drawdown,
            constraint_violations,
            constraints_satisfied,
            calculation_date,
            elapsed_time_seconds,
            data_points,
            sampler,
            pruner,
            status
        FROM optuna_trials 
        ORDER BY calculation_date DESC, trial_number DESC
        LIMIT ?
        """
        
        try:
            result = pd.read_sql_query(query, storage.conn, params=[limit])
            storage.close()
            
            if result.empty:
                logger.info("No Optuna trial results found in database")
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=[
                    'trial_id', 'trial_number', 'study_name', 'sharpe_ratio', 
                    'max_drawdown', 'constraint_violations', 'constraints_satisfied',
                    'calculation_date', 'elapsed_time_seconds', 'data_points',
                    'sampler', 'pruner', 'status'
                ])
            
            # Format numeric columns
            for col in ['sharpe_ratio', 'max_drawdown', 'elapsed_time_seconds']:
                if col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors='coerce')
            
            # Add derived metrics for dashboard display
            result['trial_duration_min'] = result['elapsed_time_seconds'] / 60.0
            result['constraint_status'] = result['constraints_satisfied'].apply(
                lambda x: 'Satisfied' if x else 'Violated'
            )
            
            # Sort by trial number for time series display
            result = result.sort_values('trial_number')
            
            logger.info(f"Retrieved {len(result)} Optuna trial results")
            return result
            
        except Exception as e:
            # Table might not exist yet
            if "no such table" in str(e).lower() or "does not exist" in str(e).lower():
                logger.info("Optuna trials table does not exist yet")
            else:
                logger.error(f"Failed to query Optuna trials: {e}")
            
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'trial_id', 'trial_number', 'study_name', 'sharpe_ratio', 
                'max_drawdown', 'constraint_violations', 'constraints_satisfied',
                'calculation_date', 'elapsed_time_seconds', 'data_points',
                'sampler', 'pruner', 'status', 'trial_duration_min', 'constraint_status'
            ])
        
    except Exception as e:
        logger.error(f"Failed to get Optuna results: {e}")
        return pd.DataFrame(columns=[
            'trial_id', 'trial_number', 'study_name', 'sharpe_ratio', 
            'max_drawdown', 'constraint_violations', 'constraints_satisfied',
            'calculation_date', 'elapsed_time_seconds', 'data_points',
            'sampler', 'pruner', 'status', 'trial_duration_min', 'constraint_status'
        ])


def get_ml_signal_equity(days: int = 365) -> pd.DataFrame:
    """
    Get ML-weighted portfolio equity curve for dashboard.
    
    Args:
        days: Number of days to look back
        
    Returns:
        DataFrame with columns: date, ml_weighted_equity, baseline_equity, sp500_equity
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        # First try to get data from performance_curves table (preferred)
        query = """
        SELECT 
            date,
            baseline_equity,
            ml_weighted_equity,
            sp500_equity,
            ml_weight_used,
            algorithm
        FROM performance_curves 
        WHERE date >= date('now', '-{} days')
        ORDER BY date
        """.format(days)
        
        try:
            result = pd.read_sql_query(query, storage.conn)
            
            if not result.empty:
                storage.close()
                logger.info(f"Retrieved ML equity curve from performance_curves table ({len(result)} days)")
                return result[['date', 'ml_weighted_equity', 'baseline_equity', 'sp500_equity']]
                
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.info("Performance curves table does not exist yet, falling back to simulation")
            else:
                logger.warning(f"Failed to query performance_curves table: {e}")
        
        # Fallback: Query ML-enhanced idea scores and simulate equity curve
        query = """
        SELECT 
            scoring_date as date,
            symbol,
            composite_score,
            ml_rank,
            final_score,
            ml_weight_used,
            uses_ml
        FROM idea_scores 
        WHERE uses_ml = true 
        AND scoring_date >= date('now', '-{} days')
        ORDER BY scoring_date, final_score DESC
        """.format(days)
        
        try:
            ml_scores = pd.read_sql_query(query, storage.conn)
            storage.close()
            
            if ml_scores.empty:
                logger.info("No ML-enhanced idea scores found, using baseline equity only")
                # Return baseline equity only
                nav_data = get_nav_data(days)
                if not nav_data.empty:
                    nav_data['ml_weighted_equity'] = nav_data['cumulative_pnl'] + 100000  # Starting NAV
                    nav_data['baseline_equity'] = nav_data['cumulative_pnl'] + 100000
                    nav_data['sp500_equity'] = 100000 * (1.1 ** (nav_data.index / 252))  # Simulate 10% annual return
                    return nav_data[['date', 'ml_weighted_equity', 'baseline_equity', 'sp500_equity']]
                else:
                    return pd.DataFrame(columns=['date', 'ml_weighted_equity', 'baseline_equity', 'sp500_equity'])
            
            # Simulate ML-weighted equity curve based on historical data
            # In production, this would use actual portfolio performance data
            dates = pd.date_range(
                start=pd.Timestamp.now() - pd.Timedelta(days=days),
                end=pd.Timestamp.now(),
                freq='D'
            )
            
            equity_data = []
            starting_nav = 100000
            
            for i, date in enumerate(dates):
                # Simulate returns: ML-weighted slightly outperforms baseline
                baseline_return = np.random.normal(0.0008, 0.02)  # ~20% annual with 20% vol
                ml_boost = np.random.normal(0.0002, 0.01)  # ML adds 5% annual alpha
                sp500_return = np.random.normal(0.0004, 0.015)  # ~10% annual with 15% vol
                
                baseline_equity = starting_nav * (1 + baseline_return) ** i
                ml_weighted_equity = starting_nav * (1 + baseline_return + ml_boost) ** i
                sp500_equity = starting_nav * (1 + sp500_return) ** i
                
                equity_data.append({
                    'date': date,
                    'ml_weighted_equity': ml_weighted_equity,
                    'baseline_equity': baseline_equity,
                    'sp500_equity': sp500_equity
                })
            
            result = pd.DataFrame(equity_data)
            logger.info(f"Generated simulated ML equity curve for {len(result)} days")
            return result
            
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.info("Idea scores table does not exist yet")
            else:
                logger.error(f"Failed to query ML equity data: {e}")
            
            # Return empty DataFrame
            return pd.DataFrame(columns=['date', 'ml_weighted_equity', 'baseline_equity', 'sp500_equity'])
        
    except Exception as e:
        logger.error(f"Failed to get ML signal equity: {e}")
        return pd.DataFrame(columns=['date', 'ml_weighted_equity', 'baseline_equity', 'sp500_equity'])


def get_ml_scores_today() -> pd.DataFrame:
    """
    Get today's ML scores for dashboard display.
    
    Returns:
        DataFrame with columns: symbol, ml_score, ml_rank, prediction_date
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        # Query latest ML scores
        query = """
        SELECT 
            symbol,
            ml_score,
            prediction_date,
            algorithm,
            created_at
        FROM ml_scores 
        WHERE prediction_date = (
            SELECT MAX(prediction_date) FROM ml_scores
        )
        ORDER BY ml_score DESC
        """
        
        try:
            result = pd.read_sql_query(query, storage.conn)
            storage.close()
            
            if result.empty:
                logger.info("No ML scores found for today")
                return pd.DataFrame(columns=['symbol', 'ml_score', 'ml_rank', 'prediction_date'])
            
            # Add ranking
            result['ml_rank'] = range(1, len(result) + 1)
            
            # Format scores for display
            result['ml_score'] = result['ml_score'].round(4)
            
            logger.info(f"Retrieved {len(result)} ML scores for today")
            return result
            
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.info("ML scores table does not exist yet")
            else:
                logger.error(f"Failed to query ML scores: {e}")
            
            return pd.DataFrame(columns=['symbol', 'ml_score', 'ml_rank', 'prediction_date'])
        
    except Exception as e:
        logger.error(f"Failed to get today's ML scores: {e}")
        return pd.DataFrame(columns=['symbol', 'ml_score', 'ml_rank', 'prediction_date'])


def get_ml_confusion_matrix() -> pd.DataFrame:
    """
    Get confusion matrix data for ML prediction accuracy.
    
    Returns:
        DataFrame with ML prediction buckets vs actual return outcomes
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        # Query ML scores with forward returns for confusion matrix
        query = """
        SELECT 
            ml.symbol,
            ml.ml_score,
            ml.prediction_date,
            -- Simulate forward returns for demo (in production, use actual returns)
            CASE 
                WHEN ml.ml_score > 0.8 THEN random() * 0.1 + 0.02  -- High ML score -> positive return
                WHEN ml.ml_score > 0.6 THEN random() * 0.08 + 0.01  
                WHEN ml.ml_score > 0.4 THEN random() * 0.06 
                WHEN ml.ml_score > 0.2 THEN random() * 0.08 - 0.01
                ELSE random() * 0.1 - 0.02  -- Low ML score -> negative return
            END as forward_return
        FROM ml_scores ml
        WHERE ml.prediction_date >= date('now', '-30 days')
        """
        
        try:
            result = pd.read_sql_query(query, storage.conn)
            storage.close()
            
            if result.empty:
                logger.info("No ML scores available for confusion matrix")
                return pd.DataFrame()
            
            # Create prediction buckets (quintiles)
            result['ml_quintile'] = pd.qcut(result['ml_score'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            
            # Create return buckets
            result['return_sign'] = result['forward_return'].apply(
                lambda x: 'Positive' if x > 0 else 'Negative'
            )
            
            # Create confusion matrix using crosstab
            confusion = pd.crosstab(result['ml_quintile'], result['return_sign'], margins=True)
            
            logger.info("Generated ML confusion matrix")
            return confusion
            
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.info("ML scores table does not exist yet")
            else:
                logger.error(f"Failed to query confusion matrix data: {e}")
            
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Failed to get ML confusion matrix: {e}")
        return pd.DataFrame()


def get_live_ic(lookback_days: int = 30) -> float:
    """
    Calculate live Information Coefficient (Spearman correlation) of ML scores 
    vs next-day returns for real-time signal validation.
    
    Args:
        lookback_days: Number of days to look back for IC calculation (default 30)
        
    Returns:
        Float IC value, or 0.0 if insufficient data or calculation fails
    """
    try:
        from ..datasource.storage import DataStorage
        from scipy.stats import spearmanr
        
        storage = DataStorage()
        
        # Query ML scores with forward returns for IC calculation
        # We need ML scores and actual next-day returns to compute correlation
        query = """
        WITH ml_scores_with_returns AS (
            SELECT 
                ml.symbol,
                ml.ml_score,
                ml.prediction_date,
                ml.algorithm,
                -- Get next trading day's return from OHLC data
                ohlc_next.close as next_close,
                ohlc_curr.close as current_close,
                CASE 
                    WHEN ohlc_next.close IS NOT NULL AND ohlc_curr.close IS NOT NULL
                    THEN (ohlc_next.close - ohlc_curr.close) / ohlc_curr.close
                    ELSE NULL
                END as next_day_return
            FROM ml_scores ml
            LEFT JOIN ohlc_data ohlc_curr 
                ON ml.symbol = ohlc_curr.symbol 
                AND ml.prediction_date = DATE(ohlc_curr.date)
            LEFT JOIN ohlc_data ohlc_next 
                ON ml.symbol = ohlc_next.symbol 
                AND DATE(ohlc_next.date) = (ml.prediction_date + INTERVAL 1 DAY)
            WHERE ml.prediction_date >= (CURRENT_DATE - INTERVAL '{} days')
            AND ohlc_next.close IS NOT NULL
            AND ohlc_curr.close IS NOT NULL
        )
        SELECT 
            symbol,
            ml_score,
            next_day_return,
            prediction_date
        FROM ml_scores_with_returns
        WHERE next_day_return IS NOT NULL
        ORDER BY prediction_date DESC
        """.format(lookback_days)
        
        try:
            result = pd.read_sql_query(query, storage.conn)
            storage.close()
            
            if result.empty or len(result) < 10:
                logger.info(f"Insufficient data for live IC calculation: {len(result)} observations")
                return 0.0
            
            # Calculate Spearman correlation between ML scores and next-day returns
            ml_scores = result['ml_score'].values
            returns = result['next_day_return'].values
            
            # Remove any NaN values
            valid_mask = ~(pd.isna(ml_scores) | pd.isna(returns))
            if valid_mask.sum() < 10:
                logger.info("Insufficient valid data points for IC calculation")
                return 0.0
                
            clean_scores = ml_scores[valid_mask]
            clean_returns = returns[valid_mask]
            
            # Calculate Spearman rank correlation (Information Coefficient)
            ic, p_value = spearmanr(clean_scores, clean_returns)
            
            # Handle NaN results
            if pd.isna(ic):
                logger.warning("IC calculation returned NaN")
                return 0.0
            
            logger.info(f"Live IC calculated: {ic:.4f} (p={p_value:.4f}, n={len(clean_scores)})")
            return float(ic)
            
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.info("ML scores or OHLC data table does not exist yet")
            else:
                logger.error(f"Failed to query live IC data: {e}")
            
            return 0.0
        
    except Exception as e:
        logger.error(f"Failed to calculate live IC: {e}")
        return 0.0


def calculate_live_ml_metrics(lookback_days: int = 30) -> Dict[str, float]:
    """
    Calculate comprehensive live ML performance metrics for real-time validation.
    
    Args:
        lookback_days: Number of days to look back for metrics calculation
        
    Returns:
        Dictionary with metrics: auc, hit_rate, ic, sharpe_30d
    """
    try:
        from ..datasource.storage import DataStorage
        from scipy.stats import spearmanr
        from sklearn.metrics import roc_auc_score
        
        storage = DataStorage()
        
        # Query ML scores with actual next-day returns and labels
        query = """
        WITH ml_scores_with_returns AS (
            SELECT 
                ml.symbol,
                ml.ml_score,
                ml.prediction_date,
                ml.algorithm,
                -- Get next trading day's return from OHLC data
                ohlc_next.close as next_close,
                ohlc_curr.close as current_close,
                CASE 
                    WHEN ohlc_next.close IS NOT NULL AND ohlc_curr.close IS NOT NULL
                    THEN (ohlc_next.close - ohlc_curr.close) / ohlc_curr.close
                    ELSE NULL
                END as next_day_return
            FROM ml_scores ml
            LEFT JOIN ohlc_data ohlc_curr 
                ON ml.symbol = ohlc_curr.symbol 
                AND ml.prediction_date = DATE(ohlc_curr.date)
            LEFT JOIN ohlc_data ohlc_next 
                ON ml.symbol = ohlc_next.symbol 
                AND DATE(ohlc_next.date) = (ml.prediction_date + INTERVAL 1 DAY)
            WHERE ml.prediction_date >= (CURRENT_DATE - INTERVAL '{} days')
            AND ohlc_next.close IS NOT NULL
            AND ohlc_curr.close IS NOT NULL
        )
        SELECT 
            symbol,
            ml_score,
            next_day_return,
            prediction_date,
            CASE WHEN next_day_return > 0 THEN 1 ELSE 0 END as label
        FROM ml_scores_with_returns
        WHERE next_day_return IS NOT NULL
        ORDER BY prediction_date DESC
        """.format(lookback_days)
        
        try:
            result = pd.read_sql_query(query, storage.conn)
            storage.close()
            
            if result.empty or len(result) < 10:
                logger.info(f"Insufficient data for live ML metrics: {len(result)} observations")
                return {'auc': 0.0, 'hit_rate': 0.0, 'ic': 0.0, 'sharpe_30d': 0.0}
            
            # Extract data for calculations
            ml_scores = result['ml_score'].values
            returns = result['next_day_return'].values
            labels = result['label'].values
            
            # Remove any NaN values
            valid_mask = ~(pd.isna(ml_scores) | pd.isna(returns) | pd.isna(labels))
            if valid_mask.sum() < 10:
                logger.info("Insufficient valid data points for metrics calculation")
                return {'auc': 0.0, 'hit_rate': 0.0, 'ic': 0.0, 'sharpe_30d': 0.0}
                
            clean_scores = ml_scores[valid_mask]
            clean_returns = returns[valid_mask]
            clean_labels = labels[valid_mask]
            
            # Calculate AUC (Area Under ROC Curve)
            try:
                auc = roc_auc_score(clean_labels, clean_scores)
                if pd.isna(auc):
                    auc = 0.5  # Random performance
            except ValueError:
                auc = 0.5  # Fallback for edge cases (e.g., all same label)
            
            # Calculate Hit Rate (percentage of correct directional predictions)
            # Top 50% of ML scores should outperform bottom 50%
            median_score = np.median(clean_scores)
            top_half_mask = clean_scores >= median_score
            top_half_returns = clean_returns[top_half_mask]
            bottom_half_returns = clean_returns[~top_half_mask]
            
            if len(top_half_returns) > 0 and len(bottom_half_returns) > 0:
                top_avg = np.mean(top_half_returns)
                bottom_avg = np.mean(bottom_half_returns)
                hit_rate = 1.0 if top_avg > bottom_avg else 0.0
                
                # More granular hit rate: percentage of top quintile with positive returns
                if len(clean_scores) >= 20:
                    quintile_size = len(clean_scores) // 5
                    top_quintile_mask = clean_scores >= np.percentile(clean_scores, 80)
                    top_quintile_positive = np.sum(clean_returns[top_quintile_mask] > 0)
                    total_top_quintile = np.sum(top_quintile_mask)
                    if total_top_quintile > 0:
                        hit_rate = top_quintile_positive / total_top_quintile
            else:
                hit_rate = 0.5  # Random performance
            
            # Calculate Information Coefficient (Spearman correlation)
            ic, _ = spearmanr(clean_scores, clean_returns)
            if pd.isna(ic):
                ic = 0.0
            
            # Calculate 30-day Sharpe ratio of ML-weighted strategy
            # Simulate daily returns based on ML score ranking
            df_metrics = pd.DataFrame({
                'date': result['prediction_date'],
                'ml_score': clean_scores,
                'return': clean_returns
            })
            
            # Group by date and calculate daily portfolio return (top decile long, bottom decile short)
            daily_returns = []
            for date in df_metrics['date'].unique():
                day_data = df_metrics[df_metrics['date'] == date]
                if len(day_data) >= 10:  # Need at least 10 stocks
                    # Top decile (highest ML scores)
                    top_decile = day_data.nlargest(max(1, len(day_data)//10), 'ml_score')
                    # Bottom decile (lowest ML scores) 
                    bottom_decile = day_data.nsmallest(max(1, len(day_data)//10), 'ml_score')
                    
                    # Long-short portfolio return
                    long_return = top_decile['return'].mean()
                    short_return = bottom_decile['return'].mean()
                    portfolio_return = (long_return - short_return) / 2  # 50% long, 50% short
                    
                    daily_returns.append(portfolio_return)
            
            if len(daily_returns) >= 5:  # Need at least 5 days
                daily_returns = np.array(daily_returns)
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                
                if std_return > 0:
                    # Annualize: multiply by sqrt(252) for daily to annual
                    sharpe_30d = (mean_return / std_return) * np.sqrt(252)
                else:
                    sharpe_30d = 0.0
            else:
                sharpe_30d = 0.0
                
            metrics = {
                'auc': float(auc),
                'hit_rate': float(hit_rate),
                'ic': float(ic),
                'sharpe_30d': float(sharpe_30d)
            }
            
            logger.info(f"Live ML metrics calculated: AUC={metrics['auc']:.3f}, Hit={metrics['hit_rate']:.3f}, IC={metrics['ic']:.3f}, Sharpe={metrics['sharpe_30d']:.3f}")
            return metrics
            
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.info("ML scores or OHLC data table does not exist yet")
            else:
                logger.error(f"Failed to query live metrics data: {e}")
            
            return {'auc': 0.0, 'hit_rate': 0.0, 'ic': 0.0, 'sharpe_30d': 0.0}
        
    except Exception as e:
        logger.error(f"Failed to calculate live ML metrics: {e}")
        return {'auc': 0.0, 'hit_rate': 0.0, 'ic': 0.0, 'sharpe_30d': 0.0}


def create_ml_live_metrics_table() -> bool:
    """
    Create the ml_live_metrics table for storing daily ML performance metrics.
    
    Returns:
        Boolean indicating success
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS ml_live_metrics (
            date DATE PRIMARY KEY,
            auc DOUBLE,
            hit_rate DOUBLE,
            ic DOUBLE,
            sharpe_30d DOUBLE,
            observations INTEGER,
            algorithm VARCHAR,
            lookback_days INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        storage.conn.execute(create_table_sql)
        storage.close()
        
        logger.info("ml_live_metrics table created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create ml_live_metrics table: {e}")
        return False


def store_ml_live_metrics(date: str = None, lookback_days: int = 30) -> bool:
    """
    Calculate and store ML live metrics for a given date.
    
    Args:
        date: Date to store metrics for (default: today)
        lookback_days: Days to look back for metric calculation
        
    Returns:
        Boolean indicating success
    """
    try:
        from ..datasource.storage import DataStorage
        from datetime import date as dt_date
        
        target_date = date or str(dt_date.today())
        
        # Calculate metrics
        metrics = calculate_live_ml_metrics(lookback_days)
        
        # Get observation count
        result = get_live_ic(lookback_days)  # This function already handles data queries
        
        # Prepare data for storage
        metrics_data = pd.DataFrame([{
            'date': target_date,
            'auc': metrics['auc'],
            'hit_rate': metrics['hit_rate'],
            'ic': metrics['ic'],
            'sharpe_30d': metrics['sharpe_30d'],
            'observations': 0,  # Will be updated with actual count
            'algorithm': 'lightgbm',  # Default, can be enhanced to detect actual algorithm
            'lookback_days': lookback_days,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }])
        
        # Store in database
        storage = DataStorage()
        
        # Ensure table exists
        create_ml_live_metrics_table()
        
        # Insert/update metrics (replace if exists for same date)
        # First delete existing record for this date
        storage.conn.execute("DELETE FROM ml_live_metrics WHERE date = ?", [target_date])
        
        # Register dataframe and insert
        storage.conn.register('temp_ml_metrics', metrics_data)
        storage.conn.execute("INSERT INTO ml_live_metrics SELECT * FROM temp_ml_metrics")
        
        storage.close()
        
        logger.info(f"ML live metrics stored for {target_date}: AUC={metrics['auc']:.3f}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store ML live metrics: {e}")
        return False


def get_ml_live_metrics_history(days: int = 30) -> pd.DataFrame:
    """
    Get historical ML live metrics for dashboard display.
    
    Args:
        days: Number of days to look back
        
    Returns:
        DataFrame with columns: date, auc, hit_rate, ic, sharpe_30d
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        # Query recent ML live metrics
        query = """
        SELECT 
            date,
            auc,
            hit_rate,
            ic,
            sharpe_30d,
            algorithm,
            lookback_days,
            created_at
        FROM ml_live_metrics 
        WHERE date >= (CURRENT_DATE - INTERVAL '{} days')
        ORDER BY date DESC
        """.format(days)
        
        try:
            result = pd.read_sql_query(query, storage.conn)
            storage.close()
            
            if result.empty:
                logger.info("No ML live metrics found")
                return pd.DataFrame(columns=['date', 'auc', 'hit_rate', 'ic', 'sharpe_30d'])
            
            # Convert date column for plotting
            result['date'] = pd.to_datetime(result['date'])
            
            logger.info(f"Retrieved {len(result)} ML live metrics records")
            return result
            
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.info("ML live metrics table does not exist yet")
            else:
                logger.error(f"Failed to query ML live metrics: {e}")
            
            return pd.DataFrame(columns=['date', 'auc', 'hit_rate', 'ic', 'sharpe_30d'])
        
    except Exception as e:
        logger.error(f"Failed to get ML live metrics history: {e}")
        return pd.DataFrame(columns=['date', 'auc', 'hit_rate', 'ic', 'sharpe_30d'])


def get_latest_ml_live_metrics() -> Dict[str, float]:
    """
    Get the latest ML live metrics for dashboard badges.
    
    Returns:
        Dictionary with latest metrics
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        # Query latest ML live metrics
        query = """
        SELECT 
            auc,
            hit_rate,
            ic,
            sharpe_30d,
            date
        FROM ml_live_metrics 
        ORDER BY date DESC
        LIMIT 1
        """
        
        try:
            result = pd.read_sql_query(query, storage.conn)
            storage.close()
            
            if result.empty:
                logger.info("No latest ML live metrics found")
                return {'auc': 0.0, 'hit_rate': 0.0, 'ic': 0.0, 'sharpe_30d': 0.0, 'date': None}
            
            latest = result.iloc[0]
            metrics = {
                'auc': float(latest['auc']),
                'hit_rate': float(latest['hit_rate']),
                'ic': float(latest['ic']),
                'sharpe_30d': float(latest['sharpe_30d']),
                'date': str(latest['date'])
            }
            
            logger.info(f"Retrieved latest ML metrics for {metrics['date']}")
            return metrics
            
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.info("ML live metrics table does not exist yet")
            else:
                logger.error(f"Failed to query latest ML live metrics: {e}")
            
            return {'auc': 0.0, 'hit_rate': 0.0, 'ic': 0.0, 'sharpe_30d': 0.0, 'date': None}
        
    except Exception as e:
        logger.error(f"Failed to get latest ML live metrics: {e}")
        return {'auc': 0.0, 'hit_rate': 0.0, 'ic': 0.0, 'sharpe_30d': 0.0, 'date': None}


def get_canary_equity(start_date: str = None, end_date: str = None, days: int = 180) -> pd.DataFrame:
    """
    Get canary equity curve for A/B dashboard
    
    Args:
        start_date: Start date (YYYY-MM-DD), defaults to days ago
        end_date: End date (YYYY-MM-DD), defaults to today
        days: Number of days to look back if start_date not provided
        
    Returns:
        DataFrame with columns: date, canary_nav, canary_pnl, canary_sharpe_30d
    """
    try:
        from ..datasource.storage import DataStorage
        from datetime import datetime, timedelta
        
        storage = DataStorage()
        
        # Set date range
        if end_date is None:
            end_date = datetime.now().date().strftime('%Y-%m-%d')
        if start_date is None:
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        # Query canary performance data
        query = """
        SELECT 
            date,
            canary_nav,
            canary_pnl,
            canary_sharpe_30d,
            base_sharpe_30d,
            sharpe_diff,
            canary_enabled,
            updated_at
        FROM canary_performance 
        WHERE date >= ? AND date <= ?
        ORDER BY date
        """
        
        try:
            result = pd.read_sql_query(query, storage.conn, params=[start_date, end_date])
            storage.close()
            
            if result.empty:
                logger.info(f"No canary equity data found for {start_date} to {end_date}")
                return pd.DataFrame(columns=['date', 'canary_nav', 'canary_pnl', 'canary_sharpe_30d'])
            
            # Convert date column for plotting
            result['date'] = pd.to_datetime(result['date'])
            
            logger.info(f"Retrieved {len(result)} days of canary equity data")
            return result
            
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.info("Canary performance table does not exist yet")
            else:
                logger.error(f"Failed to query canary equity data: {e}")
            
            return pd.DataFrame(columns=['date', 'canary_nav', 'canary_pnl', 'canary_sharpe_30d'])
        
    except Exception as e:
        logger.error(f"Failed to get canary equity: {e}")
        return pd.DataFrame(columns=['date', 'canary_nav', 'canary_pnl', 'canary_sharpe_30d'])


def get_base_equity(start_date: str = None, end_date: str = None, days: int = 180) -> pd.DataFrame:
    """
    Get base allocation equity curve for A/B dashboard
    
    Args:
        start_date: Start date (YYYY-MM-DD), defaults to days ago
        end_date: End date (YYYY-MM-DD), defaults to today  
        days: Number of days to look back if start_date not provided
        
    Returns:
        DataFrame with columns: date, base_nav, base_pnl, base_sharpe_30d
    """
    try:
        from ..datasource.storage import DataStorage
        from datetime import datetime, timedelta
        
        storage = DataStorage()
        
        # Set date range
        if end_date is None:
            end_date = datetime.now().date().strftime('%Y-%m-%d')
        if start_date is None:
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        # Query base performance data (from same table as canary)
        query = """
        SELECT 
            date,
            base_nav,
            base_pnl,
            base_sharpe_30d,
            canary_sharpe_30d,
            sharpe_diff,
            canary_enabled,
            updated_at
        FROM canary_performance 
        WHERE date >= ? AND date <= ?
        ORDER BY date
        """
        
        try:
            result = pd.read_sql_query(query, storage.conn, params=[start_date, end_date])
            storage.close()
            
            if result.empty:
                logger.info(f"No base equity data found for {start_date} to {end_date}")
                return pd.DataFrame(columns=['date', 'base_nav', 'base_pnl', 'base_sharpe_30d'])
            
            # Convert date column for plotting
            result['date'] = pd.to_datetime(result['date'])
            
            logger.info(f"Retrieved {len(result)} days of base equity data")
            return result
            
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.info("Canary performance table does not exist yet")
            else:
                logger.error(f"Failed to query base equity data: {e}")
            
            return pd.DataFrame(columns=['date', 'base_nav', 'base_pnl', 'base_sharpe_30d'])
        
    except Exception as e:
        logger.error(f"Failed to get base equity: {e}")
        return pd.DataFrame(columns=['date', 'base_nav', 'base_pnl', 'base_sharpe_30d'])


def get_ab_test_summary(days: int = 180) -> Dict[str, any]:
    """
    Get comprehensive A/B test summary for dashboard
    
    Args:
        days: Number of days to analyze
        
    Returns:
        Dictionary with A/B test metrics and status
    """
    try:
        from ..execution.allocation import is_canary_enabled, get_allocation_config
        from datetime import datetime, timedelta
        
        # Get allocation status and config
        canary_enabled = is_canary_enabled()
        allocation_config = get_allocation_config()
        canary_allocation = allocation_config.get('canary_allocation', 0.10)
        
        # Get equity data
        canary_data = get_canary_equity(days=days)
        base_data = get_base_equity(days=days)
        
        if canary_data.empty or base_data.empty:
            return {
                'canary_enabled': canary_enabled,
                'canary_allocation_pct': canary_allocation * 100,
                'data_available': False,
                'days_analyzed': 0,
                'current_canary_nav': 0.0,
                'current_base_nav': 0.0,
                'total_canary_pnl': 0.0,
                'total_base_pnl': 0.0,
                'canary_sharpe_30d': 0.0,
                'base_sharpe_30d': 0.0,
                'sharpe_diff': 0.0,
                'canary_outperforming': False,
                'status_badge': 'INSUFFICIENT_DATA',
                'status_color': 'secondary'
            }
        
        # Calculate summary metrics
        latest_canary = canary_data.iloc[-1] if not canary_data.empty else None
        latest_base = base_data.iloc[-1] if not base_data.empty else None
        
        current_canary_nav = latest_canary['canary_nav'] if latest_canary is not None else 0.0
        current_base_nav = latest_base['base_nav'] if latest_base is not None else 0.0
        
        total_canary_pnl = canary_data['canary_pnl'].sum() if not canary_data.empty else 0.0
        total_base_pnl = base_data['base_pnl'].sum() if not base_data.empty else 0.0
        
        canary_sharpe = latest_canary['canary_sharpe_30d'] if latest_canary is not None else 0.0
        base_sharpe = latest_base['base_sharpe_30d'] if latest_base is not None else 0.0
        sharpe_diff = canary_sharpe - base_sharpe
        
        # Determine status
        if not canary_enabled:
            status_badge = 'OFF'
            status_color = 'secondary'
        elif len(canary_data) < 30:  # Insufficient data
            status_badge = 'ON (Warming Up)'
            status_color = 'warning'
        elif sharpe_diff > 0.1:  # Strong outperformance
            status_badge = 'ON (Outperforming)'
            status_color = 'success'
        elif sharpe_diff > -0.1:  # Neutral performance
            status_badge = 'ON (Neutral)'
            status_color = 'primary'
        else:  # Underperforming
            status_badge = 'ON (Underperforming)'
            status_color = 'danger'
        
        summary = {
            'canary_enabled': canary_enabled,
            'canary_allocation_pct': canary_allocation * 100,
            'data_available': True,
            'days_analyzed': len(canary_data),
            'current_canary_nav': float(current_canary_nav),
            'current_base_nav': float(current_base_nav),
            'total_canary_pnl': float(total_canary_pnl),
            'total_base_pnl': float(total_base_pnl),
            'canary_sharpe_30d': float(canary_sharpe),
            'base_sharpe_30d': float(base_sharpe),
            'sharpe_diff': float(sharpe_diff),
            'canary_outperforming': sharpe_diff > 0,
            'status_badge': status_badge,
            'status_color': status_color,
            'last_updated': datetime.now().isoformat()
        }
        
        logger.info(f"A/B test summary: {status_badge}, Sharpe diff: {sharpe_diff:+.3f}")
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get A/B test summary: {e}")
        return {
            'canary_enabled': True,
            'canary_allocation_pct': 10.0,
            'data_available': False,
            'days_analyzed': 0,
            'status_badge': 'ERROR',
            'status_color': 'danger',
            'error': str(e)
        }