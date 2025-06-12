"""
Tag-based P&L and NAV computation for canary A/B testing

Computes separate performance metrics for base vs canary allocations
and stores daily results in the canary_performance table.
"""

import logging
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple

from ..datasource.storage import DataStorage
from ..execution.fill_store import FillStore

logger = logging.getLogger(__name__)


def compute_tag_based_nav(target_date: Optional[date] = None) -> Dict[str, float]:
    """
    Compute NAV (Net Asset Value) for each allocation tag.
    
    Args:
        target_date: Date to compute NAV for (default: today)
        
    Returns:
        Dictionary with NAV by tag
        
    Example:
        >>> nav_by_tag = compute_tag_based_nav(date(2025, 6, 10))
        >>> print(nav_by_tag)
        {'base': 90000.0, 'ml_canary': 10000.0}
    """
    try:
        if target_date is None:
            target_date = date.today()
        
        fill_store = FillStore()
        
        # Get all fills up to target date, grouped by tag
        query = """
            SELECT 
                tag,
                symbol,
                SUM(quantity) as net_quantity,
                SUM(quantity * price) / SUM(quantity) as avg_price,
                SUM(gross_value) as total_invested,
                SUM(total_fees) as total_fees,
                COUNT(*) as fill_count
            FROM fills 
            WHERE DATE(filled_at) <= ?
                AND tag IS NOT NULL
            GROUP BY tag, symbol
            HAVING net_quantity != 0
        """
        
        fills_df = pd.read_sql_query(query, fill_store.conn, params=[str(target_date)])
        fill_store.close()
        
        if fills_df.empty:
            logger.warning(f"No fills found for date {target_date}")
            return {'base': 0.0, 'ml_canary': 0.0}
        
        # Compute NAV by tag
        nav_by_tag = {}
        
        for tag in fills_df['tag'].unique():
            tag_fills = fills_df[fills_df['tag'] == tag]
            
            # For simplicity, assume current price = avg_price (in real system, would get current market prices)
            tag_nav = (tag_fills['net_quantity'] * tag_fills['avg_price']).sum()
            
            nav_by_tag[tag] = float(tag_nav)
            
            logger.debug(f"NAV for {tag}: ${tag_nav:,.2f} from {len(tag_fills)} positions")
        
        # Ensure both tags are present
        nav_by_tag.setdefault('base', 0.0)
        nav_by_tag.setdefault('ml_canary', 0.0)
        
        logger.info(f"NAV computed for {target_date}: Base=${nav_by_tag['base']:,.2f}, Canary=${nav_by_tag['ml_canary']:,.2f}")
        
        return nav_by_tag
        
    except Exception as e:
        logger.error(f"Failed to compute tag-based NAV: {e}")
        return {'base': 0.0, 'ml_canary': 0.0}


def compute_daily_pnl(target_date: Optional[date] = None) -> Dict[str, float]:
    """
    Compute daily P&L for each allocation tag.
    
    Args:
        target_date: Date to compute P&L for (default: today)
        
    Returns:
        Dictionary with daily P&L by tag
    """
    try:
        if target_date is None:
            target_date = date.today()
        
        # Get NAV for target date and previous day
        current_nav = compute_tag_based_nav(target_date)
        previous_nav = compute_tag_based_nav(target_date - timedelta(days=1))
        
        # Compute daily P&L as NAV change
        daily_pnl = {}
        for tag in ['base', 'ml_canary']:
            daily_pnl[tag] = current_nav[tag] - previous_nav[tag]
        
        logger.info(f"Daily P&L for {target_date}: Base=${daily_pnl['base']:,.2f}, Canary=${daily_pnl['ml_canary']:,.2f}")
        
        return daily_pnl
        
    except Exception as e:
        logger.error(f"Failed to compute daily P&L: {e}")
        return {'base': 0.0, 'ml_canary': 0.0}


def compute_rolling_sharpe(tag: str, window_days: int = 30, target_date: Optional[date] = None) -> float:
    """
    Compute rolling Sharpe ratio for a specific tag.
    
    Args:
        tag: Allocation tag ('base' or 'ml_canary')
        window_days: Rolling window size in days
        target_date: End date for calculation (default: today)
        
    Returns:
        Annualized Sharpe ratio
    """
    try:
        if target_date is None:
            target_date = date.today()
        
        storage = DataStorage()
        
        # Get historical P&L for the tag from canary_performance table
        start_date = target_date - timedelta(days=window_days + 10)  # Extra buffer for weekends
        
        query = """
            SELECT 
                date,
                CASE WHEN ? = 'base' THEN base_pnl ELSE canary_pnl END as daily_pnl,
                CASE WHEN ? = 'base' THEN base_nav ELSE canary_nav END as nav
            FROM canary_performance 
            WHERE date >= ? AND date <= ?
            ORDER BY date
        """
        
        pnl_df = pd.read_sql_query(query, storage.conn, params=[tag, tag, str(start_date), str(target_date)])
        storage.close()
        
        if len(pnl_df) < 5:  # Need minimum data points
            logger.warning(f"Insufficient data for Sharpe calculation: {len(pnl_df)} days")
            return 0.0
        
        # Calculate daily returns
        pnl_df['daily_return'] = pnl_df['daily_pnl'] / pnl_df['nav'].shift(1)
        daily_returns = pnl_df['daily_return'].dropna()
        
        if len(daily_returns) < 5:
            return 0.0
        
        # Calculate Sharpe ratio (assume risk-free rate = 0)
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        
        if std_return == 0:
            return 0.0
        
        # Annualize: sqrt(252) for daily to annual
        sharpe_ratio = (mean_return / std_return) * (252 ** 0.5)
        
        logger.debug(f"Sharpe ratio for {tag} ({window_days}d): {sharpe_ratio:.4f}")
        
        return float(sharpe_ratio)
        
    except Exception as e:
        logger.error(f"Failed to compute Sharpe ratio for {tag}: {e}")
        return 0.0


def store_daily_performance(target_date: Optional[date] = None) -> bool:
    """
    Store daily performance metrics in canary_performance table.
    
    Args:
        target_date: Date to store performance for (default: today)
        
    Returns:
        Boolean indicating success
    """
    try:
        if target_date is None:
            target_date = date.today()
        
        # Compute metrics
        nav_by_tag = compute_tag_based_nav(target_date)
        pnl_by_tag = compute_daily_pnl(target_date)
        
        # Compute 30-day Sharpe ratios
        base_sharpe = compute_rolling_sharpe('base', 30, target_date)
        canary_sharpe = compute_rolling_sharpe('ml_canary', 30, target_date)
        sharpe_diff = canary_sharpe - base_sharpe
        
        # Check if canary is enabled (for storage)
        from ..execution.allocation import is_canary_enabled
        canary_enabled = is_canary_enabled()
        
        # Count data points for Sharpe calculation
        storage = DataStorage()
        count_query = "SELECT COUNT(*) FROM canary_performance WHERE date >= ?"
        start_date = target_date - timedelta(days=30)
        days_count = storage.conn.execute(count_query, [str(start_date)]).fetchone()[0]
        
        # Store in database
        insert_query = """
            INSERT OR REPLACE INTO canary_performance (
                date, canary_pnl, canary_nav, base_pnl, base_nav,
                canary_sharpe_30d, base_sharpe_30d, sharpe_diff,
                canary_enabled, days_in_window, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        storage.conn.execute(insert_query, [
            str(target_date),
            pnl_by_tag['ml_canary'],
            nav_by_tag['ml_canary'], 
            pnl_by_tag['base'],
            nav_by_tag['base'],
            canary_sharpe,
            base_sharpe,
            sharpe_diff,
            canary_enabled,
            days_count,
            datetime.now()
        ])
        
        storage.close()
        
        logger.info(f"Stored performance for {target_date}: "
                   f"Base NAV=${nav_by_tag['base']:,.2f} (Sharpe: {base_sharpe:.3f}), "
                   f"Canary NAV=${nav_by_tag['ml_canary']:,.2f} (Sharpe: {canary_sharpe:.3f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to store daily performance: {e}")
        return False


def get_canary_performance_summary(days: int = 30) -> Dict:
    """
    Get summary of canary performance for dashboard.
    
    Args:
        days: Number of days to analyze
        
    Returns:
        Performance summary dictionary
    """
    try:
        storage = DataStorage()
        
        # Get recent performance data
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        query = """
            SELECT 
                AVG(canary_sharpe_30d) as avg_canary_sharpe,
                AVG(base_sharpe_30d) as avg_base_sharpe,
                AVG(sharpe_diff) as avg_sharpe_diff,
                SUM(canary_pnl) as total_canary_pnl,
                SUM(base_pnl) as total_base_pnl,
                MAX(canary_nav) as max_canary_nav,
                MAX(base_nav) as max_base_nav,
                COUNT(*) as data_points,
                MAX(canary_enabled) as canary_enabled,
                MAX(date) as latest_date
            FROM canary_performance
            WHERE date >= ? AND date <= ?
        """
        
        result = storage.conn.execute(query, [str(start_date), str(end_date)]).fetchone()
        storage.close()
        
        if not result or result[0] is None:
            return {
                'avg_canary_sharpe': 0.0,
                'avg_base_sharpe': 0.0,
                'avg_sharpe_diff': 0.0,
                'total_canary_pnl': 0.0,
                'total_base_pnl': 0.0,
                'canary_enabled': True,
                'data_points': 0,
                'latest_date': str(end_date)
            }
        
        return {
            'avg_canary_sharpe': float(result[0] or 0.0),
            'avg_base_sharpe': float(result[1] or 0.0),
            'avg_sharpe_diff': float(result[2] or 0.0),
            'total_canary_pnl': float(result[3] or 0.0),
            'total_base_pnl': float(result[4] or 0.0),
            'max_canary_nav': float(result[5] or 0.0),
            'max_base_nav': float(result[6] or 0.0),
            'data_points': int(result[7] or 0),
            'canary_enabled': bool(result[8] or True),
            'latest_date': str(result[9] or end_date)
        }
        
    except Exception as e:
        logger.error(f"Failed to get canary performance summary: {e}")
        return {
            'avg_canary_sharpe': 0.0,
            'avg_base_sharpe': 0.0,
            'avg_sharpe_diff': 0.0,
            'total_canary_pnl': 0.0,
            'total_base_pnl': 0.0,
            'canary_enabled': True,
            'data_points': 0,
            'latest_date': str(date.today()),
            'error': str(e)
        }


def get_allocation_breakdown() -> Dict[str, float]:
    """
    Get current allocation breakdown by tag.
    
    Returns:
        Dictionary with allocation percentages
    """
    try:
        nav_by_tag = compute_tag_based_nav()
        total_nav = sum(nav_by_tag.values())
        
        if total_nav == 0:
            return {'base': 1.0, 'ml_canary': 0.0}
        
        allocation = {
            tag: nav / total_nav 
            for tag, nav in nav_by_tag.items()
        }
        
        return allocation
        
    except Exception as e:
        logger.error(f"Failed to get allocation breakdown: {e}")
        return {'base': 1.0, 'ml_canary': 0.0}