"""
Monthly Loss Guard - Monthly Drawdown Protection

Implements monthly P&L tracking and -3% stop-loss protection.
Part of P10 Week 3 Day 3 implementation.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional, List
import calendar
import pandas as pd

from ..datasource.storage import DataStorage
from ..reporting.pnl_live import get_live_nav

logger = logging.getLogger(__name__)


def get_mtd_pnl_pct(target_date: date = None) -> float:
    """
    Calculate month-to-date PnL percentage
    
    Sum daily P&L from daily_metrics/intraday_metrics between month-start 
    and target_date - 1 day (excluding current day)
    
    Args:
        target_date: Date to calculate MTD for (default: today)
        
    Returns:
        Monthly PnL percentage (negative for losses)
    """
    if target_date is None:
        target_date = date.today()
    
    logger.info(f"Calculating MTD PnL for {target_date}")
    
    try:
        # Get month start date
        month_start = date(target_date.year, target_date.month, 1)
        
        # Calculate up to target_date - 1 day (excluding current day as specified)
        end_date = target_date - timedelta(days=1)
        
        if end_date < month_start:
            logger.warning(f"End date {end_date} is before month start {month_start}")
            return 0.0
        
        # Get month start NAV
        month_start_nav = get_month_start_nav(target_date)
        
        if month_start_nav <= 0:
            logger.error(f"Invalid month start NAV: {month_start_nav}")
            return 0.0
        
        # Get current NAV (as of end_date)
        current_nav = _get_nav_for_date(end_date)
        
        if current_nav <= 0:
            logger.warning(f"Using month start NAV as fallback for current NAV")
            current_nav = month_start_nav
        
        # Calculate MTD PnL percentage
        mtd_amount = current_nav - month_start_nav
        mtd_pct = (mtd_amount / month_start_nav) * 100
        
        logger.info(f"MTD PnL: {mtd_pct:.3f}% (${mtd_amount:,.2f})")
        logger.info(f"Month start NAV: ${month_start_nav:,.2f}, Current NAV: ${current_nav:,.2f}")
        
        return round(mtd_pct, 3)
        
    except Exception as e:
        logger.error(f"Failed to calculate MTD PnL: {e}")
        return 0.0


def get_month_start_nav(target_date: date = None) -> float:
    """
    Get baseline NAV for the start of the month
    
    Args:
        target_date: Date to get month start for (default: today)
        
    Returns:
        NAV value at start of month
    """
    if target_date is None:
        target_date = date.today()
    
    try:
        # Get first day of the month
        month_start = date(target_date.year, target_date.month, 1)
        
        # Try to get NAV from the last day of previous month
        prev_month_end = month_start - timedelta(days=1)
        
        logger.info(f"Getting month start NAV for {target_date.strftime('%Y-%m')} (prev month end: {prev_month_end})")
        
        # First try to get from daily_metrics
        nav = _get_nav_from_daily_metrics(prev_month_end)
        
        if nav > 0:
            logger.info(f"Found month start NAV from daily metrics: ${nav:,.2f}")
            return nav
        
        # Fall back to intraday_metrics
        nav = _get_nav_from_intraday_metrics(prev_month_end)
        
        if nav > 0:
            logger.info(f"Found month start NAV from intraday metrics: ${nav:,.2f}")
            return nav
        
        # Final fallback - use current live NAV (for very first month)
        logger.warning("No historical NAV found, using current live NAV as fallback")
        nav_data = get_live_nav()
        fallback_nav = nav_data.get('live_nav', 100000.0)  # Default to 100k
        
        logger.info(f"Using fallback NAV: ${fallback_nav:,.2f}")
        return fallback_nav
        
    except Exception as e:
        logger.error(f"Failed to get month start NAV: {e}")
        return 100000.0  # Default fallback


def _get_nav_for_date(target_date: date) -> float:
    """Get NAV for a specific date"""
    try:
        # Try daily metrics first
        nav = _get_nav_from_daily_metrics(target_date)
        if nav > 0:
            return nav
        
        # Fall back to intraday metrics
        nav = _get_nav_from_intraday_metrics(target_date)
        if nav > 0:
            return nav
        
        # If it's today, try live NAV
        if target_date == date.today():
            nav_data = get_live_nav()
            return nav_data.get('live_nav', 0.0)
        
        logger.warning(f"No NAV found for {target_date}")
        return 0.0
        
    except Exception as e:
        logger.error(f"Failed to get NAV for {target_date}: {e}")
        return 0.0


def _get_nav_from_daily_metrics(target_date: date) -> float:
    """Get NAV from daily_metrics table"""
    try:
        storage = DataStorage()
        
        # Check if daily_metrics table exists and query for NAV
        query = """
        SELECT nav_close 
        FROM daily_metrics 
        WHERE DATE(date) = ?
        ORDER BY date DESC
        LIMIT 1
        """
        
        result = storage.conn.execute(query, [target_date.isoformat()]).fetchone()
        storage.close()
        
        if result and result[0] is not None:
            nav = float(result[0])
            logger.debug(f"Found daily NAV for {target_date}: ${nav:,.2f}")
            return nav
        
        logger.debug(f"No daily NAV found for {target_date}")
        return 0.0
        
    except Exception as e:
        if "no such table" in str(e).lower():
            logger.debug("daily_metrics table does not exist")
        else:
            logger.error(f"Error querying daily metrics: {e}")
        return 0.0


def _get_nav_from_intraday_metrics(target_date: date) -> float:
    """Get NAV from intraday_metrics table (last entry of the day)"""
    try:
        storage = DataStorage()
        
        # Get last NAV entry for the target date
        query = """
        SELECT nav 
        FROM intraday_metrics 
        WHERE DATE(ts) = ?
        ORDER BY ts DESC
        LIMIT 1
        """
        
        result = storage.conn.execute(query, [target_date.isoformat()]).fetchone()
        storage.close()
        
        if result and result[0] is not None:
            nav = float(result[0])
            logger.debug(f"Found intraday NAV for {target_date}: ${nav:,.2f}")
            return nav
        
        logger.debug(f"No intraday NAV found for {target_date}")
        return 0.0
        
    except Exception as e:
        if "no such table" in str(e).lower():
            logger.debug("intraday_metrics table does not exist")
        else:
            logger.error(f"Error querying intraday metrics: {e}")
        return 0.0


def get_monthly_config() -> Dict[str, Any]:
    """
    Get monthly stop-loss configuration
    
    Returns:
        Configuration dictionary with threshold and settings
    """
    try:
        from .config import ConfigManager
        
        config_manager = ConfigManager()
        
        # Try to load from risk.yml
        try:
            risk_config = config_manager.load_config('risk')
            monthly_config = risk_config.get('monthly_stop', {})
        except:
            monthly_config = {}
        
        # Apply defaults
        default_config = {
            'enabled': True,
            'threshold_pct': -3.0,
            'min_history_days': 10,  # Guard ignored before day 10 of month
            'alert_enabled': True,
            'dry_run': False
        }
        
        # Merge with defaults
        for key, default_value in default_config.items():
            if key not in monthly_config:
                monthly_config[key] = default_value
        
        logger.debug(f"Monthly guard config: {monthly_config}")
        return monthly_config
        
    except Exception as e:
        logger.error(f"Failed to load monthly config: {e}")
        # Return safe defaults
        return {
            'enabled': True,
            'threshold_pct': -3.0,
            'min_history_days': 10,
            'alert_enabled': True,
            'dry_run': False
        }


def should_run_monthly_guard(target_date: date = None) -> tuple[bool, str]:
    """
    Check if monthly guard should run for the given date
    
    Args:
        target_date: Date to check (default: today)
        
    Returns:
        Tuple of (should_run: bool, reason: str)
    """
    if target_date is None:
        target_date = date.today()
    
    try:
        config = get_monthly_config()
        
        # Check if enabled
        if not config.get('enabled', True):
            return False, "Monthly guard disabled in config"
        
        # Check if we have enough history (min_history_days)
        min_days = config.get('min_history_days', 10)
        if target_date.day < min_days:
            return False, f"Too early in month (day {target_date.day} < {min_days})"
        
        # Check if kill-switch is already disabled
        from ..cli.killswitch import is_trading_enabled
        if not is_trading_enabled():
            return False, "Kill-switch already disabled"
        
        return True, "Monthly guard should run"
        
    except Exception as e:
        logger.error(f"Error checking if monthly guard should run: {e}")
        return False, f"Error: {e}"


def get_mtd_summary(target_date: date = None) -> Dict[str, Any]:
    """
    Get comprehensive month-to-date summary
    
    Args:
        target_date: Date to calculate for (default: today)
        
    Returns:
        Dictionary with MTD metrics and status
    """
    if target_date is None:
        target_date = date.today()
    
    try:
        config = get_monthly_config()
        
        # Calculate MTD PnL
        mtd_pct = get_mtd_pnl_pct(target_date)
        threshold_pct = config.get('threshold_pct', -3.0)
        
        # Determine status
        if mtd_pct <= threshold_pct:
            status = 'CRITICAL'
            status_color = 'danger'
        elif mtd_pct <= (threshold_pct * 0.8):  # 80% of threshold (-2.4% if threshold is -3%)
            status = 'WARNING'
            status_color = 'warning'
        elif mtd_pct < 0:
            status = 'NEGATIVE'
            status_color = 'info'
        else:
            status = 'POSITIVE'
            status_color = 'success'
        
        # Get additional metrics
        month_start_nav = get_month_start_nav(target_date)
        current_nav = _get_nav_for_date(target_date - timedelta(days=1))
        mtd_amount = current_nav - month_start_nav if current_nav > 0 and month_start_nav > 0 else 0
        
        # Check if guard should run
        should_run, run_reason = should_run_monthly_guard(target_date)
        
        return {
            'target_date': target_date.isoformat(),
            'month_year': target_date.strftime('%Y-%m'),
            'mtd_pct': mtd_pct,
            'mtd_amount': round(mtd_amount, 2),
            'threshold_pct': threshold_pct,
            'threshold_breached': mtd_pct <= threshold_pct,
            'status': status,
            'status_color': status_color,
            'month_start_nav': round(month_start_nav, 2),
            'current_nav': round(current_nav, 2),
            'days_in_month': calendar.monthrange(target_date.year, target_date.month)[1],
            'day_of_month': target_date.day,
            'should_run_guard': should_run,
            'run_reason': run_reason,
            'config': config,
            'last_updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get MTD summary: {e}")
        return {
            'target_date': target_date.isoformat(),
            'mtd_pct': 0.0,
            'threshold_pct': -3.0,
            'threshold_breached': False,
            'status': 'ERROR',
            'status_color': 'danger',
            'error': str(e),
            'last_updated': datetime.now().isoformat()
        }


# Test/debugging functions

def create_stub_monthly_data(target_date: date, target_mtd_pct: float) -> bool:
    """
    Create stub data for testing monthly calculations
    
    Args:
        target_date: Target date for testing
        target_mtd_pct: Desired MTD percentage (e.g., -3.2)
        
    Returns:
        True if stub data created successfully
    """
    try:
        logger.info(f"Creating stub data for {target_date} with MTD {target_mtd_pct}%")
        
        storage = DataStorage()
        
        # Ensure tables exist
        _ensure_monthly_test_tables(storage)
        
        # Calculate target NAV values
        base_nav = 100000.0  # Starting NAV
        month_start = date(target_date.year, target_date.month, 1)
        target_nav = base_nav * (1 + target_mtd_pct / 100)
        
        # Insert month start data (end of previous month)
        prev_month_end = month_start - timedelta(days=1)
        storage.conn.execute("""
            INSERT OR REPLACE INTO daily_metrics (date, nav_close)
            VALUES (?, ?)
        """, [prev_month_end.isoformat(), base_nav])
        
        # Insert target date data
        end_date = target_date - timedelta(days=1)  # MTD calculation excludes current day
        storage.conn.execute("""
            INSERT OR REPLACE INTO daily_metrics (date, nav_close)
            VALUES (?, ?)
        """, [end_date.isoformat(), target_nav])
        
        storage.conn.commit()
        storage.close()
        
        logger.info(f"Created stub data: ${base_nav:,.2f} → ${target_nav:,.2f} ({target_mtd_pct:+.1f}%)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create stub data: {e}")
        return False


def _ensure_monthly_test_tables(storage: DataStorage):
    """Ensure test tables exist for monthly calculations"""
    try:
        # Create daily_metrics table if it doesn't exist
        storage.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date DATE PRIMARY KEY,
                nav_close DECIMAL(18,2),
                pnl_amount DECIMAL(18,2),
                pnl_pct DECIMAL(6,3),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        logger.debug("Monthly test tables ensured")
        
    except Exception as e:
        logger.error(f"Failed to ensure monthly test tables: {e}")


if __name__ == "__main__":
    # Test the monthly calculator
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Monthly Loss Guard...")
    
    # Test current MTD
    mtd_pct = get_mtd_pnl_pct()
    print(f"Current MTD PnL: {mtd_pct:+.2f}%")
    
    # Test with stub data
    test_date = date.today()
    create_stub_monthly_data(test_date, -3.2)
    
    # Verify calculation
    stub_mtd = get_mtd_pnl_pct(test_date)
    print(f"Stub MTD PnL: {stub_mtd:+.2f}%")
    
    # Get full summary
    summary = get_mtd_summary(test_date)
    print(f"Summary: {summary['status']} - {summary['mtd_pct']:+.2f}%")
    
    print("✅ Monthly Loss Guard test complete")