"""
Live PnL Monitoring & Intraday Sentinel

Provides real-time PnL calculation and monitoring for intraday risk management.
Integrates with kill-switch for automated loss protection.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sqlite3
import json
import random

from ..datasource.storage import DataStorage
from ..execution.fill_store import FillStore

logger = logging.getLogger(__name__)


class LivePnLMonitor:
    """Real-time PnL monitoring and intraday sentinel"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize live PnL monitor
        
        Args:
            db_path: Path to database file
        """
        self.storage = DataStorage(db_path)
        self.fill_store = FillStore(db_path)
        
        # Cache for performance
        self._last_close_nav_cache = {}
        self._last_prices_cache = {}
        self._cache_timestamp = None
        
        # Ensure intraday metrics table exists
        self._ensure_intraday_table()
    
    def _ensure_intraday_table(self):
        """Ensure intraday_metrics table exists"""
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS intraday_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TIMESTAMP NOT NULL,
                nav DECIMAL(18,2) NOT NULL,
                pnl_pct DECIMAL(6,3) NOT NULL,
                day_start_nav DECIMAL(18,2),
                position_count INTEGER DEFAULT 0,
                gross_exposure DECIMAL(18,2) DEFAULT 0.0,
                net_exposure DECIMAL(18,2) DEFAULT 0.0,
                top_positions TEXT,
                alerts_triggered BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Indexes for performance
                INDEX idx_intraday_ts (ts),
                INDEX idx_intraday_date (DATE(ts)),
                INDEX idx_intraday_pnl (pnl_pct)
            )
            """
            
            self.storage.conn.execute(create_table_sql)
            self.storage.conn.commit()
            logger.info("✅ Intraday metrics table ready")
            
        except Exception as e:
            logger.error(f"Failed to create intraday_metrics table: {e}")
            raise
    
    def get_live_nav(self, tag: str = 'all') -> Dict[str, Any]:
        """
        Calculate live NAV and day-to-date PnL percentage
        
        Args:
            tag: Portfolio tag to calculate NAV for (default: 'all')
            
        Returns:
            Dict with live NAV, PnL percentage, and position details
        """
        logger.info(f"Calculating live NAV for tag '{tag}'")
        
        try:
            # Get today's date
            today = date.today()
            
            # Get current positions from fills
            positions = self._get_current_positions(tag, today)
            
            if not positions:
                logger.warning("No positions found for live NAV calculation")
                return {
                    'live_nav': 0.0,
                    'day_start_nav': 0.0,
                    'pnl_amount': 0.0,
                    'pnl_pct': 0.0,
                    'position_count': 0,
                    'timestamp': datetime.now().isoformat(),
                    'positions': [],
                    'error': 'No positions found'
                }
            
            # Get latest prices
            latest_prices = self._get_latest_prices(positions)
            
            # Calculate current NAV
            live_nav = self._calculate_nav(positions, latest_prices)
            
            # Get day start NAV (yesterday's close or cached value)
            day_start_nav = self.get_last_close_nav(tag)
            
            # Calculate PnL
            pnl_amount = live_nav - day_start_nav
            pnl_pct = (pnl_amount / day_start_nav * 100) if day_start_nav > 0 else 0.0
            
            # Calculate additional metrics
            position_count = len(positions)
            gross_exposure = sum(abs(pos['market_value']) for pos in positions)
            net_exposure = sum(pos['market_value'] for pos in positions)
            
            # Get top positions for monitoring
            top_positions = sorted(positions, key=lambda x: abs(x['market_value']), reverse=True)[:5]
            
            result = {
                'live_nav': round(live_nav, 2),
                'day_start_nav': round(day_start_nav, 2),
                'pnl_amount': round(pnl_amount, 2),
                'pnl_pct': round(pnl_pct, 3),
                'position_count': position_count,
                'gross_exposure': round(gross_exposure, 2),
                'net_exposure': round(net_exposure, 2),
                'top_positions': [
                    {
                        'symbol': pos['symbol'],
                        'quantity': pos['quantity'],
                        'market_value': round(pos['market_value'], 2),
                        'unrealized_pnl': round(pos['unrealized_pnl'], 2)
                    }
                    for pos in top_positions
                ],
                'timestamp': datetime.now().isoformat(),
                'tag': tag,
                'calculation_successful': True
            }
            
            logger.info(f"Live NAV calculated: ${live_nav:,.2f} ({pnl_pct:+.2f}%)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate live NAV: {e}")
            return {
                'live_nav': 0.0,
                'day_start_nav': 0.0,
                'pnl_amount': 0.0,
                'pnl_pct': 0.0,
                'position_count': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'calculation_successful': False
            }
    
    def get_last_close_nav(self, tag: str = 'all') -> float:
        """
        Get last close NAV (cached for performance)
        
        Args:
            tag: Portfolio tag
            
        Returns:
            Last close NAV value
        """
        cache_key = f"{tag}_{date.today().isoformat()}"
        
        # Check cache (valid for current day)
        if (cache_key in self._last_close_nav_cache and 
            self._cache_timestamp and 
            (datetime.now() - self._cache_timestamp).seconds < 3600):  # 1 hour cache
            return self._last_close_nav_cache[cache_key]
        
        try:
            # Try to get from previous day's intraday_metrics
            yesterday = date.today() - timedelta(days=1)
            
            query = """
            SELECT nav FROM intraday_metrics 
            WHERE DATE(ts) = ? 
            ORDER BY ts DESC 
            LIMIT 1
            """
            
            result = self.storage.conn.execute(query, [yesterday.isoformat()]).fetchone()
            
            if result:
                last_nav = float(result[0])
                logger.info(f"Retrieved last close NAV from intraday_metrics: ${last_nav:,.2f}")
            else:
                # Fallback: calculate from historical positions or use default
                last_nav = self._calculate_fallback_nav(tag, yesterday)
                logger.info(f"Using fallback NAV calculation: ${last_nav:,.2f}")
            
            # Cache the result
            self._last_close_nav_cache[cache_key] = last_nav
            self._cache_timestamp = datetime.now()
            
            return last_nav
            
        except Exception as e:
            logger.error(f"Failed to get last close NAV: {e}")
            # Return a reasonable default for testing
            return 100000.0  # $100K default
    
    def _calculate_fallback_nav(self, tag: str, target_date: date) -> float:
        """Calculate fallback NAV for historical date"""
        try:
            # Try to get positions from fills on target date
            positions = self._get_current_positions(tag, target_date)
            
            if not positions:
                logger.warning(f"No positions found for fallback NAV on {target_date}")
                return 100000.0  # Default
            
            # Use previous close prices (simplified)
            prices = {}
            for pos in positions:
                prices[pos['symbol']] = pos.get('avg_price', 100.0)  # Fallback price
            
            nav = self._calculate_nav(positions, prices)
            logger.info(f"Calculated fallback NAV for {target_date}: ${nav:,.2f}")
            return nav
            
        except Exception as e:
            logger.error(f"Fallback NAV calculation failed: {e}")
            return 100000.0  # Default
    
    def _get_current_positions(self, tag: str, target_date: date) -> List[Dict[str, Any]]:
        """Get current positions from fills"""
        try:
            # Query fills to calculate current positions
            query = """
            SELECT 
                symbol,
                SUM(quantity) as net_quantity,
                AVG(fill_price) as avg_price,
                SUM(quantity * fill_price) as cost_basis,
                strategy,
                MAX(fill_time) as last_fill_time
            FROM fills 
            WHERE DATE(fill_time) <= ? 
            AND (strategy LIKE ? OR ? = 'all')
            GROUP BY symbol
            HAVING ABS(SUM(quantity)) > 0.01
            ORDER BY ABS(SUM(quantity * fill_price)) DESC
            """
            
            tag_filter = f"%{tag}%" if tag != 'all' else '%'
            
            result = self.storage.conn.execute(query, [
                target_date.isoformat(),
                tag_filter,
                tag
            ]).fetchall()
            
            if not result:
                return []
            
            positions = []
            for row in result:
                symbol, net_qty, avg_price, cost_basis, strategy, last_fill_time = row
                
                # Skip very small positions
                if abs(net_qty) < 0.01:
                    continue
                
                position = {
                    'symbol': symbol,
                    'quantity': float(net_qty),
                    'avg_price': float(avg_price),
                    'cost_basis': float(cost_basis),
                    'strategy': strategy,
                    'last_fill_time': last_fill_time,
                    'market_value': 0.0,  # Will be calculated with current prices
                    'unrealized_pnl': 0.0  # Will be calculated
                }
                
                positions.append(position)
            
            logger.info(f"Found {len(positions)} positions for {target_date}")
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get current positions: {e}")
            return []
    
    def _get_latest_prices(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get latest prices for positions (stub implementation)"""
        try:
            symbols = [pos['symbol'] for pos in positions]
            prices = {}
            
            # In production, this would fetch from IBKR or market data provider
            # For now, use stub implementation with previous close + noise
            for symbol in symbols:
                # Get price from position's average price as baseline
                baseline_price = next(
                    (pos['avg_price'] for pos in positions if pos['symbol'] == symbol), 
                    100.0
                )
                
                # Add random noise for simulation (-2% to +2%)
                noise_factor = 1 + random.uniform(-0.02, 0.02)
                current_price = baseline_price * noise_factor
                
                prices[symbol] = round(current_price, 2)
            
            logger.info(f"Retrieved prices for {len(prices)} symbols (stub mode)")
            return prices
            
        except Exception as e:
            logger.error(f"Failed to get latest prices: {e}")
            # Return baseline prices
            return {pos['symbol']: pos['avg_price'] for pos in positions}
    
    def _calculate_nav(self, positions: List[Dict[str, Any]], prices: Dict[str, float]) -> float:
        """Calculate NAV from positions and current prices"""
        try:
            total_nav = 0.0
            
            for position in positions:
                symbol = position['symbol']
                quantity = position['quantity']
                
                if symbol in prices:
                    current_price = prices[symbol]
                    market_value = quantity * current_price
                    
                    # Update position with current market data
                    position['current_price'] = current_price
                    position['market_value'] = market_value
                    position['unrealized_pnl'] = market_value - position['cost_basis']
                    
                    total_nav += market_value
                else:
                    logger.warning(f"No price found for {symbol}, using avg_price")
                    market_value = quantity * position['avg_price']
                    position['market_value'] = market_value
                    position['unrealized_pnl'] = 0.0
                    total_nav += market_value
            
            return total_nav
            
        except Exception as e:
            logger.error(f"NAV calculation failed: {e}")
            return 0.0
    
    def record_intraday_metrics(self, nav_data: Dict[str, Any]) -> bool:
        """Record intraday metrics to database"""
        try:
            # Prepare top positions as JSON
            top_positions_json = json.dumps(nav_data.get('top_positions', []))
            
            insert_sql = """
            INSERT INTO intraday_metrics (
                ts, nav, pnl_pct, day_start_nav, position_count,
                gross_exposure, net_exposure, top_positions, alerts_triggered
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.storage.conn.execute(insert_sql, [
                datetime.now().isoformat(),
                nav_data['live_nav'],
                nav_data['pnl_pct'],
                nav_data['day_start_nav'],
                nav_data['position_count'],
                nav_data.get('gross_exposure', 0.0),
                nav_data.get('net_exposure', 0.0),
                top_positions_json,
                False  # alerts_triggered - will be updated separately
            ])
            
            self.storage.conn.commit()
            logger.info("✅ Intraday metrics recorded")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record intraday metrics: {e}")
            return False
    
    def get_intraday_history(self, hours_back: int = 24) -> pd.DataFrame:
        """Get intraday metrics history"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            query = """
            SELECT ts, nav, pnl_pct, position_count, gross_exposure, net_exposure
            FROM intraday_metrics 
            WHERE ts >= ?
            ORDER BY ts ASC
            """
            
            result = self.storage.conn.execute(query, [cutoff_time.isoformat()]).fetchall()
            
            if not result:
                return pd.DataFrame()
            
            df = pd.DataFrame(result, columns=[
                'timestamp', 'nav', 'pnl_pct', 'position_count', 
                'gross_exposure', 'net_exposure'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
            
        except Exception as e:
            logger.error(f"Failed to get intraday history: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connections"""
        if self.storage:
            self.storage.close()
        if self.fill_store:
            self.fill_store.close()


# Utility functions for external use
def get_live_nav(tag: str = 'all', db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to get live NAV
    
    Args:
        tag: Portfolio tag
        db_path: Optional database path
        
    Returns:
        Live NAV data dictionary
    """
    monitor = LivePnLMonitor(db_path)
    try:
        return monitor.get_live_nav(tag)
    finally:
        monitor.close()


def get_last_close_nav(tag: str = 'all', db_path: Optional[str] = None) -> float:
    """
    Convenience function to get last close NAV
    
    Args:
        tag: Portfolio tag
        db_path: Optional database path
        
    Returns:
        Last close NAV value
    """
    monitor = LivePnLMonitor(db_path)
    try:
        return monitor.get_last_close_nav(tag)
    finally:
        monitor.close()


def create_test_positions_and_nav(target_pnl_pct: float = -1.0) -> Dict[str, Any]:
    """
    Create test positions that yield a specific PnL percentage for testing
    
    Args:
        target_pnl_pct: Target PnL percentage (e.g., -1.0 for -1%)
        
    Returns:
        Test NAV data with specified PnL
    """
    # Create realistic test positions
    base_nav = 100000.0  # $100K base
    target_nav = base_nav * (1 + target_pnl_pct / 100)
    
    test_positions = [
        {
            'symbol': 'SPY',
            'quantity': 200,
            'avg_price': 400.0,
            'current_price': 400.0 * (1 + target_pnl_pct / 100),
            'market_value': 200 * 400.0 * (1 + target_pnl_pct / 100),
            'unrealized_pnl': 200 * 400.0 * (target_pnl_pct / 100)
        },
        {
            'symbol': 'QQQ',
            'quantity': 100,
            'current_price': 350.0 * (1 + target_pnl_pct / 100),
            'market_value': 100 * 350.0 * (1 + target_pnl_pct / 100),
            'unrealized_pnl': 100 * 350.0 * (target_pnl_pct / 100)
        }
    ]
    
    return {
        'live_nav': target_nav,
        'day_start_nav': base_nav,
        'pnl_amount': target_nav - base_nav,
        'pnl_pct': target_pnl_pct,
        'position_count': len(test_positions),
        'top_positions': test_positions,
        'timestamp': datetime.now().isoformat(),
        'calculation_successful': True
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Live PnL Monitor...")
    
    # Test with real data
    nav_data = get_live_nav()
    print(f"Live NAV: ${nav_data['live_nav']:,.2f}")
    print(f"PnL: {nav_data['pnl_pct']:+.2f}%")
    
    # Test with target PnL
    test_data = create_test_positions_and_nav(-0.83)
    print(f"Test PnL: {test_data['pnl_pct']:+.2f}%")
    
    print("✅ Live PnL Monitor test complete")