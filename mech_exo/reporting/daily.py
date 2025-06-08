"""
Daily reporting module for trading performance analysis
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from ..execution.fill_store import FillStore
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)


class DailyReport:
    """
    Generate daily trading performance reports
    
    Features:
    - Daily P&L calculation
    - Commission and fee tracking
    - Maximum drawdown analysis
    - Position summary
    - Export to JSON/CSV formats
    """

    def __init__(self, date: str = "today") -> None:
        """
        Initialize daily report for specified date
        
        Args:
            date: Date string (YYYY-MM-DD) or "today"
        """
        self.date_str = self._parse_date(date)
        self.target_date = datetime.strptime(self.date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        # Initialize data stores
        self.fill_store = FillStore()
        self.config_manager = ConfigManager()
        
        # Load data
        self.fills = self._load_fills()
        self.positions = self._load_positions()
        
        logger.info(f"DailyReport initialized for {self.date_str}")

    def _parse_date(self, date: str) -> str:
        """Parse date string and return YYYY-MM-DD format"""
        if date == "today":
            return datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Validate date format
        try:
            datetime.strptime(date, "%Y-%m-%d")
            return date
        except ValueError as e:
            raise ValueError(f"Invalid date format '{date}'. Use YYYY-MM-DD or 'today'") from e

    def _load_fills(self) -> pd.DataFrame:
        """Load fills for the target date"""
        try:
            # Define date range (full day UTC)
            start_date = self.target_date
            end_date = start_date + timedelta(days=1)
            
            # Query fills from database
            query = """
            SELECT 
                fill_id,
                order_id,
                symbol,
                quantity,
                price,
                commission,
                filled_at as timestamp,
                strategy,
                slippage_bps,
                reference_price,
                gross_value,
                total_fees,
                exchange,
                notes
            FROM fills 
            WHERE filled_at >= ? AND filled_at < ?
            ORDER BY filled_at
            """
            
            df = pd.read_sql_query(
                query, 
                self.fill_store.conn, 
                params=[start_date, end_date]
            )
            
            if not df.empty:
                # Calculate P&L for each fill
                df['notional'] = df['quantity'] * df['price']
                df['pnl'] = df['notional']  # Simplified - would need position tracking for real P&L
                
            logger.info(f"Loaded {len(df)} fills for {self.date_str}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading fills for {self.date_str}: {e}")
            return pd.DataFrame()

    def _load_positions(self) -> pd.DataFrame:
        """Load current positions (stub implementation)"""
        # For now, return empty DataFrame
        # In real implementation, would query position database
        return pd.DataFrame(columns=['symbol', 'quantity', 'avg_price', 'market_value', 'unrealized_pnl'])

    def summary(self) -> dict[str, Any]:
        """
        Generate daily summary statistics
        
        Returns:
            Dictionary with key performance metrics
        """
        if self.fills.empty:
            return {
                "date": self.date_str,
                "daily_pnl": 0.0,
                "fees": 0.0,
                "max_dd": 0.0,
                "trade_count": 0,
                "volume": 0.0,
                "avg_slippage_bps": 0.0,
                "avg_routing_latency_ms": 0.0,
                "strategies": [],
                "symbols": []
            }

        # Calculate metrics
        daily_pnl = float(self.fills['pnl'].sum())
        fees = float(self.fills['commission'].sum())
        trade_count = len(self.fills)
        volume = float(self.fills['notional'].abs().sum())
        
        # Calculate slippage averages
        avg_slippage = float(self.fills['slippage_bps'].mean()) if 'slippage_bps' in self.fills.columns and not self.fills['slippage_bps'].isna().all() else 0.0
        
        # Get unique strategies and symbols
        strategies = self.fills['strategy'].dropna().unique().tolist()
        symbols = self.fills['symbol'].unique().tolist()
        
        # Calculate maximum drawdown (simplified)
        max_dd = self._calculate_max_drawdown()
        
        return {
            "date": self.date_str,
            "daily_pnl": daily_pnl,
            "fees": fees,
            "max_dd": max_dd,
            "trade_count": trade_count,
            "volume": volume,
            "avg_slippage_bps": avg_slippage,
            "strategies": strategies,
            "symbols": symbols
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown for the day"""
        if self.fills.empty:
            return 0.0
            
        # Calculate cumulative P&L throughout the day
        fills_sorted = self.fills.sort_values('timestamp')
        fills_sorted['cumulative_pnl'] = fills_sorted['pnl'].cumsum()
        
        # Calculate running maximum and drawdown
        fills_sorted['running_max'] = fills_sorted['cumulative_pnl'].expanding().max()
        fills_sorted['drawdown'] = fills_sorted['cumulative_pnl'] - fills_sorted['running_max']
        
        return float(fills_sorted['drawdown'].min())

    def detailed_breakdown(self) -> dict[str, Any]:
        """
        Generate detailed breakdown by strategy and symbol
        
        Returns:
            Detailed performance breakdown
        """
        breakdown = {
            "date": self.date_str,
            "by_strategy": {},
            "by_symbol": {},
            "hourly_pnl": []
        }
        
        if self.fills.empty:
            return breakdown
            
        # Breakdown by strategy
        if 'strategy' in self.fills.columns:
            strategy_groups = self.fills.groupby('strategy')
            for strategy, group in strategy_groups:
                if pd.notna(strategy):
                    breakdown["by_strategy"][strategy] = {
                        "pnl": float(group['pnl'].sum()),
                        "fees": float(group['commission'].sum()),
                        "trade_count": len(group),
                        "volume": float(group['notional'].abs().sum())
                    }
        
        # Breakdown by symbol
        symbol_groups = self.fills.groupby('symbol')
        for symbol, group in symbol_groups:
            breakdown["by_symbol"][symbol] = {
                "pnl": float(group['pnl'].sum()),
                "fees": float(group['commission'].sum()),
                "trade_count": len(group),
                "volume": float(group['notional'].abs().sum())
            }
        
        # Hourly P&L (if there are fills)
        if not self.fills.empty:
            fills_with_hour = self.fills.copy()
            fills_with_hour['hour'] = pd.to_datetime(fills_with_hour['timestamp']).dt.hour
            hourly_groups = fills_with_hour.groupby('hour')
            
            for hour, group in hourly_groups:
                breakdown["hourly_pnl"].append({
                    "hour": int(hour),
                    "pnl": float(group['pnl'].sum()),
                    "trade_count": len(group)
                })
        
        return breakdown

    def to_json(self, file_path=None) -> str:
        """
        Export report to JSON format
        
        Args:
            file_path: Optional file path to save JSON
            
        Returns:
            JSON string representation
        """
        report_data = {
            "summary": self.summary(),
            "breakdown": self.detailed_breakdown(),
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "fill_count": len(self.fills),
                "position_count": len(self.positions)
            }
        }
        
        json_str = json.dumps(report_data, indent=2, default=str)
        
        if file_path:
            file_path.write_text(json_str)
            logger.info(f"Daily report saved to {file_path}")
            
        return json_str

    def to_csv(self, file_path: Path) -> None:
        """Export fills data to CSV format"""
        if self.fills.empty:
            logger.warning(f"No fills data to export for {self.date_str}")
            return
            
        self.fills.to_csv(file_path, index=False)
        logger.info(f"Fills data exported to {file_path}")


def generate_daily_report(date: str = "today") -> DailyReport:
    """
    Convenience function to generate a daily report
    
    Args:
        date: Date string (YYYY-MM-DD) or "today"
        
    Returns:
        DailyReport instance
    """
    return DailyReport(date=date)