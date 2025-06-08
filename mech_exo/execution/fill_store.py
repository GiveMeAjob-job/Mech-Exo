"""
Fill store for persisting execution data to DuckDB
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb

from ..utils import ConfigManager
from .models import Fill, Order

logger = logging.getLogger(__name__)


class FillStore:
    """
    Store for persisting fills and execution data to DuckDB
    
    Features:
    - Fill storage with complete execution details
    - Order tracking and linking
    - Slippage and cost analysis
    - Performance metrics calculation
    - Data quality monitoring
    """

    def __init__(self, db_path=None) -> None:
        self.config_manager = ConfigManager()

        # Database setup
        if db_path:
            self.db_path = db_path
        else:
            data_dir = Path(__file__).parent.parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            self.db_path = str(data_dir / "execution.db")

        self.conn = None
        self._connect()
        self._create_tables()

        logger.info(f"FillStore initialized with database: {self.db_path}")

    def _ensure_utc(self, dt: datetime) -> datetime:
        """
        Ensure datetime is UTC timezone-aware.
        
        Args:
            dt: Datetime to convert to UTC
            
        Returns:
            datetime: UTC timezone-aware datetime
        """
        if dt.tzinfo is None:
            # Assume UTC if no timezone info
            return dt.replace(tzinfo=timezone.utc)
        if dt.tzinfo != timezone.utc:
            # Convert to UTC
            return dt.astimezone(timezone.utc)
        return dt

    def _connect(self) -> None:
        """Connect to DuckDB"""
        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info("Connected to execution database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _create_tables(self) -> None:
        """Create required database tables"""
        try:
            # Fills table with timezone-aware timestamps
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    fill_id VARCHAR PRIMARY KEY,
                    order_id VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    quantity INTEGER NOT NULL,
                    price DOUBLE NOT NULL,
                    filled_at TIMESTAMPTZ NOT NULL,
                    
                    -- Broker info
                    broker_order_id VARCHAR,
                    broker_fill_id VARCHAR,
                    exchange VARCHAR,
                    
                    -- Costs
                    commission DOUBLE DEFAULT 0.0,
                    fees DOUBLE DEFAULT 0.0,
                    sec_fee DOUBLE DEFAULT 0.0,
                    
                    -- Execution quality
                    reference_price DOUBLE,
                    slippage_bps DOUBLE,
                    
                    -- Metadata
                    strategy VARCHAR,
                    notes TEXT,
                    
                    -- Derived fields
                    side VARCHAR AS (CASE WHEN quantity > 0 THEN 'BUY' ELSE 'SELL' END),
                    abs_quantity INTEGER AS (ABS(quantity)),
                    gross_value DOUBLE AS (ABS(quantity) * price),
                    total_fees DOUBLE AS (commission + fees + sec_fee),
                    net_value DOUBLE AS (ABS(quantity) * price - commission - fees - sec_fee),
                    
                    -- Timestamps (UTC)
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Orders table (for tracking) with text fields and timezone support
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id VARCHAR PRIMARY KEY,
                    symbol VARCHAR NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_type VARCHAR NOT NULL CHECK (order_type IN ('MKT', 'LMT', 'STP', 'STP LMT', 'BRACKET', 'IOC', 'GTD')),
                    
                    -- Pricing
                    limit_price DOUBLE,
                    stop_price DOUBLE,
                    
                    -- Status
                    status VARCHAR NOT NULL CHECK (status IN ('PENDING', 'SUBMITTED', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'REJECTED', 'INACTIVE')),
                    broker_order_id VARCHAR,
                    
                    -- Timing (UTC)
                    created_at TIMESTAMPTZ NOT NULL,
                    submitted_at TIMESTAMPTZ,
                    
                    -- Metadata
                    strategy VARCHAR,
                    signal_strength DOUBLE DEFAULT 1.0,
                    notes TEXT,
                    
                    -- Derived fields  
                    side VARCHAR AS (CASE WHEN quantity > 0 THEN 'BUY' ELSE 'SELL' END),
                    abs_quantity INTEGER AS (ABS(quantity)),
                    
                    -- Timestamps
                    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Execution metrics table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_metrics (
                    date DATE PRIMARY KEY,
                    
                    -- Order counts
                    total_orders INTEGER DEFAULT 0,
                    filled_orders INTEGER DEFAULT 0,
                    rejected_orders INTEGER DEFAULT 0,
                    cancelled_orders INTEGER DEFAULT 0,
                    
                    -- Fill metrics
                    total_fills INTEGER DEFAULT 0,
                    total_volume DOUBLE DEFAULT 0.0,
                    total_notional DOUBLE DEFAULT 0.0,
                    
                    -- Cost analysis
                    total_commission DOUBLE DEFAULT 0.0,
                    total_fees DOUBLE DEFAULT 0.0,
                    avg_commission_per_share DOUBLE DEFAULT 0.0,
                    
                    -- Execution quality
                    avg_slippage_bps DOUBLE DEFAULT 0.0,
                    positive_slippage_pct DOUBLE DEFAULT 0.0,
                    execution_rate DOUBLE DEFAULT 0.0,  -- filled/submitted
                    
                    -- Performance
                    gross_pnl DOUBLE DEFAULT 0.0,
                    net_pnl DOUBLE DEFAULT 0.0,
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for performance
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_symbol_date ON fills(symbol, filled_at)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_strategy ON fills(strategy)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")

            logger.info("Database tables created successfully")

        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

    def store_fill(self, fill: Fill) -> bool:
        """
        Store a fill in the database with comprehensive execution details.
        
        Args:
            fill: Fill object containing execution details
            
        Returns:
            bool: True if storage successful, False otherwise
        """
        try:
            # Ensure filled_at is UTC timezone-aware
            filled_at_utc = self._ensure_utc(fill.filled_at)

            # Insert fill
            self.conn.execute("""
                INSERT OR REPLACE INTO fills (
                    fill_id, order_id, symbol, quantity, price, filled_at,
                    broker_order_id, broker_fill_id, exchange,
                    commission, fees, sec_fee,
                    reference_price, slippage_bps,
                    strategy, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                fill.fill_id, fill.order_id, fill.symbol, fill.quantity, fill.price,
                filled_at_utc,
                fill.broker_order_id, fill.broker_fill_id, fill.exchange,
                fill.commission, fill.fees, fill.sec_fee,
                fill.reference_price, fill.slippage_bps,
                fill.strategy, fill.notes
            ])

            logger.info(f"Stored fill: {fill.symbol} {fill.quantity} @ ${fill.price}")

            # Update daily metrics
            self._update_daily_metrics(fill.filled_at.date())

            return True

        except Exception as e:
            logger.error(f"Failed to store fill {fill.fill_id}: {e}")
            return False

    def store_order(self, order: Order) -> bool:
        """
        Store or update an order in the database.
        
        Args:
            order: Order object to store/update
            
        Returns:
            bool: True if storage successful, False otherwise
        """
        try:
            # Ensure timestamps are UTC timezone-aware
            created_at_utc = self._ensure_utc(order.created_at)
            submitted_at_utc = self._ensure_utc(order.submitted_at) if order.submitted_at else None

            # Insert or update order
            self.conn.execute("""
                INSERT OR REPLACE INTO orders (
                    order_id, symbol, quantity, order_type,
                    limit_price, stop_price, status, broker_order_id,
                    created_at, submitted_at, strategy, signal_strength, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                order.order_id, order.symbol, order.quantity, order.order_type.value,
                order.limit_price, order.stop_price, order.status.value, order.broker_order_id,
                created_at_utc,
                submitted_at_utc,
                order.strategy, order.signal_strength, order.notes
            ])

            logger.debug(f"Stored order: {order.symbol} {order.quantity} - {order.status.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to store order {order.order_id}: {e}")
            return False

    def get_fills(self, symbol=None, start_date=None, end_date=None, strategy=None):
        """
        Retrieve fills with optional filtering criteria.
        
        Args:
            symbol: Filter by specific symbol (optional)
            start_date: Filter fills from this date onwards (optional)
            end_date: Filter fills up to this date (optional)
            strategy: Filter by strategy name (optional)
            
        Returns:
            List[Fill]: List of fills matching the criteria, ordered by filled_at DESC
        """
        try:
            query = "SELECT * FROM fills WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if start_date:
                query += " AND filled_at >= ?"
                params.append(self._ensure_utc(start_date))

            if end_date:
                query += " AND filled_at <= ?"
                params.append(self._ensure_utc(end_date))

            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)

            query += " ORDER BY filled_at DESC"

            results = self.conn.execute(query, params).fetchall()
            columns = [desc[0] for desc in self.conn.description]

            fills = []
            for row in results:
                row_dict = dict(zip(columns, row, strict=False))

                # Convert back to Fill object - handle timezone-aware datetime
                filled_at = row_dict["filled_at"]
                if isinstance(filled_at, str):
                    filled_at = datetime.fromisoformat(filled_at)
                if filled_at.tzinfo is None:
                    filled_at = filled_at.replace(tzinfo=timezone.utc)

                fill = Fill(
                    order_id=row_dict["order_id"],
                    symbol=row_dict["symbol"],
                    quantity=row_dict["quantity"],
                    price=row_dict["price"],
                    filled_at=filled_at,
                    fill_id=row_dict["fill_id"],
                    broker_order_id=row_dict["broker_order_id"],
                    broker_fill_id=row_dict["broker_fill_id"],
                    exchange=row_dict["exchange"],
                    commission=row_dict["commission"] or 0.0,
                    fees=row_dict["fees"] or 0.0,
                    sec_fee=row_dict["sec_fee"] or 0.0,
                    reference_price=row_dict["reference_price"],
                    slippage_bps=row_dict["slippage_bps"],
                    strategy=row_dict["strategy"],
                    notes=row_dict["notes"]
                )
                fills.append(fill)

            return fills

        except Exception as e:
            logger.error(f"Failed to retrieve fills: {e}")
            return []

    def get_execution_summary(self, start_date=None, end_date=None):
        """
        Get comprehensive execution summary statistics for a date range.
        
        Args:
            start_date: Start of analysis period (defaults to 30 days ago)
            end_date: End of analysis period (defaults to now)
            
        Returns:
            Dict containing execution metrics including fills, orders, costs, and quality
        """
        try:
            # Default to last 30 days if no dates provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=30)

            # Ensure dates are UTC timezone-aware
            start_date_utc = self._ensure_utc(start_date)
            end_date_utc = self._ensure_utc(end_date)

            # Get fill statistics
            fill_stats = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_fills,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    SUM(ABS(quantity)) as total_shares,
                    SUM(gross_value) as total_notional,
                    SUM(total_fees) as total_costs,
                    AVG(slippage_bps) as avg_slippage,
                    AVG(commission) as avg_commission,
                    SUM(CASE WHEN quantity > 0 THEN gross_value ELSE 0 END) as buy_notional,
                    SUM(CASE WHEN quantity < 0 THEN gross_value ELSE 0 END) as sell_notional
                FROM fills 
                WHERE filled_at BETWEEN ? AND ?
            """, [start_date_utc, end_date_utc]).fetchone()

            # Get order statistics
            order_stats = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_orders,
                    SUM(CASE WHEN status = 'FILLED' THEN 1 ELSE 0 END) as filled_orders,
                    SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END) as rejected_orders,
                    SUM(CASE WHEN status = 'CANCELLED' THEN 1 ELSE 0 END) as cancelled_orders
                FROM orders 
                WHERE created_at BETWEEN ? AND ?
            """, [start_date_utc, end_date_utc]).fetchone()

            # Calculate metrics
            fill_rate = order_stats[1] / order_stats[0] if order_stats[0] > 0 else 0
            cost_per_share = fill_stats[4] / fill_stats[2] if fill_stats[2] > 0 else 0
            cost_bps = (fill_stats[4] / fill_stats[3] * 10000) if fill_stats[3] > 0 else 0

            return {
                "period": {
                    "start_date": start_date.date(),
                    "end_date": end_date.date(),
                    "days": (end_date - start_date).days
                },
                "fills": {
                    "total_fills": fill_stats[0],
                    "unique_symbols": fill_stats[1],
                    "total_shares": fill_stats[2],
                    "total_notional": fill_stats[3],
                    "buy_notional": fill_stats[7],
                    "sell_notional": fill_stats[8]
                },
                "orders": {
                    "total_orders": order_stats[0],
                    "filled_orders": order_stats[1],
                    "rejected_orders": order_stats[2],
                    "cancelled_orders": order_stats[3],
                    "fill_rate": fill_rate
                },
                "costs": {
                    "total_costs": fill_stats[4],
                    "avg_commission": fill_stats[6],
                    "cost_per_share": cost_per_share,
                    "cost_bps": cost_bps
                },
                "execution_quality": {
                    "avg_slippage_bps": fill_stats[5],
                    "fill_rate": fill_rate
                }
            }

        except Exception as e:
            logger.error(f"Failed to get execution summary: {e}")
            return {}

    def get_slippage_analysis(self, symbol=None, days=30):
        """Analyze slippage patterns"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            query = """
                SELECT 
                    symbol,
                    side,
                    COUNT(*) as fill_count,
                    AVG(slippage_bps) as avg_slippage,
                    MEDIAN(slippage_bps) as median_slippage,
                    STDDEV(slippage_bps) as slippage_std,
                    MIN(slippage_bps) as min_slippage,
                    MAX(slippage_bps) as max_slippage,
                    SUM(CASE WHEN slippage_bps > 0 THEN 1 ELSE 0 END) / COUNT(*) as positive_slippage_rate
                FROM fills 
                WHERE filled_at BETWEEN ? AND ? 
                    AND slippage_bps IS NOT NULL
            """

            params = [start_date.isoformat(), end_date.isoformat()]

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            query += " GROUP BY symbol, side ORDER BY symbol, side"

            results = self.conn.execute(query, params).fetchall()
            columns = [desc[0] for desc in self.conn.description]

            analysis = []
            for row in results:
                analysis.append(dict(zip(columns, row, strict=False)))

            return {
                "period_days": days,
                "analysis": analysis,
                "total_fills_analyzed": sum(a["fill_count"] for a in analysis)
            }

        except Exception as e:
            logger.error(f"Failed to analyze slippage: {e}")
            return {}

    def _update_daily_metrics(self, date: datetime.date) -> None:
        """Update daily execution metrics"""
        try:
            # Get daily statistics
            day_start = datetime.combine(date, datetime.min.time())
            day_end = datetime.combine(date, datetime.max.time())

            # Calculate metrics for the day
            metrics = self.conn.execute("""
                SELECT 
                    COUNT(DISTINCT o.order_id) as total_orders,
                    SUM(CASE WHEN o.status = 'FILLED' THEN 1 ELSE 0 END) as filled_orders,
                    SUM(CASE WHEN o.status = 'REJECTED' THEN 1 ELSE 0 END) as rejected_orders,
                    SUM(CASE WHEN o.status = 'CANCELLED' THEN 1 ELSE 0 END) as cancelled_orders,
                    COUNT(f.fill_id) as total_fills,
                    SUM(f.abs_quantity) as total_volume,
                    SUM(f.gross_value) as total_notional,
                    SUM(f.commission) as total_commission,
                    SUM(f.total_fees) as total_fees,
                    AVG(f.slippage_bps) as avg_slippage,
                    SUM(CASE WHEN f.slippage_bps > 0 THEN 1 ELSE 0 END) / COUNT(f.fill_id) as positive_slippage_pct
                FROM orders o
                LEFT JOIN fills f ON o.order_id = f.order_id AND f.filled_at BETWEEN ? AND ?
                WHERE o.created_at BETWEEN ? AND ?
            """, [day_start.isoformat(), day_end.isoformat(),
                  day_start.isoformat(), day_end.isoformat()]).fetchone()

            # Calculate derived metrics
            execution_rate = metrics[1] / metrics[0] if metrics[0] > 0 else 0
            avg_comm_per_share = metrics[7] / metrics[5] if metrics[5] > 0 else 0

            # Insert or update daily metrics
            self.conn.execute("""
                INSERT OR REPLACE INTO execution_metrics (
                    date, total_orders, filled_orders, rejected_orders, cancelled_orders,
                    total_fills, total_volume, total_notional,
                    total_commission, total_fees, avg_commission_per_share,
                    avg_slippage_bps, positive_slippage_pct, execution_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                date, metrics[0], metrics[1], metrics[2], metrics[3],
                metrics[4], metrics[5] or 0, metrics[6] or 0,
                metrics[7] or 0, metrics[8] or 0, avg_comm_per_share,
                metrics[9] or 0, metrics[10] or 0, execution_rate
            ])

            logger.debug(f"Updated daily metrics for {date}")

        except Exception as e:
            logger.error(f"Failed to update daily metrics for {date}: {e}")

    def get_top_symbols_by_volume(self, days=30, limit=10):
        """Get top symbols by trading volume"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            results = self.conn.execute("""
                SELECT 
                    symbol,
                    COUNT(*) as fill_count,
                    SUM(abs_quantity) as total_shares,
                    SUM(gross_value) as total_notional,
                    AVG(slippage_bps) as avg_slippage,
                    SUM(total_fees) as total_costs
                FROM fills 
                WHERE filled_at BETWEEN ? AND ?
                GROUP BY symbol
                ORDER BY total_notional DESC
                LIMIT ?
            """, [start_date.isoformat(), end_date.isoformat(), limit]).fetchall()

            columns = ["symbol", "fill_count", "total_shares", "total_notional",
                      "avg_slippage", "total_costs"]

            return [dict(zip(columns, row, strict=False)) for row in results]

        except Exception as e:
            logger.error(f"Failed to get top symbols: {e}")
            return []

    def get_daily_metrics(self, date):
        """Get daily metrics for a specific date"""
        try:
            # Convert date to UTC datetime range
            day_start = datetime.combine(date, datetime.min.time()).replace(tzinfo=timezone.utc)
            day_end = datetime.combine(date, datetime.max.time()).replace(tzinfo=timezone.utc)

            # Get fills for the day
            fills_query = """
                SELECT 
                    COUNT(*) as total_fills,
                    COUNT(DISTINCT symbol) as symbols_traded,
                    SUM(abs_quantity) as total_volume,
                    SUM(gross_value) as total_notional,
                    SUM(total_fees) as total_fees,
                    AVG(slippage_bps) as avg_slippage_bps,
                    AVG(commission) as avg_commission,
                    MIN(filled_at) as first_fill_time,
                    MAX(filled_at) as last_fill_time
                FROM fills 
                WHERE filled_at BETWEEN ? AND ?
            """

            fill_metrics = self.conn.execute(fills_query, [day_start, day_end]).fetchone()

            # Get orders for the day
            orders_query = """
                SELECT 
                    COUNT(*) as total_orders,
                    SUM(CASE WHEN status = 'FILLED' THEN 1 ELSE 0 END) as filled_orders,
                    SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END) as rejected_orders,
                    SUM(CASE WHEN status = 'CANCELLED' THEN 1 ELSE 0 END) as cancelled_orders
                FROM orders 
                WHERE created_at BETWEEN ? AND ?
            """

            order_metrics = self.conn.execute(orders_query, [day_start, day_end]).fetchone()

            # Calculate derived metrics
            fill_rate = order_metrics[1] / order_metrics[0] if order_metrics[0] > 0 else 0
            avg_cost_per_share = fill_metrics[4] / fill_metrics[2] if fill_metrics[2] > 0 else 0
            cost_bps = (fill_metrics[4] / fill_metrics[3] * 10000) if fill_metrics[3] > 0 else 0

            # Format result
            return {
                "date": date,
                "fills": {
                    "total_fills": fill_metrics[0] or 0,
                    "symbols_traded": fill_metrics[1] or 0,
                    "total_volume": fill_metrics[2] or 0,
                    "total_notional": fill_metrics[3] or 0.0,
                    "first_fill_time": fill_metrics[7],
                    "last_fill_time": fill_metrics[8]
                },
                "orders": {
                    "total_orders": order_metrics[0] or 0,
                    "filled_orders": order_metrics[1] or 0,
                    "rejected_orders": order_metrics[2] or 0,
                    "cancelled_orders": order_metrics[3] or 0,
                    "fill_rate": fill_rate
                },
                "costs": {
                    "total_fees": fill_metrics[4] or 0.0,
                    "avg_commission": fill_metrics[6] or 0.0,
                    "avg_cost_per_share": avg_cost_per_share,
                    "cost_bps": cost_bps
                },
                "execution_quality": {
                    "avg_slippage_bps": fill_metrics[5] or 0.0
                }
            }

        except Exception as e:
            logger.error(f"Failed to get daily metrics for {date}: {e}")
            return {
                "date": date,
                "error": str(e),
                "fills": {"total_fills": 0},
                "orders": {"total_orders": 0},
                "costs": {"total_fees": 0.0},
                "execution_quality": {"avg_slippage_bps": 0.0}
            }

    def last_fill_ts(self, symbol=None):
        """Get timestamp of last fill, optionally for specific symbol"""
        try:
            query = "SELECT MAX(filled_at) FROM fills"
            params = []

            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol)

            result = self.conn.execute(query, params).fetchone()
            last_ts = result[0] if result and result[0] else None

            if last_ts and isinstance(last_ts, str):
                last_ts = datetime.fromisoformat(last_ts)

            if last_ts and last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)

            return last_ts

        except Exception as e:
            logger.error(f"Failed to get last fill timestamp: {e}")
            return None

    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("FillStore database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
