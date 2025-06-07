"""
Data storage utilities using DuckDB
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DataStorage:
    """Handles data storage and retrieval using DuckDB"""
    
    def __init__(self, db_path: str = "data/mech_exo.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            self.conn = duckdb.connect(str(self.db_path))
            self._create_tables()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables"""
        
        # OHLC data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlc_data (
                symbol VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                returns DOUBLE,
                volatility DOUBLE,
                atr DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)
        
        # Fundamental data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamental_data (
                symbol VARCHAR,
                fetch_date DATE,
                market_cap BIGINT,
                enterprise_value BIGINT,
                pe_ratio DOUBLE,
                forward_pe DOUBLE,
                peg_ratio DOUBLE,
                price_to_book DOUBLE,
                price_to_sales DOUBLE,
                enterprise_to_revenue DOUBLE,
                enterprise_to_ebitda DOUBLE,
                debt_to_equity DOUBLE,
                return_on_equity DOUBLE,
                return_on_assets DOUBLE,
                revenue_growth DOUBLE,
                earnings_growth DOUBLE,
                gross_margins DOUBLE,
                operating_margins DOUBLE,
                profit_margins DOUBLE,
                current_ratio DOUBLE,
                quick_ratio DOUBLE,
                total_cash BIGINT,
                total_debt BIGINT,
                total_revenue BIGINT,
                ebitda BIGINT,
                free_cashflow BIGINT,
                beta DOUBLE,
                shares_outstanding BIGINT,
                float_shares BIGINT,
                dividend_yield DOUBLE,
                sector VARCHAR,
                industry VARCHAR,
                country VARCHAR,
                currency VARCHAR,
                current_price DOUBLE,
                target_mean_price DOUBLE,
                recommendation_key VARCHAR,
                number_of_analyst_opinions INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, fetch_date)
            )
        """)
        
        # News data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR,
                headline VARCHAR,
                summary TEXT,
                url VARCHAR UNIQUE,
                source VARCHAR,
                published_at TIMESTAMP,
                image_url VARCHAR,
                category VARCHAR,
                sentiment_score DOUBLE,
                data_source VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Universe/watchlist table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS universe (
                symbol VARCHAR PRIMARY KEY,
                name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                market_cap_category VARCHAR,
                is_active BOOLEAN DEFAULT TRUE,
                added_date DATE DEFAULT CURRENT_DATE,
                notes TEXT
            )
        """)
        
        # Data quality metrics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_quality (
                symbol VARCHAR,
                data_type VARCHAR,
                check_date DATE,
                missing_data_pct DOUBLE,
                outliers_count INTEGER,
                quality_score DOUBLE,
                issues TEXT,
                PRIMARY KEY (symbol, data_type, check_date)
            )
        """)
        
        # Create indexes for better performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_date ON ohlc_data(symbol, date)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_fundamental_symbol_date ON fundamental_data(symbol, fetch_date)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_news_symbol_published ON news_data(symbol, published_at)")
        
    def store_ohlc_data(self, data: pd.DataFrame, update_mode: str = "replace") -> bool:
        """Store OHLC data"""
        try:
            if data.empty:
                logger.warning("No OHLC data to store")
                return False
                
            # Ensure required columns exist
            required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Convert date column to proper format
            data = data.copy()
            data['date'] = pd.to_datetime(data['date']).dt.date
            
            if update_mode == "replace":
                # Delete existing data for these symbols and dates
                symbols = data['symbol'].unique().tolist()
                dates = data['date'].unique().tolist()
                
                placeholders_symbols = ','.join(['?' for _ in symbols])
                placeholders_dates = ','.join(['?' for _ in dates])
                
                self.conn.execute(
                    f"DELETE FROM ohlc_data WHERE symbol IN ({placeholders_symbols}) AND date IN ({placeholders_dates})",
                    symbols + [str(d) for d in dates]
                )
            
            # Insert new data
            self.conn.register('temp_ohlc', data)
            self.conn.execute("""
                INSERT INTO ohlc_data 
                SELECT * FROM temp_ohlc
            """)
            
            logger.info(f"Stored {len(data)} OHLC records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store OHLC data: {e}")
            return False
    
    def store_fundamental_data(self, data: pd.DataFrame, update_mode: str = "replace") -> bool:
        """Store fundamental data"""
        try:
            if data.empty:
                logger.warning("No fundamental data to store")
                return False
                
            # Ensure required columns exist
            if 'symbol' not in data.columns or 'fetch_date' not in data.columns:
                logger.error("Missing required columns: symbol, fetch_date")
                return False
            
            data = data.copy()
            data['fetch_date'] = pd.to_datetime(data['fetch_date']).dt.date
            
            if update_mode == "replace":
                # Delete existing data for these symbols and dates
                symbols = data['symbol'].unique().tolist()
                dates = data['fetch_date'].unique().tolist()
                
                placeholders_symbols = ','.join(['?' for _ in symbols])
                placeholders_dates = ','.join(['?' for _ in dates])
                
                self.conn.execute(
                    f"DELETE FROM fundamental_data WHERE symbol IN ({placeholders_symbols}) AND fetch_date IN ({placeholders_dates})",
                    symbols + [str(d) for d in dates]
                )
            
            # Insert new data
            self.conn.register('temp_fundamental', data)
            self.conn.execute("""
                INSERT INTO fundamental_data 
                SELECT * FROM temp_fundamental
            """)
            
            logger.info(f"Stored {len(data)} fundamental records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store fundamental data: {e}")
            return False
    
    def store_news_data(self, data: pd.DataFrame) -> bool:
        """Store news data"""
        try:
            if data.empty:
                logger.warning("No news data to store")
                return False
                
            # Remove duplicates based on URL
            data = data.drop_duplicates(subset=['url'])
            
            # Insert with conflict resolution (ignore duplicates)
            self.conn.register('temp_news', data)
            self.conn.execute("""
                INSERT INTO news_data (symbol, headline, summary, url, source, published_at, 
                                     image_url, category, sentiment_score, data_source)
                SELECT symbol, headline, summary, url, source, published_at, 
                       image_url, category, sentiment_score, data_source
                FROM temp_news
                ON CONFLICT (url) DO NOTHING
            """)
            
            logger.info(f"Stored news data (attempted {len(data)} records)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store news data: {e}")
            return False
    
    def get_ohlc_data(self, symbols: Optional[List[str]] = None, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve OHLC data"""
        try:
            query = "SELECT * FROM ohlc_data WHERE 1=1"
            params = []
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbols)
                
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
                
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
                
            query += " ORDER BY symbol, date DESC"
            
            if limit:
                query += f" LIMIT {limit}"
                
            return self.conn.execute(query, params).df()
            
        except Exception as e:
            logger.error(f"Failed to retrieve OHLC data: {e}")
            return pd.DataFrame()
    
    def get_fundamental_data(self, symbols: Optional[List[str]] = None,
                           latest_only: bool = True) -> pd.DataFrame:
        """Retrieve fundamental data"""
        try:
            if latest_only:
                # Get latest fundamental data for each symbol
                query = """
                    SELECT f1.*
                    FROM fundamental_data f1
                    INNER JOIN (
                        SELECT symbol, MAX(fetch_date) as max_date
                        FROM fundamental_data
                        GROUP BY symbol
                    ) f2 ON f1.symbol = f2.symbol AND f1.fetch_date = f2.max_date
                    WHERE 1=1
                """
            else:
                query = "SELECT * FROM fundamental_data WHERE 1=1"
            
            params = []
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                query += f" AND f1.symbol IN ({placeholders})" if latest_only else f" AND symbol IN ({placeholders})"
                params.extend(symbols)
                
            query += " ORDER BY symbol, fetch_date DESC"
            
            return self.conn.execute(query, params).df()
            
        except Exception as e:
            logger.error(f"Failed to retrieve fundamental data: {e}")
            return pd.DataFrame()
    
    def get_news_data(self, symbols: Optional[List[str]] = None,
                      days_back: int = 7,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve news data"""
        try:
            query = """
                SELECT * FROM news_data 
                WHERE published_at >= CURRENT_DATE - INTERVAL ? DAYS
            """
            params = [days_back]
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbols)
                
            query += " ORDER BY published_at DESC"
            
            if limit:
                query += f" LIMIT {limit}"
                
            return self.conn.execute(query, params).df()
            
        except Exception as e:
            logger.error(f"Failed to retrieve news data: {e}")
            return pd.DataFrame()
    
    def get_universe(self, active_only: bool = True) -> pd.DataFrame:
        """Get trading universe/watchlist"""
        try:
            query = "SELECT * FROM universe"
            if active_only:
                query += " WHERE is_active = TRUE"
            query += " ORDER BY symbol"
            
            return self.conn.execute(query).df()
            
        except Exception as e:
            logger.error(f"Failed to retrieve universe: {e}")
            return pd.DataFrame()
    
    def add_to_universe(self, symbols: List[str], names: Optional[List[str]] = None,
                       sectors: Optional[List[str]] = None) -> bool:
        """Add symbols to trading universe"""
        try:
            data = []
            for i, symbol in enumerate(symbols):
                data.append({
                    'symbol': symbol,
                    'name': names[i] if names and i < len(names) else None,
                    'sector': sectors[i] if sectors and i < len(sectors) else None,
                    'is_active': True
                })
                
            df = pd.DataFrame(data)
            self.conn.register('temp_universe', df)
            self.conn.execute("""
                INSERT INTO universe (symbol, name, sector, is_active)
                SELECT symbol, name, sector, is_active FROM temp_universe
                ON CONFLICT (symbol) DO UPDATE SET
                    name = EXCLUDED.name,
                    sector = EXCLUDED.sector,
                    is_active = EXCLUDED.is_active
            """)
            
            logger.info(f"Added {len(symbols)} symbols to universe")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add symbols to universe: {e}")
            return False
    
    def get_data_quality_report(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get data quality report"""
        try:
            query = "SELECT * FROM data_quality WHERE 1=1"
            params = []
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbols)
                
            query += " ORDER BY symbol, data_type, check_date DESC"
            
            return self.conn.execute(query, params).df()
            
        except Exception as e:
            logger.error(f"Failed to retrieve data quality report: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")