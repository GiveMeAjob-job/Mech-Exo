"""
Retraining Data Loader

Loads historical data for strategy retraining with configurable lookback periods.
Provides clean, validated datasets for factor re-fitting and validation.
"""

import logging
import pandas as pd
import duckdb
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataLoadConfig:
    """Configuration for data loading"""
    lookback_months: int = 6
    min_symbols: int = 20
    min_data_quality: float = 0.8
    include_fundamentals: bool = True
    include_news: bool = True
    universe_filter: Optional[List[str]] = None


@dataclass  
class DataSummary:
    """Summary of loaded data"""
    start_date: date
    end_date: date
    total_records: int
    symbols_count: int
    data_quality_score: float
    fundamental_coverage: float
    news_coverage: float
    missing_data_pct: float
    universe_symbols: List[str]


class RetrainDataLoader:
    """
    Data loader for strategy retraining
    
    Provides clean, validated datasets with configurable lookback periods
    for factor re-fitting and walk-forward validation.
    """
    
    def __init__(self, db_path: str = "data/mech_exo.duckdb"):
        """
        Initialize the data loader
        
        Args:
            db_path: Path to DuckDB database
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        
    def _connect(self):
        """Connect to DuckDB database"""
        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def calculate_date_range(self, lookback_months: int, 
                           end_date: Optional[date] = None) -> Tuple[date, date]:
        """
        Calculate start and end dates for data loading
        
        Args:
            lookback_months: Number of months to look back
            end_date: End date (defaults to today)
            
        Returns:
            Tuple of (start_date, end_date)
        """
        if end_date is None:
            end_date = date.today()
        
        # Calculate start date (approximate months)
        start_date = end_date - timedelta(days=lookback_months * 30)
        
        logger.info(f"Calculated date range: {start_date} to {end_date} ({lookback_months} months)")
        
        return start_date, end_date
    
    def get_trading_universe(self, start_date: date, end_date: date,
                           min_data_coverage: float = 0.8) -> List[str]:
        """
        Get trading universe for the specified period
        
        Args:
            start_date: Start date for universe selection
            end_date: End date for universe selection  
            min_data_coverage: Minimum data coverage required
            
        Returns:
            List of symbols in trading universe
        """
        try:
            # Query to get symbols with sufficient data coverage
            query = """
            SELECT 
                symbol,
                COUNT(*) as data_points,
                COUNT(*) * 1.0 / (DATE_DIFF('day', DATE '{start_date}', DATE '{end_date}') + 1) as coverage
            FROM ohlc_data 
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
              AND volume > 0
              AND high >= low
              AND close > 0
            GROUP BY symbol
            HAVING coverage >= {min_coverage}
            ORDER BY data_points DESC
            """.format(
                start_date=start_date,
                end_date=end_date, 
                min_coverage=min_data_coverage
            )
            
            result = self.conn.execute(query).fetchdf()
            
            if result.empty:
                logger.warning("No symbols found meeting data coverage criteria")
                return []
            
            symbols = result['symbol'].tolist()
            logger.info(f"Found {len(symbols)} symbols with >{min_data_coverage:.0%} data coverage")
            
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get trading universe: {e}")
            return []
    
    def load_ohlc_data(self, start_date: date, end_date: date, 
                      symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load OHLC price data for retraining
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
            symbols: List of symbols to load (None for all)
            
        Returns:
            DataFrame with OHLC data
        """
        try:
            # Build query
            query_parts = [
                "SELECT * FROM ohlc_data",
                f"WHERE date BETWEEN '{start_date}' AND '{end_date}'"
            ]
            
            if symbols:
                symbol_list = "', '".join(symbols)
                query_parts.append(f"AND symbol IN ('{symbol_list}')")
            
            query_parts.append("ORDER BY symbol, date")
            
            query = " ".join(query_parts)
            
            logger.info(f"Loading OHLC data: {start_date} to {end_date}")
            if symbols:
                logger.info(f"Filtering to {len(symbols)} symbols")
            
            df = self.conn.execute(query).fetchdf()
            
            if df.empty:
                logger.warning("No OHLC data found for specified criteria")
                return pd.DataFrame()
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Loaded {len(df)} OHLC records for {df['symbol'].nunique()} symbols")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load OHLC data: {e}")
            return pd.DataFrame()
    
    def load_fundamental_data(self, start_date: date, end_date: date,
                            symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load fundamental data for retraining
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading  
            symbols: List of symbols to load (None for all)
            
        Returns:
            DataFrame with fundamental data
        """
        try:
            # Check if fundamental_data table exists
            tables = self.conn.execute("SHOW TABLES").fetchdf()
            if 'fundamental_data' not in tables['name'].values:
                logger.warning("fundamental_data table not found, skipping")
                return pd.DataFrame()
            
            # Build query
            query_parts = [
                "SELECT * FROM fundamental_data",
                f"WHERE date BETWEEN '{start_date}' AND '{end_date}'"
            ]
            
            if symbols:
                symbol_list = "', '".join(symbols)
                query_parts.append(f"AND symbol IN ('{symbol_list}')")
            
            query_parts.append("ORDER BY symbol, date")
            
            query = " ".join(query_parts)
            
            logger.info(f"Loading fundamental data: {start_date} to {end_date}")
            
            df = self.conn.execute(query).fetchdf()
            
            if df.empty:
                logger.warning("No fundamental data found")
                return pd.DataFrame()
            
            # Convert date column  
            df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Loaded {len(df)} fundamental records for {df['symbol'].nunique()} symbols")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load fundamental data: {e}")
            return pd.DataFrame()
    
    def load_news_data(self, start_date: date, end_date: date,
                      symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load news data for retraining
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
            symbols: List of symbols to load (None for all)
            
        Returns:
            DataFrame with news data
        """
        try:
            # Check if news_data table exists
            tables = self.conn.execute("SHOW TABLES").fetchdf()
            if 'news_data' not in tables['name'].values:
                logger.warning("news_data table not found, skipping")
                return pd.DataFrame()
            
            # Build query
            query_parts = [
                "SELECT * FROM news_data", 
                f"WHERE date BETWEEN '{start_date}' AND '{end_date}'"
            ]
            
            if symbols:
                symbol_list = "', '".join(symbols)
                query_parts.append(f"AND symbol IN ('{symbol_list}')")
            
            query_parts.append("ORDER BY symbol, date")
            
            query = " ".join(query_parts)
            
            logger.info(f"Loading news data: {start_date} to {end_date}")
            
            df = self.conn.execute(query).fetchdf()
            
            if df.empty:
                logger.warning("No news data found")
                return pd.DataFrame()
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Loaded {len(df)} news records for {df['symbol'].nunique()} symbols")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load news data: {e}")
            return pd.DataFrame()
    
    def calculate_data_quality(self, ohlc_df: pd.DataFrame, 
                             fundamental_df: pd.DataFrame,
                             news_df: pd.DataFrame,
                             symbols: List[str],
                             start_date: date, end_date: date) -> Dict[str, float]:
        """
        Calculate data quality metrics
        
        Args:
            ohlc_df: OHLC DataFrame
            fundamental_df: Fundamental DataFrame
            news_df: News DataFrame
            symbols: List of symbols expected
            start_date: Start date of period
            end_date: End date of period
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            total_days = (end_date - start_date).days + 1
            total_expected = len(symbols) * total_days
            
            # Calculate coverage metrics
            ohlc_coverage = len(ohlc_df) / total_expected if total_expected > 0 else 0
            
            fundamental_coverage = 0
            if not fundamental_df.empty and len(symbols) > 0:
                fundamental_expected = len(symbols) * (total_days // 30)  # Monthly fundamentals
                fundamental_coverage = len(fundamental_df) / max(fundamental_expected, 1)
            
            news_coverage = 0  
            if not news_df.empty and len(symbols) > 0:
                news_expected = len(symbols) * total_days * 0.1  # Expect ~10% of days have news
                news_coverage = len(news_df) / max(news_expected, 1)
            
            # Calculate missing data percentage
            missing_data_pct = 0
            if not ohlc_df.empty:
                # Check for missing values in key columns
                key_columns = ['open', 'high', 'low', 'close', 'volume']
                existing_columns = [col for col in key_columns if col in ohlc_df.columns]
                if existing_columns:
                    missing_data_pct = ohlc_df[existing_columns].isnull().sum().sum() / (len(ohlc_df) * len(existing_columns))
            
            # Overall data quality score
            quality_score = (ohlc_coverage * 0.7 + 
                           min(fundamental_coverage, 1.0) * 0.2 + 
                           min(news_coverage, 1.0) * 0.1)
            
            quality_metrics = {
                'overall_quality': min(quality_score, 1.0),
                'ohlc_coverage': min(ohlc_coverage, 1.0),
                'fundamental_coverage': min(fundamental_coverage, 1.0),
                'news_coverage': min(news_coverage, 1.0),
                'missing_data_pct': missing_data_pct
            }
            
            logger.info(f"Data quality metrics: {quality_metrics}")
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate data quality: {e}")
            return {
                'overall_quality': 0.0,
                'ohlc_coverage': 0.0,
                'fundamental_coverage': 0.0,
                'news_coverage': 0.0,
                'missing_data_pct': 1.0
            }
    
    def load_retraining_dataset(self, config: DataLoadConfig) -> Tuple[Dict[str, pd.DataFrame], DataSummary]:
        """
        Load complete dataset for retraining
        
        Args:
            config: Data loading configuration
            
        Returns:
            Tuple of (data_dict, data_summary)
        """
        logger.info(f"Loading retraining dataset with config: {config}")
        
        # Calculate date range
        start_date, end_date = self.calculate_date_range(config.lookback_months)
        
        # Get trading universe
        if config.universe_filter:
            symbols = config.universe_filter
            logger.info(f"Using provided universe filter: {len(symbols)} symbols")
        else:
            symbols = self.get_trading_universe(start_date, end_date)
            
            # Filter to minimum symbols requirement
            if len(symbols) < config.min_symbols:
                logger.warning(f"Only {len(symbols)} symbols found, minimum required: {config.min_symbols}")
                # Relax coverage requirements and try again
                symbols = self.get_trading_universe(start_date, end_date, min_data_coverage=0.5)
        
        if not symbols:
            logger.error("No symbols found for retraining dataset")
            return {}, DataSummary(
                start_date=start_date,
                end_date=end_date,
                total_records=0,
                symbols_count=0,
                data_quality_score=0.0,
                fundamental_coverage=0.0,
                news_coverage=0.0,
                missing_data_pct=1.0,
                universe_symbols=[]
            )
        
        # Load data
        data_dict = {}
        
        # Load OHLC data (required)
        ohlc_df = self.load_ohlc_data(start_date, end_date, symbols)
        data_dict['ohlc'] = ohlc_df
        
        # Load fundamental data (optional)
        fundamental_df = pd.DataFrame()
        if config.include_fundamentals:
            fundamental_df = self.load_fundamental_data(start_date, end_date, symbols)
            data_dict['fundamentals'] = fundamental_df
        
        # Load news data (optional)
        news_df = pd.DataFrame()
        if config.include_news:
            news_df = self.load_news_data(start_date, end_date, symbols)
            data_dict['news'] = news_df
        
        # Calculate quality metrics
        quality_metrics = self.calculate_data_quality(
            ohlc_df, fundamental_df, news_df, symbols, start_date, end_date
        )
        
        # Create summary
        data_summary = DataSummary(
            start_date=start_date,
            end_date=end_date,
            total_records=len(ohlc_df),
            symbols_count=len(symbols),
            data_quality_score=quality_metrics['overall_quality'],
            fundamental_coverage=quality_metrics['fundamental_coverage'],
            news_coverage=quality_metrics['news_coverage'],
            missing_data_pct=quality_metrics['missing_data_pct'],
            universe_symbols=symbols
        )
        
        # Validate quality
        if data_summary.data_quality_score < config.min_data_quality:
            logger.warning(f"Data quality {data_summary.data_quality_score:.2f} below minimum {config.min_data_quality}")
        
        logger.info(f"Retraining dataset loaded successfully: {data_summary}")
        
        return data_dict, data_summary


def load_data_for_retraining(lookback_months: int = 6,
                           min_symbols: int = 20,
                           include_fundamentals: bool = True,
                           include_news: bool = True) -> Tuple[Dict[str, pd.DataFrame], DataSummary]:
    """
    Convenience function to load retraining data
    
    Args:
        lookback_months: Number of months to look back
        min_symbols: Minimum number of symbols required
        include_fundamentals: Whether to include fundamental data
        include_news: Whether to include news data
        
    Returns:
        Tuple of (data_dict, data_summary)
    """
    config = DataLoadConfig(
        lookback_months=lookback_months,
        min_symbols=min_symbols,
        include_fundamentals=include_fundamentals,
        include_news=include_news
    )
    
    with RetrainDataLoader() as loader:
        return loader.load_retraining_dataset(config)


if __name__ == "__main__":
    # Test the data loader
    print("🔄 Testing Retraining Data Loader...")
    
    try:
        # Test with 3 months lookback
        data_dict, summary = load_data_for_retraining(
            lookback_months=3,
            min_symbols=5,  # Lower for testing
            include_fundamentals=True,
            include_news=True
        )
        
        print(f"✅ Data loading test completed:")
        print(f"   • Period: {summary.start_date} to {summary.end_date}")
        print(f"   • Records: {summary.total_records:,}")
        print(f"   • Symbols: {summary.symbols_count}")
        print(f"   • Quality: {summary.data_quality_score:.1%}")
        print(f"   • Datasets: {list(data_dict.keys())}")
        
        if data_dict.get('ohlc') is not None and not data_dict['ohlc'].empty:
            print(f"   • OHLC shape: {data_dict['ohlc'].shape}")
        
        print("🎉 Data loader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()