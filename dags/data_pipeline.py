"""
Prefect flow for nightly data pipeline
"""

from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mech_exo.datasource import (
    OHLCDownloader, 
    FundamentalFetcher, 
    NewsScraper, 
    DataStorage,
    DataValidationError
)


def load_config(config_path: str = "config") -> Dict[str, Any]:
    """Load configuration from YAML files"""
    config = {}
    config_dir = Path(config_path)
    
    # Load API keys (try local first, then template)
    api_keys_file = config_dir / "api_keys_local.yml"
    if not api_keys_file.exists():
        api_keys_file = config_dir / "api_keys.yml"
    
    if api_keys_file.exists():
        with open(api_keys_file, 'r') as f:
            config.update(yaml.safe_load(f))
    
    return config


@task(retries=3, retry_delay_seconds=60)
def get_universe_symbols() -> List[str]:
    """Get list of symbols to process"""
    logger = get_run_logger()
    
    try:
        storage = DataStorage()
        universe_df = storage.get_universe(active_only=True)
        
        if not universe_df.empty:
            symbols = universe_df['symbol'].tolist()
            logger.info(f"Retrieved {len(symbols)} symbols from universe")
            return symbols
        else:
            # Default watchlist if universe is empty
            default_symbols = [
                'SPY', 'QQQ', 'IWM', 'FXI', 'EEM', 'VEA', 'VWO',
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
                'JPM', 'BAC', 'WFC', 'GS', 'MS',
                'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB'
            ]
            
            # Add default symbols to universe
            storage.add_to_universe(default_symbols)
            logger.info(f"Added {len(default_symbols)} default symbols to universe")
            return default_symbols
            
    except Exception as e:
        logger.error(f"Failed to get universe symbols: {e}")
        raise
    finally:
        if 'storage' in locals():
            storage.close()


@task(retries=2, retry_delay_seconds=30)
def fetch_ohlc_data(symbols: List[str], config: Dict[str, Any]) -> pd.DataFrame:
    """Fetch OHLC data for symbols"""
    logger = get_run_logger()
    
    try:
        downloader = OHLCDownloader(config)
        
        # Fetch recent data (last 30 days to ensure we have latest)
        data = downloader.fetch(symbols, period="1mo", interval="1d")
        
        logger.info(f"Fetched OHLC data for {data['symbol'].nunique()} symbols, {len(data)} records")
        return data
        
    except DataValidationError as e:
        logger.error(f"Data validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch OHLC data: {e}")
        raise


@task(retries=2, retry_delay_seconds=30)
def fetch_fundamental_data(symbols: List[str], config: Dict[str, Any]) -> pd.DataFrame:
    """Fetch fundamental data for symbols"""
    logger = get_run_logger()
    
    try:
        fetcher = FundamentalFetcher(config)
        data = fetcher.fetch(symbols)
        
        logger.info(f"Fetched fundamental data for {len(data)} symbols")
        return data
        
    except DataValidationError as e:
        logger.error(f"Fundamental data validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch fundamental data: {e}")
        raise


@task(retries=2, retry_delay_seconds=30)
def fetch_news_data(symbols: List[str], config: Dict[str, Any]) -> pd.DataFrame:
    """Fetch news data for symbols"""
    logger = get_run_logger()
    
    try:
        scraper = NewsScraper(config)
        data = scraper.fetch(symbols, days_back=7)
        
        logger.info(f"Fetched {len(data)} news articles")
        return data
        
    except Exception as e:
        logger.warning(f"Failed to fetch news data: {e}")
        # News is optional, return empty DataFrame
        return pd.DataFrame()


@task
def store_ohlc_data(data: pd.DataFrame) -> bool:
    """Store OHLC data to database"""
    logger = get_run_logger()
    
    try:
        storage = DataStorage()
        success = storage.store_ohlc_data(data, update_mode="replace")
        
        if success:
            logger.info(f"Successfully stored {len(data)} OHLC records")
        else:
            logger.error("Failed to store OHLC data")
            
        return success
        
    except Exception as e:
        logger.error(f"Error storing OHLC data: {e}")
        raise
    finally:
        if 'storage' in locals():
            storage.close()


@task
def store_fundamental_data(data: pd.DataFrame) -> bool:
    """Store fundamental data to database"""
    logger = get_run_logger()
    
    try:
        storage = DataStorage()
        success = storage.store_fundamental_data(data, update_mode="replace")
        
        if success:
            logger.info(f"Successfully stored {len(data)} fundamental records")
        else:
            logger.error("Failed to store fundamental data")
            
        return success
        
    except Exception as e:
        logger.error(f"Error storing fundamental data: {e}")
        raise
    finally:
        if 'storage' in locals():
            storage.close()


@task
def store_news_data(data: pd.DataFrame) -> bool:
    """Store news data to database"""
    logger = get_run_logger()
    
    if data.empty:
        logger.info("No news data to store")
        return True
    
    try:
        storage = DataStorage()
        success = storage.store_news_data(data)
        
        if success:
            logger.info(f"Successfully stored news data")
        else:
            logger.error("Failed to store news data")
            
        return success
        
    except Exception as e:
        logger.error(f"Error storing news data: {e}")
        raise
    finally:
        if 'storage' in locals():
            storage.close()


@task
def run_data_quality_checks(symbols: List[str]) -> Dict[str, Any]:
    """Run data quality checks on stored data"""
    logger = get_run_logger()
    
    try:
        storage = DataStorage()
        
        quality_report = {
            'timestamp': datetime.now(),
            'symbols_checked': len(symbols),
            'issues': []
        }
        
        # Check OHLC data completeness
        ohlc_data = storage.get_ohlc_data(symbols, limit=1000)
        if not ohlc_data.empty:
            # Check for missing data
            expected_records = len(symbols) * 21  # Rough estimate for month
            actual_records = len(ohlc_data)
            completeness = actual_records / expected_records if expected_records > 0 else 0
            
            quality_report['ohlc_completeness'] = completeness
            
            if completeness < 0.8:
                quality_report['issues'].append(f"Low OHLC completeness: {completeness:.2%}")
        
        # Check fundamental data freshness
        fundamental_data = storage.get_fundamental_data(symbols)
        if not fundamental_data.empty:
            latest_date = pd.to_datetime(fundamental_data['fetch_date']).max()
            days_old = (datetime.now().date() - latest_date.date()).days
            
            quality_report['fundamental_freshness_days'] = days_old
            
            if days_old > 7:
                quality_report['issues'].append(f"Fundamental data is {days_old} days old")
        
        # Check news data availability
        news_data = storage.get_news_data(symbols, days_back=7)
        quality_report['news_articles_count'] = len(news_data)
        
        if len(quality_report['issues']) == 0:
            logger.info("Data quality checks passed")
        else:
            logger.warning(f"Data quality issues found: {quality_report['issues']}")
            
        return quality_report
        
    except Exception as e:
        logger.error(f"Data quality check failed: {e}")
        raise
    finally:
        if 'storage' in locals():
            storage.close()


@flow(
    name="Daily Data Pipeline",
    description="Nightly data pipeline for Mech-Exo trading system",
    task_runner=ConcurrentTaskRunner()
)
def daily_data_pipeline(config_path: str = "config") -> Dict[str, Any]:
    """Main data pipeline flow"""
    logger = get_run_logger()
    
    try:
        logger.info("Starting daily data pipeline")
        
        # Load configuration
        config = load_config(config_path)
        
        # Get symbols to process
        symbols = get_universe_symbols()
        
        if not symbols:
            raise ValueError("No symbols to process")
        
        # Fetch data concurrently
        ohlc_future = fetch_ohlc_data.submit(symbols, config)
        fundamental_future = fetch_fundamental_data.submit(symbols, config)
        news_future = fetch_news_data.submit(symbols, config)
        
        # Wait for results
        ohlc_data = ohlc_future.result()
        fundamental_data = fundamental_future.result()
        news_data = news_future.result()
        
        # Store data
        ohlc_stored = store_ohlc_data(ohlc_data)
        fundamental_stored = store_fundamental_data(fundamental_data)
        news_stored = store_news_data(news_data)
        
        # Run quality checks
        quality_report = run_data_quality_checks(symbols)
        
        # Summary
        pipeline_result = {
            'status': 'success',
            'timestamp': datetime.now(),
            'symbols_processed': len(symbols),
            'ohlc_records': len(ohlc_data),
            'fundamental_records': len(fundamental_data),
            'news_articles': len(news_data),
            'ohlc_stored': ohlc_stored,
            'fundamental_stored': fundamental_stored,
            'news_stored': news_stored,
            'quality_report': quality_report
        }
        
        logger.info(f"Data pipeline completed successfully: {pipeline_result}")
        return pipeline_result
        
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        return {
            'status': 'failed',
            'timestamp': datetime.now(),
            'error': str(e)
        }


if __name__ == "__main__":
    # Run the pipeline
    result = daily_data_pipeline()
    print(f"Pipeline result: {result}")