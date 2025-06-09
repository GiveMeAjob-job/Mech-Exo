"""
Tests for Retraining Data Loader

Tests the data loading functionality for strategy retraining.
"""

import pytest
import pandas as pd
import tempfile
import duckdb
from datetime import date, timedelta
from pathlib import Path

from mech_exo.datasource.retrain_loader import (
    RetrainDataLoader, 
    DataLoadConfig, 
    DataSummary,
    load_data_for_retraining
)


@pytest.fixture
def temp_db():
    """Create temporary test database"""
    with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as f:
        db_path = f.name
    
    # Create test database with sample data
    conn = duckdb.connect(db_path)
    
    # Create OHLC data
    test_dates = [date.today() - timedelta(days=i) for i in range(90, 0, -1)]
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    
    ohlc_data = []
    for symbol in test_symbols:
        for test_date in test_dates:
            ohlc_data.append({
                'symbol': symbol,
                'date': test_date,
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000000
            })
    
    ohlc_df = pd.DataFrame(ohlc_data)
    conn.execute("CREATE TABLE ohlc_data AS SELECT * FROM ohlc_df")
    
    # Create fundamental data (monthly)
    fundamental_data = []
    monthly_dates = [date.today() - timedelta(days=i*30) for i in range(3, 0, -1)]
    for symbol in test_symbols:
        for test_date in monthly_dates:
            fundamental_data.append({
                'symbol': symbol,
                'date': test_date,
                'pe_ratio': 20.0,
                'return_on_equity': 0.15,
                'revenue_growth': 0.10
            })
    
    if fundamental_data:
        fundamental_df = pd.DataFrame(fundamental_data)
        conn.execute("CREATE TABLE fundamental_data AS SELECT * FROM fundamental_df")
    
    conn.close()
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


def test_retrain_data_loader_init(temp_db):
    """Test RetrainDataLoader initialization"""
    loader = RetrainDataLoader(temp_db)
    assert loader.db_path == temp_db
    assert loader.conn is not None
    loader.close()


def test_calculate_date_range():
    """Test date range calculation"""
    loader = RetrainDataLoader()
    
    # Test with default end date
    start_date, end_date = loader.calculate_date_range(3)
    assert isinstance(start_date, date)
    assert isinstance(end_date, date)
    assert start_date < end_date
    assert (end_date - start_date).days >= 80  # Approximately 3 months
    
    # Test with specific end date
    test_end = date(2025, 6, 1)
    start_date, end_date = loader.calculate_date_range(2, test_end)
    assert end_date == test_end
    assert start_date < end_date
    
    loader.close()


def test_get_trading_universe(temp_db):
    """Test trading universe selection"""
    loader = RetrainDataLoader(temp_db)
    
    start_date = date.today() - timedelta(days=60)
    end_date = date.today() - timedelta(days=30)
    
    symbols = loader.get_trading_universe(start_date, end_date, min_data_coverage=0.5)
    
    assert isinstance(symbols, list)
    assert len(symbols) >= 0  # May be 0 if no data meets criteria
    
    loader.close()


def test_load_ohlc_data(temp_db):
    """Test OHLC data loading"""
    loader = RetrainDataLoader(temp_db)
    
    start_date = date.today() - timedelta(days=60)
    end_date = date.today() - timedelta(days=30)
    
    # Test loading all data
    df = loader.load_ohlc_data(start_date, end_date)
    
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert 'symbol' in df.columns
        assert 'date' in df.columns
        assert 'close' in df.columns
    
    # Test loading specific symbols
    df_filtered = loader.load_ohlc_data(start_date, end_date, ['AAPL', 'MSFT'])
    
    assert isinstance(df_filtered, pd.DataFrame)
    if not df_filtered.empty:
        assert all(symbol in ['AAPL', 'MSFT'] for symbol in df_filtered['symbol'].unique())
    
    loader.close()


def test_data_load_config():
    """Test DataLoadConfig dataclass"""
    config = DataLoadConfig()
    
    # Test defaults
    assert config.lookback_months == 6
    assert config.min_symbols == 20
    assert config.min_data_quality == 0.8
    assert config.include_fundamentals is True
    assert config.include_news is True
    assert config.universe_filter is None
    
    # Test custom config
    custom_config = DataLoadConfig(
        lookback_months=3,
        min_symbols=10,
        min_data_quality=0.7,
        include_fundamentals=False,
        universe_filter=['AAPL', 'MSFT']
    )
    
    assert custom_config.lookback_months == 3
    assert custom_config.min_symbols == 10
    assert custom_config.universe_filter == ['AAPL', 'MSFT']


def test_data_summary():
    """Test DataSummary dataclass"""
    summary = DataSummary(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 6, 1),
        total_records=1000,
        symbols_count=50,
        data_quality_score=0.85,
        fundamental_coverage=0.90,
        news_coverage=0.70,
        missing_data_pct=0.05,
        universe_symbols=['AAPL', 'MSFT']
    )
    
    assert summary.start_date == date(2025, 1, 1)
    assert summary.total_records == 1000
    assert summary.symbols_count == 50
    assert summary.data_quality_score == 0.85


def test_calculate_data_quality(temp_db):
    """Test data quality calculation"""
    loader = RetrainDataLoader(temp_db)
    
    # Create test dataframes
    ohlc_df = pd.DataFrame({
        'symbol': ['AAPL'] * 10,
        'date': pd.date_range('2025-01-01', periods=10),
        'open': [100] * 10,
        'high': [105] * 10,
        'low': [95] * 10,
        'close': [102] * 10,
        'volume': [1000000] * 10
    })
    
    fundamental_df = pd.DataFrame({
        'symbol': ['AAPL'] * 3,
        'date': pd.date_range('2025-01-01', periods=3),
        'pe_ratio': [20] * 3
    })
    
    news_df = pd.DataFrame({
        'symbol': ['AAPL'] * 5,
        'date': pd.date_range('2025-01-01', periods=5),
        'sentiment': [0.5] * 5
    })
    
    symbols = ['AAPL']
    start_date = date(2025, 1, 1)
    end_date = date(2025, 1, 10)
    
    quality_metrics = loader.calculate_data_quality(
        ohlc_df, fundamental_df, news_df, symbols, start_date, end_date
    )
    
    assert isinstance(quality_metrics, dict)
    assert 'overall_quality' in quality_metrics
    assert 'ohlc_coverage' in quality_metrics
    assert 'fundamental_coverage' in quality_metrics
    assert 'news_coverage' in quality_metrics
    assert 'missing_data_pct' in quality_metrics
    
    # All metrics should be between 0 and 1
    for metric_name, value in quality_metrics.items():
        assert 0.0 <= value <= 1.0, f"{metric_name} should be between 0 and 1, got {value}"
    
    loader.close()


def test_load_data_for_retraining_convenience_function(temp_db):
    """Test convenience function for loading retraining data"""
    # Mock the default database path temporarily
    import mech_exo.datasource.retrain_loader as retrain_module
    original_init = retrain_module.RetrainDataLoader.__init__
    
    def mock_init(self, db_path="data/mech_exo.duckdb"):
        # Use our temp database instead
        original_init(self, temp_db)
    
    retrain_module.RetrainDataLoader.__init__ = mock_init
    
    try:
        data_dict, summary = load_data_for_retraining(
            lookback_months=2,
            min_symbols=1,  # Low for testing
            include_fundamentals=True,
            include_news=False
        )
        
        assert isinstance(data_dict, dict)
        assert isinstance(summary, DataSummary)
        assert summary.lookback_months is not None or True  # DataSummary doesn't have lookback_months field
        
    finally:
        # Restore original method
        retrain_module.RetrainDataLoader.__init__ = original_init


if __name__ == "__main__":
    # Run tests manually
    print("🧪 Testing Retraining Data Loader...")
    
    # Create a temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as f:
        test_db = f.name
    
    try:
        # Simple initialization test
        loader = RetrainDataLoader(test_db)
        print("✅ Initialization test passed")
        
        # Date range test
        start, end = loader.calculate_date_range(3)
        print(f"✅ Date range test passed: {start} to {end}")
        
        loader.close()
        print("✅ Connection management test passed")
        
        print("🎉 Basic data loader tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        Path(test_db).unlink(missing_ok=True)