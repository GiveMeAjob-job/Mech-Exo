"""
Pytest configuration and shared fixtures for Mech-Exo tests
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_price_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)
    
    data = {
        "open": 100 + np.random.randn(len(dates)).cumsum() * 0.5,
        "high": None,
        "low": None, 
        "close": None,
        "volume": np.random.randint(1000000, 10000000, len(dates))
    }
    
    # Generate high/low based on open
    data["high"] = data["open"] + np.random.uniform(0, 2, len(dates))
    data["low"] = data["open"] - np.random.uniform(0, 2, len(dates))
    data["close"] = data["open"] + np.random.uniform(-1, 1, len(dates))
    
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "risk_limits": {
            "max_single_trade_risk": 0.02,
            "max_portfolio_drawdown": 0.10,
            "trailing_stop_pct": 0.25
        },
        "position_sizing": {
            "base_risk_pct": 0.01,
            "atr_multiplier": 2.0
        }
    }


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory for testing"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def mock_nav():
    """Mock NAV for position sizing tests"""
    return 100000.0


# Execution engine test fixtures

@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    temp_dir = tempfile.gettempdir()
    temp_db_path = os.path.join(temp_dir, f"test_{os.getpid()}_{datetime.now().strftime('%H%M%S%f')}.db")
    
    yield temp_db_path
    
    # Cleanup
    if os.path.exists(temp_db_path):
        os.unlink(temp_db_path)


@pytest.fixture
def mock_config_manager():
    """Mock ConfigManager for tests"""
    with patch('mech_exo.utils.config.ConfigManager') as mock_cm:
        mock_instance = Mock()
        mock_instance.load_config.return_value = {
            'test': True,
            'database': {'path': ':memory:'}
        }
        mock_cm.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_data_storage():
    """Mock DataStorage for tests"""
    with patch('mech_exo.datasource.storage.DataStorage') as mock_ds:
        mock_instance = Mock()
        # Configure common return values
        mock_instance.get_ohlc_data.return_value = pd.DataFrame()
        mock_instance.get_fundamental_data.return_value = pd.DataFrame()
        mock_instance.get_news_data.return_value = pd.DataFrame()
        mock_ds.return_value = mock_instance
        yield mock_instance


# Pytest markers and configuration
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "execution: mark test as execution engine integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers"""
    for item in items:
        # Add unit marker to tests not in integration folders
        if "integration" not in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        else:
            item.add_marker(pytest.mark.integration)


# Environment setup
@pytest.fixture(autouse=True)
def test_environment():
    """Set up test environment"""
    # Set test mode environment variables
    os.environ['EXO_MODE'] = 'stub'
    os.environ['TESTING'] = 'true'
    
    yield
    
    # Cleanup environment
    os.environ.pop('TESTING', None)