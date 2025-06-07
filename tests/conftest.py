"""
Pytest configuration and shared fixtures for Mech-Exo tests
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


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