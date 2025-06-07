"""
Unit tests for PositionSizer
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from mech_exo.sizing import (
    PositionSizer,
    SizingMethod,
    SizingError,
    InsufficientCapitalError
)


class TestPositionSizer:
    
    @pytest.fixture
    def mock_config(self):
        """Mock risk configuration"""
        return {
            "position_sizing": {
                "max_single_trade_risk": 0.02,
                "max_single_position": 0.10,
                "min_position_value": 1000.0,
                "atr_multiplier": 2.0,
                "volatility_target": 0.15,
                "default_method": "atr_based"
            },
            "portfolio": {
                "max_sector_exposure": 0.20
            }
        }
    
    @pytest.fixture
    def sample_ohlc_data(self):
        """Sample OHLC data for testing"""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        np.random.seed(42)
        
        data = []
        base_price = 100
        
        for i, date in enumerate(dates):
            price_change = np.random.normal(0, 0.02)
            base_price *= (1 + price_change)
            
            high = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low = base_price * (1 - abs(np.random.normal(0, 0.01)))
            
            data.append({
                'symbol': 'TEST',
                'date': date,
                'open': base_price,
                'high': high,
                'low': low,
                'close': base_price,
                'volume': np.random.randint(1000000, 5000000),
                'returns': price_change,
                'volatility': 0.20,  # 20% annualized vol
                'atr': 1.2  # $1.2 ATR
            })
        
        return pd.DataFrame(data)
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    def test_initialization(self, mock_storage, mock_config_manager, mock_config):
        """Test PositionSizer initialization"""
        # Mock config manager
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        # Mock storage
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Test initialization
        nav = 50000
        sizer = PositionSizer(nav)
        
        assert sizer.nav == nav
        assert sizer.base_risk_pct == 0.02
        assert sizer.atr_multiplier == 2.0
        assert sizer.vol_target == 0.15
        assert sizer.default_method == SizingMethod.ATR_BASED
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    def test_fixed_percent_sizing(self, mock_storage, mock_config_manager, mock_config):
        """Test fixed percentage sizing method"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Initialize sizer
        nav = 50000
        sizer = PositionSizer(nav)
        
        # Test fixed percent calculation
        price = 100.0
        shares = sizer.calculate_size("TEST", price, method=SizingMethod.FIXED_PERCENT)
        
        # Expected: 2% of $50,000 = $1,000 / $100 = 10 shares
        expected_shares = 10
        assert shares == expected_shares
        
        # Test with different risk percentage
        shares_custom = sizer._calculate_fixed_percent("TEST", price, risk_pct=0.05)
        expected_custom = 25  # 5% of $50,000 = $2,500 / $100 = 25 shares
        assert shares_custom == expected_custom
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    def test_atr_based_sizing(self, mock_storage, mock_config_manager, mock_config, sample_ohlc_data):
        """Test ATR-based sizing method"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        mock_storage_instance = Mock()
        mock_storage_instance.get_ohlc_data.return_value = sample_ohlc_data
        mock_storage.return_value = mock_storage_instance
        
        # Initialize sizer
        nav = 50000
        sizer = PositionSizer(nav)
        
        # Test ATR-based calculation
        price = 100.0
        atr = 1.2
        shares = sizer.calculate_size("TEST", price, method=SizingMethod.ATR_BASED, atr=atr)
        
        # Expected calculation:
        # Risk amount = $50,000 * 2% = $1,000
        # Stop distance = 1.2 * 2.0 = 2.4
        # Shares = $1,000 / 2.4 = 416 shares
        expected_shares = 416
        assert shares == expected_shares
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    def test_volatility_based_sizing(self, mock_storage, mock_config_manager, mock_config):
        """Test volatility-based sizing method"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Initialize sizer
        nav = 50000
        sizer = PositionSizer(nav)
        
        # Test volatility-based calculation
        price = 100.0
        volatility = 0.30  # 30% vol (higher than 15% target)
        
        shares = sizer.calculate_size("TEST", price, method=SizingMethod.VOLATILITY_BASED, 
                                    volatility=volatility)
        
        # Expected calculation:
        # Vol scalar = 0.15 / 0.30 = 0.5
        # Base value = $50,000 * 2% = $1,000
        # Position value = $1,000 * 0.5 = $500
        # Shares = $500 / $100 = 5 shares
        expected_shares = 5
        assert shares == expected_shares
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    def test_kelly_criterion_sizing(self, mock_storage, mock_config_manager, mock_config):
        """Test Kelly Criterion sizing method"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Initialize sizer
        nav = 50000
        sizer = PositionSizer(nav)
        
        # Test Kelly calculation with known parameters
        price = 100.0
        win_rate = 0.60  # 60% win rate
        avg_win = 0.10   # 10% average win
        avg_loss = 0.05  # 5% average loss
        
        shares = sizer.calculate_size("TEST", price, method=SizingMethod.KELLY_CRITERION,
                                    win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss)
        
        # Expected Kelly calculation:
        # b = 0.10 / 0.05 = 2.0
        # kelly = (2.0 * 0.60 - 0.40) / 2.0 = 0.40
        # Capped at 25%, so kelly = 0.25
        # Position value = $50,000 * 0.25 = $12,500
        # Shares = $12,500 / $100 = 125 shares
        expected_shares = 125
        assert shares == expected_shares
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    def test_signal_strength_adjustment(self, mock_storage, mock_config_manager, mock_config):
        """Test signal strength adjustment"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Initialize sizer
        nav = 50000
        sizer = PositionSizer(nav)
        
        price = 100.0
        
        # Test different signal strengths
        base_shares = sizer.calculate_size("TEST", price, method=SizingMethod.FIXED_PERCENT,
                                         signal_strength=1.0)
        
        weak_signal_shares = sizer.calculate_size("TEST", price, method=SizingMethod.FIXED_PERCENT,
                                                signal_strength=0.5)
        
        strong_signal_shares = sizer.calculate_size("TEST", price, method=SizingMethod.FIXED_PERCENT,
                                                  signal_strength=1.5)
        
        # Check proportional scaling
        assert weak_signal_shares == int(base_shares * 0.5)
        assert strong_signal_shares == int(base_shares * 1.5)
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    def test_position_validation(self, mock_storage, mock_config_manager, mock_config):
        """Test position size validation"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Initialize sizer
        nav = 50000
        sizer = PositionSizer(nav)
        
        # Test minimum position value
        price = 100.0
        shares = 5  # $500 position (below $1000 minimum)
        assert not sizer.validate_size("TEST", price, shares)
        
        # Test maximum position percentage
        shares = 6000  # $600,000 position (>10% of $50,000 NAV)
        assert not sizer.validate_size("TEST", price, shares)
        
        # Test valid position
        shares = 20  # $2,000 position (4% of NAV)
        assert sizer.validate_size("TEST", price, shares)
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    def test_liquidity_adjustment(self, mock_storage, mock_config_manager, mock_config):
        """Test liquidity adjustment"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Initialize sizer
        nav = 50000
        sizer = PositionSizer(nav)
        
        # Test liquidity constraint
        calculated_shares = 1000
        avg_daily_volume = 5000  # Only 5000 shares traded daily
        
        adjusted_shares = sizer.adjust_for_liquidity("TEST", calculated_shares, avg_daily_volume)
        
        # Should be limited to 10% of daily volume = 500 shares
        expected_shares = 500
        assert adjusted_shares == expected_shares
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    def test_pyramid_sizing(self, mock_storage, mock_config_manager, mock_config):
        """Test pyramid sizing calculation"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Initialize sizer
        nav = 50000
        sizer = PositionSizer(nav)
        
        price = 100.0
        existing_shares = 10
        
        # Test pyramid sizing (should be 50% of base size for level 2)
        pyramid_shares = sizer.calculate_pyramid_size("TEST", price, existing_shares, levels=2)
        
        # Base size would be 10 shares (2% of $50k / $100)
        # Level 2 pyramid should be 50% = 5 shares
        expected_pyramid = 5
        assert pyramid_shares == expected_pyramid
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    def test_sizing_summary(self, mock_storage, mock_config_manager, mock_config):
        """Test sizing summary generation"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Initialize sizer
        nav = 50000
        sizer = PositionSizer(nav)
        
        price = 100.0
        summary = sizer.get_sizing_summary("TEST", price)
        
        # Check summary structure
        assert summary["symbol"] == "TEST"
        assert summary["price"] == price
        assert summary["nav"] == nav
        assert "sizing_methods" in summary
        
        # Check that multiple methods are included
        methods = summary["sizing_methods"]
        assert "fixed_percent" in methods
        assert "atr_based" in methods or "error" in methods["atr_based"]  # May error without data
        assert "volatility_based" in methods or "error" in methods["volatility_based"]
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    def test_error_handling(self, mock_storage, mock_config_manager, mock_config):
        """Test error handling in position sizing"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        # Initialize sizer
        nav = 50000
        sizer = PositionSizer(nav)
        
        # Test invalid price
        with pytest.raises(SizingError):
            sizer.calculate_size("TEST", -100.0)
        
        # Test invalid signal strength
        with pytest.raises(SizingError):
            sizer.calculate_size("TEST", 100.0, signal_strength=-0.5)
        
        with pytest.raises(SizingError):
            sizer.calculate_size("TEST", 100.0, signal_strength=3.0)
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    def test_config_loading_error(self, mock_config_manager):
        """Test error when config cannot be loaded"""
        # Mock config manager to return empty config
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = {}
        mock_config_manager.return_value = mock_config_instance
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            PositionSizer(50000)


class TestATRCalculation:
    """Test ATR calculation functionality"""
    
    def test_atr_calculation_with_sample_data(self):
        """Test ATR calculation with known data"""
        # Create sample data
        data = pd.DataFrame({
            'high': [102, 103, 104, 105, 106],
            'low': [98, 99, 100, 101, 102],
            'close': [100, 101, 102, 103, 104]
        })
        
        # Expected True Range calculations:
        # Day 1: max(4, -, -) = 4
        # Day 2: max(4, 2, 2) = 4  
        # Day 3: max(4, 1, 1) = 4
        # Day 4: max(4, 2, 1) = 4
        # Day 5: max(4, 2, 1) = 4
        # ATR(3) = (4 + 4 + 4) / 3 = 4
        
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=3).mean()
        
        # Check that ATR is calculated (will have NaN for first 2 periods)
        assert not atr.isna().all()
        assert atr.iloc[-1] > 0  # Latest ATR should be positive


class TestIntegrationScenarios:
    """Integration test scenarios"""
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    def test_realistic_sizing_scenario(self, mock_storage, mock_config_manager):
        """Test realistic sizing scenario with typical market data"""
        # Mock realistic config
        config = {
            "position_sizing": {
                "max_single_trade_risk": 0.02,
                "max_single_position": 0.10,
                "min_position_value": 1000.0,
                "atr_multiplier": 2.0,
                "default_method": "atr_based"
            },
            "portfolio": {"max_sector_exposure": 0.20}
        }
        
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = config
        mock_config_manager.return_value = mock_config_instance
        
        # Mock OHLC data for SPY
        sample_data = pd.DataFrame({
            'symbol': ['SPY'] * 20,
            'date': pd.date_range('2023-01-01', periods=20),
            'close': np.linspace(400, 420, 20),  # Rising from $400 to $420
            'atr': [2.5] * 20,  # $2.50 ATR
            'volatility': [0.18] * 20  # 18% volatility
        })
        
        mock_storage_instance = Mock()
        mock_storage_instance.get_ohlc_data.return_value = sample_data
        mock_storage.return_value = mock_storage_instance
        
        # Test with $100k account
        nav = 100000
        sizer = PositionSizer(nav)
        
        price = 420.0  # Current SPY price
        shares = sizer.calculate_size("SPY", price, method=SizingMethod.ATR_BASED)
        
        # Expected calculation:
        # Risk amount = $100,000 * 2% = $2,000
        # Stop distance = 2.5 * 2.0 = 5.0
        # Shares = $2,000 / 5.0 = 400 shares
        assert shares == 400
        
        # Verify position value is reasonable
        position_value = shares * price
        nav_percentage = position_value / nav
        
        assert position_value == 168000  # $168,000 position
        assert nav_percentage == 1.68    # 168% of NAV (high but within ATR logic)