"""
Unit tests for StopEngine
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from mech_exo.risk.stop_engine import StopEngine
from mech_exo.risk.base import StopCalculationError


class TestStopEngine:
    
    @pytest.fixture
    def mock_config(self):
        """Mock risk configuration"""
        return {
            "stops": {
                "trailing_stop_pct": 0.25,
                "hard_stop_pct": 0.15,
                "profit_target_pct": 0.30,
                "time_stop_days": 60,
                "vol_stop_multiplier": 2.0
            }
        }
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_initialization(self, mock_config_manager, mock_config):
        """Test StopEngine initialization"""
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        engine = StopEngine()
        
        assert engine.trailing_stop_pct == 0.25
        assert engine.hard_stop_pct == 0.15
        assert engine.profit_target_pct == 0.30
        assert engine.time_stop_days == 60
        assert engine.vol_stop_multiplier == 2.0
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_generate_stops_long_position(self, mock_config_manager, mock_config):
        """Test stop generation for long position"""
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        engine = StopEngine()
        
        entry_price = 100.0
        entry_date = datetime(2023, 1, 1)
        atr = 2.0
        
        stops = engine.generate_stops(
            entry_price=entry_price,
            position_type="long",
            entry_date=entry_date,
            atr=atr
        )
        
        # Check hard stop (15% below entry)
        expected_hard_stop = 100.0 * (1 - 0.15)  # $85.00
        assert stops["hard_stop"] == expected_hard_stop
        
        # Check profit target (30% above entry)
        expected_profit_target = 100.0 * (1 + 0.30)  # $130.00
        assert stops["profit_target"] == expected_profit_target
        
        # Check trailing stop (initially same as hard stop)
        assert stops["trailing_stop"] == expected_hard_stop
        
        # Check time stop
        expected_time_stop = entry_date + timedelta(days=60)
        assert stops["time_stop_date"] == expected_time_stop
        
        # Check volatility stop (2 * ATR below entry)
        expected_vol_stop = 100.0 - (2.0 * 2.0)  # $96.00
        assert stops["volatility_stop"] == expected_vol_stop
        
        # Check risk/reward ratio
        risk = 100.0 - 85.0  # $15
        reward = 130.0 - 100.0  # $30
        expected_rr = reward / risk  # 2.0
        assert stops["risk_reward_ratio"] == expected_rr
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_generate_stops_short_position(self, mock_config_manager, mock_config):
        """Test stop generation for short position"""
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        engine = StopEngine()
        
        entry_price = 100.0
        atr = 2.0
        
        stops = engine.generate_stops(
            entry_price=entry_price,
            position_type="short",
            atr=atr
        )
        
        # Check hard stop (15% above entry for shorts)
        expected_hard_stop = 100.0 * (1 + 0.15)  # $115.00
        assert stops["hard_stop"] == expected_hard_stop
        
        # Check profit target (30% below entry for shorts)
        expected_profit_target = 100.0 * (1 - 0.30)  # $70.00
        assert stops["profit_target"] == expected_profit_target
        
        # Check volatility stop (2 * ATR above entry for shorts)
        expected_vol_stop = 100.0 + (2.0 * 2.0)  # $104.00
        assert stops["volatility_stop"] == expected_vol_stop
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_update_trailing_stop_long(self, mock_config_manager, mock_config):
        """Test trailing stop updates for long positions"""
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        engine = StopEngine()
        
        entry_price = 100.0
        current_stop = 85.0  # 15% stop
        
        # Test 1: Price moves up - stop should trail up
        new_price = 110.0
        high_water_mark = 110.0
        updated_stop = engine.update_trailing_stop(
            current_price=new_price,
            current_stop=current_stop,
            position_type="long",
            high_water_mark=high_water_mark
        )
        
        # New stop should be 25% below high water mark
        expected_stop = 110.0 * (1 - 0.25)  # $82.50
        assert updated_stop == expected_stop
        
        # Test 2: Price moves down - stop should NOT move down
        new_price = 95.0
        updated_stop = engine.update_trailing_stop(
            current_price=new_price,
            current_stop=expected_stop,  # Use previous stop
            position_type="long"
        )
        
        # Stop should remain at previous level
        assert updated_stop == expected_stop
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_update_trailing_stop_short(self, mock_config_manager, mock_config):
        """Test trailing stop updates for short positions"""
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        engine = StopEngine()
        
        current_stop = 115.0  # 15% stop above entry at 100
        
        # Test 1: Price moves down - stop should trail down for shorts
        new_price = 90.0
        low_water_mark = 90.0
        updated_stop = engine.update_trailing_stop(
            current_price=new_price,
            current_stop=current_stop,
            position_type="short",
            high_water_mark=low_water_mark  # Reusing parameter as low water mark
        )
        
        # New stop should be 25% above low water mark
        expected_stop = 90.0 * (1 + 0.25)  # $112.50
        assert updated_stop == expected_stop
        
        # Test 2: Price moves up - stop should NOT move up
        new_price = 105.0
        updated_stop = engine.update_trailing_stop(
            current_price=new_price,
            current_stop=expected_stop,
            position_type="short"
        )
        
        # Stop should remain at previous level
        assert updated_stop == expected_stop
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_check_stop_hit_long_position(self, mock_config_manager, mock_config):
        """Test stop hit detection for long positions"""
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        engine = StopEngine()
        
        stops = {
            "hard_stop": 85.0,
            "trailing_stop": 87.0,
            "profit_target": 130.0,
            "volatility_stop": 96.0,
            "time_stop_date": datetime(2023, 3, 1)
        }
        
        # Test 1: Hard stop hit
        result = engine.check_stop_hit(
            current_price=84.0,
            stops=stops,
            position_type="long"
        )
        
        assert result["status"] == "triggered"
        assert "hard_stop" in result["triggered_stops"]
        assert "Hard stop hit" in result["exit_reason"]
        
        # Test 2: Profit target hit
        result = engine.check_stop_hit(
            current_price=131.0,
            stops=stops,
            position_type="long"
        )
        
        assert result["status"] == "triggered"
        assert "profit_target" in result["triggered_stops"]
        assert "Profit target hit" in result["exit_reason"]
        
        # Test 3: Time stop hit
        result = engine.check_stop_hit(
            current_price=100.0,
            stops=stops,
            position_type="long",
            current_date=datetime(2023, 3, 2)  # After time stop date
        )
        
        assert result["status"] == "triggered"
        assert "time_stop" in result["triggered_stops"]
        assert "Time stop hit" in result["exit_reason"]
        
        # Test 4: No stops hit
        result = engine.check_stop_hit(
            current_price=100.0,
            stops=stops,
            position_type="long",
            current_date=datetime(2023, 2, 1)  # Before time stop
        )
        
        assert result["status"] == "active"
        assert len(result["triggered_stops"]) == 0
        assert result["exit_reason"] is None
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_check_stop_hit_short_position(self, mock_config_manager, mock_config):
        """Test stop hit detection for short positions"""
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        engine = StopEngine()
        
        stops = {
            "hard_stop": 115.0,
            "trailing_stop": 113.0,
            "profit_target": 70.0,
            "volatility_stop": 104.0
        }
        
        # Test 1: Hard stop hit (price goes above stop)
        result = engine.check_stop_hit(
            current_price=116.0,
            stops=stops,
            position_type="short"
        )
        
        assert result["status"] == "triggered"
        assert "hard_stop" in result["triggered_stops"]
        
        # Test 2: Profit target hit (price goes below target)
        result = engine.check_stop_hit(
            current_price=69.0,
            stops=stops,
            position_type="short"
        )
        
        assert result["status"] == "triggered"
        assert "profit_target" in result["triggered_stops"]
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_calculate_stop_distances(self, mock_config_manager, mock_config):
        """Test stop distance calculations"""
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        engine = StopEngine()
        
        entry_price = 100.0
        stops = {
            "hard_stop": 85.0,
            "profit_target": 130.0,
            "time_stop_date": datetime(2023, 3, 1)  # Should be skipped
        }
        
        distances = engine.calculate_stop_distances(entry_price, stops, "long")
        
        # Check hard stop distance
        assert distances["hard_stop"]["price"] == 85.0
        assert distances["hard_stop"]["distance"] == 15.0
        assert distances["hard_stop"]["percentage"] == 0.15
        assert distances["hard_stop"]["direction"] == "down"
        
        # Check profit target distance
        assert distances["profit_target"]["price"] == 130.0
        assert distances["profit_target"]["distance"] == 30.0
        assert distances["profit_target"]["percentage"] == 0.30
        assert distances["profit_target"]["direction"] == "up"
        
        # Time stop should be skipped
        assert "time_stop_date" not in distances
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_optimize_stops_for_symbol(self, mock_config_manager, mock_config):
        """Test stop optimization based on symbol characteristics"""
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        engine = StopEngine()
        
        entry_price = 100.0
        
        # Test with high volatility stock
        optimized_stops = engine.optimize_stops_for_symbol(
            symbol="VOLATILE",
            entry_price=entry_price,
            position_type="long",
            volatility=0.40,  # 40% volatility (high)
            beta=1.8  # High beta
        )
        
        # Stops should be wider than standard for high vol/beta stocks
        standard_stops = engine.generate_stops(entry_price, "long")
        
        # Hard stop should be further from entry for high vol stocks
        assert optimized_stops["hard_stop"] < standard_stops["hard_stop"]  # Lower stop for long
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_get_stop_summary(self, mock_config_manager, mock_config):
        """Test comprehensive stop summary generation"""
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        engine = StopEngine()
        
        entry_price = 100.0
        entry_date = datetime(2023, 1, 1)
        atr = 2.0
        
        summary = engine.get_stop_summary(
            entry_price=entry_price,
            position_type="long",
            entry_date=entry_date,
            atr=atr
        )
        
        # Check summary structure
        assert summary["entry_price"] == entry_price
        assert summary["position_type"] == "long"
        assert "stops" in summary
        assert "distances" in summary
        assert "configuration" in summary
        
        # Check configuration is included
        config = summary["configuration"]
        assert config["trailing_stop_pct"] == 0.25
        assert config["hard_stop_pct"] == 0.15
        assert config["profit_target_pct"] == 0.30
        assert config["time_stop_days"] == 60
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_error_handling(self, mock_config_manager, mock_config):
        """Test error handling in stop engine"""
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        engine = StopEngine()
        
        # Test invalid entry price
        with pytest.raises(StopCalculationError):
            engine.generate_stops(entry_price=-100.0, position_type="long")
        
        # Test invalid position type
        with pytest.raises(StopCalculationError):
            engine.generate_stops(entry_price=100.0, position_type="invalid")
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_config_loading_error(self, mock_config_manager):
        """Test error when config cannot be loaded"""
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = {}
        mock_config_manager.return_value = mock_config_instance
        
        with pytest.raises(ValueError):
            StopEngine()


class TestSpecificCalculations:
    """Test specific calculation scenarios"""
    
    def test_trailing_stop_calculation_precision(self):
        """Test precise trailing stop calculations"""
        # Test the specific example from requirements: 29.5 with 25% trailing = 22.1
        
        mock_config = {
            "stops": {
                "trailing_stop_pct": 0.25,
                "hard_stop_pct": 0.15,
                "profit_target_pct": 0.30,
                "time_stop_days": 60
            }
        }
        
        with patch('mech_exo.risk.stop_engine.ConfigManager') as mock_config_manager:
            mock_config_instance = Mock()
            mock_config_instance.load_config.return_value = mock_config
            mock_config_manager.return_value = mock_config_instance
            
            engine = StopEngine()
            
            # Test the exact scenario from requirements
            current_price = 29.5
            trailing_stop = engine._calculate_hard_stop(current_price, "long")
            
            # With 25% trailing stop: 29.5 * (1 - 0.25) = 22.125
            # But we're using hard_stop_pct (15%) in this calculation
            expected = 29.5 * (1 - 0.15)  # 25.075
            assert abs(trailing_stop - expected) < 0.01
            
            # Test actual trailing stop with 25%
            entry_price = 29.5
            current_stop = 22.125  # 25% below entry
            high_water_mark = 29.5
            
            updated_stop = engine.update_trailing_stop(
                current_price=29.5,
                current_stop=current_stop,
                position_type="long",
                high_water_mark=high_water_mark
            )
            
            # Should calculate 29.5 * (1 - 0.25) = 22.125
            expected_trailing = 29.5 * 0.75
            assert abs(updated_stop - expected_trailing) < 0.01
            
            # Round to match expected output
            assert round(updated_stop, 1) == 22.1


class TestRiskRewardCalculations:
    """Test risk/reward ratio calculations"""
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_risk_reward_scenarios(self, mock_config_manager):
        """Test various risk/reward scenarios"""
        mock_config = {
            "stops": {
                "trailing_stop_pct": 0.25,
                "hard_stop_pct": 0.10,    # 10% stop
                "profit_target_pct": 0.20, # 20% target
                "time_stop_days": 60
            }
        }
        
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_config_instance
        
        engine = StopEngine()
        
        entry_price = 100.0
        stops = engine.generate_stops(entry_price, "long")
        
        # Risk = 100 - 90 = 10
        # Reward = 120 - 100 = 20
        # R/R = 20/10 = 2.0
        assert stops["risk_reward_ratio"] == 2.0
        
        # Test edge case: zero risk (shouldn't happen in practice)
        rr = engine._calculate_risk_reward_ratio(100.0, 100.0, 120.0)
        assert rr == float('inf')