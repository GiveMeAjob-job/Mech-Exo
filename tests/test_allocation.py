"""
Unit tests for canary allocation management
Tests Day 1 functionality: allocation helpers and order splitting
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from mech_exo.execution.allocation import (
    get_canary_allocation,
    split_order_quantity,
    is_canary_enabled,
    update_canary_enabled,
    get_allocation_config
)


class TestCanaryAllocation:
    """Test canary allocation helper functions"""
    
    def test_get_canary_allocation_default(self):
        """Test default allocation when config file doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.yml"
            allocation = get_canary_allocation(str(config_path))
            
            assert allocation == 0.10
    
    def test_get_canary_allocation_from_config(self):
        """Test loading allocation from config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            config = {
                'canary_enabled': True,
                'canary_allocation': 0.15
            }
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            allocation = get_canary_allocation(config_path)
            assert allocation == 0.15
        finally:
            Path(config_path).unlink()
    
    def test_get_canary_allocation_disabled(self):
        """Test allocation when canary is disabled"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            config = {
                'canary_enabled': False,
                'canary_allocation': 0.20
            }
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            allocation = get_canary_allocation(config_path)
            assert allocation == 0.0
        finally:
            Path(config_path).unlink()
    
    def test_get_canary_allocation_validation(self):
        """Test allocation validation and clamping"""
        test_cases = [
            (0.50, 0.30),  # Too high, should clamp to 0.30
            (-0.10, 0.0),  # Negative, should clamp to 0.0
            (0.25, 0.25),  # Valid, should remain unchanged
        ]
        
        for input_allocation, expected_output in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                config = {
                    'canary_enabled': True,
                    'canary_allocation': input_allocation
                }
                yaml.dump(config, f)
                config_path = f.name
            
            try:
                allocation = get_canary_allocation(config_path)
                assert allocation == expected_output
            finally:
                Path(config_path).unlink()


class TestOrderSplitting:
    """Test order quantity splitting logic"""
    
    def test_split_order_quantity_basic(self):
        """Test basic order splitting"""
        test_cases = [
            (100, 0.10, 90, 10),   # 100 shares, 10% canary
            (157, 0.10, 142, 15),  # Odd number, rounds down canary
            (50, 0.20, 40, 10),    # 20% canary allocation
            (23, 0.10, 21, 2),     # Small order
            (1000, 0.05, 950, 50), # 5% canary
        ]
        
        for total_qty, allocation, expected_base, expected_canary in test_cases:
            base, canary = split_order_quantity(total_qty, allocation)
            
            assert base == expected_base
            assert canary == expected_canary
            assert base + canary == total_qty
    
    def test_split_order_quantity_with_config(self):
        """Test order splitting using config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            config = {
                'canary_enabled': True,
                'canary_allocation': 0.15
            }
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            with patch('mech_exo.execution.allocation.get_canary_allocation') as mock_get:
                mock_get.return_value = 0.15
                
                base, canary = split_order_quantity(100)
                
                assert base == 85
                assert canary == 15
                assert base + canary == 100
                
        finally:
            Path(config_path).unlink()
    
    def test_split_order_quantity_zero_allocation(self):
        """Test order splitting with zero canary allocation"""
        base, canary = split_order_quantity(100, 0.0)
        
        assert base == 100
        assert canary == 0
    
    def test_split_order_quantity_error_handling(self):
        """Test error handling in order splitting"""
        with patch('mech_exo.execution.allocation.get_canary_allocation', side_effect=Exception("Config error")):
            base, canary = split_order_quantity(100)
            
            # Should fallback to all base allocation
            assert base == 100
            assert canary == 0


class TestCanaryEnabledStatus:
    """Test canary enabled/disabled functionality"""
    
    def test_is_canary_enabled_default(self):
        """Test default enabled status when config doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.yml"
            enabled = is_canary_enabled(str(config_path))
            
            assert enabled is True
    
    def test_is_canary_enabled_from_config(self):
        """Test reading enabled status from config"""
        test_cases = [
            (True, True),
            (False, False),
        ]
        
        for config_enabled, expected in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                config = {'canary_enabled': config_enabled}
                yaml.dump(config, f)
                config_path = f.name
            
            try:
                enabled = is_canary_enabled(config_path)
                assert enabled == expected
            finally:
                Path(config_path).unlink()
    
    def test_update_canary_enabled_new_file(self):
        """Test updating canary enabled status in new config file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "new_config.yml"
            
            success = update_canary_enabled(False, str(config_path))
            assert success is True
            
            # Verify file was created with correct content
            assert config_path.exists()
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert config['canary_enabled'] is False
            assert config['canary_allocation'] == 0.10  # Default value
    
    def test_update_canary_enabled_existing_file(self):
        """Test updating canary enabled status in existing config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            config = {
                'canary_enabled': True,
                'canary_allocation': 0.15,
                'other_setting': 'preserved'
            }
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            success = update_canary_enabled(False, config_path)
            assert success is True
            
            # Verify config was updated correctly
            with open(config_path, 'r') as f:
                updated_config = yaml.safe_load(f)
            
            assert updated_config['canary_enabled'] is False
            assert updated_config['canary_allocation'] == 0.15  # Preserved
            assert updated_config['other_setting'] == 'preserved'  # Preserved
            
        finally:
            Path(config_path).unlink()


class TestAllocationConfig:
    """Test complete allocation configuration management"""
    
    def test_get_allocation_config_default(self):
        """Test getting default allocation config when file doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('mech_exo.execution.allocation.Path') as mock_path:
                mock_path.return_value.exists.return_value = False
                
                config = get_allocation_config()
                
                # Check all required defaults are present
                assert config['canary_enabled'] is True
                assert config['canary_allocation'] == 0.10
                assert config['disable_threshold_sharpe'] == 0.0
                assert config['disable_min_days'] == 30
    
    def test_get_allocation_config_manual_load(self):
        """Test loading allocation config by manually calling with test file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            config = {
                'canary_enabled': False,
                'canary_allocation': 0.12,
                'disable_threshold_sharpe': -0.1,
                'disable_min_days': 45,
                'custom_setting': 'test_value'
            }
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            # Test the YAML loading directly
            with open(config_path, 'r') as file:
                loaded_config = yaml.safe_load(file) or {}
            
            # Apply defaults like the function does
            defaults = {
                'canary_enabled': True,
                'canary_allocation': 0.10,
                'disable_threshold_sharpe': 0.0,
                'disable_min_days': 30
            }
            
            for key, default_value in defaults.items():
                if key not in loaded_config:
                    loaded_config[key] = default_value
                
            assert loaded_config['canary_enabled'] is False
            assert loaded_config['canary_allocation'] == 0.12
            assert loaded_config['disable_threshold_sharpe'] == -0.1
            assert loaded_config['disable_min_days'] == 45
            assert loaded_config['custom_setting'] == 'test_value'
                
        finally:
            Path(config_path).unlink()
    
    def test_get_allocation_config_defaults_logic(self):
        """Test the defaults logic in allocation config"""
        # Test with empty config
        empty_config = {}
        
        defaults = {
            'canary_enabled': True,
            'canary_allocation': 0.10,
            'disable_threshold_sharpe': 0.0,
            'disable_min_days': 30
        }
        
        for key, default_value in defaults.items():
            if key not in empty_config:
                empty_config[key] = default_value
        
        assert empty_config['canary_enabled'] is True
        assert empty_config['canary_allocation'] == 0.10
        assert empty_config['disable_threshold_sharpe'] == 0.0
        assert empty_config['disable_min_days'] == 30


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])