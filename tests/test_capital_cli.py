"""
Tests for Capital Management CLI

Validates capital whitelist functionality, account management,
and configuration persistence.
"""

import pytest
import tempfile
import os
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mech_exo.cli.capital import CapitalManager


class TestCapitalManager:
    """Test capital management functionality"""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            config_path = f.name
        
        yield config_path
        
        # Cleanup
        if os.path.exists(config_path):
            os.unlink(config_path)
    
    @pytest.fixture
    def capital_manager(self, temp_config):
        """Create capital manager with temporary config"""
        return CapitalManager(temp_config)
    
    def test_default_config_creation(self, temp_config):
        """Test that default configuration is created"""
        # Remove the file to test creation
        if os.path.exists(temp_config):
            os.unlink(temp_config)
        
        # Create manager - should create default config
        manager = CapitalManager(temp_config)
        
        # Verify file exists
        assert os.path.exists(temp_config)
        
        # Verify structure
        with open(temp_config, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'capital_limits' in config
        assert 'accounts' in config['capital_limits']
        assert 'global' in config['capital_limits']
        assert 'utilization' in config
        assert 'history' in config
    
    def test_add_account_success(self, capital_manager):
        """Test successful account addition"""
        # Add test account
        success = capital_manager.add_account("DU12345678", 100000, "USD", "Test account")
        
        assert success is True
        
        # Verify account was added
        accounts = capital_manager.list_accounts()
        assert len(accounts) == 1
        
        account = accounts[0]
        assert account['account_id'] == "DU12345678"
        assert account['max_capital'] == 100000
        assert account['currency'] == "USD"
        assert account['enabled'] is True
        assert account['notes'] == "Test account"
    
    def test_add_account_invalid_id(self, capital_manager):
        """Test account addition with invalid ID"""
        with pytest.raises(ValueError, match="Invalid account ID format"):
            capital_manager.add_account("INVALID", 100000)
    
    def test_add_account_exceeds_global_limit(self, capital_manager):
        """Test account addition that exceeds global limit"""
        # Add account that would exceed total limit
        with pytest.raises(ValueError, match="exceed global limit"):
            capital_manager.add_account("DU12345678", 600000)  # Exceeds 500k default
    
    def test_update_existing_account(self, capital_manager):
        """Test updating existing account"""
        # Add initial account
        capital_manager.add_account("DU12345678", 100000, "USD", "Initial")
        
        # Update account
        success = capital_manager.add_account("DU12345678", 150000, "USD", "Updated")
        assert success is True
        
        # Verify update
        accounts = capital_manager.list_accounts()
        assert len(accounts) == 1
        assert accounts[0]['max_capital'] == 150000
        assert accounts[0]['notes'] == "Updated"
    
    def test_remove_account_success(self, capital_manager):
        """Test successful account removal"""
        # Add account first
        capital_manager.add_account("DU12345678", 100000)
        
        # Remove account
        success = capital_manager.remove_account("DU12345678", force=True)
        assert success is True
        
        # Verify removal
        accounts = capital_manager.list_accounts()
        assert len(accounts) == 0
    
    def test_remove_nonexistent_account(self, capital_manager):
        """Test removing nonexistent account"""
        with pytest.raises(ValueError, match="not found in whitelist"):
            capital_manager.remove_account("DU99999999", force=True)
    
    def test_disable_enable_account(self, capital_manager):
        """Test disabling and enabling account"""
        # Add account
        capital_manager.add_account("DU12345678", 100000)
        
        # Disable account
        success = capital_manager.disable_account("DU12345678")
        assert success is True
        
        # Verify disabled
        accounts = capital_manager.list_accounts()
        assert accounts[0]['enabled'] is False
        
        # Enable account
        success = capital_manager.enable_account("DU12345678")
        assert success is True
        
        # Verify enabled
        accounts = capital_manager.list_accounts()
        assert accounts[0]['enabled'] is True
    
    def test_total_limits_calculation(self, capital_manager):
        """Test total limits calculation"""
        # Add multiple accounts
        capital_manager.add_account("DU12345678", 100000)
        capital_manager.add_account("DU87654321", 150000)
        capital_manager.add_account("DU11111111", 50000)
        
        # Disable one account
        capital_manager.disable_account("DU11111111")
        
        # Check totals
        totals = capital_manager.get_total_limits()
        
        assert totals['total_accounts'] == 3
        assert totals['enabled_accounts'] == 2
        assert totals['total_allocated'] == 250000  # Only enabled accounts
        assert totals['total_max_capital'] == 500000
        assert totals['remaining_capacity'] == 250000
        assert totals['utilization_pct'] == 50.0
    
    def test_change_history_logging(self, capital_manager):
        """Test that changes are logged to history"""
        # Perform various operations
        capital_manager.add_account("DU12345678", 100000)
        capital_manager.add_account("DU12345678", 150000)  # Update
        capital_manager.disable_account("DU12345678")
        capital_manager.enable_account("DU12345678")
        
        # Check history
        history = capital_manager.get_change_history()
        
        assert len(history) == 4
        assert history[0]['action'] == "add_account"
        assert history[1]['action'] == "update_account"
        assert history[2]['action'] == "disable_account"
        assert history[3]['action'] == "enable_account"
        
        # Verify data integrity
        for change in history:
            assert 'timestamp' in change
            assert 'account' in change
            assert change['account'] == "DU12345678"
    
    def test_account_id_validation(self):
        """Test account ID validation logic"""
        from mech_exo.cli.capital import CapitalManager
        
        # Valid IDs
        assert CapitalManager._validate_account_id("DU12345678") is True
        assert CapitalManager._validate_account_id("DF123456789") is True
        assert CapitalManager._validate_account_id("UA12345678") is True
        
        # Invalid IDs
        assert CapitalManager._validate_account_id("") is False
        assert CapitalManager._validate_account_id("123456789") is False  # No letters
        assert CapitalManager._validate_account_id("DU123") is False  # Too short
        assert CapitalManager._validate_account_id("DU1234567890") is False  # Too long
        assert CapitalManager._validate_account_id("D1U2345678") is False  # Numbers in prefix
        assert CapitalManager._validate_account_id("DU12345ABC") is False  # Letters in suffix
    
    def test_configuration_persistence(self, capital_manager, temp_config):
        """Test that configuration changes persist"""
        # Add account
        capital_manager.add_account("DU12345678", 100000, "USD", "Persistent test")
        
        # Create new manager instance with same config file
        new_manager = CapitalManager(temp_config)
        
        # Verify data persisted
        accounts = new_manager.list_accounts()
        assert len(accounts) == 1
        assert accounts[0]['account_id'] == "DU12345678"
        assert accounts[0]['max_capital'] == 100000
        assert accounts[0]['notes'] == "Persistent test"
    
    def test_multiple_accounts_within_limit(self, capital_manager):
        """Test adding multiple accounts within global limit"""
        # Add multiple accounts that together don't exceed limit
        capital_manager.add_account("DU12345678", 150000)
        capital_manager.add_account("DU87654321", 200000)
        capital_manager.add_account("DU11111111", 100000)
        
        # Should total 450k, within 500k limit
        totals = capital_manager.get_total_limits()
        assert totals['total_allocated'] == 450000
        assert totals['remaining_capacity'] == 50000
        
        # Try to add one more that would exceed
        with pytest.raises(ValueError, match="exceed global limit"):
            capital_manager.add_account("DU99999999", 100000)  # Would make 550k
    
    def test_empty_account_list(self, capital_manager):
        """Test behavior with no accounts configured"""
        accounts = capital_manager.list_accounts()
        assert accounts == []
        
        totals = capital_manager.get_total_limits()
        assert totals['total_accounts'] == 0
        assert totals['enabled_accounts'] == 0
        assert totals['total_allocated'] == 0
        assert totals['utilization_pct'] == 0
    
    def test_config_file_error_handling(self):
        """Test error handling for corrupted config files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("invalid: yaml: content: [")  # Malformed YAML
            config_path = f.name
        
        try:
            # Should raise exception for corrupted file
            with pytest.raises(Exception):
                CapitalManager(config_path)
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])