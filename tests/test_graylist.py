"""
Tests for greylist symbol filtering functionality
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from mech_exo.utils.greylist import GreylistManager, get_greylist_manager
from mech_exo.execution.models import Order, OrderType, TimeInForce
from mech_exo.execution.order_router import RoutingDecision


class TestGreylistManager:
    """Test cases for GreylistManager"""
    
    def setup_method(self):
        """Set up test environment"""
        # Create temporary config file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
        self.config_path = Path(self.temp_file.name)
        
        # Initial config with GME and AMC greylisted
        self.test_config = {
            'symbol_filtering': {
                'graylist_symbols': ['GME', 'AMC'],
                'emergency_overrides': {
                    'graylist_override_enabled': True,
                    'override_contact': 'ops@test.com'
                },
                'hot_reload': {
                    'enabled': True,
                    'check_interval_seconds': 10,
                    'last_reload': None
                }
            },
            'change_history': [
                {
                    'action': 'initial_config',
                    'timestamp': '2024-12-06T12:00:00Z',
                    'user': 'test',
                    'changes': 'Initial test configuration'
                }
            ]
        }
        
        # Write config to temp file
        yaml.safe_dump(self.test_config, self.temp_file, default_flow_style=False)
        self.temp_file.close()
        
        # Create manager with temp config
        self.manager = GreylistManager(str(self.config_path))
    
    def teardown_method(self):
        """Clean up test environment"""
        if self.config_path.exists():
            self.config_path.unlink()
    
    def test_get_greylist_basic(self):
        """Test basic greylist retrieval"""
        greylist = self.manager.get_greylist()
        
        assert isinstance(greylist, list)
        assert 'GME' in greylist
        assert 'AMC' in greylist
        assert len(greylist) == 2
    
    def test_is_greylisted_symbols(self):
        """Test greylist symbol checking"""
        # Test greylisted symbols
        assert self.manager.is_greylisted('GME') is True
        assert self.manager.is_greylisted('gme') is True  # Case insensitive
        assert self.manager.is_greylisted('AMC') is True
        
        # Test non-greylisted symbols
        assert self.manager.is_greylisted('AAPL') is False
        assert self.manager.is_greylisted('TSLA') is False
        assert self.manager.is_greylisted('') is False
        assert self.manager.is_greylisted(None) is False
    
    def test_is_override_enabled(self):
        """Test override enabled check"""
        assert self.manager.is_override_enabled() is True
    
    def test_add_symbol(self):
        """Test adding symbol to greylist"""
        # Add new symbol
        success = self.manager.add_symbol('DWAC', 'High volatility')
        assert success is True
        
        # Check it's in greylist
        greylist = self.manager.get_greylist()
        assert 'DWAC' in greylist
        assert len(greylist) == 3
        
        # Adding duplicate should still succeed
        success = self.manager.add_symbol('DWAC', 'Duplicate test')
        assert success is True
        assert len(self.manager.get_greylist()) == 3  # No duplicates
    
    def test_remove_symbol(self):
        """Test removing symbol from greylist"""
        # Remove existing symbol
        success = self.manager.remove_symbol('GME', 'Testing removal')
        assert success is True
        
        # Check it's removed
        greylist = self.manager.get_greylist()
        assert 'GME' not in greylist
        assert 'AMC' in greylist  # Other symbol still there
        assert len(greylist) == 1
        
        # Removing non-existent symbol should still succeed
        success = self.manager.remove_symbol('NONEXISTENT', 'Test')
        assert success is True
    
    def test_get_greylist_stats(self):
        """Test greylist statistics"""
        stats = self.manager.get_greylist_stats()
        
        assert 'total_symbols' in stats
        assert 'symbols' in stats
        assert 'override_enabled' in stats
        assert 'last_reload' in stats
        assert 'config_path' in stats
        
        assert stats['total_symbols'] == 2
        assert stats['symbols'] == ['GME', 'AMC']
        assert stats['override_enabled'] is True


class TestGreylistOrderRouting:
    """Test greylist integration with order routing"""
    
    def setup_method(self):
        """Set up test environment"""
        # Create temporary config with greylist
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
        self.config_path = Path(self.temp_file.name)
        
        test_config = {
            'symbol_filtering': {
                'graylist_symbols': ['GME', 'AMC'],
                'emergency_overrides': {
                    'graylist_override_enabled': True
                }
            }
        }
        
        yaml.safe_dump(test_config, self.temp_file, default_flow_style=False)
        self.temp_file.close()
    
    def teardown_method(self):
        """Clean up test environment"""
        if self.config_path.exists():
            self.config_path.unlink()
    
    @patch('mech_exo.execution.order_router.get_greylist_manager')
    def test_greylist_blocks_order(self, mock_get_manager):
        """Test that greylisted symbols are blocked"""
        from mech_exo.execution.order_router import OrderRouter
        from mech_exo.execution.models import Order, OrderType, TimeInForce
        
        # Mock greylist manager
        mock_manager = Mock()
        mock_manager.is_greylisted.return_value = True
        mock_manager.is_override_enabled.return_value = True
        mock_manager.get_greylist.return_value = ['GME', 'AMC']
        mock_get_manager.return_value = mock_manager
        
        # Create mock router with minimal setup
        router = Mock()
        router.execution_logger = Mock()
        
        # Import and use the actual _check_greylist method
        from mech_exo.execution.order_router import OrderRouter
        check_method = OrderRouter._check_greylist
        
        # Create test order for greylisted symbol
        order = Order(
            symbol='GME',
            quantity=100,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            strategy='test_strategy'
        )
        
        # Test greylist check
        result = check_method(router, order)
        
        assert result.decision == RoutingDecision.REJECT
        assert 'greylist' in result.rejection_reason.lower()
        assert result.original_order == order
    
    @patch('mech_exo.execution.order_router.get_greylist_manager')
    def test_greylist_allows_override(self, mock_get_manager):
        """Test that greylist override works"""
        from mech_exo.execution.order_router import OrderRouter
        
        # Mock greylist manager
        mock_manager = Mock()
        mock_manager.is_greylisted.return_value = True
        mock_manager.is_override_enabled.return_value = True
        mock_get_manager.return_value = mock_manager
        
        # Create mock router
        router = Mock()
        router.execution_logger = Mock()
        
        # Import the actual method
        check_method = OrderRouter._check_greylist
        
        # Create test order with override
        order = Order(
            symbol='GME',
            quantity=100,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            strategy='test_strategy',
            meta={'graylist_override': True}
        )
        
        # Test override
        result = check_method(router, order)
        
        assert result.decision == RoutingDecision.APPROVE
        assert 'override' in result.routing_notes.lower()
        assert result.original_order == order
    
    @patch('mech_exo.execution.order_router.get_greylist_manager')
    def test_non_greylisted_symbol_passes(self, mock_get_manager):
        """Test that non-greylisted symbols pass through"""
        from mech_exo.execution.order_router import OrderRouter
        
        # Mock greylist manager
        mock_manager = Mock()
        mock_manager.is_greylisted.return_value = False
        mock_get_manager.return_value = mock_manager
        
        # Create mock router
        router = Mock()
        router.execution_logger = Mock()
        
        # Import the actual method
        check_method = OrderRouter._check_greylist
        
        # Create test order for allowed symbol
        order = Order(
            symbol='AAPL',
            quantity=100,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            strategy='test_strategy'
        )
        
        # Test allowed symbol
        result = check_method(router, order)
        
        assert result.decision == RoutingDecision.APPROVE
        assert 'not on greylist' in result.routing_notes.lower()
        assert result.original_order == order


class TestGreylistHotReload:
    """Test hot-reload functionality"""
    
    def test_config_reload_on_file_change(self):
        """Test that config reloads when file changes"""
        # Create temp config file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
        config_path = Path(temp_file.name)
        
        try:
            # Initial config
            initial_config = {
                'symbol_filtering': {
                    'graylist_symbols': ['GME'],
                    'hot_reload': {'enabled': True, 'check_interval_seconds': 1}
                }
            }
            yaml.safe_dump(initial_config, temp_file, default_flow_style=False)
            temp_file.close()
            
            # Create manager
            manager = GreylistManager(str(config_path))
            
            # Check initial greylist
            assert manager.get_greylist() == ['GME']
            
            # Update config file
            updated_config = {
                'symbol_filtering': {
                    'graylist_symbols': ['GME', 'AMC', 'DWAC'],
                    'hot_reload': {'enabled': True, 'check_interval_seconds': 1}
                }
            }
            
            import time
            time.sleep(0.1)  # Ensure file mtime changes
            
            with open(config_path, 'w') as f:
                yaml.safe_dump(updated_config, f, default_flow_style=False)
            
            # Force reload check
            updated_greylist = manager.get_greylist()
            
            # Should have new symbols
            assert len(updated_greylist) == 3
            assert 'GME' in updated_greylist
            assert 'AMC' in updated_greylist
            assert 'DWAC' in updated_greylist
            
        finally:
            if config_path.exists():
                config_path.unlink()


def test_global_greylist_functions():
    """Test global convenience functions"""
    from mech_exo.utils.greylist import get_greylist, is_greylisted
    
    # These should work with default config or create default manager
    greylist = get_greylist()
    assert isinstance(greylist, list)
    
    # Test with likely non-greylisted symbol
    result = is_greylisted('AAPL')
    assert isinstance(result, bool)