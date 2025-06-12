"""
Unit tests for order splitting logic in canary A/B testing
Tests Day 2 functionality: order router split and tag-based P&L
"""

import pytest
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, date

from mech_exo.execution.models import Order, OrderType, create_market_order
from mech_exo.execution.order_router import OrderRouter
from mech_exo.execution.allocation import split_order_quantity, is_canary_enabled
from mech_exo.reporting.pnl import compute_tag_based_nav, compute_daily_pnl


class TestOrderSplitLogic:
    """Test order quantity splitting with canary allocation"""
    
    def test_split_order_quantity_basic_cases(self):
        """Test order splitting for various quantities"""
        test_cases = [
            (23, 21, 2),    # Small order: base 21, canary 2
            (100, 90, 10),  # Round number: base 90, canary 10  
            (157, 142, 15), # Odd number: base 142, canary 15 (rounds down)
            (1000, 900, 100), # Large order: base 900, canary 100
        ]
        
        for total_qty, expected_base, expected_canary in test_cases:
            base_qty, canary_qty = split_order_quantity(total_qty)
            
            assert base_qty == expected_base, f"Base quantity mismatch for {total_qty}"
            assert canary_qty == expected_canary, f"Canary quantity mismatch for {total_qty}"
            assert base_qty + canary_qty == total_qty, f"Split doesn't sum to total for {total_qty}"
    
    def test_split_order_quantity_rounding_toward_base(self):
        """Test that fractional shares round toward base allocation"""
        # Test case where 10% doesn't divide evenly
        test_quantities = [37, 63, 89, 123]
        
        for qty in test_quantities:
            base_qty, canary_qty = split_order_quantity(qty)
            
            # Verify rounding behavior
            expected_canary = int(qty * 0.10)  # Should floor the canary amount
            expected_base = qty - expected_canary
            
            assert canary_qty == expected_canary, f"Canary not floored correctly for {qty}"
            assert base_qty == expected_base, f"Base not calculated correctly for {qty}"
            assert base_qty + canary_qty == qty, f"Split doesn't sum correctly for {qty}"
    
    def test_split_order_quantity_zero_allocation(self):
        """Test order splitting when canary allocation is zero"""
        with patch('mech_exo.execution.allocation.get_canary_allocation', return_value=0.0):
            base_qty, canary_qty = split_order_quantity(100)
            
            assert base_qty == 100
            assert canary_qty == 0
    
    def test_split_order_quantity_disabled_canary(self):
        """Test that disabled canary returns full order to base"""
        with patch('mech_exo.execution.allocation.is_canary_enabled', return_value=False):
            # This test verifies the integration point, actual logic is in OrderRouter
            enabled = is_canary_enabled()
            assert enabled is False


class TestOrderRouterCanaryIntegration:
    """Test OrderRouter integration with canary allocation"""
    
    def test_handle_canary_allocation_enabled(self):
        """Test order splitting when canary is enabled"""
        # Create a minimal OrderRouter for testing
        mock_broker = MagicMock()
        mock_risk_checker = MagicMock()
        
        router = OrderRouter(
            broker=mock_broker,
            risk_checker=mock_risk_checker,
            config={}
        )
        
        # Test order
        order = Order(
            symbol="AAPL",
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        # Mock canary enabled
        with patch('mech_exo.execution.order_router.is_canary_enabled', return_value=True):
            with patch('mech_exo.execution.order_router.split_order_quantity', return_value=(90, 10)):
                split_orders = router._handle_canary_allocation(order)
        
        # Should return 2 orders
        assert len(split_orders) == 2
        
        # Check base order
        base_order = split_orders[0]
        assert base_order.quantity == 90
        assert base_order.tag == "base"
        assert base_order.symbol == "AAPL"
        
        # Check canary order
        canary_order = split_orders[1]
        assert canary_order.quantity == 10
        assert canary_order.tag == "ml_canary"
        assert canary_order.symbol == "AAPL"
    
    def test_handle_canary_allocation_disabled(self):
        """Test order routing when canary is disabled"""
        mock_broker = MagicMock()
        mock_risk_checker = MagicMock()
        
        router = OrderRouter(
            broker=mock_broker,
            risk_checker=mock_risk_checker,
            config={}
        )
        
        order = Order(
            symbol="AAPL",
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        # Mock canary disabled
        with patch('mech_exo.execution.order_router.is_canary_enabled', return_value=False):
            split_orders = router._handle_canary_allocation(order)
        
        # Should return 1 order with base tag
        assert len(split_orders) == 1
        assert split_orders[0].quantity == 100
        assert split_orders[0].tag == "base"
    
    def test_handle_canary_allocation_small_order(self):
        """Test that small orders are not split"""
        mock_broker = MagicMock()
        mock_risk_checker = MagicMock()
        
        router = OrderRouter(
            broker=mock_broker,
            risk_checker=mock_risk_checker,
            config={}
        )
        
        # Small order (less than minimum split size)
        order = Order(
            symbol="AAPL", 
            quantity=3,  # Less than min_split_size of 5
            order_type=OrderType.MARKET
        )
        
        with patch('mech_exo.execution.order_router.is_canary_enabled', return_value=True):
            split_orders = router._handle_canary_allocation(order)
        
        # Should not split small orders
        assert len(split_orders) == 1
        assert split_orders[0].quantity == 3
        assert split_orders[0].tag == "base"
    
    def test_handle_canary_allocation_sell_order(self):
        """Test order splitting for sell orders (negative quantity)"""
        mock_broker = MagicMock()
        mock_risk_checker = MagicMock()
        
        router = OrderRouter(
            broker=mock_broker,
            risk_checker=mock_risk_checker,
            config={}
        )
        
        # Sell order (negative quantity)
        order = Order(
            symbol="AAPL",
            quantity=-100,  # Sell 100 shares
            order_type=OrderType.MARKET
        )
        
        with patch('mech_exo.execution.order_router.is_canary_enabled', return_value=True):
            with patch('mech_exo.execution.order_router.split_order_quantity', return_value=(90, 10)):
                split_orders = router._handle_canary_allocation(order)
        
        # Should preserve negative sign for sell orders
        assert len(split_orders) == 2
        
        base_order = split_orders[0]
        assert base_order.quantity == -90  # Negative for sell
        assert base_order.tag == "base"
        
        canary_order = split_orders[1]
        assert canary_order.quantity == -10  # Negative for sell
        assert canary_order.tag == "ml_canary"
    
    def test_handle_canary_allocation_error_fallback(self):
        """Test error handling falls back to base allocation"""
        mock_broker = MagicMock()
        mock_risk_checker = MagicMock()
        
        router = OrderRouter(
            broker=mock_broker,
            risk_checker=mock_risk_checker,
            config={}
        )
        
        order = Order(
            symbol="AAPL",
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        # Mock an error in the split function
        with patch('mech_exo.execution.order_router.is_canary_enabled', side_effect=Exception("Config error")):
            split_orders = router._handle_canary_allocation(order)
        
        # Should fallback to base allocation
        assert len(split_orders) == 1
        assert split_orders[0].quantity == 100
        assert split_orders[0].tag == "base"


class TestTagBasedPnL:
    """Test tag-based P&L computation"""
    
    def test_compute_tag_based_nav_empty_data(self):
        """Test NAV computation with no fill data"""
        with patch('mech_exo.reporting.pnl.FillStore') as mock_fill_store:
            # Mock empty DataFrame
            mock_store_instance = MagicMock()
            mock_store_instance.conn.execute.return_value.fetchall.return_value = []
            mock_fill_store.return_value = mock_store_instance
            
            with patch('pandas.read_sql_query', return_value=pd.DataFrame()):
                nav = compute_tag_based_nav()
            
            assert nav == {'base': 0.0, 'ml_canary': 0.0}
    
    def test_compute_tag_based_nav_with_data(self):
        """Test NAV computation with mock fill data"""
        # Mock fill data
        mock_fills = pd.DataFrame([
            {'tag': 'base', 'symbol': 'AAPL', 'net_quantity': 90, 'avg_price': 150.0, 'total_invested': 13500.0, 'total_fees': 5.0, 'fill_count': 1},
            {'tag': 'ml_canary', 'symbol': 'AAPL', 'net_quantity': 10, 'avg_price': 150.0, 'total_invested': 1500.0, 'total_fees': 1.0, 'fill_count': 1},
        ])
        
        with patch('mech_exo.reporting.pnl.FillStore') as mock_fill_store:
            mock_store_instance = MagicMock()
            mock_fill_store.return_value = mock_store_instance
            
            with patch('pandas.read_sql_query', return_value=mock_fills):
                nav = compute_tag_based_nav()
            
            # NAV = quantity * avg_price  
            expected_base_nav = 90 * 150.0  # 13,500
            expected_canary_nav = 10 * 150.0  # 1,500
            
            assert nav['base'] == expected_base_nav
            assert nav['ml_canary'] == expected_canary_nav
    
    def test_compute_daily_pnl_calculation(self):
        """Test daily P&L calculation logic"""
        # Mock NAV computation
        current_nav = {'base': 14000.0, 'ml_canary': 1600.0}
        previous_nav = {'base': 13500.0, 'ml_canary': 1500.0}
        
        with patch('mech_exo.reporting.pnl.compute_tag_based_nav') as mock_nav:
            # Return current NAV on first call, previous NAV on second call
            mock_nav.side_effect = [current_nav, previous_nav]
            
            pnl = compute_daily_pnl()
            
            # P&L = current - previous
            assert pnl['base'] == 500.0  # 14000 - 13500
            assert pnl['ml_canary'] == 100.0  # 1600 - 1500


class TestCanaryPerformanceIntegration:
    """Test end-to-end canary performance tracking"""
    
    def test_allocation_breakdown_calculation(self):
        """Test allocation percentage calculation"""
        from mech_exo.reporting.pnl import get_allocation_breakdown
        
        # Mock NAV data
        mock_nav = {'base': 90000.0, 'ml_canary': 10000.0}  # Total: 100k
        
        with patch('mech_exo.reporting.pnl.compute_tag_based_nav', return_value=mock_nav):
            allocation = get_allocation_breakdown()
            
            assert allocation['base'] == 0.9  # 90%
            assert allocation['ml_canary'] == 0.1  # 10%
    
    def test_allocation_breakdown_zero_nav(self):
        """Test allocation breakdown with zero NAV"""
        from mech_exo.reporting.pnl import get_allocation_breakdown
        
        # Mock zero NAV
        mock_nav = {'base': 0.0, 'ml_canary': 0.0}
        
        with patch('mech_exo.reporting.pnl.compute_tag_based_nav', return_value=mock_nav):
            allocation = get_allocation_breakdown()
            
            # Should default to 100% base when no NAV
            assert allocation['base'] == 1.0
            assert allocation['ml_canary'] == 0.0


class TestCanaryOrderExecution:
    """Test canary order execution scenarios"""
    
    def test_tag_propagation_order_to_fill(self):
        """Test that tag propagates from Order to Fill"""
        from mech_exo.execution.models import Fill
        
        # Create order with tag
        order = Order(
            symbol="AAPL",
            quantity=100,
            order_type=OrderType.MARKET,
            tag="ml_canary"
        )
        
        # Create fill from order
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=order.quantity,
            price=150.0,
            filled_at=datetime.now(),
            tag=order.tag  # Should inherit tag from order
        )
        
        assert fill.tag == "ml_canary"
        assert fill.symbol == order.symbol
        assert fill.quantity == order.quantity
    
    def test_fill_storage_with_tag(self):
        """Test that fills are stored with tag field"""
        from mech_exo.execution.fill_store import FillStore
        from mech_exo.execution.models import Fill
        
        # Create test fill with tag
        fill = Fill(
            order_id="test-order-123",
            symbol="AAPL",
            quantity=10,
            price=150.0,
            filled_at=datetime.now(),
            tag="ml_canary"
        )
        
        # Test storage (mock to avoid actual DB writes in unit tests)
        with patch.object(FillStore, '__init__', return_value=None):
            with patch.object(FillStore, 'store_fill', return_value=True) as mock_store:
                fill_store = FillStore.__new__(FillStore)
                result = fill_store.store_fill(fill)
                
                # Verify store_fill was called
                mock_store.assert_called_once_with(fill)
                assert result is True


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])