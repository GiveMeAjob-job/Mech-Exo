#!/usr/bin/env python3
"""
Integration test script for execution engine
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch

from mech_exo.execution.models import create_market_order, create_limit_order, OrderStatus
from mech_exo.execution.order_router import OrderRouter
from mech_exo.execution.fill_store import FillStore
from mech_exo.risk import RiskChecker, Portfolio
from tests.stubs.broker_stub import EnhancedStubBroker


async def test_stub_broker_basic():
    """Test basic StubBroker functionality"""
    print("🧪 Testing StubBroker Basic Functionality...")
    
    try:
        config = {
            'simulate_fills': True,
            'fill_delay_ms': 50,
            'reject_probability': 0.0,
            'initial_nav': 100000.0
        }
        
        broker = EnhancedStubBroker(config)
        
        # Test connection
        connected = await broker.connect()
        assert connected, "Should connect successfully"
        assert broker.is_connected(), "Should report connected"
        print("  ✅ Connection successful")
        
        # Test account info
        account_info = await broker.get_account_info()
        assert account_info['account_id'].startswith('STUB_ACCOUNT_'), "Should have stub account ID"
        assert account_info['netliquidation'] == 100000.0, "Should have correct NAV"
        print(f"  ✅ Account: {account_info['account_id']} NAV: ${account_info['netliquidation']:,.0f}")
        
        # Test order placement
        order = create_market_order("AAPL", 100, strategy="test")
        
        result = await broker.place_order(order)
        assert result['status'] == 'SUBMITTED', "Should submit order successfully"
        assert order.broker_order_id is not None, "Should assign broker order ID"
        print(f"  ✅ Order submitted: {order.symbol} {order.quantity} -> {order.broker_order_id}")
        
        # Wait for fill
        await asyncio.sleep(0.1)
        
        # Check order status
        status = await broker.get_order_status(order.order_id)
        assert status['status'] == 'FILLED', "Order should be filled"
        print(f"  ✅ Order status: {status['status']}")
        
        # Check positions
        positions = await broker.get_positions()
        assert len(positions) == 1, "Should have one position"
        assert positions[0]['symbol'] == 'AAPL', "Should be AAPL position"
        assert positions[0]['position'] == 100, "Should have 100 shares"
        print(f"  ✅ Position: {positions[0]['symbol']} {positions[0]['position']} shares")
        
        await broker.disconnect()
        print("  ✅ StubBroker basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ StubBroker basic test failed: {e}")
        return False


async def test_order_router_integration():
    """Test OrderRouter with StubBroker integration"""
    print("\n🔄 Testing OrderRouter Integration...")
    
    try:
        broker = EnhancedStubBroker({'simulate_fills': True, 'fill_delay_ms': 50})
        await broker.connect()
        
        # Create mock risk checker
        portfolio = Portfolio(100000)
        
        with patch('mech_exo.risk.checker.ConfigManager'), \
             patch('mech_exo.risk.checker.DataStorage'):
            
            risk_checker = RiskChecker(portfolio)
            risk_checker.check_new_position = Mock(return_value={
                'pre_trade_analysis': {
                    'recommendation': 'APPROVE',
                    'violations': [],
                    'warnings': []
                }
            })
            
            # Create OrderRouter
            router = OrderRouter(broker, risk_checker, {'max_retries': 2})
            
            # Track callbacks
            order_updates = []
            fill_updates = []
            
            def on_order_update(order):
                order_updates.append(order)
                print(f"    📝 Order update: {order.symbol} - {order.status.value}")
            
            def on_fill_update(fill):
                fill_updates.append(fill)
                print(f"    💰 Fill: {fill.symbol} {fill.quantity} @ ${fill.price}")
            
            router.add_order_callback(on_order_update)
            router.add_fill_callback(on_fill_update)
            
            # Route order
            order = create_market_order("GOOGL", 50, strategy="momentum")
            routing_result = await router.route_order(order)
            
            assert routing_result.decision.value == 'APPROVE', "Should approve order"
            print(f"  ✅ Routing decision: {routing_result.decision.value}")
            
            # Wait for fill
            await asyncio.sleep(0.1)
            
            assert len(order_updates) >= 2, "Should have order updates (submitted + filled)"
            assert len(fill_updates) == 1, "Should have fill update"
            
            # Check final order status
            final_order = order_updates[-1]
            assert final_order.status == OrderStatus.FILLED, "Order should be filled"
            
            # Check fill
            fill = fill_updates[0]
            assert fill.symbol == "GOOGL", "Fill should be for GOOGL"
            assert fill.quantity == 50, "Fill should be for 50 shares"
            
            await broker.disconnect()
            risk_checker.close()
            
        print("  ✅ OrderRouter integration test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ OrderRouter integration test failed: {e}")
        return False


async def test_fill_store_integration():
    """Test FillStore integration"""
    print("\n💾 Testing FillStore Integration...")
    
    try:
        # Create temporary FillStore
        temp_dir = tempfile.gettempdir()
        temp_db = os.path.join(temp_dir, f"test_integration_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
        
        fill_store = FillStore(temp_db)
        
        # Configure broker with fill store
        broker_config = {
            'simulate_fills': True,
            'fill_delay_ms': 50,
            'write_to_fill_store': True,
            'fill_store_path': temp_db
        }
        
        broker = EnhancedStubBroker(broker_config)
        await broker.connect()
        
        # Place multiple orders
        orders = [
            create_market_order("AAPL", 100, strategy="test1"),
            create_market_order("GOOGL", 50, strategy="test2"),
            create_limit_order("MSFT", 75, 340.0, strategy="test3")
        ]
        
        for order in orders:
            result = await broker.place_order(order)
            assert result['status'] == 'SUBMITTED', f"Should submit {order.symbol} order"
            print(f"  📤 Submitted: {order.symbol} {order.quantity}")
        
        # Wait for fills
        await asyncio.sleep(0.2)
        
        # Check FillStore has recorded fills
        fills = fill_store.get_fills()
        assert len(fills) >= 3, f"Should have at least 3 fills in store, got {len(fills)}"
        print(f"  ✅ FillStore recorded {len(fills)} fills")
        
        # Check fill details
        symbols_filled = [fill.symbol for fill in fills]
        assert 'AAPL' in symbols_filled, "Should have AAPL fill"
        assert 'GOOGL' in symbols_filled, "Should have GOOGL fill"
        assert 'MSFT' in symbols_filled, "Should have MSFT fill"
        print(f"  ✅ Symbols filled: {symbols_filled}")
        
        # Test daily metrics
        today = datetime.now().date()
        daily_metrics = fill_store.get_daily_metrics(today)
        
        assert daily_metrics['fills']['total_fills'] >= 3, "Should count today's fills"
        print(f"  ✅ Daily metrics: {daily_metrics['fills']['total_fills']} fills, {daily_metrics['fills']['symbols_traded']} symbols")
        
        # Cleanup
        await broker.disconnect()
        fill_store.close()
        if os.path.exists(temp_db):
            os.unlink(temp_db)
        
        print("  ✅ FillStore integration test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ FillStore integration test failed: {e}")
        return False


async def test_rejection_scenarios():
    """Test order rejection scenarios"""
    print("\n🚫 Testing Rejection Scenarios...")
    
    try:
        broker = EnhancedStubBroker({'simulate_fills': True})
        await broker.connect()
        
        # Add specific rejection
        broker.add_rejection("REJECT_ME", "Symbol not allowed for testing")
        
        # Create mock risk checker
        portfolio = Portfolio(100000)
        with patch('mech_exo.risk.checker.ConfigManager'), \
             patch('mech_exo.risk.checker.DataStorage'):
            
            risk_checker = RiskChecker(portfolio)
            risk_checker.check_new_position = Mock(return_value={
                'pre_trade_analysis': {
                    'recommendation': 'APPROVE',
                    'violations': [],
                    'warnings': []
                }
            })
            
            router = OrderRouter(broker, risk_checker)
            
            # Try to place rejected order
            rejected_order = create_market_order("REJECT_ME", 100, strategy="test")
            routing_result = await router.route_order(rejected_order)
            
            assert routing_result.decision.value == 'REJECT', "Should reject the order"
            assert "Symbol not allowed" in routing_result.rejection_reason, "Should have rejection reason"
            print(f"  ✅ Rejected order: {routing_result.rejection_reason}")
            
            # Test with normal order
            broker.clear_rejections()
            normal_order = create_market_order("AAPL", 100, strategy="test")
            routing_result = await router.route_order(normal_order)
            
            assert routing_result.decision.value == 'APPROVE', "Should approve normal order"
            print(f"  ✅ Approved normal order: {normal_order.symbol}")
            
            await broker.disconnect()
            risk_checker.close()
            
        print("  ✅ Rejection scenarios test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Rejection scenarios test failed: {e}")
        return False


async def test_end_to_end_session():
    """Test complete end-to-end trading session"""
    print("\n🎯 Testing End-to-End Trading Session...")
    
    try:
        # Create temporary fill store
        temp_dir = tempfile.gettempdir()
        temp_db = os.path.join(temp_dir, f"test_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
        
        broker_config = {
            'simulate_fills': True,
            'fill_delay_ms': 30,
            'write_to_fill_store': True,
            'fill_store_path': temp_db,
            'simulate_slippage': True,
            'price_movement': True
        }
        
        broker = EnhancedStubBroker(broker_config)
        await broker.connect()
        
        # Create portfolio and risk checker
        portfolio = Portfolio(100000)
        with patch('mech_exo.risk.checker.ConfigManager'), \
             patch('mech_exo.risk.checker.DataStorage'):
            
            risk_checker = RiskChecker(portfolio)
            risk_checker.check_new_position = Mock(return_value={
                'pre_trade_analysis': {
                    'recommendation': 'APPROVE',
                    'violations': [],
                    'warnings': []
                }
            })
            
            router = OrderRouter(broker, risk_checker)
            
            # Simulate trading session
            session_orders = [
                create_market_order("SPY", 100, strategy="momentum"),
                create_limit_order("QQQ", 50, 380.0, strategy="mean_revert"),
                create_market_order("IWM", 75, strategy="breakout"),
                create_market_order("SPY", -50, strategy="profit_taking"),  # Partial close
                create_market_order("AAPL", 100, strategy="new_position")
            ]
            
            print(f"  📊 Processing {len(session_orders)} orders...")
            
            for i, order in enumerate(session_orders):
                print(f"    {i+1}. {order.symbol} {order.quantity} ({order.strategy})")
                result = await router.route_order(order)
                assert result.decision.value == 'APPROVE', f"Should approve order {i+1}"
                await asyncio.sleep(0.02)  # Small delay between orders
            
            # Wait for all fills
            await asyncio.sleep(0.2)
            
            # Check final state
            final_positions = await broker.get_positions()
            account_info = await broker.get_account_info()
            trading_summary = broker.get_trading_summary()
            
            # Verify positions make sense
            spy_position = next((p for p in final_positions if p['symbol'] == 'SPY'), None)
            assert spy_position is not None, "Should have SPY position"
            assert spy_position['position'] == 50, f"SPY position should be 50, got {spy_position['position']}"
            
            # Check fill store
            fill_store = FillStore(temp_db)
            session_fills = fill_store.get_fills()
            assert len(session_fills) == 5, f"Should have 5 fills recorded, got {len(session_fills)}"
            
            print(f"  📈 Session Summary:")
            print(f"    Orders: {trading_summary['total_orders']}")
            print(f"    Fills: {trading_summary['filled_orders']}")
            print(f"    Positions: {len(final_positions)}")
            print(f"    Account NAV: ${account_info['netliquidation']:,.2f}")
            
            # Cleanup
            await broker.disconnect()
            risk_checker.close()
            fill_store.close()
            if os.path.exists(temp_db):
                os.unlink(temp_db)
            
        print("  ✅ End-to-end trading session test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ End-to-end trading session test failed: {e}")
        return False


async def main():
    """Run all integration tests"""
    print("🚀 Running Execution Engine Integration Tests\n")
    
    tests = [
        test_stub_broker_basic,
        test_order_router_integration,
        test_fill_store_integration,
        test_rejection_scenarios,
        test_end_to_end_session
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Integration Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All execution integration tests PASSED!")
        print("\n✅ Verified Components:")
        print("  - EnhancedStubBroker with realistic simulation")
        print("  - OrderRouter with pre-trade risk checks")
        print("  - FillStore persistence and daily metrics")
        print("  - Order rejection handling")
        print("  - End-to-end trading session flow")
        print("\n🚀 Ready for CI and production testing!")
        return True
    else:
        print("❌ Some execution integration tests FAILED!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)