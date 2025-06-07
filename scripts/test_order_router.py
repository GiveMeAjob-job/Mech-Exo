#!/usr/bin/env python3
"""
Test script for OrderRouter functionality
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
from unittest.mock import Mock, patch
from mech_exo.execution.order_router import OrderRouter, RoutingDecision
from mech_exo.execution.broker_adapter import create_broker_adapter
from mech_exo.execution.models import create_market_order, create_limit_order
from mech_exo.risk import RiskChecker, Portfolio, Position
from mech_exo.risk.base import RiskStatus
from datetime import datetime


async def test_order_router_basic():
    """Test basic OrderRouter functionality"""
    print("🔀 Testing OrderRouter Basic Flow...")
    
    try:
        # Setup components
        broker_config = {
            'simulate_fills': True,
            'fill_delay': 0.1,
            'reject_probability': 0.0
        }
        broker = create_broker_adapter('stub', broker_config)
        await broker.connect()
        
        # Create portfolio and risk checker (mocked)
        portfolio = Portfolio(100000)
        
        with patch('mech_exo.risk.checker.ConfigManager'), \
             patch('mech_exo.risk.checker.DataStorage'):
            
            risk_checker = RiskChecker(portfolio)
            
            # Mock risk check to always approve
            risk_checker.check_new_position = Mock(return_value={
                'pre_trade_analysis': {
                    'recommendation': 'APPROVE',
                    'violations': [],
                    'warnings': []
                }
            })
            
            # Create router
            router_config = {
                'max_retries': 2,
                'retry_delay': 0.1,
                'max_daily_orders': 10,
                'max_order_value': 50000
            }
            router = OrderRouter(broker, risk_checker, router_config)
            
            # Setup callbacks
            routing_results = []
            def on_routing_result(result):
                routing_results.append(result)
                print(f"  📋 Routing: {result.decision.value} - {result.routing_notes or result.rejection_reason}")
            
            order_updates = []
            def on_order_update(order):
                order_updates.append(order)
                print(f"  📝 Order: {order.symbol} - {order.status.value}")
            
            fill_updates = []
            def on_fill_update(fill):
                fill_updates.append(fill)
                print(f"  💰 Fill: {fill.symbol} {fill.quantity} @ ${fill.price}")
            
            router.add_routing_callback(on_routing_result)
            router.add_order_callback(on_order_update)
            router.add_fill_callback(on_fill_update)
            
            # Test 1: Route valid order
            order1 = create_market_order("AAPL", 100, strategy="test")
            result1 = await router.route_order(order1)
            
            assert result1.decision == RoutingDecision.APPROVE, "Should approve valid order"
            print(f"  ✅ Order approved: {order1.symbol}")
            
            # Wait for fill
            await asyncio.sleep(0.2)
            
            # Test 2: Route order that exceeds value limit
            order2 = create_limit_order("EXPENSIVE", 1000, 100.0)  # $100k order
            result2 = await router.route_order(order2)
            
            assert result2.decision == RoutingDecision.REJECT, "Should reject expensive order"
            assert "exceeds limit" in result2.rejection_reason, "Should mention value limit"
            print(f"  ✅ Expensive order rejected: {result2.rejection_reason}")
            
            # Test 3: Check routing stats
            stats = router.get_routing_stats()
            print(f"  📊 Stats: {stats['daily_order_count']} orders today, {stats['pending_orders']} pending")
            
            # Test 4: Health check
            health = await router.health_check()
            assert health['status'] == 'healthy', "Router should be healthy"
            print(f"  🏥 Health: {health['status']}")
            
            print("  ✅ OrderRouter basic test passed!")
            
            # Cleanup
            await broker.disconnect()
            risk_checker.close()
            return True
            
    except Exception as e:
        print(f"  ❌ OrderRouter basic test failed: {e}")
        return False


async def test_risk_rejection():
    """Test order rejection due to risk violations"""
    print("\n🚫 Testing Risk Rejection...")
    
    try:
        # Setup components
        broker = create_broker_adapter('stub', {'simulate_fills': False})
        await broker.connect()
        
        portfolio = Portfolio(100000)
        
        with patch('mech_exo.risk.checker.ConfigManager'), \
             patch('mech_exo.risk.checker.DataStorage'):
            
            risk_checker = RiskChecker(portfolio)
            
            # Mock risk check to reject
            risk_checker.check_new_position = Mock(return_value={
                'pre_trade_analysis': {
                    'recommendation': 'REJECT',
                    'violations': ['Position size too large', 'Exceeds sector limit'],
                    'warnings': []
                }
            })
            
            router = OrderRouter(broker, risk_checker)
            
            # Route order that will be rejected
            order = create_market_order("RISKY", 1000, strategy="aggressive")
            result = await router.route_order(order)
            
            assert result.decision == RoutingDecision.REJECT, "Should reject risky order"
            assert "Risk violation" in result.rejection_reason, "Should mention risk violation"
            assert len(result.risk_warnings) > 0, "Should have risk warnings"
            
            print(f"  ✅ Risk rejection test passed: {result.rejection_reason}")
            
            await broker.disconnect()
            risk_checker.close()
            return True
            
    except Exception as e:
        print(f"  ❌ Risk rejection test failed: {e}")
        return False


async def test_position_modification():
    """Test order modification due to risk suggestions"""
    print("\n✏️  Testing Position Modification...")
    
    try:
        # Setup components
        broker = create_broker_adapter('stub', {'simulate_fills': False})
        await broker.connect()
        
        portfolio = Portfolio(100000)
        
        with patch('mech_exo.risk.checker.ConfigManager'), \
             patch('mech_exo.risk.checker.DataStorage'):
            
            risk_checker = RiskChecker(portfolio)
            
            # Mock risk check to suggest smaller size
            risk_checker.check_new_position = Mock(return_value={
                'pre_trade_analysis': {
                    'recommendation': 'APPROVE',
                    'violations': [],
                    'warnings': ['Position size large'],
                    'suggested_size': 50  # Suggest smaller size
                }
            })
            
            router = OrderRouter(broker, risk_checker)
            
            # Route order that will be modified
            order = create_market_order("MODIFY", 100, strategy="test")  # Want 100 shares
            result = await router.route_order(order)
            
            assert result.decision == RoutingDecision.MODIFY, "Should modify order"
            assert result.modified_order is not None, "Should have modified order"
            assert result.modified_order.quantity == 50, "Should reduce to 50 shares"
            
            print(f"  ✅ Position modification test passed: {order.quantity} -> {result.modified_order.quantity}")
            
            await broker.disconnect()
            risk_checker.close()
            return True
            
    except Exception as e:
        print(f"  ❌ Position modification test failed: {e}")
        return False


async def test_retry_logic():
    """Test retry logic for failed orders"""
    print("\n🔄 Testing Retry Logic...")
    
    try:
        # Setup broker that fails first time
        broker_config = {
            'simulate_fills': False,
            'reject_probability': 0.8  # High rejection rate to test retries
        }
        broker = create_broker_adapter('stub', broker_config)
        await broker.connect()
        
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
            
            # Configure aggressive retries
            router_config = {
                'max_retries': 5,
                'retry_delay': 0.05,  # Fast retries for testing
                'retry_on_rejection': True  # Retry even rejections for this test
            }
            router = OrderRouter(broker, risk_checker, router_config)
            
            order = create_market_order("RETRY", 100, strategy="persistent")
            result = await router.route_order(order)
            
            # With high rejection rate, this might still fail, but we should see retry attempts
            stats = router.get_routing_stats()
            print(f"  📊 Final result: {result.decision.value}")
            print(f"  🔄 Retry attempts tracked: {stats['retry_attempts']}")
            
            print("  ✅ Retry logic test completed")
            
            await broker.disconnect()
            risk_checker.close()
            return True
            
    except Exception as e:
        print(f"  ❌ Retry logic test failed: {e}")
        return False


async def test_safety_limits():
    """Test safety limit enforcement"""
    print("\n🛡️  Testing Safety Limits...")
    
    try:
        broker = create_broker_adapter('stub', {'simulate_fills': False})
        await broker.connect()
        
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
            
            # Set very low daily limit
            router_config = {
                'max_daily_orders': 2,
                'max_order_value': 1000  # $1k limit
            }
            router = OrderRouter(broker, risk_checker, router_config)
            
            # Test order value limit
            expensive_order = create_limit_order("EXPENSIVE", 20, 100.0)  # $2k order
            result1 = await router.route_order(expensive_order)
            
            assert result1.decision == RoutingDecision.REJECT, "Should reject expensive order"
            assert "exceeds limit" in result1.rejection_reason, "Should mention value limit"
            print(f"  ✅ Value limit enforced: {result1.rejection_reason}")
            
            # Test daily order limit
            for i in range(3):  # Try 3 orders with limit of 2
                order = create_market_order(f"TEST{i}", 5, strategy="limit_test")
                result = await router.route_order(order)
                
                if i < 2:
                    # First 2 should pass
                    assert result.decision == RoutingDecision.APPROVE, f"Order {i} should be approved"
                else:
                    # 3rd should be rejected
                    assert result.decision == RoutingDecision.REJECT, "Should reject after daily limit"
                    assert "Daily order limit" in result.rejection_reason, "Should mention daily limit"
            
            print("  ✅ Daily limit enforced")
            
            await broker.disconnect()
            risk_checker.close()
            return True
            
    except Exception as e:
        print(f"  ❌ Safety limits test failed: {e}")
        return False


async def main():
    """Run all OrderRouter tests"""
    print("🚀 Testing OrderRouter\n")
    
    tests = [
        test_order_router_basic,
        test_risk_rejection,
        test_position_modification,
        test_retry_logic,
        test_safety_limits
    ]
    
    results = []
    for test in tests:
        results.append(await test())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All OrderRouter tests PASSED!")
        return True
    else:
        print("❌ Some OrderRouter tests FAILED!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)