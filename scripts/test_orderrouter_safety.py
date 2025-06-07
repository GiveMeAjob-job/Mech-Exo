#!/usr/bin/env python3
"""
Test OrderRouter with integrated SafetyValve
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
import os
from datetime import datetime
from unittest.mock import Mock, patch

from mech_exo.execution.order_router import OrderRouter
from mech_exo.execution.models import create_market_order, create_limit_order
from mech_exo.risk import RiskChecker, Portfolio
from tests.stubs.broker_stub import EnhancedStubBroker


async def test_orderrouter_safety_integration():
    """Test OrderRouter with SafetyValve integration"""
    print("🔗 Testing OrderRouter + SafetyValve Integration...")
    
    try:
        # Set non-live mode first
        os.environ['EXO_MODE'] = 'stub'
        
        broker = EnhancedStubBroker({'simulate_fills': True, 'fill_delay_ms': 10})
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
            
            # Configure OrderRouter with safety settings
            router_config = {
                'max_retries': 2,
                'safety': {
                    'safety_mode': 'disabled',  # Start with disabled for testing
                    'max_daily_value': 50000.0
                }
            }
            
            router = OrderRouter(broker, risk_checker, router_config)
            
            # Test normal order in stub mode
            order = create_market_order("AAPL", 100, strategy="safety_test")
            result = await router.route_order(order)
            
            assert result.decision.value == 'APPROVE', "Should approve order in stub mode"
            print("  ✅ Order approved in stub mode")
            
            # Test safety status
            safety_status = router.get_safety_status()
            assert safety_status['mode'] == 'disabled', "Should report disabled mode"
            assert not safety_status['emergency_abort'], "Should not be in emergency abort"
            print("  ✅ Safety status reporting works")
            
            # Test health check
            health = await router.health_check()
            assert health['status'] == 'healthy', "Should be healthy"
            assert 'safety_valve' in health, "Should include safety valve in health check"
            print("  ✅ Health check includes safety valve")
            
            await broker.disconnect()
            risk_checker.close()
            
        print("  ✅ OrderRouter + SafetyValve integration test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ OrderRouter + SafetyValve integration test failed: {e}")
        return False


async def test_live_mode_authorization():
    """Test live mode authorization flow"""
    print("\n🔐 Testing Live Mode Authorization...")
    
    try:
        # Set live mode
        os.environ['EXO_MODE'] = 'live'
        
        broker = EnhancedStubBroker({'simulate_fills': True, 'fill_delay_ms': 10})
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
            
            # Configure with confirmation only (no sentinel for testing)
            router_config = {
                'safety': {
                    'safety_mode': 'confirmation_only',
                    'max_daily_value': 50000.0,
                    'confirmation_valid_minutes': 60
                }
            }
            
            router = OrderRouter(broker, risk_checker, router_config)
            
            # Mock user input to approve
            with patch('builtins.input', return_value='yes'):
                order = create_market_order("AAPL", 100, strategy="live_test")
                result = await router.route_order(order)
                
                assert result.decision.value == 'APPROVE', "Should approve after authorization"
                print("  ✅ Live mode authorization works")
            
            # Check that authorization is cached
            routing_stats = router.get_routing_stats()
            assert routing_stats['live_trading_authorized'], "Should be authorized"
            print("  ✅ Authorization caching works")
            
            await broker.disconnect()
            risk_checker.close()
            
        print("  ✅ Live mode authorization test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Live mode authorization test failed: {e}")
        return False


async def test_emergency_abort_integration():
    """Test emergency abort through OrderRouter"""
    print("\n🚨 Testing Emergency Abort Integration...")
    
    try:
        # Set live mode
        os.environ['EXO_MODE'] = 'live'
        
        broker = EnhancedStubBroker({'simulate_fills': True})
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
            
            router_config = {
                'safety': {
                    'safety_mode': 'disabled',  # Skip confirmation for testing
                    'max_daily_value': 50000.0
                }
            }
            
            router = OrderRouter(broker, risk_checker, router_config)
            
            # First authorize trading
            with patch('builtins.input', return_value='yes'):
                # This should authorize without confirmation since mode is disabled
                order1 = create_market_order("AAPL", 100, strategy="pre_abort")
                result1 = await router.route_order(order1)
                assert result1.decision.value == 'APPROVE', "Should approve before abort"
                print("  ✅ Order approved before emergency abort")
            
            # Activate emergency abort
            router.activate_emergency_abort("Testing emergency abort integration")
            print("  ✅ Emergency abort activated")
            
            # Try to place another order - should be rejected
            order2 = create_market_order("GOOGL", 50, strategy="post_abort")
            result2 = await router.route_order(order2)
            
            assert result2.decision.value == 'REJECT', "Should reject order after emergency abort"
            # The rejection could be from authorization or safety valve
            reject_reasons = ['Emergency abort', 'not authorized', 'safety valve']
            found_reason = any(reason in result2.rejection_reason for reason in reject_reasons)
            assert found_reason, f"Should mention emergency abort related rejection, got: {result2.rejection_reason}"
            print("  ✅ Order rejected after emergency abort")
            
            # Check routing stats
            stats = router.get_routing_stats()
            assert stats['safety_valve']['emergency_abort'], "Should show emergency abort in stats"
            assert not stats['live_trading_authorized'], "Authorization should be reset"
            print("  ✅ Emergency abort status reflected in stats")
            
            await broker.disconnect()
            risk_checker.close()
            
        print("  ✅ Emergency abort integration test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Emergency abort integration test failed: {e}")
        return False


async def test_daily_value_limits():
    """Test daily value limits integration"""
    print("\n💰 Testing Daily Value Limits Integration...")
    
    try:
        # Set stub mode
        os.environ['EXO_MODE'] = 'stub'
        
        broker = EnhancedStubBroker({'simulate_fills': True})
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
            
            # Configure low daily limit for testing
            router_config = {
                'safety': {
                    'safety_mode': 'disabled',
                    'max_daily_value': 5000.0  # $5k limit
                }
            }
            
            router = OrderRouter(broker, risk_checker, router_config)
            
            # Place order within limit
            broker.set_market_price("AAPL", 100.0)  # Set known price
            small_order = create_limit_order("AAPL", 30, 100.0, strategy="small")  # $3k
            result1 = await router.route_order(small_order)
            
            assert result1.decision.value == 'APPROVE', "Should approve order within limit"
            print("  ✅ Small order approved")
            
            # Try to place order that would exceed limit
            large_order = create_limit_order("GOOGL", 30, 100.0, strategy="large")  # $3k (total $6k)
            result2 = await router.route_order(large_order)
            
            assert result2.decision.value == 'REJECT', "Should reject order exceeding daily limit"
            assert 'daily limit' in result2.rejection_reason.lower(), "Should mention daily limit"
            print("  ✅ Large order rejected for exceeding daily limit")
            
            # Check safety status shows used value
            safety_status = router.get_safety_status()
            assert safety_status['daily_value_used'] > 0, "Should track daily value used"
            print(f"  ✅ Daily value tracking: ${safety_status['daily_value_used']:,.0f} used")
            
            await broker.disconnect()
            risk_checker.close()
            
        print("  ✅ Daily value limits integration test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Daily value limits integration test failed: {e}")
        return False


async def test_safety_warnings():
    """Test safety warnings integration"""
    print("\n⚠️  Testing Safety Warnings Integration...")
    
    try:
        # Set stub mode
        os.environ['EXO_MODE'] = 'stub'
        
        broker = EnhancedStubBroker({'simulate_fills': True})
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
            
            router_config = {
                'safety': {
                    'safety_mode': 'disabled',
                    'max_daily_value': 50000.0
                }
            }
            
            router = OrderRouter(broker, risk_checker, router_config)
            
            # Mock time to be outside market hours for warning
            with patch('mech_exo.execution.safety_valve.datetime') as mock_dt:
                mock_now = datetime.now().replace(hour=6, minute=0)  # 6 AM
                mock_dt.now.return_value = mock_now
                
                order = create_market_order("AAPL", 100, strategy="early_morning")
                result = await router.route_order(order)
                
                assert result.decision.value == 'APPROVE', "Should approve with warnings"
                assert len(result.risk_warnings) > 0, "Should have risk warnings"
                
                market_hours_warning = any('market hours' in warning for warning in result.risk_warnings)
                assert market_hours_warning, "Should have market hours warning"
                print("  ✅ Market hours warning propagated to routing result")
            
            await broker.disconnect()
            risk_checker.close()
            
        print("  ✅ Safety warnings integration test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Safety warnings integration test failed: {e}")
        return False


async def main():
    """Run all OrderRouter + SafetyValve integration tests"""
    print("🛡️ Running OrderRouter + SafetyValve Integration Tests\n")
    
    tests = [
        test_orderrouter_safety_integration,
        test_live_mode_authorization,
        test_emergency_abort_integration,
        test_daily_value_limits,
        test_safety_warnings
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
        print("🎉 All OrderRouter + SafetyValve integration tests PASSED!")
        print("\n✅ Verified Integration Features:")
        print("  - SafetyValve embedded in OrderRouter workflow")
        print("  - Live trading authorization flow")
        print("  - Emergency abort propagation")
        print("  - Daily value limits enforcement")
        print("  - Safety warnings in routing results")
        print("  - Health checks include safety status")
        print("  - Safety status reporting")
        print("\n🚀 Task 2-3 (Live-Mode Safety Valve) COMPLETE!")
        return True
    else:
        print("❌ Some OrderRouter + SafetyValve integration tests FAILED!")
        return False


if __name__ == "__main__":
    # Clean up environment for testing
    os.environ.pop('EXO_MODE', None)
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)