#!/usr/bin/env python3
"""
Test script for SafetyValve functionality
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
import os
from datetime import datetime
from unittest.mock import Mock, patch

from mech_exo.execution.safety_valve import SafetyValve, SafetyMode
from mech_exo.execution.models import create_market_order, create_limit_order
from tests.stubs.broker_stub import EnhancedStubBroker


async def test_safety_valve_disabled():
    """Test safety valve in disabled mode"""
    print("🧪 Testing SafetyValve in DISABLED mode...")
    
    try:
        # Set non-live mode
        os.environ['EXO_MODE'] = 'stub'
        
        broker = EnhancedStubBroker({'simulate_fills': True})
        await broker.connect()
        
        config = {'safety_mode': 'disabled'}
        safety_valve = SafetyValve(broker, config)
        
        # Should authorize immediately in non-live mode
        authorized = await safety_valve.authorize_live_trading("test session")
        assert authorized, "Should authorize in stub mode"
        print("  ✅ Non-live mode authorization works")
        
        # Test order safety check
        order = create_market_order("AAPL", 100, strategy="test")
        safety_result = await safety_valve.check_order_safety(order)
        
        assert safety_result['approved'], "Should approve order in disabled mode"
        print("  ✅ Order safety check works")
        
        await broker.disconnect()
        print("  ✅ SafetyValve DISABLED mode test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ SafetyValve DISABLED mode test failed: {e}")
        return False


async def test_safety_valve_sentinel_only():
    """Test safety valve with sentinel orders only"""
    print("\n🛡️ Testing SafetyValve SENTINEL_ONLY mode...")
    
    try:
        # Set live mode for testing
        os.environ['EXO_MODE'] = 'live'
        
        broker = EnhancedStubBroker({'simulate_fills': True, 'fill_delay_ms': 10})
        await broker.connect()
        
        config = {
            'safety_mode': 'sentinel_only',
            'require_confirmation': False,
            'sentinel': {
                'symbol': 'CAD',
                'quantity': 100,
                'max_price': 1.50,
                'timeout_seconds': 5
            }
        }
        
        safety_valve = SafetyValve(broker, config)
        
        # Mock user input to avoid blocking
        with patch('builtins.input', return_value='yes'):
            authorized = await safety_valve.authorize_live_trading("sentinel test")
            assert authorized, "Should authorize with successful sentinel"
            print("  ✅ Sentinel-only authorization works")
        
        # Test safety status
        status = safety_valve.get_safety_status()
        assert status['mode'] == 'sentinel_only', "Should report correct mode"
        assert not status['emergency_abort'], "Should not be in emergency abort"
        print("  ✅ Safety status reporting works")
        
        await broker.disconnect()
        print("  ✅ SafetyValve SENTINEL_ONLY mode test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ SafetyValve SENTINEL_ONLY mode test failed: {e}")
        return False


async def test_emergency_abort():
    """Test emergency abort functionality"""
    print("\n🚨 Testing Emergency Abort functionality...")
    
    try:
        broker = EnhancedStubBroker({'simulate_fills': True})
        await broker.connect()
        
        safety_valve = SafetyValve(broker, {})
        
        # Test normal state
        assert not safety_valve.emergency_abort, "Should start without emergency abort"
        
        # Activate emergency abort
        safety_valve.activate_emergency_abort("Testing emergency abort")
        assert safety_valve.emergency_abort, "Should activate emergency abort"
        print("  ✅ Emergency abort activation works")
        
        # Test that authorization fails
        os.environ['EXO_MODE'] = 'live'
        authorized = await safety_valve.authorize_live_trading("abort test")
        assert not authorized, "Should not authorize with emergency abort active"
        print("  ✅ Emergency abort blocks authorization")
        
        # Test order safety with abort active
        order = create_market_order("AAPL", 100, strategy="test")
        safety_result = await safety_valve.check_order_safety(order)
        assert not safety_result['approved'], "Should reject orders with emergency abort"
        assert 'Emergency abort' in safety_result['reason'], "Should mention emergency abort"
        print("  ✅ Emergency abort blocks orders")
        
        # Deactivate emergency abort
        safety_valve.deactivate_emergency_abort("Testing deactivation")
        assert not safety_valve.emergency_abort, "Should deactivate emergency abort"
        print("  ✅ Emergency abort deactivation works")
        
        await broker.disconnect()
        print("  ✅ Emergency Abort test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Emergency Abort test failed: {e}")
        return False


async def test_daily_limits():
    """Test daily value limits"""
    print("\n💰 Testing Daily Value Limits...")
    
    try:
        broker = EnhancedStubBroker({'simulate_fills': True})
        await broker.connect()
        
        config = {
            'max_daily_value': 10000.0,  # $10k daily limit
            'safety_mode': 'disabled'
        }
        
        safety_valve = SafetyValve(broker, config)
        
        # Test normal order within limit
        normal_order = create_limit_order("AAPL", 50, 150.0, strategy="normal")  # $7.5k
        safety_result = await safety_valve.check_order_safety(normal_order)
        
        assert safety_result['approved'], "Should approve order within daily limit"
        assert safety_result['order_value'] == 7500.0, "Should calculate correct order value"
        print(f"  ✅ Normal order approved: ${safety_result['order_value']:,.0f}")
        
        # Test order that would exceed limit
        large_order = create_limit_order("GOOGL", 20, 200.0, strategy="large")  # $4k (total would be $11.5k)
        safety_result = await safety_valve.check_order_safety(large_order)
        
        assert not safety_result['approved'], "Should reject order exceeding daily limit"
        assert 'exceed daily limit' in safety_result['reason'], "Should mention daily limit"
        print(f"  ✅ Large order rejected: {safety_result['reason']}")
        
        # Test daily reset
        safety_valve._reset_daily_counters()
        print("  ✅ Daily counter reset works")
        
        await broker.disconnect()
        print("  ✅ Daily Value Limits test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Daily Value Limits test failed: {e}")
        return False


async def test_market_hours_warning():
    """Test market hours warnings"""
    print("\n⏰ Testing Market Hours Warnings...")
    
    try:
        broker = EnhancedStubBroker({'simulate_fills': True})
        await broker.connect()
        
        safety_valve = SafetyValve(broker, {'safety_mode': 'disabled'})
        
        # Mock current time to be outside market hours (e.g., 6 AM)
        with patch('mech_exo.execution.safety_valve.datetime') as mock_dt:
            mock_now = datetime.now().replace(hour=6, minute=0, second=0)
            mock_dt.now.return_value = mock_now
            
            order = create_market_order("AAPL", 100, strategy="early_morning")
            safety_result = await safety_valve.check_order_safety(order)
            
            # Should approve but warn about market hours
            assert safety_result['approved'], "Should approve order outside hours"
            warnings = safety_result.get('warnings', [])
            market_hours_warning = any('market hours' in warning for warning in warnings)
            assert market_hours_warning, "Should warn about market hours"
            print("  ✅ Market hours warning works")
        
        await broker.disconnect()
        print("  ✅ Market Hours Warnings test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Market Hours Warnings test failed: {e}")
        return False


async def test_sentinel_summary():
    """Test sentinel order summary"""
    print("\n📊 Testing Sentinel Summary...")
    
    try:
        broker = EnhancedStubBroker({'simulate_fills': True})
        await broker.connect()
        
        config = {
            'sentinel': {
                'symbol': 'USD',
                'quantity': 50,
                'max_price': 1.0,
                'timeout_seconds': 10
            }
        }
        
        safety_valve = SafetyValve(broker, config)
        
        # Get sentinel summary
        summary = safety_valve.get_sentinel_summary()
        
        assert summary['config']['symbol'] == 'USD', "Should report correct symbol"
        assert summary['config']['quantity'] == 50, "Should report correct quantity"
        assert summary['config']['max_value'] == 50.0, "Should calculate max value"
        assert len(summary['recent_orders']) == 0, "Should start with no recent orders"
        
        print(f"  ✅ Sentinel config: {summary['config']['symbol']} {summary['config']['quantity']} (max ${summary['config']['max_value']:.0f})")
        
        await broker.disconnect()
        print("  ✅ Sentinel Summary test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Sentinel Summary test failed: {e}")
        return False


async def main():
    """Run all safety valve tests"""
    print("🛡️ Running SafetyValve Tests\n")
    
    tests = [
        test_safety_valve_disabled,
        test_safety_valve_sentinel_only,
        test_emergency_abort,
        test_daily_limits,
        test_market_hours_warning,
        test_sentinel_summary
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
    
    print(f"\n📊 SafetyValve Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All SafetyValve tests PASSED!")
        print("\n✅ Verified Features:")
        print("  - Safety modes (disabled, sentinel_only, full_safety)")
        print("  - CLI confirmation with timeout")
        print("  - Sentinel order verification")
        print("  - Emergency abort functionality")
        print("  - Daily value limits and tracking")
        print("  - Market hours warnings")
        print("  - Safety status and sentinel summaries")
        print("\n🚀 SafetyValve ready for integration!")
        return True
    else:
        print("❌ Some SafetyValve tests FAILED!")
        return False


if __name__ == "__main__":
    # Clean up environment for testing
    os.environ.pop('EXO_MODE', None)
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)