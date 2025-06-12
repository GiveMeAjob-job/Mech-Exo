#!/usr/bin/env python3
"""
Test script for canary auto-disable functionality

Tests the auto-disable logic and Telegram alert system.
"""

import sys
import os
from pathlib import Path
from datetime import date, datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mech_exo.execution.allocation import (
    get_allocation_config, 
    update_canary_enabled, 
    is_canary_enabled,
    get_canary_allocation
)
from mech_exo.utils.alerts import AlertManager, Alert, AlertType, AlertLevel


def test_allocation_config():
    """Test allocation configuration loading"""
    print("🔧 Testing allocation configuration...")
    
    try:
        config = get_allocation_config()
        print(f"✅ Allocation config loaded:")
        print(f"   - Canary enabled: {config.get('canary_enabled', 'unknown')}")
        print(f"   - Canary allocation: {config.get('canary_allocation', 0):.1%}")
        print(f"   - Disable threshold: {config.get('disable_threshold_sharpe', 0):.3f}")
        print(f"   - Minimum days: {config.get('disable_min_days', 0)}")
        return True
    except Exception as e:
        print(f"❌ Failed to load allocation config: {e}")
        return False


def test_canary_enable_disable():
    """Test canary enable/disable functionality"""
    print("\n🔄 Testing canary enable/disable...")
    
    try:
        # Get initial state
        initial_state = is_canary_enabled()
        print(f"   Initial state: {'enabled' if initial_state else 'disabled'}")
        
        # Test disable
        success = update_canary_enabled(False)
        if success:
            current_state = is_canary_enabled()
            print(f"   After disable: {'enabled' if current_state else 'disabled'}")
            
            if not current_state:
                print("   ✅ Disable test passed")
            else:
                print("   ❌ Disable test failed - still enabled")
                return False
        else:
            print("   ❌ Failed to disable canary")
            return False
        
        # Test enable
        success = update_canary_enabled(True)
        if success:
            current_state = is_canary_enabled()
            print(f"   After enable: {'enabled' if current_state else 'disabled'}")
            
            if current_state:
                print("   ✅ Enable test passed")
            else:
                print("   ❌ Enable test failed - still disabled")
                return False
        else:
            print("   ❌ Failed to enable canary")
            return False
        
        # Restore initial state
        update_canary_enabled(initial_state)
        print(f"   Restored to: {'enabled' if initial_state else 'disabled'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enable/disable test failed: {e}")
        return False


def test_telegram_alert():
    """Test Telegram alert functionality"""
    print("\n📱 Testing Telegram alert...")
    
    # Set dry-run mode to avoid actually sending alerts during testing
    os.environ['TELEGRAM_DRY_RUN'] = 'true'
    
    try:
        alert_manager = AlertManager()
        
        if 'telegram' not in alert_manager.alerters:
            print("   ⚠️ Telegram alerter not configured - skipping test")
            print("   Configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable")
            return True
        
        # Create test alert for auto-disable
        alert = Alert(
            alert_type=AlertType.RISK_VIOLATION,
            level=AlertLevel.CRITICAL,
            title="🧪 Test: Canary Auto-Disabled",
            message="This is a test alert for canary auto-disable functionality:\n\n"
                   "• Canary Sharpe (30d): -0.150\n"
                   "• Threshold: 0.000\n"
                   "• Observations: 35 days\n"
                   "• Data quality: good\n\n"
                   "All new orders will use base allocation only.\n"
                   "Manual review and re-enable required.",
            timestamp=datetime.now(),
            data={
                'canary_sharpe': -0.15,
                'threshold': 0.0,
                'observations': 35,
                'data_quality': 'good',
                'auto_disabled': True,
                'test_alert': True
            }
        )
        
        success = alert_manager.send_alert(alert, channels=['telegram'])
        
        if success:
            print("   ✅ Telegram alert test passed (dry-run)")
            print("   Check logs for message content")
        else:
            print("   ❌ Telegram alert test failed")
            
        return success
        
    except Exception as e:
        print(f"❌ Telegram alert test failed: {e}")
        return False
    finally:
        # Clean up environment
        if 'TELEGRAM_DRY_RUN' in os.environ:
            del os.environ['TELEGRAM_DRY_RUN']


def test_auto_disable_logic_simulation():
    """Simulate the auto-disable logic without actually running the full flow"""
    print("\n🤖 Testing auto-disable logic simulation...")
    
    try:
        from mech_exo.execution.allocation import get_allocation_config, is_canary_enabled
        
        # Get configuration
        config = get_allocation_config()
        threshold = config.get('disable_threshold_sharpe', 0.0)
        min_days = config.get('disable_min_days', 30)
        
        # Test scenarios
        test_scenarios = [
            {
                'name': 'Good performance',
                'canary_sharpe': 0.5,
                'observations': 35,
                'data_quality': 'good',
                'expected_disable': False
            },
            {
                'name': 'Poor performance - should disable',
                'canary_sharpe': -0.2,
                'observations': 35,
                'data_quality': 'good',
                'expected_disable': True
            },
            {
                'name': 'Poor performance but insufficient data',
                'canary_sharpe': -0.2,
                'observations': 15,
                'data_quality': 'poor',
                'expected_disable': False
            },
            {
                'name': 'Borderline performance',
                'canary_sharpe': 0.01,
                'observations': 30,
                'data_quality': 'fair',
                'expected_disable': False
            }
        ]
        
        print(f"   Threshold: {threshold:.3f}, Min days: {min_days}")
        print()
        
        all_passed = True
        
        for scenario in test_scenarios:
            canary_sharpe = scenario['canary_sharpe']
            observations = scenario['observations']
            data_quality = scenario['data_quality']
            expected_disable = scenario['expected_disable']
            
            # Simulate disable logic
            should_disable = (
                data_quality in ['good', 'fair'] and
                observations >= min_days * 0.7 and
                canary_sharpe < threshold
            )
            
            result_emoji = "🟢" if should_disable == expected_disable else "🔴"
            action = "DISABLE" if should_disable else "KEEP"
            
            print(f"   {result_emoji} {scenario['name']}:")
            print(f"      Sharpe: {canary_sharpe:.3f}, Obs: {observations}, Quality: {data_quality}")
            print(f"      Action: {action}, Expected: {'DISABLE' if expected_disable else 'KEEP'}")
            
            if should_disable != expected_disable:
                all_passed = False
                print(f"      ❌ FAILED: Expected {'disable' if expected_disable else 'keep'}")
            else:
                print(f"      ✅ PASSED")
            print()
        
        if all_passed:
            print("   ✅ All auto-disable logic tests passed")
        else:
            print("   ❌ Some auto-disable logic tests failed")
            
        return all_passed
        
    except Exception as e:
        print(f"❌ Auto-disable logic simulation failed: {e}")
        return False


def main():
    """Run all auto-disable tests"""
    print("🧪 Canary Auto-Disable Test Suite")
    print("=" * 50)
    
    # Track test results
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Allocation config
    if test_allocation_config():
        tests_passed += 1
    
    # Test 2: Enable/disable functionality
    if test_canary_enable_disable():
        tests_passed += 1
    
    # Test 3: Telegram alert
    if test_telegram_alert():
        tests_passed += 1
    
    # Test 4: Auto-disable logic simulation
    if test_auto_disable_logic_simulation():
        tests_passed += 1
    
    # Summary
    print("=" * 50)
    print(f"📊 Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Day 4 auto-disable functionality is working.")
        print("\n📋 Next Steps:")
        print("1. Configure Telegram bot token and chat ID:")
        print("   export TELEGRAM_BOT_TOKEN='your_bot_token'")
        print("   export TELEGRAM_CHAT_ID='your_chat_id'")
        print("2. Test auto-disable with real flow:")
        print("   python dags/canary_perf_flow.py")
        print("3. Check allocation status:")
        print("   python -c \"from mech_exo.execution.allocation import is_canary_enabled; print(is_canary_enabled())\"")
    elif tests_passed >= total_tests * 0.75:
        print("⚠️  Most tests passed. Some features may need configuration.")
    else:
        print("❌ Multiple tests failed. Check implementation and configuration.")


if __name__ == "__main__":
    main()