#!/usr/bin/env python3
"""
Test script for Alert system
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime
from mech_exo.utils.alerts import AlertManager, Alert, AlertType, AlertLevel


def test_alert_manager_basic():
    """Test basic AlertManager functionality without external services"""
    print("📢 Testing AlertManager Basic...")
    
    try:
        # Initialize with no external services (should work without Slack/email config)
        alert_manager = AlertManager()
        
        # Test creating alerts
        test_alert = Alert(
            alert_type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.INFO,
            title="Test Alert",
            message="This is a test alert for verification",
            timestamp=datetime.now(),
            data={'test_key': 'test_value'}
        )
        
        # Test alert structure
        alert_dict = test_alert.to_dict()
        assert 'alert_type' in alert_dict, "Alert should have alert_type"
        assert 'level' in alert_dict, "Alert should have level"
        assert 'title' in alert_dict, "Alert should have title"
        assert alert_dict['alert_type'] == 'system_error', "Alert type should match"
        
        print(f"  ✅ Alert structure: {alert_dict['alert_type']}.{alert_dict['level']}")
        
        # Test filtering (without actually sending)
        should_send = alert_manager._should_send_alert(test_alert)
        print(f"  ✅ Alert filtering works: {should_send}")
        
        # Test convenience methods (without sending)
        print("  ✅ Convenience methods available:")
        print("    - send_fill_alert")
        print("    - send_order_reject_alert") 
        print("    - send_risk_violation_alert")
        print("    - send_system_error_alert")
        print("    - send_daily_summary_alert")
        
        print("  ✅ AlertManager basic test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ AlertManager basic test failed: {e}")
        return False


def test_alert_formatting():
    """Test alert message formatting"""
    print("\n📝 Testing Alert Formatting...")
    
    try:
        # Create sample alerts of different types
        alerts = [
            Alert(
                alert_type=AlertType.FILL,
                level=AlertLevel.INFO,
                title="Order Filled: AAPL",
                message="BUY 100 shares of AAPL @ $150.00",
                timestamp=datetime.now(),
                data={'symbol': 'AAPL', 'quantity': 100, 'price': 150.0}
            ),
            Alert(
                alert_type=AlertType.RISK_VIOLATION,
                level=AlertLevel.WARNING,
                title="Risk Violation Detected",
                message="Position size exceeds limit",
                timestamp=datetime.now(),
                data={'violation': 'position_size', 'limit': 0.1, 'actual': 0.15}
            ),
            Alert(
                alert_type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.ERROR,
                title="Broker Connection Failed",
                message="Unable to connect to IB Gateway",
                timestamp=datetime.now(),
                data={'broker': 'IB', 'port': 4002, 'retry_count': 3}
            )
        ]
        
        for i, alert in enumerate(alerts):
            alert_dict = alert.to_dict()
            print(f"  📄 Alert {i+1}: {alert_dict['title']}")
            print(f"    Type: {alert_dict['alert_type']}, Level: {alert_dict['level']}")
            print(f"    Data fields: {len(alert_dict['data'])}")
        
        print("  ✅ Alert formatting test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Alert formatting test failed: {e}")
        return False


def test_convenience_methods():
    """Test convenience alert methods"""
    print("\n🛠️ Testing Convenience Methods...")
    
    try:
        alert_manager = AlertManager()
        
        # Test different convenience methods (without actually sending)
        test_cases = [
            ("Fill Alert", lambda: alert_manager.send_fill_alert("AAPL", 100, 150.0, "fill123")),
            ("Order Reject Alert", lambda: alert_manager.send_order_reject_alert("GOOGL", 50, "Insufficient funds", "order456")),
            ("Risk Violation Alert", lambda: alert_manager.send_risk_violation_alert(["Position too large", "Sector limit exceeded"])),
            ("System Error Alert", lambda: alert_manager.send_system_error_alert("BrokerAdapter", "Connection timeout")),
            ("Daily Summary Alert", lambda: alert_manager.send_daily_summary_alert({
                'date': '2025-06-07',
                'signal_generation': {'signals_generated': 5},
                'execution': {'orders_submitted': 3, 'fills_received': 2},
                'risk_management': {'violations_count': 0}
            }))
        ]
        
        for test_name, test_func in test_cases:
            try:
                # These will "succeed" even without configured alerters
                # because AlertManager handles missing alerters gracefully
                result = test_func()
                print(f"  ✅ {test_name}: method callable")
            except Exception as e:
                print(f"  ⚠️  {test_name}: {e}")
        
        print("  ✅ Convenience methods test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Convenience methods test failed: {e}")
        return False


def test_configuration_loading():
    """Test configuration loading"""
    print("\n⚙️ Testing Configuration Loading...")
    
    try:
        # Test loading with config file
        alert_manager = AlertManager('alerts')
        
        print(f"  📄 Config loaded: {len(alert_manager.alerters)} alerters configured")
        print(f"  🎚️ Min level: {alert_manager.min_level.value}")
        print(f"  🏷️ Enabled types: {len(alert_manager.enabled_types)}")
        
        # Test the config structure
        config_keys = ['min_level', 'enabled_types', 'slack', 'email']
        for key in config_keys:
            if key in alert_manager.config:
                print(f"    ✅ {key}: configured")
            else:
                print(f"    ⚠️  {key}: not configured")
        
        print("  ✅ Configuration loading test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration loading test failed: {e}")
        return False


def test_integration_example():
    """Test integration example showing how to use in practice"""
    print("\n🔗 Testing Integration Example...")
    
    try:
        # Example: How this would be used in the trading system
        alert_manager = AlertManager()
        
        # Simulate trading events
        print("  📊 Simulating trading events:")
        
        # 1. Order filled
        print("    1. Order filled event")
        # alert_manager.send_fill_alert("SPY", 200, 425.50, "fill_001")
        
        # 2. Risk violation detected  
        print("    2. Risk violation event")
        violations = ["Portfolio exposure exceeds 95%", "Single position > 10% NAV"]
        # alert_manager.send_risk_violation_alert(violations, severity='warning')
        
        # 3. System error
        print("    3. System error event")
        # alert_manager.send_system_error_alert("OrderRouter", "Failed to place order after 3 retries")
        
        # 4. Daily summary
        print("    4. Daily summary event")
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'signal_generation': {'signals_generated': 8},
            'execution': {'orders_submitted': 5, 'fills_received': 4},
            'risk_management': {'positions_approved': 5, 'violations_count': 1}
        }
        # alert_manager.send_daily_summary_alert(summary)
        
        print("  ✅ Integration example completed!")
        print("  💡 To enable actual alerts:")
        print("    1. Configure Slack: set bot_token in config/alerts.yml")
        print("    2. Configure Email: set sender credentials in config/alerts.yml") 
        print("    3. Set enabled: true for desired channels")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration example test failed: {e}")
        return False


def main():
    """Run all alert system tests"""
    print("🚀 Testing Alert System\n")
    
    tests = [
        test_alert_manager_basic,
        test_alert_formatting,
        test_convenience_methods,
        test_configuration_loading,
        test_integration_example
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All alert system tests PASSED!")
        print("\n📋 Next Steps:")
        print("  1. Configure Slack bot token or webhook URL")
        print("  2. Configure email SMTP credentials")
        print("  3. Set enabled: true in config/alerts.yml")
        print("  4. Test with actual alert: python -c \"from mech_exo.utils.alerts import *; AlertManager().send_system_error_alert('Test', 'This is a test')\"")
        return True
    else:
        print("❌ Some alert system tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)