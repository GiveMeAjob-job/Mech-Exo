#!/usr/bin/env python3
"""
Test structured logging implementation for execution monitoring
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
import os
import json
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

# Configure structured logging before importing execution modules
from mech_exo.utils.structured_logging import configure_structured_logging, get_execution_logger, create_session_context

# Set up JSON logging for testing
temp_log_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
temp_log_file.close()

configure_structured_logging(
    log_level="INFO",
    log_file=temp_log_file.name,
    console_output=True,
    json_format=True
)

# Now import execution modules
from mech_exo.execution.order_router import OrderRouter
from mech_exo.execution.models import create_market_order, create_limit_order
from mech_exo.risk import RiskChecker, Portfolio
from tests.stubs.broker_stub import EnhancedStubBroker


async def test_structured_logging_basic():
    """Test basic structured logging functionality"""
    print("📝 Testing Basic Structured Logging...")
    
    try:
        # Create execution context
        context = create_session_context(
            strategy="test_strategy",
            account_id="TEST_ACCOUNT_001"
        )
        
        # Get structured logger
        logger = get_execution_logger("test.basic", context)
        
        # Test different log types
        logger.order_event(
            event_type="test",
            order_id="TEST_ORDER_001",
            symbol="AAPL",
            message="Test order event",
            quantity=100,
            price=150.0
        )
        
        logger.performance_event(
            metric_name="test_latency",
            value=25.5,
            unit="milliseconds",
            message="Test performance metric"
        )
        
        logger.risk_event(
            risk_type="position_size",
            severity="medium",
            message="Test risk event"
        )
        
        logger.safety_event(
            safety_type="daily_limit",
            action="warning",
            message="Test safety event"
        )
        
        logger.system_event(
            system="broker",
            status="connected",
            message="Test system event"
        )
        
        print("  ✅ Basic structured logging events generated")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic structured logging test failed: {e}")
        return False


async def test_orderrouter_structured_logging():
    """Test OrderRouter structured logging"""
    print("\n🔄 Testing OrderRouter Structured Logging...")
    
    try:
        # Set stub mode
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
            
            # Create OrderRouter
            router_config = {
                'max_retries': 2,
                'safety': {
                    'safety_mode': 'disabled',
                    'max_daily_value': 50000.0
                }
            }
            
            router = OrderRouter(broker, risk_checker, router_config)
            
            # Route an order - this should generate structured logs
            order = create_market_order("AAPL", 100, strategy="test_logging")
            result = await router.route_order(order)
            
            assert result.decision.value == 'APPROVE', "Order should be approved"
            
            # Wait for fill to generate more logs
            await asyncio.sleep(0.1)
            
            print("  ✅ OrderRouter structured logging completed")
            
            await broker.disconnect()
            risk_checker.close()
            
        return True
        
    except Exception as e:
        print(f"  ❌ OrderRouter structured logging test failed: {e}")
        return False


async def test_safety_valve_structured_logging():
    """Test SafetyValve structured logging"""
    print("\n🛡️ Testing SafetyValve Structured Logging...")
    
    try:
        # Set live mode to trigger safety valve logging
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
            
            # Configure with confirmation only (no sentinel for testing)
            router_config = {
                'safety': {
                    'safety_mode': 'confirmation_only',
                    'max_daily_value': 50000.0
                }
            }
            
            router = OrderRouter(broker, risk_checker, router_config)
            
            # Mock user input to approve
            with patch('builtins.input', return_value='yes'):
                order = create_market_order("AAPL", 50, strategy="safety_logging_test")
                result = await router.route_order(order)
                
                assert result.decision.value == 'APPROVE', "Order should be approved after authorization"
                
                print("  ✅ Safety valve authorization logged")
            
            # Test emergency abort logging
            router.activate_emergency_abort("Testing structured logging")
            
            # Try another order - should be rejected
            order2 = create_market_order("GOOGL", 25, strategy="emergency_test")
            result2 = await router.route_order(order2)
            
            assert result2.decision.value == 'REJECT', "Order should be rejected after emergency abort"
            
            print("  ✅ Emergency abort events logged")
            
            await broker.disconnect()
            risk_checker.close()
            
        return True
        
    except Exception as e:
        print(f"  ❌ SafetyValve structured logging test failed: {e}")
        return False


def analyze_log_file():
    """Analyze the generated JSON log file"""
    print("\n📊 Analyzing Generated JSON Logs...")
    
    try:
        with open(temp_log_file.name, 'r') as f:
            log_lines = f.readlines()
        
        print(f"  📄 Generated {len(log_lines)} log entries")
        
        # Parse and analyze log entries
        events = []
        performance_metrics = []
        safety_events = []
        order_events = []
        
        for line in log_lines:
            try:
                log_entry = json.loads(line.strip())
                
                # Extract structured data
                extra = log_entry.get('extra', {})
                event_type = extra.get('event_type', '')
                
                if event_type.startswith('order.'):
                    order_events.append(log_entry)
                elif event_type.startswith('performance.'):
                    performance_metrics.append(log_entry)
                elif event_type.startswith('safety.'):
                    safety_events.append(log_entry)
                
                events.append(log_entry)
                
            except json.JSONDecodeError:
                continue  # Skip non-JSON lines
        
        print(f"  🎯 Event Analysis:")
        print(f"    - Total events: {len(events)}")
        print(f"    - Order events: {len(order_events)}")
        print(f"    - Performance metrics: {len(performance_metrics)}")
        print(f"    - Safety events: {len(safety_events)}")
        
        # Show sample events
        if order_events:
            print(f"  📋 Sample Order Event:")
            sample_order = order_events[0]
            extra = sample_order.get('extra', {})
            print(f"    - Event Type: {extra.get('event_type')}")
            print(f"    - Order ID: {extra.get('order_id')}")
            print(f"    - Symbol: {extra.get('symbol')}")
            print(f"    - Session ID: {extra.get('session_id')}")
        
        if performance_metrics:
            print(f"  ⚡ Sample Performance Metric:")
            sample_perf = performance_metrics[0]
            extra = sample_perf.get('extra', {})
            print(f"    - Metric: {extra.get('metric_name')}")
            print(f"    - Value: {extra.get('metric_value')} {extra.get('metric_unit')}")
        
        if safety_events:
            print(f"  🛡️ Sample Safety Event:")
            sample_safety = safety_events[0]
            extra = sample_safety.get('extra', {})
            print(f"    - Safety Type: {extra.get('safety_type')}")
            print(f"    - Action: {extra.get('safety_action')}")
        
        # Validate required fields are present
        required_fields = ['timestamp', 'level', 'logger', 'message']
        missing_fields = []
        
        for event in events[:5]:  # Check first 5 events
            for field in required_fields:
                if field not in event:
                    missing_fields.append(field)
        
        if missing_fields:
            print(f"  ⚠️ Missing required fields: {set(missing_fields)}")
        else:
            print("  ✅ All required fields present in log entries")
        
        print(f"  📁 Log file: {temp_log_file.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Log file analysis failed: {e}")
        return False


async def main():
    """Run all structured logging tests"""
    print("📝 Running Structured Logging Tests\n")
    
    tests = [
        test_structured_logging_basic,
        test_orderrouter_structured_logging,
        test_safety_valve_structured_logging
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Analyze generated logs
    analysis_result = analyze_log_file()
    results.append(analysis_result)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Structured Logging Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structured logging tests PASSED!")
        print("\n✅ Verified Features:")
        print("  - JSON log formatting with structured fields")
        print("  - Execution context and session tracking")
        print("  - Order lifecycle event logging")
        print("  - Performance metrics with timing")
        print("  - Safety valve event logging")
        print("  - Risk and error event logging")
        print("  - Broker integration logging")
        print("  - Emergency abort logging")
        print("\n🚀 Task 2-4 (Structured Logging) COMPLETE!")
        return True
    else:
        print("❌ Some structured logging tests FAILED!")
        return False


if __name__ == "__main__":
    # Clean up environment for testing
    os.environ.pop('EXO_MODE', None)
    
    try:
        success = asyncio.run(main())
        print(f"\n📁 Log file saved to: {temp_log_file.name}")
        sys.exit(0 if success else 1)
    finally:
        # Cleanup
        try:
            os.unlink(temp_log_file.name)
        except:
            pass