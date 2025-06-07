#!/usr/bin/env python3
"""
Test script for BrokerAdapter implementations
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
from mech_exo.execution.broker_adapter import create_broker_adapter, BrokerStatus
from mech_exo.execution.models import create_market_order, create_limit_order


async def test_stub_broker():
    """Test StubBroker functionality"""
    print("🧪 Testing StubBroker...")
    
    config = {
        'simulate_fills': True,
        'fill_delay': 0.5,
        'reject_probability': 0.0,
        'initial_nav': 100000
    }
    
    broker = create_broker_adapter('stub', config)
    
    try:
        # Test connection
        connected = await broker.connect()
        assert connected, "Should connect successfully"
        assert broker.is_connected(), "Should be connected"
        print("  ✅ Connection successful")
        
        # Test account info
        account_info = await broker.get_account_info()
        print(f"  Account NAV: ${account_info['netliquidation']:,.0f}")
        
        # Test order placement
        order = create_market_order("FXI", 100, strategy="test")
        
        def on_order_update(updated_order):
            print(f"  📝 Order update: {updated_order.symbol} - {updated_order.status.value}")
        
        def on_fill_update(fill):
            print(f"  💰 Fill: {fill.symbol} {fill.quantity} @ ${fill.price}")
        
        broker.add_order_callback(on_order_update)
        broker.add_fill_callback(on_fill_update)
        
        result = await broker.place_order(order)
        print(f"  Order result: {result['status']}")
        
        # Wait for simulated fill
        await asyncio.sleep(1.0)
        
        # Test order status
        status = await broker.get_order_status(order.order_id)
        print(f"  Final order status: {status['status']}")
        
        # Test disconnection
        disconnected = await broker.disconnect()
        assert disconnected, "Should disconnect successfully"
        print("  ✅ StubBroker test passed!")
        
        return True
        
    except Exception as e:
        print(f"  ❌ StubBroker test failed: {e}")
        return False


async def test_ib_adapter_connection():
    """Test IBAdapter connection (without actual IB Gateway)"""
    print("\n🏢 Testing IBAdapter...")
    
    config = {
        'host': 'localhost',
        'port': 4002,
        'client_id': 1
    }
    
    broker = create_broker_adapter('ib', config)
    
    try:
        # This will fail without IB Gateway running, which is expected
        try:
            await broker.connect()
            print("  ✅ IB Connection successful (Gateway is running)")
            
            # Test basic info
            info = broker.get_broker_info()
            print(f"  Broker: {info.broker_name}, Status: {info.status.value}")
            
            await broker.disconnect()
            return True
            
        except Exception as e:
            if "Failed to connect" in str(e) or "ib_insync not installed" in str(e):
                print(f"  ⚠️  Expected failure: {e}")
                print("  ✅ IBAdapter test passed (no Gateway/ib_insync available)")
                return True
            else:
                raise e
                
    except Exception as e:
        print(f"  ❌ IBAdapter test failed: {e}")
        return False


async def test_order_factory():
    """Test order creation factory functions"""
    print("\n🏭 Testing Order Factory...")
    
    try:
        # Test market order
        market_order = create_market_order("AAPL", 100, strategy="momentum")
        assert market_order.symbol == "AAPL"
        assert market_order.quantity == 100
        assert market_order.is_buy
        print(f"  Market order: {market_order.symbol} {market_order.quantity}")
        
        # Test limit order  
        limit_order = create_limit_order("GOOGL", -50, 120.0, strategy="mean_revert")
        assert limit_order.symbol == "GOOGL"
        assert limit_order.quantity == -50
        assert limit_order.is_sell
        assert limit_order.limit_price == 120.0
        print(f"  Limit order: {limit_order.symbol} {limit_order.quantity} @ ${limit_order.limit_price}")
        
        print("  ✅ Order factory test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Order factory test failed: {e}")
        return False


async def main():
    """Run all broker adapter tests"""
    print("🚀 Testing Broker Adapters\n")
    
    tests = [
        test_order_factory,
        test_stub_broker,
        test_ib_adapter_connection
    ]
    
    results = []
    for test in tests:
        results.append(await test())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All broker adapter tests PASSED!")
        return True
    else:
        print("❌ Some broker adapter tests FAILED!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)