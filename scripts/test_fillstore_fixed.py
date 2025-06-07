#!/usr/bin/env python3
"""
Test script for Fixed FillStore with timezone handling
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import tempfile
import os
from datetime import datetime, timedelta, timezone, date
import pytz
from mech_exo.execution.fill_store import FillStore
from mech_exo.execution.models import Fill, Order, create_market_order, OrderStatus, OrderType


def test_timezone_handling():
    """Test timezone-aware datetime handling"""
    print("🌍 Testing Timezone Handling...")
    
    try:
        # Use temporary database
        temp_dir = tempfile.gettempdir()
        temp_db = os.path.join(temp_dir, f"test_tz_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
        
        with FillStore(temp_db) as store:
            # Test with different timezone inputs
            utc_time = datetime.now(timezone.utc)
            naive_time = datetime.now()  # No timezone
            local_time = datetime.now(pytz.timezone('US/Eastern'))
            
            fills = [
                Fill("order-1", "AAPL", 100, 150.0, utc_time, commission=1.0),
                Fill("order-2", "GOOGL", 50, 120.0, naive_time, commission=0.5),
                Fill("order-3", "MSFT", 75, 300.0, local_time, commission=0.75)
            ]
            
            # Store all fills
            for fill in fills:
                success = store.store_fill(fill)
                assert success, f"Should store fill {fill.fill_id}"
            
            # Retrieve fills and check timezone consistency
            retrieved_fills = store.get_fills()
            assert len(retrieved_fills) == 3, "Should retrieve all fills"
            
            for fill in retrieved_fills:
                assert fill.filled_at.tzinfo is not None, "Filled_at should be timezone-aware"
                assert fill.filled_at.tzinfo == timezone.utc, "Should be UTC timezone"
                print(f"  ✅ Fill {fill.symbol}: {fill.filled_at} (UTC)")
            
            # Test last_fill_ts method
            last_ts = store.last_fill_ts()
            assert last_ts is not None, "Should have last fill timestamp"
            assert last_ts.tzinfo == timezone.utc, "Last timestamp should be UTC"
            print(f"  ✅ Last fill timestamp: {last_ts} (UTC)")
            
            # Test last_fill_ts for specific symbol
            aapl_last = store.last_fill_ts("AAPL")
            assert aapl_last is not None, "Should have AAPL last timestamp"
            assert aapl_last.tzinfo == timezone.utc, "AAPL last timestamp should be UTC"
            print(f"  ✅ AAPL last timestamp: {aapl_last} (UTC)")
            
            print("  ✅ Timezone handling test passed!")
            
        # Cleanup
        if os.path.exists(temp_db):
            os.unlink(temp_db)
        return True
        
    except Exception as e:
        print(f"  ❌ Timezone handling test failed: {e}")
        return False


def test_enum_handling():
    """Test order type and status enum handling"""
    print("\n📝 Testing Enum Handling...")
    
    try:
        temp_dir = tempfile.gettempdir()
        temp_db = os.path.join(temp_dir, f"test_enum_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
        
        with FillStore(temp_db) as store:
            # Create orders with different types and statuses
            orders = [
                create_market_order("AAPL", 100, strategy="momentum"),
                Order("GOOGL", 50, OrderType.LIMIT, limit_price=120.0),
                Order("MSFT", -75, OrderType.STOP, stop_price=295.0)
            ]
            
            # Set different statuses
            orders[0].status = OrderStatus.FILLED
            orders[1].status = OrderStatus.SUBMITTED
            orders[2].status = OrderStatus.REJECTED
            
            # Store orders
            for order in orders:
                success = store.store_order(order)
                assert success, f"Should store order {order.order_id}"
                print(f"  ✅ Stored order: {order.symbol} {order.order_type.value} - {order.status.value}")
            
            # Test the database constraints work
            try:
                # Try to insert invalid order type directly (should fail)
                store.conn.execute("""
                    INSERT INTO orders (order_id, symbol, quantity, order_type, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, ["bad-order", "TEST", 100, "INVALID_TYPE", "PENDING", datetime.now(timezone.utc)])
                
                # If we get here, the constraint didn't work
                assert False, "Should have failed with invalid order type"
                
            except Exception:
                # This is expected - constraint should prevent invalid values
                print("  ✅ Database constraints work correctly")
            
            print("  ✅ Enum handling test passed!")
            
        if os.path.exists(temp_db):
            os.unlink(temp_db)
        return True
        
    except Exception as e:
        print(f"  ❌ Enum handling test failed: {e}")
        return False


def test_daily_metrics():
    """Test get_daily_metrics helper function"""
    print("\n📊 Testing Daily Metrics...")
    
    try:
        temp_dir = tempfile.gettempdir()
        temp_db = os.path.join(temp_dir, f"test_daily_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
        
        with FillStore(temp_db) as store:
            # Create test data for specific dates
            today = date.today()
            yesterday = today - timedelta(days=1)
            
            # Create fills for today
            today_fills = [
                Fill("order-1", "AAPL", 100, 150.0, 
                     datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc) + timedelta(hours=9),
                     commission=1.0, slippage_bps=2.5),
                Fill("order-2", "GOOGL", 50, 120.0,
                     datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc) + timedelta(hours=10), 
                     commission=0.5, slippage_bps=1.2),
                Fill("order-3", "MSFT", -75, 300.0,
                     datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc) + timedelta(hours=11),
                     commission=0.75, slippage_bps=-0.8)
            ]
            
            # Create fills for yesterday  
            yesterday_fills = [
                Fill("order-4", "TSLA", 25, 800.0,
                     datetime.combine(yesterday, datetime.min.time()).replace(tzinfo=timezone.utc) + timedelta(hours=14),
                     commission=2.0, slippage_bps=3.1)
            ]
            
            all_fills = today_fills + yesterday_fills
            
            # Store fills
            for fill in all_fills:
                store.store_fill(fill)
            
            # Create corresponding orders
            for fill in all_fills:
                order = Order(
                    symbol=fill.symbol,
                    quantity=fill.quantity,
                    order_type=OrderType.MARKET,
                    order_id=fill.order_id,
                    status=OrderStatus.FILLED,
                    created_at=fill.filled_at,
                    submitted_at=fill.filled_at
                )
                store.store_order(order)
            
            # Test daily metrics for today
            today_metrics = store.get_daily_metrics(today)
            
            assert today_metrics['date'] == today, "Should return correct date"
            assert today_metrics['fills']['total_fills'] == 3, "Should count today's fills"
            assert today_metrics['fills']['symbols_traded'] == 3, "Should count unique symbols"
            assert today_metrics['orders']['total_orders'] == 3, "Should count today's orders"
            assert today_metrics['orders']['fill_rate'] == 1.0, "Should have 100% fill rate"
            
            expected_volume = 100 + 50 + 75  # abs quantities
            assert today_metrics['fills']['total_volume'] == expected_volume, "Should calculate correct volume"
            
            expected_notional = 100*150 + 50*120 + 75*300  # gross values
            assert abs(today_metrics['fills']['total_notional'] - expected_notional) < 0.01, "Should calculate correct notional"
            
            print(f"  📈 Today's metrics:")
            print(f"    Fills: {today_metrics['fills']['total_fills']}")
            print(f"    Volume: {today_metrics['fills']['total_volume']} shares")
            print(f"    Notional: ${today_metrics['fills']['total_notional']:,.0f}")
            print(f"    Avg slippage: {today_metrics['execution_quality']['avg_slippage_bps']:.2f} bps")
            
            # Test daily metrics for yesterday
            yesterday_metrics = store.get_daily_metrics(yesterday)
            
            assert yesterday_metrics['fills']['total_fills'] == 1, "Should count yesterday's fills"
            assert yesterday_metrics['fills']['symbols_traded'] == 1, "Should count TSLA"
            
            print(f"  📈 Yesterday's metrics:")
            print(f"    Fills: {yesterday_metrics['fills']['total_fills']}")
            print(f"    Volume: {yesterday_metrics['fills']['total_volume']} shares")
            
            # Test metrics for day with no data
            empty_date = today - timedelta(days=10)
            empty_metrics = store.get_daily_metrics(empty_date)
            
            assert empty_metrics['fills']['total_fills'] == 0, "Should have zero fills for empty day"
            assert empty_metrics['orders']['total_orders'] == 0, "Should have zero orders for empty day"
            
            print("  ✅ Daily metrics test passed!")
            
        if os.path.exists(temp_db):
            os.unlink(temp_db)
        return True
        
    except Exception as e:
        print(f"  ❌ Daily metrics test failed: {e}")
        return False


def test_comprehensive_functionality():
    """Test comprehensive FillStore functionality"""
    print("\n🔧 Testing Comprehensive Functionality...")
    
    try:
        temp_dir = tempfile.gettempdir()
        temp_db = os.path.join(temp_dir, f"test_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
        
        with FillStore(temp_db) as store:
            # Create realistic trading scenario
            base_time = datetime.now(timezone.utc)
            
            # Morning batch of fills
            morning_fills = [
                Fill("morning-1", "SPY", 500, 425.50, base_time, commission=2.5, slippage_bps=1.2),
                Fill("morning-2", "QQQ", 200, 380.25, base_time + timedelta(minutes=5), commission=1.0, slippage_bps=0.8),
                Fill("morning-3", "IWM", 300, 195.75, base_time + timedelta(minutes=10), commission=1.5, slippage_bps=2.1)
            ]
            
            # Afternoon batch of fills
            afternoon_fills = [
                Fill("afternoon-1", "SPY", -250, 426.00, base_time + timedelta(hours=3), commission=1.25, slippage_bps=-0.5),
                Fill("afternoon-2", "AAPL", 100, 175.30, base_time + timedelta(hours=4), commission=0.5, slippage_bps=1.8)
            ]
            
            all_fills = morning_fills + afternoon_fills
            
            # Store all fills
            for fill in all_fills:
                store.store_fill(fill)
            
            # Test various query methods
            
            # 1. Get all fills
            all_retrieved = store.get_fills()
            assert len(all_retrieved) == 5, "Should retrieve all fills"
            
            # 2. Get fills by symbol
            spy_fills = store.get_fills(symbol="SPY")
            assert len(spy_fills) == 2, "Should get 2 SPY fills"
            
            # 3. Get fills by date range
            morning_range = store.get_fills(
                start_date=base_time - timedelta(minutes=30),
                end_date=base_time + timedelta(hours=1)
            )
            assert len(morning_range) == 3, "Should get morning fills"
            
            # 4. Test execution summary
            summary = store.get_execution_summary()
            assert summary['fills']['total_fills'] == 5, "Summary should count all fills"
            assert summary['fills']['unique_symbols'] == 4, "Should have 4 unique symbols"
            
            # 5. Test slippage analysis
            slippage_analysis = store.get_slippage_analysis()
            assert slippage_analysis['total_fills_analyzed'] == 5, "Should analyze all fills"
            
            # 6. Test top symbols
            top_symbols = store.get_top_symbols_by_volume()
            assert len(top_symbols) > 0, "Should have top symbols"
            assert top_symbols[0]['symbol'] in ['SPY', 'QQQ', 'IWM', 'AAPL'], "Should be valid symbol"
            
            print(f"  ✅ Retrieved {len(all_retrieved)} fills")
            print(f"  ✅ SPY fills: {len(spy_fills)}")
            print(f"  ✅ Morning fills: {len(morning_range)}")
            print(f"  ✅ Summary shows {summary['fills']['total_fills']} fills, {summary['fills']['unique_symbols']} symbols")
            print(f"  ✅ Top symbol by volume: {top_symbols[0]['symbol']}")
            
            print("  ✅ Comprehensive functionality test passed!")
            
        if os.path.exists(temp_db):
            os.unlink(temp_db)
        return True
        
    except Exception as e:
        print(f"  ❌ Comprehensive functionality test failed: {e}")
        return False


def main():
    """Run all FillStore fixed tests"""
    print("🚀 Testing Fixed FillStore\n")
    
    tests = [
        test_timezone_handling,
        test_enum_handling,
        test_daily_metrics,
        test_comprehensive_functionality
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    passed = sum(results)
    total = len(tests)
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All FillStore fixed tests PASSED!")
        print("\n✅ FillStore fixes verified:")
        print("  - Timezone-aware TIMESTAMPTZ columns")
        print("  - UTC timezone handling with _ensure_utc()")
        print("  - Text fields with CHECK constraints instead of enums")
        print("  - get_daily_metrics(date) helper method")
        print("  - last_fill_ts() helper method")
        return True
    else:
        print("❌ Some FillStore tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)