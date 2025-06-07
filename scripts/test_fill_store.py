#!/usr/bin/env python3
"""
Test script for FillStore functionality
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import tempfile
import os
from datetime import datetime, timedelta
from mech_exo.execution.fill_store import FillStore
from mech_exo.execution.models import Fill, Order, create_market_order, OrderStatus


def test_fill_store_basic():
    """Test basic FillStore functionality"""
    print("📊 Testing FillStore Basic Operations...")
    
    try:
        # Use temporary database - create unique name but don't create file yet
        temp_dir = tempfile.gettempdir()
        temp_db = os.path.join(temp_dir, f"test_fill_store_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
        
        with FillStore(temp_db) as store:
            # Test 1: Store a fill
            fill = Fill(
                order_id="test-order-1",
                symbol="AAPL",
                quantity=100,
                price=150.0,
                filled_at=datetime.now(),
                broker_order_id="IB123",
                broker_fill_id="FILL123",
                exchange="NASDAQ",
                commission=1.0,
                fees=0.05,
                reference_price=149.8,
                strategy="momentum"
            )
            
            # Calculate slippage
            fill.calculate_slippage(149.8)
            
            success = store.store_fill(fill)
            assert success, "Should store fill successfully"
            print(f"  ✅ Stored fill: {fill.symbol} {fill.quantity} @ ${fill.price}")
            
            # Test 2: Store an order
            order = create_market_order("AAPL", 100, strategy="momentum")
            order.status = OrderStatus.FILLED
            order.submitted_at = datetime.now()
            order.broker_order_id = "IB123"
            
            success = store.store_order(order)
            assert success, "Should store order successfully"
            print(f"  ✅ Stored order: {order.symbol} {order.quantity}")
            
            # Test 3: Retrieve fills
            fills = store.get_fills(symbol="AAPL")
            assert len(fills) == 1, "Should retrieve 1 fill"
            assert fills[0].symbol == "AAPL", "Should match symbol"
            assert fills[0].quantity == 100, "Should match quantity"
            print(f"  ✅ Retrieved {len(fills)} fills for AAPL")
            
            # Test 4: Store multiple fills for analysis
            test_fills = [
                Fill("order-2", "GOOGL", 50, 120.0, datetime.now() - timedelta(days=1), 
                     commission=0.5, slippage_bps=2.5, strategy="mean_revert"),
                Fill("order-3", "MSFT", -75, 300.0, datetime.now() - timedelta(days=2), 
                     commission=0.75, slippage_bps=-1.8, strategy="momentum"),
                Fill("order-4", "AAPL", -50, 151.0, datetime.now() - timedelta(days=3), 
                     commission=0.5, slippage_bps=5.2, strategy="breakout")
            ]
            
            for test_fill in test_fills:
                store.store_fill(test_fill)
            
            print(f"  ✅ Stored {len(test_fills)} additional test fills")
            
            # Test 5: Get execution summary
            summary = store.get_execution_summary()
            assert summary, "Should return execution summary"
            assert summary['fills']['total_fills'] >= 4, "Should count all fills"
            print(f"  📈 Summary: {summary['fills']['total_fills']} fills, ${summary['fills']['total_notional']:,.0f} notional")
            
            # Test 6: Slippage analysis
            slippage = store.get_slippage_analysis()
            assert slippage, "Should return slippage analysis"
            print(f"  📉 Slippage: {slippage['total_fills_analyzed']} fills analyzed")
            
            # Test 7: Top symbols
            top_symbols = store.get_top_symbols_by_volume(days=7, limit=5)
            assert len(top_symbols) > 0, "Should return top symbols"
            print(f"  🏆 Top symbol: {top_symbols[0]['symbol']} (${top_symbols[0]['total_notional']:,.0f})")
            
            print("  ✅ FillStore basic test passed!")
            
        # Cleanup
        if os.path.exists(temp_db):
            os.unlink(temp_db)
        return True
        
    except Exception as e:
        print(f"  ❌ FillStore basic test failed: {e}")
        return False


def test_fill_store_filtering():
    """Test FillStore filtering and query functionality"""
    print("\n🔍 Testing FillStore Filtering...")
    
    try:
        temp_dir = tempfile.gettempdir()
        temp_db = os.path.join(temp_dir, f"test_filtering_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
        
        with FillStore(temp_db) as store:
            # Create test data with different dates and strategies
            base_date = datetime.now()
            test_data = [
                ("AAPL", 100, 150.0, base_date, "momentum"),
                ("AAPL", -50, 151.0, base_date - timedelta(days=1), "profit_taking"),
                ("GOOGL", 25, 120.0, base_date - timedelta(days=2), "momentum"),
                ("MSFT", 75, 300.0, base_date - timedelta(days=10), "value"),
                ("AAPL", 200, 149.0, base_date - timedelta(days=15), "momentum")
            ]
            
            for symbol, qty, price, date, strategy in test_data:
                fill = Fill(
                    order_id=f"order-{symbol}-{abs(qty)}",
                    symbol=symbol,
                    quantity=qty,
                    price=price,
                    filled_at=date,
                    strategy=strategy,
                    commission=1.0
                )
                store.store_fill(fill)
            
            print(f"  📝 Created {len(test_data)} test fills")
            
            # Test symbol filtering
            aapl_fills = store.get_fills(symbol="AAPL")
            assert len(aapl_fills) == 3, f"Should have 3 AAPL fills, got {len(aapl_fills)}"
            print(f"  ✅ Symbol filter: {len(aapl_fills)} AAPL fills")
            
            # Test date filtering
            recent_fills = store.get_fills(start_date=base_date - timedelta(days=5))
            assert len(recent_fills) == 3, f"Should have 3 recent fills, got {len(recent_fills)}"
            print(f"  ✅ Date filter: {len(recent_fills)} recent fills")
            
            # Test strategy filtering
            momentum_fills = store.get_fills(strategy="momentum")
            assert len(momentum_fills) == 3, f"Should have 3 momentum fills, got {len(momentum_fills)}"
            print(f"  ✅ Strategy filter: {len(momentum_fills)} momentum fills")
            
            # Test combined filtering
            aapl_momentum = store.get_fills(symbol="AAPL", strategy="momentum")
            assert len(aapl_momentum) == 2, f"Should have 2 AAPL momentum fills, got {len(aapl_momentum)}"
            print(f"  ✅ Combined filter: {len(aapl_momentum)} AAPL momentum fills")
            
            print("  ✅ FillStore filtering test passed!")
            
        if os.path.exists(temp_db):
            os.unlink(temp_db)
        return True
        
    except Exception as e:
        print(f"  ❌ FillStore filtering test failed: {e}")
        return False


def test_fill_store_metrics():
    """Test FillStore metrics and analysis"""
    print("\n📊 Testing FillStore Metrics...")
    
    try:
        temp_dir = tempfile.gettempdir()
        temp_db = os.path.join(temp_dir, f"test_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
        
        with FillStore(temp_db) as store:
            # Create realistic test data
            base_date = datetime.now()
            
            # AAPL trades
            aapl_fills = [
                Fill("aapl-1", "AAPL", 100, 150.0, base_date, commission=1.0, slippage_bps=2.0),
                Fill("aapl-2", "AAPL", -100, 152.0, base_date + timedelta(hours=2), commission=1.0, slippage_bps=-1.5),
                Fill("aapl-3", "AAPL", 200, 151.0, base_date + timedelta(hours=4), commission=2.0, slippage_bps=3.2)
            ]
            
            # GOOGL trades
            googl_fills = [
                Fill("googl-1", "GOOGL", 50, 120.0, base_date, commission=0.5, slippage_bps=1.8),
                Fill("googl-2", "GOOGL", -25, 122.0, base_date + timedelta(hours=1), commission=0.25, slippage_bps=-0.5)
            ]
            
            all_fills = aapl_fills + googl_fills
            
            for fill in all_fills:
                store.store_fill(fill)
            
            # Store corresponding orders
            for fill in all_fills:
                order = Order(
                    symbol=fill.symbol,
                    quantity=fill.quantity,
                    order_type="MKT",
                    order_id=fill.order_id,
                    status=OrderStatus.FILLED,
                    created_at=fill.filled_at,
                    submitted_at=fill.filled_at
                )
                store.store_order(order)
            
            print(f"  📝 Created {len(all_fills)} fills with orders")
            
            # Test execution summary
            summary = store.get_execution_summary()
            
            # Verify key metrics
            assert summary['fills']['total_fills'] == 5, "Should count all fills"
            assert summary['fills']['unique_symbols'] == 2, "Should have 2 unique symbols"
            assert summary['orders']['total_orders'] == 5, "Should count all orders"
            assert summary['orders']['fill_rate'] == 1.0, "Should have 100% fill rate"
            
            total_notional = (100*150 + 100*152 + 200*151 + 50*120 + 25*122)
            assert abs(summary['fills']['total_notional'] - total_notional) < 0.01, "Should calculate correct notional"
            
            print(f"  ✅ Summary metrics: {summary['fills']['total_fills']} fills, ${summary['fills']['total_notional']:,.0f} notional")
            print(f"    Fill rate: {summary['orders']['fill_rate']:.1%}")
            print(f"    Avg slippage: {summary['execution_quality']['avg_slippage_bps']:.2f} bps")
            
            # Test slippage analysis
            slippage = store.get_slippage_analysis()
            assert slippage['total_fills_analyzed'] == 5, "Should analyze all fills"
            print(f"  ✅ Slippage analysis: {slippage['total_fills_analyzed']} fills")
            
            # Test top symbols
            top_symbols = store.get_top_symbols_by_volume()
            assert len(top_symbols) == 2, "Should have 2 symbols"
            assert top_symbols[0]['symbol'] == 'AAPL', "AAPL should be top by volume"
            print(f"  ✅ Top symbols: {top_symbols[0]['symbol']} (${top_symbols[0]['total_notional']:,.0f})")
            
            print("  ✅ FillStore metrics test passed!")
            
        if os.path.exists(temp_db):
            os.unlink(temp_db)
        return True
        
    except Exception as e:
        print(f"  ❌ FillStore metrics test failed: {e}")
        return False


def test_fill_store_performance():
    """Test FillStore performance with bulk data"""
    print("\n⚡ Testing FillStore Performance...")
    
    try:
        temp_dir = tempfile.gettempdir()
        temp_db = os.path.join(temp_dir, f"test_performance_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
        
        with FillStore(temp_db) as store:
            # Create bulk test data
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            fill_count = 100
            
            start_time = datetime.now()
            
            for i in range(fill_count):
                symbol = symbols[i % len(symbols)]
                fill = Fill(
                    order_id=f"bulk-{i}",
                    symbol=symbol,
                    quantity=100 + i,
                    price=100.0 + (i % 50),
                    filled_at=datetime.now() - timedelta(minutes=i),
                    commission=1.0,
                    slippage_bps=(i % 10) - 5  # Range from -5 to 4 bps
                )
                store.store_fill(fill)
            
            store_time = datetime.now() - start_time
            print(f"  ⏱️  Stored {fill_count} fills in {store_time.total_seconds():.2f} seconds")
            
            # Test retrieval performance
            start_time = datetime.now()
            all_fills = store.get_fills()
            retrieve_time = datetime.now() - start_time
            
            assert len(all_fills) == fill_count, f"Should retrieve all {fill_count} fills"
            print(f"  ⏱️  Retrieved {len(all_fills)} fills in {retrieve_time.total_seconds():.3f} seconds")
            
            # Test analysis performance
            start_time = datetime.now()
            summary = store.get_execution_summary()
            slippage = store.get_slippage_analysis()
            top_symbols = store.get_top_symbols_by_volume()
            analysis_time = datetime.now() - start_time
            
            print(f"  ⏱️  Generated analytics in {analysis_time.total_seconds():.3f} seconds")
            print(f"    Summary: {summary['fills']['total_fills']} fills")
            print(f"    Slippage: {len(slippage['analysis'])} symbol/side combinations")
            print(f"    Top symbols: {len(top_symbols)} symbols")
            
            print("  ✅ FillStore performance test passed!")
            
        if os.path.exists(temp_db):
            os.unlink(temp_db)
        return True
        
    except Exception as e:
        print(f"  ❌ FillStore performance test failed: {e}")
        return False


def main():
    """Run all FillStore tests"""
    print("🚀 Testing FillStore\n")
    
    tests = [
        test_fill_store_basic,
        test_fill_store_filtering,
        test_fill_store_metrics,
        test_fill_store_performance
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All FillStore tests PASSED!")
        return True
    else:
        print("❌ Some FillStore tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)