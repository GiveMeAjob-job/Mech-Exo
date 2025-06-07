"""
Comprehensive integration tests for execution engine
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from mech_exo.execution.models import create_market_order, create_limit_order, OrderStatus
from mech_exo.execution.order_router import OrderRouter
from mech_exo.execution.fill_store import FillStore
from mech_exo.risk import RiskChecker, Portfolio
from tests.stubs.broker_stub import EnhancedStubBroker


@pytest.mark.execution
class TestExecutionIntegration:
    """Integration tests for complete execution pipeline"""
    
    @pytest.fixture
    def temp_fill_store(self):
        """Create temporary fill store for testing"""
        temp_dir = tempfile.gettempdir()
        temp_db = os.path.join(temp_dir, f"test_execution_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
        
        store = FillStore(temp_db)
        yield store
        
        store.close()
        if os.path.exists(temp_db):
            os.unlink(temp_db)
    
    @pytest.fixture
    def stub_broker_config(self):
        """Configuration for stub broker"""
        return {
            'simulate_fills': True,
            'fill_delay_ms': 50,  # Fast fills for testing
            'reject_probability': 0.0,
            'write_to_fill_store': True,
            'initial_nav': 100000.0
        }
    
    @pytest.fixture
    def mock_risk_checker(self):
        """Mock risk checker that approves all trades"""
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
            yield risk_checker
            risk_checker.close()
    
    @pytest.mark.asyncio
    async def test_stub_broker_basic_functionality(self, stub_broker_config):
        """Test basic StubBroker functionality"""
        broker = EnhancedStubBroker(stub_broker_config)
        
        try:
            # Test connection
            connected = await broker.connect()
            assert connected, "Should connect successfully"
            assert broker.is_connected(), "Should report connected"
            
            # Test account info
            account_info = await broker.get_account_info()
            assert account_info['account_id'].startswith('STUB_ACCOUNT_'), "Should have stub account ID"
            assert account_info['netliquidation'] == 100000.0, "Should have correct NAV"
            
            # Test order placement
            order = create_market_order("AAPL", 100, strategy="test")
            
            result = await broker.place_order(order)
            assert result['status'] == 'SUBMITTED', "Should submit order successfully"
            assert order.broker_order_id is not None, "Should assign broker order ID"
            
            # Wait for fill
            await asyncio.sleep(0.1)
            
            # Check order status
            status = await broker.get_order_status(order.order_id)
            assert status['status'] == 'FILLED', "Order should be filled"
            
            # Check positions
            positions = await broker.get_positions()
            assert len(positions) == 1, "Should have one position"
            assert positions[0]['symbol'] == 'AAPL', "Should be AAPL position"
            assert positions[0]['position'] == 100, "Should have 100 shares"
            
            print("✅ StubBroker basic functionality test passed")
            
        finally:
            await broker.disconnect()
    
    @pytest.mark.asyncio
    async def test_order_router_with_stub_broker(self, stub_broker_config, mock_risk_checker):
        """Test OrderRouter with StubBroker integration"""
        broker = EnhancedStubBroker(stub_broker_config)
        
        try:
            await broker.connect()
            
            # Create OrderRouter
            router_config = {
                'max_retries': 2,
                'retry_delay': 0.05,
                'max_daily_orders': 100,
                'max_order_value': 50000
            }
            router = OrderRouter(broker, mock_risk_checker, router_config)
            
            # Track callbacks
            order_updates = []
            fill_updates = []
            routing_results = []
            
            def on_order_update(order):
                order_updates.append(order)
            
            def on_fill_update(fill):
                fill_updates.append(fill)
            
            def on_routing_result(result):
                routing_results.append(result)
            
            router.add_order_callback(on_order_update)
            router.add_fill_callback(on_fill_update)
            router.add_routing_callback(on_routing_result)
            
            # Route order
            order = create_market_order("GOOGL", 50, strategy="momentum")
            routing_result = await router.route_order(order)
            
            assert routing_result.decision.value == 'APPROVE', "Should approve order"
            assert len(routing_results) == 1, "Should have routing result"
            
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
            
            print("✅ OrderRouter with StubBroker test passed")
            
        finally:
            await broker.disconnect()
    
    @pytest.mark.asyncio
    async def test_fill_store_integration(self, temp_fill_store, stub_broker_config):
        """Test FillStore integration with StubBroker"""
        # Configure broker to use specific fill store
        broker_config = stub_broker_config.copy()
        broker_config['fill_store_path'] = temp_fill_store.db_path
        
        broker = EnhancedStubBroker(broker_config)
        
        try:
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
            
            # Wait for fills
            await asyncio.sleep(0.2)
            
            # Check FillStore has recorded fills
            fills = temp_fill_store.get_fills()
            assert len(fills) >= 3, "Should have at least 3 fills in store"
            
            # Check fill details
            symbols_filled = [fill.symbol for fill in fills]
            assert 'AAPL' in symbols_filled, "Should have AAPL fill"
            assert 'GOOGL' in symbols_filled, "Should have GOOGL fill"
            assert 'MSFT' in symbols_filled, "Should have MSFT fill"
            
            # Test daily metrics
            today = datetime.now().date()
            daily_metrics = temp_fill_store.get_daily_metrics(today)
            
            assert daily_metrics['fills']['total_fills'] >= 3, "Should count today's fills"
            assert daily_metrics['fills']['symbols_traded'] >= 3, "Should count unique symbols"
            
            print("✅ FillStore integration test passed")
            
        finally:
            await broker.disconnect()
    
    @pytest.mark.asyncio
    async def test_rejection_scenarios(self, stub_broker_config, mock_risk_checker):
        """Test order rejection scenarios"""
        broker = EnhancedStubBroker(stub_broker_config)
        
        try:
            await broker.connect()
            
            # Add specific rejection
            broker.add_rejection("REJECT_ME", "Symbol not allowed for testing")
            
            router = OrderRouter(broker, mock_risk_checker)
            
            # Try to place rejected order
            rejected_order = create_market_order("REJECT_ME", 100, strategy="test")
            routing_result = await router.route_order(rejected_order)
            
            assert routing_result.decision.value == 'REJECT', "Should reject the order"
            assert "Symbol not allowed" in routing_result.rejection_reason, "Should have rejection reason"
            
            # Test with normal order
            broker.clear_rejections()
            normal_order = create_market_order("AAPL", 100, strategy="test")
            routing_result = await router.route_order(normal_order)
            
            assert routing_result.decision.value == 'APPROVE', "Should approve normal order"
            
            print("✅ Rejection scenarios test passed")
            
        finally:
            await broker.disconnect()
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_orders(self, stub_broker_config, mock_risk_checker):
        """Test handling multiple concurrent orders"""
        broker = EnhancedStubBroker(stub_broker_config)
        
        try:
            await broker.connect()
            router = OrderRouter(broker, mock_risk_checker)
            
            # Create multiple orders
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY']
            orders = [create_market_order(symbol, 50, strategy="concurrent_test") for symbol in symbols]
            
            # Submit all orders concurrently
            routing_tasks = [router.route_order(order) for order in orders]
            routing_results = await asyncio.gather(*routing_tasks)
            
            # Check all orders were processed
            assert len(routing_results) == 5, "Should process all orders"
            
            approved_count = sum(1 for result in routing_results if result.decision.value == 'APPROVE')
            assert approved_count == 5, "Should approve all orders"
            
            # Wait for fills
            await asyncio.sleep(0.2)
            
            # Check broker state
            positions = await broker.get_positions()
            trading_summary = broker.get_trading_summary()
            
            assert len(positions) == 5, "Should have 5 positions"
            assert trading_summary['filled_orders'] == 5, "Should have 5 filled orders"
            
            print("✅ Multiple concurrent orders test passed")
            
        finally:
            await broker.disconnect()
    
    @pytest.mark.asyncio
    async def test_risk_integration(self, stub_broker_config):
        """Test risk checker integration in order routing"""
        broker = EnhancedStubBroker(stub_broker_config)
        
        try:
            await broker.connect()
            
            # Create risk checker that rejects large positions
            portfolio = Portfolio(100000)
            
            with patch('mech_exo.risk.checker.ConfigManager'), \
                 patch('mech_exo.risk.checker.DataStorage'):
                
                risk_checker = RiskChecker(portfolio)
                
                # Mock to reject large orders
                def mock_risk_check(symbol, shares, price, sector):
                    if abs(shares) * price > 20000:  # > $20k orders rejected
                        return {
                            'pre_trade_analysis': {
                                'recommendation': 'REJECT',
                                'violations': ['Position too large'],
                                'warnings': []
                            }
                        }
                    else:
                        return {
                            'pre_trade_analysis': {
                                'recommendation': 'APPROVE',
                                'violations': [],
                                'warnings': []
                            }
                        }
                
                risk_checker.check_new_position = Mock(side_effect=mock_risk_check)
                
                router = OrderRouter(broker, risk_checker)
                
                # Test small order (should pass)
                small_order = create_market_order("AAPL", 50, strategy="small_test")  # ~$8k
                broker.set_market_price("AAPL", 160.0)
                
                small_result = await router.route_order(small_order)
                assert small_result.decision.value == 'APPROVE', "Should approve small order"
                
                # Test large order (should be rejected by risk)
                large_order = create_market_order("AAPL", 200, strategy="large_test")  # ~$32k
                
                large_result = await router.route_order(large_order)
                assert large_result.decision.value == 'REJECT', "Should reject large order"
                assert 'Position too large' in large_result.rejection_reason, "Should have risk violation"
                
                print("✅ Risk integration test passed")
                
                risk_checker.close()
                
        finally:
            await broker.disconnect()
    
    @pytest.mark.asyncio
    async def test_end_to_end_trading_session(self, temp_fill_store, stub_broker_config):
        """Test complete end-to-end trading session"""
        # Configure for comprehensive test
        broker_config = stub_broker_config.copy()
        broker_config['fill_store_path'] = temp_fill_store.db_path
        broker_config['simulate_slippage'] = True
        broker_config['price_movement'] = True
        
        broker = EnhancedStubBroker(broker_config)
        
        try:
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
                    # Morning batch
                    create_market_order("SPY", 100, strategy="momentum"),
                    create_limit_order("QQQ", 50, 380.0, strategy="mean_revert"),
                    create_market_order("IWM", 75, strategy="breakout"),
                    
                    # Afternoon batch
                    create_market_order("SPY", -50, strategy="profit_taking"),  # Partial close
                    create_market_order("AAPL", 100, strategy="new_position")
                ]
                
                session_results = []
                
                for i, order in enumerate(session_orders):
                    print(f"Processing order {i+1}: {order.symbol} {order.quantity}")
                    
                    result = await router.route_order(order)
                    session_results.append(result)
                    
                    # Small delay between orders
                    await asyncio.sleep(0.05)
                
                # Wait for all fills
                await asyncio.sleep(0.3)
                
                # Analyze session results
                approved_orders = [r for r in session_results if r.decision.value == 'APPROVE']
                assert len(approved_orders) == 5, "Should approve all session orders"
                
                # Check broker state
                final_positions = await broker.get_positions()
                account_info = await broker.get_account_info()
                trading_summary = broker.get_trading_summary()
                
                # Verify positions make sense
                spy_position = next((p for p in final_positions if p['symbol'] == 'SPY'), None)
                assert spy_position is not None, "Should have SPY position"
                assert spy_position['position'] == 50, "SPY position should be 50 (100 - 50)"
                
                # Check fill store
                session_fills = temp_fill_store.get_fills()
                assert len(session_fills) == 5, "Should have 5 fills recorded"
                
                # Check daily metrics
                today_metrics = temp_fill_store.get_daily_metrics(datetime.now().date())
                assert today_metrics['fills']['total_fills'] == 5, "Daily metrics should count 5 fills"
                assert today_metrics['orders']['fill_rate'] == 1.0, "Should have 100% fill rate"
                
                print("📊 Session Summary:")
                print(f"  Orders: {trading_summary['total_orders']}")
                print(f"  Fills: {trading_summary['filled_orders']}")
                print(f"  Positions: {len(final_positions)}")
                print(f"  Account NAV: ${account_info['netliquidation']:,.2f}")
                print(f"  Total Fees: ${today_metrics['costs']['total_fees']:.2f}")
                
                print("✅ End-to-end trading session test passed")
                
                risk_checker.close()
                
        finally:
            await broker.disconnect()


@pytest.mark.execution
class TestStubBrokerAdvanced:
    """Advanced tests for StubBroker functionality"""
    
    @pytest.mark.asyncio
    async def test_market_simulation(self):
        """Test market price simulation features"""
        config = {
            'price_movement': True,
            'volatility': 0.05,  # 5% volatility
            'simulate_slippage': True,
            'avg_slippage_bps': 2.0
        }
        
        broker = EnhancedStubBroker(config)
        
        try:
            await broker.connect()
            
            # Set initial price
            broker.set_market_price("TEST", 100.0)
            initial_price = broker.get_market_price("TEST")
            assert initial_price == 100.0, "Should set initial price"
            
            # Place multiple orders to see price movement
            for i in range(5):
                order = create_market_order("TEST", 10, strategy=f"test_{i}")
                await broker.place_order(order)
                await asyncio.sleep(0.01)  # Small delay
            
            # Check price has potentially moved
            final_price = broker.get_market_price("TEST")
            print(f"Price movement: ${initial_price:.2f} -> ${final_price:.2f}")
            
            # Price should be positive and reasonable
            assert final_price > 0, "Price should remain positive"
            
            print("✅ Market simulation test passed")
            
        finally:
            await broker.disconnect()
    
    @pytest.mark.asyncio
    async def test_latency_simulation(self):
        """Test latency simulation"""
        config = {
            'simulate_latency': True,
            'base_latency_ms': 50,
            'latency_variance_ms': 20
        }
        
        broker = EnhancedStubBroker(config)
        
        try:
            await broker.connect()
            
            # Measure order placement time
            start_time = datetime.now()
            
            order = create_market_order("LATENCY_TEST", 100)
            await broker.place_order(order)
            
            end_time = datetime.now()
            elapsed_ms = (end_time - start_time).total_seconds() * 1000
            
            # Should have some latency (at least base latency)
            assert elapsed_ms >= 30, f"Should have latency, got {elapsed_ms:.1f}ms"
            
            print(f"Order latency: {elapsed_ms:.1f}ms")
            print("✅ Latency simulation test passed")
            
        finally:
            await broker.disconnect()
    
    @pytest.mark.asyncio
    async def test_commission_and_slippage(self):
        """Test commission and slippage calculations"""
        config = {
            'simulate_slippage': True,
            'avg_slippage_bps': 1.5,
            'slippage_variance_bps': 1.0
        }
        
        broker = EnhancedStubBroker(config)
        
        try:
            await broker.connect()
            
            fills_received = []
            
            def on_fill(fill):
                fills_received.append(fill)
            
            broker.add_fill_callback(on_fill)
            
            # Place order with known price
            broker.set_market_price("COMMISSION_TEST", 100.0)
            order = create_limit_order("COMMISSION_TEST", 100, 100.0, strategy="commission_test")
            
            await broker.place_order(order)
            await asyncio.sleep(0.1)
            
            assert len(fills_received) == 1, "Should receive fill"
            
            fill = fills_received[0]
            
            # Check commission
            assert fill.commission > 0, "Should have commission"
            
            # Check slippage calculation if reference price was set
            if fill.slippage_bps is not None:
                print(f"Slippage: {fill.slippage_bps:.2f} bps")
            
            print(f"Commission: ${fill.commission:.2f}")
            print(f"Fill price: ${fill.price:.2f}")
            print("✅ Commission and slippage test passed")
            
        finally:
            await broker.disconnect()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-m", "execution"])