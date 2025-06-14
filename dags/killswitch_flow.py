"""
Kill-Switch Integration Flow

Demonstrates how Prefect flows can check the kill-switch before executing
order-placing tasks. This flow can be used as a template for other flows.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from prefect import flow, task
from prefect.logging import get_run_logger

from mech_exo.cli.killswitch import is_trading_enabled, get_kill_switch_status
from mech_exo.execution.models import Order, OrderType
from mech_exo.execution.order_router import OrderRouter
from mech_exo.execution.broker_adapter import BrokerAdapter

logger = logging.getLogger(__name__)


@task(name="check_kill_switch", retries=2, retry_delay_seconds=5)
def check_kill_switch_task() -> Dict[str, Any]:
    """
    Check kill-switch status before proceeding with trading operations
    
    Returns:
        Dict with kill-switch status and details
        
    Raises:
        RuntimeError: If trading is disabled by kill-switch
    """
    task_logger = get_run_logger()
    task_logger.info("🔍 Checking kill-switch status...")
    
    try:
        # Get detailed status
        status = get_kill_switch_status()
        
        task_logger.info(f"Kill-switch status: {'✅ ENABLED' if status['trading_enabled'] else '🚨 DISABLED'}")
        task_logger.info(f"Reason: {status['reason']}")
        task_logger.info(f"Last modified: {status['timestamp']}")
        
        if not status['trading_enabled']:
            # Trading is disabled - this should halt the flow
            task_logger.error(f"🚨 Trading disabled by kill-switch: {status['reason']}")
            raise RuntimeError(f"Trading disabled by kill-switch: {status['reason']}")
        
        task_logger.info("✅ Kill-switch check passed - trading is enabled")
        return status
        
    except Exception as e:
        task_logger.error(f"❌ Kill-switch check failed: {e}")
        raise


@task(name="validate_market_hours")
def validate_market_hours() -> bool:
    """Check if markets are open (simplified implementation)"""
    task_logger = get_run_logger()
    
    # Simplified check - just check if it's during business hours
    now = datetime.now()
    if now.weekday() >= 5:  # Weekend
        task_logger.warning("Markets closed - weekend")
        return False
    
    if not (9 <= now.hour <= 16):  # Rough market hours
        task_logger.warning("Markets closed - outside trading hours")
        return False
    
    task_logger.info("✅ Markets are open")
    return True


@task(name="create_sample_orders")
def create_sample_orders() -> List[Order]:
    """Create sample orders for testing"""
    task_logger = get_run_logger()
    
    orders = [
        Order(
            symbol="SPY",
            quantity=10,
            order_type=OrderType.MARKET,
            strategy="test_strategy",
            signal_strength=0.7
        ),
        Order(
            symbol="QQQ",
            quantity=5,
            order_type=OrderType.LIMIT,
            limit_price=350.0,
            strategy="test_strategy",
            signal_strength=0.6
        )
    ]
    
    task_logger.info(f"Created {len(orders)} sample orders")
    return orders


@task(name="route_orders_with_killswitch")
def route_orders_with_killswitch(orders: List[Order], dry_run: bool = True) -> Dict[str, Any]:
    """
    Route orders through the order router with kill-switch integration
    
    Args:
        orders: List of orders to route
        dry_run: If True, don't actually send orders to broker
        
    Returns:
        Dictionary with routing results
    """
    task_logger = get_run_logger()
    
    # Double-check kill-switch before routing (defense in depth)
    if not is_trading_enabled():
        task_logger.error("🚨 Kill-switch check failed in routing task")
        raise RuntimeError("Trading disabled by kill-switch")
    
    results = {
        'total_orders': len(orders),
        'routed_orders': 0,
        'rejected_orders': 0,
        'errors': [],
        'routing_results': []
    }
    
    task_logger.info(f"{'[DRY RUN]' if dry_run else '[LIVE]'} Routing {len(orders)} orders...")
    
    if dry_run:
        # In dry run mode, simulate successful routing
        for order in orders:
            task_logger.info(f"[DRY RUN] Would route: {order.symbol} {order.quantity} {order.order_type.value}")
            results['routed_orders'] += 1
            results['routing_results'].append({
                'order_id': order.order_id,
                'symbol': order.symbol,
                'status': 'SIMULATED_SUCCESS',
                'message': 'Order would be routed successfully'
            })
    else:
        # In live mode, actually use the order router
        try:
            # Initialize broker adapter and order router
            # Note: In production, these would be configured with real broker credentials
            broker_config = {
                'mode': 'paper',  # Use paper trading for safety
                'host': 'localhost',
                'port': 7497,
                'client_id': 1
            }
            
            broker = BrokerAdapter(broker_config)
            router = OrderRouter(broker)
            
            for order in orders:
                try:
                    # The order router will check the kill-switch internally
                    routing_result = router.route_order(order)
                    
                    if routing_result.decision.value == 'APPROVE':
                        results['routed_orders'] += 1
                        task_logger.info(f"✅ Order routed: {order.symbol}")
                    else:
                        results['rejected_orders'] += 1
                        task_logger.warning(f"❌ Order rejected: {order.symbol} - {routing_result.rejection_reason}")
                    
                    results['routing_results'].append({
                        'order_id': order.order_id,
                        'symbol': order.symbol,
                        'status': routing_result.decision.value,
                        'message': routing_result.rejection_reason or 'Success'
                    })
                    
                except Exception as e:
                    results['rejected_orders'] += 1
                    results['errors'].append(f"Order {order.symbol}: {str(e)}")
                    task_logger.error(f"❌ Order routing failed for {order.symbol}: {e}")
            
            # Cleanup
            broker.close()
            
        except Exception as e:
            task_logger.error(f"❌ Order routing setup failed: {e}")
            results['errors'].append(f"Routing setup failed: {str(e)}")
    
    task_logger.info(f"Routing complete: {results['routed_orders']} routed, {results['rejected_orders']} rejected")
    return results


@flow(name="killswitch-demo-flow", description="Demonstrates kill-switch integration in Prefect flows")
def killswitch_demo_flow(dry_run: bool = True, skip_market_hours: bool = False) -> Dict[str, Any]:
    """
    Demo flow showing kill-switch integration
    
    Args:
        dry_run: If True, simulate order routing without actual execution
        skip_market_hours: If True, skip market hours validation
        
    Returns:
        Flow execution results
    """
    flow_logger = get_run_logger()
    flow_logger.info(f"🚀 Starting kill-switch demo flow (dry_run={dry_run})")
    
    try:
        # Step 1: Check kill-switch (critical - will fail flow if disabled)
        kill_switch_status = check_kill_switch_task()
        
        # Step 2: Validate market hours (if not skipped)
        if not skip_market_hours:
            market_open = validate_market_hours()
            if not market_open:
                flow_logger.warning("⚠️ Markets closed - flow will continue in demo mode")
        
        # Step 3: Create sample orders
        orders = create_sample_orders()
        
        # Step 4: Route orders (with kill-switch checks)
        routing_results = route_orders_with_killswitch(orders, dry_run=dry_run)
        
        # Step 5: Summary
        flow_results = {
            'flow_status': 'SUCCESS',
            'kill_switch_status': kill_switch_status,
            'orders_processed': routing_results['total_orders'],
            'orders_routed': routing_results['routed_orders'],
            'orders_rejected': routing_results['rejected_orders'],
            'errors': routing_results['errors'],
            'execution_mode': 'DRY_RUN' if dry_run else 'LIVE',
            'timestamp': datetime.now().isoformat()
        }
        
        flow_logger.info(f"✅ Flow completed successfully")
        flow_logger.info(f"   Orders processed: {flow_results['orders_processed']}")
        flow_logger.info(f"   Orders routed: {flow_results['orders_routed']}")
        flow_logger.info(f"   Orders rejected: {flow_results['orders_rejected']}")
        
        return flow_results
        
    except Exception as e:
        flow_logger.error(f"❌ Flow failed: {e}")
        return {
            'flow_status': 'FAILED',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@flow(name="daily-trading-flow", description="Daily trading flow with kill-switch protection")
def daily_trading_flow(symbols: Optional[List[str]] = None, dry_run: bool = True) -> Dict[str, Any]:
    """
    Example daily trading flow with kill-switch integration
    
    Args:
        symbols: List of symbols to trade (default: SPY, QQQ)
        dry_run: If True, simulate trading without actual execution
        
    Returns:
        Flow execution results
    """
    flow_logger = get_run_logger()
    symbols = symbols or ["SPY", "QQQ", "IWM"]
    
    flow_logger.info(f"🌅 Starting daily trading flow for {len(symbols)} symbols")
    
    try:
        # Critical: Check kill-switch first
        kill_switch_status = check_kill_switch_task()
        
        # If we get here, trading is enabled
        flow_logger.info("✅ Kill-switch check passed - proceeding with daily flow")
        
        # Additional checks would go here:
        # - Market hours validation  
        # - Risk checks
        # - Portfolio validation
        # - Signal generation
        # - Position sizing
        
        # For demo, just create simple orders
        orders = []
        for symbol in symbols:
            order = Order(
                symbol=symbol,
                quantity=10,
                order_type=OrderType.MARKET,
                strategy="daily_flow",
                signal_strength=0.5
            )
            orders.append(order)
        
        # Route orders with kill-switch protection
        routing_results = route_orders_with_killswitch(orders, dry_run=dry_run)
        
        flow_logger.info(f"✅ Daily flow completed: {routing_results['routed_orders']}/{routing_results['total_orders']} orders routed")
        
        return {
            'flow_status': 'SUCCESS',
            'symbols_processed': len(symbols),
            'orders_generated': len(orders),
            'routing_results': routing_results,
            'kill_switch_status': kill_switch_status,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        flow_logger.error(f"❌ Daily flow failed: {e}")
        return {
            'flow_status': 'FAILED',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing kill-switch integration flow...")
    
    # Run demo flow in dry-run mode
    result = killswitch_demo_flow(dry_run=True, skip_market_hours=True)
    print(f"Demo flow result: {result['flow_status']}")
    
    # Test daily flow
    daily_result = daily_trading_flow(symbols=["SPY"], dry_run=True)
    print(f"Daily flow result: {daily_result['flow_status']}")
    
    print("✅ Kill-switch integration test completed")