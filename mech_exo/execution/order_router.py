"""
Order router with pre-trade risk checks and retry logic
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .broker_adapter import BrokerAdapter, BrokerStatus
from .models import Order, Fill, OrderStatus, validate_order, ExecutionError, RiskViolationError
from ..risk import RiskChecker, Portfolio, Position
from ..utils import ConfigManager

logger = logging.getLogger(__name__)


class RoutingDecision(Enum):
    """Order routing decision"""
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    DEFER = "DEFER"
    MODIFY = "MODIFY"


@dataclass
class RoutingResult:
    """Result of order routing decision"""
    decision: RoutingDecision
    original_order: Order
    modified_order: Optional[Order] = None
    rejection_reason: Optional[str] = None
    risk_warnings: List[str] = field(default_factory=list)
    routing_notes: Optional[str] = None


@dataclass
class RetryConfig:
    """Retry configuration for failed orders"""
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    backoff_multiplier: float = 2.0
    retry_on_gateway_error: bool = True
    retry_on_timeout: bool = True
    retry_on_rejection: bool = False  # Usually don't retry rejections


class OrderRouter:
    """
    Order router with pre-trade risk checks and retry logic
    
    Flow:
    1. Validate order format
    2. Pre-trade risk checks
    3. Route to broker
    4. Handle fills and rejections
    5. Retry logic for failures
    """
    
    def __init__(self, broker: BrokerAdapter, risk_checker: RiskChecker, config: Optional[Dict[str, Any]] = None):
        self.broker = broker
        self.risk_checker = risk_checker
        self.config = config or {}
        
        # Retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.config.get('max_retries', 3),
            retry_delay=self.config.get('retry_delay', 1.0),
            backoff_multiplier=self.config.get('backoff_multiplier', 2.0),
            retry_on_gateway_error=self.config.get('retry_on_gateway_error', True),
            retry_on_timeout=self.config.get('retry_on_timeout', True),
            retry_on_rejection=self.config.get('retry_on_rejection', False)
        )
        
        # Order tracking
        self._pending_orders: Dict[str, Order] = {}
        self._retry_attempts: Dict[str, int] = {}
        
        # Callbacks
        self.order_callbacks: List[Callable[[Order], None]] = []
        self.fill_callbacks: List[Callable[[Fill], None]] = []
        self.routing_callbacks: List[Callable[[RoutingResult], None]] = []
        
        # Setup broker callbacks
        self.broker.add_order_callback(self._on_order_update)
        self.broker.add_fill_callback(self._on_fill_update)
        
        # Safety controls
        self.max_daily_orders = self.config.get('max_daily_orders', 100)
        self.max_order_value = self.config.get('max_order_value', 50000)  # $50k
        self.daily_order_count = 0
        self.last_reset_date = datetime.now().date()
        
        logger.info("OrderRouter initialized with retry config: %s", self.retry_config)
    
    async def route_order(self, order: Order) -> RoutingResult:
        """
        Route an order through the complete workflow
        """
        try:
            # Reset daily counter if needed
            self._reset_daily_counters()
            
            logger.info(f"Routing order: {order.symbol} {order.quantity} {order.order_type.value}")
            
            # Step 1: Validate order format
            validation_errors = validate_order(order)
            if validation_errors:
                result = RoutingResult(
                    decision=RoutingDecision.REJECT,
                    original_order=order,
                    rejection_reason=f"Validation errors: {', '.join(validation_errors)}"
                )
                logger.warning(f"Order validation failed: {result.rejection_reason}")
                self._notify_routing_callbacks(result)
                return result
            
            # Step 2: Pre-trade risk checks
            routing_result = await self._pre_trade_risk_check(order)
            
            if routing_result.decision == RoutingDecision.REJECT:
                logger.warning(f"Order rejected by risk check: {routing_result.rejection_reason}")
                self._notify_routing_callbacks(routing_result)
                return routing_result
            
            # Use modified order if available
            final_order = routing_result.modified_order or order
            
            # Step 3: Safety checks
            safety_result = self._safety_checks(final_order)
            if safety_result.decision == RoutingDecision.REJECT:
                logger.warning(f"Order rejected by safety check: {safety_result.rejection_reason}")
                self._notify_routing_callbacks(safety_result)
                return safety_result
            
            # Step 4: Route to broker with retry logic
            execution_result = await self._execute_with_retry(final_order)
            
            # Update routing result with execution outcome
            if execution_result['status'] == 'SUBMITTED':
                routing_result.decision = RoutingDecision.APPROVE
                routing_result.routing_notes = f"Order submitted to broker: {execution_result.get('broker_order_id')}"
                self.daily_order_count += 1
            else:
                routing_result.decision = RoutingDecision.REJECT
                routing_result.rejection_reason = f"Broker execution failed: {execution_result.get('message', 'Unknown error')}"
            
            logger.info(f"Order routing complete: {routing_result.decision.value}")
            self._notify_routing_callbacks(routing_result)
            return routing_result
            
        except Exception as e:
            error_msg = f"Order routing failed: {e}"
            logger.error(error_msg)
            
            result = RoutingResult(
                decision=RoutingDecision.REJECT,
                original_order=order,
                rejection_reason=error_msg
            )
            self._notify_routing_callbacks(result)
            return result
    
    async def _pre_trade_risk_check(self, order: Order) -> RoutingResult:
        """Perform pre-trade risk analysis"""
        try:
            # Get current portfolio state
            portfolio = self.risk_checker.portfolio
            
            # Create hypothetical position
            entry_price = order.limit_price or order.stop_price or 100.0  # Fallback price
            
            # Check if this would be a new position or modify existing
            existing_position = None
            for pos in portfolio.positions:
                if pos.symbol == order.symbol:
                    existing_position = pos
                    break
            
            # Perform risk check for new position
            risk_analysis = self.risk_checker.check_new_position(
                symbol=order.symbol,
                shares=order.quantity,
                price=entry_price,
                sector=getattr(existing_position, 'sector', 'Unknown')
            )
            
            # Analyze results
            if risk_analysis['pre_trade_analysis']['recommendation'] == 'REJECT':
                return RoutingResult(
                    decision=RoutingDecision.REJECT,
                    original_order=order,
                    rejection_reason=f"Risk violation: {', '.join(risk_analysis['pre_trade_analysis']['violations'])}",
                    risk_warnings=risk_analysis['pre_trade_analysis']['violations']
                )
            
            elif risk_analysis['pre_trade_analysis']['recommendation'] == 'APPROVE_WITH_CAUTION':
                return RoutingResult(
                    decision=RoutingDecision.APPROVE,
                    original_order=order,
                    risk_warnings=risk_analysis['pre_trade_analysis'].get('warnings', []),
                    routing_notes="Approved with risk warnings"
                )
            
            # Check for position sizing modifications
            suggested_size = risk_analysis['pre_trade_analysis'].get('suggested_size')
            if suggested_size and abs(suggested_size) != abs(order.quantity):
                # Create modified order with suggested size
                modified_order = Order(
                    symbol=order.symbol,
                    quantity=suggested_size if order.quantity > 0 else -abs(suggested_size),
                    order_type=order.order_type,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    time_in_force=order.time_in_force,
                    strategy=order.strategy,
                    signal_strength=order.signal_strength,
                    notes=f"Size modified by risk check: {order.quantity} -> {suggested_size}"
                )
                
                return RoutingResult(
                    decision=RoutingDecision.MODIFY,
                    original_order=order,
                    modified_order=modified_order,
                    routing_notes=f"Position size modified from {order.quantity} to {suggested_size}"
                )
            
            # Approve order
            return RoutingResult(
                decision=RoutingDecision.APPROVE,
                original_order=order,
                routing_notes="Passed pre-trade risk checks"
            )
            
        except Exception as e:
            logger.error(f"Pre-trade risk check failed: {e}")
            return RoutingResult(
                decision=RoutingDecision.REJECT,
                original_order=order,
                rejection_reason=f"Risk check error: {e}"
            )
    
    def _safety_checks(self, order: Order) -> RoutingResult:
        """Additional safety checks"""
        
        # Check daily order limit
        if self.daily_order_count >= self.max_daily_orders:
            return RoutingResult(
                decision=RoutingDecision.REJECT,
                original_order=order,
                rejection_reason=f"Daily order limit exceeded: {self.daily_order_count}/{self.max_daily_orders}"
            )
        
        # Check order value limit
        if order.estimated_value and order.estimated_value > self.max_order_value:
            return RoutingResult(
                decision=RoutingDecision.REJECT,
                original_order=order,
                rejection_reason=f"Order value ${order.estimated_value:,.0f} exceeds limit ${self.max_order_value:,.0f}"
            )
        
        # Check broker connection
        if not self.broker.is_connected():
            return RoutingResult(
                decision=RoutingDecision.REJECT,
                original_order=order,
                rejection_reason="Broker not connected"
            )
        
        # All safety checks passed
        return RoutingResult(
            decision=RoutingDecision.APPROVE,
            original_order=order,
            routing_notes="Passed safety checks"
        )
    
    async def _execute_with_retry(self, order: Order) -> Dict[str, Any]:
        """Execute order with retry logic"""
        self._pending_orders[order.order_id] = order
        self._retry_attempts[order.order_id] = 0
        
        last_error = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                logger.info(f"Executing order {order.order_id}, attempt {attempt + 1}/{self.retry_config.max_retries + 1}")
                
                # Execute order
                result = await self.broker.place_order(order)
                
                if result['status'] in ['SUBMITTED', 'FILLED']:
                    logger.info(f"Order {order.order_id} successfully submitted")
                    return result
                else:
                    raise ExecutionError(f"Order execution failed: {result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                last_error = e
                self._retry_attempts[order.order_id] = attempt + 1
                
                logger.warning(f"Order execution attempt {attempt + 1} failed: {e}")
                
                # Check if we should retry
                if attempt < self.retry_config.max_retries and self._should_retry(e):
                    delay = self.retry_config.retry_delay * (self.retry_config.backoff_multiplier ** attempt)
                    logger.info(f"Retrying order {order.order_id} in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Order {order.order_id} failed after {attempt + 1} attempts")
                    break
        
        # All retries exhausted
        self._pending_orders.pop(order.order_id, None)
        self._retry_attempts.pop(order.order_id, None)
        
        return {
            'status': 'FAILED',
            'message': f"Order failed after {self.retry_config.max_retries + 1} attempts: {last_error}",
            'last_error': str(last_error)
        }
    
    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error is retryable"""
        error_str = str(error).lower()
        
        # Don't retry validation errors
        if isinstance(error, RiskViolationError):
            return False
        
        # Don't retry if configured not to
        if 'rejected' in error_str and not self.retry_config.retry_on_rejection:
            return False
        
        # Retry on gateway errors if configured
        if 'gateway' in error_str or 'connection' in error_str:
            return self.retry_config.retry_on_gateway_error
        
        # Retry on timeout errors if configured
        if 'timeout' in error_str:
            return self.retry_config.retry_on_timeout
        
        # Default: retry on most errors
        return True
    
    def _reset_daily_counters(self):
        """Reset daily counters if new day"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_order_count = 0
            self.last_reset_date = today
            logger.info("Daily order counter reset")
    
    def _on_order_update(self, order: Order):
        """Handle order updates from broker"""
        logger.info(f"Order update: {order.order_id} -> {order.status.value}")
        
        # Remove from pending if terminal state
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self._pending_orders.pop(order.order_id, None)
            self._retry_attempts.pop(order.order_id, None)
        
        # Notify callbacks
        self._notify_order_callbacks(order)
    
    def _on_fill_update(self, fill: Fill):
        """Handle fill updates from broker"""
        logger.info(f"Fill received: {fill.symbol} {fill.quantity} @ ${fill.price}")
        
        # Notify callbacks
        self._notify_fill_callbacks(fill)
    
    def add_order_callback(self, callback: Callable[[Order], None]):
        """Add order update callback"""
        self.order_callbacks.append(callback)
    
    def add_fill_callback(self, callback: Callable[[Fill], None]):
        """Add fill update callback"""
        self.fill_callbacks.append(callback)
    
    def add_routing_callback(self, callback: Callable[[RoutingResult], None]):
        """Add routing decision callback"""
        self.routing_callbacks.append(callback)
    
    def _notify_order_callbacks(self, order: Order):
        """Notify all order callbacks"""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
    
    def _notify_fill_callbacks(self, fill: Fill):
        """Notify all fill callbacks"""
        for callback in self.fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")
    
    def _notify_routing_callbacks(self, result: RoutingResult):
        """Notify all routing callbacks"""
        for callback in self.routing_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in routing callback: {e}")
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            result = await self.broker.cancel_order(order_id)
            
            # Remove from tracking
            self._pending_orders.pop(order_id, None)
            self._retry_attempts.pop(order_id, None)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return {
                'status': 'ERROR',
                'message': f"Failed to cancel order: {e}"
            }
    
    def get_pending_orders(self) -> List[Order]:
        """Get list of pending orders"""
        return list(self._pending_orders.values())
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            'pending_orders': len(self._pending_orders),
            'daily_order_count': self.daily_order_count,
            'max_daily_orders': self.max_daily_orders,
            'retry_attempts': dict(self._retry_attempts),
            'broker_connected': self.broker.is_connected(),
            'broker_status': self.broker.status.value if hasattr(self.broker, 'status') else 'unknown'
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Check broker connection
            broker_info = self.broker.get_broker_info()
            
            # Check risk checker
            risk_status = self.risk_checker.check()
            
            return {
                'status': 'healthy',
                'broker': {
                    'name': broker_info.broker_name,
                    'status': broker_info.status.value,
                    'connected': self.broker.is_connected()
                },
                'risk_checker': {
                    'status': risk_status['status'].value,
                    'violations': len(risk_status.get('violations', []))
                },
                'pending_orders': len(self._pending_orders),
                'daily_orders': self.daily_order_count,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now()
            }