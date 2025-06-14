"""
Order router with pre-trade risk checks and retry logic
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..risk import RiskChecker
from ..utils.greylist import get_greylist_manager
from ..utils.structured_logging import (
    ExecutionContext,
    ExecutionLogger,
    execution_timer,
    timed_execution,
)
from .allocation import split_order_quantity, is_canary_enabled
from .broker_adapter import BrokerAdapter
from .models import (
    ExecutionError,
    Fill,
    Order,
    OrderStatus,
    RiskViolationError,
    validate_order,
)
from .safety_valve import SafetyValve

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
    modified_order: Order | None = None
    rejection_reason: str | None = None
    risk_warnings: list[str] = field(default_factory=list)
    routing_notes: str | None = None


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

    def __init__(self, broker: BrokerAdapter, risk_checker: RiskChecker, config: dict[str, Any] | None = None) -> None:
        self.broker = broker
        self.risk_checker = risk_checker
        self.config = config or {}

        # Setup structured logging
        self.execution_context = ExecutionContext.create(
            component="order_router",
            account_id=getattr(broker, "account_id", None)
        )
        self.execution_logger = ExecutionLogger(__name__, self.execution_context)

        # Retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.config.get("max_retries", 3),
            retry_delay=self.config.get("retry_delay", 1.0),
            backoff_multiplier=self.config.get("backoff_multiplier", 2.0),
            retry_on_gateway_error=self.config.get("retry_on_gateway_error", True),
            retry_on_timeout=self.config.get("retry_on_timeout", True),
            retry_on_rejection=self.config.get("retry_on_rejection", False)
        )

        # Order tracking
        self._pending_orders: dict[str, Order] = {}
        self._retry_attempts: dict[str, int] = {}

        # Callbacks
        self.order_callbacks: list[Callable[[Order], None]] = []
        self.fill_callbacks: list[Callable[[Fill], None]] = []
        self.routing_callbacks: list[Callable[[RoutingResult], None]] = []

        # Setup broker callbacks
        self.broker.add_order_callback(self._on_order_update)
        self.broker.add_fill_callback(self._on_fill_update)

        # Safety controls
        self.max_daily_orders = self.config.get("max_daily_orders", 100)
        self.max_order_value = self.config.get("max_order_value", 50000)  # $50k
        self.daily_order_count = 0
        self.last_reset_date = datetime.now().date()

        # Initialize safety valve
        safety_config = self.config.get("safety", {})
        self.safety_valve = SafetyValve(broker, safety_config)

        # Track if live trading has been authorized
        self._live_trading_authorized = False

        # Performance tracking
        self._order_start_times: dict[str, float] = {}

        logger.info("OrderRouter initialized with retry config: %s", self.retry_config)

        # Log initialization
        self.execution_logger.system_event(
            system="order_router",
            status="initialized",
            message="OrderRouter initialized",
            max_retries=self.retry_config.max_retries,
            max_daily_orders=self.max_daily_orders,
            max_order_value=self.max_order_value
        )

    @timed_execution("route_order")
    async def route_order(self, order: Order) -> RoutingResult:
        """
        Route an order through the complete workflow.
        
        Args:
            order: The order to route through the system
            
        Returns:
            RoutingResult containing decision, modified order, and routing notes
            
        Raises:
            ExecutionError: If order routing fails due to validation or execution issues
        """
        # Start performance tracking
        self._order_start_times[order.order_id] = time.perf_counter()

        # Log order received
        self.execution_logger.order_event(
            event_type="received",
            order_id=order.order_id,
            symbol=order.symbol,
            message=f"Order received for routing: {order.symbol} {order.quantity} {order.order_type.value}",
            quantity=order.quantity,
            order_type=order.order_type.value,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            strategy=order.strategy,
            signal_strength=order.signal_strength
        )

        try:
            # Reset daily counter if needed
            self._reset_daily_counters()

            logger.info(f"Routing order: {order.symbol} {order.quantity} {order.order_type.value}")

            # Step 0: Kill-switch check (highest priority)
            with execution_timer(self.execution_logger, "killswitch_check", order_id=order.order_id):
                if not self._check_trading_enabled():
                    result = RoutingResult(
                        decision=RoutingDecision.REJECT,
                        original_order=order,
                        rejection_reason="Trading disabled by kill-switch"
                    )
                    self._log_routing_decision(order, result, "killswitch_blocked")
                    self._notify_routing_callbacks(result)
                    return result

            # Step 1: Live trading authorization (if needed)
            with execution_timer(self.execution_logger, "authorization_check", order_id=order.order_id):
                auth_result = await self._check_live_trading_authorization()
                if not auth_result:
                    result = RoutingResult(
                        decision=RoutingDecision.REJECT,
                        original_order=order,
                        rejection_reason="Live trading not authorized or safety valve blocked"
                    )
                    self._log_routing_decision(order, result, "authorization_failed")
                    self._notify_routing_callbacks(result)
                    return result

            # Step 2: Validate order format
            with execution_timer(self.execution_logger, "order_validation", order_id=order.order_id):
                validation_errors = validate_order(order)
                if validation_errors:
                    result = RoutingResult(
                        decision=RoutingDecision.REJECT,
                        original_order=order,
                        rejection_reason=f"Validation errors: {', '.join(validation_errors)}"
                    )
                    self._log_routing_decision(order, result, "validation_failed", validation_errors=validation_errors)
                    self._notify_routing_callbacks(result)
                    return result

            # Step 2.5: Greylist symbol filtering
            with execution_timer(self.execution_logger, "greylist_check", order_id=order.order_id):
                greylist_result = self._check_greylist(order)
                if greylist_result.decision == RoutingDecision.REJECT:
                    self._log_routing_decision(order, greylist_result, "greylist_rejected")
                    self._notify_routing_callbacks(greylist_result)
                    return greylist_result

            # Step 3: Pre-trade risk checks
            with execution_timer(self.execution_logger, "risk_check", order_id=order.order_id):
                routing_result = await self._pre_trade_risk_check(order)

                if routing_result.decision == RoutingDecision.REJECT:
                    self._log_routing_decision(order, routing_result, "risk_rejected")
                    self._notify_routing_callbacks(routing_result)
                    return routing_result

            # Use modified order if available
            final_order = routing_result.modified_order or order
            if routing_result.modified_order:
                self.execution_logger.order_event(
                    event_type="modified",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    message="Order modified by risk check",
                    original_quantity=order.quantity,
                    modified_quantity=final_order.quantity,
                    modification_reason=routing_result.routing_notes
                )

            # Step 4: Safety valve order check
            with execution_timer(self.execution_logger, "safety_valve_check", order_id=order.order_id):
                safety_valve_result = await self.safety_valve.check_order_safety(final_order)
                if not safety_valve_result["approved"]:
                    result = RoutingResult(
                        decision=RoutingDecision.REJECT,
                        original_order=order,
                        rejection_reason=f"Safety valve: {safety_valve_result['reason']}",
                        risk_warnings=safety_valve_result.get("warnings", [])
                    )
                    self._log_routing_decision(order, result, "safety_valve_rejected",
                                             safety_reason=safety_valve_result["reason"])
                    self._notify_routing_callbacks(result)
                    return result

            # Add safety warnings to routing result
            if safety_valve_result.get("warnings"):
                routing_result.risk_warnings.extend(safety_valve_result["warnings"])

            # Step 5: Additional safety checks
            with execution_timer(self.execution_logger, "additional_safety_checks", order_id=order.order_id):
                safety_result = self._safety_checks(final_order)
                if safety_result.decision == RoutingDecision.REJECT:
                    self._log_routing_decision(order, safety_result, "safety_checks_failed")
                    self._notify_routing_callbacks(safety_result)
                    return safety_result

            # Step 6: Check for canary allocation split
            with execution_timer(self.execution_logger, "canary_allocation_check", order_id=order.order_id):
                split_orders = self._handle_canary_allocation(final_order)
            
            # Step 7: Route to broker with retry logic (handle single or split orders)
            with execution_timer(self.execution_logger, "broker_execution", order_id=order.order_id):
                execution_result = await self._execute_orders_with_retry(split_orders)

            # Update routing result with execution outcome
            if execution_result["status"] == "SUBMITTED":
                routing_result.decision = RoutingDecision.APPROVE
                routing_result.routing_notes = f"Order submitted to broker: {execution_result.get('broker_order_id')}"
                self.daily_order_count += 1

                self._log_routing_decision(order, routing_result, "approved",
                                          broker_order_id=execution_result.get("broker_order_id"))
            else:
                routing_result.decision = RoutingDecision.REJECT
                routing_result.rejection_reason = f"Broker execution failed: {execution_result.get('message', 'Unknown error')}"

                self._log_routing_decision(order, routing_result, "broker_execution_failed",
                                          broker_error=execution_result.get("message"))

            logger.info(f"Order routing complete: {routing_result.decision.value}")
            self._notify_routing_callbacks(routing_result)

            # Log final routing performance
            self._log_routing_performance(order)

            return routing_result

        except Exception as e:
            error_msg = f"Order routing failed: {e}"
            logger.error(error_msg)

            result = RoutingResult(
                decision=RoutingDecision.REJECT,
                original_order=order,
                rejection_reason=error_msg
            )

            self.execution_logger.error_event(
                error_type="routing_exception",
                error_message=str(e),
                message="Order routing failed with exception",
                order_id=order.order_id,
                symbol=order.symbol
            )

            self._notify_routing_callbacks(result)
            return result

    async def _pre_trade_risk_check(self, order: Order) -> RoutingResult:
        """
        Perform pre-trade risk analysis.
        
        Args:
            order: Order to analyze for risk violations
            
        Returns:
            RoutingResult with risk analysis decision and any position modifications
        """
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
                sector=getattr(existing_position, "sector", "Unknown")
            )

            # Analyze results
            if risk_analysis["pre_trade_analysis"]["recommendation"] == "REJECT":
                return RoutingResult(
                    decision=RoutingDecision.REJECT,
                    original_order=order,
                    rejection_reason=f"Risk violation: {', '.join(risk_analysis['pre_trade_analysis']['violations'])}",
                    risk_warnings=risk_analysis["pre_trade_analysis"]["violations"]
                )

            if risk_analysis["pre_trade_analysis"]["recommendation"] == "APPROVE_WITH_CAUTION":
                return RoutingResult(
                    decision=RoutingDecision.APPROVE,
                    original_order=order,
                    risk_warnings=risk_analysis["pre_trade_analysis"].get("warnings", []),
                    routing_notes="Approved with risk warnings"
                )

            # Check for position sizing modifications
            suggested_size = risk_analysis["pre_trade_analysis"].get("suggested_size")
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
        """
        Perform additional safety checks beyond risk analysis.
        
        Args:
            order: Order to check for safety violations
            
        Returns:
            RoutingResult indicating if order passes safety checks
        """

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

    async def _execute_with_retry(self, order: Order) -> dict[str, Any]:
        """
        Execute order with configurable retry logic.
        
        Args:
            order: Order to execute with retry mechanism
            
        Returns:
            Dict containing execution status, broker order ID, and any error messages
        """
        self._pending_orders[order.order_id] = order
        self._retry_attempts[order.order_id] = 0

        last_error = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                logger.info(f"Executing order {order.order_id}, attempt {attempt + 1}/{self.retry_config.max_retries + 1}")

                # Execute order
                result = await self.broker.place_order(order)

                if result["status"] in ["SUBMITTED", "FILLED"]:
                    logger.info(f"Order {order.order_id} successfully submitted")
                    return result
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
                logger.error(f"Order {order.order_id} failed after {attempt + 1} attempts")
                break

        # All retries exhausted
        self._pending_orders.pop(order.order_id, None)
        self._retry_attempts.pop(order.order_id, None)

        return {
            "status": "FAILED",
            "message": f"Order failed after {self.retry_config.max_retries + 1} attempts: {last_error}",
            "last_error": str(last_error)
        }

    def _should_retry(self, error: Exception) -> bool:
        """
        Determine if an error condition warrants a retry attempt.
        
        Args:
            error: Exception that occurred during order execution
            
        Returns:
            bool: True if the error is retryable based on configuration
        """
        error_str = str(error).lower()

        # Don't retry validation errors
        if isinstance(error, RiskViolationError):
            return False

        # Don't retry if configured not to
        if "rejected" in error_str and not self.retry_config.retry_on_rejection:
            return False

        # Retry on gateway errors if configured
        if "gateway" in error_str or "connection" in error_str:
            return self.retry_config.retry_on_gateway_error

        # Retry on timeout errors if configured
        if "timeout" in error_str:
            return self.retry_config.retry_on_timeout

        # Default: retry on most errors
        return True

    def _reset_daily_counters(self) -> None:
        """Reset daily counters if new day"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_order_count = 0
            self.last_reset_date = today
            logger.info("Daily order counter reset")
    
    def _check_trading_enabled(self) -> bool:
        """Check if trading is enabled via kill-switch"""
        try:
            from ..cli.killswitch import is_trading_enabled
            enabled = is_trading_enabled()
            
            if not enabled:
                logger.warning("🚨 Trading disabled by kill-switch - blocking order")
                
                # Log structured event
                self.execution_logger.system_event(
                    system="killswitch",
                    status="blocked",
                    message="Order blocked by kill-switch",
                    trading_enabled=False
                )
            
            return enabled
            
        except Exception as e:
            logger.error(f"Kill-switch check failed: {e}")
            # Fail-safe: if kill-switch check fails, allow trading to continue
            # but log the error for investigation
            self.execution_logger.system_event(
                system="killswitch",
                status="error",
                message=f"Kill-switch check failed: {e}",
                trading_enabled=True,
                error=str(e)
            )
            return True

    def _on_order_update(self, order: Order) -> None:
        """Handle order updates from broker"""
        logger.info(f"Order update: {order.order_id} -> {order.status.value}")

        # Log structured order update
        self.execution_logger.order_event(
            event_type="status_update",
            order_id=order.order_id,
            symbol=order.symbol,
            message=f"Order status updated to {order.status.value}",
            status=order.status.value,
            broker_order_id=order.broker_order_id
        )

        # Remove from pending if terminal state
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self._pending_orders.pop(order.order_id, None)
            self._retry_attempts.pop(order.order_id, None)

            # Log order completion performance
            if order.order_id in self._order_start_times:
                self._log_order_completion_performance(order)

        # Notify callbacks
        self._notify_order_callbacks(order)

    def _on_fill_update(self, fill: Fill) -> None:
        """Handle fill updates from broker"""
        logger.info(f"Fill received: {fill.symbol} {fill.quantity} @ ${fill.price}")

        # Log structured fill event
        self.execution_logger.fill_event(
            fill_id=fill.fill_id,
            order_id=fill.order_id,
            symbol=fill.symbol,
            quantity=fill.quantity,
            price=fill.price,
            message=f"Fill received: {fill.symbol} {fill.quantity} @ ${fill.price}",
            commission=fill.commission,
            slippage_bps=fill.slippage_bps,
            exchange=fill.exchange,
            strategy=fill.strategy
        )

        # Log performance metrics if available
        if fill.slippage_bps is not None:
            self.execution_logger.performance_event(
                metric_name="execution_slippage",
                value=fill.slippage_bps,
                unit="basis_points",
                message=f"Execution slippage recorded for {fill.symbol}",
                symbol=fill.symbol,
                order_id=fill.order_id,
                fill_id=fill.fill_id
            )

        if fill.commission > 0:
            self.execution_logger.performance_event(
                metric_name="execution_commission",
                value=fill.commission,
                unit="dollars",
                message=f"Commission recorded for {fill.symbol}",
                symbol=fill.symbol,
                order_id=fill.order_id,
                fill_id=fill.fill_id
            )

        # Notify callbacks
        self._notify_fill_callbacks(fill)

    def _log_routing_decision(self, order: Order, result: RoutingResult, decision_type: str, **extra_context):
        """Log routing decision with structured data"""
        self.execution_logger.order_event(
            event_type=f"routing.{decision_type}",
            order_id=order.order_id,
            symbol=order.symbol,
            message=f"Order routing decision: {result.decision.value}",
            decision=result.decision.value,
            rejection_reason=result.rejection_reason,
            risk_warnings=result.risk_warnings,
            routing_notes=result.routing_notes,
            **extra_context
        )

    def _log_routing_performance(self, order: Order):
        """Log overall routing performance"""
        if order.order_id in self._order_start_times:
            duration_ms = (time.perf_counter() - self._order_start_times[order.order_id]) * 1000

            self.execution_logger.performance_event(
                metric_name="order_routing_duration",
                value=duration_ms,
                unit="milliseconds",
                message=f"Order routing completed in {duration_ms:.1f}ms",
                order_id=order.order_id,
                symbol=order.symbol
            )

            # Clean up tracking
            self._order_start_times.pop(order.order_id, None)

    def _log_order_completion_performance(self, order: Order):
        """Log order completion performance metrics"""
        if order.order_id in self._order_start_times:
            total_duration_ms = (time.perf_counter() - self._order_start_times[order.order_id]) * 1000

            self.execution_logger.performance_event(
                metric_name="order_lifecycle_duration",
                value=total_duration_ms,
                unit="milliseconds",
                message=f"Order lifecycle completed: {order.order_id}",
                order_id=order.order_id,
                symbol=order.symbol,
                final_status=order.status.value,
                broker_order_id=order.broker_order_id
            )

            # Clean up tracking
            self._order_start_times.pop(order.order_id, None)

    def add_order_callback(self, callback: Callable[[Order], None]) -> None:
        """Add order update callback"""
        self.order_callbacks.append(callback)

    def add_fill_callback(self, callback: Callable[[Fill], None]) -> None:
        """Add fill update callback"""
        self.fill_callbacks.append(callback)

    def add_routing_callback(self, callback: Callable[[RoutingResult], None]) -> None:
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

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
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
                "status": "ERROR",
                "message": f"Failed to cancel order: {e}"
            }

    def get_pending_orders(self) -> list[Order]:
        """Get list of pending orders"""
        return list(self._pending_orders.values())

    async def _check_live_trading_authorization(self) -> bool:
        """Check if live trading is authorized"""
        import os

        # Check if we're in live mode
        trading_mode = os.getenv("EXO_MODE", "").lower()
        if trading_mode != "live":
            return True  # Non-live modes don't need authorization

        # If already authorized, return True
        if self._live_trading_authorized:
            return True

        # Request authorization from safety valve
        try:
            authorized = await self.safety_valve.authorize_live_trading("OrderRouter trading session")
            if authorized:
                self._live_trading_authorized = True
                logger.info("Live trading authorized by safety valve")
            else:
                logger.warning("Live trading authorization denied by safety valve")
            return authorized
        except Exception as e:
            logger.error(f"Live trading authorization failed: {e}")
            return False

    def activate_emergency_abort(self, reason: str = "OrderRouter emergency stop") -> None:
        """Activate emergency abort via safety valve"""
        self.safety_valve.activate_emergency_abort(reason)
        self._live_trading_authorized = False  # Reset authorization
        logger.critical(f"Emergency abort activated: {reason}")

    def get_safety_status(self) -> dict[str, Any]:
        """Get safety valve status"""
        return self.safety_valve.get_safety_status()

    def get_routing_stats(self) -> dict[str, Any]:
        """Get routing statistics"""
        return {
            "pending_orders": len(self._pending_orders),
            "daily_order_count": self.daily_order_count,
            "max_daily_orders": self.max_daily_orders,
            "retry_attempts": dict(self._retry_attempts),
            "broker_connected": self.broker.is_connected(),
            "broker_status": self.broker.status.value if hasattr(self.broker, "status") else "unknown",
            "live_trading_authorized": self._live_trading_authorized,
            "safety_valve": self.safety_valve.get_safety_status()
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check"""
        try:
            # Check broker connection
            broker_info = self.broker.get_broker_info()

            # Check risk checker
            risk_status = self.risk_checker.check()

            # Check safety valve status
            safety_status = self.safety_valve.get_safety_status()

            return {
                "status": "healthy",
                "broker": {
                    "name": broker_info.broker_name,
                    "status": broker_info.status.value,
                    "connected": self.broker.is_connected()
                },
                "risk_checker": {
                    "status": risk_status["status"].value,
                    "violations": len(risk_status.get("violations", []))
                },
                "safety_valve": {
                    "mode": safety_status["mode"],
                    "emergency_abort": safety_status["emergency_abort"],
                    "daily_value_remaining": safety_status["daily_value_remaining"]
                },
                "pending_orders": len(self._pending_orders),
                "daily_orders": self.daily_order_count,
                "live_trading_authorized": self._live_trading_authorized,
                "timestamp": datetime.now()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now()
            }

    def _handle_canary_allocation(self, order: Order) -> list[Order]:
        """
        Handle canary allocation by splitting order if enabled.
        
        Args:
            order: Original order to potentially split
            
        Returns:
            List of orders (1 if no split, 2 if split between base and canary)
        """
        try:
            # Check if canary allocation is enabled
            if not is_canary_enabled():
                logger.debug(f"Canary allocation disabled, routing full order to base: {order.symbol}")
                # Ensure base tag is set
                order.tag = "base"
                return [order]
            
            # Check minimum order size for splitting (avoid tiny canary orders)
            min_split_size = 5  # Minimum 5 shares to enable split
            if abs(order.quantity) < min_split_size:
                logger.debug(f"Order too small for canary split ({order.quantity} < {min_split_size}), routing to base")
                order.tag = "base"
                return [order]
            
            # Split the order quantity
            base_qty, canary_qty = split_order_quantity(abs(order.quantity))
            
            # Preserve original sign for buy/sell
            if order.quantity < 0:  # Sell order
                base_qty = -base_qty
                canary_qty = -canary_qty
            
            logger.info(f"Splitting order {order.symbol}: Base={base_qty}, Canary={canary_qty}")
            
            # Create base order (copy original and update)
            base_order = Order(
                symbol=order.symbol,
                quantity=base_qty,
                order_type=order.order_type,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                strategy=order.strategy,
                signal_strength=order.signal_strength,
                notes=f"Base allocation from split order {order.order_id}",
                tag="base"
            )
            
            # Create canary order if quantity > 0
            orders = [base_order]
            
            if canary_qty != 0:
                canary_order = Order(
                    symbol=order.symbol,
                    quantity=canary_qty,
                    order_type=order.order_type,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    time_in_force=order.time_in_force,
                    strategy=order.strategy,
                    signal_strength=order.signal_strength,
                    notes=f"Canary allocation from split order {order.order_id}",
                    tag="ml_canary"
                )
                orders.append(canary_order)
            
            self.execution_logger.order_event(
                event_type="split_allocation",
                order_id=order.order_id,
                symbol=order.symbol,
                message=f"Order split: Base {base_qty}, Canary {canary_qty}",
                original_quantity=order.quantity,
                base_quantity=base_qty,
                canary_quantity=canary_qty,
                orders_created=len(orders)
            )
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to handle canary allocation for {order.symbol}: {e}")
            # Fallback to base allocation
            order.tag = "base"
            return [order]

    async def _execute_orders_with_retry(self, orders: list[Order]) -> dict[str, Any]:
        """
        Execute a list of orders with retry logic.
        
        Args:
            orders: List of orders to execute
            
        Returns:
            Execution result dictionary
        """
        try:
            executed_orders = []
            failed_orders = []
            
            # Execute each order separately
            for order in orders:
                try:
                    execution_result = await self._execute_with_retry(order)
                    
                    if execution_result["status"] == "SUBMITTED":
                        executed_orders.append({
                            "order": order,
                            "result": execution_result
                        })
                        logger.info(f"Successfully submitted {order.tag} order: {order.symbol} {order.quantity}")
                    else:
                        failed_orders.append({
                            "order": order,
                            "result": execution_result
                        })
                        logger.warning(f"Failed to submit {order.tag} order: {order.symbol} {order.quantity}")
                        
                except Exception as e:
                    logger.error(f"Exception executing {order.tag} order {order.symbol}: {e}")
                    failed_orders.append({
                        "order": order,
                        "error": str(e)
                    })
            
            # Determine overall result
            if executed_orders and not failed_orders:
                # All orders succeeded
                return {
                    "status": "SUBMITTED",
                    "executed_orders": len(executed_orders),
                    "failed_orders": 0,
                    "message": f"Successfully submitted {len(executed_orders)} orders"
                }
            elif executed_orders and failed_orders:
                # Partial success
                return {
                    "status": "PARTIAL",
                    "executed_orders": len(executed_orders),
                    "failed_orders": len(failed_orders),
                    "message": f"Partial success: {len(executed_orders)} submitted, {len(failed_orders)} failed"
                }
            else:
                # All failed
                return {
                    "status": "FAILED", 
                    "executed_orders": 0,
                    "failed_orders": len(failed_orders),
                    "message": f"All {len(failed_orders)} orders failed"
                }
                
        except Exception as e:
            logger.error(f"Failed to execute orders: {e}")
            return {
                "status": "FAILED",
                "message": f"Order execution failed: {e}"
            }

    def _check_greylist(self, order: Order) -> RoutingResult:
        """
        Check if order symbol is on the greylist
        
        Args:
            order: Order to check
            
        Returns:
            RoutingResult with decision
        """
        try:
            greylist_manager = get_greylist_manager()
            
            # Check if symbol is greylisted
            if greylist_manager.is_greylisted(order.symbol):
                # Check for emergency override
                override_enabled = greylist_manager.is_override_enabled()
                has_override = order.meta and order.meta.get("graylist_override", False)
                
                if override_enabled and has_override:
                    logger.warning(f"GRAYLIST OVERRIDE - allowing order for {order.symbol} "
                                 f"(override: {has_override})")
                    
                    self.execution_logger.order_event(
                        event_type="greylist_override",
                        order_id=order.order_id,
                        symbol=order.symbol,
                        message=f"Greylist override applied for {order.symbol}",
                        override_flag=has_override
                    )
                    
                    return RoutingResult(
                        decision=RoutingDecision.APPROVE,
                        original_order=order,
                        routing_notes=f"Greylist override applied for {order.symbol}"
                    )
                else:
                    logger.warning(f"GRAYLIST - skipping order {order.symbol}")
                    
                    self.execution_logger.order_event(
                        event_type="greylist_blocked",
                        order_id=order.order_id,
                        symbol=order.symbol,
                        message=f"Order blocked by greylist: {order.symbol}",
                        greylist_symbols=greylist_manager.get_greylist()
                    )
                    
                    return RoutingResult(
                        decision=RoutingDecision.REJECT,
                        original_order=order,
                        rejection_reason=f"Symbol {order.symbol} is on the greylist"
                    )
            
            # Symbol not greylisted, proceed
            return RoutingResult(
                decision=RoutingDecision.APPROVE,
                original_order=order,
                routing_notes=f"Symbol {order.symbol} not on greylist"
            )
            
        except Exception as e:
            logger.error(f"Greylist check failed for {order.symbol}: {e}")
            # On error, allow the order to proceed (fail-safe)
            return RoutingResult(
                decision=RoutingDecision.APPROVE,
                original_order=order,
                routing_notes=f"Greylist check error (allowing): {e}"
            )

    def _safety_checks(self, order: Order) -> RoutingResult:
        """
        Perform additional safety checks on the order
        
        Args:
            order: Order to check
            
        Returns:
            RoutingResult with decision
        """
        # Basic safety checks - order value limits
        if hasattr(self, 'max_order_value') and self.max_order_value:
            order_value = abs(order.quantity * (order.limit_price or 100))  # Estimate value
            if order_value > self.max_order_value:
                return RoutingResult(
                    decision=RoutingDecision.REJECT,
                    original_order=order,
                    rejection_reason=f"Order value ${order_value:,.0f} exceeds limit ${self.max_order_value:,.0f}"
                )
        
        # Daily order count check
        if hasattr(self, 'max_daily_orders') and self.max_daily_orders:
            if self.daily_order_count >= self.max_daily_orders:
                return RoutingResult(
                    decision=RoutingDecision.REJECT,
                    original_order=order,
                    rejection_reason=f"Daily order limit reached: {self.daily_order_count}/{self.max_daily_orders}"
                )
        
        # All safety checks passed
        return RoutingResult(
            decision=RoutingDecision.APPROVE,
            original_order=order,
            routing_notes="All safety checks passed"
        )
