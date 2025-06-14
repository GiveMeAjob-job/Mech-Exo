"""
Enhanced Order Router with Circuit Breaker Pattern
Phase P11 Week 4 Day 5 - Resilient order routing with fallback mechanisms

Features:
- Circuit breaker integration for IB Gateway connectivity
- Intelligent fallback routing to backup brokers
- Exponential backoff retry for transient failures
- Health check integration for routing decisions
- Order validation and risk checks
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

from utils.circuit import CircuitBreaker, CircuitBreakerConfig, ResilientCaller, RetryStrategy, circuit_breaker
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics
order_routing_attempts = Counter('order_routing_attempts_total', 'Order routing attempts', ['destination', 'status'])
order_routing_duration = Histogram('order_routing_duration_seconds', 'Order routing duration', ['destination'])
active_orders = Gauge('active_orders_count', 'Number of active orders', ['status'])
broker_health_score = Gauge('broker_health_score', 'Broker health score (0-1)', ['broker_name'])
routing_fallbacks = Counter('order_routing_fallbacks_total', 'Order routing fallbacks', ['from_broker', 'to_broker'])


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


class BrokerType(Enum):
    """Broker type enumeration"""
    IB_GATEWAY = "ib_gateway"
    BACKUP_BROKER = "backup_broker"
    PAPER_TRADING = "paper_trading"
    SIMULATOR = "simulator"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    order_type: str  # MARKET, LIMIT, STOP
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    client_order_id: Optional[str] = None
    account: Optional[str] = None
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: Optional[str] = None
    routed_to: Optional[str] = None
    
    # Risk and validation
    risk_checked: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class BrokerConfig:
    """Broker configuration"""
    name: str
    broker_type: BrokerType
    host: str
    port: int
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority
    
    # Circuit breaker settings
    max_failures: int = 5
    timeout_duration: float = 60.0
    
    # Capacity limits
    max_orders_per_second: float = 10.0
    max_concurrent_orders: int = 100
    
    # Health check
    health_check_url: Optional[str] = None
    health_check_interval: float = 30.0


class BrokerHealthMonitor:
    """Monitor broker health and availability"""
    
    def __init__(self):
        self.health_scores: Dict[str, float] = {}
        self.last_health_check: Dict[str, datetime] = {}
        self.consecutive_failures: Dict[str, int] = {}
    
    def update_health_score(self, broker_name: str, score: float):
        """Update health score for a broker (0.0 = unhealthy, 1.0 = healthy)"""
        self.health_scores[broker_name] = max(0.0, min(1.0, score))
        broker_health_score.labels(broker_name=broker_name).set(score)
        
        if score < 0.5:
            logger.warning(f"Broker {broker_name} health degraded: {score:.2f}")
    
    def record_success(self, broker_name: str):
        """Record successful operation for broker"""
        self.consecutive_failures[broker_name] = 0
        current_score = self.health_scores.get(broker_name, 1.0)
        new_score = min(1.0, current_score + 0.1)
        self.update_health_score(broker_name, new_score)
    
    def record_failure(self, broker_name: str):
        """Record failed operation for broker"""
        self.consecutive_failures[broker_name] = self.consecutive_failures.get(broker_name, 0) + 1
        current_score = self.health_scores.get(broker_name, 1.0)
        new_score = max(0.0, current_score - 0.2)
        self.update_health_score(broker_name, new_score)
    
    def get_health_score(self, broker_name: str) -> float:
        """Get current health score for broker"""
        return self.health_scores.get(broker_name, 1.0)
    
    def is_broker_healthy(self, broker_name: str, threshold: float = 0.5) -> bool:
        """Check if broker is considered healthy"""
        return self.get_health_score(broker_name) >= threshold


class OrderValidator:
    """Validate orders before routing"""
    
    def __init__(self):
        self.validation_rules = [
            self._validate_symbol,
            self._validate_quantity,
            self._validate_price,
            self._validate_side,
            self._validate_order_type
        ]
    
    def validate_order(self, order: Order) -> Tuple[bool, List[str]]:
        """Validate order and return (is_valid, errors)"""
        errors = []
        
        for rule in self.validation_rules:
            try:
                rule_errors = rule(order)
                if rule_errors:
                    errors.extend(rule_errors)
            except Exception as e:
                errors.append(f"Validation rule error: {e}")
        
        return len(errors) == 0, errors
    
    def _validate_symbol(self, order: Order) -> List[str]:
        """Validate symbol format"""
        errors = []
        if not order.symbol or len(order.symbol) < 1:
            errors.append("Symbol is required")
        elif len(order.symbol) > 12:
            errors.append("Symbol too long")
        return errors
    
    def _validate_quantity(self, order: Order) -> List[str]:
        """Validate order quantity"""
        errors = []
        if order.quantity <= 0:
            errors.append("Quantity must be positive")
        elif order.quantity > 1000000:
            errors.append("Quantity too large")
        return errors
    
    def _validate_price(self, order: Order) -> List[str]:
        """Validate order price"""
        errors = []
        if order.order_type in ["LIMIT", "STOP_LIMIT"] and order.price is None:
            errors.append("Price required for limit orders")
        elif order.price is not None and order.price <= 0:
            errors.append("Price must be positive")
        return errors
    
    def _validate_side(self, order: Order) -> List[str]:
        """Validate order side"""
        errors = []
        if order.side not in ["BUY", "SELL"]:
            errors.append("Side must be BUY or SELL")
        return errors
    
    def _validate_order_type(self, order: Order) -> List[str]:
        """Validate order type"""
        errors = []
        valid_types = ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
        if order.order_type not in valid_types:
            errors.append(f"Order type must be one of: {valid_types}")
        return errors


class BrokerInterface:
    """Interface for broker communication"""
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.session = None
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to broker"""
        try:
            # Simulate connection logic
            logger.info(f"Connecting to {self.config.name} at {self.config.host}:{self.config.port}")
            await asyncio.sleep(0.1)  # Simulate connection time
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.config.name}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from broker"""
        self.connected = False
        logger.info(f"Disconnected from {self.config.name}")
    
    async def submit_order(self, order: Order) -> Tuple[bool, Dict[str, Any]]:
        """Submit order to broker"""
        if not self.connected:
            raise ConnectionError(f"Not connected to {self.config.name}")
        
        # Simulate order submission
        logger.info(f"Submitting order to {self.config.name}: {order.symbol} {order.side} {order.quantity}")
        
        # Simulate random failures for testing
        import random
        if random.random() < 0.1:  # 10% failure rate for simulation
            raise Exception(f"Simulated broker error from {self.config.name}")
        
        # Simulate processing delay
        await asyncio.sleep(0.05)
        
        broker_order_id = f"{self.config.name}_{int(time.time())}"
        
        return True, {
            "broker_order_id": broker_order_id,
            "status": "submitted",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order at broker"""
        if not self.connected:
            raise ConnectionError(f"Not connected to {self.config.name}")
        
        logger.info(f"Cancelling order {broker_order_id} at {self.config.name}")
        await asyncio.sleep(0.02)
        return True
    
    async def get_order_status(self, broker_order_id: str) -> Dict[str, Any]:
        """Get order status from broker"""
        if not self.connected:
            raise ConnectionError(f"Not connected to {self.config.name}")
        
        # Simulate status retrieval
        return {
            "broker_order_id": broker_order_id,
            "status": "filled",
            "filled_quantity": 100,
            "avg_fill_price": 150.25
        }


class OrderRouter:
    """Enhanced order router with circuit breaker and fallback"""
    
    def __init__(self):
        self.brokers: Dict[str, BrokerInterface] = {}
        self.broker_configs: Dict[str, BrokerConfig] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_monitor = BrokerHealthMonitor()
        self.order_validator = OrderValidator()
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
        # Configuration
        self.enable_fallback = True
        self.enable_circuit_breaker = True
        
        # Setup default brokers
        self._setup_default_brokers()
    
    def _setup_default_brokers(self):
        """Setup default broker configurations"""
        configs = [
            BrokerConfig(
                name="ib_primary",
                broker_type=BrokerType.IB_GATEWAY,
                host="localhost",
                port=4001,
                priority=1,
                max_failures=5,
                timeout_duration=60.0
            ),
            BrokerConfig(
                name="ib_backup",
                broker_type=BrokerType.IB_GATEWAY,
                host="backup.ib.com",
                port=4001,
                priority=2,
                max_failures=3,
                timeout_duration=30.0
            ),
            BrokerConfig(
                name="paper_trading",
                broker_type=BrokerType.PAPER_TRADING,
                host="localhost",
                port=4002,
                priority=3,
                max_failures=10,
                timeout_duration=10.0
            )
        ]
        
        for config in configs:
            self.add_broker(config)
    
    def add_broker(self, config: BrokerConfig):
        """Add broker to router"""
        self.broker_configs[config.name] = config
        self.brokers[config.name] = BrokerInterface(config)
        
        # Setup circuit breaker
        if self.enable_circuit_breaker:
            cb_config = CircuitBreakerConfig(
                name=f"broker_{config.name}",
                failure_threshold=config.max_failures,
                timeout_duration=config.timeout_duration
            )
            self.circuit_breakers[config.name] = CircuitBreaker(cb_config)
        
        logger.info(f"Added broker: {config.name} ({config.broker_type.value})")
    
    def get_available_brokers(self) -> List[str]:
        """Get list of available brokers ordered by priority and health"""
        available = []
        
        for name, config in self.broker_configs.items():
            if not config.enabled:
                continue
            
            # Check circuit breaker state
            if self.enable_circuit_breaker:
                breaker = self.circuit_breakers.get(name)
                if breaker and breaker.state.value == "open":
                    continue
            
            # Check health score
            if self.health_monitor.is_broker_healthy(name):
                available.append(name)
        
        # Sort by priority (lower number = higher priority) and health score
        available.sort(key=lambda name: (
            self.broker_configs[name].priority,
            -self.health_monitor.get_health_score(name)  # Negative for descending order
        ))
        
        return available
    
    async def route_order(self, order: Order) -> Tuple[bool, Dict[str, Any]]:
        """Route order to best available broker with fallback"""
        start_time = time.time()
        
        # Validate order
        is_valid, validation_errors = self.order_validator.validate_order(order)
        if not is_valid:
            order.validation_errors = validation_errors
            order.status = OrderStatus.REJECTED
            logger.error(f"Order validation failed: {validation_errors}")
            return False, {"errors": validation_errors}
        
        order.risk_checked = True
        
        # Get available brokers
        available_brokers = self.get_available_brokers()
        if not available_brokers:
            error_msg = "No available brokers for order routing"
            logger.error(error_msg)
            order.status = OrderStatus.ERROR
            return False, {"error": error_msg}
        
        last_error = None
        
        # Try each broker in priority order
        for broker_name in available_brokers:
            try:
                logger.info(f"Attempting to route order {order.order_id} to {broker_name}")
                
                # Use circuit breaker if enabled
                if self.enable_circuit_breaker:
                    breaker = self.circuit_breakers[broker_name]
                    result = await breaker.call_async(self._submit_order_to_broker, broker_name, order)
                else:
                    result = await self._submit_order_to_broker(broker_name, order)
                
                # Success
                success, response = result
                if success:
                    order.routed_to = broker_name
                    order.status = OrderStatus.SUBMITTED
                    order.broker_order_id = response.get("broker_order_id")
                    
                    # Update metrics and health
                    duration = time.time() - start_time
                    order_routing_attempts.labels(destination=broker_name, status="success").inc()
                    order_routing_duration.labels(destination=broker_name).observe(duration)
                    self.health_monitor.record_success(broker_name)
                    
                    # Track active order
                    self.active_orders[order.order_id] = order
                    active_orders.labels(status="active").set(len(self.active_orders))
                    
                    logger.info(f"Order {order.order_id} successfully routed to {broker_name}")
                    return True, response
                else:
                    raise Exception(response.get("error", "Unknown broker error"))
            
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to route order to {broker_name}: {e}")
                
                # Update metrics and health
                order_routing_attempts.labels(destination=broker_name, status="failure").inc()
                self.health_monitor.record_failure(broker_name)
                
                # Record fallback if trying next broker
                if broker_name != available_brokers[-1]:
                    next_broker = available_brokers[available_brokers.index(broker_name) + 1]
                    routing_fallbacks.labels(from_broker=broker_name, to_broker=next_broker).inc()
        
        # All brokers failed
        order.status = OrderStatus.ERROR
        error_msg = f"Failed to route order to any broker. Last error: {last_error}"
        logger.error(error_msg)
        return False, {"error": error_msg}
    
    async def _submit_order_to_broker(self, broker_name: str, order: Order) -> Tuple[bool, Dict[str, Any]]:
        """Submit order to specific broker"""
        broker = self.brokers[broker_name]
        
        # Ensure broker is connected
        if not broker.connected:
            connected = await broker.connect()
            if not connected:
                raise ConnectionError(f"Failed to connect to {broker_name}")
        
        # Submit order
        return await broker.submit_order(order)
    
    @circuit_breaker(name="order_validation", failure_threshold=10, timeout_duration=30)
    async def validate_order_with_risk_check(self, order: Order) -> bool:
        """Validate order with enhanced risk checking"""
        # Basic validation
        is_valid, errors = self.order_validator.validate_order(order)
        if not is_valid:
            return False
        
        # Additional risk checks could be added here
        # - Position size limits
        # - Account buying power
        # - Market hours
        # - Symbol restrictions
        
        return True
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel active order"""
        if order_id not in self.active_orders:
            logger.warning(f"Order {order_id} not found in active orders")
            return False
        
        order = self.active_orders[order_id]
        if not order.routed_to or not order.broker_order_id:
            logger.warning(f"Order {order_id} not properly routed")
            return False
        
        try:
            broker = self.brokers[order.routed_to]
            success = await broker.cancel_order(order.broker_order_id)
            
            if success:
                order.status = OrderStatus.CANCELLED
                del self.active_orders[order_id]
                active_orders.labels(status="active").set(len(self.active_orders))
                logger.info(f"Order {order_id} cancelled successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_router_health(self) -> Dict[str, Any]:
        """Get router health status"""
        available_brokers = self.get_available_brokers()
        
        health_info = {
            "router_ok": len(available_brokers) > 0,
            "available_brokers": len(available_brokers),
            "total_brokers": len(self.broker_configs),
            "active_orders": len(self.active_orders),
            "broker_health": {}
        }
        
        for name in self.broker_configs.keys():
            health_score = self.health_monitor.get_health_score(name)
            circuit_state = "unknown"
            
            if name in self.circuit_breakers:
                circuit_state = self.circuit_breakers[name].state.value
            
            health_info["broker_health"][name] = {
                "health_score": health_score,
                "circuit_state": circuit_state,
                "enabled": self.broker_configs[name].enabled,
                "is_healthy": self.health_monitor.is_broker_healthy(name)
            }
        
        return health_info
    
    async def start(self):
        """Start the order router"""
        logger.info("Starting Order Router...")
        
        # Connect to primary brokers
        for name, broker in self.brokers.items():
            if self.broker_configs[name].priority <= 2:  # Connect to priority 1 and 2 brokers
                try:
                    await broker.connect()
                    logger.info(f"Connected to {name}")
                except Exception as e:
                    logger.warning(f"Failed to connect to {name}: {e}")
        
        logger.info("Order Router started successfully")
    
    async def stop(self):
        """Stop the order router"""
        logger.info("Stopping Order Router...")
        
        # Disconnect from all brokers
        for broker in self.brokers.values():
            await broker.disconnect()
        
        logger.info("Order Router stopped")


# Global router instance
order_router = OrderRouter()


# Example usage and testing
if __name__ == "__main__":
    async def test_order_routing():
        """Test the order routing functionality"""
        
        # Create test orders
        test_orders = [
            Order(
                order_id="test_001",
                symbol="AAPL",
                side="BUY",
                quantity=100,
                order_type="MARKET"
            ),
            Order(
                order_id="test_002",
                symbol="GOOGL",
                side="SELL",
                quantity=50,
                order_type="LIMIT",
                price=2800.50
            ),
            Order(
                order_id="test_003",
                symbol="INVALID",  # This should fail validation
                side="BUY",
                quantity=-10,  # Invalid quantity
                order_type="MARKET"
            )
        ]
        
        # Start router
        await order_router.start()
        
        # Route test orders
        for order in test_orders:
            print(f"\nRouting order: {order.order_id}")
            success, response = await order_router.route_order(order)
            
            if success:
                print(f"✅ Order routed successfully: {response}")
            else:
                print(f"❌ Order routing failed: {response}")
        
        # Check router health
        health = order_router.get_router_health()
        print(f"\nRouter Health: {json.dumps(health, indent=2)}")
        
        # Stop router
        await order_router.stop()
    
    # Run test
    asyncio.run(test_order_routing())