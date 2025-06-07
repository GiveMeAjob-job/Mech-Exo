"""
Broker adapter interface and implementations for order execution
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
from enum import Enum

from .models import Order, Fill, OrderStatus, OrderSide, ExecutionError, GatewayError

logger = logging.getLogger(__name__)


class BrokerStatus(Enum):
    """Broker connection status"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"


@dataclass
class BrokerInfo:
    """Broker connection information"""
    broker_name: str
    status: BrokerStatus
    account_id: Optional[str] = None
    connection_time: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    error_message: Optional[str] = None


class BrokerAdapter(ABC):
    """Base class for broker adapters"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.status = BrokerStatus.DISCONNECTED
        self.order_callbacks: List[Callable] = []
        self.fill_callbacks: List[Callable] = []
        self._orders: Dict[str, Order] = {}
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """Place an order and return status"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass
    
    def add_order_callback(self, callback: Callable[[Order], None]):
        """Add callback for order updates"""
        self.order_callbacks.append(callback)
    
    def add_fill_callback(self, callback: Callable[[Fill], None]):
        """Add callback for fill updates"""
        self.fill_callbacks.append(callback)
    
    def _notify_order_update(self, order: Order):
        """Notify all order callbacks"""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
    
    def _notify_fill_update(self, fill: Fill):
        """Notify all fill callbacks"""
        for callback in self.fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")
    
    @abstractmethod
    def get_broker_info(self) -> BrokerInfo:
        """Get broker connection info"""
        pass
    
    def is_connected(self) -> bool:
        """Check if connected to broker"""
        return self.status == BrokerStatus.CONNECTED


class IBAdapter(BrokerAdapter):
    """Interactive Brokers adapter using ib_insync"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ib = None
        self.client_id = config.get('client_id', 1)
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 4002)  # Paper trading port
        self.account_id = None
        self.connection_time = None
        
        # Order tracking
        self._broker_orders = {}  # broker_order_id -> Order
        self._pending_orders = {}  # order_id -> Order
        
    async def connect(self) -> bool:
        """Connect to IB Gateway"""
        try:
            from ib_insync import IB, util
            
            self.status = BrokerStatus.CONNECTING
            self.ib = IB()
            
            # Set up event handlers
            self.ib.orderStatusEvent += self._on_order_status
            self.ib.execDetailsEvent += self._on_execution
            self.ib.errorEvent += self._on_error
            
            # Connect to IB Gateway
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=10
            )
            
            # Get account info
            accounts = self.ib.managedAccounts()
            if accounts:
                self.account_id = accounts[0]
                logger.info(f"Connected to IB account: {self.account_id}")
            
            self.status = BrokerStatus.CONNECTED
            self.connection_time = datetime.now()
            
            logger.info(f"Successfully connected to IB Gateway at {self.host}:{self.port}")
            return True
            
        except ImportError:
            error_msg = "ib_insync not installed. Run: pip install ib_insync"
            logger.error(error_msg)
            self.status = BrokerStatus.ERROR
            raise GatewayError(error_msg)
            
        except Exception as e:
            error_msg = f"Failed to connect to IB Gateway: {e}"
            logger.error(error_msg)
            self.status = BrokerStatus.ERROR
            raise GatewayError(error_msg)
    
    async def disconnect(self) -> bool:
        """Disconnect from IB Gateway"""
        try:
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
            
            self.status = BrokerStatus.DISCONNECTED
            self.connection_time = None
            logger.info("Disconnected from IB Gateway")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from IB Gateway: {e}")
            return False
    
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """Place order with Interactive Brokers"""
        if not self.is_connected():
            raise GatewayError("Not connected to IB Gateway")
        
        try:
            from ib_insync import Stock, Order as IBOrder
            
            # Create IB contract
            contract = Stock(order.symbol, 'SMART', 'USD')
            
            # Create IB order
            ib_order = self._convert_to_ib_order(order)
            
            # Place order
            trade = self.ib.placeOrder(contract, ib_order)
            
            # Update order with broker info
            order.broker_order_id = str(trade.order.orderId)
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            
            # Track the order
            self._orders[order.order_id] = order
            self._broker_orders[order.broker_order_id] = order
            self._pending_orders[order.order_id] = order
            
            logger.info(f"Placed order {order.order_id} with IB, broker order ID: {order.broker_order_id}")
            
            # Notify callbacks
            self._notify_order_update(order)
            
            return {
                'status': 'SUBMITTED',
                'broker_order_id': order.broker_order_id,
                'message': 'Order successfully submitted to IB'
            }
            
        except Exception as e:
            error_msg = f"Failed to place order {order.order_id}: {e}"
            logger.error(error_msg)
            
            order.status = OrderStatus.REJECTED
            self._notify_order_update(order)
            
            raise ExecutionError(error_msg)
    
    def _convert_to_ib_order(self, order: Order) -> 'IBOrder':
        """Convert our Order to IB Order"""
        from ib_insync import Order as IBOrder
        
        ib_order = IBOrder()
        
        # Basic order info
        ib_order.action = 'BUY' if order.quantity > 0 else 'SELL'
        ib_order.totalQuantity = abs(order.quantity)
        ib_order.orderType = order.order_type.value
        
        # Order type specific fields
        if order.limit_price:
            ib_order.lmtPrice = order.limit_price
        
        if order.stop_price:
            ib_order.auxPrice = order.stop_price
        
        # Time in force
        ib_order.tif = order.time_in_force
        
        # Bracket order handling
        if order.parent_order_id:
            # This is a child order of a bracket
            parent_order = self._orders.get(order.parent_order_id)
            if parent_order and parent_order.broker_order_id:
                ib_order.parentId = int(parent_order.broker_order_id)
        
        return ib_order
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        if not self.is_connected():
            raise GatewayError("Not connected to IB Gateway")
        
        order = self._orders.get(order_id)
        if not order:
            raise ExecutionError(f"Order {order_id} not found")
        
        if not order.broker_order_id:
            raise ExecutionError(f"Order {order_id} has no broker order ID")
        
        try:
            # Find the trade
            trades = [t for t in self.ib.trades() if str(t.order.orderId) == order.broker_order_id]
            if not trades:
                raise ExecutionError(f"Trade not found for order {order_id}")
            
            trade = trades[0]
            self.ib.cancelOrder(trade.order)
            
            logger.info(f"Cancelled order {order_id} (broker ID: {order.broker_order_id})")
            
            return {
                'status': 'CANCELLED',
                'message': f'Order {order_id} cancelled'
            }
            
        except Exception as e:
            error_msg = f"Failed to cancel order {order_id}: {e}"
            logger.error(error_msg)
            raise ExecutionError(error_msg)
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status from IB"""
        order = self._orders.get(order_id)
        if not order:
            raise ExecutionError(f"Order {order_id} not found")
        
        try:
            # Get trade status from IB
            if order.broker_order_id:
                trades = [t for t in self.ib.trades() if str(t.order.orderId) == order.broker_order_id]
                if trades:
                    trade = trades[0]
                    return {
                        'order_id': order_id,
                        'broker_order_id': order.broker_order_id,
                        'status': self._convert_ib_status(trade.orderStatus.status),
                        'filled_quantity': trade.orderStatus.filled,
                        'remaining_quantity': trade.orderStatus.remaining,
                        'avg_fill_price': trade.orderStatus.avgFillPrice if trade.orderStatus.avgFillPrice > 0 else None
                    }
            
            # Return current order status
            return {
                'order_id': order_id,
                'status': order.status.value,
                'broker_order_id': order.broker_order_id
            }
            
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return {
                'order_id': order_id,
                'status': 'ERROR',
                'error': str(e)
            }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions from IB"""
        if not self.is_connected():
            raise GatewayError("Not connected to IB Gateway")
        
        try:
            positions = []
            for position in self.ib.positions():
                if position.position != 0:  # Only non-zero positions
                    positions.append({
                        'symbol': position.contract.symbol,
                        'position': position.position,
                        'avg_cost': position.avgCost,
                        'market_price': position.marketPrice,
                        'market_value': position.marketValue,
                        'unrealized_pnl': position.unrealizedPNL,
                        'account': position.account
                    })
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise ExecutionError(f"Failed to get positions: {e}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information from IB"""
        if not self.is_connected():
            raise GatewayError("Not connected to IB Gateway")
        
        try:
            account_values = self.ib.accountValues()
            
            # Extract key account values
            account_info = {
                'account_id': self.account_id,
                'timestamp': datetime.now()
            }
            
            for value in account_values:
                if value.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower', 'GrossPositionValue']:
                    account_info[value.tag.lower()] = float(value.value) if value.value else 0.0
            
            return account_info
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise ExecutionError(f"Failed to get account info: {e}")
    
    def _on_order_status(self, trade):
        """Handle order status updates from IB"""
        try:
            broker_order_id = str(trade.order.orderId)
            order = self._broker_orders.get(broker_order_id)
            
            if not order:
                logger.warning(f"Received status update for unknown order: {broker_order_id}")
                return
            
            # Update order status
            old_status = order.status
            order.status = self._convert_ib_status(trade.orderStatus.status)
            
            if old_status != order.status:
                logger.info(f"Order {order.order_id} status changed: {old_status.value} -> {order.status.value}")
                self._notify_order_update(order)
            
            # Remove from pending if filled or cancelled
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                self._pending_orders.pop(order.order_id, None)
            
        except Exception as e:
            logger.error(f"Error processing order status update: {e}")
    
    def _on_execution(self, trade, fill):
        """Handle execution/fill updates from IB"""
        try:
            broker_order_id = str(trade.order.orderId)
            order = self._broker_orders.get(broker_order_id)
            
            if not order:
                logger.warning(f"Received fill for unknown order: {broker_order_id}")
                return
            
            # Create Fill object
            fill_obj = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                quantity=fill.execution.shares if trade.order.action == 'BUY' else -fill.execution.shares,
                price=fill.execution.price,
                filled_at=datetime.now(),
                broker_order_id=broker_order_id,
                broker_fill_id=fill.execution.execId,
                exchange=fill.execution.exchange,
                commission=fill.commissionReport.commission if fill.commissionReport else 0.0,
                strategy=order.strategy,
                notes=f"IB execution: {fill.execution.execId}"
            )
            
            logger.info(f"Fill received: {fill_obj.symbol} {fill_obj.quantity} @ ${fill_obj.price}")
            
            # Notify callbacks
            self._notify_fill_update(fill_obj)
            
        except Exception as e:
            logger.error(f"Error processing execution update: {e}")
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle error messages from IB"""
        logger.error(f"IB Error {errorCode}: {errorString} (reqId: {reqId})")
        
        # Handle specific error cases
        if errorCode in [502, 503, 504]:  # Connection errors
            self.status = BrokerStatus.ERROR
    
    def _convert_ib_status(self, ib_status: str) -> OrderStatus:
        """Convert IB order status to our OrderStatus"""
        status_map = {
            'Submitted': OrderStatus.SUBMITTED,
            'Filled': OrderStatus.FILLED,
            'Cancelled': OrderStatus.CANCELLED,
            'PendingCancel': OrderStatus.SUBMITTED,
            'PendingSubmit': OrderStatus.PENDING,
            'Inactive': OrderStatus.INACTIVE,
            'ApiCancelled': OrderStatus.CANCELLED,
            'ApiPending': OrderStatus.PENDING
        }
        
        return status_map.get(ib_status, OrderStatus.PENDING)
    
    def get_broker_info(self) -> BrokerInfo:
        """Get broker connection info"""
        return BrokerInfo(
            broker_name="Interactive Brokers",
            status=self.status,
            account_id=self.account_id,
            connection_time=self.connection_time,
            last_heartbeat=datetime.now() if self.is_connected() else None,
            error_message=None
        )


class StubBroker(BrokerAdapter):
    """Stub broker for testing and CI"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.simulate_fills = config.get('simulate_fills', True)
        self.fill_delay = config.get('fill_delay', 1.0)  # seconds
        self.reject_probability = config.get('reject_probability', 0.0)  # 0-1
        self.account_id = "STUB_ACCOUNT"
        self._positions = []
        self._account_nav = config.get('initial_nav', 100000.0)
        
    async def connect(self) -> bool:
        """Simulate connection"""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.status = BrokerStatus.CONNECTED
        self.connection_time = datetime.now()
        logger.info("Connected to StubBroker")
        return True
    
    async def disconnect(self) -> bool:
        """Simulate disconnection"""
        self.status = BrokerStatus.DISCONNECTED
        self.connection_time = None
        logger.info("Disconnected from StubBroker")
        return True
    
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """Simulate order placement"""
        import random
        
        # Simulate rejection
        if random.random() < self.reject_probability:
            order.status = OrderStatus.REJECTED
            self._notify_order_update(order)
            return {
                'status': 'REJECTED',
                'message': 'Simulated rejection'
            }
        
        # Accept order
        order.broker_order_id = f"STUB_{order.order_id[:8]}"
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        
        self._orders[order.order_id] = order
        self._notify_order_update(order)
        
        # Simulate fill if enabled
        if self.simulate_fills:
            asyncio.create_task(self._simulate_fill(order))
        
        return {
            'status': 'SUBMITTED',
            'broker_order_id': order.broker_order_id,
            'message': 'Order submitted to StubBroker'
        }
    
    async def _simulate_fill(self, order: Order):
        """Simulate order fill after delay"""
        await asyncio.sleep(self.fill_delay)
        
        # Update order status
        order.status = OrderStatus.FILLED
        self._notify_order_update(order)
        
        # Create simulated fill
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=order.quantity,
            price=order.limit_price or 100.0,  # Use limit price or default
            filled_at=datetime.now(),
            broker_order_id=order.broker_order_id,
            broker_fill_id=f"FILL_{order.order_id[:8]}",
            exchange="STUB",
            commission=1.0,  # $1 commission
            strategy=order.strategy,
            notes="Simulated fill"
        )
        
        logger.info(f"Simulated fill: {fill.symbol} {fill.quantity} @ ${fill.price}")
        self._notify_fill_update(fill)
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Simulate order cancellation"""
        order = self._orders.get(order_id)
        if order:
            order.status = OrderStatus.CANCELLED
            self._notify_order_update(order)
        
        return {
            'status': 'CANCELLED',
            'message': f'Order {order_id} cancelled in StubBroker'
        }
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get simulated order status"""
        order = self._orders.get(order_id)
        if not order:
            raise ExecutionError(f"Order {order_id} not found")
        
        return {
            'order_id': order_id,
            'status': order.status.value,
            'broker_order_id': order.broker_order_id
        }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get simulated positions"""
        return self._positions.copy()
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get simulated account info"""
        return {
            'account_id': self.account_id,
            'netliquidation': self._account_nav,
            'totalcashvalue': self._account_nav * 0.8,
            'buyingpower': self._account_nav * 1.5,
            'grosspositionvalue': self._account_nav * 0.2,
            'timestamp': datetime.now()
        }
    
    def get_broker_info(self) -> BrokerInfo:
        """Get stub broker info"""
        return BrokerInfo(
            broker_name="StubBroker",
            status=self.status,
            account_id=self.account_id,
            connection_time=self.connection_time,
            last_heartbeat=datetime.now() if self.is_connected() else None,
            error_message=None
        )


def create_broker_adapter(broker_type: str, config: Dict[str, Any]) -> BrokerAdapter:
    """Factory function to create broker adapters"""
    if broker_type.lower() == 'ib':
        return IBAdapter(config)
    elif broker_type.lower() == 'stub':
        return StubBroker(config)
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")