"""
Enhanced StubBroker for comprehensive testing and CI
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid
import random

from mech_exo.execution.broker_adapter import BrokerAdapter, BrokerStatus, BrokerInfo
from mech_exo.execution.models import Order, Fill, OrderStatus, OrderSide
from mech_exo.execution.fill_store import FillStore

logger = logging.getLogger(__name__)


@dataclass
class StubBrokerConfig:
    """Configuration for StubBroker behavior"""
    # Fill simulation
    simulate_fills: bool = True
    fill_delay_ms: int = 100  # Milliseconds
    fill_delay_variance: int = 50  # +/- variance in milliseconds
    
    # Rejection simulation
    reject_probability: float = 0.0  # 0.0 = never reject, 1.0 = always reject
    specific_rejections: Dict[str, str] = field(default_factory=dict)  # symbol -> rejection reason
    
    # Market simulation
    price_movement: bool = True
    volatility: float = 0.02  # 2% daily volatility
    bid_ask_spread_bps: int = 5  # 5 basis points spread
    
    # Slippage simulation
    simulate_slippage: bool = True
    avg_slippage_bps: float = 1.0  # Average 1 bp slippage
    slippage_variance_bps: float = 2.0  # +/- 2 bp variance
    
    # Account simulation
    initial_nav: float = 100000.0
    initial_cash: Optional[float] = None  # Defaults to initial_nav if not specified
    margin_multiplier: float = 2.0
    
    # Latency simulation
    simulate_latency: bool = False
    base_latency_ms: int = 10
    latency_variance_ms: int = 5
    
    # Error simulation
    connection_errors: bool = False
    error_rate: float = 0.01  # 1% of operations fail
    
    # Fill store integration
    write_to_fill_store: bool = True
    fill_store_path: Optional[str] = None


class EnhancedStubBroker(BrokerAdapter):
    """
    Enhanced StubBroker for comprehensive testing
    Features:
    - Instant or delayed fills with realistic simulation
    - Configurable rejection scenarios  
    - Market data simulation with price movement
    - Slippage and commission simulation
    - FillStore integration for persistence
    - Latency and error simulation
    - Account tracking with P&L
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Convert dict config to StubBrokerConfig
        if config is None:
            config = {}
        
        self.stub_config = StubBrokerConfig(
            simulate_fills=config.get('simulate_fills', True),
            fill_delay_ms=config.get('fill_delay_ms', 100),
            fill_delay_variance=config.get('fill_delay_variance', 50),
            reject_probability=config.get('reject_probability', 0.0),
            price_movement=config.get('price_movement', True),
            volatility=config.get('volatility', 0.02),
            simulate_slippage=config.get('simulate_slippage', True),
            initial_nav=config.get('initial_nav', 100000.0),
            initial_cash=config.get('initial_cash'),  # None if not specified
            write_to_fill_store=config.get('write_to_fill_store', True),
            fill_store_path=config.get('fill_store_path')
        )
        
        # Initialize base adapter
        super().__init__(config)
        
        # Account state
        self.account_id = "STUB_ACCOUNT_" + str(uuid.uuid4())[:8]
        self.current_nav = self.stub_config.initial_nav
        # Set cash balance to initial_nav if not specified
        self.cash_balance = self.stub_config.initial_cash if self.stub_config.initial_cash is not None else self.stub_config.initial_nav
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # Market data simulation
        self.market_prices: Dict[str, float] = {}
        self.last_price_update = datetime.now()
        
        # Order tracking
        self.pending_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}
        self.order_counter = 0
        
        # Fill store integration
        self.fill_store = None
        if self.stub_config.write_to_fill_store:
            try:
                self.fill_store = FillStore(self.stub_config.fill_store_path)
                logger.info("StubBroker connected to FillStore")
            except Exception as e:
                logger.warning(f"Failed to connect to FillStore: {e}")
        
        # Connection state
        self.connection_time = None
        self.is_market_open = True  # Always open for testing
        
        logger.info(f"EnhancedStubBroker initialized: {self.account_id}")
    
    async def connect(self) -> bool:
        """Simulate broker connection"""
        try:
            if self.stub_config.simulate_latency:
                await self._simulate_latency()
            
            self.status = BrokerStatus.CONNECTED
            self.connection_time = datetime.now()
            
            # Initialize some market prices
            default_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY', 'QQQ', 'FXI']
            base_prices = [175.0, 140.0, 340.0, 250.0, 425.0, 380.0, 35.0]
            
            for symbol, price in zip(default_symbols, base_prices):
                self.market_prices[symbol] = price
            
            logger.info(f"StubBroker connected: {self.account_id}")
            return True
            
        except Exception as e:
            logger.error(f"StubBroker connection failed: {e}")
            self.status = BrokerStatus.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Simulate broker disconnection"""
        try:
            self.status = BrokerStatus.DISCONNECTED
            self.connection_time = None
            
            # Close fill store
            if self.fill_store:
                self.fill_store.close()
                self.fill_store = None
            
            logger.info("StubBroker disconnected")
            return True
            
        except Exception as e:
            logger.error(f"StubBroker disconnect error: {e}")
            return False
    
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """Place order with realistic simulation"""
        if not self.is_connected():
            raise Exception("StubBroker not connected")
        
        try:
            if self.stub_config.simulate_latency:
                await self._simulate_latency()
            
            # Check for specific rejections
            if order.symbol in self.stub_config.specific_rejections:
                reason = self.stub_config.specific_rejections[order.symbol]
                order.status = OrderStatus.REJECTED
                self._notify_order_update(order)
                return {
                    'status': 'REJECTED',
                    'broker_order_id': None,
                    'message': reason
                }
            
            # Simulate random rejections
            if random.random() < self.stub_config.reject_probability:
                reasons = [
                    "Insufficient buying power",
                    "Symbol not found",
                    "Order size too small",
                    "Market closed",
                    "Risk management rejection"
                ]
                reason = random.choice(reasons)
                
                order.status = OrderStatus.REJECTED
                self._notify_order_update(order)
                return {
                    'status': 'REJECTED',
                    'broker_order_id': None,
                    'message': reason
                }
            
            # Accept order
            self.order_counter += 1
            order.broker_order_id = f"STUB_{self.order_counter:06d}"
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            
            self.pending_orders[order.order_id] = order
            self._notify_order_update(order)
            
            logger.info(f"StubBroker accepted order: {order.symbol} {order.quantity} -> {order.broker_order_id}")
            
            # Schedule fill simulation
            if self.stub_config.simulate_fills:
                asyncio.create_task(self._simulate_fill(order))
            
            return {
                'status': 'SUBMITTED',
                'broker_order_id': order.broker_order_id,
                'message': 'Order accepted by StubBroker'
            }
            
        except Exception as e:
            logger.error(f"StubBroker place_order failed: {e}")
            order.status = OrderStatus.REJECTED
            self._notify_order_update(order)
            
            return {
                'status': 'REJECTED',
                'broker_order_id': None,
                'message': f"StubBroker error: {e}"
            }
    
    async def _simulate_fill(self, order: Order):
        """Simulate order fill with realistic timing and pricing"""
        try:
            # Calculate fill delay
            base_delay = self.stub_config.fill_delay_ms / 1000.0  # Convert to seconds
            variance = self.stub_config.fill_delay_variance / 1000.0
            fill_delay = max(0.001, base_delay + random.uniform(-variance, variance))
            
            await asyncio.sleep(fill_delay)
            
            # Update market price (simulate movement)
            if self.stub_config.price_movement:
                self._update_market_price(order.symbol)
            
            # Determine fill price
            fill_price = self._calculate_fill_price(order)
            
            # Update order status
            order.status = OrderStatus.FILLED
            self.pending_orders.pop(order.order_id, None)
            self.completed_orders[order.order_id] = order
            
            # Create fill
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                quantity=order.quantity,
                price=fill_price,
                filled_at=datetime.now(),
                broker_order_id=order.broker_order_id,
                broker_fill_id=f"FILL_{self.order_counter:06d}",
                exchange="STUB_EXCHANGE",
                commission=self._calculate_commission(order),
                strategy=order.strategy,
                notes="Simulated fill by StubBroker"
            )
            
            # Add slippage calculation
            if self.stub_config.simulate_slippage and order.limit_price:
                fill.calculate_slippage(order.limit_price)
            
            # Update account positions
            self._update_position(fill)
            
            # Store fill to FillStore if enabled
            if self.fill_store:
                try:
                    self.fill_store.store_fill(fill)
                    self.fill_store.store_order(order)
                except Exception as e:
                    logger.warning(f"Failed to store fill to FillStore: {e}")
            
            # Notify callbacks
            self._notify_order_update(order)
            self._notify_fill_update(fill)
            
            logger.info(f"StubBroker filled order: {fill.symbol} {fill.quantity} @ ${fill.price}")
            
        except Exception as e:
            logger.error(f"StubBroker fill simulation failed: {e}")
    
    def _calculate_fill_price(self, order: Order) -> float:
        """Calculate realistic fill price with slippage"""
        # Get current market price
        if order.symbol not in self.market_prices:
            # Initialize with reasonable default
            if order.limit_price:
                base_price = order.limit_price
            elif order.stop_price:
                base_price = order.stop_price
            else:
                base_price = 100.0  # Default fallback
            
            self.market_prices[order.symbol] = base_price
        
        market_price = self.market_prices[order.symbol]
        
        # For market orders, simulate bid-ask spread and slippage
        if order.order_type.value == 'MKT':
            spread_bps = self.stub_config.bid_ask_spread_bps
            spread = market_price * (spread_bps / 10000)
            
            if order.quantity > 0:  # Buy order
                base_price = market_price + spread / 2  # Pay the ask
            else:  # Sell order
                base_price = market_price - spread / 2  # Receive the bid
            
            # Add slippage
            if self.stub_config.simulate_slippage:
                slippage_bps = random.uniform(
                    -self.stub_config.slippage_variance_bps,
                    self.stub_config.slippage_variance_bps
                ) + self.stub_config.avg_slippage_bps
                
                slippage = base_price * (slippage_bps / 10000)
                
                if order.quantity > 0:  # Buy order - positive slippage is bad
                    fill_price = base_price + abs(slippage)
                else:  # Sell order - negative slippage is bad
                    fill_price = base_price - abs(slippage)
            else:
                fill_price = base_price
                
        else:
            # For limit orders, fill at limit price or better
            fill_price = order.limit_price
        
        return round(fill_price, 2)
    
    def _calculate_commission(self, order: Order) -> float:
        """Calculate commission based on order"""
        # Simple commission structure
        base_commission = 1.0
        per_share = 0.005
        
        return base_commission + abs(order.quantity) * per_share
    
    def _update_market_price(self, symbol: str):
        """Simulate market price movement"""
        if symbol not in self.market_prices:
            return
        
        current_price = self.market_prices[symbol]
        
        # Simple random walk with drift
        dt = 1 / 252 / 6.5 / 3600  # Assume 1-second intervals during trading hours
        volatility = self.stub_config.volatility
        
        # Random price movement
        random_change = random.normalvariate(0, volatility * (dt ** 0.5))
        new_price = current_price * (1 + random_change)
        
        self.market_prices[symbol] = max(0.01, new_price)  # Prevent negative prices
    
    def _update_position(self, fill: Fill):
        """Update account position from fill"""
        symbol = fill.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            }
        
        position = self.positions[symbol]
        old_quantity = position['quantity']
        old_avg_price = position['avg_price']
        
        # Update position
        new_quantity = old_quantity + fill.quantity
        
        if new_quantity == 0:
            # Position closed
            position['quantity'] = 0
            position['avg_price'] = 0.0
            position['realized_pnl'] += (fill.price - old_avg_price) * abs(fill.quantity)
        elif old_quantity == 0:
            # New position
            position['quantity'] = new_quantity
            position['avg_price'] = fill.price
        elif (old_quantity > 0 and fill.quantity > 0) or (old_quantity < 0 and fill.quantity < 0):
            # Adding to position
            total_cost = old_avg_price * abs(old_quantity) + fill.price * abs(fill.quantity)
            total_quantity = abs(new_quantity)
            position['avg_price'] = total_cost / total_quantity
            position['quantity'] = new_quantity
        else:
            # Reducing position
            position['quantity'] = new_quantity
            position['realized_pnl'] += (fill.price - old_avg_price) * abs(fill.quantity)
        
        # Update cash balance
        self.cash_balance -= fill.gross_value + fill.total_fees
    
    async def _simulate_latency(self):
        """Simulate network latency"""
        if not self.stub_config.simulate_latency:
            return
        
        base_latency = self.stub_config.base_latency_ms / 1000.0
        variance = self.stub_config.latency_variance_ms / 1000.0
        latency = max(0.001, base_latency + random.uniform(-variance, variance))
        
        await asyncio.sleep(latency)
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order"""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            
            self.pending_orders.pop(order_id)
            self.completed_orders[order_id] = order
            
            self._notify_order_update(order)
            
            return {
                'status': 'CANCELLED',
                'message': f'Order {order_id} cancelled'
            }
        else:
            return {
                'status': 'ERROR',
                'message': f'Order {order_id} not found or already completed'
            }
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
        elif order_id in self.completed_orders:
            order = self.completed_orders[order_id]
        else:
            return {
                'status': 'ERROR',
                'message': 'Order not found'
            }
        
        return {
            'order_id': order_id,
            'status': order.status.value,
            'broker_order_id': order.broker_order_id,
            'symbol': order.symbol,
            'quantity': order.quantity
        }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        positions = []
        
        for symbol, pos_data in self.positions.items():
            if pos_data['quantity'] != 0:
                market_price = self.market_prices.get(symbol, pos_data['avg_price'])
                market_value = abs(pos_data['quantity']) * market_price
                unrealized_pnl = (market_price - pos_data['avg_price']) * pos_data['quantity']
                
                positions.append({
                    'symbol': symbol,
                    'position': pos_data['quantity'],
                    'avg_cost': pos_data['avg_price'],
                    'market_price': market_price,
                    'market_value': market_value,
                    'unrealized_pnl': unrealized_pnl,
                    'realized_pnl': pos_data['realized_pnl'],
                    'account': self.account_id
                })
        
        return positions
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        # Calculate total position value
        total_position_value = 0
        total_unrealized_pnl = 0
        
        for symbol, pos_data in self.positions.items():
            if pos_data['quantity'] != 0:
                market_price = self.market_prices.get(symbol, pos_data['avg_price'])
                position_value = abs(pos_data['quantity']) * market_price
                unrealized_pnl = (market_price - pos_data['avg_price']) * pos_data['quantity']
                
                total_position_value += position_value
                total_unrealized_pnl += unrealized_pnl
        
        # Calculate NAV
        current_nav = self.cash_balance + total_position_value + total_unrealized_pnl
        
        return {
            'account_id': self.account_id,
            'netliquidation': current_nav,
            'totalcashvalue': self.cash_balance,
            'buyingpower': self.cash_balance * self.stub_config.margin_multiplier,
            'grosspositionvalue': total_position_value,
            'unrealized_pnl': total_unrealized_pnl,
            'timestamp': datetime.now()
        }
    
    def get_broker_info(self) -> BrokerInfo:
        """Get broker connection info"""
        return BrokerInfo(
            broker_name="Enhanced StubBroker",
            status=self.status,
            account_id=self.account_id,
            connection_time=self.connection_time,
            last_heartbeat=datetime.now() if self.is_connected() else None,
            error_message=None
        )
    
    def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        return self.market_prices.get(symbol)
    
    def set_market_price(self, symbol: str, price: float):
        """Set market price for testing"""
        self.market_prices[symbol] = price
    
    def add_rejection(self, symbol: str, reason: str):
        """Add specific rejection for testing"""
        self.stub_config.specific_rejections[symbol] = reason
    
    def clear_rejections(self):
        """Clear all specific rejections"""
        self.stub_config.specific_rejections.clear()
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get trading session summary"""
        total_orders = len(self.completed_orders) + len(self.pending_orders)
        filled_orders = len([o for o in self.completed_orders.values() if o.status == OrderStatus.FILLED])
        rejected_orders = len([o for o in self.completed_orders.values() if o.status == OrderStatus.REJECTED])
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'rejected_orders': rejected_orders,
            'pending_orders': len(self.pending_orders),
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0,
            'symbols_traded': len(self.positions),
            'account_id': self.account_id
        }