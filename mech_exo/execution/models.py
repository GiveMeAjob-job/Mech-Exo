"""
Order and Fill data models for execution engine
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order types"""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    BRACKET = "BRACKET"
    IOC = "IOC"  # Immediate or Cancel
    GTD = "GTD"  # Good Till Date


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    INACTIVE = "INACTIVE"


class ExecutionError(Exception):
    """Base execution error"""


class RiskViolationError(ExecutionError):
    """Risk limit violation error"""


class GatewayError(ExecutionError):
    """Gateway connection error"""


@dataclass
class Order:
    """Order model with all necessary fields for execution"""

    # Required fields
    symbol: str
    quantity: int  # Positive for BUY, negative for SELL
    order_type: OrderType

    # Auto-generated fields
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    # Optional fields
    limit_price = None
    stop_price = None
    time_in_force: str = "DAY"

    # Bracket order fields
    parent_order_id = None
    profit_target = None
    stop_loss = None

    # Status and tracking
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id = None
    submitted_at = None

    # Metadata
    strategy = None
    signal_strength: float = 1.0
    notes = None

    @property
    def side(self) -> OrderSide:
        """Order side based on quantity"""
        return OrderSide.BUY if self.quantity > 0 else OrderSide.SELL

    @property
    def abs_quantity(self) -> int:
        """Absolute quantity"""
        return abs(self.quantity)

    @property
    def is_buy(self) -> bool:
        """Check if order is a buy"""
        return self.quantity > 0

    @property
    def is_sell(self) -> bool:
        """Check if order is a sell"""
        return self.quantity < 0

    @property
    def estimated_value(self):
        """Estimated order value"""
        if self.order_type == OrderType.MARKET:
            return None  # Unknown for market orders

        price = self.limit_price or self.stop_price
        if price:
            return abs(self.quantity) * price
        return None

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)

        # Convert enums to strings
        data["order_type"] = self.order_type.value
        data["status"] = self.status.value

        # Convert datetime to ISO string
        data["created_at"] = self.created_at.isoformat()
        if self.submitted_at:
            data["submitted_at"] = self.submitted_at.isoformat()

        return data

    @classmethod
    def from_dict(cls, data):
        """Create Order from dictionary"""
        # Convert string enums back
        data["order_type"] = OrderType(data["order_type"])
        data["status"] = OrderStatus(data["status"])

        # Convert ISO strings back to datetime
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("submitted_at"):
            data["submitted_at"] = datetime.fromisoformat(data["submitted_at"])

        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Order":
        """Create Order from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Fill:
    """Fill/execution model"""

    # Required fields
    order_id: str
    symbol: str
    quantity: int  # Positive for BUY fills, negative for SELL fills
    price: float
    filled_at: datetime

    # Auto-generated fields
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Broker fields
    broker_order_id = None
    broker_fill_id = None
    exchange = None

    # Cost tracking
    commission: float = 0.0
    fees: float = 0.0
    sec_fee: float = 0.0

    # Execution quality
    reference_price = None  # Price at time of order submission
    slippage_bps = None     # Slippage in basis points

    # Metadata
    strategy = None
    notes = None

    @property
    def side(self) -> OrderSide:
        """Fill side based on quantity"""
        return OrderSide.BUY if self.quantity > 0 else OrderSide.SELL

    @property
    def abs_quantity(self) -> int:
        """Absolute quantity"""
        return abs(self.quantity)

    @property
    def gross_value(self) -> float:
        """Gross value of fill"""
        return abs(self.quantity) * self.price

    @property
    def total_fees(self) -> float:
        """Total fees and commissions"""
        return self.commission + self.fees + self.sec_fee

    @property
    def net_value(self) -> float:
        """Net value after fees"""
        return self.gross_value - self.total_fees

    def calculate_slippage(self, reference_price: float) -> float:
        """Calculate slippage in basis points"""
        if reference_price <= 0:
            return 0.0

        price_diff = self.price - reference_price

        # For buys, positive slippage is bad (paid more)
        # For sells, negative slippage is bad (received less)
        if self.quantity > 0:  # Buy
            slippage = price_diff / reference_price * 10000  # bps
        else:  # Sell
            slippage = -price_diff / reference_price * 10000  # bps

        self.slippage_bps = slippage
        return slippage

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)

        # Convert datetime to ISO string
        data["filled_at"] = self.filled_at.isoformat()

        return data

    @classmethod
    def from_dict(cls, data):
        """Create Fill from dictionary"""
        # Convert ISO string back to datetime
        data["filled_at"] = datetime.fromisoformat(data["filled_at"])

        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Fill":
        """Create Fill from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Position:
    """Position model (different from risk.Position, focused on execution)"""

    symbol: str
    quantity: int  # Net position (positive = long, negative = short)
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Tracking
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        """Market value at current price"""
        return abs(self.quantity) * self.avg_price

    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat"""
        return self.quantity == 0


# Factory functions for common order types
def create_market_order(symbol: str, quantity: int, **kwargs) -> Order:
    """Create a market order"""
    return Order(
        symbol=symbol,
        quantity=quantity,
        order_type=OrderType.MARKET,
        **kwargs
    )


def create_limit_order(symbol: str, quantity: int, limit_price: float, **kwargs) -> Order:
    """Create a limit order"""
    return Order(
        symbol=symbol,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        limit_price=limit_price,
        **kwargs
    )


def create_bracket_order(symbol: str, quantity: int, limit_price: float,
                        profit_target: float, stop_loss: float, **kwargs):
    """Create a bracket order (parent + profit target + stop loss)"""

    # Parent order
    parent = Order(
        symbol=symbol,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        limit_price=limit_price,
        **kwargs
    )

    # Profit target order (opposite side)
    profit_order = Order(
        symbol=symbol,
        quantity=-quantity,  # Opposite side
        order_type=OrderType.LIMIT,
        limit_price=profit_target,
        parent_order_id=parent.order_id,
        **kwargs
    )

    # Stop loss order (opposite side)
    stop_order = Order(
        symbol=symbol,
        quantity=-quantity,  # Opposite side
        order_type=OrderType.STOP,
        stop_price=stop_loss,
        parent_order_id=parent.order_id,
        **kwargs
    )

    return [parent, profit_order, stop_order]


def create_ioc_order(symbol: str, quantity: int, limit_price: float, **kwargs) -> Order:
    """Create an Immediate or Cancel order"""
    return Order(
        symbol=symbol,
        quantity=quantity,
        order_type=OrderType.IOC,
        limit_price=limit_price,
        time_in_force="IOC",
        **kwargs
    )


# Validation functions
def validate_order(order: Order):
    """Validate order and return list of errors"""
    errors = []

    if not order.symbol:
        errors.append("Symbol is required")

    if order.quantity == 0:
        errors.append("Quantity cannot be zero")

    if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and not order.limit_price:
        errors.append("Limit price required for limit orders")

    if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and not order.stop_price:
        errors.append("Stop price required for stop orders")

    if order.limit_price and order.limit_price <= 0:
        errors.append("Limit price must be positive")

    if order.stop_price and order.stop_price <= 0:
        errors.append("Stop price must be positive")

    # Bracket order validation
    if order.order_type == OrderType.BRACKET:
        if not order.profit_target:
            errors.append("Profit target required for bracket orders")
        if not order.stop_loss:
            errors.append("Stop loss required for bracket orders")

    return errors


def validate_fill(fill: Fill):
    """Validate fill and return list of errors"""
    errors = []

    if not fill.symbol:
        errors.append("Symbol is required")

    if fill.quantity == 0:
        errors.append("Fill quantity cannot be zero")

    if fill.price <= 0:
        errors.append("Fill price must be positive")

    if fill.commission < 0:
        errors.append("Commission cannot be negative")

    if fill.fees < 0:
        errors.append("Fees cannot be negative")

    return errors
