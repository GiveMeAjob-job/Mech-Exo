"""
Live-mode safety valve with confirmation and sentinel orders
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .models import Order, Fill, OrderStatus, OrderType, create_limit_order
from .broker_adapter import BrokerAdapter
from ..utils.structured_logging import ExecutionLogger, ExecutionContext

logger = logging.getLogger(__name__)


class SafetyMode(Enum):
    """Safety modes for live trading"""
    DISABLED = "disabled"
    CONFIRMATION_ONLY = "confirmation_only" 
    SENTINEL_ONLY = "sentinel_only"
    FULL_SAFETY = "full_safety"  # Both confirmation and sentinel


@dataclass
class SentinelConfig:
    """Configuration for sentinel orders"""
    symbol: str = "CAD"  # Currency symbol for sentinel
    quantity: int = 100  # Number of units
    max_price: float = 1.50  # Maximum price per unit (CAD ~$1.50 USD)
    order_type: OrderType = OrderType.IOC  # Immediate or Cancel
    timeout_seconds: int = 30  # How long to wait for sentinel response
    
    @property
    def max_value(self) -> float:
        """Maximum value of sentinel order"""
        return abs(self.quantity) * self.max_price


class SafetyValve:
    """
    Live-mode safety valve for production trading
    
    Features:
    - CLI double confirmation for live mode
    - Sentinel order validation before main orders
    - Emergency abort mechanisms
    - Activity logging and monitoring
    """
    
    def __init__(self, broker: BrokerAdapter, config: Optional[Dict[str, Any]] = None):
        self.broker = broker
        self.config = config or {}
        
        # Setup structured logging
        self.execution_context = ExecutionContext.create(
            component="safety_valve",
            account_id=getattr(broker, 'account_id', None)
        )
        self.execution_logger = ExecutionLogger(__name__, self.execution_context)
        
        # Safety configuration
        self.mode = SafetyMode(self.config.get('safety_mode', 'full_safety'))
        self.require_confirmation = self.config.get('require_confirmation', True)
        self.use_sentinel_orders = self.config.get('use_sentinel_orders', True)
        
        # Sentinel configuration
        sentinel_config = self.config.get('sentinel', {})
        self.sentinel_config = SentinelConfig(
            symbol=sentinel_config.get('symbol', 'CAD'),
            quantity=sentinel_config.get('quantity', 100),
            max_price=sentinel_config.get('max_price', 1.50),
            order_type=OrderType(sentinel_config.get('order_type', 'IOC')),
            timeout_seconds=sentinel_config.get('timeout_seconds', 30)
        )
        
        # State tracking
        self.last_confirmation_time = None
        self.confirmation_valid_minutes = self.config.get('confirmation_valid_minutes', 60)
        self.sentinel_orders: Dict[str, Order] = {}
        self.aborted_sessions = []
        
        # Emergency controls
        self.emergency_abort = False
        self.max_daily_value = self.config.get('max_daily_value', 100000.0)  # $100k daily limit
        self.daily_order_value = 0.0
        self.last_reset_date = datetime.now().date()
        
        logger.info(f"SafetyValve initialized: mode={self.mode.value}, sentinel={self.use_sentinel_orders}")
        
        # Log initialization
        self.execution_logger.system_event(
            system="safety_valve",
            status="initialized",
            message="SafetyValve initialized",
            safety_mode=self.mode.value,
            use_sentinel_orders=self.use_sentinel_orders,
            max_daily_value=self.max_daily_value,
            sentinel_symbol=self.sentinel_config.symbol,
            sentinel_quantity=self.sentinel_config.quantity
        )
    
    async def authorize_live_trading(self, session_description: str = "trading session") -> bool:
        """
        Authorize live trading with safety checks
        
        Returns:
            bool: True if authorized, False if aborted
        """
        try:
            # Check if we're actually in live mode
            trading_mode = os.getenv('EXO_MODE', '').lower()
            if trading_mode != 'live':
                logger.info(f"Not in live mode (mode={trading_mode}), skipping safety valve")
                return True
            
            logger.warning("🚨 LIVE TRADING MODE ACTIVATED 🚨")
            
            # Log live trading activation attempt
            self.execution_logger.safety_event(
                safety_type="live_trading",
                action="authorization_requested",
                message=f"Live trading authorization requested: {session_description}",
                session_description=session_description,
                safety_mode=self.mode.value
            )
            
            # Reset daily counters if needed
            self._reset_daily_counters()
            
            # Check for emergency abort
            if self.emergency_abort:
                logger.error("Emergency abort is active - trading disabled")
                self.execution_logger.safety_event(
                    safety_type="emergency_abort",
                    action="authorization_blocked",
                    message="Live trading authorization blocked by emergency abort",
                    session_description=session_description
                )
                return False
            
            # Step 1: CLI Confirmation (if required)
            if self.mode in [SafetyMode.CONFIRMATION_ONLY, SafetyMode.FULL_SAFETY]:
                if not await self._get_cli_confirmation(session_description):
                    logger.warning("Live trading aborted by user")
                    self.execution_logger.safety_event(
                        safety_type="confirmation",
                        action="authorization_denied",
                        message="Live trading authorization denied by user",
                        session_description=session_description
                    )
                    return False
            
            # Step 2: Sentinel Order Check (if required)
            if self.mode in [SafetyMode.SENTINEL_ONLY, SafetyMode.FULL_SAFETY]:
                if not await self._verify_sentinel_order():
                    logger.error("Sentinel order verification failed - aborting live trading")
                    self.execution_logger.safety_event(
                        safety_type="sentinel",
                        action="authorization_failed",
                        message="Live trading authorization failed - sentinel order verification failed",
                        session_description=session_description
                    )
                    return False
            
            logger.info("✅ Live trading authorized - safety checks passed")
            self.execution_logger.safety_event(
                safety_type="live_trading",
                action="authorization_granted",
                message="Live trading authorization granted - all safety checks passed",
                session_description=session_description,
                safety_mode=self.mode.value
            )
            return True
            
        except Exception as e:
            logger.error(f"Safety valve authorization failed: {e}")
            return False
    
    async def _get_cli_confirmation(self, session_description: str) -> bool:
        """Get CLI confirmation from user"""
        try:
            # Check if confirmation is still valid
            if (self.last_confirmation_time and 
                datetime.now() - self.last_confirmation_time < timedelta(minutes=self.confirmation_valid_minutes)):
                logger.info("Using cached confirmation (still valid)")
                return True
            
            print("\n" + "="*60)
            print("🚨 LIVE TRADING CONFIRMATION REQUIRED 🚨")
            print("="*60)
            print(f"Session: {session_description}")
            print(f"Broker: {self.broker.get_broker_info().broker_name}")
            print(f"Account: {self.broker.get_broker_info().account_id}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Mode: {os.getenv('EXO_MODE', 'unknown')}")
            
            # Show daily limits
            print(f"\nDaily Limits:")
            print(f"  Max daily value: ${self.max_daily_value:,.0f}")
            print(f"  Used today: ${self.daily_order_value:,.0f}")
            print(f"  Remaining: ${self.max_daily_value - self.daily_order_value:,.0f}")
            
            # Show safety configuration
            print(f"\nSafety Configuration:")
            print(f"  Safety mode: {self.mode.value}")
            if self.use_sentinel_orders:
                print(f"  Sentinel: {self.sentinel_config.symbol} {self.sentinel_config.quantity} (max ${self.sentinel_config.max_value:.0f})")
            
            print("\n⚠️  WARNING: This will place REAL ORDERS with REAL MONEY ⚠️")
            print("Double-check all configurations before proceeding.")
            print("\n" + "="*60)
            
            # Get user confirmation
            while True:
                try:
                    response = input("\nDo you want to proceed with LIVE TRADING? (yes/no): ").strip().lower()
                    
                    if response in ['yes', 'y']:
                        # Require exact "yes" for safety
                        if response == 'yes':
                            self.last_confirmation_time = datetime.now()
                            logger.info("✅ Live trading confirmed by user")
                            return True
                        else:
                            print("Please type 'yes' exactly (not just 'y') to confirm live trading.")
                            continue
                    
                    elif response in ['no', 'n', 'abort', 'cancel', 'exit']:
                        logger.warning("❌ Live trading aborted by user")
                        return False
                    
                    else:
                        print("Please respond with 'yes' or 'no'")
                        continue
                        
                except (KeyboardInterrupt, EOFError):
                    print("\n❌ Live trading aborted (Ctrl+C)")
                    logger.warning("Live trading aborted by keyboard interrupt")
                    return False
            
        except Exception as e:
            logger.error(f"CLI confirmation failed: {e}")
            return False
    
    async def _verify_sentinel_order(self) -> bool:
        """Verify broker connectivity with small sentinel order"""
        try:
            logger.info(f"Sending sentinel order: {self.sentinel_config.symbol} {self.sentinel_config.quantity}")
            
            # Create sentinel order
            sentinel_order = create_limit_order(
                symbol=self.sentinel_config.symbol,
                quantity=self.sentinel_config.quantity,
                limit_price=self.sentinel_config.max_price,
                time_in_force="IOC",
                strategy="sentinel_safety_check",
                notes="Safety valve sentinel order for live trading verification"
            )
            
            # Track sentinel order
            self.sentinel_orders[sentinel_order.order_id] = sentinel_order
            
            # Submit sentinel order
            start_time = datetime.now()
            result = await self.broker.place_order(sentinel_order)
            
            if result['status'] not in ['SUBMITTED', 'FILLED']:
                logger.error(f"Sentinel order failed to submit: {result.get('message', 'Unknown error')}")
                return False
            
            logger.info(f"Sentinel order submitted: {result.get('broker_order_id')}")
            
            # Wait for sentinel order to complete
            timeout_time = start_time + timedelta(seconds=self.sentinel_config.timeout_seconds)
            
            while datetime.now() < timeout_time:
                status = await self.broker.get_order_status(sentinel_order.order_id)
                
                if status['status'] == 'FILLED':
                    logger.info("✅ Sentinel order filled - broker connectivity verified")
                    return True
                elif status['status'] == 'CANCELLED':
                    logger.info("✅ Sentinel order cancelled - broker connectivity verified")
                    return True
                elif status['status'] in ['REJECTED']:
                    logger.error(f"❌ Sentinel order rejected: {status}")
                    return False
                
                # Wait a bit before checking again
                await asyncio.sleep(0.5)
            
            # Timeout reached
            logger.warning(f"⏰ Sentinel order timeout after {self.sentinel_config.timeout_seconds}s")
            
            # Try to cancel the sentinel order
            try:
                cancel_result = await self.broker.cancel_order(sentinel_order.order_id)
                logger.info(f"Cancelled sentinel order: {cancel_result}")
            except Exception as e:
                logger.warning(f"Failed to cancel sentinel order: {e}")
            
            return False
            
        except Exception as e:
            logger.error(f"Sentinel order verification failed: {e}")
            return False
    
    async def check_order_safety(self, order: Order) -> Dict[str, Any]:
        """
        Check if an order is safe to execute in live mode
        
        Returns:
            Dict with 'approved', 'reason', and 'warnings' keys
        """
        try:
            warnings = []
            
            # Check emergency abort
            if self.emergency_abort:
                return {
                    'approved': False,
                    'reason': 'Emergency abort is active',
                    'warnings': warnings
                }
            
            # Check daily value limits
            order_value = abs(order.quantity) * (order.limit_price or order.stop_price or 100.0)
            
            if self.daily_order_value + order_value > self.max_daily_value:
                return {
                    'approved': False,
                    'reason': f'Order would exceed daily limit: ${order_value:,.0f} + ${self.daily_order_value:,.0f} > ${self.max_daily_value:,.0f}',
                    'warnings': warnings
                }
            
            # Check order size reasonableness
            if order_value > self.max_daily_value * 0.2:  # Single order > 20% of daily limit
                warnings.append(f'Large order: ${order_value:,.0f} (>{self.max_daily_value*0.2:,.0f})')
            
            # Check order timing (market hours, etc.)
            current_time = datetime.now()
            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            if not (market_open <= current_time <= market_close):
                warnings.append('Order submitted outside regular market hours')
            
            # Update daily tracking if approved
            self.daily_order_value += order_value
            
            return {
                'approved': True,
                'reason': 'Safety checks passed',
                'warnings': warnings,
                'order_value': order_value,
                'daily_total': self.daily_order_value
            }
            
        except Exception as e:
            logger.error(f"Order safety check failed: {e}")
            return {
                'approved': False,
                'reason': f'Safety check error: {e}',
                'warnings': []
            }
    
    def activate_emergency_abort(self, reason: str = "Manual activation"):
        """Activate emergency abort - stops all live trading"""
        self.emergency_abort = True
        self.aborted_sessions.append({
            'timestamp': datetime.now(),
            'reason': reason
        })
        logger.critical(f"🚨 EMERGENCY ABORT ACTIVATED: {reason}")
        
        # Log emergency abort activation
        self.execution_logger.safety_event(
            safety_type="emergency_abort",
            action="activated",
            message=f"Emergency abort activated: {reason}",
            abort_reason=reason,
            timestamp=datetime.now().isoformat(),
            total_aborted_sessions=len(self.aborted_sessions)
        )
    
    def deactivate_emergency_abort(self, reason: str = "Manual deactivation"):
        """Deactivate emergency abort"""
        if self.emergency_abort:
            self.emergency_abort = False
            logger.warning(f"Emergency abort deactivated: {reason}")
        else:
            logger.info("Emergency abort was not active")
    
    def _reset_daily_counters(self):
        """Reset daily counters if new day"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_order_value = 0.0
            self.last_reset_date = today
            logger.info("Daily safety counters reset")
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety valve status"""
        return {
            'mode': self.mode.value,
            'emergency_abort': self.emergency_abort,
            'daily_value_used': self.daily_order_value,
            'daily_value_limit': self.max_daily_value,
            'daily_value_remaining': self.max_daily_value - self.daily_order_value,
            'confirmation_valid': (
                self.last_confirmation_time and 
                datetime.now() - self.last_confirmation_time < timedelta(minutes=self.confirmation_valid_minutes)
            ),
            'last_confirmation': self.last_confirmation_time,
            'sentinel_orders_count': len(self.sentinel_orders),
            'aborted_sessions_count': len(self.aborted_sessions),
            'trading_mode': os.getenv('EXO_MODE', 'unknown')
        }
    
    def get_sentinel_summary(self) -> Dict[str, Any]:
        """Get sentinel order summary"""
        return {
            'config': {
                'symbol': self.sentinel_config.symbol,
                'quantity': self.sentinel_config.quantity,
                'max_price': self.sentinel_config.max_price,
                'max_value': self.sentinel_config.max_value,
                'timeout_seconds': self.sentinel_config.timeout_seconds
            },
            'recent_orders': [
                {
                    'order_id': order.order_id,
                    'created_at': order.created_at,
                    'status': order.status.value
                }
                for order in list(self.sentinel_orders.values())[-5:]  # Last 5 sentinel orders
            ]
        }