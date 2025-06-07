"""
Core position sizing implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mech_exo.utils import ConfigManager
from mech_exo.datasource import DataStorage
from .base import BaseSizer, SizingMethod, PositionType, SizingError, InsufficientCapitalError

logger = logging.getLogger(__name__)


class PositionSizer(BaseSizer):
    """Main position sizing engine with multiple sizing methods"""
    
    def __init__(self, nav: float, risk_config_path: str = "config/risk_limits.yml"):
        # Load risk configuration
        config_manager = ConfigManager()
        risk_config = config_manager.load_config("risk_limits")
        
        if not risk_config:
            raise ValueError(f"Could not load risk configuration from {risk_config_path}")
        
        # Extract sizing config
        sizing_config = risk_config.get("position_sizing", {})
        sizing_config["nav"] = nav
        
        super().__init__(sizing_config)
        
        # Initialize data storage for fetching price/vol data
        self.storage = DataStorage()
        
        # Sizing parameters
        self.base_risk_pct = sizing_config.get("max_single_trade_risk", 0.02)  # 2%
        self.atr_multiplier = sizing_config.get("atr_multiplier", 2.0)
        self.vol_target = sizing_config.get("volatility_target", 0.15)  # 15% annualized
        self.default_method = SizingMethod(sizing_config.get("default_method", "atr_based"))
        
        # Risk limits
        self.max_sector_exposure = risk_config.get("portfolio", {}).get("max_sector_exposure", 0.20)
        self.max_single_position = sizing_config.get("max_single_position", 0.10)
        
        logger.info(f"PositionSizer initialized with NAV=${nav:,.0f}, base_risk={self.base_risk_pct:.1%}")
    
    def calculate_size(self, symbol: str, price: float, method: Optional[SizingMethod] = None,
                      signal_strength: float = 1.0, **kwargs) -> int:
        """
        Calculate position size using specified method
        
        Args:
            symbol: Trading symbol
            price: Current price
            method: Sizing method to use
            signal_strength: Signal strength multiplier (0.0 to 2.0)
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Position size in shares (positive for long, negative for short)
        """
        if method is None:
            method = self.default_method
            
        # Validate inputs
        if price <= 0:
            raise SizingError(f"Invalid price: {price}")
            
        if not 0 <= signal_strength <= 2.0:
            raise SizingError(f"Signal strength must be between 0 and 2, got {signal_strength}")
        
        try:
            # Calculate base size using specified method
            if method == SizingMethod.FIXED_PERCENT:
                shares = self._calculate_fixed_percent(symbol, price, **kwargs)
            elif method == SizingMethod.ATR_BASED:
                shares = self._calculate_atr_based(symbol, price, **kwargs)
            elif method == SizingMethod.VOLATILITY_BASED:
                shares = self._calculate_volatility_based(symbol, price, **kwargs)
            elif method == SizingMethod.KELLY_CRITERION:
                shares = self._calculate_kelly(symbol, price, **kwargs)
            else:
                raise SizingError(f"Unsupported sizing method: {method}")
            
            # Apply signal strength
            shares = int(shares * signal_strength)
            
            # Validate and adjust for constraints
            if not self.validate_size(symbol, price, shares):
                # Try to scale down to meet constraints
                max_allowed = self._calculate_max_allowed_shares(symbol, price)
                shares = min(abs(shares), max_allowed) * (1 if shares >= 0 else -1)
                
                if abs(shares) == 0:
                    logger.warning(f"Position size for {symbol} reduced to zero due to constraints")
                    return 0
            
            # Adjust for liquidity if volume data available
            avg_volume = kwargs.get("avg_daily_volume")
            if avg_volume:
                shares = self.adjust_for_liquidity(symbol, shares, avg_volume)
            
            logger.info(f"Calculated size for {symbol}: {shares} shares @ ${price:.2f} "
                       f"(${abs(shares * price):,.0f}, {abs(shares * price) / self.nav:.1%} of NAV)")
            
            return shares
            
        except Exception as e:
            logger.error(f"Position sizing failed for {symbol}: {e}")
            raise SizingError(f"Position sizing failed for {symbol}: {e}")
    
    def _calculate_fixed_percent(self, symbol: str, price: float, 
                               risk_pct: Optional[float] = None) -> int:
        """Calculate position size as fixed percentage of NAV"""
        risk_pct = risk_pct or self.base_risk_pct
        position_value = self.nav * risk_pct
        shares = int(position_value / price)
        
        logger.debug(f"Fixed % sizing: {risk_pct:.1%} of ${self.nav:,.0f} = {shares} shares")
        return shares
    
    def _calculate_atr_based(self, symbol: str, price: float, 
                           atr: Optional[float] = None,
                           atr_multiplier: Optional[float] = None) -> int:
        """Calculate position size based on ATR (Average True Range)"""
        atr_mult = atr_multiplier or self.atr_multiplier
        
        # Get ATR if not provided
        if atr is None:
            atr = self._get_atr(symbol)
            if atr is None:
                logger.warning(f"No ATR data for {symbol}, falling back to fixed %")
                return self._calculate_fixed_percent(symbol, price)
        
        # Risk amount in dollars
        risk_amount = self.nav * self.base_risk_pct
        
        # Stop distance based on ATR
        stop_distance = atr * atr_mult
        
        # Position size = Risk Amount / Stop Distance
        shares = int(risk_amount / stop_distance)
        
        logger.debug(f"ATR sizing: Risk=${risk_amount:,.0f}, ATR={atr:.2f}, "
                    f"Stop={stop_distance:.2f}, Shares={shares}")
        
        return shares
    
    def _calculate_volatility_based(self, symbol: str, price: float,
                                  volatility: Optional[float] = None) -> int:
        """Calculate position size based on volatility targeting"""
        # Get volatility if not provided
        if volatility is None:
            volatility = self._get_volatility(symbol)
            if volatility is None:
                logger.warning(f"No volatility data for {symbol}, falling back to fixed %")
                return self._calculate_fixed_percent(symbol, price)
        
        # Volatility scaling: inverse relationship
        # Higher vol = smaller position
        vol_scalar = self.vol_target / max(volatility, 0.05)  # Min 5% vol to avoid huge positions
        vol_scalar = min(vol_scalar, 2.0)  # Cap at 2x
        
        # Base position value adjusted by volatility
        base_value = self.nav * self.base_risk_pct
        position_value = base_value * vol_scalar
        shares = int(position_value / price)
        
        logger.debug(f"Vol sizing: Vol={volatility:.1%}, Target={self.vol_target:.1%}, "
                    f"Scalar={vol_scalar:.2f}, Shares={shares}")
        
        return shares
    
    def _calculate_kelly(self, symbol: str, price: float,
                        win_rate: Optional[float] = None,
                        avg_win: Optional[float] = None,
                        avg_loss: Optional[float] = None) -> int:
        """Calculate position size using Kelly Criterion"""
        # Use provided parameters or defaults
        win_rate = win_rate or 0.55  # 55% win rate default
        avg_win = avg_win or 0.08    # 8% average win
        avg_loss = avg_loss or 0.04  # 4% average loss
        
        # Kelly fraction = (bp - q) / b
        # where: b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Cap Kelly at 25% for safety (full Kelly can be too aggressive)
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        # Convert to position size
        position_value = self.nav * kelly_fraction
        shares = int(position_value / price)
        
        logger.debug(f"Kelly sizing: WinRate={p:.1%}, AvgWin={avg_win:.1%}, "
                    f"AvgLoss={avg_loss:.1%}, Kelly={kelly_fraction:.1%}, Shares={shares}")
        
        return shares
    
    def _get_atr(self, symbol: str, periods: int = 14) -> Optional[float]:
        """Get latest ATR for symbol"""
        try:
            # Get recent OHLC data
            ohlc_data = self.storage.get_ohlc_data([symbol], limit=30)
            
            if ohlc_data.empty:
                return None
                
            # Get latest ATR value
            symbol_data = ohlc_data[ohlc_data['symbol'] == symbol].sort_values('date')
            
            if 'atr' in symbol_data.columns and not symbol_data['atr'].isna().all():
                return float(symbol_data['atr'].iloc[-1])
            
            # Calculate ATR if not available
            if len(symbol_data) >= periods:
                high_low = symbol_data['high'] - symbol_data['low']
                high_close = (symbol_data['high'] - symbol_data['close'].shift()).abs()
                low_close = (symbol_data['low'] - symbol_data['close'].shift()).abs()
                
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=periods).mean().iloc[-1]
                
                return float(atr)
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting ATR for {symbol}: {e}")
            return None
    
    def _get_volatility(self, symbol: str, periods: int = 20) -> Optional[float]:
        """Get latest volatility for symbol"""
        try:
            # Get recent OHLC data
            ohlc_data = self.storage.get_ohlc_data([symbol], limit=30)
            
            if ohlc_data.empty:
                return None
                
            symbol_data = ohlc_data[ohlc_data['symbol'] == symbol].sort_values('date')
            
            # Use pre-calculated volatility if available
            if 'volatility' in symbol_data.columns and not symbol_data['volatility'].isna().all():
                return float(symbol_data['volatility'].iloc[-1])
            
            # Calculate volatility from returns
            if 'returns' in symbol_data.columns and len(symbol_data) >= periods:
                vol = symbol_data['returns'].rolling(window=periods).std().iloc[-1] * np.sqrt(252)
                return float(vol)
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting volatility for {symbol}: {e}")
            return None
    
    def _calculate_max_allowed_shares(self, symbol: str, price: float) -> int:
        """Calculate maximum allowed shares based on position limits"""
        max_value_nav = self.nav * self.max_single_position
        max_shares_nav = int(max_value_nav / price)
        
        # Could add sector limits here if we track current positions
        
        return max_shares_nav
    
    def calculate_pyramid_size(self, symbol: str, price: float, 
                             existing_shares: int, levels: int = 3) -> int:
        """Calculate size for pyramiding into existing position"""
        # Base size for new entry
        base_size = self.calculate_size(symbol, price)
        
        # Reduce size for each pyramid level
        # Level 1: 100%, Level 2: 50%, Level 3: 25%
        current_level = min(levels, 3)  # Max 3 levels
        pyramid_scalar = 1.0 / (2 ** (current_level - 1))
        
        pyramid_size = int(base_size * pyramid_scalar)
        
        # Ensure total position doesn't exceed limits
        total_shares = existing_shares + pyramid_size
        max_allowed = self._calculate_max_allowed_shares(symbol, price)
        
        if abs(total_shares) > max_allowed:
            pyramid_size = max_allowed - abs(existing_shares)
            pyramid_size = max(0, pyramid_size)  # No negative pyramid
        
        logger.info(f"Pyramid size for {symbol}: {pyramid_size} shares "
                   f"(level {current_level}, existing {existing_shares})")
        
        return pyramid_size
    
    def get_sizing_summary(self, symbol: str, price: float) -> Dict[str, Any]:
        """Get sizing summary with multiple methods for comparison"""
        try:
            summary = {
                "symbol": symbol,
                "price": price,
                "nav": self.nav,
                "sizing_methods": {}
            }
            
            # Calculate with different methods
            methods = [
                SizingMethod.FIXED_PERCENT,
                SizingMethod.ATR_BASED,
                SizingMethod.VOLATILITY_BASED
            ]
            
            for method in methods:
                try:
                    shares = self.calculate_size(symbol, price, method=method)
                    position_value = abs(shares * price)
                    nav_pct = position_value / self.nav
                    
                    summary["sizing_methods"][method.value] = {
                        "shares": shares,
                        "position_value": position_value,
                        "nav_percentage": nav_pct
                    }
                except Exception as e:
                    summary["sizing_methods"][method.value] = {"error": str(e)}
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating sizing summary for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def close(self):
        """Close database connections"""
        if hasattr(self, 'storage'):
            self.storage.close()