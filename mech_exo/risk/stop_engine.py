"""
Stop loss engine for generating various types of stops
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
from .base import BaseStopEngine, StopType, StopCalculationError

logger = logging.getLogger(__name__)


class StopEngine(BaseStopEngine):
    """Main stop loss engine with multiple stop types"""
    
    def __init__(self, risk_config_path: str = "config/risk_limits.yml"):
        # Load risk configuration
        config_manager = ConfigManager()
        risk_config = config_manager.load_config("risk_limits")
        
        if not risk_config:
            raise ValueError(f"Could not load risk configuration from {risk_config_path}")
        
        super().__init__(risk_config)
        
        # Stop configuration
        stops_config = risk_config.get("stops", {})
        
        self.trailing_stop_pct = stops_config.get("trailing_stop_pct", 0.25)  # 25%
        self.hard_stop_pct = stops_config.get("hard_stop_pct", 0.15)        # 15%
        self.profit_target_pct = stops_config.get("profit_target_pct", 0.30) # 30%
        self.time_stop_days = stops_config.get("time_stop_days", 60)         # 60 days
        
        # Volatility-based stops
        self.vol_stop_multiplier = stops_config.get("vol_stop_multiplier", 2.0)  # 2x ATR
        
        logger.info(f"StopEngine initialized: trailing={self.trailing_stop_pct:.1%}, "
                   f"hard={self.hard_stop_pct:.1%}, profit={self.profit_target_pct:.1%}")
    
    def generate_stops(self, entry_price: float, position_type: str = "long", 
                      entry_date: Optional[datetime] = None, 
                      atr: Optional[float] = None,
                      **kwargs) -> Dict[str, float]:
        """
        Generate comprehensive stop loss levels
        
        Args:
            entry_price: Entry price for the position
            position_type: "long" or "short"
            entry_date: Entry date for time stops
            atr: Average True Range for volatility stops
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with stop levels for different stop types
        """
        if entry_price <= 0:
            raise StopCalculationError(f"Invalid entry price: {entry_price}")
            
        if position_type not in ["long", "short"]:
            raise StopCalculationError(f"Invalid position type: {position_type}")
        
        stops = {}
        
        try:
            # Hard stop loss
            stops["hard_stop"] = self._calculate_hard_stop(entry_price, position_type)
            
            # Initial trailing stop (same as hard stop initially)
            stops["trailing_stop"] = stops["hard_stop"]
            
            # Profit target
            stops["profit_target"] = self._calculate_profit_target(entry_price, position_type)
            
            # Time stop
            if entry_date:
                stops["time_stop_date"] = self._calculate_time_stop(entry_date)
            
            # Volatility-based stop
            if atr:
                stops["volatility_stop"] = self._calculate_volatility_stop(
                    entry_price, position_type, atr)
            
            # Risk/reward ratio
            stops["risk_reward_ratio"] = self._calculate_risk_reward_ratio(
                entry_price, stops["hard_stop"], stops["profit_target"])
            
            logger.debug(f"Generated stops for {position_type} @ {entry_price:.2f}: {stops}")
            
            return stops
            
        except Exception as e:
            logger.error(f"Stop generation failed: {e}")
            raise StopCalculationError(f"Stop generation failed: {e}")
    
    def _calculate_hard_stop(self, entry_price: float, position_type: str) -> float:
        """Calculate hard stop loss level"""
        if position_type == "long":
            return entry_price * (1 - self.hard_stop_pct)
        else:  # short
            return entry_price * (1 + self.hard_stop_pct)
    
    def _calculate_profit_target(self, entry_price: float, position_type: str) -> float:
        """Calculate profit target level"""
        if position_type == "long":
            return entry_price * (1 + self.profit_target_pct)
        else:  # short
            return entry_price * (1 - self.profit_target_pct)
    
    def _calculate_time_stop(self, entry_date: datetime) -> datetime:
        """Calculate time stop date"""
        return entry_date + timedelta(days=self.time_stop_days)
    
    def _calculate_volatility_stop(self, entry_price: float, position_type: str, 
                                 atr: float) -> float:
        """Calculate volatility-based stop using ATR"""
        stop_distance = atr * self.vol_stop_multiplier
        
        if position_type == "long":
            return entry_price - stop_distance
        else:  # short
            return entry_price + stop_distance
    
    def _calculate_risk_reward_ratio(self, entry_price: float, stop_price: float, 
                                   target_price: float) -> float:
        """Calculate risk/reward ratio"""
        risk = abs(entry_price - stop_price)
        reward = abs(target_price - entry_price)
        
        if risk == 0:
            return float('inf')
            
        return reward / risk
    
    def update_trailing_stop(self, current_price: float, current_stop: float, 
                           position_type: str, high_water_mark: Optional[float] = None) -> float:
        """
        Update trailing stop based on current price
        
        Args:
            current_price: Current market price
            current_stop: Current stop loss level
            position_type: "long" or "short"
            high_water_mark: Highest price since entry (for longs)
            
        Returns:
            Updated trailing stop level
        """
        try:
            if position_type == "long":
                # For long positions, trail stop up as price rises
                reference_price = high_water_mark if high_water_mark else current_price
                new_stop = reference_price * (1 - self.trailing_stop_pct)
                
                # Only move stop up, never down
                return max(current_stop, new_stop)
                
            else:  # short
                # For short positions, trail stop down as price falls
                low_water_mark = high_water_mark  # Reuse parameter as low water mark
                reference_price = low_water_mark if low_water_mark else current_price
                new_stop = reference_price * (1 + self.trailing_stop_pct)
                
                # Only move stop down, never up
                return min(current_stop, new_stop)
                
        except Exception as e:
            logger.error(f"Trailing stop update failed: {e}")
            return current_stop  # Return current stop if update fails
    
    def check_stop_hit(self, current_price: float, stops: Dict[str, float], 
                      position_type: str, current_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Check if any stops have been hit
        
        Args:
            current_price: Current market price
            stops: Dictionary of stop levels
            position_type: "long" or "short"
            current_date: Current date for time stop check
            
        Returns:
            Dictionary with stop status and triggered stops
        """
        triggered_stops = []
        stop_status = "active"
        exit_reason = None
        
        try:
            if position_type == "long":
                # Check if price hit any downside stops
                if "hard_stop" in stops and current_price <= stops["hard_stop"]:
                    triggered_stops.append("hard_stop")
                    exit_reason = f"Hard stop hit at {stops['hard_stop']:.2f}"
                
                if "trailing_stop" in stops and current_price <= stops["trailing_stop"]:
                    triggered_stops.append("trailing_stop")
                    exit_reason = f"Trailing stop hit at {stops['trailing_stop']:.2f}"
                
                if "volatility_stop" in stops and current_price <= stops["volatility_stop"]:
                    triggered_stops.append("volatility_stop")
                    exit_reason = f"Volatility stop hit at {stops['volatility_stop']:.2f}"
                
                # Check if price hit profit target
                if "profit_target" in stops and current_price >= stops["profit_target"]:
                    triggered_stops.append("profit_target")
                    exit_reason = f"Profit target hit at {stops['profit_target']:.2f}"
                    
            else:  # short
                # Check if price hit any upside stops
                if "hard_stop" in stops and current_price >= stops["hard_stop"]:
                    triggered_stops.append("hard_stop")
                    exit_reason = f"Hard stop hit at {stops['hard_stop']:.2f}"
                
                if "trailing_stop" in stops and current_price >= stops["trailing_stop"]:
                    triggered_stops.append("trailing_stop")
                    exit_reason = f"Trailing stop hit at {stops['trailing_stop']:.2f}"
                
                if "volatility_stop" in stops and current_price >= stops["volatility_stop"]:
                    triggered_stops.append("volatility_stop")
                    exit_reason = f"Volatility stop hit at {stops['volatility_stop']:.2f}"
                
                # Check if price hit profit target
                if "profit_target" in stops and current_price <= stops["profit_target"]:
                    triggered_stops.append("profit_target")
                    exit_reason = f"Profit target hit at {stops['profit_target']:.2f}"
            
            # Check time stop
            if (current_date and "time_stop_date" in stops and 
                current_date >= stops["time_stop_date"]):
                triggered_stops.append("time_stop")
                exit_reason = f"Time stop hit on {stops['time_stop_date'].date()}"
            
            # Determine overall status
            if triggered_stops:
                stop_status = "triggered"
                
            return {
                "status": stop_status,
                "triggered_stops": triggered_stops,
                "exit_reason": exit_reason,
                "current_price": current_price,
                "stops": stops
            }
            
        except Exception as e:
            logger.error(f"Stop check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "current_price": current_price
            }
    
    def calculate_stop_distances(self, entry_price: float, stops: Dict[str, float], 
                               position_type: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate distances and percentages for all stops
        
        Returns:
            Dictionary with stop distances and percentages
        """
        distances = {}
        
        for stop_name, stop_price in stops.items():
            if stop_name.endswith("_date"):  # Skip date fields
                continue
                
            if isinstance(stop_price, (int, float)):
                distance = abs(entry_price - stop_price)
                percentage = distance / entry_price
                
                distances[stop_name] = {
                    "price": stop_price,
                    "distance": distance,
                    "percentage": percentage,
                    "direction": "up" if stop_price > entry_price else "down"
                }
        
        return distances
    
    def optimize_stops_for_symbol(self, symbol: str, entry_price: float, 
                                position_type: str, 
                                volatility: Optional[float] = None,
                                beta: Optional[float] = None) -> Dict[str, float]:
        """
        Optimize stop levels based on symbol characteristics
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_type: Position type
            volatility: Historical volatility
            beta: Beta vs market
            
        Returns:
            Optimized stop levels
        """
        # Start with standard stops
        stops = self.generate_stops(entry_price, position_type)
        
        try:
            # Adjust stops based on volatility
            if volatility:
                vol_adjustment = self._calculate_volatility_adjustment(volatility)
                
                # Widen stops for high volatility stocks
                if volatility > 0.30:  # > 30% volatility
                    stops["hard_stop"] = self._adjust_stop_for_volatility(
                        entry_price, stops["hard_stop"], position_type, vol_adjustment)
                    stops["trailing_stop"] = stops["hard_stop"]
            
            # Adjust stops based on beta
            if beta:
                beta_adjustment = self._calculate_beta_adjustment(beta)
                
                # Adjust for high beta stocks
                if abs(beta) > 1.5:
                    stops["hard_stop"] = self._adjust_stop_for_beta(
                        entry_price, stops["hard_stop"], position_type, beta_adjustment)
                    stops["trailing_stop"] = stops["hard_stop"]
            
            logger.info(f"Optimized stops for {symbol}: vol={volatility}, beta={beta}")
            
            return stops
            
        except Exception as e:
            logger.error(f"Stop optimization failed for {symbol}: {e}")
            return stops  # Return standard stops if optimization fails
    
    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """Calculate adjustment factor based on volatility"""
        # Base adjustment: 1.0 for 20% vol, scale linearly
        base_vol = 0.20
        return max(0.5, min(2.0, volatility / base_vol))
    
    def _calculate_beta_adjustment(self, beta: float) -> float:
        """Calculate adjustment factor based on beta"""
        # Adjust based on absolute beta
        return max(0.7, min(1.5, abs(beta)))
    
    def _adjust_stop_for_volatility(self, entry_price: float, current_stop: float, 
                                  position_type: str, vol_adjustment: float) -> float:
        """Adjust stop based on volatility"""
        stop_distance = abs(entry_price - current_stop)
        adjusted_distance = stop_distance * vol_adjustment
        
        if position_type == "long":
            return entry_price - adjusted_distance
        else:
            return entry_price + adjusted_distance
    
    def _adjust_stop_for_beta(self, entry_price: float, current_stop: float, 
                            position_type: str, beta_adjustment: float) -> float:
        """Adjust stop based on beta"""
        stop_distance = abs(entry_price - current_stop)
        adjusted_distance = stop_distance * beta_adjustment
        
        if position_type == "long":
            return entry_price - adjusted_distance
        else:
            return entry_price + adjusted_distance
    
    def get_stop_summary(self, entry_price: float, position_type: str, 
                        entry_date: Optional[datetime] = None,
                        atr: Optional[float] = None) -> Dict[str, Any]:
        """Get comprehensive stop summary for analysis"""
        try:
            stops = self.generate_stops(entry_price, position_type, entry_date, atr)
            distances = self.calculate_stop_distances(entry_price, stops, position_type)
            
            summary = {
                "entry_price": entry_price,
                "position_type": position_type,
                "stops": stops,
                "distances": distances,
                "configuration": {
                    "trailing_stop_pct": self.trailing_stop_pct,
                    "hard_stop_pct": self.hard_stop_pct,
                    "profit_target_pct": self.profit_target_pct,
                    "time_stop_days": self.time_stop_days
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating stop summary: {e}")
            return {"error": str(e)}