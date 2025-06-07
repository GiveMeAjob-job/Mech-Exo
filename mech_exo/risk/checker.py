"""
Risk checker for monitoring portfolio exposure vs limits
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mech_exo.utils import ConfigManager
from mech_exo.datasource import DataStorage
from .base import BaseRiskChecker, RiskStatus, RiskViolationError

logger = logging.getLogger(__name__)


class Position:
    """Represents a trading position"""
    
    def __init__(self, symbol: str, shares: int, entry_price: float, 
                 current_price: float, entry_date: datetime,
                 sector: Optional[str] = None):
        self.symbol = symbol
        self.shares = shares  # Positive for long, negative for short
        self.entry_price = entry_price
        self.current_price = current_price
        self.entry_date = entry_date
        self.sector = sector or "Unknown"
        
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return abs(self.shares) * self.current_price
    
    @property
    def notional_value(self) -> float:
        """Notional value (includes direction)"""
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L"""
        return self.shares * (self.current_price - self.entry_price)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L percentage"""
        return (self.current_price / self.entry_price - 1) * np.sign(self.shares)
    
    @property
    def position_type(self) -> str:
        """Position type"""
        return "long" if self.shares > 0 else "short"
    
    @property
    def days_held(self) -> int:
        """Days position has been held"""
        return (datetime.now() - self.entry_date).days


class Portfolio:
    """Represents current portfolio state"""
    
    def __init__(self, nav: float):
        self.nav = nav
        self.positions: Dict[str, Position] = {}
        self.cash = nav  # Start with all cash
        
    def add_position(self, position: Position):
        """Add or update position"""
        if position.symbol in self.positions:
            # Update existing position (could be adding to or reducing)
            existing = self.positions[position.symbol]
            total_shares = existing.shares + position.shares
            
            if total_shares == 0:
                # Position closed
                del self.positions[position.symbol]
            else:
                # Update position with weighted average entry price
                total_cost = (existing.shares * existing.entry_price + 
                            position.shares * position.entry_price)
                avg_entry = total_cost / total_shares
                
                existing.shares = total_shares
                existing.entry_price = avg_entry
                existing.current_price = position.current_price
        else:
            self.positions[position.symbol] = position
        
        # Update cash
        self.cash -= position.shares * position.entry_price
    
    @property
    def total_market_value(self) -> float:
        """Total market value of all positions"""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_notional_value(self) -> float:
        """Total notional value (long - short)"""
        return sum(pos.notional_value for pos in self.positions.values())
    
    @property
    def gross_exposure(self) -> float:
        """Gross exposure (long + short)"""
        return sum(abs(pos.notional_value) for pos in self.positions.values())
    
    @property
    def net_exposure(self) -> float:
        """Net exposure (long - short)"""
        return self.total_notional_value
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def current_nav(self) -> float:
        """Current NAV including unrealized P&L"""
        return self.nav + self.total_unrealized_pnl
    
    def get_sector_exposure(self) -> Dict[str, float]:
        """Get exposure by sector"""
        sector_exposure = defaultdict(float)
        for pos in self.positions.values():
            sector_exposure[pos.sector] += abs(pos.notional_value)
        return dict(sector_exposure)
    
    def get_positions_summary(self) -> pd.DataFrame:
        """Get positions as DataFrame"""
        if not self.positions:
            return pd.DataFrame()
            
        data = []
        for pos in self.positions.values():
            data.append({
                'symbol': pos.symbol,
                'shares': pos.shares,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'sector': pos.sector,
                'position_type': pos.position_type,
                'days_held': pos.days_held
            })
        
        return pd.DataFrame(data)


class RiskChecker(BaseRiskChecker):
    """Main risk checking engine"""
    
    def __init__(self, portfolio: Portfolio, risk_config_path: str = "config/risk_limits.yml"):
        # Load risk configuration
        config_manager = ConfigManager()
        risk_config = config_manager.load_config("risk_limits")
        
        if not risk_config:
            raise ValueError(f"Could not load risk configuration from {risk_config_path}")
        
        super().__init__(risk_config)
        
        self.portfolio = portfolio
        self.storage = DataStorage()
        
        # Extract risk limits
        self.position_limits = risk_config.get("position_sizing", {})
        self.portfolio_limits = risk_config.get("portfolio", {})
        self.option_limits = risk_config.get("options", {})
        self.stop_limits = risk_config.get("stops", {})
        self.cost_limits = risk_config.get("costs", {})
        self.operational_limits = risk_config.get("operational", {})
        self.volatility_limits = risk_config.get("volatility", {})
        self.margin_limits = risk_config.get("margin", {})
        
        logger.info(f"RiskChecker initialized for portfolio NAV=${portfolio.nav:,.0f}")
    
    def check(self, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive risk check"""
        try:
            risk_report = {
                "timestamp": datetime.now(),
                "portfolio_nav": self.portfolio.current_nav,
                "status": RiskStatus.OK,
                "checks": {},
                "violations": [],
                "warnings": [],
                "summary": {}
            }
            
            # Run all risk checks
            checks = [
                self._check_position_limits,
                self._check_portfolio_limits,
                self._check_sector_limits,
                self._check_concentration_limits,
                self._check_drawdown_limits,
                self._check_leverage_limits,
                self._check_volatility_limits
            ]
            
            for check_func in checks:
                check_name = check_func.__name__[7:]  # Remove '_check_' prefix
                check_result = check_func()
                risk_report["checks"][check_name] = check_result
                
                # Collect violations and warnings
                if check_result["status"] == RiskStatus.BREACH:
                    risk_report["violations"].extend(check_result.get("violations", []))
                elif check_result["status"] == RiskStatus.WARNING:
                    risk_report["warnings"].extend(check_result.get("warnings", []))
            
            # Determine overall status
            if risk_report["violations"]:
                risk_report["status"] = RiskStatus.BREACH
            elif risk_report["warnings"]:
                risk_report["status"] = RiskStatus.WARNING
            
            # Generate summary
            risk_report["summary"] = self._generate_risk_summary()
            
            logger.info(f"Risk check completed: {risk_report['status'].value}")
            
            return risk_report
            
        except Exception as e:
            logger.error(f"Risk check failed: {e}")
            return {
                "timestamp": datetime.now(),
                "status": RiskStatus.CRITICAL,
                "error": str(e)
            }
    
    def _check_position_limits(self) -> Dict[str, Any]:
        """Check individual position limits"""
        violations = []
        warnings = []
        
        max_single_risk = self.position_limits.get("max_single_trade_risk", 0.02)
        max_single_position = self.position_limits.get("max_single_position", 0.10)
        min_position_size = self.position_limits.get("min_position_size", 100)
        
        for pos in self.portfolio.positions.values():
            position_pct = pos.market_value / self.portfolio.current_nav
            
            # Check maximum position size
            if position_pct > max_single_position:
                violations.append(
                    f"{pos.symbol}: Position size {position_pct:.2%} exceeds limit {max_single_position:.2%}"
                )
            elif position_pct > max_single_position * 0.8:  # 80% of limit
                warnings.append(
                    f"{pos.symbol}: Position size {position_pct:.2%} approaching limit {max_single_position:.2%}"
                )
            
            # Check minimum position size
            if pos.market_value < min_position_size:
                warnings.append(
                    f"{pos.symbol}: Position value ${pos.market_value:.0f} below minimum ${min_position_size}"
                )
        
        status = RiskStatus.BREACH if violations else (RiskStatus.WARNING if warnings else RiskStatus.OK)
        
        return {
            "status": status,
            "violations": violations,
            "warnings": warnings,
            "metrics": {
                "position_count": len(self.portfolio.positions),
                "max_position_pct": max(
                    (pos.market_value / self.portfolio.current_nav for pos in self.portfolio.positions.values()),
                    default=0
                ),
                "limits": {
                    "max_single_position": max_single_position,
                    "min_position_size": min_position_size
                }
            }
        }
    
    def _check_portfolio_limits(self) -> Dict[str, Any]:
        """Check portfolio-level limits"""
        violations = []
        warnings = []
        
        max_gross_exposure = self.portfolio_limits.get("max_gross_exposure", 1.5)
        max_net_exposure = self.portfolio_limits.get("max_net_exposure", 1.0)
        
        current_gross = self.portfolio.gross_exposure / self.portfolio.current_nav
        current_net = abs(self.portfolio.net_exposure) / self.portfolio.current_nav
        
        # Check gross exposure
        if current_gross > max_gross_exposure:
            violations.append(
                f"Gross exposure {current_gross:.2%} exceeds limit {max_gross_exposure:.2%}"
            )
        elif current_gross > max_gross_exposure * 0.9:
            warnings.append(
                f"Gross exposure {current_gross:.2%} approaching limit {max_gross_exposure:.2%}"
            )
        
        # Check net exposure
        if current_net > max_net_exposure:
            violations.append(
                f"Net exposure {current_net:.2%} exceeds limit {max_net_exposure:.2%}"
            )
        elif current_net > max_net_exposure * 0.9:
            warnings.append(
                f"Net exposure {current_net:.2%} approaching limit {max_net_exposure:.2%}"
            )
        
        status = RiskStatus.BREACH if violations else (RiskStatus.WARNING if warnings else RiskStatus.OK)
        
        return {
            "status": status,
            "violations": violations,
            "warnings": warnings,
            "metrics": {
                "gross_exposure": current_gross,
                "net_exposure": current_net,
                "total_market_value": self.portfolio.total_market_value,
                "limits": {
                    "max_gross_exposure": max_gross_exposure,
                    "max_net_exposure": max_net_exposure
                }
            }
        }
    
    def _check_sector_limits(self) -> Dict[str, Any]:
        """Check sector concentration limits"""
        violations = []
        warnings = []
        
        max_sector_exposure = self.portfolio_limits.get("max_sector_exposure", 0.20)
        sector_exposures = self.portfolio.get_sector_exposure()
        
        for sector, exposure in sector_exposures.items():
            exposure_pct = exposure / self.portfolio.current_nav
            
            if exposure_pct > max_sector_exposure:
                violations.append(
                    f"{sector} sector exposure {exposure_pct:.2%} exceeds limit {max_sector_exposure:.2%}"
                )
            elif exposure_pct > max_sector_exposure * 0.8:
                warnings.append(
                    f"{sector} sector exposure {exposure_pct:.2%} approaching limit {max_sector_exposure:.2%}"
                )
        
        status = RiskStatus.BREACH if violations else (RiskStatus.WARNING if warnings else RiskStatus.OK)
        
        return {
            "status": status,
            "violations": violations,
            "warnings": warnings,
            "metrics": {
                "sector_exposures": {
                    sector: exposure / self.portfolio.current_nav 
                    for sector, exposure in sector_exposures.items()
                },
                "limits": {
                    "max_sector_exposure": max_sector_exposure
                }
            }
        }
    
    def _check_concentration_limits(self) -> Dict[str, Any]:
        """Check concentration risk"""
        violations = []
        warnings = []
        
        correlation_limit = self.portfolio_limits.get("correlation_limit", 0.7)
        
        # Check for correlated positions (simplified - could enhance with actual correlation data)
        tech_symbols = [pos.symbol for pos in self.portfolio.positions.values() 
                       if pos.sector == "Technology"]
        
        if len(tech_symbols) > 5:  # Many tech positions
            warnings.append(
                f"High concentration in Technology sector: {len(tech_symbols)} positions"
            )
        
        status = RiskStatus.WARNING if warnings else RiskStatus.OK
        
        return {
            "status": status,
            "violations": violations,
            "warnings": warnings,
            "metrics": {
                "sector_position_counts": self._get_sector_position_counts(),
                "limits": {
                    "correlation_limit": correlation_limit
                }
            }
        }
    
    def _check_drawdown_limits(self) -> Dict[str, Any]:
        """Check drawdown limits"""
        violations = []
        warnings = []
        
        max_drawdown = self.portfolio_limits.get("max_drawdown", 0.10)
        
        # Calculate current drawdown from initial NAV
        current_drawdown = max(0, (self.portfolio.nav - self.portfolio.current_nav) / self.portfolio.nav)
        
        if current_drawdown > max_drawdown:
            violations.append(
                f"Portfolio drawdown {current_drawdown:.2%} exceeds limit {max_drawdown:.2%}"
            )
        elif current_drawdown > max_drawdown * 0.7:
            warnings.append(
                f"Portfolio drawdown {current_drawdown:.2%} approaching limit {max_drawdown:.2%}"
            )
        
        status = RiskStatus.BREACH if violations else (RiskStatus.WARNING if warnings else RiskStatus.OK)
        
        return {
            "status": status,
            "violations": violations,
            "warnings": warnings,
            "metrics": {
                "current_drawdown": current_drawdown,
                "initial_nav": self.portfolio.nav,
                "current_nav": self.portfolio.current_nav,
                "unrealized_pnl": self.portfolio.total_unrealized_pnl,
                "limits": {
                    "max_drawdown": max_drawdown
                }
            }
        }
    
    def _check_leverage_limits(self) -> Dict[str, Any]:
        """Check leverage limits"""
        violations = []
        warnings = []
        
        # Portfolio leverage is gross exposure / NAV
        current_leverage = self.portfolio.gross_exposure / self.portfolio.current_nav
        max_leverage = self.portfolio_limits.get("max_gross_exposure", 1.5)
        
        if current_leverage > max_leverage:
            violations.append(
                f"Portfolio leverage {current_leverage:.2f}x exceeds limit {max_leverage:.2f}x"
            )
        elif current_leverage > max_leverage * 0.9:
            warnings.append(
                f"Portfolio leverage {current_leverage:.2f}x approaching limit {max_leverage:.2f}x"
            )
        
        status = RiskStatus.BREACH if violations else (RiskStatus.WARNING if warnings else RiskStatus.OK)
        
        return {
            "status": status,
            "violations": violations,
            "warnings": warnings,
            "metrics": {
                "current_leverage": current_leverage,
                "gross_exposure": self.portfolio.gross_exposure,
                "limits": {
                    "max_leverage": max_leverage
                }
            }
        }
    
    def _check_volatility_limits(self) -> Dict[str, Any]:
        """Check portfolio volatility limits"""
        warnings = []
        
        max_portfolio_vol = self.volatility_limits.get("max_portfolio_vol", 0.20)
        
        # Simplified portfolio volatility check
        # In practice, would calculate weighted average volatility
        high_vol_positions = [
            pos.symbol for pos in self.portfolio.positions.values()
            if self._get_symbol_volatility(pos.symbol) > 0.40  # > 40% vol
        ]
        
        if high_vol_positions:
            warnings.append(
                f"High volatility positions detected: {', '.join(high_vol_positions)}"
            )
        
        status = RiskStatus.WARNING if warnings else RiskStatus.OK
        
        return {
            "status": status,
            "violations": [],
            "warnings": warnings,
            "metrics": {
                "high_vol_positions": high_vol_positions,
                "limits": {
                    "max_portfolio_vol": max_portfolio_vol
                }
            }
        }
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """Get volatility for symbol (simplified)"""
        try:
            # Get recent OHLC data
            ohlc_data = self.storage.get_ohlc_data([symbol], limit=5)
            if not ohlc_data.empty and 'volatility' in ohlc_data.columns:
                return float(ohlc_data['volatility'].iloc[-1])
            return 0.20  # Default volatility
        except Exception:
            return 0.20
    
    def _get_sector_position_counts(self) -> Dict[str, int]:
        """Get count of positions by sector"""
        sector_counts = defaultdict(int)
        for pos in self.portfolio.positions.values():
            sector_counts[pos.sector] += 1
        return dict(sector_counts)
    
    def _generate_risk_summary(self) -> Dict[str, Any]:
        """Generate risk summary metrics"""
        return {
            "portfolio_metrics": {
                "nav": self.portfolio.current_nav,
                "positions": len(self.portfolio.positions),
                "gross_exposure": self.portfolio.gross_exposure,
                "net_exposure": self.portfolio.net_exposure,
                "unrealized_pnl": self.portfolio.total_unrealized_pnl,
                "cash": self.portfolio.cash
            },
            "risk_utilization": {
                "gross_exposure_pct": self.portfolio.gross_exposure / self.portfolio.current_nav,
                "max_position_pct": max(
                    (pos.market_value / self.portfolio.current_nav for pos in self.portfolio.positions.values()),
                    default=0
                ),
                "sector_concentration": max(
                    (exp / self.portfolio.current_nav for exp in self.portfolio.get_sector_exposure().values()),
                    default=0
                )
            }
        }
    
    def check_new_position(self, symbol: str, shares: int, price: float, 
                          sector: Optional[str] = None) -> Dict[str, Any]:
        """Check if new position would violate risk limits"""
        # Create temporary position
        temp_position = Position(
            symbol=symbol,
            shares=shares,
            entry_price=price,
            current_price=price,
            entry_date=datetime.now(),
            sector=sector
        )
        
        # Create temporary portfolio with new position
        temp_portfolio = Portfolio(self.portfolio.nav)
        temp_portfolio.positions = self.portfolio.positions.copy()
        temp_portfolio.cash = self.portfolio.cash
        temp_portfolio.add_position(temp_position)
        
        # Create temporary risk checker
        temp_checker = RiskChecker(temp_portfolio)
        
        # Run risk check on temporary portfolio
        risk_result = temp_checker.check()
        
        # Add pre-trade analysis
        position_value = abs(shares * price)
        position_pct = position_value / self.portfolio.current_nav
        
        risk_result["pre_trade_analysis"] = {
            "symbol": symbol,
            "position_value": position_value,
            "position_pct": position_pct,
            "would_violate": risk_result["status"] == RiskStatus.BREACH,
            "recommendation": "REJECT" if risk_result["status"] == RiskStatus.BREACH else "APPROVE"
        }
        
        return risk_result
    
    def get_risk_status_summary(self) -> str:
        """Get concise risk status for CLI display"""
        try:
            risk_report = self.check()
            
            if risk_report["status"] == RiskStatus.OK:
                return "✅ All risk checks PASSED"
            elif risk_report["status"] == RiskStatus.WARNING:
                warnings = risk_report.get("warnings", [])
                return f"⚠️  {len(warnings)} WARNING(S): {'; '.join(warnings[:2])}"
            elif risk_report["status"] == RiskStatus.BREACH:
                violations = risk_report.get("violations", [])
                return f"❌ {len(violations)} VIOLATION(S): {'; '.join(violations[:2])}"
            else:
                return "🔥 CRITICAL RISK STATUS"
                
        except Exception as e:
            return f"❌ Risk check failed: {e}"
    
    def close(self):
        """Close database connections"""
        if hasattr(self, 'storage'):
            self.storage.close()