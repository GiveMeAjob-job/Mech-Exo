"""
Individual factor calculation implementations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from .base import BaseFactor, FactorCalculationError

logger = logging.getLogger(__name__)


class PERatioFactor(BaseFactor):
    """P/E Ratio factor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="pe_ratio",
            weight=config.get("weight", 15),
            direction=config.get("direction", "lower_better")
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate P/E ratio factor scores"""
        if 'pe_ratio' not in data.columns:
            raise FactorCalculationError("pe_ratio column not found")
            
        pe_ratios = data['pe_ratio'].copy()
        
        # Filter out extreme values
        pe_ratios = pe_ratios[(pe_ratios > 0) & (pe_ratios < 100)]
        
        return pe_ratios


class PriceToBookFactor(BaseFactor):
    """Price-to-Book factor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="price_to_book",
            weight=config.get("weight", 10),
            direction=config.get("direction", "lower_better")
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Price-to-Book factor scores"""
        if 'price_to_book' not in data.columns:
            raise FactorCalculationError("price_to_book column not found")
            
        pb_ratios = data['price_to_book'].copy()
        
        # Filter out extreme values
        pb_ratios = pb_ratios[(pb_ratios > 0) & (pb_ratios < 20)]
        
        return pb_ratios


class ROEFactor(BaseFactor):
    """Return on Equity factor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="return_on_equity", 
            weight=config.get("weight", 18),
            direction=config.get("direction", "higher_better")
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ROE factor scores"""
        if 'return_on_equity' not in data.columns:
            raise FactorCalculationError("return_on_equity column not found")
            
        roe = data['return_on_equity'].copy()
        
        # Convert to percentage if needed and filter extreme values
        if roe.max() <= 1:  # Assuming decimal format
            roe = roe * 100
            
        roe = roe[(roe > -50) & (roe < 100)]  # Filter extreme values
        
        return roe


class RevenueGrowthFactor(BaseFactor):
    """Revenue Growth factor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="revenue_growth",
            weight=config.get("weight", 15),
            direction=config.get("direction", "higher_better")
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Revenue Growth factor scores"""
        if 'revenue_growth' not in data.columns:
            raise FactorCalculationError("revenue_growth column not found")
            
        growth = data['revenue_growth'].copy()
        
        # Convert to percentage if needed
        if growth.max() <= 1:
            growth = growth * 100
            
        # Filter extreme values
        growth = growth[(growth > -100) & (growth < 500)]
        
        return growth


class EarningsGrowthFactor(BaseFactor):
    """Earnings Growth factor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="earnings_growth",
            weight=config.get("weight", 20),
            direction=config.get("direction", "higher_better")
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Earnings Growth factor scores"""
        if 'earnings_growth' not in data.columns:
            raise FactorCalculationError("earnings_growth column not found")
            
        growth = data['earnings_growth'].copy()
        
        # Convert to percentage if needed
        if growth.max() <= 1:
            growth = growth * 100
            
        # Filter extreme values
        growth = growth[(growth > -100) & (growth < 1000)]
        
        return growth


class DebtToEquityFactor(BaseFactor):
    """Debt-to-Equity factor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="debt_to_equity",
            weight=config.get("weight", 12),
            direction=config.get("direction", "lower_better")
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Debt-to-Equity factor scores"""
        if 'debt_to_equity' not in data.columns:
            raise FactorCalculationError("debt_to_equity column not found")
            
        de_ratio = data['debt_to_equity'].copy()
        
        # Filter out extreme values
        de_ratio = de_ratio[(de_ratio >= 0) & (de_ratio < 10)]
        
        return de_ratio


class RSIFactor(BaseFactor):
    """RSI technical factor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="rsi_14",
            weight=config.get("weight", 8),
            direction=config.get("direction", "mean_revert")
        )
        self.oversold_threshold = config.get("oversold_threshold", 30)
        self.overbought_threshold = config.get("overbought_threshold", 70)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI factor scores"""
        if 'close' not in data.columns:
            raise FactorCalculationError("close price column not found")
            
        # Calculate RSI for each symbol
        rsi_values = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].sort_values('date')
            if len(symbol_data) >= 14:
                rsi = self._calculate_rsi(symbol_data['close'])
                rsi_values.append({
                    'symbol': symbol,
                    'rsi': rsi.iloc[-1] if not rsi.empty else 50  # Use latest RSI
                })
        
        rsi_df = pd.DataFrame(rsi_values)
        
        if rsi_df.empty:
            return pd.Series(dtype=float)
            
        # Merge back with original data
        merged = data.merge(rsi_df, on='symbol', how='left')
        
        return merged['rsi'].fillna(50)  # Neutral RSI for missing values
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class MomentumFactor(BaseFactor):
    """12-month momentum factor (excluding last month)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="momentum_12_1",
            weight=config.get("weight", 12),
            direction=config.get("direction", "higher_better")
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate 12-1 momentum factor scores"""
        if 'close' not in data.columns or 'date' not in data.columns:
            raise FactorCalculationError("close price or date column not found")
            
        momentum_values = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].sort_values('date')
            
            if len(symbol_data) >= 252:  # Need at least a year of data
                # 12-1 momentum: return from 12 months ago to 1 month ago
                current_price = symbol_data['close'].iloc[-22]  # 1 month ago (22 trading days)
                year_ago_price = symbol_data['close'].iloc[-252]  # 12 months ago
                
                momentum = (current_price / year_ago_price - 1) * 100
                
                momentum_values.append({
                    'symbol': symbol,
                    'momentum': momentum
                })
        
        momentum_df = pd.DataFrame(momentum_values)
        
        if momentum_df.empty:
            return pd.Series(dtype=float)
            
        # Merge back with original data
        merged = data.merge(momentum_df, on='symbol', how='left')
        
        return merged['momentum'].fillna(0)  # Neutral momentum for missing values


class VolatilityFactor(BaseFactor):
    """Volatility factor (lower is generally better for risk-adjusted returns)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="volatility_ratio",
            weight=config.get("weight", 6),
            direction=config.get("direction", "lower_better")
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volatility factor scores"""
        if 'volatility' in data.columns:
            # Use pre-calculated volatility
            return data['volatility'].fillna(data['volatility'].median())
        elif 'returns' in data.columns:
            # Calculate from returns
            vol_values = []
            
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol]
                if len(symbol_data) >= 20:
                    vol = symbol_data['returns'].std() * np.sqrt(252)  # Annualized
                    vol_values.append({
                        'symbol': symbol,
                        'volatility': vol
                    })
            
            vol_df = pd.DataFrame(vol_values)
            if vol_df.empty:
                return pd.Series(dtype=float)
                
            merged = data.merge(vol_df, on='symbol', how='left')
            return merged['volatility'].fillna(merged['volatility'].median())
        else:
            raise FactorCalculationError("Neither volatility nor returns column found")


class FactorFactory:
    """Factory for creating factor instances"""
    
    FACTOR_CLASSES = {
        'pe_ratio': PERatioFactor,
        'price_to_book': PriceToBookFactor,
        'debt_to_equity': DebtToEquityFactor,
        'return_on_equity': ROEFactor,
        'revenue_growth': RevenueGrowthFactor,
        'earnings_growth': EarningsGrowthFactor,
        'rsi_14': RSIFactor,
        'momentum_12_1': MomentumFactor,
        'volatility_ratio': VolatilityFactor
    }
    
    @classmethod
    def create_factor(cls, factor_name: str, config: Dict[str, Any]) -> BaseFactor:
        """Create a factor instance"""
        if factor_name not in cls.FACTOR_CLASSES:
            raise ValueError(f"Unknown factor: {factor_name}")
            
        factor_class = cls.FACTOR_CLASSES[factor_name]
        return factor_class(config)
    
    @classmethod
    def create_all_factors(cls, factor_config: Dict[str, Dict[str, Any]]) -> Dict[str, BaseFactor]:
        """Create all factors from configuration"""
        factors = {}
        
        for factor_name, config in factor_config.items():
            if factor_name in cls.FACTOR_CLASSES:
                try:
                    factors[factor_name] = cls.create_factor(factor_name, config)
                    logger.info(f"Created factor: {factor_name}")
                except Exception as e:
                    logger.error(f"Failed to create factor {factor_name}: {e}")
                    
        return factors