"""
Rotational Alpha Signal Generator - Phase P11 Week 2

Implements a simple sector-momentum factor for multi-strategy portfolio allocation.
This signal identifies sectors with strong momentum characteristics and generates
rotation signals for tactical asset allocation.

Features:
- Sector momentum scoring based on price and volume indicators
- Dynamic sector rotation signals
- Integration with existing factor pipeline
- Risk-adjusted momentum scoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class SectorMomentumCalculator:
    """Calculates sector momentum scores for rotational alpha strategy"""
    
    def __init__(self, lookback_days: int = 20, min_volume_ratio: float = 1.2):
        self.lookback_days = lookback_days
        self.min_volume_ratio = min_volume_ratio
        
        # Standard sector mapping (GICS Level 1)
        self.sector_mapping = {
            'XLK': 'Technology',           # Technology Select Sector SPDR
            'XLF': 'Financials',           # Financial Select Sector SPDR  
            'XLV': 'Healthcare',           # Health Care Select Sector SPDR
            'XLI': 'Industrials',          # Industrial Select Sector SPDR
            'XLE': 'Energy',               # Energy Select Sector SPDR
            'XLB': 'Materials',            # Materials Select Sector SPDR
            'XLU': 'Utilities',            # Utilities Select Sector SPDR
            'XLP': 'ConsumerStaples',      # Consumer Staples Select Sector SPDR
            'XLY': 'ConsumerDiscretionary', # Consumer Discretionary Select Sector SPDR
            'XLRE': 'RealEstate',          # Real Estate Select Sector SPDR
            'XLC': 'Communication'         # Communication Services Select Sector SPDR
        }
        
        logger.info(f"📊 SectorMomentumCalculator initialized with {lookback_days}-day lookback")
        
    def generate_mock_sector_data(self, symbols: List[str], days: int = 30) -> pd.DataFrame:
        """Generate mock sector price and volume data for development"""
        
        np.random.seed(42)  # Reproducible results
        dates = pd.date_range(end=datetime.now().date(), periods=days, freq='D')
        
        data = []
        for symbol in symbols:
            # Generate realistic price series with sector-specific characteristics
            base_price = np.random.uniform(50, 200)
            daily_returns = np.random.normal(0.001, 0.02, days)  # ~0.1% daily return, 2% volatility
            
            # Add sector momentum characteristics
            sector_momentum = self._get_sector_momentum_trend(symbol, days)
            adjusted_returns = daily_returns + sector_momentum
            
            prices = base_price * np.cumprod(1 + adjusted_returns)
            
            # Generate volume data
            base_volume = np.random.uniform(1_000_000, 10_000_000)
            volume_noise = np.random.lognormal(0, 0.3, days)
            volumes = base_volume * volume_noise
            
            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'close': prices[i],
                    'volume': volumes[i],
                    'sector': self.sector_mapping.get(symbol, 'Unknown')
                })
        
        df = pd.DataFrame(data)
        logger.info(f"📊 Generated mock data: {len(df)} records for {len(symbols)} symbols")
        return df
        
    def _get_sector_momentum_trend(self, symbol: str, days: int) -> np.ndarray:
        """Generate sector-specific momentum trends"""
        
        # Sector momentum patterns (simplified)
        momentum_profiles = {
            'XLK': 0.0005,    # Technology - strong momentum
            'XLV': 0.0003,    # Healthcare - moderate momentum  
            'XLF': 0.0001,    # Financials - weak momentum
            'XLI': 0.0002,    # Industrials - moderate momentum
            'XLE': -0.0001,   # Energy - negative momentum
            'XLB': 0.0001,    # Materials - weak momentum
            'XLU': -0.0002,   # Utilities - defensive/negative
            'XLP': 0.0000,    # Consumer Staples - neutral
            'XLY': 0.0003,    # Consumer Discretionary - moderate momentum
            'XLRE': -0.0001,  # Real Estate - weak
            'XLC': 0.0002     # Communications - moderate
        }
        
        base_trend = momentum_profiles.get(symbol, 0.0)
        
        # Add some cyclical variation
        time_series = np.arange(days)
        cyclical = 0.0001 * np.sin(2 * np.pi * time_series / 10)  # 10-day cycle
        
        return np.full(days, base_trend) + cyclical
        
    def calculate_momentum_score(self, price_data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate comprehensive momentum score for a sector/symbol"""
        
        symbol_data = price_data[price_data['symbol'] == symbol].copy()
        if len(symbol_data) < self.lookback_days:
            logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} days")
            return self._default_score(symbol)
            
        symbol_data = symbol_data.sort_values('date')
        symbol_data['returns'] = symbol_data['close'].pct_change()
        
        # 1. Price momentum (returns-based)
        recent_returns = symbol_data['returns'].tail(self.lookback_days)
        price_momentum = recent_returns.mean() * np.sqrt(252)  # Annualized
        
        # 2. Price trend strength (regression slope)
        prices = symbol_data['close'].tail(self.lookback_days).values
        x = np.arange(len(prices))
        if len(prices) > 1:
            slope, _ = np.polyfit(x, prices, 1)
            trend_strength = slope / prices[0]  # Normalized by starting price
        else:
            trend_strength = 0.0
            
        # 3. Volume momentum (volume surge indicator)
        recent_volume = symbol_data['volume'].tail(self.lookback_days)
        avg_volume = symbol_data['volume'].tail(self.lookback_days * 2).head(self.lookback_days).mean()
        volume_ratio = recent_volume.mean() / avg_volume if avg_volume > 0 else 1.0
        volume_momentum = max(0, (volume_ratio - 1.0))  # Positive if volume increasing
        
        # 4. Volatility adjustment (lower vol = higher quality momentum)
        volatility = recent_returns.std() * np.sqrt(252)  # Annualized volatility
        volatility_penalty = min(volatility / 0.20, 1.0)  # Penalty if vol > 20%
        
        # 5. Composite momentum score
        momentum_score = (
            0.4 * price_momentum +
            0.3 * trend_strength + 
            0.2 * volume_momentum -
            0.1 * volatility_penalty
        )
        
        # 6. Risk-adjusted momentum (Sharpe-like ratio)
        risk_adjusted_momentum = price_momentum / volatility if volatility > 0 else 0.0
        
        score_dict = {
            'symbol': symbol,
            'sector': self.sector_mapping.get(symbol, 'Unknown'),
            'momentum_score': momentum_score,
            'price_momentum': price_momentum,
            'trend_strength': trend_strength,
            'volume_momentum': volume_momentum,
            'volatility': volatility,
            'risk_adjusted_momentum': risk_adjusted_momentum,
            'volume_ratio': volume_ratio,
            'data_quality': min(len(symbol_data) / self.lookback_days, 1.0)
        }
        
        logger.debug(f"📊 {symbol}: momentum={momentum_score:.4f}, trend={trend_strength:.4f}")
        return score_dict
        
    def _default_score(self, symbol: str) -> Dict[str, float]:
        """Return default score for symbols with insufficient data"""
        return {
            'symbol': symbol,
            'sector': self.sector_mapping.get(symbol, 'Unknown'),
            'momentum_score': 0.0,
            'price_momentum': 0.0,
            'trend_strength': 0.0,
            'volume_momentum': 0.0,
            'volatility': 0.20,  # Default 20% volatility
            'risk_adjusted_momentum': 0.0,
            'volume_ratio': 1.0,
            'data_quality': 0.0
        }
        
    def generate_rotation_signals(self, momentum_scores: List[Dict]) -> pd.DataFrame:
        """Generate sector rotation signals based on momentum scores"""
        
        df = pd.DataFrame(momentum_scores)
        
        # Rank sectors by momentum score
        df['momentum_rank'] = df['momentum_score'].rank(ascending=False)
        df['momentum_percentile'] = df['momentum_score'].rank(pct=True)
        
        # Generate signals
        df['signal'] = 'NEUTRAL'
        df.loc[df['momentum_percentile'] >= 0.8, 'signal'] = 'STRONG_BUY'
        df.loc[df['momentum_percentile'] >= 0.6, 'signal'] = 'BUY'
        df.loc[df['momentum_percentile'] <= 0.2, 'signal'] = 'SELL'
        df.loc[df['momentum_percentile'] <= 0.1, 'signal'] = 'STRONG_SELL'
        
        # Allocation weights (for rotational strategy)
        total_momentum = df['momentum_score'].sum()
        if total_momentum > 0:
            df['weight'] = np.maximum(df['momentum_score'] / total_momentum, 0.0)
        else:
            # Equal weight if no momentum signal
            df['weight'] = 1.0 / len(df)
            
        # Ensure weights sum to 1
        df['weight'] = df['weight'] / df['weight'].sum()
        
        # Add metadata
        df['timestamp'] = datetime.now().isoformat()
        df['strategy'] = 'rotational_alpha'
        
        logger.info(f"🔄 Generated rotation signals for {len(df)} sectors")
        logger.info(f"   Strong Buy: {len(df[df['signal'] == 'STRONG_BUY'])}")
        logger.info(f"   Buy: {len(df[df['signal'] == 'BUY'])}")
        logger.info(f"   Neutral: {len(df[df['signal'] == 'NEUTRAL'])}")
        logger.info(f"   Sell: {len(df[df['signal'] == 'SELL'])}")
        
        return df


def generate_rotational_alpha_scores(universe_symbols: Optional[List[str]] = None,
                                   output_file: str = "data/rot_alpha_scores.csv") -> pd.DataFrame:
    """Main function to generate rotational alpha scores"""
    
    logger.info("🔄 Generating Rotational Alpha scores...")
    
    # Default sector ETF universe
    if universe_symbols is None:
        universe_symbols = ['XLK', 'XLF', 'XLV', 'XLI', 'XLE', 'XLB', 'XLU', 'XLP', 'XLY', 'XLRE', 'XLC']
    
    # Initialize calculator
    calculator = SectorMomentumCalculator(lookback_days=20)
    
    # Generate mock price data (in production, this would fetch real data)
    price_data = calculator.generate_mock_sector_data(universe_symbols, days=40)
    
    # Calculate momentum scores for each symbol
    momentum_scores = []
    for symbol in universe_symbols:
        score = calculator.calculate_momentum_score(price_data, symbol)
        momentum_scores.append(score)
    
    # Generate rotation signals
    rotation_df = calculator.generate_rotation_signals(momentum_scores)
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rotation_df.to_csv(output_path, index=False)
    
    logger.info(f"💾 Saved rotational alpha scores to {output_path}")
    logger.info(f"📊 Top 3 momentum sectors:")
    
    top_sectors = rotation_df.nlargest(3, 'momentum_score')
    for _, row in top_sectors.iterrows():
        logger.info(f"   {row['symbol']} ({row['sector']}): {row['momentum_score']:.4f}")
    
    return rotation_df


def test_rotational_alpha_signal():
    """Test function for rotational alpha signal generation"""
    print("🧪 Testing Rotational Alpha Signal Generation...")
    
    try:
        # Generate scores
        scores_df = generate_rotational_alpha_scores()
        
        # Validate results
        assert len(scores_df) == 11, f"Expected 11 sectors, got {len(scores_df)}"
        assert abs(scores_df['weight'].sum() - 1.0) < 0.001, "Weights don't sum to 1"
        assert all(scores_df['data_quality'] >= 0.0), "Invalid data quality scores"
        
        # Check signal distribution
        signal_counts = scores_df['signal'].value_counts()
        print(f"📊 Signal distribution: {signal_counts.to_dict()}")
        
        # Performance summary
        avg_momentum = scores_df['momentum_score'].mean()
        top_momentum = scores_df['momentum_score'].max()
        
        print(f"📈 Average momentum score: {avg_momentum:.4f}")
        print(f"📈 Top momentum score: {top_momentum:.4f}")
        
        print("✅ Rotational Alpha signal test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Rotational Alpha signal test FAILED: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        success = test_rotational_alpha_signal()
        sys.exit(0 if success else 1)
    else:
        # Generate scores
        result_df = generate_rotational_alpha_scores()
        print(f"\n📊 Generated {len(result_df)} rotational alpha scores")
        print("\nTop 5 sectors by momentum:")
        print(result_df.nlargest(5, 'momentum_score')[['symbol', 'sector', 'momentum_score', 'signal', 'weight']])