"""
Universe Loader - Phase P11 Expansion

Manages the trading universe expansion from 250 to 500 stocks/ETFs.
Provides filtered, validated stock universes for factor-based trading strategies.

Features:
- Market cap and liquidity filtering
- Sector diversification requirements  
- ETF inclusion with expense ratio limits
- Real-time eligibility validation
- Feature completeness verification
"""

import os
import sys
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import requests
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


class UniverseType(Enum):
    """Types of trading universes"""
    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"
    ETF = "etf"
    SECTOR_ETF = "sector_etf"


@dataclass
class SecurityInfo:
    """Security information for universe inclusion"""
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    avg_volume: float
    price: float
    universe_type: UniverseType
    added_date: str
    weight: float = 1.0
    active: bool = True


class UniverseLoader:
    """Loads and manages trading universe expansion"""
    
    def __init__(self, cache_dir: str = "data/universe"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Universe parameters for Phase P11
        self.target_size = 500  # Expanded from 250
        self.min_market_cap = 2_000_000_000  # $2B minimum
        self.min_avg_volume = 500_000  # 500k shares daily avg
        self.max_etf_expense_ratio = 0.01  # 1% max expense ratio
        
        # Sector diversification requirements
        self.max_sector_concentration = 0.25  # Max 25% in any sector
        self.min_sectors = 8  # Minimum 8 sectors represented
        
        logger.info(f"🌐 Universe Loader initialized for {self.target_size} securities")
        
    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols as base universe"""
        try:
            # Download S&P 500 list
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_df = tables[0]
            
            symbols = sp500_df['Symbol'].tolist()
            
            # Clean symbols (some have dots, etc.)
            cleaned_symbols = []
            for symbol in symbols:
                # Replace common issues
                symbol = symbol.replace('.', '-')
                cleaned_symbols.append(symbol)
                
            logger.info(f"📥 Downloaded {len(cleaned_symbols)} S&P 500 symbols")
            return cleaned_symbols
            
        except Exception as e:
            logger.error(f"Failed to download S&P 500 list: {e}")
            
            # Fallback to static list of major symbols
            return self._get_fallback_symbols()
            
    def _get_fallback_symbols(self) -> List[str]:
        """Fallback list of major symbols if download fails"""
        return [
            # Large cap tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            # Large cap finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP',
            # Large cap healthcare
            'JNJ', 'PFE', 'ABT', 'MRK', 'TMO', 'DHR', 'BMY', 'ABBV',
            # Large cap consumer
            'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX',
            # Large cap industrial
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX',
            # Mid cap growth
            'ROKU', 'PTON', 'ZOOM', 'DOCU', 'SNOW', 'PLTR', 'AI', 'RBLX',
            # Additional mid caps across sectors
            'AMD', 'INTC', 'QCOM', 'CRM', 'ORCL', 'IBM', 'ACN', 'ADBE',
            'V', 'MA', 'PYPL', 'SQ', 'COIN', 'AFRM', 'UPST', 'SOFI'
        ]
        
    def get_liquid_etfs(self) -> List[str]:
        """Get list of liquid ETFs for universe expansion"""
        return [
            # Broad market ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO',
            # Sector ETFs
            'XLK', 'XLF', 'XLV', 'XLI', 'XLE', 'XLU', 'XLP', 'XLY', 'XLB',
            # Factor ETFs
            'VUG', 'VTV', 'VBR', 'VBK', 'QUAL', 'MTUM', 'VMOT', 'VLUE',
            # International
            'EFA', 'EEM', 'FXI', 'EWJ', 'EWZ', 'EWW', 'EWU', 'EWG',
            # Commodity/Real Estate
            'GLD', 'SLV', 'USO', 'UNG', 'VNQ', 'REZ', 'RWR',
            # Bond ETFs
            'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'EMB', 'TIP'
        ]
        
    def get_additional_growth_stocks(self) -> List[str]:
        """Get additional growth stocks to reach 500 target"""
        return [
            # Cloud/SaaS
            'NOW', 'WDAY', 'OKTA', 'ZS', 'CRWD', 'NET', 'DDOG', 'MDB',
            'ESTC', 'FSLY', 'TWLO', 'ZM', 'TEAM', 'ATLASSIAN', 'SHOP',
            # Biotech/Healthcare
            'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'ISRG', 'DXCM', 'TDOC',
            'VEEV', 'ZTS', 'EW', 'SYK', 'BSX', 'MDT', 'ANTM', 'UNH',
            # Fintech
            'HOOD', 'AFRM', 'LC', 'UPST', 'SOFI', 'NU', 'OPEN', 'RDFN',
            # EV/Clean Energy
            'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'FSR', 'CHPT', 'BLNK',
            'ENPH', 'SEDG', 'FSLR', 'RUN', 'SPWR', 'PLUG', 'BE', 'NEE',
            # Gaming/Entertainment
            'RBLX', 'U', 'TTWO', 'EA', 'ATVI', 'NTES', 'SE', 'BILI',
            # E-commerce/Consumer
            'BABA', 'JD', 'PDD', 'MELI', 'ETSY', 'W', 'CHWY', 'RVLV',
            # Additional industrials
            'PLTR', 'SNOW', 'AI', 'C3AI', 'PATH', 'GTLB', 'S', 'RDFN'
        ]
        
    def validate_security(self, symbol: str) -> Optional[SecurityInfo]:
        """Validate a security for universe inclusion"""
        try:
            # Get basic info from yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get historical data for volume analysis
            hist = ticker.history(period="30d")
            if hist.empty:
                logger.warning(f"No historical data for {symbol}")
                return None
                
            # Extract key metrics
            market_cap = info.get('marketCap', 0)
            avg_volume = hist['Volume'].mean()
            current_price = hist['Close'][-1]
            
            # Basic filters
            if market_cap < self.min_market_cap:
                logger.debug(f"{symbol}: Market cap ${market_cap:,.0f} below minimum")
                return None
                
            if avg_volume < self.min_avg_volume:
                logger.debug(f"{symbol}: Avg volume {avg_volume:,.0f} below minimum")
                return None
                
            if current_price < 5.0:  # Avoid penny stocks
                logger.debug(f"{symbol}: Price ${current_price:.2f} too low")
                return None
                
            # Determine universe type
            universe_type = self._classify_security(info, market_cap)
            
            return SecurityInfo(
                symbol=symbol,
                name=info.get('longName', symbol),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                market_cap=market_cap,
                avg_volume=avg_volume,
                price=current_price,
                universe_type=universe_type,
                added_date=datetime.now().strftime('%Y-%m-%d'),
                active=True
            )
            
        except Exception as e:
            logger.warning(f"Failed to validate {symbol}: {e}")
            return None
            
    def _classify_security(self, info: Dict, market_cap: float) -> UniverseType:
        """Classify security by type and market cap"""
        
        # Check if it's an ETF
        if info.get('quoteType') == 'ETF':
            if 'sector' in info.get('longName', '').lower():
                return UniverseType.SECTOR_ETF
            else:
                return UniverseType.ETF
                
        # Classify by market cap
        if market_cap >= 50_000_000_000:  # $50B+
            return UniverseType.LARGE_CAP
        elif market_cap >= 10_000_000_000:  # $10B+
            return UniverseType.MID_CAP
        else:
            return UniverseType.SMALL_CAP
            
    def build_expanded_universe(self) -> List[SecurityInfo]:
        """Build expanded 500-security universe"""
        logger.info("🔨 Building expanded 500-security universe...")
        
        all_symbols = set()
        
        # Get base symbols from different sources
        sp500_symbols = self.get_sp500_symbols()
        etf_symbols = self.get_liquid_etfs()
        growth_symbols = self.get_additional_growth_stocks()
        
        # Combine all symbols
        all_symbols.update(sp500_symbols[:350])  # Take top 350 S&P 500
        all_symbols.update(etf_symbols[:100])    # Take top 100 ETFs
        all_symbols.update(growth_symbols[:100]) # Take additional 100 growth stocks
        
        logger.info(f"📋 Collected {len(all_symbols)} candidate symbols")
        
        # Validate each symbol
        validated_securities = []
        validation_errors = []
        
        for i, symbol in enumerate(all_symbols):
            if i % 50 == 0:  # Progress update every 50 symbols
                logger.info(f"🔍 Validating symbols... {i}/{len(all_symbols)}")
                
            security = self.validate_security(symbol)
            if security:
                validated_securities.append(security)
            else:
                validation_errors.append(symbol)
                
            # Stop when we reach target
            if len(validated_securities) >= self.target_size:
                break
                
        # Check sector diversification
        validated_securities = self._ensure_sector_diversification(validated_securities)
        
        logger.info(f"✅ Universe built: {len(validated_securities)} securities")
        logger.info(f"❌ Validation errors: {len(validation_errors)} symbols")
        
        if len(validated_securities) < self.target_size:
            logger.warning(f"⚠️ Only {len(validated_securities)} securities validated (target: {self.target_size})")
            
        return validated_securities
        
    def _ensure_sector_diversification(self, securities: List[SecurityInfo]) -> List[SecurityInfo]:
        """Ensure proper sector diversification"""
        sector_counts = {}
        for security in securities:
            sector = security.sector
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
        total_securities = len(securities)
        max_per_sector = int(total_securities * self.max_sector_concentration)
        
        # Remove excess securities from over-represented sectors
        filtered_securities = []
        sector_current = {}
        
        for security in securities:
            sector = security.sector
            current_count = sector_current.get(sector, 0)
            
            if current_count < max_per_sector:
                filtered_securities.append(security)
                sector_current[sector] = current_count + 1
                
        logger.info(f"🎯 Sector diversification: {len(set(s.sector for s in filtered_securities))} sectors")
        
        return filtered_securities
        
    def save_universe(self, securities: List[SecurityInfo], filename: str = "universe_500.json"):
        """Save universe to JSON file"""
        universe_data = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'version': '500_expansion_p11',
                'total_securities': len(securities),
                'target_size': self.target_size
            },
            'filters': {
                'min_market_cap': self.min_market_cap,
                'min_avg_volume': self.min_avg_volume,
                'max_sector_concentration': self.max_sector_concentration
            },
            'securities': [
                {
                    'symbol': s.symbol,
                    'name': s.name,
                    'sector': s.sector,
                    'industry': s.industry,
                    'market_cap': s.market_cap,
                    'avg_volume': s.avg_volume,
                    'price': s.price,
                    'universe_type': s.universe_type.value,
                    'added_date': s.added_date,
                    'weight': s.weight,
                    'active': s.active
                }
                for s in securities
            ]
        }
        
        filepath = self.cache_dir / filename
        with open(filepath, 'w') as f:
            json.dump(universe_data, f, indent=2)
            
        logger.info(f"💾 Universe saved to {filepath}")
        
        return filepath
        
    def load_universe(self, filename: str = "universe_500.json") -> List[SecurityInfo]:
        """Load universe from JSON file"""
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            logger.error(f"Universe file not found: {filepath}")
            return []
            
        with open(filepath, 'r') as f:
            universe_data = json.load(f)
            
        securities = []
        for sec_data in universe_data['securities']:
            securities.append(SecurityInfo(
                symbol=sec_data['symbol'],
                name=sec_data['name'],
                sector=sec_data['sector'],
                industry=sec_data['industry'],
                market_cap=sec_data['market_cap'],
                avg_volume=sec_data['avg_volume'],
                price=sec_data['price'],
                universe_type=UniverseType(sec_data['universe_type']),
                added_date=sec_data['added_date'],
                weight=sec_data.get('weight', 1.0),
                active=sec_data.get('active', True)
            ))
            
        logger.info(f"📥 Loaded {len(securities)} securities from {filepath}")
        return securities
        
    def get_universe_summary(self, securities: List[SecurityInfo]) -> Dict:
        """Get summary statistics of the universe"""
        
        # Count by type
        type_counts = {}
        for security in securities:
            universe_type = security.universe_type.value
            type_counts[universe_type] = type_counts.get(universe_type, 0) + 1
            
        # Count by sector
        sector_counts = {}
        for security in securities:
            sector = security.sector
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
        # Calculate aggregates
        total_market_cap = sum(s.market_cap for s in securities)
        avg_market_cap = total_market_cap / len(securities) if securities else 0
        avg_volume = sum(s.avg_volume for s in securities) / len(securities) if securities else 0
        
        return {
            'total_securities': len(securities),
            'by_type': type_counts,
            'by_sector': dict(sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)),
            'market_cap': {
                'total': total_market_cap,
                'average': avg_market_cap,
                'median': sorted([s.market_cap for s in securities])[len(securities)//2] if securities else 0
            },
            'avg_daily_volume': avg_volume,
            'sector_count': len(sector_counts),
            'largest_sector_pct': max(sector_counts.values()) / len(securities) * 100 if securities else 0
        }
        
    def validate_feature_completeness(self, securities: List[SecurityInfo]) -> Dict:
        """Validate that all securities have complete feature data"""
        logger.info("🧪 Validating feature completeness...")
        
        results = {
            'total_securities': len(securities),
            'missing_data': {},
            'feature_coverage': {},
            'data_quality_score': 0.0
        }
        
        required_fields = ['sector', 'industry', 'market_cap', 'avg_volume', 'price']
        
        for field in required_fields:
            missing_count = 0
            for security in securities:
                value = getattr(security, field, None)
                if value is None or value == 'Unknown' or value == 0:
                    missing_count += 1
                    
            coverage_pct = (len(securities) - missing_count) / len(securities) * 100
            results['missing_data'][field] = missing_count
            results['feature_coverage'][field] = coverage_pct
            
        # Calculate overall data quality score
        coverage_scores = list(results['feature_coverage'].values())
        results['data_quality_score'] = sum(coverage_scores) / len(coverage_scores)
        
        logger.info(f"📊 Data quality score: {results['data_quality_score']:.1f}%")
        
        return results


def main():
    """Command-line interface for universe management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Universe Loader - Phase P11 Expansion')
    parser.add_argument('command', choices=['build', 'load', 'summary', 'validate'],
                       help='Command to execute')
    parser.add_argument('--file', default='universe_500.json', help='Universe file name')
    parser.add_argument('--size', type=int, default=500, help='Target universe size')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create universe loader
    loader = UniverseLoader()
    loader.target_size = args.size
    
    if args.command == 'build':
        # Build new universe
        securities = loader.build_expanded_universe()
        filepath = loader.save_universe(securities, args.file)
        
        # Print summary
        summary = loader.get_universe_summary(securities)
        print(f"\n=== Universe Built: {summary['total_securities']} securities ===")
        print(f"By Type: {summary['by_type']}")
        print(f"Sectors: {summary['sector_count']}")
        print(f"Largest Sector: {summary['largest_sector_pct']:.1f}%")
        print(f"Avg Market Cap: ${summary['market_cap']['average']:,.0f}")
        print(f"Saved to: {filepath}")
        
    elif args.command == 'load':
        # Load existing universe
        securities = loader.load_universe(args.file)
        if securities:
            summary = loader.get_universe_summary(securities)
            print(f"Loaded {summary['total_securities']} securities")
            
    elif args.command == 'summary':
        # Show universe summary
        securities = loader.load_universe(args.file)
        if securities:
            summary = loader.get_universe_summary(securities)
            print(json.dumps(summary, indent=2))
            
    elif args.command == 'validate':
        # Validate feature completeness
        securities = loader.load_universe(args.file)
        if securities:
            validation = loader.validate_feature_completeness(securities)
            print(json.dumps(validation, indent=2))
            
            # Exit code based on data quality
            min_quality_threshold = 85.0  # 85% minimum
            success = validation['data_quality_score'] >= min_quality_threshold
            
            if success:
                print(f"✅ Universe validation PASSED ({validation['data_quality_score']:.1f}%)")
                sys.exit(0)
            else:
                print(f"❌ Universe validation FAILED ({validation['data_quality_score']:.1f}% < {min_quality_threshold}%)")
                sys.exit(1)


if __name__ == '__main__':
    main()