#!/usr/bin/env python3
"""
Create sample training data for ML pipeline testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import Mock

from mech_exo.ml.features import FeatureBuilder

def create_extended_sample_data():
    """Create extended sample data for ML training."""
    
    # Generate 90 days of data
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Create OHLC data with trending behavior
    ohlc_data = []
    for symbol in symbols:
        base_price = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 140, 'AMZN': 120, 'NVDA': 500}[symbol]
        trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])  # Random trend
        
        for i, date_val in enumerate(dates):
            # Price with trend and noise
            trend_component = trend * i * 0.002  # Small daily trend
            noise = np.random.normal(0, 0.02)    # 2% daily volatility
            price_return = trend_component + noise
            
            price = base_price * (1 + price_return) ** i
            
            ohlc_data.append({
                'symbol': symbol,
                'date': date_val,
                'open': price * 0.999,
                'high': price * 1.015,
                'low': price * 0.985,
                'close': price,
                'volume': int(1000000 + np.random.normal(0, 200000)),
                'atr_20': price * np.random.uniform(0.015, 0.025)
            })
    
    ohlc_df = pd.DataFrame(ohlc_data)
    
    # Create fundamental data (quarterly updates)
    fund_data = []
    for symbol in symbols:
        for quarter_start in pd.date_range('2024-01-01', '2024-03-31', freq='Q'):
            fund_data.append({
                'symbol': symbol,
                'date': quarter_start,
                'pe_ratio': np.random.uniform(15, 40),
                'return_on_equity': np.random.uniform(0.10, 0.50),
                'revenue_growth': np.random.uniform(-0.05, 0.25),
                'earnings_growth': np.random.uniform(-0.10, 0.30),
                'debt_to_equity': np.random.uniform(0.2, 1.5),
                'current_ratio': np.random.uniform(1.0, 3.0),
                'gross_margin': np.random.uniform(0.20, 0.60),
                'profit_margin': np.random.uniform(0.05, 0.25)
            })
    
    fund_df = pd.DataFrame(fund_data)
    
    # Create news data (daily with some randomness)
    news_data = []
    for symbol in symbols:
        for i, date_val in enumerate(dates):
            # Not every symbol has news every day
            if np.random.random() > 0.3:  # 70% chance of news
                news_data.append({
                    'symbol': symbol,
                    'date': date_val,
                    'sentiment_score': np.random.beta(2, 2) * 0.8 + 0.1,  # Skewed toward neutral
                    'article_count': np.random.poisson(8) + 1
                })
    
    news_df = pd.DataFrame(news_data)
    
    return ohlc_df, fund_df, news_df

def main():
    """Create sample features for ML training."""
    print("🏗️ Creating extended sample data for ML training...")
    
    # Create sample data
    ohlc_df, fund_df, news_df = create_extended_sample_data()
    
    print(f"   • OHLC records: {len(ohlc_df)}")
    print(f"   • Fundamental records: {len(fund_df)}")
    print(f"   • News records: {len(news_df)}")
    
    # Mock storage
    mock_storage = Mock()
    mock_storage.get_ohlc_data.return_value = ohlc_df
    mock_storage.get_fundamental_data.return_value = fund_df
    mock_storage.get_news_data.return_value = news_df
    
    # Create feature builder
    builder = FeatureBuilder(storage=mock_storage, output_dir='data/features')
    
    # Build features for the date range
    print(f"\n📊 Building features for 90-day period...")
    
    feature_files = builder.build_features('2024-01-01', '2024-03-30')
    
    if feature_files:
        print(f"✅ Features built successfully!")
        print(f"   • Files created: {len(feature_files)}")
        
        # Show sample feature file
        sample_file = list(feature_files.values())[0]
        if sample_file.suffix == '.csv':
            sample_df = pd.read_csv(sample_file)
        else:
            sample_df = pd.read_parquet(sample_file)
            
        print(f"   • Sample file: {sample_file.name}")
        print(f"   • Symbols per file: {len(sample_df)}")
        print(f"   • Features per symbol: {sample_df['feature_count'].iloc[0] if 'feature_count' in sample_df.columns else 'N/A'}")
        
        # Show feature summary
        summary = builder.get_feature_summary(sample_df)
        print(f"   • Feature names: {summary['feature_names'][:8]}..." if len(summary['feature_names']) > 8 else f"   • Feature names: {summary['feature_names']}")
        
        return True
        
    else:
        print("❌ No features generated")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)