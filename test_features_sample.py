#!/usr/bin/env python3
"""
Test ML feature building with sample data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import Mock

from mech_exo.ml.features import FeatureBuilder

def create_sample_data():
    """Create sample OHLC, fundamental, and news data."""
    
    # Sample date range
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Create OHLC data
    ohlc_data = []
    for symbol in symbols:
        base_price = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 140}[symbol]
        
        for i, date_val in enumerate(dates):
            price = base_price + i * 0.5 + np.random.normal(0, 2)
            ohlc_data.append({
                'symbol': symbol,
                'date': date_val,
                'open': price * 0.999,
                'high': price * 1.01,
                'low': price * 0.99,
                'close': price,
                'volume': 1000000 + np.random.randint(-100000, 100000),
                'atr_20': price * 0.02
            })
    
    ohlc_df = pd.DataFrame(ohlc_data)
    
    # Create fundamental data
    fund_data = []
    for symbol in symbols:
        fund_data.append({
            'symbol': symbol,
            'date': dates[5],  # Mid-point date
            'pe_ratio': np.random.uniform(20, 35),
            'return_on_equity': np.random.uniform(0.15, 0.45),
            'revenue_growth': np.random.uniform(0.05, 0.20),
            'earnings_growth': np.random.uniform(0.08, 0.25),
            'debt_to_equity': np.random.uniform(0.3, 1.2),
            'current_ratio': np.random.uniform(1.1, 2.5)
        })
    
    fund_df = pd.DataFrame(fund_data)
    
    # Create news data
    news_data = []
    for symbol in symbols:
        for i in range(5):  # 5 news items per symbol
            news_data.append({
                'symbol': symbol,
                'date': dates[i + 2],
                'sentiment_score': np.random.uniform(0.2, 0.8),
                'article_count': np.random.randint(3, 15)
            })
    
    news_df = pd.DataFrame(news_data)
    
    return ohlc_df, fund_df, news_df

def main():
    """Test feature building with sample data."""
    print("🧪 Testing ML Feature Builder with sample data...")
    
    # Create sample data
    ohlc_df, fund_df, news_df = create_sample_data()
    
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
    
    # Test feature building for a single date
    target_date = pd.Timestamp('2024-01-08')
    
    # Prepare data dict
    data_dict = {
        'ohlc': ohlc_df,
        'fundamentals': fund_df,
        'news': news_df
    }
    
    # Build features for the target date
    print(f"\n📊 Building features for {target_date.strftime('%Y-%m-%d')}...")
    
    features_df = builder._build_features_for_date(data_dict, target_date)
    
    if features_df is not None and len(features_df) > 0:
        print(f"✅ Features built successfully!")
        print(f"   • Symbols: {len(features_df)}")
        print(f"   • Features: {len(features_df.columns)}")
        print(f"   • Feature names: {list(features_df.columns)}")
        
        # Show sample features
        print(f"\n📈 Sample features for {features_df.iloc[0]['symbol']}:")
        sample_row = features_df.iloc[0]
        for col in features_df.columns:
            if col not in ['symbol', 'feature_date', 'feature_count']:
                value = sample_row[col]
                if isinstance(value, (int, float)) and not pd.isna(value):
                    print(f"   • {col}: {value:.4f}")
                else:
                    print(f"   • {col}: {value}")
        
        # Test feature summary
        summary = builder.get_feature_summary(features_df)
        print(f"\n📋 Feature Summary:")
        print(f"   • Total symbols: {summary['total_symbols']}")
        print(f"   • Total features: {summary['total_features']}")
        print(f"   • Missing data: {max(summary['missing_data_pct'].values()):.1f}% max")
        
        # Save to file for inspection
        output_file = Path('data/features/test_features.csv')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(output_file, index=False)
        print(f"   • Saved to: {output_file}")
        
    else:
        print("❌ No features generated")
        return False
    
    print("\n🎉 Feature building test completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)