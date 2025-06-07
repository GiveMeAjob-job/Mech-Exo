#!/usr/bin/env python3
"""
Quick test script for the data pipeline
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from mech_exo.datasource import OHLCDownloader, DataStorage
from mech_exo.utils import ConfigManager
import pandas as pd


def test_basic_pipeline():
    """Test basic pipeline functionality"""
    print("🚀 Testing Mech-Exo Data Pipeline...")
    
    # Initialize components
    config_manager = ConfigManager()
    api_config = config_manager.get_api_config()
    
    # Create storage
    storage = DataStorage("data/test_mech_exo.duckdb")
    
    # Test OHLC downloader
    downloader = OHLCDownloader(api_config)
    
    try:
        print("📊 Fetching sample OHLC data...")
        symbols = ['SPY', 'QQQ']  # Simple, reliable symbols
        data = downloader.fetch(symbols, period="5d", interval="1d")
        
        print(f"✅ Fetched {len(data)} OHLC records for {data['symbol'].nunique()} symbols")
        print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
        
        # Test storage
        print("💾 Testing data storage...")
        success = storage.store_ohlc_data(data)
        
        if success:
            print("✅ Data stored successfully")
            
            # Test retrieval
            retrieved = storage.get_ohlc_data(symbols)
            print(f"✅ Retrieved {len(retrieved)} records from database")
            
        else:
            print("❌ Failed to store data")
            
        # Add symbols to universe
        print("🌍 Adding symbols to universe...")
        universe_success = storage.add_to_universe(symbols)
        
        if universe_success:
            universe = storage.get_universe()
            print(f"✅ Universe now contains {len(universe)} symbols")
        
        print("\n🎉 Data pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False
        
    finally:
        storage.close()


if __name__ == "__main__":
    success = test_basic_pipeline()
    sys.exit(0 if success else 1)