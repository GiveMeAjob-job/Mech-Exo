#!/usr/bin/env python3
"""
Quick test of greylist functionality
"""

from mech_exo.utils.greylist import GreylistManager
import tempfile
import yaml
from pathlib import Path

def test_greylist():
    """Test basic greylist functionality"""
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        config_path = f.name
    
    try:
        print("🧪 Testing Greylist Manager...")
        
        # Initialize manager (will create default config)
        manager = GreylistManager(config_path)
        
        # Test getting greylist from default allocation.yml
        print("\n📋 Current greylist from config:")
        greylist = manager.get_greylist()
        print(f"   Symbols: {greylist}")
        print(f"   Count: {len(greylist)}")
        
        # Test checking symbols
        test_symbols = ['GME', 'AMC', 'AAPL', 'TSLA']
        print(f"\n🔍 Testing symbol checks:")
        for symbol in test_symbols:
            is_grey = manager.is_greylisted(symbol)
            status = "🚫 BLOCKED" if is_grey else "✅ ALLOWED"
            print(f"   {symbol}: {status}")
        
        # Test override setting
        override_enabled = manager.is_override_enabled()
        print(f"\n🔧 Override enabled: {override_enabled}")
        
        # Test stats
        stats = manager.get_greylist_stats()
        print(f"\n📊 Stats:")
        print(f"   Total symbols: {stats['total_symbols']}")
        print(f"   Override enabled: {stats['override_enabled']}")
        
        print("\n🎉 Greylist test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Greylist test failed: {e}")
        return False
    
    finally:
        # Cleanup
        config_file = Path(config_path)
        if config_file.exists():
            config_file.unlink()

if __name__ == "__main__":
    test_greylist()