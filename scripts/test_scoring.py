#!/usr/bin/env python3
"""
Quick test script for the scoring system
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from mech_exo.scoring import IdeaScorer
from mech_exo.datasource import DataStorage
import pandas as pd


def test_scoring_system():
    """Test the scoring system"""
    print("🎯 Testing Mech-Exo Scoring System...")
    
    try:
        # Initialize scorer
        scorer = IdeaScorer()
        print(f"✅ Scorer initialized with {len(scorer.factors)} factors")
        
        # Display factor information
        print("\n📊 Available Factors:")
        for name, factor in scorer.factors.items():
            print(f"  - {name}: weight={factor.weight}, direction={factor.direction}")
        
        # Check if we have data to score
        storage = scorer.storage
        universe = storage.get_universe()
        
        if universe.empty:
            print("\n⚠️ No symbols in universe. Adding test symbols...")
            test_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT']
            storage.add_to_universe(test_symbols)
            universe = storage.get_universe()
        
        if not universe.empty:
            print(f"\n🌍 Universe contains {len(universe)} symbols")
            
            # Try scoring with a few symbols
            test_symbols = universe['symbol'].head(3).tolist()
            print(f"\n🔍 Testing scoring with symbols: {test_symbols}")
            
            try:
                ranking = scorer.score(test_symbols)
                
                if not ranking.empty:
                    print(f"✅ Successfully scored {len(ranking)} symbols")
                    print("\n🏆 Ranking Results:")
                    print(ranking[['rank', 'symbol', 'composite_score']].to_string(index=False))
                    
                    # Save results
                    scorer.save_ranking(ranking, "data/test_ranking.csv")
                    print("\n💾 Results saved to data/test_ranking.csv")
                    
                else:
                    print("❌ No ranking results generated")
                    
            except Exception as e:
                print(f"❌ Scoring failed: {e}")
                print("💡 This might be due to missing fundamental data.")
                print("   Run the data pipeline first to populate data.")
        
        else:
            print("❌ No symbols available for testing")
            
        print("\n🎉 Scoring system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Scoring test failed: {e}")
        return False
        
    finally:
        if 'scorer' in locals():
            scorer.close()


if __name__ == "__main__":
    success = test_scoring_system()
    sys.exit(0 if success else 1)