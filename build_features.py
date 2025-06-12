#!/usr/bin/env python3
"""
Build ML features script for Mech-Exo.

Usage:
    python build_features.py --start 2022-01-01 --end 2025-01-01
    python build_features.py --start 2024-01-01 --end 2024-12-31 --symbols AAPL,MSFT,GOOGL
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mech_exo.ml.features import build_features_cli

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description='Build ML feature matrices')
    
    parser.add_argument('--start', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, 
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, default=None,
                       help='Comma-separated list of symbols (optional)')
    parser.add_argument('--output-dir', type=str, default='data/features',
                       help='Output directory (default: data/features)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        
    try:
        build_features_cli(
            start_date=args.start,
            end_date=args.end,
            symbols=symbols,
            output_dir=args.output_dir
        )
        print(f"✅ Feature building completed successfully!")
        print(f"📁 Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Feature building failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()