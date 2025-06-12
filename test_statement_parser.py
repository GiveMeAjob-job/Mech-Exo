#!/usr/bin/env python3
"""
Quick test of statement parser functionality
"""

from mech_exo.reconciliation.ib_statement_parser import parse_ib_statement, validate_ib_statement
from pathlib import Path

def test_statement_parser():
    """Test the statement parser with sample files"""
    
    print("🧪 Testing IB Statement Parser...")
    
    # Test CSV parsing
    csv_path = "tests/fixtures/sample_ib_statement.csv"
    if Path(csv_path).exists():
        print(f"\n📄 Testing CSV parsing: {csv_path}")
        try:
            df_csv = parse_ib_statement(csv_path)
            print(f"   ✅ Parsed {len(df_csv)} trades from CSV")
            print(f"   Columns: {list(df_csv.columns)}")
            if not df_csv.empty:
                print(f"   Sample symbols: {list(df_csv['symbol'].head())}")
                print(f"   Sample quantities: {list(df_csv['qty'].head())}")
                print(f"   Sample prices: {list(df_csv['price'].head())}")
                
                # Validate
                validation = validate_ib_statement(df_csv)
                print(f"   Validation: {'✅ PASS' if validation['is_valid'] else '❌ FAIL'}")
                if validation['warnings']:
                    print(f"   Warnings: {validation['warnings']}")
        except Exception as e:
            print(f"   ❌ CSV parsing failed: {e}")
    else:
        print(f"   ⚠️ CSV test file not found: {csv_path}")
    
    # Test OFX parsing
    ofx_path = "tests/fixtures/sample_ib_statement.ofx"
    if Path(ofx_path).exists():
        print(f"\n📄 Testing OFX parsing: {ofx_path}")
        try:
            df_ofx = parse_ib_statement(ofx_path)
            print(f"   ✅ Parsed {len(df_ofx)} trades from OFX")
            print(f"   Columns: {list(df_ofx.columns)}")
            if not df_ofx.empty:
                print(f"   Sample symbols: {list(df_ofx['symbol'].head())}")
                print(f"   Sample quantities: {list(df_ofx['qty'].head())}")
                print(f"   Sample prices: {list(df_ofx['price'].head())}")
                
                # Validate
                validation = validate_ib_statement(df_ofx)
                print(f"   Validation: {'✅ PASS' if validation['is_valid'] else '❌ FAIL'}")
                if validation['warnings']:
                    print(f"   Warnings: {validation['warnings']}")
        except Exception as e:
            print(f"   ❌ OFX parsing failed: {e}")
    else:
        print(f"   ⚠️ OFX test file not found: {ofx_path}")
    
    print("\n🎉 Statement parser test completed!")

if __name__ == "__main__":
    test_statement_parser()