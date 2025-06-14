#!/usr/bin/env python3
"""
Quick test of reconciliation functionality
"""

import pandas as pd
from datetime import date, timedelta
from mech_exo.reconciliation.reconciler import reconcile_trades

def test_reconciliation():
    """Test basic reconciliation functionality"""
    
    print("🧪 Testing Trade Reconciliation...")
    
    # Create test internal fills
    internal_fills = pd.DataFrame([
        {
            'fill_id': 'FILL001',
            'symbol': 'AAPL',
            'quantity': 100,
            'fill_price': 150.25,
            'commission': 1.00,
            'net_cash': -15026.00,
            'trade_id': 'TXN001'
        },
        {
            'fill_id': 'FILL002',
            'symbol': 'MSFT', 
            'quantity': -50,
            'fill_price': 305.80,
            'commission': 1.50,
            'net_cash': 15288.50,
            'trade_id': 'TXN002'
        }
    ])
    
    # Create test broker statement (matching)
    broker_statement = pd.DataFrame([
        {
            'symbol': 'AAPL',
            'qty': 100,
            'price': 150.25,
            'commission': 1.00,
            'net_cash': -15026.00,
            'trade_id': 'TXN001',
            'currency': 'USD'
        },
        {
            'symbol': 'MSFT',
            'qty': -50,
            'price': 305.80,
            'commission': 1.50,
            'net_cash': 15288.50,
            'trade_id': 'TXN002',
            'currency': 'USD'
        }
    ])
    
    print(f"\n📊 Test Data:")
    print(f"   Internal fills: {len(internal_fills)}")
    print(f"   Broker trades: {len(broker_statement)}")
    
    # Test perfect match scenario
    print(f"\n🎯 Testing perfect match scenario...")
    result = reconcile_trades(internal_fills, broker_statement)
    
    print(f"   Status: {result.status.value}")
    print(f"   Matched trades: {len(result.trade_matches)}")
    print(f"   Total diff: {result.total_diff_bps:.1f} basis points")
    print(f"   Unmatched internal: {len(result.unmatched_internal)}")
    print(f"   Unmatched broker: {len(result.unmatched_broker)}")
    
    assert result.status.value == 'pass', "Perfect match should pass"
    assert len(result.trade_matches) == 2, "Should match 2 trades"
    assert result.total_diff_bps < 5, "Should be under 5bp threshold"
    
    # Test scenario with differences
    print(f"\n⚠️ Testing scenario with differences...")
    broker_with_diff = broker_statement.copy()
    broker_with_diff.loc[0, 'commission'] = 2.00  # Increase commission
    broker_with_diff.loc[0, 'net_cash'] = -15027.00  # Adjust net cash
    
    result_diff = reconcile_trades(internal_fills, broker_with_diff)
    
    print(f"   Status: {result_diff.status.value}")
    print(f"   Total diff: {result_diff.total_diff_bps:.1f} basis points")
    print(f"   Commission diff: ${result_diff.total_commission_diff:.2f}")
    print(f"   Net cash diff: ${result_diff.total_net_cash_diff:.2f}")
    
    # Test unmatched scenario
    print(f"\n❌ Testing unmatched scenario...")
    broker_partial = broker_statement.iloc[:1].copy()  # Only first trade
    
    result_unmatched = reconcile_trades(internal_fills, broker_partial)
    
    print(f"   Status: {result_unmatched.status.value}")
    print(f"   Matched trades: {len(result_unmatched.trade_matches)}")
    print(f"   Unmatched internal: {len(result_unmatched.unmatched_internal)}")
    print(f"   Unmatched broker: {len(result_unmatched.unmatched_broker)}")
    
    assert result_unmatched.status.value == 'fail', "Unmatched trades should fail"
    assert len(result_unmatched.unmatched_internal) > 0, "Should have unmatched internal"
    
    print("\n🎉 Reconciliation tests completed successfully!")
    
    return True

if __name__ == "__main__":
    test_reconciliation()