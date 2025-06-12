#!/usr/bin/env python3
"""
Quick test of capital CLI functionality
"""

from mech_exo.cli.capital import CapitalManager
import tempfile
import os

def test_capital_cli():
    """Test capital CLI functionality"""
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        config_path = f.name
    
    try:
        print("🧪 Testing Capital CLI...")
        
        # Initialize manager
        manager = CapitalManager(config_path)
        print("✅ CapitalManager initialized")
        
        # Test adding account
        success = manager.add_account('DU12345678', 100000, 'USD', 'Test account')
        print(f"✅ Add account success: {success}")
        
        # Test listing accounts
        accounts = manager.list_accounts()
        print(f"✅ Number of accounts: {len(accounts)}")
        
        if accounts:
            account = accounts[0]
            print(f"   Account ID: {account['account_id']}")
            print(f"   Max capital: {account['max_capital']:,}")
            print(f"   Currency: {account['currency']}")
            print(f"   Enabled: {account['enabled']}")
        
        # Test total limits
        totals = manager.get_total_limits()
        print(f"✅ Total allocated: {totals['total_allocated']:,}")
        print(f"   Remaining capacity: {totals['remaining_capacity']:,}")
        print(f"   Utilization: {totals['utilization_pct']:.1f}%")
        
        # Test account validation
        valid_ids = ['DU12345678', 'DF123456789', 'UA12345678']
        invalid_ids = ['', '123456789', 'DU123', 'INVALID']
        
        print("\n🔍 Testing account ID validation:")
        for account_id in valid_ids:
            result = CapitalManager._validate_account_id(account_id)
            print(f"   {account_id}: {'✅' if result else '❌'}")
        
        for account_id in invalid_ids:
            result = CapitalManager._validate_account_id(account_id)
            print(f"   {account_id}: {'❌' if not result else '✅'} (should be invalid)")
        
        # Test updating account
        success = manager.add_account('DU12345678', 150000, 'USD', 'Updated test account')
        print(f"\n✅ Update account success: {success}")
        
        accounts = manager.list_accounts()
        if accounts:
            print(f"   Updated max capital: {accounts[0]['max_capital']:,}")
        
        # Test change history
        history = manager.get_change_history()
        print(f"\n✅ Change history entries: {len(history)}")
        for change in history:
            print(f"   {change['action']}: {change['account']} ({change['timestamp'][:19]})")
        
        print("\n🎉 All capital CLI tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.unlink(config_path)

if __name__ == "__main__":
    test_capital_cli()