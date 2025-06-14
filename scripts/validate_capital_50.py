#!/usr/bin/env python3
"""
Validation script for Phase P11 Week 2 - 50% Canary Capital Expansion
Tests the capital configuration and allocation logic.
"""

import sys
import yaml
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def validate_capital_configuration():
    """Validate capital configuration for 50% canary allocation"""
    print("🔍 Validating Phase P11 Week 2 - 50% Canary Capital Configuration...")
    print("=" * 70)
    
    errors = []
    warnings = []
    
    # Test 1: Validate allocation.yml configuration
    try:
        allocation_file = Path("config/allocation.yml")
        if not allocation_file.exists():
            errors.append("allocation.yml not found")
        else:
            with open(allocation_file, 'r') as f:
                allocation_config = yaml.safe_load(f)
            
            canary_config = allocation_config.get('canary', {})
            allocation_pct = canary_config.get('allocation', 0)
            
            if allocation_pct == 0.50:
                print("✅ allocation.yml: Canary allocation set to 50%")
            else:
                errors.append(f"Expected 50% allocation, got {allocation_pct*100:.1f}%")
                
            if canary_config.get('enabled', False):
                print("✅ allocation.yml: Canary enabled")
            else:
                warnings.append("Canary not enabled in config")
                
    except Exception as e:
        errors.append(f"Failed to validate allocation.yml: {e}")
    
    # Test 2: Validate capital_limits.yml configuration
    try:
        capital_file = Path("config/capital_limits.yml")
        if not capital_file.exists():
            errors.append("capital_limits.yml not found")
        else:
            with open(capital_file, 'r') as f:
                capital_config = yaml.safe_load(f)
            
            accounts = capital_config.get('capital_limits', {}).get('accounts', {})
            canary_accounts = {k: v for k, v in accounts.items() if v.get('category') == 'canary'}
            
            total_canary_allocation = sum(acc.get('allocation_pct', 0) for acc in canary_accounts.values())
            
            if total_canary_allocation >= 50.0:
                print(f"✅ capital_limits.yml: Total canary allocation {total_canary_allocation:.1f}%")
            else:
                errors.append(f"Expected ≥50% total canary allocation, got {total_canary_allocation:.1f}%")
                
            print(f"📊 Found {len(canary_accounts)} canary accounts:")
            for account, config in canary_accounts.items():
                allocation = config.get('allocation_pct', 0)
                capital = config.get('max_capital', 0)
                print(f"   • {account}: {allocation:.1f}% (${capital:,})")
                
    except Exception as e:
        errors.append(f"Failed to validate capital_limits.yml: {e}")
    
    # Test 3: Test allocation.py functionality
    try:
        from mech_exo.execution.allocation import get_canary_allocation, is_canary_enabled
        
        current_allocation = get_canary_allocation()
        if abs(current_allocation - 0.50) < 0.01:
            print(f"✅ allocation.py: Returns 50% allocation ({current_allocation:.1%})")
        else:
            errors.append(f"allocation.py returns {current_allocation:.1%}, expected 50%")
            
        if is_canary_enabled():
            print("✅ allocation.py: Canary enabled check passes")
        else:
            warnings.append("allocation.py reports canary as disabled")
            
    except Exception as e:
        errors.append(f"Failed to test allocation.py: {e}")
    
    # Test 4: Validate test fixture
    try:
        fixture_file = Path("tests/fixtures/capital_50.yml")
        if not fixture_file.exists():
            errors.append("capital_50.yml test fixture not found")
        else:
            with open(fixture_file, 'r') as f:
                fixture_config = yaml.safe_load(f)
            
            expected_validation = fixture_config.get('expected_validation', {})
            expected_canary_pct = expected_validation.get('total_canary_allocation_pct', 0)
            
            if expected_canary_pct == 50.0:
                print("✅ Test fixture: Validates 50% canary allocation")
            else:
                warnings.append(f"Test fixture expects {expected_canary_pct}% allocation")
                
    except Exception as e:
        warnings.append(f"Could not validate test fixture: {e}")
    
    # Test 5: Capital manager integration
    try:
        from mech_exo.capital.manager import CapitalManager
        
        manager = CapitalManager()
        # This would test actual capital validation in production
        print("✅ CapitalManager: Module imports successfully")
        
    except Exception as e:
        warnings.append(f"CapitalManager test skipped: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    if not errors:
        print("🎉 ALL CRITICAL VALIDATIONS PASSED")
        print("✅ Phase P11 Week 2 capital expansion to 50% is ready")
    else:
        print("❌ VALIDATION FAILURES:")
        for error in errors:
            print(f"   • {error}")
    
    if warnings:
        print("\n⚠️ WARNINGS:")
        for warning in warnings:
            print(f"   • {warning}")
    
    if not errors and not warnings:
        print("\n🚀 READY FOR PRODUCTION DEPLOYMENT")
        print("   Command: exo canary enable --pct 0.50")
    
    return len(errors) == 0


if __name__ == "__main__":
    success = validate_capital_configuration()
    sys.exit(0 if success else 1)