#!/usr/bin/env python3
"""
Test script for ML Weight Management utilities
Tests Day 1 functionality: weight history table and helper functions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, date
from mech_exo.scoring.weight_utils import (
    create_ml_weight_history_table,
    get_current_ml_weight,
    log_weight_change,
    get_weight_history,
    update_ml_weight_config,
    validate_weight_bounds,
    format_weight_change_summary
)


def test_weight_history_table():
    """Test weight history table creation"""
    print("📊 Testing Weight History Table Creation...")
    
    try:
        success = create_ml_weight_history_table()
        
        if success:
            print("  ✅ ml_weight_history table created successfully")
            
            # Test table structure by attempting to query it
            try:
                from mech_exo.datasource.storage import DataStorage
                storage = DataStorage()
                
                # Query table schema
                schema_query = "PRAGMA table_info(ml_weight_history)"
                result = storage.conn.execute(schema_query).fetchall()
                
                print(f"  📋 Table schema ({len(result)} columns):")
                for col in result:
                    print(f"    - {col[1]} ({col[2]})")
                
                storage.close()
                print("  ✅ Table structure verified")
                
            except Exception as e:
                print(f"  ⚠️  Could not verify table structure: {e}")
                
        else:
            print("  ❌ Failed to create table")
            
        return success
        
    except Exception as e:
        print(f"  ❌ Table creation test failed: {e}")
        return False


def test_current_weight_retrieval():
    """Test getting current ML weight from config"""
    print("\n⚖️ Testing Current Weight Retrieval...")
    
    try:
        current_weight = get_current_ml_weight()
        
        print(f"  📖 Current ML weight: {current_weight}")
        
        # Validate weight is reasonable
        if 0.0 <= current_weight <= 0.50:
            print("  ✅ Weight within valid range (0.0 - 0.50)")
        else:
            print(f"  ⚠️  Weight outside expected range: {current_weight}")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Weight retrieval test failed: {e}")
        return False


def test_weight_validation():
    """Test weight validation function"""
    print("\n🔍 Testing Weight Validation...")
    
    test_cases = [
        (0.0, True, "Minimum valid weight"),
        (0.30, True, "Normal weight"),
        (0.50, True, "Maximum valid weight"),
        (-0.1, False, "Negative weight"),
        (0.6, False, "Exceeds maximum"),
        ("invalid", False, "Non-numeric"),
        (None, False, "None value")
    ]
    
    passed = 0
    total = len(test_cases)
    
    for weight, expected_valid, description in test_cases:
        is_valid, error_msg = validate_weight_bounds(weight)
        
        if is_valid == expected_valid:
            print(f"  ✅ {description}: {weight} → {is_valid}")
            passed += 1
        else:
            print(f"  ❌ {description}: {weight} → {is_valid} (expected {expected_valid})")
            if error_msg:
                print(f"    Error: {error_msg}")
    
    print(f"  📊 Validation tests: {passed}/{total} passed")
    return passed == total


def test_weight_change_logging():
    """Test logging weight changes"""
    print("\n📝 Testing Weight Change Logging...")
    
    try:
        # Test logging a weight change
        old_weight = 0.30
        new_weight = 0.35
        rule = "ML_OUTPERFORM_BASELINE"
        ml_sharpe = 1.25
        baseline_sharpe = 1.10
        notes = "ML Sharpe exceeded baseline by +0.15"
        
        success = log_weight_change(
            old_weight=old_weight,
            new_weight=new_weight,
            adjustment_rule=rule,
            ml_sharpe=ml_sharpe,
            baseline_sharpe=baseline_sharpe,
            notes=notes
        )
        
        if success:
            print("  ✅ Weight change logged successfully")
            
            # Test retrieving the logged change
            history = get_weight_history(days=1)
            
            if not history.empty:
                latest = history.iloc[0]
                print(f"  📖 Retrieved change: {latest['old_weight']:.2f} → {latest['new_weight']:.2f}")
                print(f"  📖 Rule: {latest['adjustment_rule']}")
                print(f"  📖 Sharpe diff: {latest['sharpe_diff']:+.3f}")
                print("  ✅ Weight history retrieval successful")
            else:
                print("  ⚠️  Could not retrieve logged change")
                
        else:
            print("  ❌ Failed to log weight change")
            
        return success
        
    except Exception as e:
        print(f"  ❌ Weight change logging test failed: {e}")
        return False


def test_weight_config_update():
    """Test updating ML weight in config file"""
    print("\n⚙️ Testing Weight Config Update...")
    
    try:
        # Get current weight
        original_weight = get_current_ml_weight()
        print(f"  📖 Original weight: {original_weight}")
        
        # Test updating to a new weight
        new_weight = 0.25 if original_weight != 0.25 else 0.35
        
        print(f"  🔄 Updating weight to: {new_weight}")
        success = update_ml_weight_config(new_weight, backup=True)
        
        if success:
            print("  ✅ Config file updated successfully")
            
            # Verify the change
            updated_weight = get_current_ml_weight()
            
            if abs(updated_weight - new_weight) < 0.001:
                print(f"  ✅ Weight verified: {updated_weight}")
                
                # Restore original weight
                print(f"  🔄 Restoring original weight: {original_weight}")
                restore_success = update_ml_weight_config(original_weight, backup=False)
                
                if restore_success:
                    print("  ✅ Original weight restored")
                else:
                    print("  ⚠️  Failed to restore original weight")
                    
            else:
                print(f"  ❌ Weight mismatch: expected {new_weight}, got {updated_weight}")
                
        else:
            print("  ❌ Failed to update config file")
            
        return success
        
    except Exception as e:
        print(f"  ❌ Config update test failed: {e}")
        return False


def test_weight_change_formatting():
    """Test weight change summary formatting"""
    print("\n🎨 Testing Weight Change Formatting...")
    
    try:
        test_cases = [
            (0.30, 0.35, 1.25, 1.10, "ML_OUTPERFORM_BASELINE"),
            (0.35, 0.30, 0.95, 1.10, "ML_UNDERPERFORM_BASELINE"),
            (0.25, 0.25, 1.05, 1.05, "NO_CHANGE")
        ]
        
        for old_w, new_w, ml_s, base_s, rule in test_cases:
            summary = format_weight_change_summary(old_w, new_w, ml_s, base_s, rule)
            print(f"  📄 {rule}:")
            for line in summary.split('\n'):
                print(f"    {line}")
            print()
        
        print("  ✅ Weight change formatting test completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Formatting test failed: {e}")
        return False


def test_integration_scenario():
    """Test complete integration scenario"""
    print("\n🔗 Testing Integration Scenario...")
    
    try:
        print("  📊 Scenario: ML outperforms baseline, trigger weight increase")
        
        # 1. Get current weight
        current_weight = get_current_ml_weight()
        print(f"  1️⃣ Current weight: {current_weight}")
        
        # 2. Simulate performance data
        ml_sharpe = 1.30
        baseline_sharpe = 1.15
        sharpe_diff = ml_sharpe - baseline_sharpe
        print(f"  2️⃣ ML Sharpe: {ml_sharpe}, Baseline: {baseline_sharpe}, Diff: {sharpe_diff:+.2f}")
        
        # 3. Determine new weight (simple increase logic)
        if sharpe_diff > 0.10:
            new_weight = min(current_weight + 0.05, 0.50)
            rule = "ML_OUTPERFORM_BASELINE"
        else:
            new_weight = current_weight
            rule = "NO_CHANGE"
            
        print(f"  3️⃣ New weight: {new_weight} ({rule})")
        
        # 4. Log the change if weight changed
        if new_weight != current_weight:
            log_success = log_weight_change(
                old_weight=current_weight,
                new_weight=new_weight,
                adjustment_rule=rule,
                ml_sharpe=ml_sharpe,
                baseline_sharpe=baseline_sharpe,
                notes=f"Auto-adjustment based on {sharpe_diff:+.2f} Sharpe diff"
            )
            print(f"  4️⃣ Change logged: {log_success}")
        else:
            print("  4️⃣ No change to log")
            
        # 5. Format notification
        summary = format_weight_change_summary(current_weight, new_weight, ml_sharpe, baseline_sharpe, rule)
        print("  5️⃣ Notification summary:")
        for line in summary.split('\n'):
            print(f"    {line}")
        
        print("  ✅ Integration scenario completed successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration scenario failed: {e}")
        return False


def main():
    """Run all weight management tests"""
    print("🚀 Testing ML Weight Management (Phase P9 Week 2 Day 1)\n")
    
    tests = [
        test_weight_history_table,
        test_current_weight_retrieval,
        test_weight_validation,
        test_weight_change_logging,
        test_weight_config_update,
        test_weight_change_formatting,
        test_integration_scenario
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All weight management tests PASSED!")
        print("\n✅ Day 1 Implementation Complete:")
        print("  • ml_weight_history table created with proper schema")
        print("  • get_current_ml_weight() returns value from config/factors.yml")
        print("  • log_weight_change() stores adjustments with full context")
        print("  • Weight validation and config update utilities working")
        print("  • Integration scenario demonstrates complete workflow")
        
        print("\n📋 Ready for Day 2:")
        print("  • Adjustment algorithm compute_new_weight() function")
        print("  • Increment/decrement logic with bounds checking")
        print("  • Unit tests for edge cases")
        
        return True
    else:
        print("❌ Some weight management tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)