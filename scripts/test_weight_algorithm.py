#!/usr/bin/env python3
"""
Test script for weight adjustment algorithm
Tests Day 2 functionality: compute_new_weight function and edge cases
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import math
from mech_exo.scoring.weight_utils import compute_new_weight, auto_adjust_ml_weight


def test_basic_algorithm():
    """Test basic weight adjustment algorithm"""
    print("🧮 Testing Basic Algorithm Logic...")
    
    test_cases = [
        {
            'name': 'ML outperforms (+0.12 delta)',
            'baseline': 1.00,
            'ml': 1.12,
            'current': 0.30,
            'expected_weight': 0.35,
            'expected_rule': 'ML_OUTPERFORM_BASELINE'
        },
        {
            'name': 'ML underperforms (-0.07 delta)', 
            'baseline': 1.00,
            'ml': 0.93,
            'current': 0.30,
            'expected_weight': 0.25,
            'expected_rule': 'ML_UNDERPERFORM_BASELINE'
        },
        {
            'name': 'Performance within band (+0.03 delta)',
            'baseline': 1.00,
            'ml': 1.03,
            'current': 0.30,
            'expected_weight': 0.30,
            'expected_rule': 'PERFORMANCE_WITHIN_BAND'
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test in test_cases:
        new_weight, rule = compute_new_weight(
            baseline_sharpe=test['baseline'],
            ml_sharpe=test['ml'],
            current_w=test['current']
        )
        
        weight_match = abs(new_weight - test['expected_weight']) < 0.001
        rule_match = rule == test['expected_rule']
        
        if weight_match and rule_match:
            print(f"  ✅ {test['name']}: {test['current']:.2f} → {new_weight:.2f} ({rule})")
            passed += 1
        else:
            print(f"  ❌ {test['name']}: Expected {test['expected_weight']:.2f}/{test['expected_rule']}, got {new_weight:.2f}/{rule}")
    
    print(f"  📊 Basic algorithm tests: {passed}/{total} passed")
    return passed == total


def test_boundary_conditions():
    """Test boundary and edge cases"""
    print("\n🔒 Testing Boundary Conditions...")
    
    test_cases = [
        {
            'name': 'Upper cap (0.48 → 0.50)',
            'baseline': 1.00,
            'ml': 1.15,  # Trigger increase
            'current': 0.48,
            'expected_weight': 0.50
        },
        {
            'name': 'Lower floor (0.02 → 0.00)',
            'baseline': 1.00,
            'ml': 0.85,  # Trigger decrease
            'current': 0.02,
            'expected_weight': 0.00
        },
        {
            'name': 'Already at max (0.50 → 0.50)',
            'baseline': 1.00,
            'ml': 1.20,  # Would increase but capped
            'current': 0.50,
            'expected_weight': 0.50
        },
        {
            'name': 'Already at min (0.00 → 0.00)',
            'baseline': 1.00,
            'ml': 0.80,  # Would decrease but floored
            'current': 0.00,
            'expected_weight': 0.00
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test in test_cases:
        new_weight, rule = compute_new_weight(
            baseline_sharpe=test['baseline'],
            ml_sharpe=test['ml'],
            current_w=test['current']
        )
        
        weight_match = abs(new_weight - test['expected_weight']) < 0.001
        
        if weight_match:
            print(f"  ✅ {test['name']}: {test['current']:.2f} → {new_weight:.2f}")
            passed += 1
        else:
            print(f"  ❌ {test['name']}: Expected {test['expected_weight']:.2f}, got {new_weight:.2f}")
    
    print(f"  📊 Boundary tests: {passed}/{total} passed")
    return passed == total


def test_nan_handling():
    """Test NaN and None value handling"""
    print("\n🚫 Testing NaN/None Handling...")
    
    test_cases = [
        {
            'name': 'NaN baseline Sharpe',
            'baseline': float('nan'),
            'ml': 1.20,
            'current': 0.30,
            'expected_rule': 'INVALID_SHARPE_VALUES'
        },
        {
            'name': 'NaN ML Sharpe',
            'baseline': 1.00,
            'ml': float('nan'),
            'current': 0.30,
            'expected_rule': 'INVALID_SHARPE_VALUES'
        },
        {
            'name': 'None baseline Sharpe',
            'baseline': None,
            'ml': 1.20,
            'current': 0.30,
            'expected_rule': 'INVALID_SHARPE_VALUES'
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test in test_cases:
        new_weight, rule = compute_new_weight(
            baseline_sharpe=test['baseline'],
            ml_sharpe=test['ml'],
            current_w=test['current']
        )
        
        # Should return current weight unchanged with invalid rule
        weight_unchanged = abs(new_weight - test['current']) < 0.001
        rule_match = rule == test['expected_rule']
        
        if weight_unchanged and rule_match:
            print(f"  ✅ {test['name']}: Weight unchanged ({new_weight:.2f}), rule={rule}")
            passed += 1
        else:
            print(f"  ❌ {test['name']}: Expected unchanged/{test['expected_rule']}, got {new_weight:.2f}/{rule}")
    
    print(f"  📊 NaN handling tests: {passed}/{total} passed")
    return passed == total


def test_threshold_precision():
    """Test threshold boundary precision"""
    print("\n🎯 Testing Threshold Precision...")
    
    test_cases = [
        {
            'name': 'Exact up threshold (+0.10)',
            'baseline': 1.00,
            'ml': 1.10,  # Exactly +0.10
            'expected_rule': 'ML_OUTPERFORM_BASELINE'
        },
        {
            'name': 'Just below up threshold (+0.099)',
            'baseline': 1.00,
            'ml': 1.099,  # Just below +0.10
            'expected_rule': 'PERFORMANCE_WITHIN_BAND'
        },
        {
            'name': 'Exact down threshold (-0.05)',
            'baseline': 1.00,
            'ml': 0.95,  # Exactly -0.05
            'expected_rule': 'ML_UNDERPERFORM_BASELINE'
        },
        {
            'name': 'Just above down threshold (-0.049)',
            'baseline': 1.00,
            'ml': 0.951,  # Just above -0.05
            'expected_rule': 'PERFORMANCE_WITHIN_BAND'
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test in test_cases:
        new_weight, rule = compute_new_weight(
            baseline_sharpe=test['baseline'],
            ml_sharpe=test['ml'],
            current_w=0.30
        )
        
        rule_match = rule == test['expected_rule']
        delta = test['ml'] - test['baseline']
        
        if rule_match:
            print(f"  ✅ {test['name']}: Delta={delta:+.3f} → {rule}")
            passed += 1
        else:
            print(f"  ❌ {test['name']}: Delta={delta:+.3f} → Expected {test['expected_rule']}, got {rule}")
    
    print(f"  📊 Threshold precision tests: {passed}/{total} passed")
    return passed == total


def test_custom_parameters():
    """Test custom step sizes and thresholds"""
    print("\n⚙️ Testing Custom Parameters...")
    
    # Test custom step size
    new_weight, rule = compute_new_weight(
        baseline_sharpe=1.00,
        ml_sharpe=1.15,  # Trigger increase
        current_w=0.30,
        step=0.10  # Custom step
    )
    
    step_test_passed = abs(new_weight - 0.40) < 0.001  # 0.30 + 0.10
    
    # Test custom thresholds
    new_weight2, rule2 = compute_new_weight(
        baseline_sharpe=1.00,
        ml_sharpe=1.12,  # +0.12 delta
        current_w=0.30,
        up_thresh=0.15,  # Higher threshold
        down_thresh=-0.10
    )
    
    threshold_test_passed = rule2 == 'PERFORMANCE_WITHIN_BAND'  # Should not trigger with higher threshold
    
    if step_test_passed:
        print("  ✅ Custom step size (0.10): 0.30 → 0.40")
    else:
        print(f"  ❌ Custom step size: Expected 0.40, got {new_weight:.2f}")
    
    if threshold_test_passed:
        print("  ✅ Custom thresholds: Delta +0.12 with thresh 0.15 → No change")
    else:
        print(f"  ❌ Custom thresholds: Expected no change, got {rule2}")
    
    passed = sum([step_test_passed, threshold_test_passed])
    print(f"  📊 Custom parameter tests: {passed}/2 passed")
    return passed == 2


def test_cli_integration():
    """Test CLI utility integration"""
    print("\n🖥️ Testing CLI Integration...")
    
    try:
        from mech_exo.cli_weight import main
        print("  ✅ CLI module imports successfully")
        
        # Test basic CLI argument parsing (we won't actually run commands)
        print("  ✅ CLI weight adjustment utility available")
        print("    Usage: python -m mech_exo.cli_weight adjust --baseline 1.00 --ml 1.15 --current 0.30 --dry-run")
        print("    Usage: python -m mech_exo.cli_weight auto --dry-run")
        print("    Usage: python -m mech_exo.cli_weight current")
        
        return True
        
    except Exception as e:
        print(f"  ❌ CLI integration failed: {e}")
        return False


def test_realistic_scenarios():
    """Test realistic market scenarios"""
    print("\n📈 Testing Realistic Scenarios...")
    
    scenarios = [
        {
            'name': 'Bull market: ML helps momentum',
            'baseline': 1.50,
            'ml': 1.65,
            'expected_increase': True
        },
        {
            'name': 'Bear market: ML provides downside protection',
            'baseline': -0.20,
            'ml': 0.10,
            'expected_increase': True  # +0.30 delta > +0.10 threshold
        },
        {
            'name': 'Sideways market: ML adds little value',
            'baseline': 0.80,
            'ml': 0.83,
            'expected_increase': False  # +0.03 delta within band
        },
        {
            'name': 'Volatile market: ML hurts performance',
            'baseline': 0.60,
            'ml': 0.45,
            'expected_increase': False  # -0.15 delta triggers decrease
        }
    ]
    
    passed = 0
    total = len(scenarios)
    
    for scenario in scenarios:
        new_weight, rule = compute_new_weight(
            baseline_sharpe=scenario['baseline'],
            ml_sharpe=scenario['ml'],
            current_w=0.30
        )
        
        delta = scenario['ml'] - scenario['baseline']
        actual_increase = new_weight > 0.30
        expected_increase = scenario['expected_increase']
        
        if actual_increase == expected_increase:
            direction = "increased" if actual_increase else "decreased/unchanged"
            print(f"  ✅ {scenario['name']}: Weight {direction} (delta: {delta:+.2f})")
            passed += 1
        else:
            print(f"  ❌ {scenario['name']}: Expected {'increase' if expected_increase else 'decrease/unchanged'}, got {'increase' if actual_increase else 'decrease/unchanged'}")
    
    print(f"  📊 Realistic scenario tests: {passed}/{total} passed")
    return passed == total


def main():
    """Run all weight adjustment algorithm tests"""
    print("🚀 Testing ML Weight Adjustment Algorithm (Phase P9 Week 2 Day 2)\n")
    
    tests = [
        test_basic_algorithm,
        test_boundary_conditions,
        test_nan_handling,
        test_threshold_precision,
        test_custom_parameters,
        test_cli_integration,
        test_realistic_scenarios
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All weight adjustment algorithm tests PASSED!")
        print("\n✅ Day 2 Implementation Complete:")
        print("  • compute_new_weight() function with full algorithm logic")
        print("  • Edge case protection (NaN, bounds, clamping)")
        print("  • Comprehensive unit test coverage")
        print("  • CLI utility for dry-run testing")
        print("  • Integration hooks ready for Prefect flow")
        
        print("\n📋 Ready for Day 3:")
        print("  • Prefect task auto_adjust_ml_weight integration")
        print("  • ML reweight flow creation")
        print("  • YAML configuration updates")
        
        return True
    else:
        print("❌ Some weight adjustment algorithm tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)