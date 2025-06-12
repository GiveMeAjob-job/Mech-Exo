#!/usr/bin/env python3
"""
Test script for canary performance flow

Runs the canary performance flow manually and tests the health endpoint integration.
"""

import sys
import json
import requests
from pathlib import Path
from datetime import date, datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mech_exo.reporting.query import get_latest_canary_metrics, get_health_data


def test_canary_metrics():
    """Test the canary metrics function directly"""
    print("🧪 Testing canary metrics function...")
    
    try:
        metrics = get_latest_canary_metrics()
        print(f"✅ Canary metrics retrieved:")
        print(f"   - Canary Sharpe (30d): {metrics.get('canary_sharpe_30d', 0):.3f}")
        print(f"   - Base Sharpe (30d): {metrics.get('base_sharpe_30d', 0):.3f}")
        print(f"   - Sharpe difference: {metrics.get('sharpe_diff', 0):+.3f}")
        print(f"   - Canary enabled: {metrics.get('canary_enabled', True)}")
        print(f"   - Data quality: {metrics.get('canary_data_quality', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ Failed to get canary metrics: {e}")
        return False


def test_health_endpoint():
    """Test the health endpoint includes canary fields"""
    print("\n🌐 Testing health endpoint...")
    
    try:
        health_data = get_health_data()
        
        # Check if canary fields are present
        canary_sharpe = health_data.get('canary_sharpe_30d')
        canary_enabled = health_data.get('canary_enabled')
        
        if canary_sharpe is not None and canary_enabled is not None:
            print(f"✅ Health endpoint includes canary fields:")
            print(f"   - canary_sharpe_30d: {canary_sharpe:.3f}")
            print(f"   - canary_enabled: {canary_enabled}")
            return True
        else:
            print(f"❌ Health endpoint missing canary fields")
            print(f"   Available fields: {list(health_data.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test health endpoint: {e}")
        return False


def test_manual_flow():
    """Test running the canary performance flow manually"""
    print("\n🎯 Testing manual canary performance flow...")
    
    try:
        # Import and run the manual function
        from dags.canary_perf_flow import run_manual_canary_performance
        
        # Run for today
        target_date = str(date.today())
        result = run_manual_canary_performance(target_date, 30)
        
        if result.get('overall_status') in ['success', 'partial_success']:
            print(f"✅ Flow completed with status: {result.get('overall_status')}")
            
            # Check if performance metrics are available
            perf_metrics = result.get('performance_metrics', {})
            if perf_metrics:
                print(f"   - Total fills: {perf_metrics.get('total_fills', 0)}")
                print(f"   - Canary allocation: {perf_metrics.get('canary_allocation_pct', 0):.1f}%")
                print(f"   - Alpha: {perf_metrics.get('alpha_bps', 0):+.1f} bps")
                print(f"   - Sharpe difference: {perf_metrics.get('sharpe_diff', 0):+.3f}")
            
            return True
        else:
            print(f"❌ Flow failed with status: {result.get('overall_status')}")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to run manual flow: {e}")
        return False


def test_health_cache():
    """Test the health cache file creation"""
    print("\n💾 Testing health cache file...")
    
    cache_file = Path("data/canary_health_cache.json")
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            print(f"✅ Health cache file exists:")
            print(f"   - File: {cache_file}")
            print(f"   - Last updated: {cache_data.get('last_updated', 'unknown')}")
            print(f"   - Canary Sharpe: {cache_data.get('canary_sharpe_30d', 0):.3f}")
            print(f"   - Data quality: {cache_data.get('data_quality', 'unknown')}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to read health cache: {e}")
            return False
    else:
        print(f"⚠️  Health cache file not found: {cache_file}")
        print("   This is expected if the flow hasn't run yet.")
        return False


def main():
    """Run all tests"""
    print("🧪 Canary Performance Flow Test Suite")
    print("=" * 50)
    
    # Track test results
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Canary metrics function
    if test_canary_metrics():
        tests_passed += 1
    
    # Test 2: Health endpoint integration
    if test_health_endpoint():
        tests_passed += 1
    
    # Test 3: Manual flow execution
    if test_manual_flow():
        tests_passed += 1
    
    # Test 4: Health cache file
    if test_health_cache():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Day 3 deliverables are working.")
    elif tests_passed >= total_tests * 0.75:
        print("⚠️  Most tests passed. Some features may need data or configuration.")
    else:
        print("❌ Multiple tests failed. Check implementation and configuration.")
    
    # Instructions for deployment
    if tests_passed >= 2:
        print("\n📋 Next Steps:")
        print("1. To schedule the flow in Prefect:")
        print("   python deployments/canary_perf_deployment.py")
        print("2. To test health endpoint via HTTP:")
        print("   curl -H 'Accept: application/json' http://localhost:8050/healthz")
        print("3. Manual flow execution:")
        print("   python dags/canary_perf_flow.py")


if __name__ == "__main__":
    main()