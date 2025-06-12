#!/usr/bin/env python3
"""
CI Capital Fixture Script

Emits JSON payloads for testing capital guard flow in CI environment.
Provides both pass and fail test cases for capital limit validation.
"""

import json
import sys
import argparse
from datetime import datetime
from typing import Dict, Any


def create_pass_payload(account_id: str = "DU12345678") -> Dict[str, Any]:
    """
    Create a passing capital test case
    
    BuyingPower = 80k, Limit = 100k -> 20% utilization (OK)
    
    Args:
        account_id: IB account ID for testing
        
    Returns:
        JSON payload for passing test case
    """
    return {
        "test_case": "pass",
        "description": "Capital utilization below warning threshold",
        "account_id": account_id,
        "buying_power": 80000.0,
        "max_capital": 100000.0,
        "currency": "USD",
        "expected_utilization_pct": 20.0,
        "expected_status": "ok",
        "expected_capital_ok": True,
        "timestamp": datetime.now().isoformat(),
        "source": "ci_fixture"
    }


def create_fail_payload(account_id: str = "DU12345678") -> Dict[str, Any]:
    """
    Create a failing capital test case
    
    BuyingPower = 20k, Limit = 100k -> 80% utilization (WARNING/CRITICAL)
    
    Args:
        account_id: IB account ID for testing
        
    Returns:
        JSON payload for failing test case
    """
    return {
        "test_case": "fail",
        "description": "Capital utilization above critical threshold",
        "account_id": account_id,
        "buying_power": 20000.0,  # Low buying power = high utilization
        "max_capital": 100000.0,
        "currency": "USD",
        "expected_utilization_pct": 80.0,
        "expected_status": "critical",
        "expected_capital_ok": False,
        "timestamp": datetime.now().isoformat(),
        "source": "ci_fixture"
    }


def create_edge_case_payload(account_id: str = "DU12345678") -> Dict[str, Any]:
    """
    Create an edge case test (exactly at warning threshold)
    
    BuyingPower = 40k, Limit = 100k -> 60% utilization (WARNING)
    
    Args:
        account_id: IB account ID for testing
        
    Returns:
        JSON payload for edge case test
    """
    return {
        "test_case": "warning",
        "description": "Capital utilization at warning threshold",
        "account_id": account_id,
        "buying_power": 40000.0,
        "max_capital": 100000.0,
        "currency": "USD",
        "expected_utilization_pct": 60.0,
        "expected_status": "warning",
        "expected_capital_ok": True,  # Warning still allows trading
        "timestamp": datetime.now().isoformat(),
        "source": "ci_fixture"
    }


def inject_fixture_data(payload: Dict[str, Any]) -> None:
    """
    Inject fixture data into environment for capital guard flow
    
    Sets CI_CAPITAL_FIXTURE environment variable with JSON payload
    
    Args:
        payload: Test case payload to inject
    """
    import os
    
    # Set environment variable for capital guard flow to use
    os.environ['CI_CAPITAL_FIXTURE'] = json.dumps(payload)
    
    print(f"✅ Injected fixture data for test case: {payload['test_case']}")
    print(f"   Account: {payload['account_id']}")
    print(f"   Buying Power: ${payload['buying_power']:,.0f}")
    print(f"   Max Capital: ${payload['max_capital']:,.0f}")
    print(f"   Expected Utilization: {payload['expected_utilization_pct']:.1f}%")
    print(f"   Expected Status: {payload['expected_status']}")
    print(f"   Expected capital_ok: {payload['expected_capital_ok']}")


def run_capital_guard_test(test_case: str = "pass") -> int:
    """
    Run capital guard flow with fixture data
    
    Args:
        test_case: Test case to run ("pass", "fail", "warning")
        
    Returns:
        Exit code (0 = success, 1 = failure)
    """
    try:
        import sys
        from pathlib import Path
        
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        # Create and inject fixture data
        if test_case == "pass":
            payload = create_pass_payload()
        elif test_case == "fail":
            payload = create_fail_payload()
        elif test_case == "warning":
            payload = create_edge_case_payload()
        else:
            raise ValueError(f"Unknown test case: {test_case}")
        
        inject_fixture_data(payload)
        
        # Import and run capital guard flow
        from dags.capital_guard_flow import capital_guard_flow
        
        print(f"\n🏦 Running capital guard flow with {test_case} test case...")
        
        # Run flow in stub mode with CI fixture
        result = capital_guard_flow(stub_mode=True)
        
        print(f"\n📊 Flow result: {json.dumps(result, indent=2)}")
        
        # Validate results
        expected_capital_ok = payload['expected_capital_ok']
        actual_capital_ok = result.get('capital_ok', False)
        
        if actual_capital_ok == expected_capital_ok:
            print(f"✅ Test {test_case} PASSED: capital_ok = {actual_capital_ok} (expected: {expected_capital_ok})")
            return 0
        else:
            print(f"❌ Test {test_case} FAILED: capital_ok = {actual_capital_ok} (expected: {expected_capital_ok})")
            return 1
            
    except Exception as e:
        print(f"❌ Test {test_case} ERROR: {e}")
        return 1


def check_health_endpoint() -> Dict[str, Any]:
    """
    Check health endpoint for capital_ok status
    
    Returns:
        Health endpoint data or error info
    """
    try:
        from mech_exo.reporting.dash_app import _get_capital_health_status
        
        capital_ok = _get_capital_health_status()
        
        health_data = {
            'capital_ok': capital_ok,
            'timestamp': datetime.now().isoformat(),
            'source': 'health_endpoint'
        }
        
        print(f"🏥 Health endpoint check:")
        print(f"   capital_ok: {capital_ok}")
        
        return health_data
        
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return {
            'capital_ok': None,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Main CLI interface for CI capital fixture"""
    parser = argparse.ArgumentParser(description="CI Capital Fixture for testing capital guard flow")
    
    parser.add_argument("--test-case", choices=["pass", "fail", "warning"], default="pass",
                       help="Test case to run (default: pass)")
    parser.add_argument("--emit-only", action="store_true",
                       help="Only emit JSON payload, don't run test")
    parser.add_argument("--check-health", action="store_true",
                       help="Check health endpoint after test")
    parser.add_argument("--account-id", default="DU12345678",
                       help="IB account ID for testing (default: DU12345678)")
    
    args = parser.parse_args()
    
    try:
        if args.emit_only:
            # Just emit the JSON payload
            if args.test_case == "pass":
                payload = create_pass_payload(args.account_id)
            elif args.test_case == "fail":
                payload = create_fail_payload(args.account_id)
            elif args.test_case == "warning":
                payload = create_edge_case_payload(args.account_id)
            
            print(json.dumps(payload, indent=2))
            return 0
        
        # Run the test
        print(f"🧪 Starting CI capital smoke test: {args.test_case}")
        
        exit_code = run_capital_guard_test(args.test_case)
        
        if args.check_health:
            print("\n" + "="*50)
            health_data = check_health_endpoint()
            print(f"📊 Health check result: {json.dumps(health_data, indent=2)}")
        
        if exit_code == 0:
            print(f"\n🎉 CI capital smoke test PASSED for {args.test_case} case")
        else:
            print(f"\n💥 CI capital smoke test FAILED for {args.test_case} case")
        
        return exit_code
        
    except Exception as e:
        print(f"❌ CI capital fixture failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())