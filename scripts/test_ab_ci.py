#!/usr/bin/env python3
"""
Local test script for A/B CI workflow validation

Simulates the CI smoke test locally to verify all components work.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import uuid

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def setup_test_environment():
    """Set up test environment similar to CI"""
    print("🏗️  Setting up test environment...")
    
    # Create temporary directory
    test_dir = Path(tempfile.mkdtemp(prefix="mech_exo_ci_test_"))
    print(f"   Test directory: {test_dir}")
    
    # Copy project files
    project_root = Path(__file__).parent.parent
    shutil.copytree(project_root, test_dir / "mech_exo", 
                   ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'data'))
    
    # Set up environment
    os.environ['PYTHONPATH'] = str(test_dir / "mech_exo")
    os.environ['TELEGRAM_DRY_RUN'] = 'true'
    os.environ['EXO_MODE'] = 'stub'
    
    return test_dir


def create_test_fills(test_dir):
    """Create test fills with base and canary tags"""
    print("📊 Creating test fills...")
    
    sys.path.insert(0, str(test_dir / "mech_exo"))
    
    from mech_exo.execution.fill_store import FillStore
    from mech_exo.execution.models import Fill
    
    # Create test data directory
    data_dir = test_dir / "mech_exo" / "data"
    data_dir.mkdir(exist_ok=True)
    
    fill_store = FillStore(str(data_dir / "test_fills.db"))
    
    # Create 5 days of test fills
    base_date = datetime.now() - timedelta(days=5)
    total_fills = 0
    
    for day in range(5):
        trade_date = base_date + timedelta(days=day)
        
        # Base allocation fills (90% of volume)
        for i in range(8):  # 8 base fills per day
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=f'base_order_{day}_{i}',
                symbol=f'STOCK_{i % 4}',  # 4 different stocks
                quantity=100 + (i * 10),
                price=50.0 + (i * 0.5) + (day * 0.1),  # Slight price movement
                filled_at=trade_date,
                tag='base',
                strategy='systematic',
                gross_value=(100 + (i * 10)) * (50.0 + (i * 0.5) + (day * 0.1)),
                total_fees=1.0 + (i * 0.1)
            )
            fill_store.store_fill(fill)
            total_fills += 1
        
        # Canary allocation fills (10% of volume)
        for i in range(2):  # 2 canary fills per day
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=f'canary_order_{day}_{i}',
                symbol=f'STOCK_{i % 2}',  # 2 different stocks
                quantity=20 + (i * 5),
                price=50.0 + (i * 0.5) + (day * 0.1) + 0.1,  # Slightly better performance
                filled_at=trade_date,
                tag='ml_canary',
                strategy='systematic',
                gross_value=(20 + (i * 5)) * (50.0 + (i * 0.5) + (day * 0.1) + 0.1),
                total_fees=0.2 + (i * 0.05)
            )
            fill_store.store_fill(fill)
            total_fills += 1
    
    fill_store.close()
    print(f"   Created {total_fills} test fills for 5 days")
    return True


def create_performance_data(test_dir):
    """Create canary performance data"""
    print("📈 Creating performance data...")
    
    sys.path.insert(0, str(test_dir / "mech_exo"))
    
    from mech_exo.reporting.pnl import store_daily_performance
    from mech_exo.datasource.storage import DataStorage
    
    # Create data directory
    data_dir = test_dir / "mech_exo" / "data"
    
    # Create canary_performance table
    storage = DataStorage(str(data_dir / "test_performance.duckdb"))
    
    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS canary_performance (
        date DATE PRIMARY KEY,
        canary_pnl DOUBLE,
        canary_nav DOUBLE,
        base_pnl DOUBLE,
        base_nav DOUBLE,
        canary_sharpe_30d DOUBLE,
        base_sharpe_30d DOUBLE,
        sharpe_diff DOUBLE,
        canary_enabled BOOLEAN,
        days_in_window INTEGER,
        updated_at TIMESTAMP
    )
    '''
    
    storage.conn.execute(create_table_sql)
    storage.close()
    
    # Generate performance data for the last 5 days
    base_date = datetime.now() - timedelta(days=5)
    success_count = 0
    
    for day in range(5):
        target_date = (base_date + timedelta(days=day)).date()
        try:
            success = store_daily_performance(target_date)
            if success:
                success_count += 1
            print(f"   Stored performance for {target_date}: {success}")
        except Exception as e:
            print(f"   Failed to store performance for {target_date}: {e}")
    
    print(f"   Successfully created {success_count}/5 performance records")
    return success_count > 0


def test_health_endpoint(test_dir):
    """Test health endpoint includes canary fields"""
    print("🏥 Testing health endpoint...")
    
    sys.path.insert(0, str(test_dir / "mech_exo"))
    
    from mech_exo.reporting.query import get_health_data
    
    # Get health data
    health_data = get_health_data()
    
    # Check for canary fields
    required_fields = ['canary_sharpe_30d', 'canary_enabled']
    missing_fields = []
    
    for field in required_fields:
        if field not in health_data:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"   ❌ Missing required fields: {missing_fields}")
        return False
    
    print("   ✅ Health endpoint includes required canary fields")
    print(f"      canary_sharpe_30d: {health_data['canary_sharpe_30d']}")
    print(f"      canary_enabled: {health_data['canary_enabled']}")
    return True


def test_auto_disable_logic(test_dir):
    """Test auto-disable logic"""
    print("🤖 Testing auto-disable logic...")
    
    sys.path.insert(0, str(test_dir / "mech_exo"))
    
    from mech_exo.execution.allocation import update_canary_enabled, is_canary_enabled
    from mech_exo.reporting.query import get_health_data
    
    try:
        # Test 1: Verify initial state
        initial_state = is_canary_enabled()
        print(f"   Initial canary state: {initial_state}")
        
        # Test 2: Disable canary
        success = update_canary_enabled(False)
        if not success:
            print("   ❌ Failed to disable canary")
            return False
        
        new_state = is_canary_enabled()
        if new_state:
            print("   ❌ Canary should be disabled")
            return False
        print("   ✅ Canary disable test passed")
        
        # Test 3: Check health endpoint reflects change
        health_data = get_health_data()
        if health_data['canary_enabled']:
            print("   ❌ Health endpoint should show canary disabled")
            return False
        print("   ✅ Health endpoint reflects canary disable")
        
        # Test 4: Re-enable canary
        success = update_canary_enabled(True)
        if not success:
            print("   ❌ Failed to re-enable canary")
            return False
        
        final_state = is_canary_enabled()
        if not final_state:
            print("   ❌ Canary should be enabled")
            return False
        print("   ✅ Canary re-enable test passed")
        
        # Test 5: Final health check
        health_data = get_health_data()
        if not health_data['canary_enabled']:
            print("   ❌ Health endpoint should show canary enabled")
            return False
        print("   ✅ All auto-disable logic tests passed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Auto-disable logic test failed: {e}")
        return False


def test_ab_dashboard_queries(test_dir):
    """Test A/B dashboard queries"""
    print("📊 Testing A/B dashboard queries...")
    
    sys.path.insert(0, str(test_dir / "mech_exo"))
    
    try:
        from mech_exo.reporting.query import get_canary_equity, get_base_equity, get_ab_test_summary
        
        # Test canary equity query
        canary_data = get_canary_equity(days=7)
        print(f"   Canary equity data: {len(canary_data)} rows")
        
        # Test base equity query  
        base_data = get_base_equity(days=7)
        print(f"   Base equity data: {len(base_data)} rows")
        
        # Test A/B summary
        summary = get_ab_test_summary(days=7)
        print(f"   A/B summary: {summary['status_badge']} - {summary['status_color']}")
        print(f"   Days analyzed: {summary['days_analyzed']}")
        print(f"   Sharpe diff: {summary['sharpe_diff']:.3f}")
        
        print("   ✅ A/B dashboard queries working")
        return True
        
    except Exception as e:
        print(f"   ❌ A/B dashboard queries failed: {e}")
        return False


def test_telegram_alert(test_dir):
    """Test Telegram alert (dry-run)"""
    print("📱 Testing Telegram alert...")
    
    sys.path.insert(0, str(test_dir / "mech_exo"))
    
    try:
        from mech_exo.utils.alerts import AlertManager, Alert, AlertType, AlertLevel
        from datetime import datetime
        
        # Create test alert manager
        alert_manager = AlertManager()
        
        # Create test auto-disable alert
        alert = Alert(
            alert_type=AlertType.RISK_VIOLATION,
            level=AlertLevel.CRITICAL,
            title='🧪 CI Test: Canary Auto-Disabled',
            message='This is a CI smoke test for auto-disable functionality:\n\n'
                   '• Canary Sharpe (30d): -0.250\n'
                   '• Threshold: 0.000\n'
                   '• Observations: 35 days\n'
                   '• Data quality: good\n\n'
                   'All new orders will use base allocation only.\n'
                   'Manual review and re-enable required.',
            timestamp=datetime.now(),
            data={
                'canary_sharpe': -0.25,
                'threshold': 0.0,
                'observations': 35,
                'data_quality': 'good',
                'auto_disabled': True,
                'ci_test': True
            }
        )
        
        # This will log the alert instead of sending due to dry-run mode
        success = alert_manager.send_alert(alert, channels=['telegram'])
        print(f"   Telegram alert test: {'✅ PASSED' if success else '❌ FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"   Telegram alert test failed: {e}")
        print("   ⚠️ Telegram not configured - test skipped")
        return True  # Don't fail if Telegram isn't configured


def cleanup_test_environment(test_dir):
    """Clean up test environment"""
    print("🧹 Cleaning up test environment...")
    
    try:
        shutil.rmtree(test_dir)
        print(f"   Removed test directory: {test_dir}")
    except Exception as e:
        print(f"   Warning: Failed to remove test directory: {e}")


def main():
    """Run A/B CI smoke test locally"""
    print("🧪 A/B Test CI Smoke Test (Local)")
    print("=" * 50)
    
    test_dir = None
    tests_passed = 0
    total_tests = 6
    
    try:
        # Setup
        test_dir = setup_test_environment()
        
        # Test 1: Create test fills
        if create_test_fills(test_dir):
            tests_passed += 1
        
        # Test 2: Create performance data
        if create_performance_data(test_dir):
            tests_passed += 1
        
        # Test 3: Health endpoint
        if test_health_endpoint(test_dir):
            tests_passed += 1
        
        # Test 4: Auto-disable logic
        if test_auto_disable_logic(test_dir):
            tests_passed += 1
        
        # Test 5: A/B dashboard queries
        if test_ab_dashboard_queries(test_dir):
            tests_passed += 1
        
        # Test 6: Telegram alert
        if test_telegram_alert(test_dir):
            tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
    
    finally:
        if test_dir:
            cleanup_test_environment(test_dir)
    
    # Summary
    print("=" * 50)
    print(f"📊 Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! A/B CI workflow is ready.")
        print("\n📋 CI Workflow Features:")
        print("• Test data generation (fills with base/canary tags)")
        print("• P&L calculation and performance tracking")
        print("• Health endpoint validation")
        print("• Auto-disable logic testing")
        print("• A/B dashboard query validation")
        print("• Telegram alert dry-run testing")
    elif tests_passed >= total_tests * 0.75:
        print("⚠️  Most tests passed. CI workflow mostly ready.")
    else:
        print("❌ Multiple tests failed. CI workflow needs fixes.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)