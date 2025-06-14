"""
Monthly Drawdown Guard Flow

End-of-day Prefect flow that monitors monthly P&L and triggers kill-switch
at -3% threshold. Scheduled daily at 23:10 UTC.

Part of P10 Week 3 Day 3 implementation.
"""

import logging
import subprocess
import os
from datetime import datetime, date, timedelta
from typing import Dict, Any

from prefect import flow, task
from prefect.logging import get_run_logger

from mech_exo.utils.monthly_loss_guard import (
    get_mtd_pnl_pct, 
    get_monthly_config, 
    should_run_monthly_guard,
    get_mtd_summary
)

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_THRESHOLD_PCT = -3.0  # -3% monthly stop-loss threshold


@task(name="calc_mtd_loss", retries=1, retry_delay_seconds=30)
def calc_mtd_loss_task(target_date: date = None) -> Dict[str, Any]:
    """
    Calculate month-to-date loss percentage
    
    Args:
        target_date: Date to calculate MTD for (default: today)
        
    Returns:
        Dictionary with MTD calculation results
    """
    task_logger = get_run_logger()
    
    if target_date is None:
        target_date = date.today()
    
    task_logger.info(f"📊 Calculating MTD loss for {target_date}")
    
    try:
        # Get configuration
        config = get_monthly_config()
        
        # Check if guard should run
        should_run, run_reason = should_run_monthly_guard(target_date)
        
        if not should_run:
            task_logger.info(f"⏭️ Monthly guard skipped: {run_reason}")
            return {
                'mtd_pct': 0.0,
                'should_run': False,
                'skip_reason': run_reason,
                'threshold_pct': config.get('threshold_pct', DEFAULT_THRESHOLD_PCT),
                'target_date': target_date.isoformat()
            }
        
        # Calculate MTD PnL
        mtd_pct = get_mtd_pnl_pct(target_date)
        threshold_pct = config.get('threshold_pct', DEFAULT_THRESHOLD_PCT)
        
        result = {
            'mtd_pct': mtd_pct,
            'threshold_pct': threshold_pct,
            'threshold_breached': mtd_pct <= threshold_pct,
            'should_run': True,
            'target_date': target_date.isoformat(),
            'config': config,
            'calculation_successful': True
        }
        
        # Log results
        task_logger.info(f"MTD PnL: {mtd_pct:+.3f}% (threshold: {threshold_pct}%)")
        
        if result['threshold_breached']:
            task_logger.warning(f"🚨 MONTHLY THRESHOLD BREACHED: {mtd_pct:+.3f}% ≤ {threshold_pct}%")
        else:
            task_logger.info(f"✅ MTD within limits: {mtd_pct:+.3f}% > {threshold_pct}%")
        
        return result
        
    except Exception as e:
        task_logger.error(f"❌ Failed to calculate MTD loss: {e}")
        return {
            'mtd_pct': 0.0,
            'should_run': False,
            'error': str(e),
            'calculation_successful': False,
            'target_date': target_date.isoformat()
        }


@task(name="check_monthly_stop", retries=1, retry_delay_seconds=30)
def check_monthly_stop_task(mtd_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check monthly stop-loss threshold and trigger kill-switch if breached
    
    Args:
        mtd_data: MTD calculation results from calc_mtd_loss_task
        
    Returns:
        Dictionary with stop-loss check results and actions taken
    """
    task_logger = get_run_logger()
    
    try:
        # Check if we should proceed
        if not mtd_data.get('should_run', False):
            task_logger.info(f"⏭️ Monthly stop check skipped: {mtd_data.get('skip_reason', 'Unknown')}")
            return {
                'action_taken': 'skipped',
                'reason': mtd_data.get('skip_reason', 'Guard not enabled'),
                'mtd_pct': mtd_data.get('mtd_pct', 0.0)
            }
        
        if not mtd_data.get('calculation_successful', False):
            task_logger.error("❌ MTD calculation failed, cannot proceed with stop check")
            return {
                'action_taken': 'calculation_failed',
                'reason': mtd_data.get('error', 'Unknown calculation error'),
                'mtd_pct': 0.0
            }
        
        mtd_pct = mtd_data['mtd_pct']
        threshold_pct = mtd_data['threshold_pct']
        threshold_breached = mtd_data.get('threshold_breached', False)
        
        task_logger.info(f"🔍 Checking monthly stop: {mtd_pct:+.3f}% vs {threshold_pct}%")
        
        result = {
            'mtd_pct': mtd_pct,
            'threshold_pct': threshold_pct,
            'threshold_breached': threshold_breached,
            'killswitch_triggered': False,
            'alert_sent': False,
            'action_taken': 'none',
            'timestamp': datetime.now().isoformat()
        }
        
        if threshold_breached:
            task_logger.error(f"🚨 MONTHLY STOP-LOSS TRIGGERED: {mtd_pct:+.3f}% ≤ {threshold_pct}%")
            
            # Check if this is a dry run
            config = mtd_data.get('config', {})
            dry_run = config.get('dry_run', False)
            
            if dry_run:
                task_logger.info("🧪 DRY RUN MODE - Kill-switch not actually triggered")
                result.update({
                    'action_taken': 'dry_run_triggered',
                    'killswitch_triggered': True,
                    'dry_run': True
                })
            else:
                # Trigger kill-switch
                killswitch_result = _trigger_monthly_killswitch(mtd_pct, task_logger)
                result.update(killswitch_result)
            
            # Send alert
            alert_result = _send_monthly_alert(mtd_pct, threshold_pct, task_logger)
            result.update(alert_result)
            
        else:
            task_logger.info(f"✅ Monthly stop within limits: {mtd_pct:+.3f}% > {threshold_pct}%")
            result['action_taken'] = 'within_limits'
        
        return result
        
    except Exception as e:
        task_logger.error(f"❌ Monthly stop check failed: {e}")
        return {
            'action_taken': 'check_failed',
            'error': str(e),
            'mtd_pct': mtd_data.get('mtd_pct', 0.0),
            'timestamp': datetime.now().isoformat()
        }


def _trigger_monthly_killswitch(mtd_pct: float, task_logger) -> Dict[str, Any]:
    """Trigger kill-switch for monthly stop-loss"""
    try:
        task_logger.error(f"🚨 Triggering kill-switch for {mtd_pct:+.3f}% monthly loss")
        
        # Prepare kill-switch command
        reason = f"{mtd_pct:+.3f}% monthly stop"
        cmd = [
            "python", "-m", "mech_exo.cli", "kill", "off",
            "--reason", reason,
            "--verbose"
        ]
        
        task_logger.info(f"Executing: {' '.join(cmd)}")
        
        # Run the kill-switch command
        proc_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd()
        )
        
        if proc_result.returncode == 0:
            task_logger.error(f"✅ Kill-switch triggered successfully")
            task_logger.info(f"Command output: {proc_result.stdout}")
            return {
                'killswitch_triggered': True,
                'action_taken': 'killswitch_triggered',
                'killswitch_reason': reason
            }
        else:
            task_logger.error(f"❌ Kill-switch command failed: {proc_result.stderr}")
            return {
                'killswitch_triggered': False,
                'action_taken': 'killswitch_failed',
                'killswitch_error': proc_result.stderr
            }
            
    except subprocess.TimeoutExpired:
        task_logger.error("❌ Kill-switch command timed out")
        return {
            'killswitch_triggered': False,
            'action_taken': 'killswitch_timeout'
        }
    except Exception as e:
        task_logger.error(f"❌ Kill-switch execution failed: {e}")
        return {
            'killswitch_triggered': False,
            'action_taken': 'killswitch_error',
            'killswitch_error': str(e)
        }


def _send_monthly_alert(mtd_pct: float, threshold_pct: float, task_logger) -> Dict[str, Any]:
    """Send monthly loss alert"""
    try:
        task_logger.info(f"📨 Sending monthly loss alert for {mtd_pct:+.3f}%")
        
        # Import alert function
        from mech_exo.utils.alerts import send_monthly_loss_alert
        
        success = send_monthly_loss_alert(mtd_pct, threshold_pct)
        
        if success:
            task_logger.info("✅ Monthly loss alert sent successfully")
            return {
                'alert_sent': True,
                'alert_status': 'sent'
            }
        else:
            task_logger.warning("⚠️ Monthly loss alert sending failed")
            return {
                'alert_sent': False,
                'alert_status': 'failed'
            }
            
    except ImportError:
        task_logger.warning("⚠️ Alert system not available - continuing without alerts")
        return {
            'alert_sent': False,
            'alert_status': 'not_available'
        }
    except Exception as e:
        task_logger.error(f"❌ Failed to send monthly alert: {e}")
        return {
            'alert_sent': False,
            'alert_status': 'error',
            'alert_error': str(e)
        }


@flow(
    name="monthly-drawdown-guard",
    description="End-of-day monthly P&L monitoring with -3% stop-loss protection",
    retries=1,
    retry_delay_seconds=30
)
def monthly_drawdown_guard_flow(
    target_date: date = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Monthly drawdown guard flow
    
    Monitors monthly P&L and triggers kill-switch at -3% threshold.
    Scheduled daily at 23:10 UTC.
    
    Args:
        target_date: Date to check (default: today)
        dry_run: If True, don't actually trigger kill-switch
        
    Returns:
        Flow execution results
    """
    flow_logger = get_run_logger()
    
    if target_date is None:
        target_date = date.today()
    
    flow_logger.info(f"🚀 Starting monthly drawdown guard for {target_date}")
    
    if dry_run:
        flow_logger.info("🧪 DRY RUN MODE - Kill-switch will not be triggered")
    
    try:
        # Step 1: Calculate month-to-date loss
        mtd_data = calc_mtd_loss_task(target_date)
        
        # Override dry_run setting if specified
        if dry_run and 'config' in mtd_data:
            mtd_data['config']['dry_run'] = True
        
        # Step 2: Check monthly stop threshold
        stop_data = check_monthly_stop_task(mtd_data)
        
        # Compile flow results
        flow_results = {
            'flow_status': 'SUCCESS',
            'target_date': target_date.isoformat(),
            'timestamp': datetime.now().isoformat(),
            'mtd_calculation': mtd_data,
            'stop_check': stop_data,
            'summary': {
                'mtd_pct': mtd_data.get('mtd_pct', 0.0),
                'threshold_breached': stop_data.get('threshold_breached', False),
                'killswitch_triggered': stop_data.get('killswitch_triggered', False),
                'alert_sent': stop_data.get('alert_sent', False),
                'action_taken': stop_data.get('action_taken', 'none')
            }
        }
        
        # Log flow summary
        mtd_pct = mtd_data.get('mtd_pct', 0.0)
        action_taken = stop_data.get('action_taken', 'none')
        
        if stop_data.get('threshold_breached', False):
            status_emoji = "🔴"
            flow_logger.error(f"MONTHLY STOP-LOSS TRIGGERED: {mtd_pct:+.3f}%")
        elif mtd_pct < -2.0:  # Warning level
            status_emoji = "🟡"
            flow_logger.warning(f"Monthly loss approaching threshold: {mtd_pct:+.3f}%")
        else:
            status_emoji = "🟢"
            flow_logger.info(f"Monthly P&L within normal range: {mtd_pct:+.3f}%")
        
        flow_logger.info(f"{status_emoji} Monthly guard complete: {mtd_pct:+.3f}% MTD")
        flow_logger.info(f"   Action taken: {action_taken}")
        flow_logger.info(f"   Kill-switch triggered: {stop_data.get('killswitch_triggered', False)}")
        flow_logger.info(f"   Alert sent: {stop_data.get('alert_sent', False)}")
        
        return flow_results
        
    except Exception as e:
        flow_logger.error(f"❌ Monthly drawdown guard flow failed: {e}")
        return {
            'flow_status': 'FAILED',
            'error': str(e),
            'target_date': target_date.isoformat(),
            'timestamp': datetime.now().isoformat()
        }


@flow(name="test-monthly-guard", description="Test monthly guard with specific MTD scenarios")
def test_monthly_guard_flow(test_scenarios: list = None) -> Dict[str, Any]:
    """
    Test the monthly guard with different MTD loss scenarios
    
    Args:
        test_scenarios: List of MTD percentages to test (default: [-3.5, -1.0, +1.0])
        
    Returns:
        Test results for all scenarios
    """
    flow_logger = get_run_logger()
    
    if test_scenarios is None:
        test_scenarios = [-3.5, -1.0, 1.0]  # Critical, warning, positive
    
    flow_logger.info(f"🧪 Testing monthly guard with scenarios: {test_scenarios}")
    
    test_results = []
    
    for mtd_pct in test_scenarios:
        flow_logger.info(f"📊 Testing scenario: {mtd_pct:+.1f}% MTD")
        
        try:
            # Create stub data for this scenario
            from mech_exo.utils.monthly_loss_guard import create_stub_monthly_data
            test_date = date.today()
            
            success = create_stub_monthly_data(test_date, mtd_pct)
            
            if not success:
                flow_logger.error(f"Failed to create stub data for {mtd_pct:+.1f}%")
                continue
            
            # Run the guard flow with dry_run=True
            result = monthly_drawdown_guard_flow(
                target_date=test_date,
                dry_run=True  # Always dry run for testing
            )
            
            scenario_result = {
                'test_mtd_pct': mtd_pct,
                'flow_status': result['flow_status'],
                'actual_mtd_pct': result['summary'].get('mtd_pct', 0.0),
                'threshold_breached': result['summary'].get('threshold_breached', False),
                'killswitch_triggered': result['summary'].get('killswitch_triggered', False),
                'action_taken': result['summary'].get('action_taken', 'none')
            }
            
            test_results.append(scenario_result)
            
            flow_logger.info(f"Scenario {mtd_pct:+.1f}%: {scenario_result['action_taken']}")
            
        except Exception as e:
            flow_logger.error(f"Scenario {mtd_pct:+.1f}% failed: {e}")
            test_results.append({
                'test_mtd_pct': mtd_pct,
                'flow_status': 'FAILED',
                'error': str(e)
            })
    
    return {
        'test_status': 'COMPLETE',
        'scenarios_tested': len(test_scenarios),
        'results': test_results,
        'timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Monthly Drawdown Guard Flow...")
    
    # Test with -3.5% MTD (should trigger kill-switch)
    print("\n1. Testing -3.5% MTD scenario...")
    result = test_monthly_guard_flow([-3.5])
    print(f"Result: {result['results'][0]['action_taken']}")
    
    # Test with +1% MTD (should pass)
    print("\n2. Testing +1% MTD scenario...")
    result = test_monthly_guard_flow([1.0])
    print(f"Result: {result['results'][0]['action_taken']}")
    
    print("✅ Monthly Drawdown Guard test complete")