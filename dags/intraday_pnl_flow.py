"""
Intraday PnL Sentinel Flow

5-minute monitoring flow that tracks intraday PnL and triggers kill-switch 
at -0.8% day-loss threshold. Includes alerts and metrics storage.
"""

import logging
import subprocess
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import os

from prefect import flow, task
from prefect.logging import get_run_logger

from mech_exo.reporting.pnl_live import LivePnLMonitor, get_live_nav, create_test_positions_and_nav
from mech_exo.cli.killswitch import is_trading_enabled, get_kill_switch_status

logger = logging.getLogger(__name__)

# Configuration
LOSS_THRESHOLD_PCT = -0.8  # -0.8% day-loss threshold
WARNING_THRESHOLD_PCT = -0.4  # Warning threshold at -0.4%


@task(name="pull_live_nav", retries=2, retry_delay_seconds=10)
def pull_live_nav_task(tag: str = 'all', test_mode: bool = False, test_pnl_pct: Optional[float] = None) -> Dict[str, Any]:
    """
    Pull live NAV and calculate day-to-date PnL
    
    Args:
        tag: Portfolio tag to monitor
        test_mode: If True, use test data
        test_pnl_pct: Test PnL percentage for simulation
        
    Returns:
        Dict with NAV data and PnL metrics
    """
    task_logger = get_run_logger()
    task_logger.info(f"📊 Pulling live NAV for tag '{tag}' (test_mode={test_mode})")
    
    try:
        if test_mode and test_pnl_pct is not None:
            # Use test data for simulation
            task_logger.info(f"Using test data with PnL: {test_pnl_pct:+.2f}%")
            nav_data = create_test_positions_and_nav(test_pnl_pct)
        else:
            # Get real live NAV
            nav_data = get_live_nav(tag)
        
        if not nav_data.get('calculation_successful', False):
            task_logger.error(f"NAV calculation failed: {nav_data.get('error', 'Unknown error')}")
            raise RuntimeError(f"NAV calculation failed: {nav_data.get('error', 'Unknown error')}")
        
        # Log key metrics
        task_logger.info(f"Live NAV: ${nav_data['live_nav']:,.2f}")
        task_logger.info(f"Day PnL: {nav_data['pnl_pct']:+.3f}%")
        task_logger.info(f"Positions: {nav_data['position_count']}")
        
        return nav_data
        
    except Exception as e:
        task_logger.error(f"❌ Failed to pull live NAV: {e}")
        raise


@task(name="check_day_loss", retries=1, retry_delay_seconds=5)
def check_day_loss_task(nav_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if day loss exceeds threshold and trigger kill-switch if needed
    
    Args:
        nav_data: NAV data from pull_live_nav_task
        
    Returns:
        Dict with threshold check results and actions taken
    """
    task_logger = get_run_logger()
    
    pnl_pct = nav_data['pnl_pct']
    task_logger.info(f"🔍 Checking day loss: {pnl_pct:+.3f}% vs threshold {LOSS_THRESHOLD_PCT}%")
    
    result = {
        'pnl_pct': pnl_pct,
        'threshold_breached': False,
        'warning_level': False,
        'killswitch_triggered': False,
        'killswitch_already_off': False,
        'action_taken': 'none',
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Check current kill-switch status
        trading_enabled = is_trading_enabled()
        
        if pnl_pct <= LOSS_THRESHOLD_PCT:
            # Critical threshold breached
            result['threshold_breached'] = True
            task_logger.warning(f"🚨 CRITICAL: Day loss {pnl_pct:+.3f}% exceeds threshold {LOSS_THRESHOLD_PCT}%")
            
            if trading_enabled:
                # Trigger kill-switch
                task_logger.error(f"🚨 Triggering kill-switch for {pnl_pct:+.3f}% day-loss")
                
                try:
                    # Call exo kill command
                    cmd = [
                        "python", "-m", "mech_exo.cli", "kill", "off",
                        "--reason", f"{pnl_pct:+.3f}% day-loss breach",
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
                        result['killswitch_triggered'] = True
                        result['action_taken'] = 'killswitch_triggered'
                        task_logger.error(f"✅ Kill-switch triggered successfully")
                        task_logger.info(f"Command output: {proc_result.stdout}")
                    else:
                        task_logger.error(f"❌ Kill-switch command failed: {proc_result.stderr}")
                        result['action_taken'] = 'killswitch_failed'
                        
                except subprocess.TimeoutExpired:
                    task_logger.error("❌ Kill-switch command timed out")
                    result['action_taken'] = 'killswitch_timeout'
                except Exception as e:
                    task_logger.error(f"❌ Kill-switch execution failed: {e}")
                    result['action_taken'] = 'killswitch_error'
            else:
                task_logger.warning("⚠️ Kill-switch already disabled - no action needed")
                result['killswitch_already_off'] = True
                result['action_taken'] = 'already_disabled'
                
        elif pnl_pct <= WARNING_THRESHOLD_PCT:
            # Warning threshold
            result['warning_level'] = True
            task_logger.warning(f"⚠️ WARNING: Day loss {pnl_pct:+.3f}% exceeds warning threshold {WARNING_THRESHOLD_PCT}%")
            result['action_taken'] = 'warning_logged'
            
        else:
            # Normal operation
            task_logger.info(f"✅ Day loss {pnl_pct:+.3f}% within normal range")
            result['action_taken'] = 'normal_operation'
        
        return result
        
    except Exception as e:
        task_logger.error(f"❌ Day loss check failed: {e}")
        result['action_taken'] = 'check_failed'
        result['error'] = str(e)
        return result


@task(name="push_metrics", retries=2, retry_delay_seconds=5)
def push_metrics_task(nav_data: Dict[str, Any], threshold_result: Dict[str, Any]) -> bool:
    """
    Write intraday metrics to DuckDB/SQLite database
    
    Args:
        nav_data: NAV data from pull_live_nav_task
        threshold_result: Threshold check results
        
    Returns:
        True if metrics were successfully recorded
    """
    task_logger = get_run_logger()
    task_logger.info("💾 Recording intraday metrics to database")
    
    try:
        monitor = LivePnLMonitor()
        
        # Add threshold check results to nav_data
        enhanced_nav_data = nav_data.copy()
        enhanced_nav_data.update({
            'threshold_breached': threshold_result.get('threshold_breached', False),
            'warning_level': threshold_result.get('warning_level', False),
            'alerts_triggered': threshold_result.get('killswitch_triggered', False)
        })
        
        success = monitor.record_intraday_metrics(enhanced_nav_data)
        monitor.close()
        
        if success:
            task_logger.info("✅ Metrics recorded successfully")
        else:
            task_logger.error("❌ Failed to record metrics")
            
        return success
        
    except Exception as e:
        task_logger.error(f"❌ Failed to push metrics: {e}")
        return False


@task(name="send_alerts", retries=1, retry_delay_seconds=5)
def send_alerts_task(nav_data: Dict[str, Any], threshold_result: Dict[str, Any]) -> bool:
    """
    Send alerts for threshold breaches
    
    Args:
        nav_data: NAV data
        threshold_result: Threshold check results
        
    Returns:
        True if alerts were sent successfully
    """
    task_logger = get_run_logger()
    
    # Only send alerts for significant events
    if not (threshold_result.get('threshold_breached', False) or threshold_result.get('warning_level', False)):
        task_logger.info("No alerts needed - PnL within normal range")
        return True
    
    try:
        from mech_exo.utils.alerts import send_intraday_loss_alert
        
        pnl_pct = nav_data['pnl_pct']
        action_taken = threshold_result.get('action_taken', 'none')
        
        task_logger.info(f"🚨 Sending intraday loss alert for {pnl_pct:+.3f}%")
        
        success = send_intraday_loss_alert(
            pnl_pct=pnl_pct,
            nav_data=nav_data,
            threshold_result=threshold_result
        )
        
        if success:
            task_logger.info("✅ Alert sent successfully")
        else:
            task_logger.warning("⚠️ Alert sending failed")
            
        return success
        
    except ImportError:
        task_logger.warning("Alert system not available - continuing without alerts")
        return True
    except Exception as e:
        task_logger.error(f"❌ Failed to send alerts: {e}")
        return False


@flow(
    name="intraday-pnl-sentinel",
    description="5-minute intraday PnL monitoring with kill-switch protection",
    retries=1,
    retry_delay_seconds=30
)
def intraday_pnl_sentinel_flow(
    tag: str = 'all', 
    test_mode: bool = False, 
    test_pnl_pct: Optional[float] = None,
    send_alerts: bool = True
) -> Dict[str, Any]:
    """
    Intraday PnL sentinel flow
    
    Monitors portfolio PnL every 5 minutes and triggers kill-switch at -0.8% loss.
    
    Args:
        tag: Portfolio tag to monitor
        test_mode: If True, use test data
        test_pnl_pct: Test PnL percentage for simulation
        send_alerts: If True, send alerts for threshold breaches
        
    Returns:
        Flow execution results
    """
    flow_logger = get_run_logger()
    flow_logger.info(f"🚀 Starting intraday PnL sentinel (tag={tag}, test_mode={test_mode})")
    
    try:
        # Step 1: Pull live NAV
        nav_data = pull_live_nav_task(tag, test_mode, test_pnl_pct)
        
        # Step 2: Check for day loss threshold breach
        threshold_result = check_day_loss_task(nav_data)
        
        # Step 3: Record metrics to database
        metrics_recorded = push_metrics_task(nav_data, threshold_result)
        
        # Step 4: Send alerts if needed
        alerts_sent = True
        if send_alerts:
            alerts_sent = send_alerts_task(nav_data, threshold_result)
        
        # Step 5: Compile results
        flow_results = {
            'flow_status': 'SUCCESS',
            'timestamp': datetime.now().isoformat(),
            'nav_data': nav_data,
            'threshold_result': threshold_result,
            'metrics_recorded': metrics_recorded,
            'alerts_sent': alerts_sent,
            'monitoring_config': {
                'loss_threshold_pct': LOSS_THRESHOLD_PCT,
                'warning_threshold_pct': WARNING_THRESHOLD_PCT,
                'tag': tag,
                'test_mode': test_mode
            }
        }
        
        # Log summary
        pnl_pct = nav_data['pnl_pct']
        status_emoji = "🔴" if threshold_result.get('threshold_breached') else ("🟡" if threshold_result.get('warning_level') else "🟢")
        
        flow_logger.info(f"{status_emoji} Sentinel complete: {pnl_pct:+.3f}% PnL")
        flow_logger.info(f"   Action taken: {threshold_result.get('action_taken', 'none')}")
        flow_logger.info(f"   Metrics recorded: {metrics_recorded}")
        flow_logger.info(f"   Alerts sent: {alerts_sent}")
        
        return flow_results
        
    except Exception as e:
        flow_logger.error(f"❌ Intraday PnL sentinel flow failed: {e}")
        return {
            'flow_status': 'FAILED',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@flow(name="test-pnl-sentinel", description="Test intraday PnL sentinel with specific scenarios")
def test_pnl_sentinel_flow(scenarios: list = None) -> Dict[str, Any]:
    """
    Test the PnL sentinel with different loss scenarios
    
    Args:
        scenarios: List of PnL percentages to test (default: [-1.0, -0.5, +0.2])
        
    Returns:
        Test results for all scenarios
    """
    flow_logger = get_run_logger()
    
    if scenarios is None:
        scenarios = [-1.0, -0.5, 0.2]  # -1%, -0.5%, +0.2%
    
    flow_logger.info(f"🧪 Testing PnL sentinel with scenarios: {scenarios}")
    
    test_results = []
    
    for pnl_pct in scenarios:
        flow_logger.info(f"📊 Testing scenario: {pnl_pct:+.1f}%")
        
        try:
            result = intraday_pnl_sentinel_flow(
                tag='test',
                test_mode=True,
                test_pnl_pct=pnl_pct,
                send_alerts=False  # Don't spam alerts during testing
            )
            
            scenario_result = {
                'test_pnl_pct': pnl_pct,
                'flow_status': result['flow_status'],
                'killswitch_triggered': result['threshold_result'].get('killswitch_triggered', False),
                'warning_level': result['threshold_result'].get('warning_level', False),
                'action_taken': result['threshold_result'].get('action_taken', 'none')
            }
            
            test_results.append(scenario_result)
            
            flow_logger.info(f"Scenario {pnl_pct:+.1f}%: {scenario_result['action_taken']}")
            
        except Exception as e:
            flow_logger.error(f"Scenario {pnl_pct:+.1f}% failed: {e}")
            test_results.append({
                'test_pnl_pct': pnl_pct,
                'flow_status': 'FAILED',
                'error': str(e)
            })
    
    return {
        'test_status': 'COMPLETE',
        'scenarios_tested': len(scenarios),
        'results': test_results,
        'timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Intraday PnL Sentinel Flow...")
    
    # Test with -1% loss (should trigger kill-switch)
    print("\n1. Testing -1% loss scenario...")
    result = intraday_pnl_sentinel_flow(
        tag='test',
        test_mode=True,
        test_pnl_pct=-1.0,
        send_alerts=False
    )
    print(f"Result: {result['threshold_result']['action_taken']}")
    
    # Test with +0.2% gain (should be normal)
    print("\n2. Testing +0.2% gain scenario...")
    result = intraday_pnl_sentinel_flow(
        tag='test',
        test_mode=True,
        test_pnl_pct=0.2,
        send_alerts=False
    )
    print(f"Result: {result['threshold_result']['action_taken']}")
    
    print("✅ Intraday PnL Sentinel test complete")