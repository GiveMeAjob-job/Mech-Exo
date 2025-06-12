"""
Capital Guard Flow - Limit Sentinel

Daily Prefect flow that monitors capital utilization against configured limits.
Runs at 08:45 UTC to check IB buying power and update health status.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import os

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

logger = logging.getLogger(__name__)


@task(name="load_capital_limits", retries=2, retry_delay_seconds=30)
def load_capital_limits() -> Dict[str, Any]:
    """
    Load capital limits configuration
    
    Returns:
        Capital limits configuration
    """
    logger = get_run_logger()
    
    try:
        from mech_exo.cli.capital import CapitalManager
        
        # Load configuration
        manager = CapitalManager()
        config = manager.config
        
        logger.info("✅ Capital limits configuration loaded")
        return {
            'accounts': config['capital_limits']['accounts'],
            'global_settings': config['capital_limits']['global'],
            'manager': manager  # Pass manager for updates
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to load capital limits: {e}")
        raise


@task(name="fetch_buying_power", retries=3, retry_delay_seconds=60)
def fetch_buying_power(account_id: str, stub_mode: bool = None) -> Dict[str, Any]:
    """
    Fetch buying power from Interactive Brokers
    
    Args:
        account_id: IB account ID
        stub_mode: Override for testing (None = auto-detect)
        
    Returns:
        Account buying power and status
    """
    logger = get_run_logger()
    
    try:
        # Auto-detect stub mode if not specified
        if stub_mode is None:
            stub_mode = os.getenv('IB_STUB_MODE', 'false').lower() == 'true'
        
        if stub_mode:
            # Check for CI fixture data first
            ci_fixture = os.getenv('CI_CAPITAL_FIXTURE')
            if ci_fixture:
                import json
                try:
                    fixture_data = json.loads(ci_fixture)
                    if fixture_data.get('account_id') == account_id:
                        buying_power = fixture_data['buying_power']
                        logger.info(f"🧪 CI FIXTURE: Using buying power for {account_id}: ${buying_power:,.0f}")
                        logger.info(f"   Test case: {fixture_data.get('test_case', 'unknown')}")
                        logger.info(f"   Expected status: {fixture_data.get('expected_status', 'unknown')}")
                        
                        return {
                            'account_id': account_id,
                            'buying_power': buying_power,
                            'currency': fixture_data.get('currency', 'USD'),
                            'timestamp': datetime.now().isoformat(),
                            'status': 'success',
                            'source': 'ci_fixture',
                            'test_case': fixture_data.get('test_case'),
                            'expected_utilization_pct': fixture_data.get('expected_utilization_pct'),
                            'expected_status': fixture_data.get('expected_status')
                        }
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Invalid CI_CAPITAL_FIXTURE JSON: {e}")
            
            # Stub mode - return mock data for testing
            import random
            
            # Generate realistic buying power (80k-120k range)
            buying_power = random.uniform(80000, 120000)
            
            logger.info(f"🧪 STUB MODE: Generated buying power for {account_id}: ${buying_power:,.0f}")
            
            return {
                'account_id': account_id,
                'buying_power': buying_power,
                'currency': 'USD',
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'source': 'stub'
            }
        else:
            # Production mode - fetch from IB
            return _fetch_ib_buying_power(account_id)
            
    except Exception as e:
        logger.error(f"❌ Failed to fetch buying power for {account_id}: {e}")
        return {
            'account_id': account_id,
            'buying_power': None,
            'currency': 'USD',
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e),
            'source': 'ib'
        }


def _fetch_ib_buying_power(account_id: str) -> Dict[str, Any]:
    """
    Fetch buying power from Interactive Brokers API
    
    Args:
        account_id: IB account ID
        
    Returns:
        Account buying power information
    """
    logger = get_run_logger()
    
    try:
        # Import IB connection utilities
        from mech_exo.execution.broker_adapter import get_broker
        
        # Get broker connection
        broker = get_broker()
        
        # Request account summary for buying power
        # Note: This is a simplified version - production would need proper IB API calls
        if hasattr(broker, 'get_account_summary'):
            account_summary = broker.get_account_summary(account_id)
            buying_power = account_summary.get('BuyingPower', 0.0)
        else:
            # Fallback method
            logger.warning("Broker doesn't support get_account_summary, using fallback")
            buying_power = 100000.0  # Default fallback
        
        logger.info(f"✅ Fetched buying power for {account_id}: ${buying_power:,.0f}")
        
        return {
            'account_id': account_id,
            'buying_power': float(buying_power),
            'currency': 'USD',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'source': 'ib_api'
        }
        
    except Exception as e:
        logger.error(f"❌ IB API error for {account_id}: {e}")
        raise


@task(name="check_capital_utilization", retries=1)
def check_capital_utilization(account_config: Dict[str, Any], 
                            buying_power_data: Dict[str, Any],
                            global_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check capital utilization against limits
    
    Args:
        account_config: Account configuration
        buying_power_data: Current buying power data
        global_settings: Global limit settings
        
    Returns:
        Utilization analysis and status
    """
    logger = get_run_logger()
    
    try:
        account_id = buying_power_data['account_id']
        buying_power = buying_power_data.get('buying_power')
        max_capital = account_config['max_capital']
        
        if buying_power is None:
            return {
                'account_id': account_id,
                'status': 'error',
                'message': 'No buying power data available',
                'capital_ok': False
            }
        
        # Calculate utilization
        used_capital = max_capital - buying_power  # Simplified calculation
        if used_capital < 0:
            used_capital = 0  # Can't use negative capital
        
        utilization_pct = (used_capital / max_capital * 100) if max_capital > 0 else 0
        
        # Determine status based on thresholds
        warning_threshold = global_settings['alerts']['warning_threshold_pct']
        critical_threshold = global_settings['alerts']['critical_threshold_pct']
        
        if utilization_pct >= critical_threshold:
            status = 'critical'
            capital_ok = False
            message = f"Critical: {utilization_pct:.1f}% utilization (≥{critical_threshold}%)"
        elif utilization_pct >= warning_threshold:
            status = 'warning'
            capital_ok = True  # Still OK but warned
            message = f"Warning: {utilization_pct:.1f}% utilization (≥{warning_threshold}%)"
        else:
            status = 'ok'
            capital_ok = True
            message = f"OK: {utilization_pct:.1f}% utilization"
        
        result = {
            'account_id': account_id,
            'buying_power': buying_power,
            'max_capital': max_capital,
            'used_capital': used_capital,
            'utilization_pct': utilization_pct,
            'status': status,
            'capital_ok': capital_ok,
            'message': message,
            'last_updated': datetime.now().isoformat(),
            'thresholds': {
                'warning': warning_threshold,
                'critical': critical_threshold
            }
        }
        
        logger.info(f"✅ {account_id}: {message}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to check utilization: {e}")
        return {
            'account_id': buying_power_data.get('account_id', 'unknown'),
            'status': 'error',
            'message': f'Utilization check failed: {e}',
            'capital_ok': False
        }


@task(name="update_utilization_data", retries=2)
def update_utilization_data(utilization_results: List[Dict[str, Any]], 
                           capital_manager) -> bool:
    """
    Update utilization data in configuration
    
    Args:
        utilization_results: List of utilization check results
        capital_manager: CapitalManager instance
        
    Returns:
        True if successful
    """
    logger = get_run_logger()
    
    try:
        # Update utilization section in config
        if 'utilization' not in capital_manager.config:
            capital_manager.config['utilization'] = {'accounts': {}}
        
        capital_manager.config['utilization']['last_check'] = datetime.now().isoformat()
        
        # Update each account's utilization data
        for result in utilization_results:
            account_id = result['account_id']
            
            # Only update if we have valid data
            if result.get('buying_power') is not None:
                capital_manager.config['utilization']['accounts'][account_id] = {
                    'buying_power': result['buying_power'],
                    'used_capital': result.get('used_capital', 0),
                    'utilization_pct': result.get('utilization_pct', 0),
                    'last_updated': result['last_updated'],
                    'status': result['status']
                }
        
        # Save configuration
        capital_manager._save_config()
        
        logger.info(f"✅ Updated utilization data for {len(utilization_results)} accounts")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update utilization data: {e}")
        return False


@task(name="send_capital_alerts", retries=2)
def send_capital_alerts(utilization_results: List[Dict[str, Any]]) -> bool:
    """
    Send alerts for capital limit violations
    
    Args:
        utilization_results: List of utilization check results
        
    Returns:
        True if alerts sent successfully
    """
    logger = get_run_logger()
    
    try:
        from mech_exo.utils.alerts import AlertManager, Alert, AlertType, AlertLevel
        
        alert_manager = AlertManager()
        alerts_sent = 0
        
        for result in utilization_results:
            account_id = result['account_id']
            status = result.get('status', 'unknown')
            
            # Send alerts for warning and critical statuses
            if status in ['warning', 'critical']:
                alert_level = AlertLevel.CRITICAL if status == 'critical' else AlertLevel.WARNING
                
                alert = Alert(
                    alert_type=AlertType.SYSTEM_ALERT,
                    level=alert_level,
                    title=f"💰 Capital Limit Alert - {account_id}",
                    message=f"Account: {account_id}\n"
                           f"Status: {status.upper()}\n"
                           f"Utilization: {result.get('utilization_pct', 0):.1f}%\n"
                           f"Buying Power: ${result.get('buying_power', 0):,.0f}\n"
                           f"Max Capital: ${result.get('max_capital', 0):,.0f}\n\n"
                           f"Message: {result.get('message', 'No details available')}",
                    timestamp=datetime.now(),
                    data=result
                )
                
                # Send alert with escalation support
                success = alert_manager.send_alert_with_escalation(
                    alert, 
                    channels=['telegram'],
                    respect_quiet_hours=False,  # Capital alerts are always urgent
                    force_send=True
                )
                
                if success:
                    alerts_sent += 1
                    logger.info(f"✅ Capital alert sent for {account_id}: {status}")
                else:
                    logger.warning(f"⚠️ Failed to send capital alert for {account_id}")
        
        logger.info(f"✅ Sent {alerts_sent} capital alerts")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to send capital alerts: {e}")
        return False


@task(name="update_health_status", retries=1)
def update_health_status(utilization_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Update global health status based on capital checks
    
    Args:
        utilization_results: List of utilization check results
        
    Returns:
        Health status summary
    """
    logger = get_run_logger()
    
    try:
        # Calculate overall capital health
        total_accounts = len(utilization_results)
        healthy_accounts = sum(1 for r in utilization_results if r.get('capital_ok', False))
        critical_accounts = sum(1 for r in utilization_results if r.get('status') == 'critical')
        warning_accounts = sum(1 for r in utilization_results if r.get('status') == 'warning')
        
        # Overall capital_ok status
        capital_ok = critical_accounts == 0  # No critical violations
        
        health_status = {
            'capital_ok': capital_ok,
            'total_accounts': total_accounts,
            'healthy_accounts': healthy_accounts,
            'warning_accounts': warning_accounts,
            'critical_accounts': critical_accounts,
            'last_check': datetime.now().isoformat(),
            'check_type': 'daily_capital_guard'
        }
        
        # This status will be picked up by the /healthz endpoint
        # Store in a location accessible by the health endpoint
        _store_health_status(health_status)
        
        status_msg = f"Capital Health: {healthy_accounts}/{total_accounts} OK"
        if critical_accounts > 0:
            status_msg += f", {critical_accounts} CRITICAL"
        if warning_accounts > 0:
            status_msg += f", {warning_accounts} WARNING"
        
        logger.info(f"✅ {status_msg}")
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Failed to update health status: {e}")
        return {
            'capital_ok': False,
            'error': str(e),
            'last_check': datetime.now().isoformat()
        }


def _store_health_status(health_status: Dict[str, Any]):
    """
    Store health status for access by health endpoint
    
    Args:
        health_status: Health status data
    """
    try:
        # Store in database for health endpoint access
        from mech_exo.datasource.storage import DataStorage
        
        storage = DataStorage()
        
        # Create or update capital_health table
        storage.conn.execute("""
            CREATE TABLE IF NOT EXISTS capital_health (
                id INTEGER PRIMARY KEY,
                capital_ok BOOLEAN,
                total_accounts INTEGER,
                healthy_accounts INTEGER,
                warning_accounts INTEGER, 
                critical_accounts INTEGER,
                last_check TIMESTAMP,
                data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert or update health status
        import json
        storage.conn.execute("""
            INSERT OR REPLACE INTO capital_health 
            (id, capital_ok, total_accounts, healthy_accounts, warning_accounts, 
             critical_accounts, last_check, data)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?)
        """, [
            health_status['capital_ok'],
            health_status['total_accounts'],
            health_status['healthy_accounts'],
            health_status['warning_accounts'],
            health_status['critical_accounts'],
            health_status['last_check'],
            json.dumps(health_status)
        ])
        
        storage.conn.commit()
        storage.close()
        
        logger.info("✅ Health status stored in database")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to store health status: {e}")


@flow(
    name="capital-guard-flow",
    description="Daily capital limit monitoring and health checks",
    task_runner=SequentialTaskRunner(),
    log_prints=True,
    retries=1,
    retry_delay_seconds=300
)
def capital_guard_flow(stub_mode: bool = None) -> Dict[str, Any]:
    """
    Main capital guard flow
    
    Monitors capital utilization against configured limits and updates health status.
    Designed to run daily at 08:45 UTC.
    
    Args:
        stub_mode: Force stub mode for testing (None = auto-detect)
        
    Returns:
        Flow execution summary
    """
    logger = get_run_logger()
    
    logger.info("🏦 Starting Capital Guard Flow - Limit Sentinel")
    logger.info(f"Execution time: {datetime.now().isoformat()}")
    
    try:
        # Load capital limits configuration
        config_data = load_capital_limits()
        accounts = config_data['accounts']
        global_settings = config_data['global_settings']
        capital_manager = config_data['manager']
        
        if not accounts:
            logger.warning("⚠️ No accounts configured for capital monitoring")
            return {
                'status': 'success',
                'message': 'No accounts to monitor',
                'accounts_checked': 0,
                'capital_ok': True
            }
        
        logger.info(f"📋 Monitoring {len(accounts)} accounts")
        
        # Fetch buying power for all enabled accounts
        buying_power_futures = []
        enabled_accounts = {}
        
        for account_id, account_config in accounts.items():
            if account_config.get('enabled', True):
                enabled_accounts[account_id] = account_config
                future = fetch_buying_power.submit(account_id, stub_mode)
                buying_power_futures.append(future)
        
        # Wait for all buying power fetches to complete
        buying_power_results = [future.result() for future in buying_power_futures]
        
        # Check capital utilization for each account
        utilization_futures = []
        for buying_power_data in buying_power_results:
            account_id = buying_power_data['account_id']
            if account_id in enabled_accounts:
                future = check_capital_utilization.submit(
                    enabled_accounts[account_id],
                    buying_power_data,
                    global_settings
                )
                utilization_futures.append(future)
        
        # Wait for all utilization checks to complete
        utilization_results = [future.result() for future in utilization_futures]
        
        # Update utilization data in configuration
        update_success = update_utilization_data(utilization_results, capital_manager)
        
        # Send alerts for any violations
        alerts_success = send_capital_alerts(utilization_results)
        
        # Update global health status
        health_status = update_health_status(utilization_results)
        
        # Calculate summary statistics
        total_checked = len(utilization_results)
        healthy_count = sum(1 for r in utilization_results if r.get('capital_ok', False))
        critical_count = sum(1 for r in utilization_results if r.get('status') == 'critical')
        
        summary = {
            'status': 'success',
            'accounts_checked': total_checked,
            'healthy_accounts': healthy_count,
            'critical_accounts': critical_count,
            'capital_ok': health_status.get('capital_ok', False),
            'update_success': update_success,
            'alerts_success': alerts_success,
            'execution_time': datetime.now().isoformat(),
            'health_status': health_status
        }
        
        logger.info(f"✅ Capital Guard Flow completed successfully")
        logger.info(f"📊 Summary: {healthy_count}/{total_checked} accounts healthy, "
                   f"{critical_count} critical violations")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Capital Guard Flow failed: {e}")
        
        # Update health status to indicate failure
        error_health = {
            'capital_ok': False,
            'error': str(e),
            'last_check': datetime.now().isoformat(),
            'check_type': 'daily_capital_guard_failed'
        }
        _store_health_status(error_health)
        
        return {
            'status': 'failed',
            'error': str(e),
            'accounts_checked': 0,
            'capital_ok': False,
            'execution_time': datetime.now().isoformat()
        }


# Deployment configuration for scheduled execution
if __name__ == "__main__":
    # Test execution
    result = capital_guard_flow(stub_mode=True)
    print(f"Flow result: {result}")