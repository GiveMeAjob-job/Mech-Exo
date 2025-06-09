"""
Prefect flow for drift monitoring automation

Runs daily drift calculation to compare live performance vs backtest,
stores metrics in DuckDB, and sends alerts for significant drift.
"""

import logging
import json
from datetime import datetime, timedelta, date
from typing import Dict, Optional

try:
    from prefect import flow, task
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    # Create dummy decorators for when Prefect is not available
    def flow(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func
    
    def task(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

try:
    # Try relative imports first (when run as part of package)
    from ..datasource.storage import DataStorage
    from ..reporting.drift import calculate_daily_drift, get_drift_status
    from ..utils.alerts import AlertManager
except ImportError:
    # Fall back to absolute imports (when run standalone)
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from mech_exo.datasource.storage import DataStorage
    from mech_exo.reporting.drift import calculate_daily_drift, get_drift_status
    from mech_exo.utils.alerts import AlertManager

logger = logging.getLogger(__name__)


@task(name="calc_drift", retries=2, retry_delay_seconds=60)
def calc_drift(target_date: str = None) -> Dict:
    """
    Calculate drift metrics using DriftMetricEngine
    
    Args:
        target_date: Date to calculate drift for (YYYY-MM-DD, defaults to today)
        
    Returns:
        Dict with drift metrics
    """
    try:
        if target_date:
            target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
        else:
            target_date_obj = date.today()
        
        logger.info(f"Calculating drift metrics for {target_date_obj}")
        
        # Calculate drift metrics
        metrics = calculate_daily_drift(target_date_obj)
        
        # Add calculated timestamp
        metrics['calculated_at'] = datetime.now().isoformat()
        
        logger.info(f"Drift calculation completed: drift={metrics['drift_pct']:.1f}%, "
                   f"IR={metrics['information_ratio']:.2f}, "
                   f"quality={metrics['data_quality']}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to calculate drift metrics: {e}")
        # Return default metrics on error
        return {
            'date': target_date or date.today().isoformat(),
            'live_cagr': 0.0,
            'backtest_cagr': 0.0,
            'drift_pct': 0.0,
            'information_ratio': 0.0,
            'excess_return_mean': 0.0,
            'excess_return_std': 0.0,
            'tracking_error': 0.0,
            'data_quality': 'error',
            'days_analyzed': 0,
            'calculated_at': datetime.now().isoformat()
        }


@task(name="store_drift_metrics", retries=3, retry_delay_seconds=30)
def store_drift_metrics(drift_data: Dict) -> bool:
    """
    Store drift metrics in DuckDB drift_metrics table
    
    Args:
        drift_data: Dictionary containing drift metrics
        
    Returns:
        True if successful
    """
    try:
        # Connect to DuckDB
        storage = DataStorage()
        
        # Create table if it doesn't exist
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS drift_metrics (
            drift_date DATE,
            calculated_at TIMESTAMP,
            live_cagr DECIMAL,
            backtest_cagr DECIMAL,
            drift_pct DECIMAL,
            information_ratio DECIMAL,
            excess_return_mean DECIMAL,
            excess_return_std DECIMAL,
            tracking_error DECIMAL,
            data_quality VARCHAR,
            days_analyzed INTEGER,
            drift_status VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (drift_date)
        );
        """
        
        storage.conn.execute(create_table_sql)
        
        # Calculate drift status
        drift_status = get_drift_status(drift_data['drift_pct'], drift_data['information_ratio'])
        
        # Insert or replace metrics (upsert on drift_date)
        insert_sql = """
        INSERT OR REPLACE INTO drift_metrics (
            drift_date, calculated_at, live_cagr, backtest_cagr, drift_pct,
            information_ratio, excess_return_mean, excess_return_std, tracking_error,
            data_quality, days_analyzed, drift_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        storage.conn.execute(insert_sql, [
            drift_data['date'],
            drift_data['calculated_at'],
            drift_data['live_cagr'],
            drift_data['backtest_cagr'],
            drift_data['drift_pct'],
            drift_data['information_ratio'],
            drift_data['excess_return_mean'],
            drift_data['excess_return_std'],
            drift_data['tracking_error'],
            drift_data['data_quality'],
            drift_data['days_analyzed'],
            drift_status
        ])
        
        logger.info(f"Stored drift metrics for {drift_data['date']} with status: {drift_status}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store drift metrics: {e}")
        return False


@task(name="alert_if_breach", retries=2, retry_delay_seconds=30)
def alert_if_breach(drift_data: Dict) -> bool:
    """
    Send Slack alert if drift exceeds thresholds
    
    Args:
        drift_data: Dictionary containing drift metrics
        
    Returns:
        True if alert was sent or not needed, False if alert failed
    """
    try:
        drift_pct = drift_data['drift_pct']
        information_ratio = drift_data['information_ratio']
        data_quality = drift_data['data_quality']
        
        # Get drift status
        drift_status = get_drift_status(drift_pct, information_ratio)
        
        # Only alert on WARN or BREACH
        if drift_status not in ['WARN', 'BREACH']:
            logger.info(f"Drift status OK ({drift_pct:.1f}%), no alert needed")
            return True
        
        # Skip alerts for poor data quality
        if data_quality in ['error', 'no_backtest', 'no_overlap']:
            logger.warning(f"Poor data quality ({data_quality}), skipping alert")
            return True
        
        # Prepare alert message
        if drift_status == 'BREACH':
            icon = "🚨"
            severity = "BREACH"
        else:
            icon = "⚠️"
            severity = "WARNING"
        
        # Format the alert message
        message = f"""{icon} **Drift Alert: {severity}**

**Date**: {drift_data['date']}
**Drift**: {drift_pct:.1f}% (threshold: ±10%)
**Information Ratio**: {information_ratio:.2f} (threshold: >0.2)
**Status**: {drift_status}

**Performance Comparison**:
• Live CAGR: {drift_data['live_cagr']:.2%}
• Backtest CAGR: {drift_data['backtest_cagr']:.2%}
• Tracking Error: {drift_data['tracking_error']:.2%}

**Analysis Quality**: {data_quality} ({drift_data['days_analyzed']} days)

Live performance is drifting from backtest expectations — please investigate."""
        
        # Send alert via AlertManager
        alert_manager = AlertManager()
        
        # Try to send alert
        success = alert_manager.send_alert(
            subject=f"Drift {severity}: {drift_pct:.1f}%",
            message=message,
            urgency="high" if drift_status == 'BREACH' else "medium"
        )
        
        if success:
            logger.info(f"Sent {drift_status} alert for {drift_pct:.1f}% drift")
        else:
            logger.error(f"Failed to send {drift_status} alert")
            
        return success
        
    except Exception as e:
        logger.error(f"Failed to process drift alert: {e}")
        return False


@flow(name="drift_monitor_flow", retries=1, retry_delay_seconds=120)
def drift_monitor_flow(target_date: str = None) -> Dict:
    """
    Daily drift monitoring flow
    
    Calculates drift metrics, stores in database, and sends alerts for breaches.
    Scheduled to run daily at 08:45 UTC (03:45 EST) after market close.
    
    Args:
        target_date: Date to analyze (YYYY-MM-DD, defaults to today)
        
    Returns:
        Dict with flow results
    """
    logger.info(f"🔍 Starting drift monitor flow for {target_date or 'today'}")
    
    try:
        # Step 1: Calculate drift metrics
        drift_metrics = calc_drift(target_date)
        
        # Step 2: Store metrics in database
        store_success = store_drift_metrics(drift_metrics)
        
        # Step 3: Check for alerts
        alert_success = alert_if_breach(drift_metrics)
        
        # Summary
        results = {
            'flow_status': 'completed',
            'drift_date': drift_metrics['date'],
            'drift_pct': drift_metrics['drift_pct'],
            'drift_status': get_drift_status(drift_metrics['drift_pct'], drift_metrics['information_ratio']),
            'information_ratio': drift_metrics['information_ratio'],
            'data_quality': drift_metrics['data_quality'],
            'days_analyzed': drift_metrics['days_analyzed'],
            'store_success': store_success,
            'alert_success': alert_success,
            'completed_at': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Drift monitor flow completed: "
                   f"drift={drift_metrics['drift_pct']:.1f}%, "
                   f"status={results['drift_status']}, "
                   f"stored={store_success}, alerted={alert_success}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Drift monitor flow failed: {e}")
        return {
            'flow_status': 'failed',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        }


def create_drift_monitor_deployment():
    """
    Create Prefect deployment for drift monitoring
    
    Scheduled to run daily at 08:45 UTC (03:45 EST)
    """
    if not PREFECT_AVAILABLE:
        logger.warning("Prefect not available, cannot create deployment")
        return None
    
    try:
        deployment = Deployment.build_from_flow(
            flow=drift_monitor_flow,
            name="drift-monitor-daily",
            schedule=CronSchedule(cron="45 8 * * *"),  # 08:45 UTC = 03:45 EST
            parameters={},
            work_pool_name="default-agent-pool",
            description="Daily drift monitoring between live and backtest performance",
            tags=["drift", "monitoring", "daily", "alerts"]
        )
        
        deployment_id = deployment.apply()
        logger.info(f"Created drift monitor deployment: {deployment_id}")
        return deployment_id
        
    except Exception as e:
        logger.error(f"Failed to create drift monitor deployment: {e}")
        return None


def run_manual_drift_monitor(target_date: str = None) -> Dict:
    """
    Run drift monitor manually for testing
    
    Args:
        target_date: Date to analyze (YYYY-MM-DD, defaults to today)
        
    Returns:
        Dict with results
    """
    logger.info(f"🧪 Running manual drift monitor for {target_date or 'today'}")
    
    # Run the flow without Prefect orchestration
    return drift_monitor_flow(target_date)


if __name__ == "__main__":
    # Test the drift monitor flow
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Testing Drift Monitor Flow...")
    
    # Test manual run
    result = run_manual_drift_monitor()
    
    print(f"\n📊 Flow Results:")
    print(f"   • Status: {result.get('flow_status', 'unknown')}")
    print(f"   • Date: {result.get('drift_date', 'unknown')}")
    print(f"   • Drift: {result.get('drift_pct', 0):.1f}%")
    print(f"   • Drift Status: {result.get('drift_status', 'unknown')}")
    print(f"   • IR: {result.get('information_ratio', 0):.2f}")
    print(f"   • Data Quality: {result.get('data_quality', 'unknown')}")
    print(f"   • Days Analyzed: {result.get('days_analyzed', 0)}")
    print(f"   • Stored: {result.get('store_success', False)}")
    print(f"   • Alert: {result.get('alert_success', False)}")
    
    print("\n✅ Drift monitor flow test completed!")
    
    # Test deployment creation if Prefect is available
    if PREFECT_AVAILABLE:
        try:
            deployment_id = create_drift_monitor_deployment()
            if deployment_id:
                print(f"✅ Deployment created: {deployment_id}")
            else:
                print("⚠️  Deployment creation skipped")
        except Exception as e:
            print(f"⚠️  Deployment creation failed: {e}")
    else:
        print("ℹ️  Prefect not available, skipping deployment test")