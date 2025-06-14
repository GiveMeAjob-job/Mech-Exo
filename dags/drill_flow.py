"""
Rollback Drill Prefect Flow - Day 4 Module 3

Manual flow for executing rollback drills with metadata logging.
Supports quarterly drill scheduling with configurable intervals.
"""

import os
import subprocess
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.blocks.system import Secret

# Project imports
from mech_exo.datasource.storage import DataStorage

logger = logging.getLogger(__name__)


@task(name="launch-drill", description="Execute rollback drill script")
def launch_drill_task(dry_run: bool = True, wait_seconds: int = 120) -> Dict[str, Any]:
    """
    Launch the rollback drill script
    
    Args:
        dry_run: Whether to run in dry-run mode
        wait_seconds: Wait time between disable and restore
        
    Returns:
        Dictionary with drill execution results
    """
    task_logger = get_run_logger()
    task_logger.info(f"🚀 Launching rollback drill (dry_run={dry_run}, wait={wait_seconds}s)")
    
    start_time = datetime.now()
    
    try:
        # Build command
        script_path = Path("/Users/binwspacerace/PycharmProjects/Mech-Exo/scripts/rollback_drill.py")
        
        if not script_path.exists():
            raise FileNotFoundError(f"Drill script not found: {script_path}")
        
        cmd = ["python", str(script_path), "--wait", str(wait_seconds)]
        if dry_run:
            cmd.append("--dry-run")
        
        task_logger.info(f"🔧 Executing command: {' '.join(cmd)}")
        
        # Execute drill script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=wait_seconds + 300,  # Extra timeout buffer
            cwd=script_path.parent.parent
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Parse results
        drill_passed = result.returncode == 0
        
        # Try to find the generated report file
        report_pattern = f"drill_{start_time.strftime('%Y%m%d_%H%M')}*.md"
        project_root = Path("/Users/binwspacerace/PycharmProjects/Mech-Exo")
        report_files = list(project_root.glob(report_pattern))
        report_path = str(report_files[0]) if report_files else None
        
        execution_result = {
            'drill_passed': drill_passed,
            'execution_time_seconds': execution_time,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'report_path': report_path,
            'dry_run': dry_run,
            'wait_seconds': wait_seconds,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
        
        if drill_passed:
            task_logger.info(f"✅ Drill completed successfully in {execution_time:.1f}s")
        else:
            task_logger.error(f"❌ Drill failed after {execution_time:.1f}s")
            task_logger.error(f"Error output: {result.stderr}")
        
        return execution_result
        
    except subprocess.TimeoutExpired:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        task_logger.error(f"⏰ Drill timed out after {execution_time:.1f}s")
        return {
            'drill_passed': False,
            'execution_time_seconds': execution_time,
            'return_code': -1,
            'stdout': '',
            'stderr': 'Drill execution timed out',
            'report_path': None,
            'dry_run': dry_run,
            'wait_seconds': wait_seconds,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'error': 'timeout'
        }
        
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        task_logger.error(f"💥 Drill execution failed: {e}")
        return {
            'drill_passed': False,
            'execution_time_seconds': execution_time,
            'return_code': -1,
            'stdout': '',
            'stderr': str(e),
            'report_path': None,
            'dry_run': dry_run,
            'wait_seconds': wait_seconds,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'error': str(e)
        }


@task(name="store-drill-metadata", description="Log drill metadata to database")
def store_drill_meta_task(drill_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store drill execution metadata in database
    
    Args:
        drill_result: Results from launch_drill_task
        
    Returns:
        Dictionary with storage results
    """
    task_logger = get_run_logger()
    task_logger.info("📊 Storing drill metadata to database")
    
    try:
        # Create drill_log table if it doesn't exist
        storage = DataStorage()
        
        # Create table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS drill_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drill_date DATE NOT NULL,
            drill_timestamp TIMESTAMP NOT NULL,
            duration_seconds REAL NOT NULL,
            passed BOOLEAN NOT NULL,
            file_path TEXT,
            dry_run BOOLEAN NOT NULL DEFAULT FALSE,
            wait_seconds INTEGER DEFAULT 120,
            return_code INTEGER,
            error_message TEXT,
            stdout_snippet TEXT,
            stderr_snippet TEXT,
            flow_run_id TEXT,
            prefect_state TEXT DEFAULT 'completed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Indexes
            INDEX idx_drill_log_date (drill_date),
            INDEX idx_drill_log_passed (passed),
            INDEX idx_drill_log_dry_run (dry_run)
        );
        """
        
        storage.conn.execute(create_table_sql)
        storage.conn.commit()
        
        # Insert drill record
        drill_date = datetime.fromisoformat(drill_result['start_time']).date()
        drill_timestamp = datetime.fromisoformat(drill_result['start_time'])
        
        # Truncate output for storage (keep first 500 chars)
        stdout_snippet = drill_result.get('stdout', '')[:500] if drill_result.get('stdout') else None
        stderr_snippet = drill_result.get('stderr', '')[:500] if drill_result.get('stderr') else None
        
        insert_sql = """
        INSERT INTO drill_log (
            drill_date, drill_timestamp, duration_seconds, passed, file_path,
            dry_run, wait_seconds, return_code, error_message, 
            stdout_snippet, stderr_snippet, flow_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Get flow run ID if available
        flow_run_id = os.getenv('PREFECT_FLOW_RUN_ID', 'manual')
        
        storage.conn.execute(insert_sql, (
            drill_date,
            drill_timestamp,
            drill_result['execution_time_seconds'],
            drill_result['drill_passed'],
            drill_result.get('report_path'),
            drill_result['dry_run'],
            drill_result['wait_seconds'],
            drill_result['return_code'],
            drill_result.get('error'),
            stdout_snippet,
            stderr_snippet,
            flow_run_id
        ))
        
        storage.conn.commit()
        storage.close()
        
        task_logger.info(f"✅ Drill metadata stored successfully")
        
        return {
            'storage_success': True,
            'drill_date': drill_date.isoformat(),
            'record_id': storage.conn.lastrowid if hasattr(storage.conn, 'lastrowid') else None
        }
        
    except Exception as e:
        task_logger.error(f"❌ Failed to store drill metadata: {e}")
        return {
            'storage_success': False,
            'error': str(e)
        }


@task(name="check-drill-frequency", description="Check if drill is due based on interval")
def check_drill_frequency_task(interval_days: int = 90) -> Dict[str, Any]:
    """
    Check if a drill is due based on configured interval
    
    Args:
        interval_days: Days between required drills (default: 90 for quarterly)
        
    Returns:
        Dictionary with frequency check results
    """
    task_logger = get_run_logger()
    task_logger.info(f"📅 Checking drill frequency (interval: {interval_days} days)")
    
    try:
        storage = DataStorage()
        
        # Get last successful drill
        query_sql = """
        SELECT drill_date, passed, dry_run 
        FROM drill_log 
        WHERE passed = TRUE AND dry_run = FALSE
        ORDER BY drill_date DESC 
        LIMIT 1
        """
        
        result = storage.conn.execute(query_sql).fetchone()
        storage.close()
        
        if result:
            last_drill_date = date.fromisoformat(result[0])
            days_since_last = (date.today() - last_drill_date).days
            
            is_due = days_since_last >= interval_days
            is_overdue = days_since_last > interval_days + 7  # Grace period
            
            task_logger.info(f"📊 Last drill: {last_drill_date} ({days_since_last} days ago)")
            
            if is_overdue:
                task_logger.warning(f"⚠️ Drill overdue by {days_since_last - interval_days} days!")
            elif is_due:
                task_logger.info(f"⏰ Drill due (≥{interval_days} days)")
            else:
                days_until_due = interval_days - days_since_last
                task_logger.info(f"✅ Drill not due yet ({days_until_due} days remaining)")
            
            return {
                'frequency_check_success': True,
                'last_drill_date': last_drill_date.isoformat(),
                'days_since_last': days_since_last,
                'is_due': is_due,
                'is_overdue': is_overdue,
                'days_until_due': max(0, interval_days - days_since_last),
                'interval_days': interval_days
            }
        else:
            task_logger.warning("⚠️ No previous successful drills found")
            return {
                'frequency_check_success': True,
                'last_drill_date': None,
                'days_since_last': None,
                'is_due': True,  # Always due if no previous drill
                'is_overdue': True,
                'days_until_due': 0,
                'interval_days': interval_days,
                'first_drill': True
            }
            
    except Exception as e:
        task_logger.error(f"❌ Failed to check drill frequency: {e}")
        return {
            'frequency_check_success': False,
            'error': str(e)
        }


@flow(name="rollback-drill", description="Manual rollback drill execution with metadata logging")
def rollback_drill_flow(
    dry_run: bool = True,
    wait_seconds: int = 120,
    interval_days: int = 90,
    skip_frequency_check: bool = False
) -> Dict[str, Any]:
    """
    Execute rollback drill with comprehensive logging and tracking
    
    Args:
        dry_run: Whether to run in dry-run mode (default: True for safety)
        wait_seconds: Wait time between disable and restore (default: 120)
        interval_days: Days between required drills (default: 90 for quarterly)
        skip_frequency_check: Skip frequency check and force drill (default: False)
        
    Returns:
        Dictionary with complete flow results
    """
    flow_logger = get_run_logger()
    flow_logger.info(f"🚀 Starting rollback drill flow")
    flow_logger.info(f"   Parameters: dry_run={dry_run}, wait={wait_seconds}s, interval={interval_days}d")
    
    flow_start_time = datetime.now()
    
    try:
        # Step 1: Check drill frequency (unless skipped)
        frequency_result = None
        if not skip_frequency_check:
            frequency_result = check_drill_frequency_task(interval_days)
            
            if frequency_result.get('frequency_check_success'):
                if not frequency_result.get('is_due', True):
                    flow_logger.info("⏭️ Drill not due yet, skipping execution")
                    return {
                        'flow_success': True,
                        'drill_executed': False,
                        'skip_reason': 'not_due',
                        'frequency_result': frequency_result,
                        'next_due_date': (date.today() + timedelta(days=frequency_result.get('days_until_due', 0))).isoformat()
                    }
        
        # Step 2: Execute drill
        flow_logger.info("🔄 Executing rollback drill")
        drill_result = launch_drill_task(dry_run=dry_run, wait_seconds=wait_seconds)
        
        # Step 3: Store metadata
        flow_logger.info("📊 Storing drill metadata")
        storage_result = store_drill_meta_task(drill_result)
        
        # Determine overall success
        overall_success = (
            drill_result.get('drill_passed', False) and 
            storage_result.get('storage_success', False)
        )
        
        end_time = datetime.now()
        total_duration = (end_time - flow_start_time).total_seconds()
        
        if overall_success:
            flow_logger.info(f"✅ Rollback drill flow completed successfully in {total_duration:.1f}s")
        else:
            flow_logger.error(f"❌ Rollback drill flow failed after {total_duration:.1f}s")
        
        return {
            'flow_success': overall_success,
            'drill_executed': True,
            'total_duration_seconds': total_duration,
            'frequency_result': frequency_result,
            'drill_result': drill_result,
            'storage_result': storage_result,
            'flow_start_time': flow_start_time.isoformat(),
            'flow_end_time': end_time.isoformat()
        }
        
    except Exception as e:
        end_time = datetime.now()
        total_duration = (end_time - flow_start_time).total_seconds()
        
        flow_logger.error(f"💥 Rollback drill flow failed with exception: {e}")
        
        return {
            'flow_success': False,
            'drill_executed': False,
            'total_duration_seconds': total_duration,
            'error': str(e),
            'flow_start_time': flow_start_time.isoformat(),
            'flow_end_time': end_time.isoformat()
        }


# Utility functions for external access

def get_last_drill_info() -> Optional[Dict[str, Any]]:
    """
    Get information about the last drill execution
    
    Returns:
        Dictionary with last drill info or None if no drills found
    """
    try:
        storage = DataStorage()
        
        query_sql = """
        SELECT drill_date, passed, dry_run, duration_seconds, file_path
        FROM drill_log 
        ORDER BY drill_timestamp DESC 
        LIMIT 1
        """
        
        result = storage.conn.execute(query_sql).fetchone()
        storage.close()
        
        if result:
            drill_date, passed, dry_run, duration, file_path = result
            return {
                'drill_date': drill_date,
                'passed': bool(passed),
                'dry_run': bool(dry_run),
                'duration_seconds': duration,
                'file_path': file_path,
                'days_ago': (date.today() - date.fromisoformat(drill_date)).days
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get last drill info: {e}")
        return None


def get_drill_summary(days: int = 30) -> Dict[str, Any]:
    """
    Get drill summary statistics
    
    Args:
        days: Number of days to look back
        
    Returns:
        Dictionary with drill statistics
    """
    try:
        storage = DataStorage()
        
        since_date = (date.today() - timedelta(days=days)).isoformat()
        
        query_sql = """
        SELECT 
            COUNT(*) as total_drills,
            SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed_drills,
            SUM(CASE WHEN dry_run = 0 THEN 1 ELSE 0 END) as live_drills,
            AVG(duration_seconds) as avg_duration,
            MAX(drill_date) as last_drill_date
        FROM drill_log 
        WHERE drill_date >= ?
        """
        
        result = storage.conn.execute(query_sql, (since_date,)).fetchone()
        storage.close()
        
        if result:
            total, passed, live, avg_duration, last_date = result
            return {
                'total_drills': total or 0,
                'passed_drills': passed or 0,
                'live_drills': live or 0,
                'success_rate': (passed / total * 100) if total > 0 else 0,
                'avg_duration_seconds': avg_duration or 0,
                'last_drill_date': last_date,
                'days_period': days
            }
        
        return {
            'total_drills': 0,
            'passed_drills': 0,
            'live_drills': 0,
            'success_rate': 0,
            'avg_duration_seconds': 0,
            'last_drill_date': None,
            'days_period': days
        }
        
    except Exception as e:
        logger.error(f"Failed to get drill summary: {e}")
        return {
            'total_drills': 0,
            'passed_drills': 0,
            'live_drills': 0,
            'success_rate': 0,
            'avg_duration_seconds': 0,
            'last_drill_date': None,
            'days_period': days,
            'error': str(e)
        }


if __name__ == "__main__":
    # Manual execution example
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        print("🚨 WARNING: Running LIVE drill (not dry-run)")
        response = input("Are you sure? Type 'YES' to continue: ")
        if response != "YES":
            print("❌ Aborted")
            sys.exit(1)
        
        result = rollback_drill_flow(dry_run=False, wait_seconds=5)
    else:
        print("🧪 Running DRY-RUN drill")
        result = rollback_drill_flow(dry_run=True, wait_seconds=1)
    
    print(f"\n📊 Flow Result: {result}")