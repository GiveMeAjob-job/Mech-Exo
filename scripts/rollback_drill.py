#!/usr/bin/env python3
"""
Rollback Drill Script - Day 4 Module 1

Automated kill-switch rollback drill with comprehensive reporting.
Tests the complete kill-switch cycle: backup → disable → wait → restore → verify.
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
import fcntl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DrillLockManager:
    """Manages file locking to prevent concurrent drills"""
    
    def __init__(self, lock_file: str = ".drill_lock"):
        self.lock_file = Path(lock_file)
        self.lock_fd = None
    
    def __enter__(self):
        """Acquire lock"""
        try:
            self.lock_fd = open(self.lock_file, 'w')
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_fd.write(f"Drill started: {datetime.now().isoformat()}\n")
            self.lock_fd.flush()
            logger.info(f"✅ Acquired drill lock: {self.lock_file}")
            return self
        except (IOError, OSError) as e:
            if self.lock_fd:
                self.lock_fd.close()
            raise RuntimeError(f"Another drill is already running (lock: {self.lock_file})") from e
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock"""
        if self.lock_fd:
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
            self.lock_fd.close()
            self.lock_fd = None
            
            # Clean up lock file
            try:
                self.lock_file.unlink()
                logger.info(f"✅ Released drill lock: {self.lock_file}")
            except FileNotFoundError:
                pass


class RollbackDrill:
    """Main rollback drill orchestrator"""
    
    def __init__(self, dry_run: bool = False, wait_seconds: int = 120):
        self.dry_run = dry_run
        self.wait_seconds = wait_seconds
        self.start_time = datetime.now()
        self.steps_log: List[Dict[str, Any]] = []
        self.drill_passed = False
        
        # Paths
        self.project_root = Path("/Users/binwspacerace/PycharmProjects/Mech-Exo")
        self.killswitch_config = self.project_root / "config" / "killswitch.yml"
        self.backup_config = None
        self.report_path = None
        
        # Initialize report
        timestamp = self.start_time.strftime("%Y%m%d_%H%M")
        self.report_path = self.project_root / f"drill_{timestamp}.md"
        
        logger.info(f"🚀 Rollback Drill initialized")
        logger.info(f"   Dry run: {self.dry_run}")
        logger.info(f"   Wait time: {self.wait_seconds}s")
        logger.info(f"   Report: {self.report_path}")
    
    def log_step(self, step_id: str, description: str, status: str, 
                 details: Optional[str] = None, error: Optional[str] = None):
        """Log a drill step with timestamp"""
        step_time = datetime.now()
        step_entry = {
            'step_id': step_id,
            'description': description,
            'status': status,  # 'start', 'success', 'fail', 'skip'
            'timestamp': step_time,
            'elapsed_ms': int((step_time - self.start_time).total_seconds() * 1000),
            'details': details,
            'error': error
        }
        
        self.steps_log.append(step_entry)
        
        status_emoji = {
            'start': '🔄',
            'success': '✅',
            'fail': '❌',
            'skip': '⏭️'
        }.get(status, '❓')
        
        logger.info(f"{status_emoji} Step {step_id}: {description} [{status.upper()}]")
        if details:
            logger.info(f"   Details: {details}")
        if error:
            logger.error(f"   Error: {error}")
    
    def run_command(self, command: List[str], description: str) -> Dict[str, Any]:
        """Run shell command and capture result"""
        try:
            if self.dry_run:
                logger.info(f"🧪 DRY-RUN: Would execute: {' '.join(command)}")
                return {
                    'returncode': 0,
                    'stdout': f"DRY-RUN: {description}",
                    'stderr': '',
                    'success': True
                }
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.project_root
            )
            
            return {
                'returncode': result.returncode,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'success': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': 'Command timeout',
                'success': False
            }
        except Exception as e:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }
    
    def step_a_backup_config(self) -> bool:
        """Step A: Save current killswitch.yml as backup"""
        self.log_step('A', 'Backup killswitch configuration', 'start')
        
        try:
            if not self.killswitch_config.exists():
                self.log_step('A', 'Backup killswitch configuration', 'fail', 
                             error=f"Killswitch config not found: {self.killswitch_config}")
                return False
            
            # Create backup with timestamp
            backup_timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            self.backup_config = self.killswitch_config.parent / f"killswitch_backup_{backup_timestamp}.yml"
            
            if self.dry_run:
                self.log_step('A', 'Backup killswitch configuration', 'success',
                             details=f"DRY-RUN: Would backup to {self.backup_config}")
                return True
            
            # Create actual backup
            shutil.copy2(self.killswitch_config, self.backup_config)
            
            # Verify backup
            if self.backup_config.exists():
                backup_size = self.backup_config.stat().st_size
                original_size = self.killswitch_config.stat().st_size
                
                if backup_size == original_size:
                    self.log_step('A', 'Backup killswitch configuration', 'success',
                                 details=f"Backup created: {self.backup_config} ({backup_size} bytes)")
                    return True
                else:
                    self.log_step('A', 'Backup killswitch configuration', 'fail',
                                 error=f"Backup size mismatch: {backup_size} vs {original_size}")
                    return False
            else:
                self.log_step('A', 'Backup killswitch configuration', 'fail',
                             error="Backup file was not created")
                return False
                
        except Exception as e:
            self.log_step('A', 'Backup killswitch configuration', 'fail', error=str(e))
            return False
    
    def step_b_disable_trading(self) -> bool:
        """Step B: Run 'exo kill off --reason DRILL'"""
        self.log_step('B', 'Disable trading via kill-switch', 'start')
        
        # First check current state
        status_result = self.run_command(['python', '-m', 'mech_exo.cli', 'kill', 'status'], 
                                       'Check kill-switch status')
        
        if not status_result['success']:
            self.log_step('B', 'Disable trading via kill-switch', 'fail',
                         error=f"Failed to check kill-switch status: {status_result['stderr']}")
            return False
        
        # Check if already disabled
        try:
            with open(self.killswitch_config) as f:
                config = yaml.safe_load(f)
                trading_enabled = config.get('trading_enabled', True)
                
            if not trading_enabled:
                self.log_step('B', 'Disable trading via kill-switch', 'success',
                             details="Trading already disabled - no action needed")
                return True
                
        except Exception as e:
            logger.warning(f"Could not read killswitch config: {e}")
        
        # Disable trading
        disable_result = self.run_command(
            ['python', '-m', 'mech_exo.cli', 'kill', 'off', '--reason', 'ROLLBACK_DRILL'],
            'Disable trading'
        )
        
        if disable_result['success']:
            self.log_step('B', 'Disable trading via kill-switch', 'success',
                         details=f"Trading disabled: {disable_result['stdout']}")
            return True
        else:
            self.log_step('B', 'Disable trading via kill-switch', 'fail',
                         error=f"Failed to disable trading: {disable_result['stderr']}")
            return False
    
    def step_c_wait_period(self) -> bool:
        """Step C: Wait n seconds"""
        self.log_step('C', f'Wait period ({self.wait_seconds}s)', 'start')
        
        try:
            if self.dry_run:
                logger.info(f"🧪 DRY-RUN: Would wait {self.wait_seconds} seconds")
                time.sleep(0.1)  # Minimal wait for dry-run
            else:
                logger.info(f"⏳ Waiting {self.wait_seconds} seconds...")
                
                # Wait with progress indicators
                for i in range(self.wait_seconds):
                    time.sleep(1)
                    if (i + 1) % 30 == 0:  # Log every 30 seconds
                        remaining = self.wait_seconds - (i + 1)
                        logger.info(f"⏳ Wait progress: {i + 1}/{self.wait_seconds}s ({remaining}s remaining)")
            
            self.log_step('C', f'Wait period ({self.wait_seconds}s)', 'success',
                         details=f"Completed {self.wait_seconds} second wait period")
            return True
            
        except KeyboardInterrupt:
            self.log_step('C', f'Wait period ({self.wait_seconds}s)', 'fail',
                         error="Wait period interrupted by user")
            return False
        except Exception as e:
            self.log_step('C', f'Wait period ({self.wait_seconds}s)', 'fail', error=str(e))
            return False
    
    def step_d_restore_trading(self) -> bool:
        """Step D: Run 'exo kill on --reason DRILL COMPLETE' and verify"""
        self.log_step('D', 'Restore trading via kill-switch', 'start')
        
        # Enable trading
        enable_result = self.run_command(
            ['python', '-m', 'mech_exo.cli', 'kill', 'on', '--reason', 'ROLLBACK_DRILL_COMPLETE'],
            'Enable trading'
        )
        
        if not enable_result['success']:
            self.log_step('D', 'Restore trading via kill-switch', 'fail',
                         error=f"Failed to enable trading: {enable_result['stderr']}")
            return False
        
        # Verify trading is enabled
        time.sleep(1)  # Brief pause for config to update
        
        try:
            if self.dry_run:
                self.log_step('D', 'Restore trading via kill-switch', 'success',
                             details="DRY-RUN: Would verify trading_enabled=true")
                return True
            
            with open(self.killswitch_config) as f:
                config = yaml.safe_load(f)
                trading_enabled = config.get('trading_enabled', False)
            
            if trading_enabled:
                self.log_step('D', 'Restore trading via kill-switch', 'success',
                             details=f"Trading restored: {enable_result['stdout']}")
                return True
            else:
                self.log_step('D', 'Restore trading via kill-switch', 'fail',
                             error="Verification failed: trading_enabled still false")
                return False
                
        except Exception as e:
            self.log_step('D', 'Restore trading via kill-switch', 'fail',
                         error=f"Verification failed: {e}")
            return False
    
    def emergency_restore(self):
        """Emergency restore from backup if drill fails"""
        logger.error("🚨 Emergency restore triggered")
        
        if not self.backup_config or not self.backup_config.exists():
            logger.error("❌ No backup available for emergency restore")
            return False
        
        try:
            if not self.dry_run:
                shutil.copy2(self.backup_config, self.killswitch_config)
                logger.info(f"✅ Emergency restore completed from {self.backup_config}")
            else:
                logger.info(f"🧪 DRY-RUN: Would restore from {self.backup_config}")
            
            self.log_step('EMERGENCY', 'Emergency config restore', 'success',
                         details=f"Restored from backup: {self.backup_config}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Emergency restore failed: {e}")
            self.log_step('EMERGENCY', 'Emergency config restore', 'fail', error=str(e))
            return False
    
    def generate_report(self) -> str:
        """Generate Markdown report"""
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        
        # Determine overall result
        failed_steps = [s for s in self.steps_log if s['status'] == 'fail']
        self.drill_passed = len(failed_steps) == 0
        
        result_emoji = "✅" if self.drill_passed else "❌"
        result_text = "PASS" if self.drill_passed else "FAIL"
        
        # Build Markdown report
        report = f"""# Rollback Drill Report
        
## Summary
**Result**: {result_emoji} **{result_text}**  
**Start Time**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}  
**End Time**: {end_time.strftime('%Y-%m-%d %H:%M:%S')}  
**Duration**: {total_duration.total_seconds():.1f} seconds  
**Mode**: {'DRY-RUN' if self.dry_run else 'LIVE'}  
**Wait Period**: {self.wait_seconds} seconds  

## Test Configuration
- **Killswitch Config**: `{self.killswitch_config}`
- **Backup Created**: `{self.backup_config or 'N/A'}`
- **Report Path**: `{self.report_path}`

## Step-by-Step Results

| Step | Description | Status | Time | Details |
|------|-------------|--------|------|---------|
"""
        
        for step in self.steps_log:
            step_emoji = {
                'start': '🔄',
                'success': '✅', 
                'fail': '❌',
                'skip': '⏭️'
            }.get(step['status'], '❓')
            
            elapsed_str = f"{step['elapsed_ms']}ms"
            details = step.get('details', '') or step.get('error', '')
            
            report += f"| {step['step_id']} | {step['description']} | {step_emoji} {step['status'].upper()} | {elapsed_str} | {details} |\n"
        
        # Add failure analysis if any
        if failed_steps:
            report += f"\n## ❌ Failure Analysis\n\n"
            for step in failed_steps:
                report += f"**Step {step['step_id']}**: {step['description']}  \n"
                report += f"**Error**: {step.get('error', 'Unknown error')}  \n\n"
        
        # Add recommendations
        report += f"\n## Recommendations\n\n"
        if self.drill_passed:
            report += "- ✅ Kill-switch system functioning correctly\n"
            report += "- ✅ Backup and restore procedures verified\n"
            report += "- ✅ No action required\n"
        else:
            report += "- ❌ Kill-switch system requires attention\n"
            report += "- 🔧 Review failed steps and fix underlying issues\n"
            report += "- 🔄 Re-run drill after fixes are implemented\n"
        
        report += f"\n## Dashboard Screenshot\n\n"
        report += f"![dashboard](TODO_SCREENSHOT.png)\n"
        report += f"*Note: Operations team should replace with actual dashboard screenshot*\n\n"
        
        report += f"## Next Drill\n\n"
        next_drill = self.start_time + timedelta(days=90)
        report += f"**Recommended Next Drill**: {next_drill.strftime('%Y-%m-%d')} (90 days)\n\n"
        
        report += f"---\n"
        report += f"*Generated by Mech-Exo Rollback Drill System*  \n"
        report += f"*Report ID*: `drill_{self.start_time.strftime('%Y%m%d_%H%M')}`\n"
        
        return report
    
    def save_report(self, report_content: str):
        """Save report to file"""
        try:
            with open(self.report_path, 'w') as f:
                f.write(report_content)
            
            file_size = self.report_path.stat().st_size
            logger.info(f"📄 Report saved: {self.report_path} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
    
    def send_telegram_alert(self):
        """Send Telegram alert about drill completion"""
        try:
            from mech_exo.utils.alerts import send_drill_report
            
            result_emoji = "✅" if self.drill_passed else "⚠️"
            success = send_drill_report(str(self.report_path), self.drill_passed)
            
            if success:
                logger.info(f"{result_emoji} Telegram alert sent successfully")
            else:
                logger.error("❌ Failed to send Telegram alert")
                
        except ImportError:
            logger.warning("⚠️ Telegram alerts not available (alerts module not found)")
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram alert: {e}")
    
    def cleanup(self):
        """Clean up backup files if drill passed"""
        if self.drill_passed and self.backup_config and self.backup_config.exists() and not self.dry_run:
            try:
                self.backup_config.unlink()
                logger.info(f"🧹 Cleaned up backup file: {self.backup_config}")
            except Exception as e:
                logger.warning(f"⚠️ Could not clean up backup: {e}")
    
    def run(self) -> bool:
        """Execute the complete rollback drill"""
        logger.info("🚀 Starting Rollback Drill")
        
        try:
            # Execute all steps
            success_a = self.step_a_backup_config()
            if not success_a:
                return False
            
            success_b = self.step_b_disable_trading()
            if not success_b:
                self.emergency_restore()
                return False
            
            success_c = self.step_c_wait_period()
            if not success_c:
                self.emergency_restore()
                return False
            
            success_d = self.step_d_restore_trading()
            if not success_d:
                self.emergency_restore()
                return False
            
            # All steps passed
            self.drill_passed = True
            logger.info("✅ Rollback drill completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Drill failed with exception: {e}")
            self.log_step('EXCEPTION', 'Unexpected error', 'fail', error=str(e))
            self.emergency_restore()
            return False
        
        finally:
            # Always generate report
            report_content = self.generate_report()
            self.save_report(report_content)
            
            # Send alerts if not dry-run
            if not self.dry_run:
                self.send_telegram_alert()
            
            # Clean up if successful
            self.cleanup()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Rollback Drill Script - Test kill-switch backup/restore cycle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate drill without making changes'
    )
    
    parser.add_argument(
        '--wait',
        type=int,
        default=120,
        help='Wait time in seconds between disable and restore'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate wait time
    if args.wait < 1 or args.wait > 3600:
        logger.error(f"❌ Invalid wait time: {args.wait}s (must be 1-3600)")
        sys.exit(1)
    
    try:
        # Use file lock to prevent concurrent drills
        with DrillLockManager():
            drill = RollbackDrill(dry_run=args.dry_run, wait_seconds=args.wait)
            success = drill.run()
            
            if success:
                logger.info("🎉 Drill completed successfully")
                sys.exit(0)
            else:
                logger.error("💥 Drill failed")
                sys.exit(1)
                
    except RuntimeError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.error("❌ Drill interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()