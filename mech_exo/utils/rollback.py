"""
Incident Rollback Tooling

Provides rollback capabilities for configuration files, database states,
and flow deployments with Git integration and safety checks.
"""

import logging
import os
import shutil
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class RollbackManager:
    """Manages rollback operations for incident recovery"""
    
    def __init__(self, repo_path: str = ".", dry_run: bool = False):
        """
        Initialize rollback manager
        
        Args:
            repo_path: Path to Git repository
            dry_run: If True, only simulate operations
        """
        self.repo_path = Path(repo_path)
        self.dry_run = dry_run
        self.rollback_log = []
        
        # Verify Git repository
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a Git repository: {repo_path}")
    
    def rollback_flow_deployment(self, flow_name: str, target_timestamp: str) -> Dict[str, Any]:
        """
        Rollback a flow deployment to a specific timestamp
        
        Args:
            flow_name: Name of flow to rollback (e.g., 'ml_reweight', 'canary_perf')
            target_timestamp: Target timestamp (ISO format: 2025-06-15T09:55:00)
            
        Returns:
            Rollback operation results
        """
        logger.info(f"Rolling back {flow_name} to {target_timestamp} (dry_run={self.dry_run})")
        
        try:
            # Parse target timestamp
            target_dt = datetime.fromisoformat(target_timestamp.replace('T', ' '))
            
            # Find relevant files for the flow
            flow_files = self._get_flow_files(flow_name)
            
            if not flow_files:
                return {
                    'success': False,
                    'error': f'No files found for flow: {flow_name}',
                    'available_flows': self._get_available_flows()
                }
            
            # Find Git commit closest to target timestamp
            target_commit = self._find_commit_by_timestamp(target_dt)
            
            if not target_commit:
                return {
                    'success': False,
                    'error': f'No commit found near timestamp: {target_timestamp}'
                }
            
            # Create rollback plan
            rollback_plan = self._create_rollback_plan(flow_files, target_commit)
            
            # Execute rollback if not dry run
            if not self.dry_run:
                success = self._execute_rollback_plan(rollback_plan, flow_name, target_commit)
                
                if success:
                    # Send notification
                    self._send_rollback_notification(flow_name, target_timestamp, target_commit)
                
                return {
                    'success': success,
                    'flow_name': flow_name,
                    'target_timestamp': target_timestamp,
                    'target_commit': target_commit,
                    'rollback_plan': rollback_plan,
                    'files_changed': len(rollback_plan['files']),
                    'notification_sent': success
                }
            else:
                return {
                    'success': True,
                    'dry_run': True,
                    'flow_name': flow_name,
                    'target_timestamp': target_timestamp,
                    'target_commit': target_commit,
                    'rollback_plan': rollback_plan,
                    'files_to_change': len(rollback_plan['files'])
                }
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def rollback_config_file(self, config_file: str, target_timestamp: str) -> Dict[str, Any]:
        """
        Rollback a specific configuration file
        
        Args:
            config_file: Path to config file (e.g., 'config/factors.yml')
            target_timestamp: Target timestamp
            
        Returns:
            Rollback operation results
        """
        logger.info(f"Rolling back config {config_file} to {target_timestamp}")
        
        try:
            target_dt = datetime.fromisoformat(target_timestamp.replace('T', ' '))
            config_path = self.repo_path / config_file
            
            if not config_path.exists():
                return {
                    'success': False,
                    'error': f'Config file not found: {config_file}'
                }
            
            # Find target commit
            target_commit = self._find_commit_by_timestamp(target_dt)
            
            if not target_commit:
                return {
                    'success': False,
                    'error': f'No commit found near timestamp: {target_timestamp}'
                }
            
            # Get file content at target commit
            old_content = self._get_file_at_commit(config_file, target_commit)
            
            if old_content is None:
                return {
                    'success': False,
                    'error': f'File not found in commit {target_commit}: {config_file}'
                }
            
            # Create backup of current file
            backup_path = self._create_backup(config_path)
            
            if not self.dry_run:
                # Write old content to file
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write(old_content)
                
                # Commit the rollback
                commit_msg = f"Rollback {config_file} to {target_timestamp} (commit: {target_commit[:8]})"
                self._commit_changes([config_file], commit_msg)
                
                # Send notification
                self._send_rollback_notification(f"config:{config_file}", target_timestamp, target_commit)
            
            return {
                'success': True,
                'dry_run': self.dry_run,
                'config_file': config_file,
                'target_commit': target_commit,
                'backup_path': str(backup_path) if backup_path else None,
                'changes_preview': self._diff_content(config_path.read_text(), old_content) if self.dry_run else None
            }
            
        except Exception as e:
            logger.error(f"Config rollback failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def rollback_database_state(self, target_timestamp: str, flags_only: bool = True) -> Dict[str, Any]:
        """
        Rollback database state (flags and configuration)
        
        Args:
            target_timestamp: Target timestamp
            flags_only: If True, only rollback configuration flags
            
        Returns:
            Rollback operation results
        """
        logger.info(f"Rolling back database state to {target_timestamp} (flags_only={flags_only})")
        
        try:
            from ..datasource.storage import DataStorage
            from ..execution.allocation import update_canary_enabled, reset_breach_counter
            
            target_dt = datetime.fromisoformat(target_timestamp.replace('T', ' '))
            
            if flags_only:
                # Only reset configuration flags, not historical data
                changes = []
                
                if not self.dry_run:
                    # Reset canary to enabled state
                    success = update_canary_enabled(True)
                    if success:
                        changes.append("canary_enabled: true")
                    
                    # Reset breach counter
                    reset_breach_counter()
                    changes.append("consecutive_breach_days: 0")
                    
                    # Log the rollback in database
                    storage = DataStorage()
                    try:
                        rollback_sql = """
                            INSERT INTO system_events (timestamp, event_type, description, data)
                            VALUES (?, 'rollback', ?, ?)
                        """
                        
                        event_data = {
                            'target_timestamp': target_timestamp,
                            'flags_reset': changes,
                            'rollback_type': 'database_flags'
                        }
                        
                        storage.conn.execute(rollback_sql, [
                            datetime.now().isoformat(),
                            f"Database flags rollback to {target_timestamp}",
                            str(event_data)
                        ])
                        storage.conn.commit()
                        
                    finally:
                        storage.close()
                
                return {
                    'success': True,
                    'dry_run': self.dry_run,
                    'rollback_type': 'database_flags',
                    'target_timestamp': target_timestamp,
                    'changes': changes if not self.dry_run else ['canary_enabled: true', 'consecutive_breach_days: 0']
                }
            else:
                return {
                    'success': False,
                    'error': 'Full database rollback not implemented for safety reasons'
                }
                
        except Exception as e:
            logger.error(f"Database rollback failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_flow_files(self, flow_name: str) -> List[str]:
        """Get files associated with a flow"""
        flow_mappings = {
            'ml_reweight': [
                'config/ml_weights.yml',
                'dags/ml_weight_flow.py',
                'mech_exo/scoring/ml_reweight.py'
            ],
            'canary_perf': [
                'config/allocation.yml',
                'dags/canary_perf_flow.py',
                'mech_exo/execution/allocation.py'
            ],
            'daily_flow': [
                'dags/daily_flow.py',
                'config/factors.yml'
            ],
            'data_pipeline': [
                'dags/data_pipeline.py',
                'config/api_keys.yml'
            ]
        }
        
        return flow_mappings.get(flow_name, [])
    
    def _get_available_flows(self) -> List[str]:
        """Get list of available flows for rollback"""
        return [
            'ml_reweight',
            'canary_perf', 
            'daily_flow',
            'data_pipeline'
        ]
    
    def _find_commit_by_timestamp(self, target_dt: datetime) -> Optional[str]:
        """Find Git commit closest to target timestamp"""
        try:
            # Get commits within 24 hours of target
            since_dt = target_dt - timedelta(hours=24)
            until_dt = target_dt + timedelta(hours=24)
            
            cmd = [
                'git', 'log',
                '--format=%H %ci',
                f'--since={since_dt.isoformat()}',
                f'--until={until_dt.isoformat()}',
                '--max-count=50'
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            if not result.stdout.strip():
                # If no commits in range, get the most recent before target
                cmd = [
                    'git', 'log',
                    '--format=%H %ci',
                    f'--until={target_dt.isoformat()}',
                    '--max-count=1'
                ]
                
                result = subprocess.run(
                    cmd,
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
            
            lines = result.stdout.strip().split('\n')
            if not lines or not lines[0]:
                return None
            
            # Parse commits and find closest
            commits = []
            for line in lines:
                if line.strip():
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        commit_hash = parts[0]
                        commit_date_str = parts[1]
                        commit_dt = datetime.fromisoformat(commit_date_str.replace(' +', '+').replace(' -', '-')[:-3])
                        commits.append((commit_hash, commit_dt))
            
            if not commits:
                return None
            
            # Find closest commit
            closest_commit = min(commits, key=lambda x: abs((x[1] - target_dt).total_seconds()))
            return closest_commit[0]
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to find commit by timestamp: {e}")
            return None
    
    def _create_rollback_plan(self, files: List[str], target_commit: str) -> Dict[str, Any]:
        """Create a rollback execution plan"""
        plan = {
            'target_commit': target_commit,
            'files': [],
            'timestamp': datetime.now().isoformat()
        }
        
        for file_path in files:
            file_info = {
                'path': file_path,
                'exists_current': (self.repo_path / file_path).exists(),
                'exists_target': self._file_exists_at_commit(file_path, target_commit),
                'action': 'unknown'
            }
            
            if file_info['exists_target']:
                if file_info['exists_current']:
                    file_info['action'] = 'revert'
                else:
                    file_info['action'] = 'restore'
            else:
                if file_info['exists_current']:
                    file_info['action'] = 'delete'
                else:
                    file_info['action'] = 'skip'
            
            plan['files'].append(file_info)
        
        return plan
    
    def _execute_rollback_plan(self, plan: Dict[str, Any], flow_name: str, target_commit: str) -> bool:
        """Execute the rollback plan"""
        try:
            changed_files = []
            
            for file_info in plan['files']:
                file_path = file_info['path']
                action = file_info['action']
                
                if action == 'revert' or action == 'restore':
                    # Get file content from target commit
                    old_content = self._get_file_at_commit(file_path, target_commit)
                    
                    if old_content is not None:
                        # Create backup
                        full_path = self.repo_path / file_path
                        if full_path.exists():
                            self._create_backup(full_path)
                        
                        # Ensure directory exists
                        full_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Write old content
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(old_content)
                        
                        changed_files.append(file_path)
                        logger.info(f"Reverted {file_path} to commit {target_commit[:8]}")
                
                elif action == 'delete':
                    # Remove file that didn't exist in target commit
                    full_path = self.repo_path / file_path
                    if full_path.exists():
                        self._create_backup(full_path)
                        full_path.unlink()
                        changed_files.append(file_path)
                        logger.info(f"Deleted {file_path} (not in target commit)")
            
            # Commit changes
            if changed_files:
                commit_msg = f"Rollback {flow_name} to {target_commit[:8]} - Emergency recovery"
                self._commit_changes(changed_files, commit_msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute rollback plan: {e}")
            return False
    
    def _get_file_at_commit(self, file_path: str, commit_hash: str) -> Optional[str]:
        """Get file content at specific commit"""
        try:
            cmd = ['git', 'show', f'{commit_hash}:{file_path}']
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            return result.stdout
            
        except subprocess.CalledProcessError:
            # File might not exist at that commit
            return None
        except Exception as e:
            logger.error(f"Failed to get file content: {e}")
            return None
    
    def _file_exists_at_commit(self, file_path: str, commit_hash: str) -> bool:
        """Check if file exists at specific commit"""
        try:
            cmd = ['git', 'cat-file', '-e', f'{commit_hash}:{file_path}']
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                check=False
            )
            
            return result.returncode == 0
            
        except Exception:
            return False
    
    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """Create backup of current file"""
        try:
            if not file_path.exists():
                return None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{file_path.name}.backup.{timestamp}"
            backup_path = file_path.parent / backup_name
            
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def _commit_changes(self, files: List[str], commit_message: str) -> bool:
        """Commit changes to Git"""
        try:
            # Add files
            for file_path in files:
                cmd = ['git', 'add', file_path]
                subprocess.run(cmd, cwd=self.repo_path, check=True)
            
            # Commit
            cmd = ['git', 'commit', '-m', commit_message]
            subprocess.run(cmd, cwd=self.repo_path, check=True)
            
            logger.info(f"Committed rollback: {commit_message}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit changes: {e}")
            return False
    
    def _send_rollback_notification(self, component: str, timestamp: str, commit: str):
        """Send Telegram notification about rollback"""
        try:
            from ..utils.alerts import AlertManager, Alert, AlertType, AlertLevel
            
            alert_manager = AlertManager()
            
            alert = Alert(
                alert_type=AlertType.SYSTEM_INFO,
                level=AlertLevel.CRITICAL,
                title=f"🔄 Emergency Rollback Executed",
                message=f"Component: {component}\n"
                       f"Target time: {timestamp}\n"
                       f"Restored to: {commit[:8]}\n"
                       f"Executed by: rollback tool\n\n"
                       f"Please verify system status and functionality.",
                timestamp=datetime.now(),
                data={
                    'component': component,
                    'target_timestamp': timestamp,
                    'target_commit': commit,
                    'rollback_tool': True
                }
            )
            
            success = alert_manager.send_alert(alert, channels=['telegram'])
            logger.info(f"Rollback notification sent: {success}")
            
        except Exception as e:
            logger.error(f"Failed to send rollback notification: {e}")
    
    def _diff_content(self, current: str, target: str) -> Dict[str, Any]:
        """Generate diff preview"""
        import difflib
        
        diff = list(difflib.unified_diff(
            current.splitlines(keepends=True),
            target.splitlines(keepends=True),
            fromfile='current',
            tofile='target',
            n=3
        ))
        
        return {
            'has_changes': len(diff) > 0,
            'diff_lines': len(diff),
            'preview': ''.join(diff[:20])  # First 20 lines of diff
        }
    
    def get_rollback_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent rollback operations"""
        try:
            # Look for rollback commits in Git history
            since_dt = datetime.now() - timedelta(days=days)
            
            cmd = [
                'git', 'log',
                '--format=%H|%ci|%s',
                f'--since={since_dt.isoformat()}',
                '--grep=Rollback',
                '--grep=rollback',
                '--grep=Emergency recovery',
                '-i'  # case insensitive
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            rollbacks = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split('|', 2)
                    if len(parts) == 3:
                        commit_hash, commit_date, commit_msg = parts
                        rollbacks.append({
                            'commit': commit_hash,
                            'timestamp': commit_date,
                            'message': commit_msg,
                            'type': 'git_rollback'
                        })
            
            return rollbacks
            
        except Exception as e:
            logger.error(f"Failed to get rollback history: {e}")
            return []


def confirm_rollback(operation_desc: str) -> bool:
    """Get user confirmation for rollback operation"""
    print(f"\n⚠️  ROLLBACK CONFIRMATION REQUIRED")
    print(f"Operation: {operation_desc}")
    print(f"This action cannot be easily undone.")
    
    response = input("\nType 'CONFIRM' to proceed: ").strip()
    return response == 'CONFIRM'


def validate_timestamp(timestamp_str: str) -> bool:
    """Validate timestamp format"""
    try:
        datetime.fromisoformat(timestamp_str.replace('T', ' '))
        return True
    except ValueError:
        return False