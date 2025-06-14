"""
Kill-Switch CLI Module

Provides command-line interface for emergency trading halt functionality.
Allows operators to quickly disable trading system-wide with proper logging.
"""

import argparse
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys

from ..utils.alerts import AlertManager, Alert, AlertType, AlertLevel

logger = logging.getLogger(__name__)


class KillSwitchManager:
    """Manages trading kill-switch state and operations"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize kill-switch manager
        
        Args:
            config_path: Path to killswitch.yml file
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path(__file__).parent.parent.parent / "config" / "killswitch.yml"
        
        self.alert_manager = None
        try:
            self.alert_manager = AlertManager()
        except Exception as e:
            logger.warning(f"Alert manager not available: {e}")
        
        # Ensure config file exists
        self._ensure_config_exists()
    
    def _ensure_config_exists(self):
        """Ensure kill-switch config file exists with defaults"""
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            default_config = {
                'trading_enabled': True,
                'reason': 'System operational',
                'timestamp': datetime.now().isoformat(),
                'last_modified_by': 'system',
                'auto_triggered': False,
                'trigger_source': None,
                'history': [{
                    'timestamp': datetime.now().isoformat(),
                    'action': 'enable',
                    'reason': 'System initialization',
                    'triggered_by': 'system',
                    'auto_triggered': False
                }]
            }
            self._write_config(default_config)
            logger.info(f"Created default kill-switch config: {self.config_path}")
    
    def _read_config(self) -> Dict[str, Any]:
        """Read kill-switch configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to read kill-switch config: {e}")
            return {
                'trading_enabled': False,  # Fail-safe to disabled
                'reason': f'Config read error: {e}',
                'timestamp': datetime.now().isoformat(),
                'last_modified_by': 'system',
                'auto_triggered': False,
                'trigger_source': 'error',
                'history': []
            }
    
    def _write_config(self, config: Dict[str, Any]):
        """Write kill-switch configuration with backup"""
        try:
            # Create backup
            if self.config_path.exists():
                backup_path = self.config_path.with_suffix('.yml.backup')
                self.config_path.rename(backup_path)
            
            # Write new config
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Kill-switch config updated: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to write kill-switch config: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current kill-switch status"""
        config = self._read_config()
        return {
            'trading_enabled': config.get('trading_enabled', False),
            'reason': config.get('reason', 'Unknown'),
            'timestamp': config.get('timestamp', datetime.now().isoformat()),
            'last_modified_by': config.get('last_modified_by', 'unknown'),
            'auto_triggered': config.get('auto_triggered', False),
            'trigger_source': config.get('trigger_source'),
            'config_path': str(self.config_path),
            'last_checked': datetime.now().isoformat()
        }
    
    def enable_trading(self, reason: str = "Manual enable", 
                      triggered_by: str = "operator", 
                      dry_run: bool = False) -> Dict[str, Any]:
        """
        Enable trading system-wide
        
        Args:
            reason: Reason for enabling trading
            triggered_by: Who triggered this action
            dry_run: If True, don't actually modify config
            
        Returns:
            Result dictionary with status
        """
        if dry_run:
            logger.info(f"[DRY RUN] Would enable trading: {reason}")
            return {
                'success': True,
                'action': 'enable',
                'reason': reason,
                'dry_run': True,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            config = self._read_config()
            timestamp = datetime.now().isoformat()
            
            # Update main config
            config.update({
                'trading_enabled': True,
                'reason': reason,
                'timestamp': timestamp,
                'last_modified_by': triggered_by,
                'auto_triggered': triggered_by != 'operator',
                'trigger_source': 'manual' if triggered_by == 'operator' else triggered_by
            })
            
            # Add to history
            history_entry = {
                'timestamp': timestamp,
                'action': 'enable',
                'reason': reason,
                'triggered_by': triggered_by,
                'auto_triggered': triggered_by != 'operator'
            }
            
            if 'history' not in config:
                config['history'] = []
            config['history'].append(history_entry)
            
            # Keep only last 10 history entries
            config['history'] = config['history'][-10:]
            
            self._write_config(config)
            
            # Send alert
            if self.alert_manager:
                self._send_alert('enable', reason, triggered_by)
            
            logger.info(f"✅ Trading ENABLED by {triggered_by}: {reason}")
            
            return {
                'success': True,
                'action': 'enable',
                'reason': reason,
                'triggered_by': triggered_by,
                'timestamp': timestamp,
                'trading_enabled': True
            }
            
        except Exception as e:
            logger.error(f"Failed to enable trading: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'enable',
                'timestamp': datetime.now().isoformat()
            }
    
    def disable_trading(self, reason: str = "Manual disable", 
                       triggered_by: str = "operator", 
                       dry_run: bool = False) -> Dict[str, Any]:
        """
        Disable trading system-wide (kill-switch ON)
        
        Args:
            reason: Reason for disabling trading
            triggered_by: Who triggered this action
            dry_run: If True, don't actually modify config
            
        Returns:
            Result dictionary with status
        """
        if dry_run:
            logger.info(f"[DRY RUN] Would disable trading: {reason}")
            return {
                'success': True,
                'action': 'disable',
                'reason': reason,
                'dry_run': True,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            config = self._read_config()
            timestamp = datetime.now().isoformat()
            
            # Update main config
            config.update({
                'trading_enabled': False,
                'reason': reason,
                'timestamp': timestamp,
                'last_modified_by': triggered_by,
                'auto_triggered': triggered_by != 'operator',
                'trigger_source': 'manual' if triggered_by == 'operator' else triggered_by
            })
            
            # Add to history
            history_entry = {
                'timestamp': timestamp,
                'action': 'disable',
                'reason': reason,
                'triggered_by': triggered_by,
                'auto_triggered': triggered_by != 'operator'
            }
            
            if 'history' not in config:
                config['history'] = []
            config['history'].append(history_entry)
            
            # Keep only last 10 history entries
            config['history'] = config['history'][-10:]
            
            self._write_config(config)
            
            # Send alert
            if self.alert_manager:
                self._send_alert('disable', reason, triggered_by)
            
            logger.warning(f"🚨 Trading DISABLED by {triggered_by}: {reason}")
            
            return {
                'success': True,
                'action': 'disable',
                'reason': reason,
                'triggered_by': triggered_by,
                'timestamp': timestamp,
                'trading_enabled': False
            }
            
        except Exception as e:
            logger.error(f"Failed to disable trading: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'disable',
                'timestamp': datetime.now().isoformat()
            }
    
    def _send_alert(self, action: str, reason: str, triggered_by: str):
        """Send alert for kill-switch action"""
        if not self.alert_manager:
            return
        
        try:
            emoji = "🚨" if action == "disable" else "✅"
            action_text = "DISABLED" if action == "disable" else "ENABLED"
            level = AlertLevel.CRITICAL if action == "disable" else AlertLevel.INFO
            
            message = f"""🔄 **TRADING KILL-SWITCH UPDATE**

**Status:** {emoji} Trading {action_text}
**Reason:** {reason}
**Triggered By:** {triggered_by}
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'⚠️ ALL NEW ORDERS WILL BE BLOCKED' if action == 'disable' else '✅ Trading operations resumed'}"""
            
            alert = Alert(
                alert_type=AlertType.SYSTEM_ALERT,
                level=level,
                title=f"Kill-Switch {action_text}",
                message=message,
                timestamp=datetime.now(),
                data={
                    'action': action,
                    'reason': reason,
                    'triggered_by': triggered_by,
                    'trading_enabled': action == 'enable'
                }
            )
            
            self.alert_manager.send_alert_with_escalation(
                alert,
                channels=['telegram'],
                respect_quiet_hours=False,
                force_send=True
            )
            
        except Exception as e:
            logger.error(f"Failed to send kill-switch alert: {e}")
    
    def is_trading_enabled(self) -> bool:
        """Check if trading is currently enabled"""
        try:
            config = self._read_config()
            return config.get('trading_enabled', False)
        except Exception as e:
            logger.error(f"Failed to check trading status: {e}")
            return False  # Fail-safe to disabled
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get kill-switch action history"""
        try:
            config = self._read_config()
            history = config.get('history', [])
            return history[-limit:] if history else []
        except Exception as e:
            logger.error(f"Failed to get kill-switch history: {e}")
            return []


def create_kill_switch_cli() -> argparse.ArgumentParser:
    """Create kill-switch CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="Emergency trading kill-switch control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  exo kill status                    # Check current status
  exo kill on --reason "System OK"   # Enable trading
  exo kill off --reason "Market halt" # Disable trading
  exo kill off --reason "Risk breach" --dry-run  # Test disable
        """
    )
    
    parser.add_argument(
        'action',
        choices=['on', 'off', 'status', 'history'],
        help='Kill-switch action: on=enable trading, off=disable trading, status=show current state, history=show recent actions'
    )
    
    parser.add_argument(
        '--reason',
        type=str,
        help='Reason for the action (required for on/off actions)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to kill-switch config file (default: config/killswitch.yml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser


def handle_kill_switch_command(args: argparse.Namespace) -> int:
    """Handle kill-switch CLI command"""
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize kill-switch manager
        manager = KillSwitchManager(args.config)
        
        if args.action == 'status':
            # Show current status
            status = manager.get_status()
            
            print(f"\n🔄 Kill-Switch Status")
            print(f"{'='*50}")
            print(f"Trading Enabled: {'✅ YES' if status['trading_enabled'] else '🚨 NO'}")
            print(f"Reason: {status['reason']}")
            print(f"Last Modified: {status['timestamp']}")
            print(f"Modified By: {status['last_modified_by']}")
            print(f"Auto Triggered: {'Yes' if status['auto_triggered'] else 'No'}")
            if status['trigger_source']:
                print(f"Trigger Source: {status['trigger_source']}")
            print(f"Config File: {status['config_path']}")
            
            return 0 if status['trading_enabled'] else 1
            
        elif args.action == 'history':
            # Show action history
            history = manager.get_history()
            
            print(f"\n📊 Kill-Switch History (Last {len(history)} actions)")
            print(f"{'='*70}")
            
            if not history:
                print("No history available")
                return 0
            
            for entry in reversed(history):  # Most recent first
                action_icon = "✅" if entry['action'] == 'enable' else "🚨"
                auto_text = " [AUTO]" if entry.get('auto_triggered', False) else ""
                print(f"{action_icon} {entry['timestamp']} - {entry['action'].upper()}{auto_text}")
                print(f"   Reason: {entry['reason']}")
                print(f"   By: {entry['triggered_by']}")
                print()
            
            return 0
            
        elif args.action in ['on', 'off']:
            # Enable or disable trading
            if not args.reason and not args.dry_run:
                print("Error: --reason is required for on/off actions")
                return 1
            
            reason = args.reason or f"Test {args.action} action"
            
            if args.action == 'on':
                result = manager.enable_trading(reason, 'operator', args.dry_run)
            else:
                result = manager.disable_trading(reason, 'operator', args.dry_run)
            
            if result['success']:
                action_text = "ENABLED" if args.action == 'on' else "DISABLED"
                dry_text = " [DRY RUN]" if args.dry_run else ""
                print(f"\n{'✅' if args.action == 'on' else '🚨'} Trading {action_text}{dry_text}")
                print(f"Reason: {result['reason']}")
                print(f"Timestamp: {result['timestamp']}")
                
                if not args.dry_run:
                    print(f"\n💡 Verify status with: exo kill status")
                
                return 0
            else:
                print(f"\n❌ Failed to {args.action} trading: {result.get('error', 'Unknown error')}")
                return 1
        
        else:
            print(f"Unknown action: {args.action}")
            return 1
            
    except Exception as e:
        logger.error(f"Kill-switch command failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1


# Utility function for other modules
def is_trading_enabled(config_path: Optional[str] = None) -> bool:
    """
    Quick check if trading is enabled
    
    Args:
        config_path: Optional path to kill-switch config
        
    Returns:
        True if trading is enabled, False otherwise
    """
    try:
        manager = KillSwitchManager(config_path)
        return manager.is_trading_enabled()
    except Exception as e:
        logger.error(f"Failed to check trading status: {e}")
        return False  # Fail-safe to disabled


def get_kill_switch_status(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get kill-switch status for health checks and APIs
    
    Args:
        config_path: Optional path to kill-switch config
        
    Returns:
        Status dictionary
    """
    try:
        manager = KillSwitchManager(config_path)
        return manager.get_status()
    except Exception as e:
        logger.error(f"Failed to get kill-switch status: {e}")
        return {
            'trading_enabled': False,
            'reason': f'Error: {e}',
            'timestamp': datetime.now().isoformat(),
            'last_modified_by': 'system',
            'auto_triggered': False,
            'trigger_source': 'error',
            'error': str(e)
        }


if __name__ == "__main__":
    parser = create_kill_switch_cli()
    args = parser.parse_args()
    sys.exit(handle_kill_switch_command(args))