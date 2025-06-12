"""
Capital Management CLI

Manages capital limits and account whitelisting for production trading.
Provides commands to add/remove accounts and set capital allocation limits.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

logger = logging.getLogger(__name__)


class CapitalManager:
    """Manages capital limits configuration"""
    
    def __init__(self, config_path: str = "config/capital_limits.yml"):
        """
        Initialize capital manager
        
        Args:
            config_path: Path to capital limits configuration file
        """
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure config file exists
        if not self.config_path.exists():
            self._create_default_config()
        
        self.config = self._load_config()
    
    def _create_default_config(self):
        """Create default capital limits configuration"""
        default_config = {
            'capital_limits': {
                'accounts': {},
                'global': {
                    'total_max_capital': 500000,
                    'default_currency': 'USD',
                    'safety_margin_pct': 10,
                    'alerts': {
                        'warning_threshold_pct': 80,
                        'critical_threshold_pct': 95
                    },
                    'check_frequency_minutes': 60
                }
            },
            'utilization': {
                'last_check': None,
                'accounts': {}
            },
            'history': {
                'changes': []
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load capital limits configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _save_config(self):
        """Save capital limits configuration"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
    
    def _log_change(self, action: str, account: str, old_limit: Optional[float], 
                   new_limit: Optional[float], user: str = "cli"):
        """Log capital limit change"""
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'account': account,
            'old_limit': old_limit,
            'new_limit': new_limit,
            'user': user
        }
        
        if 'history' not in self.config:
            self.config['history'] = {'changes': []}
        
        self.config['history']['changes'].append(change_record)
        
        # Keep only last 100 changes
        if len(self.config['history']['changes']) > 100:
            self.config['history']['changes'] = self.config['history']['changes'][-100:]
    
    def add_account(self, account_id: str, max_capital: float, 
                   currency: str = "USD", notes: str = "") -> bool:
        """
        Add account to capital whitelist
        
        Args:
            account_id: IB account ID (e.g., DU12345678)
            max_capital: Maximum capital allocation
            currency: Account currency
            notes: Optional notes
            
        Returns:
            True if successful
        """
        try:
            # Validate account ID format
            if not self._validate_account_id(account_id):
                raise ValueError(f"Invalid account ID format: {account_id}")
            
            # Check if account already exists
            accounts = self.config['capital_limits']['accounts']
            old_limit = accounts.get(account_id, {}).get('max_capital')
            
            # Validate total capital doesn't exceed global limit
            total_max = self.config['capital_limits']['global']['total_max_capital']
            current_total = sum(acc.get('max_capital', 0) for acc in accounts.values() 
                              if acc.get('enabled', True))
            
            if account_id not in accounts:
                # New account
                if current_total + max_capital > total_max:
                    raise ValueError(f"Total capital would exceed global limit: "
                                   f"{current_total + max_capital} > {total_max}")
            else:
                # Existing account - adjust calculation
                old_capital = accounts[account_id].get('max_capital', 0)
                if current_total - old_capital + max_capital > total_max:
                    raise ValueError(f"Total capital would exceed global limit")
            
            # Add/update account
            accounts[account_id] = {
                'max_capital': max_capital,
                'currency': currency,
                'enabled': True,
                'added_date': datetime.now().strftime('%Y-%m-%d'),
                'notes': notes
            }
            
            # Log the change
            action = "update_account" if old_limit is not None else "add_account"
            self._log_change(action, account_id, old_limit, max_capital)
            
            # Save configuration
            self._save_config()
            
            logger.info(f"Account {account_id} {'updated' if old_limit else 'added'} "
                       f"with max capital: {max_capital:,.0f} {currency}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add account {account_id}: {e}")
            raise
    
    def remove_account(self, account_id: str, force: bool = False) -> bool:
        """
        Remove account from capital whitelist
        
        Args:
            account_id: IB account ID
            force: Skip safety checks
            
        Returns:
            True if successful
        """
        try:
            accounts = self.config['capital_limits']['accounts']
            
            if account_id not in accounts:
                raise ValueError(f"Account {account_id} not found in whitelist")
            
            # Safety check - warn if account has recent activity
            if not force:
                utilization = self.config.get('utilization', {}).get('accounts', {})
                if account_id in utilization:
                    last_check = utilization[account_id].get('last_updated')
                    if last_check:
                        # Account has recent activity - require force flag
                        logger.warning(f"Account {account_id} has recent activity. Use --force to remove.")
                        return False
            
            # Store old limit for logging
            old_limit = accounts[account_id].get('max_capital')
            
            # Remove account
            del accounts[account_id]
            
            # Clean up utilization data
            if 'utilization' in self.config and 'accounts' in self.config['utilization']:
                if account_id in self.config['utilization']['accounts']:
                    del self.config['utilization']['accounts'][account_id]
            
            # Log the change
            self._log_change("remove_account", account_id, old_limit, None)
            
            # Save configuration
            self._save_config()
            
            logger.info(f"Account {account_id} removed from whitelist")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove account {account_id}: {e}")
            raise
    
    def disable_account(self, account_id: str) -> bool:
        """
        Disable account without removing from whitelist
        
        Args:
            account_id: IB account ID
            
        Returns:
            True if successful
        """
        try:
            accounts = self.config['capital_limits']['accounts']
            
            if account_id not in accounts:
                raise ValueError(f"Account {account_id} not found in whitelist")
            
            old_enabled = accounts[account_id].get('enabled', True)
            accounts[account_id]['enabled'] = False
            
            # Log the change
            self._log_change("disable_account", account_id, 
                           accounts[account_id]['max_capital'], 
                           accounts[account_id]['max_capital'])
            
            # Save configuration
            self._save_config()
            
            logger.info(f"Account {account_id} disabled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable account {account_id}: {e}")
            raise
    
    def enable_account(self, account_id: str) -> bool:
        """
        Enable previously disabled account
        
        Args:
            account_id: IB account ID
            
        Returns:
            True if successful
        """
        try:
            accounts = self.config['capital_limits']['accounts']
            
            if account_id not in accounts:
                raise ValueError(f"Account {account_id} not found in whitelist")
            
            accounts[account_id]['enabled'] = True
            
            # Log the change
            self._log_change("enable_account", account_id,
                           accounts[account_id]['max_capital'],
                           accounts[account_id]['max_capital'])
            
            # Save configuration
            self._save_config()
            
            logger.info(f"Account {account_id} enabled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable account {account_id}: {e}")
            raise
    
    def list_accounts(self) -> List[Dict[str, Any]]:
        """
        List all accounts in whitelist
        
        Returns:
            List of account information
        """
        accounts = self.config['capital_limits']['accounts']
        utilization = self.config.get('utilization', {}).get('accounts', {})
        
        result = []
        for account_id, config in accounts.items():
            account_info = {
                'account_id': account_id,
                'max_capital': config['max_capital'],
                'currency': config['currency'],
                'enabled': config.get('enabled', True),
                'added_date': config.get('added_date'),
                'notes': config.get('notes', ''),
            }
            
            # Add utilization data if available
            if account_id in utilization:
                util = utilization[account_id]
                account_info.update({
                    'buying_power': util.get('buying_power'),
                    'used_capital': util.get('used_capital'),
                    'utilization_pct': util.get('utilization_pct'),
                    'status': util.get('status'),
                    'last_updated': util.get('last_updated')
                })
            
            result.append(account_info)
        
        return result
    
    def get_total_limits(self) -> Dict[str, Any]:
        """
        Get total capital limits summary
        
        Returns:
            Summary of capital allocation
        """
        accounts = self.config['capital_limits']['accounts']
        global_config = self.config['capital_limits']['global']
        
        total_allocated = sum(acc['max_capital'] for acc in accounts.values() 
                            if acc.get('enabled', True))
        total_max = global_config['total_max_capital']
        
        enabled_accounts = sum(1 for acc in accounts.values() if acc.get('enabled', True))
        total_accounts = len(accounts)
        
        return {
            'total_accounts': total_accounts,
            'enabled_accounts': enabled_accounts,
            'total_allocated': total_allocated,
            'total_max_capital': total_max,
            'remaining_capacity': total_max - total_allocated,
            'utilization_pct': (total_allocated / total_max * 100) if total_max > 0 else 0,
            'safety_margin_pct': global_config.get('safety_margin_pct', 10)
        }
    
    def get_change_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent capital limit changes
        
        Args:
            limit: Maximum number of changes to return
            
        Returns:
            List of recent changes
        """
        changes = self.config.get('history', {}).get('changes', [])
        return changes[-limit:] if len(changes) > limit else changes
    
    @staticmethod
    def _validate_account_id(account_id: str) -> bool:
        """
        Validate Interactive Brokers account ID format
        
        Args:
            account_id: Account ID to validate
            
        Returns:
            True if valid format
        """
        # IB account IDs are typically 8-9 characters, start with letters
        if not account_id or len(account_id) < 8 or len(account_id) > 9:
            return False
        
        # Should start with letters (DU, DF, etc.)
        if not account_id[:2].isalpha():
            return False
        
        # Rest should be digits
        if not account_id[2:].isdigit():
            return False
        
        return True


def create_capital_parser(subparsers) -> argparse.ArgumentParser:
    """Create capital management argument parser"""
    
    capital_parser = subparsers.add_parser("capital", 
                                         help="Manage capital limits and account whitelist")
    capital_subparsers = capital_parser.add_subparsers(dest="capital_action", 
                                                     help="Capital management actions")
    
    # Add account command
    add_parser = capital_subparsers.add_parser("add", help="Add account to whitelist")
    add_parser.add_argument("account_id", help="IB account ID (e.g., DU12345678)")
    add_parser.add_argument("--max", type=float, required=True, 
                          help="Maximum capital allocation")
    add_parser.add_argument("--currency", default="USD", 
                          help="Account currency (default: USD)")
    add_parser.add_argument("--notes", default="", 
                          help="Optional notes")
    
    # Remove account command
    remove_parser = capital_subparsers.add_parser("remove", help="Remove account from whitelist")
    remove_parser.add_argument("account_id", help="IB account ID")
    remove_parser.add_argument("--force", action="store_true", 
                             help="Force removal even with recent activity")
    
    # Enable/disable commands
    enable_parser = capital_subparsers.add_parser("enable", help="Enable account")
    enable_parser.add_argument("account_id", help="IB account ID")
    
    disable_parser = capital_subparsers.add_parser("disable", help="Disable account")
    disable_parser.add_argument("account_id", help="IB account ID")
    
    # List command
    list_parser = capital_subparsers.add_parser("list", help="List all accounts")
    list_parser.add_argument("--format", choices=["table", "json"], default="table",
                           help="Output format")
    
    # Status command
    status_parser = capital_subparsers.add_parser("status", help="Show capital allocation status")
    
    # History command
    history_parser = capital_subparsers.add_parser("history", help="Show change history")
    history_parser.add_argument("--limit", type=int, default=20,
                               help="Number of changes to show")
    
    return capital_parser


def handle_capital_command(args):
    """Handle capital management commands"""
    try:
        manager = CapitalManager()
        
        if args.capital_action == "add":
            success = manager.add_account(
                args.account_id, 
                args.max, 
                args.currency, 
                args.notes
            )
            if success:
                print(f"✅ Account {args.account_id} added with max capital: {args.max:,.0f} {args.currency}")
            
        elif args.capital_action == "remove":
            success = manager.remove_account(args.account_id, args.force)
            if success:
                print(f"✅ Account {args.account_id} removed from whitelist")
            else:
                print(f"❌ Failed to remove account {args.account_id}")
                
        elif args.capital_action == "enable":
            success = manager.enable_account(args.account_id)
            if success:
                print(f"✅ Account {args.account_id} enabled")
                
        elif args.capital_action == "disable":
            success = manager.disable_account(args.account_id)
            if success:
                print(f"✅ Account {args.account_id} disabled")
                
        elif args.capital_action == "list":
            accounts = manager.list_accounts()
            
            if args.format == "json":
                import json
                print(json.dumps(accounts, indent=2, default=str))
            else:
                # Table format
                print("📋 Capital Whitelist")
                print("=" * 80)
                
                if not accounts:
                    print("No accounts configured")
                    return
                
                # Header
                print(f"{'Account ID':<12} {'Max Capital':<12} {'Currency':<8} {'Status':<8} {'Usage':<10} {'Added':<12}")
                print("-" * 80)
                
                # Accounts
                for acc in accounts:
                    status = "Enabled" if acc['enabled'] else "Disabled"
                    usage = f"{acc.get('utilization_pct', 0):.1f}%" if 'utilization_pct' in acc else "N/A"
                    
                    print(f"{acc['account_id']:<12} {acc['max_capital']:>11,.0f} {acc['currency']:<8} "
                          f"{status:<8} {usage:<10} {acc.get('added_date', 'N/A'):<12}")
                
        elif args.capital_action == "status":
            summary = manager.get_total_limits()
            
            print("📊 Capital Allocation Status")
            print("=" * 40)
            print(f"Total Accounts: {summary['enabled_accounts']}/{summary['total_accounts']}")
            print(f"Total Allocated: {summary['total_allocated']:,.0f}")
            print(f"Maximum Capacity: {summary['total_max_capital']:,.0f}")
            print(f"Remaining: {summary['remaining_capacity']:,.0f}")
            print(f"Utilization: {summary['utilization_pct']:.1f}%")
            print(f"Safety Margin: {summary['safety_margin_pct']}%")
            
        elif args.capital_action == "history":
            changes = manager.get_change_history(args.limit)
            
            print(f"📜 Recent Capital Changes (Last {len(changes)})")
            print("=" * 60)
            
            if not changes:
                print("No changes recorded")
                return
            
            for change in reversed(changes):  # Most recent first
                timestamp = change['timestamp'][:19]  # Remove microseconds
                action = change['action'].replace('_', ' ').title()
                account = change['account']
                
                print(f"{timestamp} - {action}: {account}")
                
                if change['old_limit'] and change['new_limit']:
                    print(f"   {change['old_limit']:,.0f} → {change['new_limit']:,.0f}")
                elif change['new_limit']:
                    print(f"   Added: {change['new_limit']:,.0f}")
                elif change['old_limit']:
                    print(f"   Removed: {change['old_limit']:,.0f}")
                
                print()
        
        else:
            print("❌ Unknown capital action")
            
    except Exception as e:
        print(f"❌ Capital command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Test the capital manager
    manager = CapitalManager()
    
    # Test adding an account
    manager.add_account("DU12345678", 100000, "USD", "Test account")
    
    # List accounts
    accounts = manager.list_accounts()
    print("Accounts:", accounts)
    
    # Get status
    status = manager.get_total_limits()
    print("Status:", status)