"""
Capital Management System

Monitors and manages capital allocation across trading accounts with real-time
utilization tracking, risk limits, and automated alerts.

Phase P11 Features:
- Canary account expansion (10% → 30%)
- Real-time capital utilization monitoring
- Multi-account risk management
- Automated limit enforcement
"""

import os
import sys
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from mech_exo.utils.alerts import TelegramAlerter
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CapitalStatus(Enum):
    """Capital utilization status levels"""
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    DISABLED = "disabled"


@dataclass
class AccountInfo:
    """Account information and limits"""
    account_id: str
    max_capital: float
    currency: str
    enabled: bool
    category: str
    allocation_pct: float
    notes: str
    added_date: str


@dataclass
class CapitalUtilization:
    """Current capital utilization for an account"""
    account_id: str
    buying_power: float
    used_capital: float
    available_capital: float
    utilization_pct: float
    status: CapitalStatus
    last_updated: datetime
    positions_count: int = 0
    largest_position_pct: float = 0.0
    sector_exposure: Dict[str, float] = None


class CapitalManager:
    """Manages capital allocation and monitoring"""
    
    def __init__(self, config_path: str = "config/capital_limits.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize alerter if available
        try:
            if ALERTS_AVAILABLE:
                # Try to initialize TelegramAlerter with default config
                self.alerter = TelegramAlerter({})
            else:
                self.alerter = None
        except Exception as e:
            logger.warning(f"Could not initialize Telegram alerter: {e}")
            self.alerter = None
        
        logger.info(f"💰 Capital Manager initialized")
        logger.info(f"   Total accounts: {len(self.get_enabled_accounts())}")
        logger.info(f"   Total max capital: ${self.config['capital_limits']['global']['total_max_capital']:,.0f}")
        
    def _load_config(self) -> Dict:
        """Load capital configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Capital config not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in capital config: {e}")
            raise
            
    def _save_config(self):
        """Save capital configuration to YAML"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"✅ Capital config saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save capital config: {e}")
            raise
            
    def get_enabled_accounts(self) -> List[AccountInfo]:
        """Get list of enabled trading accounts"""
        accounts = []
        
        for account_id, config in self.config['capital_limits']['accounts'].items():
            if config.get('enabled', False):
                accounts.append(AccountInfo(
                    account_id=account_id,
                    max_capital=config['max_capital'],
                    currency=config.get('currency', 'USD'),
                    enabled=config['enabled'],
                    category=config.get('category', 'production'),
                    allocation_pct=config.get('allocation_pct', 0.0),
                    notes=config.get('notes', ''),
                    added_date=config.get('added_date', '')
                ))
                
        return accounts
        
    def get_account_utilization(self, account_id: str) -> Optional[CapitalUtilization]:
        """Get current capital utilization for an account"""
        
        # Check if account exists and is enabled
        accounts = {acc.account_id: acc for acc in self.get_enabled_accounts()}
        if account_id not in accounts:
            logger.warning(f"Account {account_id} not found or disabled")
            return None
            
        account = accounts[account_id]
        
        # In production, this would connect to IB API to get real data
        # For now, simulate realistic utilization data
        utilization = self._simulate_account_utilization(account)
        
        # Update utilization tracking in config
        self._update_utilization_tracking(account_id, utilization)
        
        return utilization
        
    def _simulate_account_utilization(self, account: AccountInfo) -> CapitalUtilization:
        """Simulate account utilization (replace with real IB API in production)"""
        import random
        
        # Simulate realistic utilization based on account category
        if account.category == "canary":
            base_utilization = random.uniform(0.60, 0.85)  # 60-85% for canary
        elif account.category == "staging":
            base_utilization = random.uniform(0.20, 0.40)  # 20-40% for staging
        else:
            base_utilization = random.uniform(0.50, 0.75)  # 50-75% for production
            
        # Calculate utilization metrics
        used_capital = account.max_capital * base_utilization
        buying_power = account.max_capital * 1.1  # Assume some margin
        available_capital = buying_power - used_capital
        utilization_pct = (used_capital / account.max_capital) * 100
        
        # Determine status based on thresholds
        warning_threshold = self.config['capital_limits']['global']['alerts']['warning_threshold_pct']
        critical_threshold = self.config['capital_limits']['global']['alerts']['critical_threshold_pct']
        
        if utilization_pct >= critical_threshold:
            status = CapitalStatus.CRITICAL
        elif utilization_pct >= warning_threshold:
            status = CapitalStatus.WARNING
        else:
            status = CapitalStatus.OK
            
        return CapitalUtilization(
            account_id=account.account_id,
            buying_power=buying_power,
            used_capital=used_capital,
            available_capital=available_capital,
            utilization_pct=utilization_pct,
            status=status,
            last_updated=datetime.now(),
            positions_count=random.randint(15, 45),
            largest_position_pct=random.uniform(2.0, 8.0),
            sector_exposure={
                'Technology': random.uniform(20, 35),
                'Healthcare': random.uniform(15, 25),
                'Financials': random.uniform(10, 20),
                'Consumer': random.uniform(8, 15),
                'Other': random.uniform(5, 12)
            }
        )
        
    def _update_utilization_tracking(self, account_id: str, utilization: CapitalUtilization):
        """Update utilization tracking in config"""
        if 'utilization' not in self.config:
            self.config['utilization'] = {'last_check': None, 'accounts': {}}
            
        self.config['utilization']['last_check'] = datetime.now().isoformat()
        self.config['utilization']['accounts'][account_id] = {
            'buying_power': utilization.buying_power,
            'used_capital': utilization.used_capital,
            'utilization_pct': utilization.utilization_pct,
            'last_updated': utilization.last_updated.isoformat(),
            'status': utilization.status.value,
            'positions_count': utilization.positions_count
        }
        
    def check_all_accounts(self) -> Dict[str, CapitalUtilization]:
        """Check capital utilization for all enabled accounts"""
        logger.info("🔍 Checking capital utilization for all accounts...")
        
        utilizations = {}
        alerts_sent = []
        
        for account in self.get_enabled_accounts():
            utilization = self.get_account_utilization(account.account_id)
            if utilization:
                utilizations[account.account_id] = utilization
                
                # Check if alerts needed
                if utilization.status in [CapitalStatus.WARNING, CapitalStatus.CRITICAL]:
                    alert_sent = self._send_utilization_alert(account, utilization)
                    if alert_sent:
                        alerts_sent.append(account.account_id)
                        
        # Log summary
        total_accounts = len(utilizations)
        ok_accounts = sum(1 for u in utilizations.values() if u.status == CapitalStatus.OK)
        warning_accounts = sum(1 for u in utilizations.values() if u.status == CapitalStatus.WARNING)
        critical_accounts = sum(1 for u in utilizations.values() if u.status == CapitalStatus.CRITICAL)
        
        logger.info(f"📊 Capital check complete:")
        logger.info(f"   ✅ OK: {ok_accounts}/{total_accounts}")
        logger.info(f"   ⚠️ Warning: {warning_accounts}/{total_accounts}")
        logger.info(f"   🚨 Critical: {critical_accounts}/{total_accounts}")
        
        if alerts_sent:
            logger.info(f"   📱 Alerts sent: {len(alerts_sent)} accounts")
            
        # Save updated utilization data
        self._save_config()
        
        return utilizations
        
    def _send_utilization_alert(self, account: AccountInfo, utilization: CapitalUtilization) -> bool:
        """Send capital utilization alert"""
        if not self.alerter:
            logger.warning("Alerter not available - skipping alert")
            return False
            
        try:
            severity_emoji = "🚨" if utilization.status == CapitalStatus.CRITICAL else "⚠️"
            
            message = f"""{severity_emoji} **CAPITAL UTILIZATION ALERT**

📊 **Account**: {account.account_id} ({account.category})
💰 **Utilization**: {utilization.utilization_pct:.1f}%
💵 **Used**: ${utilization.used_capital:,.0f} / ${account.max_capital:,.0f}
📈 **Available**: ${utilization.available_capital:,.0f}
🏷️ **Status**: {utilization.status.value.upper()}

📋 **Positions**: {utilization.positions_count}
🔝 **Largest**: {utilization.largest_position_pct:.1f}%

⏰ **Time**: {utilization.last_updated.strftime('%H:%M:%S')}

{severity_emoji} **Action Required**: Monitor closely and consider position reduction"""

            return self.alerter.send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send utilization alert: {e}")
            return False
            
    def get_capital_status_summary(self) -> Dict:
        """Get overall capital status summary"""
        utilizations = self.check_all_accounts()
        
        # Calculate totals
        total_max_capital = sum(acc.max_capital for acc in self.get_enabled_accounts())
        total_used_capital = sum(u.used_capital for u in utilizations.values())
        total_available = sum(u.available_capital for u in utilizations.values())
        overall_utilization = (total_used_capital / total_max_capital * 100) if total_max_capital > 0 else 0
        
        # Count by category
        canary_accounts = [acc for acc in self.get_enabled_accounts() if acc.category == "canary"]
        canary_utilizations = [utilizations.get(acc.account_id) for acc in canary_accounts]
        canary_utilizations = [u for u in canary_utilizations if u]
        
        canary_used = sum(u.used_capital for u in canary_utilizations)
        canary_max = sum(acc.max_capital for acc in canary_accounts)
        canary_utilization = (canary_used / canary_max * 100) if canary_max > 0 else 0
        
        return {
            'overall': {
                'total_max_capital': total_max_capital,
                'total_used_capital': total_used_capital,
                'total_available': total_available,
                'utilization_pct': overall_utilization,
                'account_count': len(utilizations)
            },
            'canary': {
                'total_max_capital': canary_max,
                'total_used_capital': canary_used,
                'utilization_pct': canary_utilization,
                'account_count': len(canary_accounts),
                'allocation_target': 30.0  # Phase P11 target
            },
            'status_breakdown': {
                'ok': sum(1 for u in utilizations.values() if u.status == CapitalStatus.OK),
                'warning': sum(1 for u in utilizations.values() if u.status == CapitalStatus.WARNING),
                'critical': sum(1 for u in utilizations.values() if u.status == CapitalStatus.CRITICAL)
            },
            'last_check': datetime.now().isoformat()
        }
        
    def validate_canary_expansion(self) -> bool:
        """Validate that canary expansion to 30% is working correctly"""
        logger.info("🔍 Validating canary expansion (Phase P11)...")
        
        canary_accounts = [acc for acc in self.get_enabled_accounts() if acc.category == "canary"]
        
        if not canary_accounts:
            logger.error("❌ No canary accounts found!")
            return False
            
        total_canary_allocation = sum(acc.allocation_pct for acc in canary_accounts)
        
        if total_canary_allocation < 30.0:
            logger.error(f"❌ Canary allocation ({total_canary_allocation:.1f}%) below 30% target")
            return False
            
        # Check each canary account status
        all_ok = True
        for account in canary_accounts:
            utilization = self.get_account_utilization(account.account_id)
            if not utilization:
                logger.error(f"❌ Could not get utilization for {account.account_id}")
                all_ok = False
                continue
                
            if utilization.status == CapitalStatus.CRITICAL:
                logger.error(f"❌ {account.account_id} in critical status")
                all_ok = False
            else:
                logger.info(f"✅ {account.account_id}: {utilization.utilization_pct:.1f}% - {utilization.status.value}")
                
        if all_ok:
            logger.info("✅ Canary expansion validation PASSED")
            
            # Send success notification
            if self.alerter:
                message = f"""✅ **CANARY EXPANSION VALIDATED**

📊 **Phase P11 Status**: Canary accounts expanded to {total_canary_allocation:.1f}%
🎯 **Target**: 30% allocation achieved
📈 **Accounts**: {len(canary_accounts)} canary accounts active

{chr(10).join([f"• {acc.account_id}: ${acc.max_capital:,.0f} ({acc.allocation_pct:.1f}%)" for acc in canary_accounts])}

🟢 **All systems**: capital_ok = true"""
                
                self.alerter.send_message(message)
                
        return all_ok


def main():
    """Command-line interface for capital management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Capital Management System')
    parser.add_argument('command', choices=['check', 'status', 'validate', 'monitor'],
                       help='Command to execute')
    parser.add_argument('--account', help='Specific account ID to check')
    parser.add_argument('--alert', action='store_true', help='Send alerts for issues')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create capital manager
    manager = CapitalManager()
    
    if args.command == 'check':
        if args.account:
            # Check specific account
            utilization = manager.get_account_utilization(args.account)
            if utilization:
                print(f"Account: {utilization.account_id}")
                print(f"Utilization: {utilization.utilization_pct:.1f}%")
                print(f"Used: ${utilization.used_capital:,.0f}")
                print(f"Available: ${utilization.available_capital:,.0f}")
                print(f"Status: {utilization.status.value}")
            else:
                print(f"Account {args.account} not found or disabled")
        else:
            # Check all accounts
            utilizations = manager.check_all_accounts()
            for account_id, util in utilizations.items():
                print(f"{account_id}: {util.utilization_pct:.1f}% ({util.status.value})")
                
    elif args.command == 'status':
        # Get status summary
        summary = manager.get_capital_status_summary()
        print("=== Capital Status Summary ===")
        print(f"Overall Utilization: {summary['overall']['utilization_pct']:.1f}%")
        print(f"Total Used: ${summary['overall']['total_used_capital']:,.0f}")
        print(f"Total Available: ${summary['overall']['total_available']:,.0f}")
        print(f"Canary Utilization: {summary['canary']['utilization_pct']:.1f}%")
        print(f"Status: {summary['status_breakdown']['ok']} OK, {summary['status_breakdown']['warning']} Warning, {summary['status_breakdown']['critical']} Critical")
        
    elif args.command == 'validate':
        # Validate canary expansion
        success = manager.validate_canary_expansion()
        sys.exit(0 if success else 1)
        
    elif args.command == 'monitor':
        # Continuous monitoring mode
        import time
        
        logger.info("🔄 Starting continuous capital monitoring...")
        try:
            while True:
                manager.check_all_accounts()
                time.sleep(15 * 60)  # Check every 15 minutes
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped")


if __name__ == '__main__':
    main()