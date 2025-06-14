"""
Test module for monthly drawdown guard functionality

Tests for Day 3 Module 6: Monthly stop-loss protection and alert system
"""

import unittest
from unittest.mock import patch, MagicMock, call
from datetime import datetime, date, timedelta
import json
import yaml

from mech_exo.utils.monthly_loss_guard import (
    get_mtd_pnl_pct, 
    get_month_start_nav, 
    get_monthly_config,
    should_run_monthly_guard,
    get_mtd_summary,
    create_stub_monthly_data
)


class TestMonthlyLossGuard(unittest.TestCase):
    """Test monthly loss guard calculations"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_date = date(2025, 1, 15)  # Mid-month test date
        self.month_start = date(2025, 1, 1)
        self.prev_month_end = date(2024, 12, 31)
    
    @patch('mech_exo.utils.monthly_loss_guard.DataStorage')
    def test_get_month_start_nav_from_daily_metrics(self, mock_storage):
        """Test getting month start NAV from daily metrics"""
        # Mock storage and database response
        mock_storage_instance = MagicMock()
        mock_storage.return_value = mock_storage_instance
        mock_storage_instance.conn.execute.return_value.fetchone.return_value = (100000.0,)
        
        result = get_month_start_nav(self.test_date)
        
        # Verify correct NAV returned
        self.assertEqual(result, 100000.0)
        mock_storage_instance.conn.execute.assert_called()
        mock_storage_instance.close.assert_called()
    
    @patch('mech_exo.utils.monthly_loss_guard.DataStorage')
    def test_get_month_start_nav_fallback(self, mock_storage):
        """Test month start NAV fallback behavior"""
        # Mock storage with no data found
        mock_storage_instance = MagicMock()
        mock_storage.return_value = mock_storage_instance
        mock_storage_instance.conn.execute.return_value.fetchone.return_value = None
        
        # Mock live NAV fallback
        with patch('mech_exo.utils.monthly_loss_guard.get_live_nav') as mock_live_nav:
            mock_live_nav.return_value = {'live_nav': 105000.0}
            
            result = get_month_start_nav(self.test_date)
            
            # Should use fallback NAV
            self.assertEqual(result, 105000.0)
    
    def test_create_stub_monthly_data(self):
        """Test creation of stub data for testing"""
        target_mtd_pct = -3.2
        
        with patch('mech_exo.utils.monthly_loss_guard.DataStorage') as mock_storage:
            mock_storage_instance = MagicMock()
            mock_storage.return_value = mock_storage_instance
            
            # Test stub data creation
            result = create_stub_monthly_data(self.test_date, target_mtd_pct)
            
            # Verify success
            self.assertTrue(result)
            
            # Verify database calls were made
            self.assertTrue(mock_storage_instance.conn.execute.called)
            self.assertTrue(mock_storage_instance.conn.commit.called)
            
            # Check that correct values were calculated
            expected_calls = mock_storage_instance.conn.execute.call_args_list
            self.assertEqual(len(expected_calls), 2)  # Month start + target date
    
    @patch('mech_exo.utils.monthly_loss_guard._get_nav_for_date')
    @patch('mech_exo.utils.monthly_loss_guard.get_month_start_nav')
    def test_get_mtd_pnl_pct_calculation(self, mock_month_start, mock_nav_for_date):
        """Test MTD PnL percentage calculation"""
        # Mock month start and current NAV
        mock_month_start.return_value = 100000.0
        mock_nav_for_date.return_value = 96800.0  # -3.2% loss
        
        result = get_mtd_pnl_pct(self.test_date)
        
        # Verify calculation: (96800 - 100000) / 100000 * 100 = -3.2%
        self.assertAlmostEqual(result, -3.2, places=2)
    
    @patch('mech_exo.utils.monthly_loss_guard._get_nav_for_date')
    @patch('mech_exo.utils.monthly_loss_guard.get_month_start_nav')
    def test_get_mtd_pnl_pct_positive(self, mock_month_start, mock_nav_for_date):
        """Test MTD PnL calculation with positive returns"""
        # Mock positive return scenario
        mock_month_start.return_value = 100000.0
        mock_nav_for_date.return_value = 102500.0  # +2.5% gain
        
        result = get_mtd_pnl_pct(self.test_date)
        
        # Verify positive calculation
        self.assertAlmostEqual(result, 2.5, places=2)
    
    def test_get_monthly_config_defaults(self):
        """Test monthly configuration loading with defaults"""
        with patch('mech_exo.utils.monthly_loss_guard.ConfigManager') as mock_config_manager:
            # Mock config manager that returns empty config
            mock_manager_instance = MagicMock()
            mock_config_manager.return_value = mock_manager_instance
            mock_manager_instance.load_config.side_effect = Exception("Config not found")
            
            config = get_monthly_config()
            
            # Verify defaults are applied
            self.assertTrue(config['enabled'])
            self.assertEqual(config['threshold_pct'], -3.0)
            self.assertEqual(config['min_history_days'], 10)
            self.assertTrue(config['alert_enabled'])
            self.assertFalse(config['dry_run'])
    
    @patch('mech_exo.utils.monthly_loss_guard.is_trading_enabled')
    def test_should_run_monthly_guard_early_month(self, mock_trading_enabled):
        """Test guard doesn't run early in month"""
        mock_trading_enabled.return_value = True
        
        # Test early in month (day 5, min_history_days=10)
        early_date = date(2025, 1, 5)
        
        should_run, reason = should_run_monthly_guard(early_date)
        
        self.assertFalse(should_run)
        self.assertIn("Too early in month", reason)
    
    @patch('mech_exo.utils.monthly_loss_guard.is_trading_enabled')
    @patch('mech_exo.utils.monthly_loss_guard.get_monthly_config')
    def test_should_run_monthly_guard_killswitch_disabled(self, mock_config, mock_trading_enabled):
        """Test guard doesn't run when kill-switch already disabled"""
        mock_config.return_value = {'enabled': True, 'min_history_days': 10}
        mock_trading_enabled.return_value = False  # Already disabled
        
        should_run, reason = should_run_monthly_guard(self.test_date)
        
        self.assertFalse(should_run)
        self.assertIn("Kill-switch already disabled", reason)
    
    @patch('mech_exo.utils.monthly_loss_guard.is_trading_enabled')
    @patch('mech_exo.utils.monthly_loss_guard.get_monthly_config')
    def test_should_run_monthly_guard_normal(self, mock_config, mock_trading_enabled):
        """Test guard runs under normal conditions"""
        mock_config.return_value = {'enabled': True, 'min_history_days': 10}
        mock_trading_enabled.return_value = True
        
        should_run, reason = should_run_monthly_guard(self.test_date)
        
        self.assertTrue(should_run)
        self.assertIn("Monthly guard should run", reason)


class TestMonthlyGuardFlow(unittest.TestCase):
    """Test monthly guard Prefect flow functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_date = date(2025, 1, 15)
    
    @patch('dags.monthly_guard_flow.get_mtd_pnl_pct')
    @patch('dags.monthly_guard_flow.should_run_monthly_guard')
    @patch('dags.monthly_guard_flow.get_monthly_config')
    def test_calc_mtd_loss_task_normal(self, mock_config, mock_should_run, mock_mtd_pnl):
        """Test MTD loss calculation task under normal conditions"""
        # Mock normal operation
        mock_config.return_value = {'threshold_pct': -3.0}
        mock_should_run.return_value = (True, "Should run")
        mock_mtd_pnl.return_value = -1.5  # Normal loss level
        
        from dags.monthly_guard_flow import calc_mtd_loss_task
        
        result = calc_mtd_loss_task(self.test_date)
        
        # Verify normal result
        self.assertTrue(result['calculation_successful'])
        self.assertEqual(result['mtd_pct'], -1.5)
        self.assertFalse(result['threshold_breached'])  # -1.5% > -3.0%
        self.assertTrue(result['should_run'])
    
    @patch('dags.monthly_guard_flow.get_mtd_pnl_pct')
    @patch('dags.monthly_guard_flow.should_run_monthly_guard')
    @patch('dags.monthly_guard_flow.get_monthly_config')
    def test_calc_mtd_loss_task_threshold_breach(self, mock_config, mock_should_run, mock_mtd_pnl):
        """Test MTD loss calculation task with threshold breach"""
        # Mock threshold breach
        mock_config.return_value = {'threshold_pct': -3.0}
        mock_should_run.return_value = (True, "Should run")
        mock_mtd_pnl.return_value = -3.5  # Breach threshold
        
        from dags.monthly_guard_flow import calc_mtd_loss_task
        
        result = calc_mtd_loss_task(self.test_date)
        
        # Verify breach detected
        self.assertTrue(result['calculation_successful'])
        self.assertEqual(result['mtd_pct'], -3.5)
        self.assertTrue(result['threshold_breached'])  # -3.5% ≤ -3.0%
        self.assertTrue(result['should_run'])
    
    @patch('dags.monthly_guard_flow._trigger_monthly_killswitch')
    @patch('dags.monthly_guard_flow._send_monthly_alert')
    def test_check_monthly_stop_task_breach(self, mock_send_alert, mock_trigger_killswitch):
        """Test monthly stop check task with threshold breach"""
        # Mock successful kill-switch and alert
        mock_trigger_killswitch.return_value = {
            'killswitch_triggered': True,
            'action_taken': 'killswitch_triggered'
        }
        mock_send_alert.return_value = {
            'alert_sent': True,
            'alert_status': 'sent'
        }
        
        from dags.monthly_guard_flow import check_monthly_stop_task
        
        # Input data showing threshold breach
        mtd_data = {
            'should_run': True,
            'calculation_successful': True,
            'mtd_pct': -3.2,
            'threshold_pct': -3.0,
            'threshold_breached': True,
            'config': {'dry_run': False}
        }
        
        result = check_monthly_stop_task(mtd_data)
        
        # Verify actions taken
        self.assertTrue(result['threshold_breached'])
        self.assertTrue(result['killswitch_triggered'])
        self.assertTrue(result['alert_sent'])
        mock_trigger_killswitch.assert_called_once()
        mock_send_alert.assert_called_once()
    
    @patch('subprocess.run')
    def test_trigger_monthly_killswitch_success(self, mock_subprocess):
        """Test successful kill-switch triggering"""
        # Mock successful subprocess call
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Kill-switch disabled successfully"
        mock_subprocess.return_value = mock_result
        
        from dags.monthly_guard_flow import _trigger_monthly_killswitch
        
        mock_logger = MagicMock()
        result = _trigger_monthly_killswitch(-3.2, mock_logger)
        
        # Verify kill-switch command called
        self.assertTrue(result['killswitch_triggered'])
        self.assertEqual(result['action_taken'], 'killswitch_triggered')
        mock_subprocess.assert_called_once()
        
        # Verify correct command structure
        call_args = mock_subprocess.call_args[0][0]
        self.assertIn('mech_exo.cli', call_args)
        self.assertIn('kill', call_args)
        self.assertIn('off', call_args)


class TestMonthlyAlertSystem(unittest.TestCase):
    """Test monthly alert system"""
    
    @patch('mech_exo.utils.alerts.AlertManager')
    @patch('subprocess.check_output')
    def test_send_monthly_loss_alert(self, mock_git, mock_alert_manager):
        """Test monthly loss alert sending"""
        # Mock git hash
        mock_git.return_value = b'abc1234\n'
        
        # Mock alert manager
        mock_manager_instance = MagicMock()
        mock_alert_manager.return_value = mock_manager_instance
        mock_manager_instance.send_alert_with_escalation.return_value = True
        
        from mech_exo.utils.alerts import send_monthly_loss_alert
        
        result = send_monthly_loss_alert(-3.2, -3.0)
        
        # Verify alert sent
        self.assertTrue(result)
        mock_manager_instance.send_alert_with_escalation.assert_called_once()
        
        # Verify alert structure
        call_args = mock_manager_instance.send_alert_with_escalation.call_args
        alert = call_args[0][0]  # First positional argument is the alert
        
        # Check alert properties
        self.assertEqual(alert.alert_type.value, 'system_alert')
        self.assertEqual(alert.level.value, 'critical')
        self.assertIn('-3.2%', alert.title)
        self.assertIn('MONTHLY', alert.message)
        self.assertIn('KILL-SWITCH', alert.message)
        
        # Verify alert data
        self.assertEqual(alert.data['mtd_pct'], -3.2)
        self.assertEqual(alert.data['threshold_pct'], -3.0)
        self.assertEqual(alert.data['alert_type'], 'monthly_stop_loss')
    
    def test_send_monthly_loss_alert_message_format(self):
        """Test monthly alert message format and length"""
        from mech_exo.utils.alerts import send_monthly_loss_alert
        
        with patch('mech_exo.utils.alerts.AlertManager') as mock_alert_manager:
            mock_manager_instance = MagicMock()
            mock_alert_manager.return_value = mock_alert_manager_instance = mock_manager_instance
            mock_manager_instance.send_alert_with_escalation.return_value = True
            
            # Test with different MTD values
            send_monthly_loss_alert(-3.15, -3.0)
            
            # Get the alert message
            call_args = mock_manager_instance.send_alert_with_escalation.call_args
            alert = call_args[0][0]
            message = alert.message
            
            # Verify message format
            self.assertIn('3.15%', message)  # MTD percentage with 2 decimal places
            self.assertIn('-3%', message)    # Threshold
            self.assertIn('0.15%', message)  # Breach amount (diff to threshold)
            
            # Verify message length (Telegram limit is 4096 chars)
            self.assertLessEqual(len(message), 4096)
            
            # Verify MarkdownV2 escaping
            self.assertIn('\\-', message)  # Escaped hyphens
            self.assertIn('**', message)   # Bold text
            self.assertNotIn('_', message.replace('_', '')) # Should be escaped if present


class TestKillSwitchIntegration(unittest.TestCase):
    """Test kill-switch integration with monthly guard"""
    
    @patch('mech_exo.cli.killswitch.Path')
    def test_killswitch_yaml_update(self, mock_path):
        """Test that kill-switch YAML is properly updated"""
        # Mock file system
        mock_config_file = MagicMock()
        mock_path.return_value = mock_config_file
        mock_config_file.exists.return_value = True
        
        test_config = {
            'trading_enabled': True,
            'reason': 'System operational',
            'history': []
        }
        
        # Mock file reading and writing
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            with patch('yaml.safe_load', return_value=test_config):
                with patch('yaml.safe_dump') as mock_dump:
                    from mech_exo.cli.killswitch import KillSwitchManager
                    
                    manager = KillSwitchManager()
                    result = manager.disable_trading(
                        reason="-3.2% monthly stop",
                        triggered_by="monthly_guard"
                    )
                    
                    # Verify kill-switch was disabled
                    self.assertTrue(result['success'])
                    self.assertEqual(result['action'], 'disable')
                    
                    # Verify YAML was written
                    mock_dump.assert_called()


if __name__ == '__main__':
    # Run specific test scenarios for monthly guard functionality
    
    # Test MTD calculation with stub data  
    print("🧪 Testing Monthly Guard with stub data...")
    
    # Create test scenarios
    test_scenarios = [
        ("Normal operation", -1.5, False),
        ("At threshold", -3.0, True), 
        ("Breach threshold", -3.2, True),
        ("Positive return", +2.1, False)
    ]
    
    for scenario_name, mtd_pct, should_trigger in test_scenarios:
        print(f"\n📊 {scenario_name}: {mtd_pct:+.1f}%")
        
        # Mock the calculation to return our test value
        with patch('mech_exo.utils.monthly_loss_guard.get_mtd_pnl_pct', return_value=mtd_pct):
            from mech_exo.utils.monthly_loss_guard import get_mtd_summary
            
            summary = get_mtd_summary()
            threshold_breached = summary['threshold_breached']
            
            print(f"   Expected trigger: {should_trigger}")
            print(f"   Actual trigger: {threshold_breached}")
            print(f"   Status: {summary['status']}")
            
            assert threshold_breached == should_trigger, f"Trigger mismatch for {scenario_name}"
    
    print("\n✅ Monthly Guard tests passed")
    
    # Run full unittest suite
    unittest.main(verbosity=2)