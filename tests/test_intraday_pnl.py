"""
Test module for intraday PnL monitoring and sentinel functionality

Tests for Day 2 Module 5: Kill-switch and intraday PnL integration
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, date
import json

from mech_exo.reporting.pnl_live import LivePnLMonitor, get_live_nav, create_test_positions_and_nav
from mech_exo.cli.killswitch import is_trading_enabled, get_kill_switch_status


class TestIntradayPnLMonitoring(unittest.TestCase):
    """Test intraday PnL monitoring functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.monitor = None
    
    def tearDown(self):
        """Clean up test environment"""
        if self.monitor:
            self.monitor.close()
    
    def test_create_test_positions_and_nav(self):
        """Test test position creation with target PnL"""
        # Test -1% PnL
        test_data = create_test_positions_and_nav(-1.0)
        
        self.assertIsInstance(test_data, dict)
        self.assertEqual(test_data['pnl_pct'], -1.0)
        self.assertAlmostEqual(test_data['live_nav'], 99000.0, places=0)  # 100k * 0.99
        self.assertEqual(test_data['day_start_nav'], 100000.0)
        self.assertEqual(test_data['pnl_amount'], -1000.0)
        self.assertTrue(test_data['calculation_successful'])
        self.assertGreater(test_data['position_count'], 0)
        self.assertIn('top_positions', test_data)
    
    def test_create_test_positions_positive_pnl(self):
        """Test test position creation with positive PnL"""
        test_data = create_test_positions_and_nav(+0.5)
        
        self.assertEqual(test_data['pnl_pct'], +0.5)
        self.assertAlmostEqual(test_data['live_nav'], 100500.0, places=0)  # 100k * 1.005
        self.assertEqual(test_data['pnl_amount'], 500.0)
    
    @patch('mech_exo.reporting.pnl_live.DataStorage')
    @patch('mech_exo.reporting.pnl_live.FillStore')
    def test_live_pnl_monitor_initialization(self, mock_fill_store, mock_data_storage):
        """Test LivePnLMonitor initialization"""
        # Mock the storage classes
        mock_storage_instance = MagicMock()
        mock_fill_store_instance = MagicMock()
        mock_data_storage.return_value = mock_storage_instance
        mock_fill_store.return_value = mock_fill_store_instance
        
        # Create monitor
        self.monitor = LivePnLMonitor()
        
        # Verify initialization
        self.assertIsNotNone(self.monitor)
        self.assertIsNotNone(self.monitor.storage)
        self.assertIsNotNone(self.monitor.fill_store)
        mock_data_storage.assert_called_once()
        mock_fill_store.assert_called_once()
    
    @patch('mech_exo.reporting.pnl_live.get_live_nav')
    def test_get_live_nav_function(self, mock_get_live_nav):
        """Test convenience function for getting live NAV"""
        # Mock return data
        expected_data = {
            'live_nav': 99500.0,
            'day_start_nav': 100000.0,
            'pnl_amount': -500.0,
            'pnl_pct': -0.5,
            'position_count': 5,
            'calculation_successful': True
        }
        mock_get_live_nav.return_value = expected_data
        
        # Call function
        result = get_live_nav('test_tag')
        
        # Verify results
        self.assertEqual(result, expected_data)
        mock_get_live_nav.assert_called_once_with('test_tag')
    
    def test_threshold_calculations(self):
        """Test PnL threshold calculations"""
        # Test data at various PnL levels
        test_scenarios = [
            (-1.0, False, True),   # -1.0%: critical threshold breach, should trigger kill-switch
            (-0.8, False, True),   # -0.8%: at critical threshold, should trigger kill-switch  
            (-0.5, True, True),    # -0.5%: warning level, should NOT trigger kill-switch
            (-0.2, True, False),   # -0.2%: normal, no warnings
            (+0.1, True, False),   # +0.1%: positive, no warnings
        ]
        
        for pnl_pct, expected_day_loss_ok, expected_warning in test_scenarios:
            with self.subTest(pnl_pct=pnl_pct):
                # Test thresholds
                day_loss_ok = pnl_pct > -0.8  # -0.8% threshold
                warning_level = -0.8 < pnl_pct <= -0.4  # Warning range
                
                self.assertEqual(day_loss_ok, expected_day_loss_ok, 
                               f"day_loss_ok mismatch for {pnl_pct}%")
                self.assertEqual(warning_level, expected_warning,
                               f"warning_level mismatch for {pnl_pct}%")


class TestKillSwitchIntegration(unittest.TestCase):
    """Test kill-switch integration with intraday PnL monitoring"""
    
    @patch('mech_exo.cli.killswitch.Path')
    def test_kill_switch_status_reading(self, mock_path):
        """Test reading kill-switch status"""
        # Mock configuration file
        mock_config_file = MagicMock()
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        
        test_config = {
            'trading_enabled': True,
            'reason': 'System operational',
            'timestamp': '2025-01-12T10:30:00',
            'last_modified_by': 'system'
        }
        
        # Mock file reading
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(test_config))):
            with patch('yaml.safe_load', return_value=test_config):
                status = get_kill_switch_status()
        
                # Verify status
                self.assertIsInstance(status, dict)
                self.assertTrue(status.get('trading_enabled', False))
                self.assertEqual(status.get('reason'), 'System operational')
    
    @patch('mech_exo.cli.killswitch.Path')
    def test_is_trading_enabled(self, mock_path):
        """Test is_trading_enabled function"""
        # Mock configuration file
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        
        test_config = {'trading_enabled': False}
        
        # Mock file reading  
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(test_config))):
            with patch('yaml.safe_load', return_value=test_config):
                enabled = is_trading_enabled()
                
                # Verify disabled state
                self.assertFalse(enabled)


class TestHealthEndpointIntegration(unittest.TestCase):
    """Test health endpoint integration with kill-switch and PnL data"""
    
    @patch('mech_exo.reporting.query.get_live_nav')
    @patch('mech_exo.reporting.query.get_kill_switch_status')
    def test_health_endpoint_data_structure(self, mock_kill_switch, mock_live_nav):
        """Test health endpoint returns required data structure"""
        # Mock kill-switch status
        mock_kill_switch.return_value = {
            'trading_enabled': True,
            'reason': 'System operational'
        }
        
        # Mock live NAV data
        mock_live_nav.return_value = {
            'pnl_pct': -0.3,
            'live_nav': 99700.0,
            'day_start_nav': 100000.0,
            'calculation_successful': True
        }
        
        # Import and call health function
        from mech_exo.reporting.query import get_health_data
        
        with patch('mech_exo.reporting.query.DashboardDataProvider') as mock_provider:
            mock_provider_instance = MagicMock()
            mock_provider.return_value = mock_provider_instance
            mock_provider_instance.get_system_health.return_value = {
                'system_status': 'operational',
                'last_updated': datetime.now().isoformat()
            }
            
            # Mock other health data functions
            with patch('mech_exo.reporting.query.get_risk_data', return_value=({}, {})):
                with patch('mech_exo.reporting.query.get_latest_backtest_metrics', return_value={}):
                    with patch('mech_exo.reporting.query.get_latest_drift_metrics', return_value={}):
                        with patch('mech_exo.reporting.query.get_latest_canary_metrics', return_value={}):
                            health_data = get_health_data()
            
            # Verify required fields are present
            self.assertIn('trading_enabled', health_data)
            self.assertIn('killswitch_reason', health_data)
            self.assertIn('day_loss_pct', health_data)
            self.assertIn('day_loss_ok', health_data)
            self.assertIn('live_nav', health_data)
            self.assertIn('day_start_nav', health_data)
            
            # Verify values
            self.assertTrue(health_data['trading_enabled'])
            self.assertEqual(health_data['killswitch_reason'], 'System operational')
            self.assertEqual(health_data['day_loss_pct'], -0.3)
            self.assertTrue(health_data['day_loss_ok'])  # -0.3% > -0.8%
            self.assertEqual(health_data['live_nav'], 99700.0)
            self.assertEqual(health_data['day_start_nav'], 100000.0)


class TestIntradayMetricsStorage(unittest.TestCase):
    """Test intraday metrics database storage"""
    
    @patch('mech_exo.reporting.pnl_live.DataStorage')
    def test_record_intraday_metrics(self, mock_data_storage):
        """Test recording intraday metrics to database"""
        # Mock storage
        mock_storage_instance = MagicMock()
        mock_data_storage.return_value = mock_storage_instance
        
        # Create monitor with mocked storage
        with patch('mech_exo.reporting.pnl_live.FillStore'):
            monitor = LivePnLMonitor()
            
            # Test data
            nav_data = {
                'live_nav': 99200.0,
                'day_start_nav': 100000.0,
                'pnl_pct': -0.8,
                'position_count': 3,
                'gross_exposure': 150000.0,
                'net_exposure': 50000.0,
                'top_positions': [
                    {'symbol': 'SPY', 'quantity': 100, 'market_value': 40000.0}
                ]
            }
            
            # Call record function
            result = monitor.record_intraday_metrics(nav_data)
            
            # Verify database call was made
            mock_storage_instance.conn.execute.assert_called()
            mock_storage_instance.conn.commit.assert_called()
            self.assertTrue(result)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)