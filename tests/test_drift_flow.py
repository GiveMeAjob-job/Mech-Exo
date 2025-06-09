"""
Unit tests for drift monitoring flow
"""

import pytest
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock
import json

from dags.drift_flow import (
    calc_drift, store_drift_metrics, alert_if_breach, 
    drift_monitor_flow, run_manual_drift_monitor
)


class TestDriftFlow:
    """Test cases for drift monitoring flow"""

    def test_calc_drift_with_date(self):
        """Test drift calculation with specific date"""
        with patch('dags.drift_flow.calculate_daily_drift') as mock_calc:
            expected_metrics = {
                'date': '2024-01-15',
                'drift_pct': 5.0,
                'information_ratio': 1.2,
                'data_quality': 'good'
            }
            mock_calc.return_value = expected_metrics
            
            result = calc_drift('2024-01-15')
            
            assert result['date'] == '2024-01-15'
            assert result['drift_pct'] == 5.0
            assert 'calculated_at' in result
            mock_calc.assert_called_once()

    def test_calc_drift_error_handling(self):
        """Test drift calculation error handling"""
        with patch('dags.drift_flow.calculate_daily_drift') as mock_calc:
            mock_calc.side_effect = Exception("Test error")
            
            result = calc_drift()
            
            # Should return default metrics on error
            assert result['drift_pct'] == 0.0
            assert result['information_ratio'] == 0.0
            assert result['data_quality'] == 'error'

    def test_store_drift_metrics_success(self):
        """Test successful drift metrics storage"""
        mock_drift_data = {
            'date': '2024-01-15',
            'calculated_at': '2024-01-15T10:00:00',
            'live_cagr': 0.15,
            'backtest_cagr': 0.12,
            'drift_pct': 10.0,
            'information_ratio': 0.8,
            'excess_return_mean': 0.001,
            'excess_return_std': 0.02,
            'tracking_error': 0.15,
            'data_quality': 'good',
            'days_analyzed': 25
        }
        
        with patch('dags.drift_flow.DataStorage') as mock_storage_class:
            mock_storage = Mock()
            mock_conn = Mock()
            mock_storage.conn = mock_conn
            mock_storage_class.return_value = mock_storage
            
            result = store_drift_metrics(mock_drift_data)
            
            assert result == True
            assert mock_conn.execute.call_count == 2  # CREATE TABLE + INSERT

    def test_store_drift_metrics_error(self):
        """Test drift metrics storage error handling"""
        with patch('dags.drift_flow.DataStorage') as mock_storage_class:
            mock_storage_class.side_effect = Exception("Database error")
            
            result = store_drift_metrics({})
            
            assert result == False

    def test_alert_if_breach_no_alert_needed(self):
        """Test alert when no alert is needed (OK status)"""
        drift_data = {
            'drift_pct': 5.0,
            'information_ratio': 1.5,
            'data_quality': 'good'
        }
        
        result = alert_if_breach(drift_data)
        assert result == True  # No alert needed but function succeeds

    def test_alert_if_breach_warn_status(self):
        """Test alert for WARN status drift"""
        drift_data = {
            'date': '2024-01-15',
            'drift_pct': 12.0,  # Should trigger WARN
            'information_ratio': 0.8,
            'live_cagr': 0.20,
            'backtest_cagr': 0.08,
            'tracking_error': 0.15,
            'data_quality': 'good',
            'days_analyzed': 25
        }
        
        with patch('dags.drift_flow.AlertManager') as mock_alert_class:
            mock_alert_manager = Mock()
            mock_alert_manager.send_alert.return_value = True
            mock_alert_class.return_value = mock_alert_manager
            
            result = alert_if_breach(drift_data)
            
            assert result == True
            mock_alert_manager.send_alert.assert_called_once()
            
            # Check alert message content
            call_args = mock_alert_manager.send_alert.call_args
            assert "WARNING" in call_args[1]['subject']
            assert "12.0%" in call_args[1]['message']

    def test_alert_if_breach_breach_status(self):
        """Test alert for BREACH status drift"""
        drift_data = {
            'date': '2024-01-15',
            'drift_pct': 25.0,  # Should trigger BREACH
            'information_ratio': -0.2,  # Negative IR
            'live_cagr': 0.30,
            'backtest_cagr': 0.05,
            'tracking_error': 0.25,
            'data_quality': 'good',
            'days_analyzed': 30
        }
        
        with patch('dags.drift_flow.AlertManager') as mock_alert_class:
            mock_alert_manager = Mock()
            mock_alert_manager.send_alert.return_value = True
            mock_alert_class.return_value = mock_alert_manager
            
            result = alert_if_breach(drift_data)
            
            assert result == True
            mock_alert_manager.send_alert.assert_called_once()
            
            # Check alert message content
            call_args = mock_alert_manager.send_alert.call_args
            assert "BREACH" in call_args[1]['subject']
            assert "🚨" in call_args[1]['message']

    def test_alert_if_breach_poor_data_quality(self):
        """Test alert skipping for poor data quality"""
        drift_data = {
            'drift_pct': 25.0,
            'information_ratio': -0.2,
            'data_quality': 'no_backtest'  # Poor quality should skip alert
        }
        
        result = alert_if_breach(drift_data)
        assert result == True  # Skipped but considered successful

    def test_drift_monitor_flow_success(self):
        """Test complete drift monitor flow success"""
        mock_drift_data = {
            'date': '2024-01-15',
            'drift_pct': 8.0,
            'information_ratio': 1.2,
            'data_quality': 'good',
            'days_analyzed': 25
        }
        
        with patch('dags.drift_flow.calc_drift') as mock_calc:
            mock_calc.return_value = mock_drift_data
            
            with patch('dags.drift_flow.store_drift_metrics') as mock_store:
                mock_store.return_value = True
                
                with patch('dags.drift_flow.alert_if_breach') as mock_alert:
                    mock_alert.return_value = True
                    
                    result = drift_monitor_flow('2024-01-15')
                    
                    assert result['flow_status'] == 'completed'
                    assert result['drift_date'] == '2024-01-15'
                    assert result['drift_pct'] == 8.0
                    assert result['drift_status'] == 'OK'  # 8% should be OK
                    assert result['store_success'] == True
                    assert result['alert_success'] == True
                    assert 'completed_at' in result

    def test_drift_monitor_flow_failure(self):
        """Test drift monitor flow error handling"""
        with patch('dags.drift_flow.calc_drift') as mock_calc:
            mock_calc.side_effect = Exception("Test flow error")
            
            result = drift_monitor_flow()
            
            assert result['flow_status'] == 'failed'
            assert 'error' in result
            assert 'completed_at' in result

    def test_run_manual_drift_monitor(self):
        """Test manual drift monitor execution"""
        with patch('dags.drift_flow.drift_monitor_flow') as mock_flow:
            expected_result = {'flow_status': 'completed', 'drift_pct': 5.0}
            mock_flow.return_value = expected_result
            
            result = run_manual_drift_monitor('2024-01-15')
            
            assert result == expected_result
            mock_flow.assert_called_once_with('2024-01-15')


class TestDriftThresholds:
    """Test drift threshold logic"""

    def test_drift_status_thresholds(self):
        """Test that drift status thresholds work correctly"""
        from mech_exo.reporting.drift import get_drift_status
        
        # OK cases
        assert get_drift_status(5.0, 1.5) == 'OK'
        assert get_drift_status(-5.0, 0.8) == 'OK'
        
        # WARN cases
        assert get_drift_status(15.0, 1.0) == 'WARN'  # High drift
        assert get_drift_status(5.0, 0.1) == 'WARN'   # Low IR
        
        # BREACH cases
        assert get_drift_status(25.0, 1.0) == 'BREACH'  # Very high drift
        assert get_drift_status(5.0, -0.5) == 'BREACH'  # Negative IR


@pytest.fixture
def sample_drift_data():
    """Sample drift data for testing"""
    return {
        'date': '2024-01-15',
        'calculated_at': '2024-01-15T10:00:00',
        'live_cagr': 0.15,
        'backtest_cagr': 0.12,
        'drift_pct': 10.0,
        'information_ratio': 0.8,
        'excess_return_mean': 0.001,
        'excess_return_std': 0.02,
        'tracking_error': 0.15,
        'data_quality': 'good',
        'days_analyzed': 25
    }


def test_integration_flow_with_sample_data(sample_drift_data):
    """Integration test with sample data"""
    # This would test the full flow end-to-end
    # For now, just verify the data structure
    assert 'date' in sample_drift_data
    assert 'drift_pct' in sample_drift_data
    assert 'information_ratio' in sample_drift_data
    assert sample_drift_data['data_quality'] == 'good'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])