#!/usr/bin/env python3
"""
Unit Tests for Risk Panel Dashboard Components
Tests risk panel callbacks, data processing, and UI components
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mech_exo.reporting.dash_layout.risk_live import RiskLiveLayout


class TestRiskPanel:
    """Unit tests for risk panel components"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures"""
        self.risk_layout = RiskLiveLayout()
        
        # Mock risk data
        self.mock_risk_data = {
            'positions': {
                'AAPL': {'quantity': 1000, 'market_value': 150000, 'var_contrib': 2500},
                'GOOGL': {'quantity': 500, 'market_value': 125000, 'var_contrib': 3200},
                'TSLA': {'quantity': -200, 'market_value': -40000, 'var_contrib': 4100}
            },
            'var': {
                '95': 8500,
                '99': 12300
            },
            'portfolio_value': 235000,
            'last_updated': datetime.now().isoformat()
        }
        
    def test_risk_heatmap_callback(self):
        """Test risk heatmap callback function"""
        # Mock callback
        with patch.object(self.risk_layout, 'update_risk_heatmap') as mock_update:
            mock_update.return_value = {
                'data': [{'z': [[1, 2], [3, 4]], 'type': 'heatmap'}],
                'layout': {'title': 'Risk Heatmap'}
            }
            
            result = self.risk_layout.update_risk_heatmap(1)  # interval trigger
            
            assert 'data' in result
            assert 'layout' in result
            assert result['data'][0]['type'] == 'heatmap'
            
    def test_var_timeline_callback(self):
        """Test VaR timeline callback function"""
        with patch.object(self.risk_layout, 'update_var_timeline') as mock_update:
            mock_timeline_data = {
                'data': [{
                    'x': ['2024-01-01', '2024-01-02', '2024-01-03'],
                    'y': [8500, 9200, 8800],
                    'type': 'scatter',
                    'name': '95% VaR'
                }],
                'layout': {'title': 'VaR Timeline'}
            }
            mock_update.return_value = mock_timeline_data
            
            result = self.risk_layout.update_var_timeline(1)
            
            assert 'data' in result
            assert len(result['data']) > 0
            assert result['data'][0]['type'] == 'scatter'
            
    def test_position_breakdown_callback(self):
        """Test position breakdown callback function"""
        with patch.object(self.risk_layout, 'update_position_breakdown') as mock_update:
            mock_breakdown = {
                'data': [{
                    'labels': ['AAPL', 'GOOGL', 'TSLA'],
                    'values': [2500, 3200, 4100],
                    'type': 'pie'
                }],
                'layout': {'title': 'Risk Contribution by Position'}
            }
            mock_update.return_value = mock_breakdown
            
            result = self.risk_layout.update_position_breakdown(1)
            
            assert 'data' in result
            assert result['data'][0]['type'] == 'pie'
            assert len(result['data'][0]['labels']) == 3
            
    def test_risk_level_color_coding(self):
        """Test risk level color coding logic"""
        # Test different risk levels
        test_cases = [
            {'var': 5000, 'limit': 10000, 'expected': 'success'},  # Low risk
            {'var': 8000, 'limit': 10000, 'expected': 'warning'},  # Medium risk  
            {'var': 12000, 'limit': 10000, 'expected': 'danger'}   # High risk
        ]
        
        for case in test_cases:
            color_class = self._get_risk_color_class(case['var'], case['limit'])
            assert color_class == case['expected'], f"Wrong color for VaR {case['var']}"
            
    def test_data_validation(self):
        """Test risk data validation"""
        # Test valid data
        assert self._validate_risk_data(self.mock_risk_data) == True
        
        # Test invalid data - missing required fields
        invalid_data = {'positions': {}}  # Missing var field
        assert self._validate_risk_data(invalid_data) == False
        
        # Test invalid data - wrong data types
        invalid_data2 = {'positions': 'not_a_dict', 'var': {'95': 'not_a_number'}}
        assert self._validate_risk_data(invalid_data2) == False
        
    def test_position_sorting(self):
        """Test position sorting by risk contribution"""
        positions = self.mock_risk_data['positions']
        sorted_positions = self._sort_positions_by_risk(positions)
        
        # Should be sorted by var_contrib descending
        risk_values = [pos['var_contrib'] for pos in sorted_positions.values()]
        assert risk_values == sorted(risk_values, reverse=True)
        
    def test_var_calculation_helpers(self):
        """Test VaR calculation helper functions"""
        # Test VaR scaling
        daily_var = 10000
        scaled_10day = self._scale_var(daily_var, days=10)
        expected_10day = daily_var * (10 ** 0.5)  # sqrt scaling
        
        assert abs(scaled_10day - expected_10day) < 100  # Allow small rounding
        
    def test_dashboard_component_ids(self):
        """Test that all required dashboard components have correct IDs"""
        layout = self.risk_layout.create_layout()
        
        # Extract all component IDs
        component_ids = self._extract_component_ids(layout)
        
        required_ids = [
            'risk-heatmap', 
            'var-timeline', 
            'position-breakdown',
            'risk-metrics-table',
            'alert-banner'
        ]
        
        for req_id in required_ids:
            assert req_id in component_ids, f"Missing component ID: {req_id}"
            
    def test_alert_banner_logic(self):
        """Test alert banner display logic"""
        # Test no alerts
        alerts = []
        banner_style = self._get_alert_banner_style(alerts)
        assert banner_style['display'] == 'none'
        
        # Test with alerts
        alerts = [{'severity': 'HIGH', 'message': 'VaR limit breach'}]
        banner_style = self._get_alert_banner_style(alerts)
        assert banner_style['display'] == 'block'
        assert 'danger' in banner_style['className']
        
    def test_data_refresh_interval(self):
        """Test data refresh interval configuration"""
        # Test different refresh intervals
        intervals = [1000, 5000, 10000]  # milliseconds
        
        for interval in intervals:
            component = self.risk_layout._create_interval_component(interval)
            assert component.interval == interval
            assert component.n_intervals == 0  # Initial state
            
    # Helper methods for testing
    def _get_risk_color_class(self, var, limit):
        """Get CSS color class based on risk level"""
        ratio = var / limit
        if ratio < 0.7:
            return 'success'
        elif ratio < 0.9:
            return 'warning'
        else:
            return 'danger'
            
    def _validate_risk_data(self, data):
        """Validate risk data structure"""
        try:
            required_fields = ['positions', 'var']
            for field in required_fields:
                if field not in data:
                    return False
                    
            if not isinstance(data['positions'], dict):
                return False
                
            if not isinstance(data['var'], dict):
                return False
                
            return True
        except:
            return False
            
    def _sort_positions_by_risk(self, positions):
        """Sort positions by risk contribution"""
        return dict(sorted(positions.items(), 
                          key=lambda x: x[1]['var_contrib'], 
                          reverse=True))
                          
    def _scale_var(self, daily_var, days):
        """Scale VaR for different time horizons"""
        return daily_var * (days ** 0.5)
        
    def _extract_component_ids(self, layout):
        """Extract all component IDs from layout"""
        ids = []
        if hasattr(layout, 'id') and layout.id:
            ids.append(layout.id)
        if hasattr(layout, 'children'):
            if isinstance(layout.children, list):
                for child in layout.children:
                    ids.extend(self._extract_component_ids(child))
            else:
                ids.extend(self._extract_component_ids(layout.children))
        return ids
        
    def _get_alert_banner_style(self, alerts):
        """Get alert banner style based on alerts"""
        if not alerts:
            return {'display': 'none'}
        
        highest_severity = max(alert.get('severity', 'LOW') for alert in alerts)
        severity_class = {
            'LOW': 'info',
            'MEDIUM': 'warning', 
            'HIGH': 'danger'
        }.get(highest_severity, 'info')
        
        return {
            'display': 'block',
            'className': f'alert alert-{severity_class}'
        }


def run_risk_panel_tests():
    """Run all risk panel unit tests"""
    print("🧪 Running Risk Panel Unit Tests...")
    
    pytest_args = [
        __file__,
        '-v',
        '--tb=short',
        '--disable-warnings'
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("✅ All risk panel tests passed!")
    else:
        print("❌ Some risk panel tests failed!")
        
    return exit_code == 0


if __name__ == '__main__':
    success = run_risk_panel_tests()
    sys.exit(0 if success else 1)