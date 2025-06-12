"""
Unit tests for ML weight badge helper functions
Tests Day 5 functionality: Dashboard badge and health endpoint integration
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, date


class TestMLWeightBadgeHelpers:
    """Test ML weight badge helper functions"""
    
    def test_get_current_ml_weight_info_success(self):
        """Test successful weight info retrieval with all components"""
        from mech_exo.reporting.query import get_current_ml_weight_info
        
        # Mock the weight utility functions
        with patch('mech_exo.scoring.weight_utils.get_current_ml_weight', return_value=0.35):
            with patch('mech_exo.scoring.weight_utils.get_latest_weight_change') as mock_latest:
                mock_latest.return_value = {
                    'date': date(2025, 6, 8),
                    'old_weight': 0.30,
                    'new_weight': 0.35,
                    'adjustment_rule': 'ML_OUTPERFORM_BASELINE',
                    'ml_sharpe': 1.15,
                    'baseline_sharpe': 1.00
                }
                
                info = get_current_ml_weight_info()
                
                # Verify all required fields are present
                assert 'current_weight' in info
                assert 'weight_percentage' in info
                assert 'badge_color' in info
                assert 'badge_status' in info
                assert 'tooltip_info' in info
                assert 'latest_change' in info
                
                # Verify specific values
                assert info['current_weight'] == 0.35
                assert info['weight_percentage'] == "35.0%"
                assert info['badge_color'] == "success"  # 0.35 is in green range (≥ 0.25)
                assert info['badge_status'] == "high"
                
                # Verify tooltip contains key information
                tooltip = info['tooltip_info']
                assert "Current ML Weight: 35.0%" in tooltip
                assert "Last change: 2025-06-08" in tooltip
                assert "Reason: ML_OUTPERFORM_BASELINE" in tooltip
    
    def test_badge_color_logic_green_range(self):
        """Test badge color logic for green range (≥ 0.25)"""
        from mech_exo.reporting.query import get_current_ml_weight_info
        
        test_weights = [0.25, 0.30, 0.35, 0.45, 0.50]
        
        for weight in test_weights:
            with patch('mech_exo.scoring.weight_utils.get_current_ml_weight', return_value=weight):
                with patch('mech_exo.scoring.weight_utils.get_latest_weight_change', return_value=None):
                    info = get_current_ml_weight_info()
                    
                    assert info['badge_color'] == "success", f"Weight {weight} should be green"
                    assert info['badge_status'] == "high", f"Weight {weight} should be high status"
    
    def test_badge_color_logic_yellow_range(self):
        """Test badge color logic for yellow range (0.05-0.25)"""
        from mech_exo.reporting.query import get_current_ml_weight_info
        
        test_weights = [0.05, 0.10, 0.15, 0.20, 0.24]
        
        for weight in test_weights:
            with patch('mech_exo.scoring.weight_utils.get_current_ml_weight', return_value=weight):
                with patch('mech_exo.scoring.weight_utils.get_latest_weight_change', return_value=None):
                    info = get_current_ml_weight_info()
                    
                    assert info['badge_color'] == "warning", f"Weight {weight} should be yellow"
                    assert info['badge_status'] == "medium", f"Weight {weight} should be medium status"
    
    def test_badge_color_logic_grey_range(self):
        """Test badge color logic for grey range (< 0.05)"""
        from mech_exo.reporting.query import get_current_ml_weight_info
        
        test_weights = [0.0, 0.01, 0.03, 0.04, 0.049]
        
        for weight in test_weights:
            with patch('mech_exo.scoring.weight_utils.get_current_ml_weight', return_value=weight):
                with patch('mech_exo.scoring.weight_utils.get_latest_weight_change', return_value=None):
                    info = get_current_ml_weight_info()
                    
                    assert info['badge_color'] == "secondary", f"Weight {weight} should be grey"
                    assert info['badge_status'] == "low", f"Weight {weight} should be low status"
    
    def test_get_current_ml_weight_info_no_change_history(self):
        """Test weight info when no change history is available"""
        from mech_exo.reporting.query import get_current_ml_weight_info
        
        with patch('mech_exo.scoring.weight_utils.get_current_ml_weight', return_value=0.25):
            with patch('mech_exo.scoring.weight_utils.get_latest_weight_change', return_value=None):
                info = get_current_ml_weight_info()
                
                assert info['current_weight'] == 0.25
                assert info['weight_percentage'] == "25.0%"
                assert info['latest_change'] is None
                assert "No change history available" in info['tooltip_info']
    
    def test_get_current_ml_weight_info_error_handling(self):
        """Test error handling when weight utilities fail"""
        from mech_exo.reporting.query import get_current_ml_weight_info
        
        with patch('mech_exo.scoring.weight_utils.get_current_ml_weight', side_effect=Exception("Config error")):
            info = get_current_ml_weight_info()
            
            # Should return default fallback values
            assert info['current_weight'] == 0.30
            assert info['weight_percentage'] == "30.0%"
            assert info['badge_color'] == "warning"
            assert info['badge_status'] == "medium"
            assert "Error loading weight info" in info['tooltip_info']
    
    def test_tooltip_info_formatting_with_datetime(self):
        """Test tooltip formatting with datetime objects"""
        from mech_exo.reporting.query import get_current_ml_weight_info
        
        with patch('mech_exo.scoring.weight_utils.get_current_ml_weight', return_value=0.40):
            with patch('mech_exo.scoring.weight_utils.get_latest_weight_change') as mock_latest:
                # Mock with datetime object (has strftime method)
                mock_latest.return_value = {
                    'date': datetime(2025, 6, 10, 15, 30, 0),
                    'adjustment_rule': 'PERFORMANCE_WITHIN_BAND'
                }
                
                info = get_current_ml_weight_info()
                
                # Should format datetime properly
                assert "Last change: 2025-06-10" in info['tooltip_info']
                assert "Reason: PERFORMANCE_WITHIN_BAND" in info['tooltip_info']
    
    def test_tooltip_info_formatting_with_string_date(self):
        """Test tooltip formatting with string date objects"""
        from mech_exo.reporting.query import get_current_ml_weight_info
        
        with patch('mech_exo.scoring.weight_utils.get_current_ml_weight', return_value=0.15):
            with patch('mech_exo.scoring.weight_utils.get_latest_weight_change') as mock_latest:
                # Mock with string date (no strftime method)
                mock_latest.return_value = {
                    'date': "2025-06-09T10:30:00",  # String with extra time info
                    'adjustment_rule': 'ML_UNDERPERFORM_BASELINE'
                }
                
                info = get_current_ml_weight_info()
                
                # Should truncate to first 10 characters for YYYY-MM-DD
                assert "Last change: 2025-06-09" in info['tooltip_info']
                assert "Reason: ML_UNDERPERFORM_BASELINE" in info['tooltip_info']
    
    def test_percentage_formatting_precision(self):
        """Test percentage formatting precision for different weights"""
        from mech_exo.reporting.query import get_current_ml_weight_info
        
        test_cases = [
            (0.123, "12.3%"),
            (0.1234, "12.3%"),  # Should round to 1 decimal
            (0.1567, "15.7%"),  # Should round to 1 decimal
            (0.0, "0.0%"),
            (0.5, "50.0%"),
            (0.33333, "33.3%")
        ]
        
        for weight, expected_percentage in test_cases:
            with patch('mech_exo.scoring.weight_utils.get_current_ml_weight', return_value=weight):
                with patch('mech_exo.scoring.weight_utils.get_latest_weight_change', return_value=None):
                    info = get_current_ml_weight_info()
                    assert info['weight_percentage'] == expected_percentage, \
                        f"Weight {weight} should format as {expected_percentage}, got {info['weight_percentage']}"


class TestMLWeightHealthEndpoint:
    """Test ML weight exposure in health endpoint"""
    
    def test_health_endpoint_includes_ml_weight(self):
        """Test that health endpoint includes ML weight in JSON response"""
        from mech_exo.reporting.dash_app import create_app
        
        app = create_app()
        
        with app.test_client() as client:
            with patch('mech_exo.reporting.query.get_health_data') as mock_health:
                with patch('mech_exo.reporting.query.get_current_ml_weight_info') as mock_weight:
                    # Mock health data
                    mock_health.return_value = {
                        'system_status': 'operational',
                        'risk_ok': True,
                        'last_updated': '2025-06-10T12:00:00Z',
                        'fills_today': 5
                    }
                    
                    # Mock weight info
                    mock_weight.return_value = {
                        'current_weight': 0.40,
                        'weight_percentage': "40.0%",
                        'badge_color': "success"
                    }
                    
                    # Make JSON request to health endpoint
                    response = client.get('/healthz', headers={'Accept': 'application/json'})
                    
                    assert response.status_code == 200
                    data = response.get_json()
                    
                    # Verify ML weight is included
                    assert 'ml_weight' in data
                    assert data['ml_weight'] == 0.40
                    
                    # Verify other standard fields are still present
                    assert data['status'] == 'operational'
                    assert data['risk_ok'] is True
                    assert data['fills_today'] == 5
                    assert data['ml_signal'] is True
    
    def test_health_endpoint_ml_weight_fallback(self):
        """Test health endpoint fallback when ML weight retrieval fails"""
        from mech_exo.reporting.dash_app import create_app
        
        app = create_app()
        
        with app.test_client() as client:
            with patch('mech_exo.reporting.query.get_health_data') as mock_health:
                with patch('mech_exo.reporting.query.get_current_ml_weight_info', side_effect=Exception("Config error")):
                    # Mock health data
                    mock_health.return_value = {
                        'system_status': 'operational',
                        'risk_ok': True,
                        'last_updated': '2025-06-10T12:00:00Z',
                        'fills_today': 0
                    }
                    
                    # Make JSON request to health endpoint
                    response = client.get('/healthz', headers={'Accept': 'application/json'})
                    
                    assert response.status_code == 200
                    data = response.get_json()
                    
                    # Should use fallback ML weight
                    assert 'ml_weight' in data
                    assert data['ml_weight'] == 0.30  # Default fallback
    
    def test_health_endpoint_text_response_unchanged(self):
        """Test that text response from health endpoint is unchanged"""
        from mech_exo.reporting.dash_app import create_app
        
        app = create_app()
        
        with app.test_client() as client:
            with patch('mech_exo.reporting.query.get_health_data') as mock_health:
                mock_health.return_value = {
                    'system_status': 'operational',
                    'risk_ok': True,
                    'last_updated': '2025-06-10T12:00:00Z',
                    'fills_today': 0
                }
                
                # Make text request (no Accept header or non-JSON)
                response = client.get('/healthz')
                
                assert response.status_code == 200
                assert response.get_data(as_text=True) == 'OK'


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])