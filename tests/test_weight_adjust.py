"""
Unit tests for ML weight adjustment algorithm
Tests Day 2 functionality: compute_new_weight and edge cases
"""

import pytest
import math
from mech_exo.scoring.weight_utils import (
    compute_new_weight,
    auto_adjust_ml_weight,
    validate_weight_bounds
)


class TestComputeNewWeight:
    """Test cases for compute_new_weight function"""
    
    def test_ml_outperforms_increase_weight(self):
        """Test weight increase when ML outperforms baseline"""
        # ML Sharpe = 1.20, Baseline = 1.00, delta = +0.20 (> +0.10 threshold)
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=1.20,
            current_w=0.30
        )
        
        assert new_weight == pytest.approx(0.35, rel=1e-3)
        assert rule == "ML_OUTPERFORM_BASELINE"
    
    def test_ml_underperforms_decrease_weight(self):
        """Test weight decrease when ML underperforms baseline"""
        # ML Sharpe = 0.85, Baseline = 1.00, delta = -0.15 (< -0.05 threshold)
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=0.85,
            current_w=0.30
        )
        
        assert new_weight == pytest.approx(0.25, rel=1e-3)
        assert rule == "ML_UNDERPERFORM_BASELINE"
    
    def test_performance_within_band_no_change(self):
        """Test no change when performance within acceptable band"""
        # ML Sharpe = 1.03, Baseline = 1.00, delta = +0.03 (within band)
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=1.03,
            current_w=0.30
        )
        
        assert new_weight == pytest.approx(0.30, rel=1e-3)
        assert rule == "PERFORMANCE_WITHIN_BAND"
    
    def test_upper_bound_cap(self):
        """Test weight capped at upper bound"""
        # Starting at 0.48, increase should cap at 0.50
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=1.20,  # Trigger increase
            current_w=0.48,
            upper=0.50
        )
        
        assert new_weight == pytest.approx(0.50, rel=1e-3)
        assert rule == "ML_OUTPERFORM_BASELINE"
    
    def test_lower_bound_floor(self):
        """Test weight floored at lower bound"""
        # Starting at 0.02, decrease should floor at 0.00
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=0.80,  # Trigger decrease
            current_w=0.02,
            lower=0.00
        )
        
        assert new_weight == pytest.approx(0.00, rel=1e-3)
        assert rule == "ML_UNDERPERFORM_BASELINE"
    
    def test_already_at_upper_limit(self):
        """Test no change when already at upper limit"""
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=1.20,  # Would trigger increase
            current_w=0.50,  # Already at max
            upper=0.50
        )
        
        assert new_weight == pytest.approx(0.50, rel=1e-3)
        assert rule == "ML_OUTPERFORM_BASELINE"
    
    def test_already_at_lower_limit(self):
        """Test no change when already at lower limit"""
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=0.80,  # Would trigger decrease
            current_w=0.00,  # Already at min
            lower=0.00
        )
        
        assert new_weight == pytest.approx(0.00, rel=1e-3)
        assert rule == "ML_UNDERPERFORM_BASELINE"
    
    def test_nan_sharpe_values(self):
        """Test handling of NaN Sharpe values"""
        # Test with NaN baseline
        new_weight, rule = compute_new_weight(
            baseline_sharpe=float('nan'),
            ml_sharpe=1.20,
            current_w=0.30
        )
        
        assert new_weight == pytest.approx(0.30, rel=1e-3)
        assert rule == "INVALID_SHARPE_VALUES"
        
        # Test with NaN ML Sharpe
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=float('nan'),
            current_w=0.30
        )
        
        assert new_weight == pytest.approx(0.30, rel=1e-3)
        assert rule == "INVALID_SHARPE_VALUES"
    
    def test_none_sharpe_values(self):
        """Test handling of None Sharpe values"""
        new_weight, rule = compute_new_weight(
            baseline_sharpe=None,
            ml_sharpe=1.20,
            current_w=0.30
        )
        
        assert new_weight == pytest.approx(0.30, rel=1e-3)
        assert rule == "INVALID_SHARPE_VALUES"
    
    def test_current_weight_clamping(self):
        """Test clamping of current weight to valid range"""
        # Current weight above upper bound
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=1.03,  # No change scenario
            current_w=0.60,  # Above upper bound
            upper=0.50
        )
        
        assert new_weight == pytest.approx(0.50, rel=1e-3)
        assert rule == "PERFORMANCE_WITHIN_BAND"
        
        # Current weight below lower bound
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=1.03,  # No change scenario
            current_w=-0.10,  # Below lower bound
            lower=0.00
        )
        
        assert new_weight == pytest.approx(0.00, rel=1e-3)
        assert rule == "PERFORMANCE_WITHIN_BAND"
    
    def test_custom_thresholds(self):
        """Test custom up/down thresholds"""
        # Custom thresholds: up=0.15, down=-0.10
        
        # Delta +0.12 should not trigger increase (< 0.15)
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=1.12,
            current_w=0.30,
            up_thresh=0.15,
            down_thresh=-0.10
        )
        
        assert new_weight == pytest.approx(0.30, rel=1e-3)
        assert rule == "PERFORMANCE_WITHIN_BAND"
        
        # Delta -0.12 should trigger decrease (< -0.10)
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=0.88,
            current_w=0.30,
            up_thresh=0.15,
            down_thresh=-0.10
        )
        
        assert new_weight == pytest.approx(0.25, rel=1e-3)
        assert rule == "ML_UNDERPERFORM_BASELINE"
    
    def test_custom_step_size(self):
        """Test custom step size"""
        # Custom step size of 0.10
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=1.20,  # Trigger increase
            current_w=0.30,
            step=0.10
        )
        
        assert new_weight == pytest.approx(0.40, rel=1e-3)
        assert rule == "ML_OUTPERFORM_BASELINE"
    
    def test_boundary_thresholds(self):
        """Test exact boundary threshold values"""
        # Exact up threshold (+0.10)
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=1.10,  # Exactly +0.10
            current_w=0.30
        )
        
        assert new_weight == pytest.approx(0.35, rel=1e-3)
        assert rule == "ML_OUTPERFORM_BASELINE"
        
        # Exact down threshold (-0.05)
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=0.95,  # Exactly -0.05
            current_w=0.30
        )
        
        assert new_weight == pytest.approx(0.25, rel=1e-3)
        assert rule == "ML_UNDERPERFORM_BASELINE"
    
    def test_weight_rounding(self):
        """Test weight rounding to 2 decimal places"""
        # Result should be rounded to avoid tiny YAML diffs
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=1.20,
            current_w=0.333333  # Floating point precision
        )
        
        # Should round to 2 decimal places
        assert new_weight == pytest.approx(0.38, rel=1e-3)  # 0.333333 + 0.05 = 0.383333 → 0.38


class TestValidationEdgeCases:
    """Test edge cases and validation"""
    
    def test_extreme_sharpe_values(self):
        """Test handling of extreme Sharpe values"""
        # Very high ML Sharpe
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=5.00,  # Extremely high
            current_w=0.30
        )
        
        assert new_weight == pytest.approx(0.35, rel=1e-3)
        assert rule == "ML_OUTPERFORM_BASELINE"
        
        # Very negative baseline Sharpe
        new_weight, rule = compute_new_weight(
            baseline_sharpe=-2.00,
            ml_sharpe=0.50,
            current_w=0.30
        )
        
        # Delta = 0.50 - (-2.00) = +2.50 (> 0.10)
        assert new_weight == pytest.approx(0.35, rel=1e-3)
        assert rule == "ML_OUTPERFORM_BASELINE"
    
    def test_zero_sharpe_values(self):
        """Test handling of zero Sharpe values"""
        new_weight, rule = compute_new_weight(
            baseline_sharpe=0.00,
            ml_sharpe=0.00,
            current_w=0.30
        )
        
        # Delta = 0.00 (within band)
        assert new_weight == pytest.approx(0.30, rel=1e-3)
        assert rule == "PERFORMANCE_WITHIN_BAND"
    
    def test_very_small_differences(self):
        """Test very small Sharpe differences"""
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.000000,
            ml_sharpe=1.000001,  # Tiny difference
            current_w=0.30
        )
        
        # Delta = +0.000001 (within band)
        assert new_weight == pytest.approx(0.30, rel=1e-3)
        assert rule == "PERFORMANCE_WITHIN_BAND"


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    def test_bull_market_scenario(self):
        """Test weight adjustment in bull market scenario"""
        # ML performs well in trending market
        scenarios = [
            (1.20, 1.40, 0.30, 0.35, "ML_OUTPERFORM_BASELINE"),  # Strong ML performance
            (1.15, 1.23, 0.35, 0.35, "PERFORMANCE_WITHIN_BAND"),  # Marginal difference
            (1.30, 1.20, 0.35, 0.30, "ML_UNDERPERFORM_BASELINE")  # ML struggles
        ]
        
        for baseline, ml, current, expected, expected_rule in scenarios:
            new_weight, rule = compute_new_weight(baseline, ml, current)
            assert new_weight == pytest.approx(expected, rel=1e-3)
            assert rule == expected_rule
    
    def test_bear_market_scenario(self):
        """Test weight adjustment in bear market scenario"""
        # Lower absolute Sharpe ratios
        scenarios = [
            (0.20, 0.35, 0.25, 0.30, "ML_OUTPERFORM_BASELINE"),  # ML helps in downtrend
            (-0.10, -0.05, 0.30, 0.30, "PERFORMANCE_WITHIN_BAND"),  # Both negative
            (0.10, -0.10, 0.30, 0.25, "ML_UNDERPERFORM_BASELINE")  # ML hurts performance
        ]
        
        for baseline, ml, current, expected, expected_rule in scenarios:
            new_weight, rule = compute_new_weight(baseline, ml, current)
            assert new_weight == pytest.approx(expected, rel=1e-3)
            assert rule == expected_rule
    
    def test_progressive_weight_changes(self):
        """Test progressive weight changes over time"""
        # Simulate multiple adjustment cycles
        current_weight = 0.30
        
        # Week 1: ML outperforms
        current_weight, rule = compute_new_weight(1.00, 1.15, current_weight)
        assert current_weight == pytest.approx(0.35, rel=1e-3)
        
        # Week 2: Continues to outperform
        current_weight, rule = compute_new_weight(1.05, 1.20, current_weight)
        assert current_weight == pytest.approx(0.40, rel=1e-3)
        
        # Week 3: Reaches cap
        current_weight, rule = compute_new_weight(1.10, 1.25, current_weight)
        assert current_weight == pytest.approx(0.45, rel=1e-3)
        
        # Week 4: At cap
        current_weight, rule = compute_new_weight(1.10, 1.25, current_weight)
        assert current_weight == pytest.approx(0.50, rel=1e-3)  # Capped


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])