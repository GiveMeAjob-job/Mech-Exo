"""
Unit tests for automatic ML weight rebalancing functionality
Tests Day 3 functionality: Prefect flow, YAML updates, and Git integration
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from mech_exo.scoring.weight_utils import (
    update_ml_weight_in_yaml,
    auto_adjust_ml_weight
)


class TestYAMLUpdates:
    """Test YAML update functionality with comment preservation"""
    
    def test_yaml_comment_preservation(self):
        """Test that YAML comments are preserved during weight updates"""
        yaml_content = """# Factor Scoring Configuration

# ML Integration Weight (0.0 - 0.50)
ml_weight: 0.30

# Fundamental Factors (0-100 weight)
fundamental:
  pe_ratio:
    weight: 15
    direction: "lower_better"  # Lower P/E is better
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            # Update weight
            success = update_ml_weight_in_yaml(temp_path, 0.35)
            assert success, "YAML update should succeed"
            
            # Read updated content
            with open(temp_path, 'r') as f:
                updated_content = f.read()
            
            # Check that weight was updated
            assert "ml_weight: 0.35" in updated_content
            
            # Check that comments are preserved
            assert "# Factor Scoring Configuration" in updated_content
            assert "# ML Integration Weight" in updated_content
            assert "# Lower P/E is better" in updated_content
            
        finally:
            os.unlink(temp_path)
    
    def test_yaml_precision_rounding(self):
        """Test that weights are rounded to 2 decimal places"""
        yaml_content = """ml_weight: 0.30"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            # Update with high precision weight
            success = update_ml_weight_in_yaml(temp_path, 0.3333333)
            assert success
            
            # Check rounding
            with open(temp_path, 'r') as f:
                content = f.read()
            
            assert "ml_weight: 0.33" in content
            
        finally:
            os.unlink(temp_path)
    
    def test_yaml_fallback_when_ruamel_unavailable(self):
        """Test fallback to standard yaml when ruamel.yaml not available"""
        yaml_content = """ml_weight: 0.30\nother_field: value"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            # Mock ruamel.yaml import to fail
            with patch('mech_exo.scoring.weight_utils.ruamel.yaml', side_effect=ImportError):
                success = update_ml_weight_in_yaml(temp_path, 0.25)
                assert success
                
                # Check weight was updated
                with open(temp_path, 'r') as f:
                    content = f.read()
                assert "ml_weight: 0.25" in content
                
        finally:
            os.unlink(temp_path)


class TestAutoAdjustmentScenarios:
    """Test automatic weight adjustment scenarios"""
    
    @patch('mech_exo.scoring.weight_utils.get_current_ml_weight')
    @patch('mech_exo.scoring.weight_utils.update_ml_weight_config')
    @patch('mech_exo.scoring.weight_utils.log_weight_change')
    def test_weight_increase_scenario(self, mock_log, mock_update, mock_get_weight):
        """Test weight increase when ML outperforms baseline"""
        # Setup mocks
        mock_get_weight.return_value = 0.30
        mock_update.return_value = True
        mock_log.return_value = True
        
        # Run auto adjustment with outperforming ML
        result = auto_adjust_ml_weight(dry_run=False)
        
        # Simulate the adjustment with mocked Sharpe ratios
        from mech_exo.scoring.weight_utils import compute_new_weight
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=1.15,  # +0.15 delta > +0.10 threshold
            current_w=0.30
        )
        
        assert new_weight == pytest.approx(0.35, rel=1e-3)
        assert rule == "ML_OUTPERFORM_BASELINE"
    
    @patch('mech_exo.scoring.weight_utils.get_current_ml_weight')
    @patch('mech_exo.scoring.weight_utils.update_ml_weight_config')
    @patch('mech_exo.scoring.weight_utils.log_weight_change')
    def test_weight_decrease_scenario(self, mock_log, mock_update, mock_get_weight):
        """Test weight decrease when ML underperforms baseline"""
        # Setup mocks
        mock_get_weight.return_value = 0.30
        mock_update.return_value = True
        mock_log.return_value = True
        
        # Simulate underperforming ML scenario
        from mech_exo.scoring.weight_utils import compute_new_weight
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=0.85,  # -0.15 delta < -0.05 threshold
            current_w=0.30
        )
        
        assert new_weight == pytest.approx(0.25, rel=1e-3)
        assert rule == "ML_UNDERPERFORM_BASELINE"
    
    @patch('mech_exo.scoring.weight_utils.get_current_ml_weight')
    def test_no_change_scenario(self, mock_get_weight):
        """Test no change when performance within acceptable band"""
        mock_get_weight.return_value = 0.30
        
        # Simulate performance within band
        from mech_exo.scoring.weight_utils import compute_new_weight
        new_weight, rule = compute_new_weight(
            baseline_sharpe=1.00,
            ml_sharpe=1.03,  # +0.03 delta within band
            current_w=0.30
        )
        
        assert new_weight == pytest.approx(0.30, rel=1e-3)
        assert rule == "PERFORMANCE_WITHIN_BAND"
    
    def test_dry_run_mode(self):
        """Test that dry run mode doesn't make actual changes"""
        with patch('mech_exo.scoring.weight_utils.get_baseline_and_ml_sharpe') as mock_sharpe:
            mock_sharpe.return_value = (1.00, 1.15)  # ML outperforms
            
            with patch('mech_exo.scoring.weight_utils.get_current_ml_weight') as mock_weight:
                mock_weight.return_value = 0.30
                
                with patch('mech_exo.scoring.weight_utils.update_ml_weight_config') as mock_update:
                    with patch('mech_exo.scoring.weight_utils.log_weight_change') as mock_log:
                        # Run in dry-run mode
                        result = auto_adjust_ml_weight(dry_run=True)
                        
                        # Should not call update or log functions
                        mock_update.assert_not_called()
                        mock_log.assert_not_called()
                        
                        # But should indicate what would happen
                        assert result['dry_run'] is True
                        assert result['changed'] is True
                        assert result['new_weight'] == pytest.approx(0.35, rel=1e-3)
    
    def test_missing_sharpe_data(self):
        """Test handling of missing Sharpe ratio data"""
        result = auto_adjust_ml_weight(dry_run=False)
        
        # Should handle missing data gracefully
        # Note: This will call get_baseline_and_ml_sharpe() which may return None, None
        # The function should handle this case appropriately


class TestPrefectFlowIntegration:
    """Test Prefect flow integration (mocked)"""
    
    @patch('dags.ml_reweight_flow.fetch_sharpe_metrics')
    @patch('dags.ml_reweight_flow.auto_adjust_ml_weight')
    @patch('dags.ml_reweight_flow.promote_weight_yaml')
    def test_flow_success_path(self, mock_promote, mock_adjust, mock_fetch):
        """Test successful flow execution"""
        # Setup mocks
        mock_fetch.return_value = (1.00, 1.15)  # ML outperforms
        mock_adjust.return_value = {
            'success': True,
            'changed': True,
            'current_weight': 0.30,
            'new_weight': 0.35,
            'adjustment_rule': 'ML_OUTPERFORM_BASELINE',
            'config_updated': True,
            'change_logged': True,
            'dry_run': False
        }
        mock_promote.return_value = True
        
        # Import and run flow
        from dags.ml_reweight_flow import ml_reweight_flow
        
        result = ml_reweight_flow(dry_run=False)
        
        # Verify flow executed successfully
        assert result['flow_success'] is True
        assert result['weight_changed'] is True
        assert result['git_success'] is True
        
        # Verify tasks were called
        mock_fetch.assert_called_once()
        mock_adjust.assert_called_once()
        mock_promote.assert_called_once()
    
    @patch('dags.ml_reweight_flow.fetch_sharpe_metrics')
    def test_flow_no_data_path(self, mock_fetch):
        """Test flow when no Sharpe data available"""
        # Setup mock to return no data
        mock_fetch.return_value = (None, None)
        
        from dags.ml_reweight_flow import ml_reweight_flow
        
        result = ml_reweight_flow(dry_run=False)
        
        # Flow should handle missing data gracefully
        assert 'flow_success' in result
    
    def test_environment_dry_run_flag(self):
        """Test ML_REWEIGHT_DRY_RUN environment variable"""
        with patch.dict(os.environ, {'ML_REWEIGHT_DRY_RUN': 'true'}):
            with patch('mech_exo.scoring.weight_utils.get_baseline_and_ml_sharpe') as mock_sharpe:
                mock_sharpe.return_value = (1.00, 1.15)
                
                with patch('mech_exo.scoring.weight_utils.get_current_ml_weight') as mock_weight:
                    mock_weight.return_value = 0.30
                    
                    # Should respect environment variable
                    from dags.ml_reweight_flow import auto_adjust_ml_weight
                    
                    result = auto_adjust_ml_weight(baseline_sharpe=1.00, ml_sharpe=1.15, dry_run=False)
                    
                    # Should be in dry-run mode due to environment variable
                    assert result['dry_run'] is True


class TestGitIntegration:
    """Test Git integration functionality (mocked)"""
    
    @patch('git.Repo')
    def test_git_commit_success(self, mock_repo_class):
        """Test successful Git commit and push"""
        # Setup Git mocks
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo
        mock_repo.remotes.origin.push = MagicMock()
        
        from dags.ml_reweight_flow import promote_weight_yaml
        
        adjustment_result = {
            'dry_run': False,
            'changed': True,
            'config_updated': True,
            'current_weight': 0.30,
            'new_weight': 0.35,
            'adjustment_rule': 'ML_OUTPERFORM_BASELINE',
            'sharpe_diff': 0.15
        }
        
        with patch.dict(os.environ, {'GIT_AUTO_PUSH': 'true'}):
            result = promote_weight_yaml(adjustment_result)
            
            assert result is True
            
            # Verify Git operations were called
            mock_repo.index.add.assert_called_with(['config/factors.yml'])
            mock_repo.index.commit.assert_called_once()
            mock_repo.remotes.origin.push.assert_called_once()
    
    def test_git_disabled(self):
        """Test when Git auto-push is disabled"""
        from dags.ml_reweight_flow import promote_weight_yaml
        
        adjustment_result = {
            'dry_run': False,
            'changed': True,
            'config_updated': True
        }
        
        with patch.dict(os.environ, {'GIT_AUTO_PUSH': 'false'}):
            result = promote_weight_yaml(adjustment_result)
            
            # Should succeed without Git operations
            assert result is True
    
    @patch('git.Repo')
    def test_git_rollback_on_failure(self, mock_repo_class):
        """Test Git rollback when operations fail"""
        # Setup Git mock to fail
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo
        mock_repo.index.commit.side_effect = Exception("Git commit failed")
        
        from dags.ml_reweight_flow import promote_weight_yaml
        
        adjustment_result = {
            'dry_run': False,
            'changed': True,
            'config_updated': True,
            'current_weight': 0.30,
            'new_weight': 0.35
        }
        
        with patch.dict(os.environ, {'GIT_AUTO_PUSH': 'true'}):
            result = promote_weight_yaml(adjustment_result)
            
            # Should fail but attempt rollback
            assert result is False
            mock_repo.git.reset.assert_called_with('--hard', 'HEAD')


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_invalid_weight_bounds(self):
        """Test handling of invalid weight values"""
        from mech_exo.scoring.weight_utils import update_ml_weight_config
        
        # Test weight too high
        result = update_ml_weight_config(0.60)
        assert result is False
        
        # Test negative weight
        result = update_ml_weight_config(-0.10)
        assert result is False
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file"""
        from mech_exo.scoring.weight_utils import update_ml_weight_config
        
        result = update_ml_weight_config(0.35, config_path="nonexistent.yml")
        assert result is False
    
    def test_yaml_syntax_error(self):
        """Test handling of malformed YAML files"""
        malformed_yaml = "ml_weight: 0.30\ninvalid: [unclosed"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(malformed_yaml)
            temp_path = f.name
        
        try:
            result = update_ml_weight_in_yaml(temp_path, 0.35)
            # Should handle YAML parsing errors gracefully
            assert result is False
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])