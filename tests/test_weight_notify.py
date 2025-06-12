"""
Unit tests for ML weight change notifications
Tests Day 4 functionality: Telegram notifications and Prefect integration
"""

import os
import tempfile
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, call

from mech_exo.utils.alerts import TelegramAlerter


class TestTelegramWeightNotifications:
    """Test Telegram weight change notification functionality"""
    
    def setup_method(self):
        """Setup test configuration"""
        self.telegram_config = {
            'bot_token': 'test_token_123',
            'chat_id': 'test_chat_456'
        }
        self.alerter = TelegramAlerter(self.telegram_config)
    
    def test_send_weight_change_increase(self):
        """Test weight increase notification"""
        with patch.object(self.alerter, 'send_message') as mock_send:
            mock_send.return_value = True
            
            success = self.alerter.send_weight_change(
                old_w=0.30,
                new_w=0.35,
                sharpe_ml=1.15,
                sharpe_base=1.00,
                adjustment_rule="ML_OUTPERFORM_BASELINE",
                dry_run=False
            )
            
            assert success is True
            mock_send.assert_called_once()
            
            # Check message content
            call_args = mock_send.call_args
            message = call_args[0][0]
            
            assert "⚖️ *ML Weight Auto\\-Adjusted*" in message
            assert "0.30 ↗️ 0.35" in message
            assert "+0.150" in message  # Delta Sharpe
            assert "ML\\_OUTPERFORM\\_BASELINE" in message  # Escaped version
    
    def test_send_weight_change_decrease(self):
        """Test weight decrease notification"""
        with patch.object(self.alerter, 'send_message') as mock_send:
            mock_send.return_value = True
            
            success = self.alerter.send_weight_change(
                old_w=0.35,
                new_w=0.30,
                sharpe_ml=0.85,
                sharpe_base=1.00,
                adjustment_rule="ML_UNDERPERFORM_BASELINE",
                dry_run=False
            )
            
            assert success is True
            
            # Check message content
            call_args = mock_send.call_args
            message = call_args[0][0]
            
            assert "0.35 ↘️ 0.30" in message
            assert "-0.150" in message  # Negative delta
            assert "ML\\_UNDERPERFORM\\_BASELINE" in message
    
    def test_send_weight_change_no_change(self):
        """Test notification when weight stays the same"""
        with patch.object(self.alerter, 'send_message') as mock_send:
            mock_send.return_value = True
            
            success = self.alerter.send_weight_change(
                old_w=0.30,
                new_w=0.30,
                sharpe_ml=1.03,
                sharpe_base=1.00,
                adjustment_rule="PERFORMANCE_WITHIN_BAND",
                dry_run=False
            )
            
            assert success is True
            
            # Check message content
            call_args = mock_send.call_args
            message = call_args[0][0]
            
            assert "0.30 ➡️ 0.30" in message
            assert "+0.030" in message
            assert "PERFORMANCE\\_WITHIN\\_BAND" in message
    
    def test_dry_run_mode(self):
        """Test dry-run mode logging instead of sending"""
        with patch.object(self.alerter, 'send_message') as mock_send:
            with patch('mech_exo.utils.alerts.logger') as mock_logger:
                
                success = self.alerter.send_weight_change(
                    old_w=0.30,
                    new_w=0.35,
                    sharpe_ml=1.15,
                    sharpe_base=1.00,
                    adjustment_rule="ML_OUTPERFORM_BASELINE",
                    dry_run=True
                )
                
                assert success is True
                mock_send.assert_not_called()  # Should not send actual message
                
                # Check that dry-run was logged
                mock_logger.info.assert_any_call("TELEGRAM_DRY_RUN=true - logging weight change message")
    
    def test_environment_dry_run_override(self):
        """Test TELEGRAM_DRY_RUN environment variable override"""
        with patch.dict(os.environ, {'TELEGRAM_DRY_RUN': 'true'}):
            with patch.object(self.alerter, 'send_message') as mock_send:
                with patch('mech_exo.utils.alerts.logger') as mock_logger:
                    
                    success = self.alerter.send_weight_change(
                        old_w=0.30,
                        new_w=0.35,
                        sharpe_ml=1.15,
                        sharpe_base=1.00,
                        adjustment_rule="ML_OUTPERFORM_BASELINE",
                        dry_run=False  # dry_run=False but env overrides
                    )
                    
                    assert success is True
                    mock_send.assert_not_called()
                    mock_logger.info.assert_any_call("TELEGRAM_DRY_RUN=true - logging weight change message")
    
    def test_git_hash_inclusion(self):
        """Test Git commit hash is included in message"""
        with patch.object(self.alerter, 'send_message') as mock_send:
            with patch('subprocess.check_output') as mock_git:
                mock_send.return_value = True
                mock_git.return_value = b'abc123\n'
                
                success = self.alerter.send_weight_change(
                    old_w=0.30,
                    new_w=0.35,
                    sharpe_ml=1.15,
                    sharpe_base=1.00,
                    adjustment_rule="ML_OUTPERFORM_BASELINE",
                    dry_run=False
                )
                
                assert success is True
                
                # Check message includes Git hash
                call_args = mock_send.call_args
                message = call_args[0][0]
                assert "commit `abc123`" in message
    
    def test_git_hash_failure_graceful(self):
        """Test graceful handling when Git hash cannot be retrieved"""
        with patch.object(self.alerter, 'send_message') as mock_send:
            with patch('subprocess.check_output') as mock_git:
                mock_send.return_value = True
                mock_git.side_effect = Exception("Git not available")
                
                success = self.alerter.send_weight_change(
                    old_w=0.30,
                    new_w=0.35,
                    sharpe_ml=1.15,
                    sharpe_base=1.00,
                    adjustment_rule="ML_OUTPERFORM_BASELINE",
                    dry_run=False
                )
                
                assert success is True  # Should still succeed without Git hash
                
                # Check message doesn't include Git hash
                call_args = mock_send.call_args
                message = call_args[0][0]
                assert "commit" not in message
    
    def test_markdown_escaping(self):
        """Test proper Markdown character escaping"""
        with patch.object(self.alerter, 'send_message') as mock_send:
            mock_send.return_value = True
            
            # Test with rule containing special characters
            rule_with_special_chars = "TEST_RULE-WITH.SPECIAL_CHARS"
            
            success = self.alerter.send_weight_change(
                old_w=0.30,
                new_w=0.35,
                sharpe_ml=1.15,
                sharpe_base=1.00,
                adjustment_rule=rule_with_special_chars,
                dry_run=False
            )
            
            assert success is True
            
            # Check message has escaped characters
            call_args = mock_send.call_args
            message = call_args[0][0]
            
            # Periods and underscores should be escaped
            assert "TEST\\_RULE\\-WITH\\.SPECIAL\\_CHARS" in message
    
    def test_message_send_failure(self):
        """Test handling of message send failure"""
        with patch.object(self.alerter, 'send_message') as mock_send:
            mock_send.return_value = False  # Simulate send failure
            
            success = self.alerter.send_weight_change(
                old_w=0.30,
                new_w=0.35,
                sharpe_ml=1.15,
                sharpe_base=1.00,
                adjustment_rule="ML_OUTPERFORM_BASELINE",
                dry_run=False
            )
            
            assert success is False
    
    def test_exception_handling(self):
        """Test exception handling in send_weight_change"""
        with patch.object(self.alerter, 'send_message') as mock_send:
            mock_send.side_effect = Exception("Telegram API error")
            
            success = self.alerter.send_weight_change(
                old_w=0.30,
                new_w=0.35,
                sharpe_ml=1.15,
                sharpe_base=1.00,
                adjustment_rule="ML_OUTPERFORM_BASELINE",
                dry_run=False
            )
            
            assert success is False


class TestPrefectNotificationTask:
    """Test Prefect notification task integration"""
    
    def test_notify_weight_change_success(self):
        """Test successful notification task execution"""
        from dags.ml_reweight_flow import notify_weight_change
        
        adjustment_result = {
            'success': True,
            'changed': True,
            'current_weight': 0.30,
            'new_weight': 0.35,
            'baseline_sharpe': 1.00,
            'ml_sharpe': 1.15,
            'adjustment_rule': 'ML_OUTPERFORM_BASELINE',
            'dry_run': False
        }
        
        with patch('dags.ml_reweight_flow.TelegramAlerter') as mock_alerter_class:
            with patch('dags.ml_reweight_flow.ConfigManager') as mock_config:
                mock_alerter = MagicMock()
                mock_alerter_class.return_value = mock_alerter
                mock_alerter.send_weight_change.return_value = True
                
                # Mock config
                mock_config_manager = MagicMock()
                mock_config.return_value = mock_config_manager
                mock_config_manager.load_config.return_value = {
                    'telegram': {'enabled': True, 'bot_token': 'test', 'chat_id': 'test'}
                }
                
                success = notify_weight_change(adjustment_result)
                
                assert success is True
                mock_alerter.send_weight_change.assert_called_once_with(
                    old_w=0.30,
                    new_w=0.35,
                    sharpe_ml=1.15,
                    sharpe_base=1.00,
                    adjustment_rule='ML_OUTPERFORM_BASELINE',
                    dry_run=False
                )
    
    def test_notify_no_change_skipped(self):
        """Test notification skipped when no weight change"""
        from dags.ml_reweight_flow import notify_weight_change
        
        adjustment_result = {
            'success': True,
            'changed': False,  # No change
            'current_weight': 0.30,
            'new_weight': 0.30,
            'dry_run': False
        }
        
        with patch('dags.ml_reweight_flow.TelegramAlerter') as mock_alerter_class:
            mock_alerter = MagicMock()
            mock_alerter_class.return_value = mock_alerter
            
            success = notify_weight_change(adjustment_result)
            
            assert success is True
            mock_alerter.send_weight_change.assert_not_called()
    
    def test_notify_weekend_disabled(self):
        """Test weekend notification disable functionality"""
        from dags.ml_reweight_flow import notify_weight_change
        
        adjustment_result = {
            'success': True,
            'changed': True,
            'current_weight': 0.30,
            'new_weight': 0.35,
            'dry_run': False
        }
        
        # Mock Saturday (weekday=5)
        with patch('dags.ml_reweight_flow.datetime') as mock_datetime:
            mock_now = MagicMock()
            mock_now.weekday.return_value = 5  # Saturday
            mock_datetime.now.return_value = mock_now
            
            with patch('dags.ml_reweight_flow.ConfigManager') as mock_config:
                mock_config_manager = MagicMock()
                mock_config.return_value = mock_config_manager
                mock_config_manager.load_config.return_value = {
                    'telegram_disable_on_weekend': True
                }
                
                with patch('dags.ml_reweight_flow.TelegramAlerter') as mock_alerter_class:
                    mock_alerter = MagicMock()
                    mock_alerter_class.return_value = mock_alerter
                    
                    success = notify_weight_change(adjustment_result)
                    
                    assert success is True
                    mock_alerter.send_weight_change.assert_not_called()
    
    def test_notify_telegram_disabled(self):
        """Test notification skipped when Telegram disabled"""
        from dags.ml_reweight_flow import notify_weight_change
        
        adjustment_result = {
            'success': True,
            'changed': True,
            'dry_run': False
        }
        
        with patch('dags.ml_reweight_flow.ConfigManager') as mock_config:
            mock_config_manager = MagicMock()
            mock_config.return_value = mock_config_manager
            mock_config_manager.load_config.return_value = {
                'telegram': {'enabled': False}
            }
            
            with patch('dags.ml_reweight_flow.TelegramAlerter') as mock_alerter_class:
                mock_alerter = MagicMock()
                mock_alerter_class.return_value = mock_alerter
                
                success = notify_weight_change(adjustment_result)
                
                assert success is True
                mock_alerter.send_weight_change.assert_not_called()
    
    def test_notify_fallback_env_vars(self):
        """Test fallback to environment variables when config unavailable"""
        from dags.ml_reweight_flow import notify_weight_change
        
        adjustment_result = {
            'success': True,
            'changed': True,
            'current_weight': 0.30,
            'new_weight': 0.35,
            'baseline_sharpe': 1.00,
            'ml_sharpe': 1.15,
            'adjustment_rule': 'ML_OUTPERFORM_BASELINE',
            'dry_run': False
        }
        
        with patch.dict(os.environ, {
            'TELEGRAM_BOT_TOKEN': 'env_token',
            'TELEGRAM_CHAT_ID': 'env_chat'
        }):
            with patch('dags.ml_reweight_flow.ConfigManager') as mock_config:
                mock_config_manager = MagicMock()
                mock_config.return_value = mock_config_manager
                mock_config_manager.load_config.side_effect = Exception("Config not found")
                
                with patch('dags.ml_reweight_flow.TelegramAlerter') as mock_alerter_class:
                    mock_alerter = MagicMock()
                    mock_alerter_class.return_value = mock_alerter
                    mock_alerter.send_weight_change.return_value = True
                    
                    success = notify_weight_change(adjustment_result)
                    
                    assert success is True
                    
                    # Check TelegramAlerter was created with env vars
                    mock_alerter_class.assert_called_with({
                        'bot_token': 'env_token',
                        'chat_id': 'env_chat'
                    })


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])