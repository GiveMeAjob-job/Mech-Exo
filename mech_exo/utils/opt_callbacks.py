"""
Optuna optimization callbacks for progress tracking and notifications
"""

import logging
import os
from datetime import datetime
from typing import Optional
import optuna

logger = logging.getLogger(__name__)


class OptunaDashboardCallback:
    """
    Callback for Optuna optimization with progress tracking and optional notifications
    """
    
    def __init__(self, progress_interval: int = 10, notify_progress: bool = False):
        """
        Initialize callback
        
        Args:
            progress_interval: Print progress every N trials
            notify_progress: Send Telegram notifications on progress
        """
        self.progress_interval = progress_interval
        self.notify_progress = notify_progress
        self.start_time = None
        self.best_value = float('-inf')
        self.trial_count = 0
        
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Callback function called after each trial
        
        Args:
            study: Optuna study object
            trial: Completed trial object
        """
        try:
            if self.start_time is None:
                self.start_time = datetime.now()
            
            self.trial_count += 1
            
            # Update best value
            if trial.value is not None and trial.value > self.best_value:
                self.best_value = trial.value
                self._log_improvement(study, trial)
            
            # Print progress at intervals
            if self.trial_count % self.progress_interval == 0:
                self._print_progress(study, trial)
                
                # Send Telegram notification if enabled
                if self.notify_progress:
                    self._send_progress_notification(study, trial)
                    
        except Exception as e:
            logger.error(f"Callback error: {e}")
    
    def _log_improvement(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Log when a new best trial is found"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        logger.info(f"🎯 New best trial #{trial.number}: {trial.value:.4f}")
        
        # Log key metrics from trial attributes
        if trial.user_attrs:
            max_dd = trial.user_attrs.get('max_drawdown', 'N/A')
            violations = trial.user_attrs.get('constraint_violations', 'N/A')
            logger.info(f"   Max DD: {max_dd}, Violations: {violations}")
    
    def _print_progress(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Print progress summary"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.trial_count / elapsed if elapsed > 0 else 0
        
        print(f"📊 Progress: {self.trial_count} trials completed")
        print(f"   • Best Sharpe ratio: {self.best_value:.4f}")
        print(f"   • Rate: {rate:.1f} trials/sec")
        print(f"   • Elapsed: {elapsed:.0f}s")
        
        # Show best trial parameters (abbreviated)
        if study.best_trial:
            best_params = study.best_trial.params
            fund_weights = {k: v for k, v in best_params.items() if k.startswith('weight_') and 'pe_ratio' in k or 'equity' in k}
            if fund_weights:
                print(f"   • Key weights: {fund_weights}")
    
    def _send_progress_notification(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Send Telegram progress notification"""
        try:
            # Check for dry-run mode
            if os.getenv('TELEGRAM_DRY_RUN', 'false').lower() == 'true':
                logger.info("TELEGRAM_DRY_RUN=true - logging progress message instead of sending")
                
                message = self._create_progress_message(study, trial)
                logger.info(f"Dry-run Telegram progress message:\n{message}")
                return
            
            # Send actual notification if not in dry-run mode
            from mech_exo.utils.alerts import TelegramAlerter
            
            telegram_config = {
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID')
            }
            
            if telegram_config['bot_token'] and telegram_config['chat_id']:
                alerter = TelegramAlerter(telegram_config)
                message = self._create_progress_message(study, trial)
                
                success = alerter.send_message(message)
                if success:
                    logger.info(f"📱 Progress notification sent")
                else:
                    logger.warning(f"Failed to send progress notification")
            else:
                logger.info(f"Telegram credentials not set - skipping notification")
                
        except Exception as e:
            logger.warning(f"Failed to send progress notification: {e}")
    
    def _create_progress_message(self, study: optuna.Study, trial: optuna.Trial) -> str:
        """Create formatted progress message for Telegram"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        message = f"🔬 *Optuna Progress Update*\n\n"
        message += f"📊 *Trials*: {self.trial_count}\n"
        message += f"🎯 *Best Sharpe*: `{self.best_value:.4f}`\n"
        message += f"⏱️ *Elapsed*: {elapsed/60:.1f} minutes\n"
        
        if study.best_trial and study.best_trial.user_attrs:
            max_dd = study.best_trial.user_attrs.get('max_drawdown', 0)
            if isinstance(max_dd, (int, float)):
                message += f"📉 *Best Max DD*: {max_dd:.1%}\n"
        
        message += f"\n🔄 Optimization continuing..."
        
        return message


def create_optuna_callback(progress_interval: int = 10, 
                          notify_progress: bool = False) -> OptunaDashboardCallback:
    """
    Create Optuna callback with specified settings
    
    Args:
        progress_interval: Print progress every N trials
        notify_progress: Send Telegram notifications
        
    Returns:
        Configured callback instance
    """
    return OptunaDashboardCallback(progress_interval, notify_progress)


def interrupt_handler(study: optuna.Study, output_file: str) -> None:
    """
    Handle KeyboardInterrupt by saving current best YAML before exit
    
    Args:
        study: Optuna study object
        output_file: YAML output file path
    """
    try:
        if study.best_trial:
            logger.info(f"🛑 Interrupted! Saving current best trial to {output_file}")
            
            from pathlib import Path
            import yaml
            from datetime import date
            
            # Create best trial export
            best_trial = study.best_trial
            
            # Extract factor weights and hyperparameters
            factor_weights = {}
            other_params = {}
            
            for key, value in best_trial.params.items():
                if key.startswith('weight_'):
                    factor_name = key.replace('weight_', '')
                    factor_weights[factor_name] = value
                else:
                    other_params[key] = value
            
            # Create YAML structure
            export_data = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'optimization_status': 'interrupted',
                    'study_name': study.study_name,
                    'best_trial_number': best_trial.number,
                    'best_sharpe_ratio': study.best_value,
                    'trials_completed': len(study.trials),
                    'interrupted_at': datetime.now().isoformat()
                },
                'factors': {
                    'fundamental': {
                        name: {'weight': factor_weights.get(name, 0), 'direction': 'higher_better'}
                        for name in ['pe_ratio', 'return_on_equity', 'revenue_growth', 'earnings_growth']
                    },
                    'technical': {
                        name: {'weight': factor_weights.get(name, 0), 'direction': 'higher_better'}
                        for name in ['rsi_14', 'momentum_12_1', 'volatility_ratio']
                    },
                    'sentiment': {
                        name: {'weight': factor_weights.get(name, 0), 'direction': 'higher_better'}
                        for name in ['news_sentiment', 'analyst_revisions']
                    }
                },
                'hyperparameters': other_params
            }
            
            # Write to YAML file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False, indent=2, sort_keys=False)
            
            print(f"✅ Best trial saved to {output_path}")
            
        else:
            logger.warning("No best trial available to save")
            
    except Exception as e:
        logger.error(f"Failed to save interrupted trial: {e}")


def send_optimization_summary_with_chart(study: optuna.Study, 
                                        export_info: dict,
                                        chart_path: Optional[str] = None) -> bool:
    """
    Send Telegram summary with optional PNG chart attachment
    
    Args:
        study: Completed Optuna study
        export_info: Export information from optimization flow
        chart_path: Optional path to PNG chart file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check for dry-run mode
        if os.getenv('TELEGRAM_DRY_RUN', 'false').lower() == 'true':
            logger.info("TELEGRAM_DRY_RUN=true - logging summary instead of sending")
            
            message = _create_summary_message(study, export_info)
            logger.info(f"Dry-run Telegram summary:\n{message}")
            
            if chart_path and os.path.exists(chart_path):
                logger.info(f"Would send chart: {chart_path} ({os.path.getsize(chart_path)} bytes)")
            
            return True
        
        from mech_exo.utils.alerts import TelegramAlerter
        
        # Get Telegram credentials
        telegram_config = {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID')
        }
        
        if not telegram_config['bot_token'] or not telegram_config['chat_id']:
            logger.warning("Telegram credentials not available - skipping summary")
            return False
        
        alerter = TelegramAlerter(telegram_config)
        
        # Send text summary
        message = _create_summary_message(study, export_info)
        success = alerter.send_message(message)
        
        if not success:
            logger.error("Failed to send summary message")
            return False
        
        # Send chart if available and < 1MB
        if chart_path and os.path.exists(chart_path):
            file_size_mb = os.path.getsize(chart_path) / (1024 * 1024)
            
            if file_size_mb < 1.0:  # Telegram limit enforcement
                chart_success = alerter.send_document(
                    chart_path,
                    caption=f"📊 Optimization Progress Chart"
                )
                
                if chart_success:
                    logger.info(f"📊 Chart sent successfully: {chart_path}")
                else:
                    logger.warning(f"Failed to send chart: {chart_path}")
            else:
                logger.info(f"Chart too large ({file_size_mb:.2f}MB) - providing link instead")
                # Could add a follow-up message with a link here
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to send optimization summary: {e}")
        return False


def _create_summary_message(study: optuna.Study, export_info: dict) -> str:
    """Create formatted summary message for Telegram"""
    
    def escape_markdown_v2(text: str) -> str:
        """Escape MarkdownV2 special characters"""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
    
    # Get study metrics
    best_trial = study.best_trial
    total_trials = len(study.trials)
    best_sharpe = study.best_value
    
    # Calculate completion time
    if hasattr(study, '_creation_time') and hasattr(best_trial, 'datetime_complete'):
        elapsed_time = best_trial.datetime_complete - study._creation_time
        elapsed_minutes = elapsed_time.total_seconds() / 60
    else:
        elapsed_minutes = 0  # Fallback
    
    # Get constraints status
    constraints_satisfied = best_trial.user_attrs.get('constraints_satisfied', False) if best_trial.user_attrs else False
    violations = best_trial.user_attrs.get('constraint_violations', 0) if best_trial.user_attrs else 0
    max_dd = best_trial.user_attrs.get('max_drawdown', 0) if best_trial.user_attrs else 0
    
    # Create message
    message = f"🎯 *Optuna Optimization Complete*\n\n"
    message += f"📊 *Study*: {escape_markdown_v2(study.study_name)}\n"
    message += f"🔬 *Trials*: {total_trials}\n"
    message += f"⏱️ *Duration*: {elapsed_minutes:.1f} minutes\n\n"
    
    message += f"📈 *Best Results*:\n"
    message += f"• Sharpe Ratio: `{best_sharpe:.4f}`\n"
    message += f"• Max Drawdown: {max_dd:.1%}\n"
    message += f"• Violations: {violations}\n"
    message += f"• Constraints: {'✅ Satisfied' if constraints_satisfied else '❌ Violated'}\n\n"
    
    # Add export info
    if export_info:
        yaml_filename = os.path.basename(export_info.get('yaml_path', 'N/A'))
        message += f"📄 *Export*: `{escape_markdown_v2(yaml_filename)}`\n"
        message += f"💾 *Size*: {export_info.get('file_size_mb', 0):.2f} MB\n\n"
    
    message += f"🚀 *Ready for deployment and validation*"
    
    return message


if __name__ == "__main__":
    # Test the callback
    print("🧪 Testing OptunaDashboardCallback...")
    
    callback = create_optuna_callback(progress_interval=2, notify_progress=True)
    print(f"✅ Callback created: interval={callback.progress_interval}, notify={callback.notify_progress}")
    
    print("🎯 Callback test completed!")