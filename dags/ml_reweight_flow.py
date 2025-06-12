"""
Prefect flow for automatic ML weight adjustment based on performance metrics.

This flow runs weekly to dynamically adjust ML influence based on 
real-time performance comparison between ML-enhanced and baseline strategies.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

logger = logging.getLogger(__name__)


@task(name="fetch_sharpe_metrics", description="Fetch latest 30-day Sharpe ratios for comparison")
def fetch_sharpe_metrics(lookback_days: int = 30) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetch baseline and ML Sharpe ratios for weight adjustment decision.
    
    Args:
        lookback_days: Number of days to look back for Sharpe calculation
        
    Returns:
        Tuple of (baseline_sharpe, ml_sharpe) or (None, None) if unavailable
    """
    logger = get_run_logger()
    
    try:
        from mech_exo.scoring.weight_utils import get_baseline_and_ml_sharpe
        
        baseline_sharpe, ml_sharpe = get_baseline_and_ml_sharpe(days=lookback_days)
        
        if baseline_sharpe is not None and ml_sharpe is not None:
            logger.info(f"📊 Fetched Sharpe ratios - Baseline: {baseline_sharpe:.3f}, ML: {ml_sharpe:.3f}")
            logger.info(f"📈 Performance delta: {ml_sharpe - baseline_sharpe:+.3f}")
        else:
            logger.warning("⚠️ Unable to fetch Sharpe ratios - insufficient data")
        
        return baseline_sharpe, ml_sharpe
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch Sharpe metrics: {e}")
        return None, None


@task(name="auto_adjust_ml_weight", 
      description="Automatically adjust ML weight based on performance comparison",
      retries=2,
      retry_delay_seconds=30)
def auto_adjust_ml_weight(baseline_sharpe: Optional[float], 
                         ml_sharpe: Optional[float],
                         dry_run: bool = False) -> Dict[str, any]:
    """
    Automatically adjust ML weight based on Sharpe ratio comparison.
    
    Args:
        baseline_sharpe: Baseline strategy 30-day Sharpe ratio
        ml_sharpe: ML-enhanced strategy 30-day Sharpe ratio
        dry_run: If True, only simulate the adjustment
        
    Returns:
        Dictionary with adjustment results
    """
    logger = get_run_logger()
    
    try:
        from mech_exo.scoring.weight_utils import (
            get_current_ml_weight,
            compute_new_weight,
            update_ml_weight_config,
            log_weight_change
        )
        
        # Check for dry-run environment variable
        env_dry_run = os.getenv('ML_REWEIGHT_DRY_RUN', 'false').lower() == 'true'
        effective_dry_run = dry_run or env_dry_run
        
        if effective_dry_run:
            logger.info("🔍 DRY RUN MODE - No changes will be made")
        
        # Validate inputs
        if baseline_sharpe is None or ml_sharpe is None:
            logger.warning("⚠️ Missing Sharpe ratios - cannot adjust weight")
            return {
                'success': False,
                'error': 'Missing Sharpe ratio data',
                'changed': False,
                'dry_run': effective_dry_run
            }
        
        # Get current weight
        current_weight = get_current_ml_weight()
        logger.info(f"⚖️ Current ML weight: {current_weight:.3f}")
        
        # Compute new weight using Day 2 algorithm
        new_weight, rule = compute_new_weight(
            baseline_sharpe=baseline_sharpe,
            ml_sharpe=ml_sharpe,
            current_w=current_weight
        )
        
        # Check if weight changed
        weight_changed = abs(new_weight - current_weight) > 0.001
        sharpe_diff = ml_sharpe - baseline_sharpe
        
        result = {
            'success': True,
            'current_weight': current_weight,
            'new_weight': new_weight,
            'changed': weight_changed,
            'adjustment_rule': rule,
            'baseline_sharpe': baseline_sharpe,
            'ml_sharpe': ml_sharpe,
            'sharpe_diff': sharpe_diff,
            'dry_run': effective_dry_run
        }
        
        if weight_changed:
            change_direction = "↗️" if new_weight > current_weight else "↘️"
            logger.info(f"🎯 Weight adjustment triggered: {current_weight:.3f} {change_direction} {new_weight:.3f}")
            logger.info(f"🔧 Rule: {rule}")
            logger.info(f"📊 Sharpe delta: {sharpe_diff:+.3f}")
            
            if not effective_dry_run:
                # Update YAML configuration
                config_success = update_ml_weight_config(new_weight)
                
                if config_success:
                    logger.info("✅ Configuration file updated")
                    
                    # Log the change to history table
                    log_success = log_weight_change(
                        old_weight=current_weight,
                        new_weight=new_weight,
                        adjustment_rule=rule,
                        ml_sharpe=ml_sharpe,
                        baseline_sharpe=baseline_sharpe,
                        notes=f"Prefect auto-adjustment: {sharpe_diff:+.3f} Sharpe diff"
                    )
                    
                    if log_success:
                        logger.info("✅ Weight change logged to history")
                    else:
                        logger.warning("⚠️ Failed to log weight change")
                    
                    result.update({
                        'config_updated': True,
                        'change_logged': log_success
                    })
                else:
                    logger.error("❌ Failed to update configuration file")
                    result.update({
                        'success': False,
                        'error': 'Config update failed',
                        'config_updated': False,
                        'change_logged': False
                    })
            else:
                logger.info("🔍 DRY RUN: Would update config and log change")
                result.update({
                    'config_updated': False,
                    'change_logged': False
                })
        else:
            logger.info(f"ℹ️ No weight adjustment needed ({rule})")
            result.update({
                'config_updated': False,
                'change_logged': False
            })
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Weight adjustment failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'changed': False,
            'dry_run': effective_dry_run
        }


@task(name="promote_weight_yaml",
      description="Commit and push weight changes to Git (optional)",
      retries=1)
def promote_weight_yaml(adjustment_result: Dict[str, any]) -> bool:
    """
    Optionally commit and push weight changes to Git repository.
    
    Args:
        adjustment_result: Result from auto_adjust_ml_weight task
        
    Returns:
        Boolean indicating Git operations success
    """
    logger = get_run_logger()
    
    try:
        # Check if Git auto-push is enabled
        git_auto_push = os.getenv('GIT_AUTO_PUSH', 'false').lower() == 'true'
        dry_run = adjustment_result.get('dry_run', False)
        weight_changed = adjustment_result.get('changed', False)
        config_updated = adjustment_result.get('config_updated', False)
        
        if not git_auto_push:
            logger.info("ℹ️ Git auto-push disabled (GIT_AUTO_PUSH != true)")
            return True
        
        if dry_run:
            logger.info("🔍 DRY RUN: Would commit and push weight changes")
            return True
        
        if not weight_changed or not config_updated:
            logger.info("ℹ️ No weight changes to commit")
            return True
        
        # Import GitPython
        try:
            import git
        except ImportError:
            logger.warning("⚠️ GitPython not available - skipping Git operations")
            return True
        
        # Git operations
        repo_path = Path(__file__).parent.parent  # Project root
        repo = git.Repo(repo_path)
        
        old_weight = adjustment_result['current_weight']
        new_weight = adjustment_result['new_weight']
        rule = adjustment_result['adjustment_rule']
        
        # Stage the factors.yml file
        repo.index.add(['config/factors.yml'])
        
        # Create commit message
        commit_msg = f"Auto-adjust ML weight {old_weight:.2f} → {new_weight:.2f}\n\n"
        commit_msg += f"Rule: {rule}\n"
        commit_msg += f"Sharpe diff: {adjustment_result['sharpe_diff']:+.3f}\n"
        commit_msg += f"Generated by Prefect ml_reweight_flow"
        
        # Set Git author
        git_author_name = os.getenv('GIT_AUTHOR_NAME', 'MechExoBot')
        git_author_email = os.getenv('GIT_AUTHOR_EMAIL', 'bot@mechexo.local')
        
        # Create commit
        repo.index.commit(
            commit_msg,
            author=git.Actor(git_author_name, git_author_email)
        )
        
        logger.info("✅ Weight change committed to Git")
        
        # Push to remote (if configured)
        try:
            origin = repo.remotes.origin
            origin.push()
            logger.info("✅ Changes pushed to remote repository")
        except Exception as push_error:
            logger.warning(f"⚠️ Failed to push to remote: {push_error}")
            # Don't fail the task if push fails
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Git operations failed: {e}")
        
        # Attempt rollback if YAML was updated but Git failed
        if adjustment_result.get('config_updated'):
            try:
                import git
                repo_path = Path(__file__).parent.parent
                repo = git.Repo(repo_path)
                repo.git.reset('--hard', 'HEAD')
                logger.info("🔄 Rolled back YAML changes due to Git failure")
            except Exception as rollback_error:
                logger.error(f"❌ Rollback also failed: {rollback_error}")
        
        return False


@task(name="notify_weight_change",
      description="Send Telegram notification for ML weight changes",
      retries=1)
def notify_weight_change(adjustment_result: Dict[str, any]) -> bool:
    """
    Send Telegram notification when ML weight changes.
    
    Args:
        adjustment_result: Result from auto_adjust_ml_weight task
        
    Returns:
        Boolean indicating notification success
    """
    logger = get_run_logger()
    
    try:
        from datetime import datetime
        from mech_exo.utils.alerts import TelegramAlerter
        from mech_exo.utils.config import ConfigManager
        
        # Check if notification should be sent
        weight_changed = adjustment_result.get('changed', False)
        dry_run = adjustment_result.get('dry_run', False)
        success = adjustment_result.get('success', False)
        
        if not success:
            logger.info("ℹ️ Adjustment not successful - skipping notification")
            return True
        
        if not weight_changed:
            logger.info("ℹ️ Weight unchanged - skipping notification")
            return True
        
        # Check weekend disable setting
        config_manager = ConfigManager()
        try:
            reweight_config = config_manager.load_config('reweight')
            weekend_disable = reweight_config.get('telegram_disable_on_weekend', True)
            
            # Check if it's weekend (Saturday=5, Sunday=6)
            if weekend_disable and datetime.now().weekday() >= 5:
                logger.info("📅 Weekend notification disabled - skipping Telegram alert")
                return True
        except:
            # If config doesn't exist, continue with notification
            pass
        
        # Load Telegram configuration
        try:
            telegram_config = config_manager.load_config('alerts').get('telegram', {})
            if not telegram_config.get('enabled', False):
                logger.info("📱 Telegram alerts disabled - skipping notification")
                return True
        except:
            logger.warning("⚠️ Could not load Telegram config - attempting with env vars")
            telegram_config = {
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID')
            }
            
            if not telegram_config['bot_token'] or not telegram_config['chat_id']:
                logger.warning("⚠️ Telegram credentials not available - skipping notification")
                return True
        
        # Create TelegramAlerter
        alerter = TelegramAlerter(telegram_config)
        
        # Extract adjustment details
        old_weight = adjustment_result['current_weight']
        new_weight = adjustment_result['new_weight']
        baseline_sharpe = adjustment_result['baseline_sharpe']
        ml_sharpe = adjustment_result['ml_sharpe']
        adjustment_rule = adjustment_result['adjustment_rule']
        
        # Send weight change notification
        success = alerter.send_weight_change(
            old_w=old_weight,
            new_w=new_weight,
            sharpe_ml=ml_sharpe,
            sharpe_base=baseline_sharpe,
            adjustment_rule=adjustment_rule,
            dry_run=dry_run
        )
        
        if success:
            logger.info(f"📱 Weight change notification sent: {old_weight:.2f} → {new_weight:.2f}")
        else:
            logger.warning("⚠️ Failed to send weight change notification")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Notification task failed: {e}")
        # Don't fail the flow if notification fails
        return False


@flow(name="ml-weight-rebalance",
      description="Weekly ML weight adjustment based on performance metrics",
      task_runner=SequentialTaskRunner())
def ml_reweight_flow(dry_run: bool = False, lookback_days: int = 30) -> Dict[str, any]:
    """
    Main flow for automatic ML weight adjustment.
    
    Args:
        dry_run: If True, only simulate adjustments without making changes
        lookback_days: Number of days to look back for Sharpe calculation
        
    Returns:
        Flow execution summary
    """
    logger = get_run_logger()
    
    logger.info("🚀 Starting ML weight rebalancing flow")
    logger.info(f"🔍 Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    logger.info(f"📅 Lookback period: {lookback_days} days")
    
    # Step 1: Fetch performance metrics
    baseline_sharpe, ml_sharpe = fetch_sharpe_metrics(lookback_days)
    
    # Step 2: Auto-adjust ML weight
    adjustment_result = auto_adjust_ml_weight(baseline_sharpe, ml_sharpe, dry_run)
    
    # Step 3: Send Telegram notification (conditional)
    notification_success = True
    if adjustment_result.get('success') and adjustment_result.get('changed'):
        notification_success = notify_weight_change(adjustment_result)
    
    # Step 4: Promote changes to Git (conditional)
    git_success = True
    if adjustment_result.get('success') and adjustment_result.get('changed'):
        git_success = promote_weight_yaml(adjustment_result)
    
    # Flow summary
    summary = {
        'flow_success': adjustment_result.get('success', False),
        'weight_changed': adjustment_result.get('changed', False),
        'notification_success': notification_success,
        'git_success': git_success,
        'dry_run': dry_run,
        'execution_time': datetime.now().isoformat()
    }
    
    # Add metrics if available
    if baseline_sharpe is not None and ml_sharpe is not None:
        summary.update({
            'baseline_sharpe': baseline_sharpe,
            'ml_sharpe': ml_sharpe,
            'sharpe_diff': ml_sharpe - baseline_sharpe
        })
    
    # Add weight info if adjustment was attempted
    if adjustment_result.get('success'):
        summary.update({
            'old_weight': adjustment_result.get('current_weight'),
            'new_weight': adjustment_result.get('new_weight'),
            'adjustment_rule': adjustment_result.get('adjustment_rule')
        })
    
    # Log final summary
    if summary['flow_success']:
        if summary['weight_changed']:
            logger.info(f"✅ Flow completed: Weight adjusted {summary['old_weight']:.3f} → {summary['new_weight']:.3f}")
        else:
            logger.info("✅ Flow completed: No weight adjustment needed")
    else:
        logger.error("❌ Flow failed: See task logs for details")
    
    return summary


# Deployment configuration for scheduling
def create_deployment():
    """Create Prefect deployment for weekly ML weight rebalancing"""
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule
    
    deployment = Deployment.build_from_flow(
        flow=ml_reweight_flow,
        name="ml-weight-weekly-rebalance",
        description="Weekly automatic ML weight adjustment based on performance metrics",
        tags=["ml", "weight-adjustment", "automated"],
        schedule=CronSchedule(cron="55 9 * * 1", timezone="UTC"),  # Every Monday 09:55 UTC
        parameters={
            "dry_run": False,
            "lookback_days": 30
        },
        work_pool_name="default-agent-pool"
    )
    
    return deployment


# Manual execution for testing
if __name__ == "__main__":
    import sys
    
    # Check for dry-run argument
    dry_run = "--dry-run" in sys.argv
    
    print(f"🚀 Running ML reweight flow (dry_run={dry_run})")
    
    # Run the flow
    result = ml_reweight_flow(dry_run=dry_run)
    
    print(f"📊 Flow result: {result}")
    
    if result.get('flow_success'):
        sys.exit(0)
    else:
        sys.exit(1)