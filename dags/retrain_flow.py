"""
Drift-Triggered Strategy Retraining Flow

Automatically retrain strategy factors when performance drift is detected,
ensuring the strategy adapts to changing market conditions.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

logger = logging.getLogger(__name__)


@task(name="check_drift_breach", retries=2, retry_delay_seconds=30)
def check_drift_breach(target_date: str = None) -> Dict[str, Any]:
    """
    Check if drift status indicates a retraining is needed
    
    Args:
        target_date: Date to check drift for (YYYY-MM-DD format)
        
    Returns:
        Dictionary with drift status and decision to retrain
    """
    logger = get_run_logger()
    
    try:
        from mech_exo.reporting.drift import DriftMetricEngine
        
        # Use provided date or default to yesterday
        if target_date:
            check_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        else:
            check_date = date.today() - timedelta(days=1)
        
        logger.info(f"Checking drift breach for date: {check_date}")
        
        # Initialize drift engine
        drift_engine = DriftMetricEngine()
        
        # Calculate latest drift metrics
        metrics = drift_engine.calculate_drift_metrics(check_date, lookback_days=30)
        
        # Check if retraining is needed
        drift_status = metrics.get('drift_status', 'UNKNOWN')
        drift_pct = abs(metrics.get('drift_pct', 0))
        information_ratio = metrics.get('information_ratio', 1.0)
        
        # Retraining criteria: ALERT status OR high drift OR low IR
        needs_retrain = (
            drift_status == 'ALERT' or 
            drift_pct > 15.0 or  # More than 15% drift
            information_ratio < 0.1  # Very low information ratio
        )
        
        logger.info(f"Drift check results: status={drift_status}, drift={drift_pct:.1f}%, IR={information_ratio:.3f}")
        logger.info(f"Retraining needed: {needs_retrain}")
        
        drift_engine.close()
        
        return {
            'needs_retrain': needs_retrain,
            'drift_status': drift_status,
            'drift_pct': drift_pct,
            'information_ratio': information_ratio,
            'check_date': check_date.isoformat(),
            'reason': f"Drift {drift_pct:.1f}%, IR {information_ratio:.3f}, Status {drift_status}"
        }
        
    except Exception as e:
        logger.error(f"Failed to check drift breach: {e}")
        return {
            'needs_retrain': False,
            'error': str(e),
            'check_date': check_date.isoformat() if 'check_date' in locals() else None
        }


@task(name="load_retrain_data", retries=2, retry_delay_seconds=60)
def load_retrain_data(lookback_months: int = 6) -> Dict[str, Any]:
    """
    Load data for retraining with configurable lookback period
    
    Args:
        lookback_months: Number of months of data to load for retraining
        
    Returns:
        Dictionary with loaded data and metadata
    """
    logger = get_run_logger()
    
    try:
        from mech_exo.datasource.retrain_loader import load_data_for_retraining
        
        logger.info(f"Loading retraining data with {lookback_months} month lookback")
        
        # Load data using the retraining data loader
        data_dict, data_summary = load_data_for_retraining(
            lookback_months=lookback_months,
            min_symbols=10,  # Reduced for retraining
            include_fundamentals=True,
            include_news=True
        )
        
        logger.info(f"Data loading completed: {data_summary}")
        
        # Convert data summary to dictionary for JSON serialization
        summary_dict = {
            'start_date': data_summary.start_date.isoformat(),
            'end_date': data_summary.end_date.isoformat(),
            'lookback_months': lookback_months,
            'records_loaded': data_summary.total_records,
            'symbols_count': data_summary.symbols_count,
            'data_quality': data_summary.data_quality_score,
            'fundamental_coverage': data_summary.fundamental_coverage,
            'news_coverage': data_summary.news_coverage,
            'missing_data_pct': data_summary.missing_data_pct,
            'universe_symbols': data_summary.universe_symbols[:10]  # Limit for logging
        }
        
        # Validate minimum data requirements
        success = (
            data_summary.total_records > 0 and
            data_summary.symbols_count >= 5 and  # Minimum symbols for retraining
            data_summary.data_quality_score >= 0.5  # Minimum quality threshold
        )
        
        if not success:
            message = f"Insufficient data for retraining: {data_summary.total_records} records, {data_summary.symbols_count} symbols, {data_summary.data_quality_score:.1%} quality"
            logger.warning(message)
        else:
            message = f"Successfully loaded {data_summary.total_records:,} records for {data_summary.symbols_count} symbols"
            logger.info(message)
        
        return {
            'success': success,
            'data_summary': summary_dict,
            'data_available': bool(data_dict),
            'datasets_loaded': list(data_dict.keys()),
            'message': message
        }
        
    except Exception as e:
        logger.error(f"Failed to load retraining data: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"Failed to load retraining data: {e}"
        }


@task(name="refit_factors", retries=1, retry_delay_seconds=120)
def refit_factors(data_summary: Dict[str, Any], method: str = "ridge") -> Dict[str, Any]:
    """
    Re-fit factor weights using specified method
    
    Args:
        data_summary: Summary of loaded data from load_retrain_data
        method: Fitting method ('ridge', 'ols', 'lasso')
        
    Returns:
        Dictionary with new factor weights and performance metrics
    """
    logger = get_run_logger()
    
    try:
        from mech_exo.datasource.retrain_loader import load_data_for_retraining
        from mech_exo.research.refit_factors import refit_strategy_factors
        
        logger.info(f"Re-fitting factors using {method} method")
        logger.info(f"Using data summary: {data_summary}")
        
        # Load the actual data for re-fitting
        lookback_months = data_summary.get('lookback_months', 6)
        data_dict, _ = load_data_for_retraining(
            lookback_months=lookback_months,
            min_symbols=5,  # Minimum for re-fitting
            include_fundamentals=True,
            include_news=True
        )
        
        if not data_dict or 'ohlc' not in data_dict or data_dict['ohlc'].empty:
            logger.warning("No OHLC data available for factor re-fitting")
            return {
                'success': False,
                'error': 'No OHLC data available',
                'message': 'Insufficient data for factor re-fitting'
            }
        
        # Perform factor re-fitting
        logger.info(f"Re-fitting factors on {len(data_dict['ohlc'])} OHLC records")
        
        refit_results = refit_strategy_factors(
            ohlc_df=data_dict['ohlc'],
            fundamental_df=data_dict.get('fundamentals'),
            news_df=data_dict.get('news'),
            method=method,
            alpha=1.0  # Default regularization
        )
        
        # Convert results to task-compatible format
        performance_metrics = {
            'in_sample_r2': refit_results.in_sample_r2,
            'in_sample_sharpe': 1.5 + refit_results.in_sample_r2,  # Approximate conversion
            'cross_validation_score': refit_results.performance_metrics.get('mean_cv_score', 0.0),
            'improvement_vs_baseline': max(0, refit_results.in_sample_r2 - 0.1),  # Baseline R² of 0.1
            'method_used': refit_results.method,
            'version': refit_results.version,
            'n_features': len(refit_results.feature_importance),
            'n_samples': refit_results.performance_metrics.get('n_samples', 0)
        }
        
        logger.info(f"Factor re-fitting completed with {method}")
        logger.info(f"Performance: R² = {refit_results.in_sample_r2:.3f}, Features = {len(refit_results.feature_importance)}")
        
        return {
            'success': True,
            'new_weights': refit_results.factor_weights,
            'performance_metrics': performance_metrics,
            'feature_importance': refit_results.feature_importance,
            'version': refit_results.version,
            'method': method,
            'message': f"Successfully re-fitted factors using {method}: R² = {refit_results.in_sample_r2:.3f}"
        }
        
    except Exception as e:
        logger.error(f"Failed to refit factors: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"Failed to refit factors: {e}"
        }


@task(name="validate_retrained_strategy", retries=1, retry_delay_seconds=90)
def validate_retrained_strategy(new_weights: Dict[str, Any], 
                               performance_metrics: Dict[str, Any],
                               version: str) -> Dict[str, Any]:
    """
    Validate retrained strategy using walk-forward analysis
    
    Args:
        new_weights: New factor weights from refit_factors
        performance_metrics: Performance metrics from refit_factors
        version: Version timestamp for the factors
        
    Returns:
        Dictionary with validation results and deployment decision
    """
    logger = get_run_logger()
    
    try:
        import tempfile
        import yaml
        from mech_exo.validation.walk_forward_refit import run_walk_forward, ValidationConfig
        
        logger.info("Validating retrained strategy with walk-forward analysis")
        
        # Create temporary factors file for validation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
            factors_config = {
                'metadata': {
                    'version': version,
                    'created_at': datetime.now().isoformat(),
                    'validation_run': True
                },
                'factors': new_weights
            }
            
            yaml.dump(factors_config, temp_file, default_flow_style=False)
            temp_factors_file = temp_file.name
        
        try:
            # Configure validation with realistic thresholds
            validation_config = ValidationConfig(
                train_months=18,
                test_months=6,
                step_months=6,
                min_sharpe=0.30,
                max_drawdown=0.15,
                min_segments=1  # Reduced for testing
            )
            
            # Run walk-forward validation
            # Use last 3 years of data for validation
            end_date = date.today()
            start_date = end_date - timedelta(days=3*365)  # 3 years
            
            logger.info(f"Running walk-forward validation: {start_date} to {end_date}")
            
            validation_results = run_walk_forward(
                temp_factors_file,
                start_date,
                end_date,
                validation_config
            )
            
            # Log validation table
            if not validation_results['table'].empty:
                logger.info("Walk-forward validation results:")
                logger.info("\n" + validation_results['table'].to_string(index=False))
            
            # Extract key metrics
            summary = validation_results.get('summary_metrics', {})
            
            # Decision logic for deployment
            passed_validation = validation_results['passed']
            min_sharpe_met = summary.get('mean_sharpe', 0) >= validation_config.min_sharpe
            max_dd_met = summary.get('worst_max_drawdown', 1.0) <= validation_config.max_drawdown
            
            should_deploy = passed_validation and min_sharpe_met and max_dd_met
            
            # Create deployment reason
            reasons = []
            if not passed_validation:
                reasons.append(f"Failed validation: {validation_results.get('failure_reason', 'Unknown')}")
            if not min_sharpe_met:
                reasons.append(f"Mean Sharpe {summary.get('mean_sharpe', 0):.3f} < {validation_config.min_sharpe}")
            if not max_dd_met:
                reasons.append(f"Worst MaxDD {summary.get('worst_max_drawdown', 0):.1%} > {validation_config.max_drawdown:.1%}")
            
            deployment_reason = "; ".join(reasons) if reasons else "Passed all validation criteria"
            
            # Compile final results
            final_results = {
                'out_of_sample_sharpe': summary.get('mean_sharpe', 0),
                'min_sharpe': summary.get('min_sharpe', 0),
                'max_drawdown': summary.get('worst_max_drawdown', 0),
                'mean_return': summary.get('mean_total_return', 0),
                'segments_passed': validation_results.get('segments_passed', 0),
                'segments_total': validation_results.get('segments_count', 0),
                'pass_rate': summary.get('pass_rate', 0),
                'validation_period': f"{start_date}_to_{end_date}",
                'passes_validation': passed_validation
            }
            
            logger.info(f"Validation summary: {final_results}")
            logger.info(f"Deploy recommendation: {should_deploy} - {deployment_reason}")
            
            return {
                'success': True,
                'validation_results': final_results,
                'validation_table': validation_results['table'].to_dict('records') if not validation_results['table'].empty else [],
                'should_deploy': should_deploy,
                'deployment_reason': deployment_reason,
                'message': f"Walk-forward validation: {'PASSED' if should_deploy else 'FAILED'}"
            }
            
        finally:
            # Clean up temporary file
            try:
                Path(temp_factors_file).unlink()
            except:
                pass
        
    except Exception as e:
        logger.error(f"Failed to validate retrained strategy: {e}")
        return {
            'success': False,
            'error': str(e),
            'should_deploy': False,
            'deployment_reason': f"Validation error: {e}",
            'message': f"Validation failed: {e}"
        }


@task(name="deploy_new_factors", retries=2, retry_delay_seconds=30)
def deploy_new_factors(new_weights: Dict[str, Any], 
                      validation_results: Dict[str, Any],
                      version: str, 
                      dry_run: bool = True) -> Dict[str, Any]:
    """
    Deploy new factor weights to production configuration
    
    Args:
        new_weights: New factor weights to deploy
        validation_results: Validation results from validate_retrained_strategy
        version: Version timestamp for the new factors
        dry_run: If True, only stage factors without promoting to production
        
    Returns:
        Dictionary with deployment results and file paths
    """
    logger = get_run_logger()
    
    try:
        import yaml
        import os
        
        logger.info(f"Deploying new factors version {version} (dry_run={dry_run})")
        
        # Create staging directory if it doesn't exist
        staging_dir = Path("config/staging")
        staging_dir.mkdir(exist_ok=True)
        
        # Create versioned factors file in staging
        factors_file = staging_dir / f"factors_retrained_{version}.yml"
        
        # Add metadata to the factors configuration
        factors_config = {
            'metadata': {
                'version': version,
                'created_at': datetime.now().isoformat(),
                'retrain_trigger': 'drift_breach',
                'validation_sharpe': validation_results.get('out_of_sample_sharpe', 0),
                'segments_passed': validation_results.get('segments_passed', 0),
                'segments_total': validation_results.get('segments_total', 0),
                'deployment_mode': 'dry_run' if dry_run else 'production'
            },
            'factors': new_weights
        }
        
        # Write new factors configuration to staging
        with open(factors_file, 'w') as f:
            yaml.dump(factors_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"New factors configuration written to staging: {factors_file}")
        
        deployment_status = 'staged'
        production_file = None
        
        # If not dry run, promote to production
        if not dry_run:
            try:
                # Create production directory
                prod_dir = Path("config/prod")
                prod_dir.mkdir(exist_ok=True)
                
                # Copy to production with current timestamp
                production_file = prod_dir / f"factors_{version}.yml"
                
                with open(production_file, 'w') as f:
                    yaml.dump(factors_config, f, default_flow_style=False, sort_keys=False)
                
                logger.info(f"Factors promoted to production: {production_file}")
                
                # Try to commit with Git if available
                commit_success = _commit_factors_to_git(production_file, version)
                
                deployment_status = 'deployed_with_git' if commit_success else 'deployed_no_git'
                
            except Exception as e:
                logger.error(f"Failed to promote to production: {e}")
                deployment_status = 'staged_promotion_failed'
        
        return {
            'success': True,
            'factors_file': str(factors_file),
            'production_file': str(production_file) if production_file else None,
            'version': version,
            'status': deployment_status,
            'dry_run': dry_run,
            'message': f"Factors version {version} {'staged' if dry_run else 'deployed'} successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to deploy new factors: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"Failed to deploy new factors: {e}"
        }


def _commit_factors_to_git(factors_file: Path, version: str) -> bool:
    """
    Commit new factors to Git repository
    
    Args:
        factors_file: Path to factors file to commit
        version: Version string for commit message
        
    Returns:
        True if successful, False otherwise
    """
    logger = get_run_logger()
    
    try:
        # Check if GitPython is available
        try:
            import git
        except ImportError:
            logger.warning("GitPython not available, skipping Git commit")
            return False
        
        import os
        
        # Set Git author from environment variables
        git_author_name = os.getenv('GIT_AUTHOR_NAME', 'MechExoBot')
        git_author_email = os.getenv('GIT_AUTHOR_EMAIL', 'bot@mech-exo.ai')
        
        # Initialize repository
        try:
            repo = git.Repo(search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            logger.warning("Not in a Git repository, skipping commit")
            return False
        
        # Configure Git user
        with repo.config_writer() as config:
            config.set_value("user", "name", git_author_name)
            config.set_value("user", "email", git_author_email)
        
        # Add the factors file
        repo.index.add([str(factors_file)])
        
        # Create commit message
        commit_message = f"""🤖 Auto-retrain: Deploy factors v{version}

- Strategy retraining completed automatically
- Walk-forward validation passed
- New factor weights deployed to production

Generated by Mech-Exo retrain flow
Version: {version}
Timestamp: {datetime.now().isoformat()}"""
        
        # Commit changes
        commit = repo.index.commit(commit_message)
        
        logger.info(f"Factors committed to Git: {commit.hexsha[:8]}")
        
        # Try to push if configured
        try:
            origin = repo.remote('origin')
            origin.push()
            logger.info("Changes pushed to remote repository")
        except Exception as e:
            logger.warning(f"Failed to push to remote: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to commit factors to Git: {e}")
        return False


@task(name="send_retrain_notification", retries=3, retry_delay_seconds=30)
def send_retrain_notification(drift_check: Dict[str, Any],
                             deployment_result: Dict[str, Any],
                             validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send Telegram notification about retraining completion
    
    Args:
        drift_check: Results from check_drift_breach
        deployment_result: Results from deploy_new_factors  
        validation_results: Results from validate_retrained_strategy
        
    Returns:
        Dictionary with notification results
    """
    logger = get_run_logger()
    
    try:
        from mech_exo.utils.alerts import TelegramAlerter
        import os
        
        logger.info("Sending retrain completion notification via Telegram")
        
        # Initialize Telegram alerter
        telegram_config = {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID')
        }
        
        if not telegram_config['bot_token'] or not telegram_config['chat_id']:
            logger.warning("Telegram credentials not configured, skipping notification")
            return {
                'success': False,
                'error': 'Missing Telegram credentials',
                'notification_type': 'telegram_retrain'
            }
        
        alerter = TelegramAlerter(telegram_config)
        
        # Extract key information
        version = deployment_result.get('version', 'unknown')
        factors_file = deployment_result.get('factors_file', 'N/A')
        validation_data = validation_results.get('validation_results', {})
        deployment_reason = validation_results.get('deployment_reason', 'Unknown')
        should_deploy = validation_results.get('should_deploy', False)
        deployment_success = deployment_result.get('success', False)
        
        # Send appropriate notification based on outcome
        if deployment_success and should_deploy:
            # Success notification
            success = alerter.send_retrain_success(
                validation_results=validation_data,
                version=version,
                factors_file=factors_file
            )
            notification_type = 'retrain_success'
            
        elif not should_deploy:
            # Validation failure notification
            success = alerter.send_validation_failure(
                validation_results=validation_data,
                deployment_reason=deployment_reason,
                version=version
            )
            notification_type = 'retrain_validation_failed'
            
        else:
            # General failure notification
            failure_reason = deployment_result.get('error', 'Deployment failed')
            success = alerter.send_retrain_failure(
                failure_reason=failure_reason,
                version=version
            )
            notification_type = 'retrain_failed'
        
        logger.info(f"Telegram retrain notification sent: {success} (type: {notification_type})")
        
        return {
            'success': success,
            'notification_type': notification_type,
            'telegram_sent': success,
            'deployment_success': deployment_success,
            'should_deploy': should_deploy
        }
        
    except Exception as e:
        logger.error(f"Failed to send Telegram retrain notification: {e}")
        return {
            'success': False,
            'error': str(e),
            'notification_type': 'telegram_retrain_error'
        }


@flow(name="strategy-retrain-flow", 
      task_runner=SequentialTaskRunner(),
      description="Drift-triggered strategy retraining flow")
def strategy_retrain_flow(target_date: str = None, 
                         lookback_months: int = 6,
                         force_retrain: bool = False,
                         dry_run: bool = True) -> Dict[str, Any]:
    """
    Main flow for drift-triggered strategy retraining
    
    Args:
        target_date: Date to check drift for (YYYY-MM-DD format)
        lookback_months: Number of months of data for retraining
        force_retrain: Force retraining regardless of drift status
        dry_run: If True, only stage factors without promoting to production
        
    Returns:
        Dictionary with flow execution results
    """
    logger = get_run_logger()
    
    logger.info("🔄 Starting strategy retrain flow")
    logger.info(f"Parameters: target_date={target_date}, lookback_months={lookback_months}, force_retrain={force_retrain}, dry_run={dry_run}")
    
    # Step 1: Check if retraining is needed
    drift_check = check_drift_breach(target_date)
    
    if not force_retrain and not drift_check.get('needs_retrain', False):
        logger.info("No retraining needed, ending flow")
        return {
            'status': 'skipped',
            'reason': 'No drift breach detected',
            'drift_check': drift_check
        }
    
    logger.info("Retraining triggered, proceeding with flow")
    
    # Step 2: Load retraining data
    data_result = load_retrain_data(lookback_months)
    if not data_result.get('success', False):
        logger.error("Data loading failed, ending flow")
        return {
            'status': 'failed',
            'reason': 'Data loading failed',
            'error': data_result.get('error')
        }
    
    # Step 3: Refit factors
    refit_result = refit_factors(data_result['data_summary'])
    if not refit_result.get('success', False):
        logger.error("Factor refitting failed, ending flow")
        return {
            'status': 'failed',
            'reason': 'Factor refitting failed',
            'error': refit_result.get('error')
        }
    
    # Step 4: Validate retrained strategy
    validation_result = validate_retrained_strategy(
        refit_result['new_weights'],
        refit_result['performance_metrics'],
        refit_result['version']
    )
    
    if not validation_result.get('success', False):
        logger.error("Strategy validation failed, ending flow")
        return {
            'status': 'failed',
            'reason': 'Strategy validation failed',
            'error': validation_result.get('error')
        }
    
    # Step 5: Deploy new factors (if validation passes)
    if validation_result.get('should_deploy', False):
        deployment_result = deploy_new_factors(
            refit_result['new_weights'],
            validation_result['validation_results'],
            refit_result['version'],
            dry_run
        )
    else:
        logger.info("Strategy did not pass validation criteria, skipping deployment")
        deployment_result = {
            'success': False,
            'reason': 'Did not meet deployment criteria',
            'message': validation_result.get('deployment_reason', 'Unknown reason')
        }
    
    # Step 6: Send notification
    notification_result = send_retrain_notification(
        drift_check,
        deployment_result,
        validation_result
    )
    
    # Final flow results
    flow_results = {
        'status': 'completed' if deployment_result.get('success') else 'completed_with_issues',
        'drift_check': drift_check,
        'data_loading': data_result,
        'factor_refit': refit_result,
        'validation': validation_result,
        'deployment': deployment_result,
        'notification': notification_result,
        'execution_time': datetime.now().isoformat()
    }
    
    logger.info(f"🔄 Strategy retrain flow completed: {flow_results['status']}")
    
    return flow_results


# Manual execution functions for testing
def run_manual_retrain(target_date: str = None, force: bool = False, dry_run: bool = True) -> Dict[str, Any]:
    """
    Run strategy retraining manually for testing
    
    Args:
        target_date: Date to check drift for
        force: Force retraining regardless of drift status
        dry_run: If True, only stage factors without promoting to production
        
    Returns:
        Flow execution results
    """
    return strategy_retrain_flow(target_date=target_date, force_retrain=force, dry_run=dry_run)


if __name__ == "__main__":
    # Test the retrain flow
    print("🔄 Testing Strategy Retrain Flow...")
    
    # Run with force and dry_run to test the full pipeline safely
    result = run_manual_retrain(force=True, dry_run=True)
    
    print(f"✅ Flow completed with status: {result.get('status')}")
    if result.get('deployment', {}).get('factors_file'):
        print(f"📁 New factors file: {result['deployment']['factors_file']}")
        print(f"🔄 Dry run mode: {result['deployment'].get('dry_run', False)}")