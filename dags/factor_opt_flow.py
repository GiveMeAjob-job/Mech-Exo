"""
Prefect Flow for Optuna Factor Weight Optimization

Automated workflow for running batch Optuna optimization with Telegram reporting
and optional staging integration.
"""

import logging
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional
import json

try:
    from prefect import flow, task, get_run_logger
    from prefect.blocks.system import Secret
    PREFECT_AVAILABLE = True
except ImportError:
    # Fallback for environments without Prefect
    def flow(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def task(**kwargs):
        def decorator(func):
            return func
        return decorator
    
    def get_run_logger():
        return logging.getLogger(__name__)
    
    PREFECT_AVAILABLE = False

logger = logging.getLogger(__name__)


@task(timeout_seconds=14400)  # 4 hour timeout for long optimizations
def run_optuna_batch(n_trials: int = 50, n_jobs: int = 4, study_name: str = "factor_weight_optimization",
                     study_file: str = "studies/factor_opt.db", notify_progress: bool = False,
                     progress_interval: int = 10, run_id: str = None) -> Dict[str, Any]:
    """
    Run Optuna batch optimization with specified parameters
    
    Args:
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs
        study_name: Name of the optimization study
        study_file: Path to study database file
        notify_progress: Send progress notifications
        progress_interval: Progress update interval
        run_id: Unique run identifier
        
    Returns:
        Dictionary with optimization results
    """
    task_logger = get_run_logger()
    task_logger.info(f"Starting Optuna batch optimization: {n_trials} trials, {n_jobs} jobs")
    
    try:
        from optimize.opt_factor_weights import FactorWeightOptimizer
        from mech_exo.utils.opt_callbacks import create_optuna_callback
        import time
        import os
        
        # Cap n_jobs to reasonable limits
        max_jobs = min(n_jobs, os.cpu_count() or 1, 8)
        if max_jobs != n_jobs:
            task_logger.warning(f"Capping n_jobs from {n_jobs} to {max_jobs}")
            n_jobs = max_jobs
        
        # Initialize optimizer
        optimizer = FactorWeightOptimizer(study_file)
        
        # Load historical data
        task_logger.info("Loading historical data...")
        data = optimizer.load_historical_data()
        
        if len(data['factor_scores']) == 0:
            raise ValueError("No factor data available for optimization")
        
        # Create enhanced study
        study = optimizer.create_enhanced_study(study_name)
        
        # Create progress callback
        callback = create_optuna_callback(progress_interval, notify_progress)
        
        task_logger.info(f"Running {n_trials} trials with {n_jobs} parallel jobs...")
        start_time = time.time()
        
        # Define objective function
        def objective(trial):
            return optimizer.objective_function(trial, data['factor_scores'], data['returns'])
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=[callback],
            show_progress_bar=False  # Disable in Prefect environment
        )
        
        elapsed_time = time.time() - start_time
        
        # Compile results
        results = {
            'run_id': run_id,
            'study_name': study_name,
            'n_trials': n_trials,
            'n_jobs': n_jobs,
            'elapsed_time': elapsed_time,
            'best_value': study.best_value,
            'best_trial_number': study.best_trial.number,
            'total_trials': len(study.trials),
            'best_params': study.best_trial.params,
            'best_user_attrs': study.best_trial.user_attrs or {},
            'sampler': study.sampler.__class__.__name__,
            'pruner': study.pruner.__class__.__name__,
            'data_points': len(data['factor_scores'])
        }
        
        task_logger.info(f"Optimization completed: Best Sharpe={study.best_value:.4f}")
        
        return results
        
    except Exception as e:
        task_logger.error(f"Optuna optimization failed: {e}")
        raise


@task()
def export_yaml(optimization_results: Dict[str, Any], export_path: str = None) -> Dict[str, Any]:
    """
    Export best trial to YAML file
    
    Args:
        optimization_results: Results from optimization task
        export_path: Custom export path (optional)
        
    Returns:
        Dictionary with export information
    """
    task_logger = get_run_logger()
    
    try:
        import yaml
        from pathlib import Path
        
        # Generate export path if not provided
        if not export_path:
            today = date.today().strftime('%Y-%m-%d')
            export_path = f"factors/factors_opt_{today}.yml"
        
        task_logger.info(f"Exporting optimization results to {export_path}")
        
        # Create output directory
        output_path = Path(export_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract factor weights and hyperparameters
        best_params = optimization_results['best_params']
        best_attrs = optimization_results['best_user_attrs']
        
        factor_weights = {}
        hyperparameters = {}
        
        for key, value in best_params.items():
            if key.startswith('weight_'):
                factor_name = key.replace('weight_', '')
                factor_weights[factor_name] = value
            else:
                hyperparameters[key] = value
        
        # Create comprehensive YAML structure
        export_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'optimization_method': 'optuna_tpe_prefect',
                'run_id': optimization_results['run_id'],
                'study_name': optimization_results['study_name'],
                'best_trial_number': optimization_results['best_trial_number'],
                'best_sharpe_ratio': optimization_results['best_value'],
                'max_drawdown': best_attrs.get('max_drawdown', 0),
                'total_return': best_attrs.get('total_return', 0),
                'volatility': best_attrs.get('volatility', 0),
                'constraints_satisfied': best_attrs.get('constraints_satisfied', False),
                'constraint_violations': best_attrs.get('constraint_violations', 0),
                'total_trials': optimization_results['total_trials'],
                'elapsed_time_seconds': optimization_results['elapsed_time'],
                'sampler': optimization_results['sampler'],
                'pruner': optimization_results['pruner'],
                'data_points': optimization_results['data_points']
            },
            'factors': {
                'fundamental': {
                    name: {
                        'weight': round(factor_weights.get(name, 0), 4),
                        'direction': 'higher_better',
                        'category': 'fundamental'
                    }
                    for name in ['pe_ratio', 'return_on_equity', 'revenue_growth', 'earnings_growth']
                },
                'technical': {
                    name: {
                        'weight': round(factor_weights.get(name, 0), 4),
                        'direction': 'higher_better',
                        'category': 'technical'
                    }
                    for name in ['rsi_14', 'momentum_12_1', 'volatility_ratio']
                },
                'sentiment': {
                    name: {
                        'weight': round(factor_weights.get(name, 0), 4),
                        'direction': 'higher_better',
                        'category': 'sentiment'
                    }
                    for name in ['news_sentiment', 'analyst_revisions']
                }
            },
            'hyperparameters': {
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in hyperparameters.items()
            }
        }
        
        # Write YAML file
        with open(output_path, 'w') as f:
            yaml.dump(export_data, f, default_flow_style=False, indent=2, sort_keys=False)
        
        # Get file size
        file_size = output_path.stat().st_size
        
        export_info = {
            'yaml_path': str(output_path),
            'file_size': file_size,
            'factor_count': len(factor_weights),
            'hyperparameter_count': len(hyperparameters),
            'file_size_mb': file_size / (1024 * 1024)
        }
        
        task_logger.info(f"YAML exported: {export_info['yaml_path']} ({file_size} bytes)")
        
        return export_info
        
    except Exception as e:
        task_logger.error(f"YAML export failed: {e}")
        raise


@task()
def store_opt_result(optimization_results: Dict[str, Any], export_info: Dict[str, Any]) -> bool:
    """
    Store optimization results in DuckDB opt_results table
    
    Args:
        optimization_results: Results from optimization
        export_info: YAML export information
        
    Returns:
        True if successful
    """
    task_logger = get_run_logger()
    
    try:
        import duckdb
        
        # Connect to main database
        db_path = "data/mech_exo.duckdb"
        conn = duckdb.connect(db_path)
        
        # Create opt_results table if it doesn't exist
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS opt_results (
            run_id VARCHAR PRIMARY KEY,
            date DATE,
            best_sharpe DOUBLE,
            best_trial_id INTEGER,
            yaml_path VARCHAR,
            trials_total INTEGER,
            elapsed_time_seconds DOUBLE,
            n_jobs INTEGER,
            data_points INTEGER,
            constraints_satisfied BOOLEAN,
            constraint_violations INTEGER,
            max_drawdown DOUBLE,
            total_return DOUBLE,
            volatility DOUBLE,
            study_name VARCHAR,
            sampler VARCHAR,
            pruner VARCHAR,
            file_size_bytes INTEGER,
            created_at TIMESTAMP
        )
        """
        
        conn.execute(create_table_sql)
        
        # Insert optimization results
        insert_sql = """
        INSERT OR REPLACE INTO opt_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        best_attrs = optimization_results['best_user_attrs']
        
        result_data = (
            optimization_results['run_id'],
            date.today(),
            optimization_results['best_value'],
            optimization_results['best_trial_number'],
            export_info['yaml_path'],
            optimization_results['total_trials'],
            optimization_results['elapsed_time'],
            optimization_results['n_jobs'],
            optimization_results['data_points'],
            best_attrs.get('constraints_satisfied', False),
            best_attrs.get('constraint_violations', 0),
            best_attrs.get('max_drawdown', 0),
            best_attrs.get('total_return', 0),
            best_attrs.get('volatility', 0),
            optimization_results['study_name'],
            optimization_results['sampler'],
            optimization_results['pruner'],
            export_info['file_size'],
            datetime.now()
        )
        
        conn.execute(insert_sql, result_data)
        conn.close()
        
        task_logger.info(f"Optimization results stored in database: run_id={optimization_results['run_id']}")
        
        return True
        
    except Exception as e:
        task_logger.error(f"Failed to store optimization results: {e}")
        raise


@task()
def notify_telegram(optimization_results: Dict[str, Any], export_info: Dict[str, Any], 
                   send_file: bool = True) -> bool:
    """
    Send Telegram notification with optimization summary
    
    Args:
        optimization_results: Results from optimization
        export_info: YAML export information  
        send_file: Whether to send YAML file as attachment
        
    Returns:
        True if successful
    """
    task_logger = get_run_logger()
    
    try:
        import os
        from mech_exo.utils.alerts import TelegramAlerter
        
        # Check for dry-run mode
        if os.getenv('TELEGRAM_DRY_RUN', 'false').lower() == 'true':
            task_logger.info("TELEGRAM_DRY_RUN=true - logging message instead of sending")
            
            message = _create_telegram_message(optimization_results, export_info)
            task_logger.info(f"Dry-run Telegram message:\n{message}")
            
            if send_file and export_info['file_size_mb'] < 1.0:
                task_logger.info(f"Would send file: {export_info['yaml_path']} ({export_info['file_size']} bytes)")
            
            return True
        
        # Get Telegram credentials
        try:
            if PREFECT_AVAILABLE:
                # Try to get from Prefect Secret blocks
                telegram_token_block = Secret.load("telegram-bot-token")
                telegram_chat_block = Secret.load("telegram-chat-id")
                
                telegram_config = {
                    'bot_token': telegram_token_block.get(),
                    'chat_id': telegram_chat_block.get()
                }
            else:
                # Fallback to environment variables
                telegram_config = {
                    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                    'chat_id': os.getenv('TELEGRAM_CHAT_ID')
                }
        except:
            # Fallback to environment variables
            telegram_config = {
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID')
            }
        
        if not telegram_config['bot_token'] or not telegram_config['chat_id']:
            task_logger.warning("Telegram credentials not available - skipping notification")
            return False
        
        # Initialize alerter
        alerter = TelegramAlerter(telegram_config)
        
        # Create message
        message = _create_telegram_message(optimization_results, export_info)
        
        # Send message
        success = alerter.send_message(message)
        
        if success:
            task_logger.info("Telegram message sent successfully")
            
            # Send file if requested and small enough
            if send_file and export_info['file_size_mb'] < 1.0:
                file_success = alerter.send_document(
                    export_info['yaml_path'],
                    caption=f"Optimized factors from run {optimization_results['run_id'][:8]}"
                )
                
                if file_success:
                    task_logger.info("YAML file sent successfully")
                else:
                    task_logger.warning("Failed to send YAML file")
            elif send_file:
                task_logger.info(f"YAML file too large ({export_info['file_size_mb']:.2f}MB) - skipping upload")
        else:
            task_logger.error("Failed to send Telegram message")
            
        return success
        
    except Exception as e:
        task_logger.error(f"Telegram notification failed: {e}")
        return False


@task()
def promote_yaml_to_staging(export_info: Dict[str, Any], stage: bool = True) -> bool:
    """
    Promote YAML to staging with git integration
    
    Args:
        export_info: YAML export information
        stage: Whether to promote to staging
        
    Returns:
        True if successful
    """
    task_logger = get_run_logger()
    
    if not stage:
        task_logger.info("Staging disabled - skipping promotion")
        return True
    
    try:
        from pathlib import Path
        from datetime import datetime
        import shutil
        
        task_logger.info("Promoting YAML to staging with git integration...")
        
        # Create staging directory
        staging_dir = Path("config/staging")
        staging_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped staging file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        staged_name = f"factors_optuna_{timestamp}.yml"
        staged_path = staging_dir / staged_name
        
        # Copy file to staging
        shutil.copy2(export_info['yaml_path'], staged_path)
        task_logger.info(f"Copied to staging: {staged_path}")
        
        # Git integration
        try:
            import git
            
            repo = git.Repo(".")
            
            # Add staged file
            repo.index.add([str(staged_path)])
            
            # Create commit
            commit_msg = f"Add Optuna factors {date.today().strftime('%Y-%m-%d')}"
            commit = repo.index.commit(commit_msg)
            
            task_logger.info(f"Git commit created: {commit.hexsha[:8]} - {commit_msg}")
            
            # Try to push if remote is configured
            try:
                origin = repo.remote('origin')
                origin.push()
                task_logger.info("Pushed to remote successfully")
            except Exception as e:
                task_logger.info(f"Remote push skipped: {e}")
                
        except ImportError:
            task_logger.warning("GitPython not available - skipping git operations")
        except Exception as e:
            task_logger.warning(f"Git operations failed: {e}")
        
        return True
        
    except Exception as e:
        task_logger.error(f"Staging promotion failed: {e}")
        return False


def _create_telegram_message(optimization_results: Dict[str, Any], 
                           export_info: Dict[str, Any]) -> str:
    """Create formatted Telegram message"""
    
    def escape_markdown_v2(text: str) -> str:
        """Escape MarkdownV2 special characters"""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
    
    # Get key metrics
    best_sharpe = optimization_results['best_value']
    trials = optimization_results['total_trials']
    elapsed_min = optimization_results['elapsed_time'] / 60
    yaml_filename = Path(export_info['yaml_path']).name
    
    best_attrs = optimization_results['best_user_attrs']
    max_dd = best_attrs.get('max_drawdown', 0)
    violations = best_attrs.get('constraint_violations', 0)
    constraints_ok = best_attrs.get('constraints_satisfied', False)
    
    # Create message
    message = f"🎯 *Optuna Optimization Finished*\n\n"
    message += f"📊 *Trials*: {trials}\n"
    message += f"⏱️ *Time*: {elapsed_min:.1f} minutes\n"
    message += f"📈 *Best Sharpe*: `{best_sharpe:.4f}`\n"
    message += f"📉 *Max Drawdown*: {max_dd:.1%}\n"
    message += f"⚠️ *Violations*: {violations}\n"
    message += f"✅ *Constraints*: {'Satisfied' if constraints_ok else 'Failed'}\n\n"
    message += f"📄 *YAML*: `{escape_markdown_v2(yaml_filename)}`\n"
    message += f"💾 *Size*: {export_info['file_size_mb']:.2f} MB\n\n"
    message += f"🤖 *Run ID*: `{optimization_results['run_id'][:8]}`"
    
    return message


@flow(name="factor-optimization")
def factor_optimization_flow(
    n_trials: int = 50,
    n_jobs: int = 4,
    stage: bool = False,
    notify_progress: bool = False,
    progress_interval: int = 10,
    send_telegram: bool = True,
    study_name: str = "factor_weight_optimization"
) -> Dict[str, Any]:
    """
    Prefect flow for Optuna factor weight optimization
    
    Args:
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs
        stage: Whether to promote to staging
        notify_progress: Send progress notifications during optimization
        progress_interval: Progress update interval
        send_telegram: Send Telegram summary
        study_name: Name of optimization study
        
    Returns:
        Flow execution summary
    """
    flow_logger = get_run_logger()
    
    # Generate unique run ID
    run_id = str(uuid.uuid4())
    flow_logger.info(f"Starting factor optimization flow: run_id={run_id}")
    
    try:
        # Step 1: Run Optuna batch optimization
        optimization_results = run_optuna_batch(
            n_trials=n_trials,
            n_jobs=n_jobs,
            study_name=study_name,
            notify_progress=notify_progress,
            progress_interval=progress_interval,
            run_id=run_id
        )
        
        # Step 2: Export results to YAML
        export_info = export_yaml(optimization_results)
        
        # Step 3: Store results in database
        store_success = store_opt_result(optimization_results, export_info)
        
        # Step 4: Promote to staging if requested
        staging_success = promote_yaml_to_staging(export_info, stage)
        
        # Step 5: Send Telegram notification
        telegram_success = False
        if send_telegram:
            telegram_success = notify_telegram(optimization_results, export_info)
        
        # Compile flow summary
        flow_summary = {
            'run_id': run_id,
            'status': 'success',
            'best_sharpe': optimization_results['best_value'],
            'total_trials': optimization_results['total_trials'],
            'yaml_path': export_info['yaml_path'],
            'database_stored': store_success,
            'staged': staging_success if stage else False,
            'telegram_sent': telegram_success,
            'elapsed_time': optimization_results['elapsed_time']
        }
        
        flow_logger.info(f"Flow completed successfully: {flow_summary}")
        
        return flow_summary
        
    except Exception as e:
        flow_logger.error(f"Flow failed: {e}")
        raise


if __name__ == "__main__":
    # Test the flow locally
    print("🧪 Testing factor optimization flow...")
    
    # Run with minimal parameters for testing
    result = factor_optimization_flow(
        n_trials=3,
        n_jobs=1,
        stage=False,
        send_telegram=True  # Will use dry-run mode
    )
    
    print(f"✅ Flow test result: {result}")