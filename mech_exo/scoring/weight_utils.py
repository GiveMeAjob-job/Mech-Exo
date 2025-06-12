"""
ML Weight Management Utilities

Handles dynamic adjustment of ML weight based on performance metrics.
Provides functions for weight history tracking and configuration management.
"""

import logging
import yaml
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def create_ml_weight_history_table() -> bool:
    """
    Create the ml_weight_history table for tracking weight adjustments.
    
    Returns:
        Boolean indicating success
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS ml_weight_history (
            date DATE PRIMARY KEY,
            old_weight DOUBLE,
            new_weight DOUBLE,
            adjustment_rule VARCHAR,
            ml_sharpe DOUBLE,
            baseline_sharpe DOUBLE,
            sharpe_diff DOUBLE,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        storage.conn.execute(create_table_sql)
        storage.close()
        
        logger.info("ml_weight_history table created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create ml_weight_history table: {e}")
        return False


def get_current_ml_weight() -> float:
    """
    Get current ML weight from config/factors.yml.
    
    Returns:
        Current ML weight value, defaults to 0.30 if not found
    """
    try:
        from ..utils.config import ConfigManager
        
        config_manager = ConfigManager()
        
        # Load factors configuration
        factors_config = config_manager.load_config('factors')
        
        # Get ML weight, default to 0.30
        ml_weight = factors_config.get('ml_weight', 0.30)
        
        logger.info(f"Current ML weight: {ml_weight}")
        return float(ml_weight)
        
    except Exception as e:
        logger.error(f"Failed to get current ML weight: {e}")
        logger.info("Using default ML weight: 0.30")
        return 0.30


def log_weight_change(old_weight: float, new_weight: float, 
                     adjustment_rule: str, ml_sharpe: float, 
                     baseline_sharpe: float, notes: str = "",
                     change_date: Optional[str] = None) -> bool:
    """
    Log ML weight change to history table.
    
    Args:
        old_weight: Previous ML weight
        new_weight: New ML weight
        adjustment_rule: Rule that triggered the change
        ml_sharpe: ML strategy Sharpe ratio
        baseline_sharpe: Baseline strategy Sharpe ratio
        notes: Additional notes about the change
        change_date: Date of change (default: today)
        
    Returns:
        Boolean indicating success
    """
    try:
        from ..datasource.storage import DataStorage
        from datetime import date as dt_date
        
        target_date = change_date or str(dt_date.today())
        sharpe_diff = ml_sharpe - baseline_sharpe
        
        # Prepare data for storage
        weight_data = pd.DataFrame([{
            'date': target_date,
            'old_weight': old_weight,
            'new_weight': new_weight,
            'adjustment_rule': adjustment_rule,
            'ml_sharpe': ml_sharpe,
            'baseline_sharpe': baseline_sharpe,
            'sharpe_diff': sharpe_diff,
            'notes': notes,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }])
        
        # Store in database
        storage = DataStorage()
        
        # Ensure table exists
        create_ml_weight_history_table()
        
        # Insert/update weight change (replace if exists for same date)
        storage.conn.execute("DELETE FROM ml_weight_history WHERE date = ?", [target_date])
        
        # Register dataframe and insert
        storage.conn.register('temp_weight_history', weight_data)
        storage.conn.execute("INSERT INTO ml_weight_history SELECT * FROM temp_weight_history")
        
        storage.close()
        
        logger.info(f"Weight change logged: {old_weight:.3f} → {new_weight:.3f} ({adjustment_rule})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to log weight change: {e}")
        return False


def get_weight_history(days: int = 30) -> pd.DataFrame:
    """
    Get ML weight adjustment history.
    
    Args:
        days: Number of days to look back
        
    Returns:
        DataFrame with weight history
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        query = """
        SELECT 
            date,
            old_weight,
            new_weight,
            adjustment_rule,
            ml_sharpe,
            baseline_sharpe,
            sharpe_diff,
            notes,
            created_at
        FROM ml_weight_history 
        WHERE date >= (CURRENT_DATE - INTERVAL '{} days')
        ORDER BY date DESC
        """.format(days)
        
        try:
            result = pd.read_sql_query(query, storage.conn)
            storage.close()
            
            if result.empty:
                logger.info("No weight history found")
                return pd.DataFrame(columns=[
                    'date', 'old_weight', 'new_weight', 'adjustment_rule',
                    'ml_sharpe', 'baseline_sharpe', 'sharpe_diff', 'notes', 'created_at'
                ])
            
            # Convert date column for consistency
            result['date'] = pd.to_datetime(result['date'])
            
            logger.info(f"Retrieved {len(result)} weight history records")
            return result
            
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.info("Weight history table does not exist yet")
            else:
                logger.error(f"Failed to query weight history: {e}")
            
            return pd.DataFrame(columns=[
                'date', 'old_weight', 'new_weight', 'adjustment_rule',
                'ml_sharpe', 'baseline_sharpe', 'sharpe_diff', 'notes', 'created_at'
            ])
        
    except Exception as e:
        logger.error(f"Failed to get weight history: {e}")
        return pd.DataFrame(columns=[
            'date', 'old_weight', 'new_weight', 'adjustment_rule',
            'ml_sharpe', 'baseline_sharpe', 'sharpe_diff', 'notes', 'created_at'
        ])


def update_ml_weight_config(new_weight: float, config_path: str = "config/factors.yml",
                           backup: bool = True) -> bool:
    """
    Update ML weight in factors.yml configuration file using ruamel.yaml to preserve comments.
    
    Args:
        new_weight: New ML weight value (0.0 - 0.50)
        config_path: Path to factors.yml file
        backup: Whether to create backup before updating
        
    Returns:
        Boolean indicating success
    """
    try:
        # Validate weight range
        if not (0.0 <= new_weight <= 0.50):
            logger.error(f"Invalid ML weight: {new_weight}. Must be between 0.0 and 0.50")
            return False
        
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.error(f"Config file not found: {config_path}")
            return False
        
        # Create backup if requested
        if backup:
            backup_path = config_file.with_suffix(f'.yml.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            backup_path.write_text(config_file.read_text())
            logger.info(f"Config backup created: {backup_path}")
        
        # Use ruamel.yaml to preserve comments and formatting
        return update_ml_weight_in_yaml(config_path, new_weight)
        
    except Exception as e:
        logger.error(f"Failed to update ML weight config: {e}")
        return False


def update_ml_weight_in_yaml(yaml_path: str, new_weight: float) -> bool:
    """
    Update ML weight in YAML file while preserving comments and formatting.
    
    Args:
        yaml_path: Path to YAML configuration file
        new_weight: New ML weight value (rounded to 2 decimal places)
        
    Returns:
        Boolean indicating success
    """
    try:
        from ruamel.yaml import YAML
        
        # Initialize YAML processor with comment preservation
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.map_indent = 2
        yaml.sequence_indent = 4
        yaml.sequence_dash_offset = 2
        yaml.width = 4096  # Prevent line wrapping
        
        # Round weight to 2 decimal places
        rounded_weight = round(float(new_weight), 2)
        
        # Read current config
        with open(yaml_path, 'r') as f:
            config = yaml.load(f)
        
        # Ensure config was loaded properly
        if config is None:
            config = {}
        
        # Get old weight for logging
        old_weight = config.get('ml_weight', 0.30) if isinstance(config, dict) else 0.30
        
        # Update ML weight 
        config['ml_weight'] = rounded_weight
        
        # Write updated config back to file
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"YAML updated: ml_weight {old_weight:.3f} → {rounded_weight:.3f}")
        return True
        
    except ImportError:
        logger.warning("ruamel.yaml not available, falling back to standard yaml")
        return _update_ml_weight_fallback(yaml_path, new_weight)
    except Exception as e:
        logger.error(f"Failed to update YAML with ruamel.yaml: {e}")
        # Try fallback on any ruamel.yaml error
        logger.info("Attempting fallback YAML update method")
        return _update_ml_weight_fallback(yaml_path, new_weight)


def _update_ml_weight_fallback(yaml_path: str, new_weight: float) -> bool:
    """
    Fallback YAML update using standard yaml (comments will be lost).
    
    Args:
        yaml_path: Path to YAML configuration file
        new_weight: New ML weight value
        
    Returns:
        Boolean indicating success
    """
    try:
        import yaml
        
        # Read current config
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get old weight for logging
        old_weight = config.get('ml_weight', 0.30)
        
        # Update ML weight
        config['ml_weight'] = round(float(new_weight), 2)
        
        # Write updated config
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.warning("YAML updated using fallback method (comments lost)")
        logger.info(f"ml_weight {old_weight:.3f} → {new_weight:.3f}")
        return True
        
    except Exception as e:
        logger.error(f"Fallback YAML update failed: {e}")
        return False


def get_latest_weight_change() -> Optional[Dict]:
    """
    Get the most recent weight change from history.
    
    Returns:
        Dictionary with latest weight change data or None
    """
    try:
        history = get_weight_history(days=7)  # Look back 7 days
        
        if history.empty:
            return None
        
        latest = history.iloc[0]
        return {
            'date': latest['date'],
            'old_weight': latest['old_weight'],
            'new_weight': latest['new_weight'],
            'adjustment_rule': latest['adjustment_rule'],
            'ml_sharpe': latest['ml_sharpe'],
            'baseline_sharpe': latest['baseline_sharpe'],
            'sharpe_diff': latest['sharpe_diff'],
            'notes': latest['notes']
        }
        
    except Exception as e:
        logger.error(f"Failed to get latest weight change: {e}")
        return None


def validate_weight_bounds(weight: float) -> Tuple[bool, str]:
    """
    Validate ML weight is within acceptable bounds.
    
    Args:
        weight: Weight value to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(weight, (int, float)):
        return False, f"Weight must be numeric, got {type(weight)}"
    
    if weight < 0.0:
        return False, f"Weight cannot be negative: {weight}"
    
    if weight > 0.50:
        return False, f"Weight cannot exceed 0.50: {weight}"
    
    return True, ""


def compute_new_weight(baseline_sharpe: float,
                      ml_sharpe: float,
                      current_w: float,
                      step: float = 0.05,
                      upper: float = 0.50,
                      lower: float = 0.0,
                      up_thresh: float = 0.10,
                      down_thresh: float = -0.05) -> Tuple[float, str]:
    """
    Compute new ML weight based on performance comparison.
    
    Algorithm:
    1. Calculate delta = ml_sharpe - baseline_sharpe
    2. If delta >= up_thresh: increase weight by step (capped at upper)
    3. If delta <= down_thresh: decrease weight by step (floored at lower)
    4. Otherwise: keep current weight unchanged
    
    Args:
        baseline_sharpe: Baseline strategy 30-day Sharpe ratio
        ml_sharpe: ML-enhanced strategy 30-day Sharpe ratio
        current_w: Current ML weight (0.0 - 0.50)
        step: Adjustment step size (default: 0.05)
        upper: Maximum allowed weight (default: 0.50)
        lower: Minimum allowed weight (default: 0.0)
        up_thresh: Minimum Sharpe advantage to increase weight (default: 0.10)
        down_thresh: Maximum Sharpe disadvantage to decrease weight (default: -0.05)
        
    Returns:
        Tuple of (new_weight, adjustment_rule)
    """
    import math
    
    # Edge case protection: handle NaN/None values
    if (baseline_sharpe is None or ml_sharpe is None or 
        math.isnan(baseline_sharpe) or math.isnan(ml_sharpe)):
        logger.warning("Invalid Sharpe values provided, keeping current weight")
        return current_w, "INVALID_SHARPE_VALUES"
    
    # Clamp current weight to valid range
    current_w = max(lower, min(current_w, upper))
    
    # Calculate performance delta
    delta = ml_sharpe - baseline_sharpe
    
    logger.info(f"Weight adjustment analysis: ML={ml_sharpe:.3f}, Baseline={baseline_sharpe:.3f}, Delta={delta:+.3f}")
    
    # Apply adjustment rules
    if delta >= up_thresh:
        # ML significantly outperforms baseline -> increase weight
        new_weight = min(current_w + step, upper)
        rule = "ML_OUTPERFORM_BASELINE"
        logger.info(f"ML outperforms by {delta:+.3f} (>= {up_thresh:+.3f}), increasing weight")
        
    elif delta <= down_thresh:
        # ML significantly underperforms baseline -> decrease weight
        new_weight = max(current_w - step, lower)
        rule = "ML_UNDERPERFORM_BASELINE"
        logger.info(f"ML underperforms by {delta:+.3f} (<= {down_thresh:+.3f}), decreasing weight")
        
    else:
        # Performance difference within acceptable band -> no change
        new_weight = current_w
        rule = "PERFORMANCE_WITHIN_BAND"
        logger.info(f"Performance delta {delta:+.3f} within band [{down_thresh:+.3f}, {up_thresh:+.3f}], no change")
    
    # Round to 2 decimal places to avoid tiny diffs in YAML
    new_weight = round(new_weight, 2)
    
    return new_weight, rule


def get_baseline_and_ml_sharpe(days: int = 30) -> Tuple[Optional[float], Optional[float]]:
    """
    Get baseline and ML strategy Sharpe ratios for comparison.
    
    Args:
        days: Number of days to look back for Sharpe calculation
        
    Returns:
        Tuple of (baseline_sharpe, ml_sharpe) or (None, None) if unavailable
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        # Query latest ML live metrics for ML Sharpe
        ml_query = """
        SELECT sharpe_30d
        FROM ml_live_metrics 
        ORDER BY date DESC
        LIMIT 1
        """
        
        # Query performance curves for baseline Sharpe (simplified: use latest entry)
        baseline_query = """
        SELECT 
            (baseline_equity - LAG(baseline_equity, {}) OVER (ORDER BY date)) / 
            (LAG(baseline_equity, {}) OVER (ORDER BY date)) as daily_return
        FROM performance_curves 
        WHERE date >= (CURRENT_DATE - INTERVAL '{} days')
        ORDER BY date DESC
        """.format(days, days, days)
        
        try:
            # Get ML Sharpe
            ml_result = pd.read_sql_query(ml_query, storage.conn)
            ml_sharpe = float(ml_result.iloc[0]['sharpe_30d']) if not ml_result.empty else None
            
            # For baseline Sharpe, use a simplified approach: get from ml_live_metrics table
            # which should contain both baseline and ML metrics
            baseline_query_simple = """
            SELECT ic as baseline_sharpe_proxy
            FROM ml_live_metrics
            ORDER BY date DESC  
            LIMIT 1
            """
            
            # Simplified: assume baseline Sharpe is approximated by IC * 10 (rough proxy)
            # In production, this would be calculated from actual baseline performance
            baseline_result = pd.read_sql_query(baseline_query_simple, storage.conn)
            if not baseline_result.empty:
                ic = float(baseline_result.iloc[0]['baseline_sharpe_proxy'])
                baseline_sharpe = abs(ic) * 10  # Convert IC to rough Sharpe proxy
            else:
                baseline_sharpe = None
                
            storage.close()
            
            logger.info(f"Retrieved Sharpe ratios - Baseline: {baseline_sharpe}, ML: {ml_sharpe}")
            return baseline_sharpe, ml_sharpe
            
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.info("Metrics tables do not exist yet")
            else:
                logger.error(f"Failed to query Sharpe ratios: {e}")
            
            storage.close()
            return None, None
        
    except Exception as e:
        logger.error(f"Failed to get Sharpe ratios: {e}")
        return None, None


def auto_adjust_ml_weight(dry_run: bool = False) -> Dict[str, any]:
    """
    Automatically adjust ML weight based on performance comparison.
    
    Args:
        dry_run: If True, only simulate the adjustment without making changes
        
    Returns:
        Dictionary with adjustment results
    """
    try:
        # Get current weight
        current_weight = get_current_ml_weight()
        
        # Get performance metrics
        baseline_sharpe, ml_sharpe = get_baseline_and_ml_sharpe()
        
        if baseline_sharpe is None or ml_sharpe is None:
            logger.warning("Unable to retrieve Sharpe ratios for weight adjustment")
            return {
                'success': False,
                'error': 'Missing Sharpe ratio data',
                'current_weight': current_weight,
                'new_weight': current_weight,
                'changed': False
            }
        
        # Compute new weight
        new_weight, rule = compute_new_weight(baseline_sharpe, ml_sharpe, current_weight)
        
        # Check if weight changed
        weight_changed = abs(new_weight - current_weight) > 0.001
        
        result = {
            'success': True,
            'current_weight': current_weight,
            'new_weight': new_weight,
            'changed': weight_changed,
            'adjustment_rule': rule,
            'baseline_sharpe': baseline_sharpe,
            'ml_sharpe': ml_sharpe,
            'sharpe_diff': ml_sharpe - baseline_sharpe,
            'dry_run': dry_run
        }
        
        if weight_changed and not dry_run:
            # Update config file
            config_success = update_ml_weight_config(new_weight)
            
            if config_success:
                # Log the change
                log_success = log_weight_change(
                    old_weight=current_weight,
                    new_weight=new_weight,
                    adjustment_rule=rule,
                    ml_sharpe=ml_sharpe,
                    baseline_sharpe=baseline_sharpe,
                    notes=f"Auto-adjustment: {ml_sharpe - baseline_sharpe:+.3f} Sharpe diff"
                )
                
                result['config_updated'] = config_success
                result['change_logged'] = log_success
                
                logger.info(f"ML weight auto-adjusted: {current_weight:.3f} → {new_weight:.3f} ({rule})")
            else:
                result['success'] = False
                result['error'] = 'Failed to update config file'
                
        elif weight_changed and dry_run:
            logger.info(f"DRY RUN: Would adjust ML weight: {current_weight:.3f} → {new_weight:.3f} ({rule})")
            result['config_updated'] = False
            result['change_logged'] = False
            
        else:
            logger.info(f"No weight adjustment needed: {current_weight:.3f} ({rule})")
            result['config_updated'] = False
            result['change_logged'] = False
        
        return result
        
    except Exception as e:
        logger.error(f"Auto weight adjustment failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'current_weight': get_current_ml_weight(),
            'new_weight': get_current_ml_weight(),
            'changed': False
        }


def format_weight_change_summary(old_weight: float, new_weight: float,
                                ml_sharpe: float, baseline_sharpe: float,
                                adjustment_rule: str) -> str:
    """
    Format weight change summary for notifications.
    
    Args:
        old_weight: Previous weight
        new_weight: New weight
        ml_sharpe: ML strategy Sharpe
        baseline_sharpe: Baseline strategy Sharpe
        adjustment_rule: Rule that triggered change
        
    Returns:
        Formatted summary string
    """
    change_direction = "↗️" if new_weight > old_weight else "↘️" if new_weight < old_weight else "➡️"
    sharpe_diff = ml_sharpe - baseline_sharpe
    
    summary = f"⚖️ ML Weight Auto-Adjusted: {old_weight:.2f} {change_direction} {new_weight:.2f}\n"
    summary += f"📊 ML Sharpe: {ml_sharpe:.3f} vs Baseline: {baseline_sharpe:.3f} (diff: {sharpe_diff:+.3f})\n"
    summary += f"🔧 Rule: {adjustment_rule}"
    
    return summary