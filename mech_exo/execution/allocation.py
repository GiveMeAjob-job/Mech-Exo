"""
Allocation Management for Canary A/B Testing

Manages the split between base portfolio and ML canary allocations.
Supports dynamic enable/disable of canary based on performance.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def get_canary_allocation(config_path: str = "config/allocation.yml") -> float:
    """
    Get current canary allocation percentage from configuration.
    
    Args:
        config_path: Path to allocation configuration file
        
    Returns:
        Canary allocation as decimal (0.10 = 10%)
        
    Example:
        >>> allocation = get_canary_allocation()
        >>> print(f"Canary gets {allocation*100:.1f}% of each order")
    """
    try:
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Allocation config not found: {config_path}, using default 10%")
            return 0.10
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if canary is enabled (support both old and new config structure)
        canary_config = config.get('canary', {})
        canary_enabled = canary_config.get('enabled', config.get('canary_enabled', True))
        if not canary_enabled:
            logger.info("Canary allocation disabled in config")
            return 0.0
        
        # Get allocation percentage (support both old and new config structure)
        allocation = canary_config.get('allocation', config.get('canary_allocation', 0.10))
        
        # Validate range - Phase P11 Week 2: Increased limit to 60% for scaled beta
        if not (0.0 <= allocation <= 0.60):
            logger.warning(f"Invalid canary allocation {allocation}, clamping to [0.0, 0.60]")
            allocation = max(0.0, min(0.60, allocation))
        
        logger.debug(f"Canary allocation: {allocation:.1%}")
        return allocation
        
    except Exception as e:
        logger.error(f"Failed to load canary allocation: {e}")
        logger.info("Using default canary allocation: 10%")
        return 0.10


def split_order_quantity(total_quantity: int, canary_allocation: Optional[float] = None) -> Tuple[int, int]:
    """
    Split order quantity between base and canary allocations.
    
    Args:
        total_quantity: Total shares to allocate
        canary_allocation: Canary percentage (if None, will load from config)
        
    Returns:
        Tuple of (base_quantity, canary_quantity)
        
    Note:
        Rounds toward base to avoid fractional shares.
        
    Example:
        >>> base_qty, canary_qty = split_order_quantity(100)
        >>> print(f"Base: {base_qty}, Canary: {canary_qty}")
        Base: 90, Canary: 10
    """
    try:
        if canary_allocation is None:
            canary_allocation = get_canary_allocation()
        
        # Calculate canary quantity (round down to avoid fractional shares)
        canary_quantity = int(total_quantity * canary_allocation)
        
        # Base gets the remainder
        base_quantity = total_quantity - canary_quantity
        
        logger.debug(f"Split {total_quantity} shares -> Base: {base_quantity}, Canary: {canary_quantity}")
        
        return base_quantity, canary_quantity
        
    except Exception as e:
        logger.error(f"Failed to split order quantity: {e}")
        # Failsafe: all to base
        return total_quantity, 0


def is_canary_enabled(config_path: str = "config/allocation.yml") -> bool:
    """
    Check if canary allocation is currently enabled.
    
    Args:
        config_path: Path to allocation configuration file
        
    Returns:
        Boolean indicating if canary is enabled
    """
    try:
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Allocation config not found: {config_path}, assuming enabled")
            return True
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Support both old and new config structure
        canary_config = config.get('canary', {})
        return canary_config.get('enabled', config.get('canary_enabled', True))
        
    except Exception as e:
        logger.error(f"Failed to check canary status: {e}")
        return True


def update_canary_enabled(enabled: bool, config_path: str = "config/allocation.yml") -> bool:
    """
    Update canary enabled status in configuration file.
    
    Args:
        enabled: Whether to enable or disable canary
        config_path: Path to allocation configuration file
        
    Returns:
        Boolean indicating success
    """
    try:
        config_file = Path(config_path)
        
        # Load existing config or create default
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
            # Ensure directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Update enabled status
        config['canary_enabled'] = enabled
        
        # Ensure other defaults are present
        if 'canary_allocation' not in config:
            config['canary_allocation'] = 0.10
        
        # Write updated config
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Canary {'enabled' if enabled else 'disabled'} in {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update canary enabled status: {e}")
        return False


def get_allocation_config() -> Dict:
    """
    Get complete allocation configuration.
    
    Returns:
        Dictionary with allocation settings
    """
    try:
        config_path = "config/allocation.yml"
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Return default config with new hysteresis structure
            return {
                'canary_enabled': True,
                'canary_allocation': 0.10,
                'disable_rule': {
                    'sharpe_low': 0.0,
                    'confirm_days': 2,
                    'max_dd_pct': 2.0,
                    'min_observations': 21
                },
                'created_at': 'default'
            }
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        # Ensure all required keys are present with defaults
        defaults = {
            'canary_enabled': True,
            'canary_allocation': 0.10
        }
        
        # Handle both old and new config formats
        if 'disable_rule' not in config:
            # Convert old format to new
            config['disable_rule'] = {
                'sharpe_low': config.get('disable_threshold_sharpe', 0.0),
                'confirm_days': 2,  # New hysteresis parameter
                'max_dd_pct': 2.0,  # New max drawdown threshold
                'min_observations': config.get('disable_min_days', 21)
            }
            
            # Remove old keys if present
            config.pop('disable_threshold_sharpe', None)
            config.pop('disable_min_days', None)
        
        # Ensure disable_rule has all required keys
        disable_rule_defaults = {
            'sharpe_low': 0.0,
            'confirm_days': 2,
            'max_dd_pct': 2.0,
            'min_observations': 21
        }
        
        for key, default_value in disable_rule_defaults.items():
            if key not in config['disable_rule']:
                config['disable_rule'][key] = default_value
        
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get allocation config: {e}")
        return {
            'canary_enabled': True,
            'canary_allocation': 0.10,
            'disable_rule': {
                'sharpe_low': 0.0,
                'confirm_days': 2,
                'max_dd_pct': 2.0,
                'min_observations': 21
            },
            'error': str(e)
        }


def get_consecutive_breach_days(config_path: str = "config/allocation.yml") -> int:
    """
    Get current consecutive breach days counter for hysteresis logic.
    
    Args:
        config_path: Path to allocation configuration file
        
    Returns:
        Number of consecutive days below threshold
    """
    try:
        config_file = Path(config_path)
        
        if not config_file.exists():
            return 0
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        return config.get('consecutive_breach_days', 0)
        
    except Exception as e:
        logger.error(f"Failed to get consecutive breach days: {e}")
        return 0


def update_consecutive_breach_days(days: int, config_path: str = "config/allocation.yml") -> bool:
    """
    Update consecutive breach days counter in configuration.
    
    Args:
        days: Number of consecutive breach days
        config_path: Path to allocation configuration file
        
    Returns:
        Boolean indicating success
    """
    try:
        config_file = Path(config_path)
        
        # Load existing config or create default
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
            config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Update breach counter
        config['consecutive_breach_days'] = days
        
        # Write updated config
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.debug(f"Updated consecutive breach days to {days}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update consecutive breach days: {e}")
        return False


def reset_breach_counter(config_path: str = "config/allocation.yml") -> bool:
    """
    Reset consecutive breach days counter to 0.
    
    Args:
        config_path: Path to allocation configuration file
        
    Returns:
        Boolean indicating success
    """
    return update_consecutive_breach_days(0, config_path)


def increment_breach_counter(config_path: str = "config/allocation.yml") -> int:
    """
    Increment consecutive breach days counter by 1.
    
    Args:
        config_path: Path to allocation configuration file
        
    Returns:
        New consecutive breach days count
    """
    current_days = get_consecutive_breach_days(config_path)
    new_days = current_days + 1
    
    if update_consecutive_breach_days(new_days, config_path):
        return new_days
    else:
        return current_days


def check_hysteresis_trigger(canary_sharpe: float, config_path: str = "config/allocation.yml") -> Dict[str, any]:
    """
    Check hysteresis logic for auto-disable trigger.
    
    Args:
        canary_sharpe: Current canary Sharpe ratio
        config_path: Path to allocation configuration file
        
    Returns:
        Dictionary with hysteresis check results
    """
    try:
        config = get_allocation_config()
        disable_rule = config['disable_rule']
        
        sharpe_threshold = disable_rule['sharpe_low']
        required_days = disable_rule['confirm_days']
        
        current_breach_days = get_consecutive_breach_days(config_path)
        
        # Check if current performance breaches threshold
        is_breach = canary_sharpe < sharpe_threshold
        
        if is_breach:
            # Increment breach counter
            new_breach_days = increment_breach_counter(config_path)
            should_trigger = new_breach_days >= required_days
        else:
            # Reset breach counter on good performance
            reset_breach_counter(config_path)
            new_breach_days = 0
            should_trigger = False
        
        result = {
            'is_breach': is_breach,
            'canary_sharpe': canary_sharpe,
            'threshold': sharpe_threshold,
            'current_breach_days': new_breach_days,
            'required_days': required_days,
            'should_trigger': should_trigger,
            'hysteresis_active': new_breach_days > 0
        }
        
        logger.info(f"Hysteresis check: Sharpe {canary_sharpe:.3f} vs {sharpe_threshold:.3f}, "
                   f"breach days {new_breach_days}/{required_days}, trigger: {should_trigger}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to check hysteresis trigger: {e}")
        return {
            'is_breach': False,
            'canary_sharpe': canary_sharpe,
            'threshold': 0.0,
            'current_breach_days': 0,
            'required_days': 2,
            'should_trigger': False,
            'hysteresis_active': False,
            'error': str(e)
        }