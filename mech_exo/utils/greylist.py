"""
Greylist Symbol Filtering

Provides hot-reloadable symbol filtering for order routing.
Symbols on the greylist are blocked from trading unless explicitly overridden.
"""

import logging
import os
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class GreylistManager:
    """
    Manager for greylist symbol filtering with hot-reload capability
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize greylist manager
        
        Args:
            config_path: Path to allocation.yml config file
        """
        if config_path is None:
            # Default to project config directory
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "allocation.yml"
        
        self.config_path = Path(config_path)
        self._config_cache = None
        self._last_mtime = None
        self._last_reload = None
        self._lock = Lock()
        
        # Initialize cache
        self._reload_if_needed()
    
    def _reload_if_needed(self) -> bool:
        """
        Reload configuration if file has changed
        
        Returns:
            True if config was reloaded
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Greylist config not found: {self.config_path}")
                return False
            
            current_mtime = self.config_path.stat().st_mtime
            
            # Check if we need to reload
            should_reload = (
                self._config_cache is None or
                self._last_mtime is None or
                current_mtime > self._last_mtime or
                self._should_check_reload()
            )
            
            if not should_reload:
                return False
            
            with self._lock:
                # Double-check after acquiring lock
                if self._last_mtime is not None and current_mtime <= self._last_mtime:
                    return False
                
                # Load configuration
                with open(self.config_path, 'r') as f:
                    self._config_cache = yaml.safe_load(f)
                
                self._last_mtime = current_mtime
                self._last_reload = datetime.now()
                
                # Update hot-reload timestamp in config
                self._update_reload_timestamp()
                
                logger.info(f"✅ Greylist config reloaded from {self.config_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to reload greylist config: {e}")
            return False
    
    def _should_check_reload(self) -> bool:
        """
        Check if enough time has passed to check for reload
        
        Returns:
            True if we should check for file changes
        """
        if self._last_reload is None:
            return True
        
        config = self._config_cache or {}
        hot_reload = config.get('symbol_filtering', {}).get('hot_reload', {})
        check_interval = hot_reload.get('check_interval_seconds', 600)  # Default 10 minutes
        
        return datetime.now() - self._last_reload > timedelta(seconds=check_interval)
    
    def _update_reload_timestamp(self):
        """
        Update the last_reload timestamp in the config file
        """
        try:
            if not self._config_cache:
                return
            
            # Update timestamp in memory
            if 'symbol_filtering' not in self._config_cache:
                self._config_cache['symbol_filtering'] = {}
            if 'hot_reload' not in self._config_cache['symbol_filtering']:
                self._config_cache['symbol_filtering']['hot_reload'] = {}
            
            self._config_cache['symbol_filtering']['hot_reload']['last_reload'] = datetime.now().isoformat()
            
            # Write back to file (optional - for visibility)
            # Note: In production, we might skip this to avoid file write conflicts
            
        except Exception as e:
            logger.debug(f"Could not update reload timestamp: {e}")
    
    def get_greylist(self) -> List[str]:
        """
        Get current greylist symbols
        
        Returns:
            List of greylisted symbol strings
        """
        self._reload_if_needed()
        
        if not self._config_cache:
            return []
        
        symbol_filtering = self._config_cache.get('symbol_filtering', {})
        greylist = symbol_filtering.get('graylist_symbols', [])
        
        # Ensure all symbols are uppercase
        return [symbol.upper() for symbol in greylist]
    
    def is_greylisted(self, symbol: str) -> bool:
        """
        Check if a symbol is on the greylist
        
        Args:
            symbol: Stock symbol to check
            
        Returns:
            True if symbol is greylisted
        """
        if not symbol:
            return False
        
        greylist = self.get_greylist()
        return symbol.upper() in greylist
    
    def is_override_enabled(self) -> bool:
        """
        Check if greylist override is enabled
        
        Returns:
            True if override is enabled
        """
        self._reload_if_needed()
        
        if not self._config_cache:
            return False
        
        emergency = self._config_cache.get('symbol_filtering', {}).get('emergency_overrides', {})
        return emergency.get('graylist_override_enabled', False)
    
    def get_greylist_stats(self) -> Dict[str, Any]:
        """
        Get greylist statistics for dashboard display
        
        Returns:
            Dictionary with greylist stats
        """
        greylist = self.get_greylist()
        
        return {
            'total_symbols': len(greylist),
            'symbols': greylist,
            'override_enabled': self.is_override_enabled(),
            'last_reload': self._last_reload.isoformat() if self._last_reload else None,
            'config_path': str(self.config_path)
        }
    
    def add_symbol(self, symbol: str, reason: str = "") -> bool:
        """
        Add a symbol to the greylist
        
        Args:
            symbol: Symbol to add
            reason: Reason for adding (for audit trail)
            
        Returns:
            True if successful
        """
        try:
            self._reload_if_needed()
            
            if not self._config_cache:
                logger.error("No config loaded, cannot add symbol")
                return False
            
            symbol = symbol.upper()
            
            # Get current greylist
            if 'symbol_filtering' not in self._config_cache:
                self._config_cache['symbol_filtering'] = {}
            if 'graylist_symbols' not in self._config_cache['symbol_filtering']:
                self._config_cache['symbol_filtering']['graylist_symbols'] = []
            
            greylist = self._config_cache['symbol_filtering']['graylist_symbols']
            
            if symbol not in greylist:
                greylist.append(symbol)
                
                # Add to change history
                if 'change_history' not in self._config_cache:
                    self._config_cache['change_history'] = []
                
                self._config_cache['change_history'].append({
                    'action': 'add_greylist_symbol',
                    'symbol': symbol,
                    'reason': reason,
                    'timestamp': datetime.now().isoformat(),
                    'user': 'greylist_manager'
                })
                
                # Update metadata
                self._config_cache['updated_at'] = datetime.now().isoformat()
                
                # Save to file
                with self._lock:
                    with open(self.config_path, 'w') as f:
                        yaml.safe_dump(self._config_cache, f, default_flow_style=False, sort_keys=False)
                
                logger.info(f"✅ Added {symbol} to greylist: {reason}")
                return True
            else:
                logger.info(f"Symbol {symbol} already on greylist")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add symbol {symbol} to greylist: {e}")
            return False
    
    def remove_symbol(self, symbol: str, reason: str = "") -> bool:
        """
        Remove a symbol from the greylist
        
        Args:
            symbol: Symbol to remove
            reason: Reason for removal (for audit trail)
            
        Returns:
            True if successful
        """
        try:
            self._reload_if_needed()
            
            if not self._config_cache:
                logger.error("No config loaded, cannot remove symbol")
                return False
            
            symbol = symbol.upper()
            
            greylist = self._config_cache.get('symbol_filtering', {}).get('graylist_symbols', [])
            
            if symbol in greylist:
                greylist.remove(symbol)
                
                # Add to change history
                if 'change_history' not in self._config_cache:
                    self._config_cache['change_history'] = []
                
                self._config_cache['change_history'].append({
                    'action': 'remove_greylist_symbol',
                    'symbol': symbol,
                    'reason': reason,
                    'timestamp': datetime.now().isoformat(),
                    'user': 'greylist_manager'
                })
                
                # Update metadata
                self._config_cache['updated_at'] = datetime.now().isoformat()
                
                # Save to file
                with self._lock:
                    with open(self.config_path, 'w') as f:
                        yaml.safe_dump(self._config_cache, f, default_flow_style=False, sort_keys=False)
                
                logger.info(f"✅ Removed {symbol} from greylist: {reason}")
                return True
            else:
                logger.info(f"Symbol {symbol} not on greylist")
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove symbol {symbol} from greylist: {e}")
            return False


# Global instance for easy access
_greylist_manager = None

def get_greylist_manager() -> GreylistManager:
    """
    Get global greylist manager instance
    
    Returns:
        GreylistManager instance
    """
    global _greylist_manager
    if _greylist_manager is None:
        _greylist_manager = GreylistManager()
    return _greylist_manager

def get_greylist() -> List[str]:
    """
    Convenience function to get current greylist
    
    Returns:
        List of greylisted symbols
    """
    return get_greylist_manager().get_greylist()

def is_greylisted(symbol: str) -> bool:
    """
    Convenience function to check if symbol is greylisted
    
    Args:
        symbol: Symbol to check
        
    Returns:
        True if symbol is greylisted
    """
    return get_greylist_manager().is_greylisted(symbol)