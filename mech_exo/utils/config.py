"""
Configuration utilities
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._config = {}
        
    def load_config(self, config_type: str) -> Dict[str, Any]:
        """Load configuration of specified type"""
        try:
            # Try local config first (with secrets), then fall back to template
            local_file = self.config_dir / f"{config_type}_local.yml"
            template_file = self.config_dir / f"{config_type}.yml"
            
            config_file = local_file if local_file.exists() else template_file
            
            if not config_file.exists():
                logger.error(f"Config file not found: {config_file}")
                return {}
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
                
            logger.info(f"Loaded config from {config_file}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config {config_type}: {e}")
            return {}
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.load_config("api_keys")
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk limits configuration"""
        return self.load_config("risk_limits")
    
    def get_factor_config(self) -> Dict[str, Any]:
        """Get factor scoring configuration"""
        return self.load_config("factors")
    
    def validate_api_keys(self, required_apis: Optional[list] = None) -> bool:
        """Validate that required API keys are present"""
        config = self.get_api_config()
        
        if not required_apis:
            required_apis = ['interactive_brokers']  # Minimum requirement
            
        missing_keys = []
        
        for api in required_apis:
            if api not in config or not config[api]:
                missing_keys.append(api)
                
        if missing_keys:
            logger.error(f"Missing API configurations: {missing_keys}")
            return False
            
        return True