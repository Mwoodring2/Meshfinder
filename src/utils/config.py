"""
ModelFinder Configuration
Centralized configuration management for the ModelFinder project.
"""

import os
from pathlib import Path
from typing import Dict, Any
import json

class Config:
    """Configuration manager for ModelFinder."""
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize configuration with optional config file."""
        self.config_file = Path(config_file)
        self.config = self.load_default_config()
        
        if self.config_file.exists():
            self.load_config()
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            "database": {
                "path": "db/modelfinder.db",
                "backup_interval": 24,  # hours
                "max_backups": 5
            },
            "scanning": {
                "supported_extensions": [
                    ".stl", ".obj", ".fbx", ".glb", ".ply", 
                    ".3ds", ".dae", ".blend", ".max", ".ma", 
                    ".mb", ".ztl", ".3mf"
                ],
                "max_file_size": 100 * 1024 * 1024,  # 100MB
                "parallel_workers": 4
            },
            "conversion": {
                "output_format": "glb",
                "normalize_scale": True,
                "center_origin": True,
                "generate_previews": True,
                "preview_size": (512, 512)
            },
            "search": {
                "default_limit": 10,
                "max_results": 100,
                "similarity_threshold": 0.8
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/modelfinder.log"
            }
        }
    
    def load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
                self.merge_config(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    def merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing config."""
        def merge_dict(base: dict, update: dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self.config, new_config)
    
    def save_config(self):
        """Save current configuration to file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'database.path')."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def get_database_path(self) -> Path:
        """Get the database file path."""
        return Path(self.get("database.path", "db/modelfinder.db"))
    
    def get_supported_extensions(self) -> list:
        """Get list of supported file extensions."""
        return self.get("scanning.supported_extensions", [])
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get("logging", {})

# Global configuration instance
config = Config()














