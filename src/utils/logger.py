"""
ModelFinder Logging
Centralized logging configuration for the ModelFinder project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from .config import config

def setup_logger(name: str = "modelfinder", 
                level: Optional[str] = None,
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting and output.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
    
    # Set logging level
    log_level = level or config.get("logging.level", "INFO")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    log_format = config.get("logging.format", 
                          "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    elif config.get("logging.file"):
        # Use configured log file
        log_path = Path(config.get("logging.file"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "modelfinder") -> logging.Logger:
    """Get a logger instance."""
    return setup_logger(name)

# Default logger
logger = get_logger()














