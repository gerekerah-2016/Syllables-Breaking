"""
Logging configuration.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import logging
import sys
from pathlib import Path
from src.params import get_run_params

_logger = None

def initialize_logger():
    """Initialize the logger with file and console handlers."""
    global _logger
    
    experiment_name = get_run_params("EXPERIMENT_NAME")
    log_dir = Path(get_run_params("OUTPUT_BASE_DIR")) / experiment_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    _logger = logging.getLogger("ethiopic_pipeline")
    _logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    _logger.handlers.clear()
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    _logger.addHandler(console)
    
    # File handler
    log_file = log_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    _logger.addHandler(file_handler)
    
    _logger.info(f"Logger initialized. Log file: {log_file}")

def get_logger():
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        initialize_logger()
    return _logger