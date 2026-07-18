"""
Common utility functions, config loaders, and helper utilities.
"""

import os
import logging

def setup_logger(name: str = "glance_retrieval", log_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a simple console logger for the project.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
