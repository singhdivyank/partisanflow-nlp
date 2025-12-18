"""
Centralise logging configurations.
This module is to be imported by all other modules that need logging.
"""

import logging
import sys

_logging_configured = False

def setup_logger(name: str = None) -> logging.Logger:
    """
    Get or create a logger with standardized configuration.

    Parameters:
    name (str, optional): Name of the logger. If None, the root logger is used.

    Returns:
    logging.Logger: Configured logger instance.
    """

    global _logging_configured

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger