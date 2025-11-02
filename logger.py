"""
Structured logging for production debugging and monitoring.
"""

import logging
import sys
from typing import Optional
from config import Config

class Logger:
    """Centralized logging with consistent formatting."""

    _loggers = {}

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger with standard configuration."""
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)

        # Set log level from config
        log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(log_level)

        # Only add handler if logger doesn't have one
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        cls._loggers[name] = logger
        return logger


def get_logger(module_name: str) -> logging.Logger:
    """Convenience function to get a logger for a module."""
    return Logger.get_logger(module_name)
