"""
Unified Observability System

Simplified logging and context management for the orchestrator.
Uses SimpleSessionLogger for file output, console for debugging.

Author: AI System
Version: 3.0 - Minimal
"""

import os
from typing import Optional
from pathlib import Path

from .logging_config import (
    configure_logging,
    get_logger,
    LogContext
)


# ============================================================================
# OBSERVABILITY SYSTEM
# ============================================================================

class ObservabilitySystem:
    """
    Unified observability system - minimal version.

    Provides:
    - Console logging configuration
    - Session context management
    """

    def __init__(
        self,
        session_id: str,
        log_dir: str = "logs",
        log_level: str = "INFO",
        verbose: bool = False
    ):
        """
        Initialize observability system

        Args:
            session_id: Session ID
            log_dir: Base directory for logs
            log_level: Logging level
            verbose: Enable verbose logging
        """
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.verbose = verbose

        # Initialize console logging only
        self._init_logging(log_level)

        # Set session context
        LogContext.set_session(session_id)

        # Get main logger
        self.logger = get_logger(__name__)
        if verbose:
            self.logger.info(f"Observability system initialized for session: {session_id}")

    def _init_logging(self, log_level: str):
        """Initialize logging system - console only"""
        configure_logging({
            'log_level': log_level,
            'log_dir': str(self.log_dir),
            'enable_file_logging': False,
            'enable_json_logging': False,
            'enable_console': True,
            'enable_colors': True
        })

    def cleanup(self):
        """Cleanup observability resources"""
        if self.verbose:
            self.logger.info("Observability system cleanup complete")


# ============================================================================
# GLOBAL OBSERVABILITY INSTANCE
# ============================================================================

_global_observability: Optional[ObservabilitySystem] = None


def initialize_observability(
    session_id: str,
    log_dir: str = "logs",
    log_level: Optional[str] = None,
    verbose: bool = False,
    **kwargs  # Accept but ignore legacy parameters
) -> ObservabilitySystem:
    """
    Initialize the global observability system

    Args:
        session_id: Session ID
        log_dir: Log directory
        log_level: Log level (defaults to env var or INFO)
        verbose: Enable verbose logging

    Returns:
        ObservabilitySystem instance
    """
    global _global_observability

    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", "INFO")

    _global_observability = ObservabilitySystem(
        session_id=session_id,
        log_dir=log_dir,
        log_level=log_level,
        verbose=verbose
    )

    return _global_observability


def get_observability() -> Optional[ObservabilitySystem]:
    """Get the global observability system"""
    return _global_observability
