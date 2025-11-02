"""
Production configuration for the orchestration system.
All hardcoded values moved here for easy tuning.
"""

import os
from typing import Dict, Any

class Config:
    """Central configuration for all system components."""

    # Agent Operation Timeouts (seconds)
    AGENT_OPERATION_TIMEOUT = float(os.getenv('AGENT_TIMEOUT', '120.0'))
    ENRICHMENT_TIMEOUT = float(os.getenv('ENRICHMENT_TIMEOUT', '5.0'))
    LLM_OPERATION_TIMEOUT = float(os.getenv('LLM_TIMEOUT', '30.0'))

    # Confirmation Queue Settings
    BATCH_TIMEOUT_MS = int(os.getenv('BATCH_TIMEOUT_MS', '1000'))
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', '10'))
    MAX_PENDING_ACTIONS = int(os.getenv('MAX_PENDING_ACTIONS', '100'))

    # Retry Configuration
    MAX_RETRY_ATTEMPTS = int(os.getenv('MAX_RETRIES', '3'))
    RETRY_BACKOFF_FACTOR = float(os.getenv('RETRY_BACKOFF', '2.0'))
    INITIAL_RETRY_DELAY = float(os.getenv('INITIAL_RETRY_DELAY', '1.0'))

    # Input Validation
    MAX_INSTRUCTION_LENGTH = int(os.getenv('MAX_INSTRUCTION_LENGTH', '10000'))
    MAX_PARAMETER_VALUE_LENGTH = int(os.getenv('MAX_PARAM_LENGTH', '5000'))

    # Enrichment Settings
    REQUIRE_ENRICHMENT_FOR_HIGH_RISK = os.getenv('REQUIRE_ENRICHMENT_HIGH_RISK', 'true').lower() == 'true'
    FAIL_OPEN_ON_ENRICHMENT_ERROR = os.getenv('FAIL_OPEN_ENRICHMENT', 'false').lower() == 'true'

    # Security Settings
    ENABLE_INPUT_SANITIZATION = os.getenv('ENABLE_SANITIZATION', 'true').lower() == 'true'
    MAX_REGEX_PATTERN_LENGTH = int(os.getenv('MAX_REGEX_LENGTH', '1000'))

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    VERBOSE = os.getenv('VERBOSE', 'false').lower() == 'true'

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Return all config values as a dictionary."""
        return {
            'agent_timeout': cls.AGENT_OPERATION_TIMEOUT,
            'enrichment_timeout': cls.ENRICHMENT_TIMEOUT,
            'llm_timeout': cls.LLM_OPERATION_TIMEOUT,
            'batch_timeout_ms': cls.BATCH_TIMEOUT_MS,
            'max_batch_size': cls.MAX_BATCH_SIZE,
            'max_retry_attempts': cls.MAX_RETRY_ATTEMPTS,
            'max_instruction_length': cls.MAX_INSTRUCTION_LENGTH,
            'verbose': cls.VERBOSE,
        }
