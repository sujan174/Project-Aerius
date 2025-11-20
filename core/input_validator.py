"""
Input validation and sanitization for production safety.
Prevents injection attacks and malformed data.
"""

from typing import Tuple, Optional
from config import Config

class InputValidator:
    """Validates and sanitizes user inputs."""

    @staticmethod
    def validate_instruction(instruction: str) -> Tuple[bool, Optional[str]]:
        """
        Validate instruction string.
        Returns: (is_valid, error_message)
        """
        if not instruction:
            return False, "Instruction cannot be empty"

        if not isinstance(instruction, str):
            return False, f"Instruction must be string, got {type(instruction)}"

        if len(instruction) > Config.MAX_INSTRUCTION_LENGTH:
            return False, f"Instruction exceeds max length of {Config.MAX_INSTRUCTION_LENGTH}"

        # Check for null bytes
        if '\x00' in instruction:
            return False, "Instruction contains null bytes"

        # Check for excessive special characters (potential injection)
        # Use a more lenient threshold that scales with instruction length
        # to allow legitimate code snippets, file paths, and quoted text
        special_count = sum(1 for c in instruction if c in ';\'"\\')
        max_allowed_special_chars = max(100, len(instruction) // 5)  # At least 100, or 20% of length
        if special_count > max_allowed_special_chars:
            return False, "Instruction contains excessive special characters"

        return True, None
