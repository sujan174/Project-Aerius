"""
UI Module - Ultra Modern Edition v4.0

A comprehensive, world-class terminal UI system with:
- Unified design language
- Beautiful components
- Rich interactions
- Professional polish

Components:
- EnhancedTerminalUI: Main terminal interface
- InteractiveActionEditor: Parameter editing
- NotificationManager: User feedback system
- DesignSystem: Unified visual language

Author: AI System
Version: 4.0
"""

# Design System (Foundation)
from ui.design_system import (
    ds,
    DesignSystem,
    ColorPalette,
    Typography,
    Spacing,
    BoxStyles,
    Animation,
    Icons,
    Semantic,
    build_status_text,
    build_badge,
    build_divider,
    build_key_value,
    Layout
)

# Main UI Components
from ui.enhanced_terminal_ui import EnhancedTerminalUI, enhanced_ui
from ui.interactive_editor import InteractiveActionEditor
from ui.notifications import NotificationManager, ProgressNotification, notifications

__all__ = [
    # Design System
    'ds',
    'DesignSystem',
    'ColorPalette',
    'Typography',
    'Spacing',
    'BoxStyles',
    'Animation',
    'Icons',
    'Semantic',
    'build_status_text',
    'build_badge',
    'build_divider',
    'build_key_value',
    'Layout',

    # UI Components
    'EnhancedTerminalUI',
    'enhanced_ui',
    'InteractiveActionEditor',
    'NotificationManager',
    'ProgressNotification',
    'notifications',
]

__version__ = '4.0.0'
