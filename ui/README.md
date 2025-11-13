# UI System v4.0 - Ultra Modern Edition

A world-class terminal UI system with unified design language and professional polish.

## 🎨 Overview

The UI system provides a comprehensive set of components for building beautiful, consistent terminal interfaces. Every component follows the same design language, ensuring a cohesive user experience.

### Features

- **Unified Design Language**: Consistent colors, typography, spacing, and components
- **Beautiful Components**: Rich, modern UI elements with smooth interactions
- **Professional Polish**: Attention to detail in every interaction
- **Accessibility**: High contrast, readable typography, clear visual hierarchy
- **Extensible**: Easy to add new components while maintaining consistency

## 📦 Components

### 1. Design System (`design_system.py`)

The foundation of the entire UI system. Provides:

#### Color Palette
- **Primary Colors**: Brand identity (`primary_500` to `primary_900`)
- **Accent Colors**: Purple, pink, teal, amber for highlights
- **Semantic Colors**: Success, warning, error, info (with light/dark variants)
- **Neutral Colors**: Gray scale (`gray_50` to `gray_900`)
- **Syntax Highlighting**: Consistent code block colors

#### Typography
- Monospace font family
- Size scale (xs to 3xl)
- Weight variants (normal, medium, semibold, bold)

#### Spacing System
- Consistent spacing scale (xs to xxxl)
- Padding presets

#### Icons
- Comprehensive unicode icon library
- Status icons (success, error, warning, info)
- Action icons (send, edit, delete, copy)
- Entity icons (user, agent, file, folder)

#### Usage Example:
```python
from ui.design_system import ds

# Use colors
console.print(f"[{ds.colors.success}]Success![/]")

# Use icons
print(f"{ds.icons.success} Operation completed")

# Use semantic styles
panel = Panel(..., border_style=ds.semantic.border['success'])
```

### 2. Enhanced Terminal UI (`enhanced_terminal_ui.py`)

The main interface component providing:

#### Features
- **Stunning Header**: ASCII art banner with gradient colors
- **Session Info**: Track time, stats, and session ID
- **Agent Discovery**: Beautiful table showing loaded agents
- **Chat Interface**: Modern prompt and response display
- **Tool Calls**: Visual feedback for agent operations
- **Help System**: Comprehensive keyboard shortcuts
- **Session Dashboard**: Analytics and statistics
- **Notifications**: Context-aware user feedback

#### Usage Example:
```python
from ui import EnhancedTerminalUI

ui = EnhancedTerminalUI(verbose=False)

# Show header
ui.print_header(session_id="abc-123")

# Show agent table
ui.print_agent_table(loaded_agents, failed_agents)

# User interaction
ui.print_prompt()
user_input = input()

ui.print_thinking()
# ... process ...

ui.print_response("Here's the result...")

# Tool calls
ui.print_tool_call("slack", "send_message")
ui.print_tool_result(success=True, message="Sent!")

# Session stats
ui.print_session_stats()
ui.print_goodbye()
```

### 3. Confirmation UI (`confirmation_ui.py`)

Beautiful action confirmation dialogs:

#### Features
- **Action Cards**: Rich preview of what will be executed
- **Risk Indicators**: Color-coded risk levels (low/medium/high)
- **Batch Review**: Review multiple actions at once
- **Inline Editing**: Edit actions before confirmation
- **Keyboard Controls**: Intuitive shortcuts (a=approve, e=edit, r=review, c=cancel)

#### Usage Example:
```python
from ui import ConfirmationUI

ui = ConfirmationUI()

# Show batch
ui.present_batch(actions)

# Collect decisions
decisions = ui.collect_decisions(actions)

# Process decisions
for action_id, confirmed in decisions['confirmed'].items():
    # Execute confirmed actions
    pass

for action_id, edits in decisions['edited'].items():
    # Apply edits
    pass
```

### 4. Interactive Editor (`interactive_editor.py`)

Field-by-field parameter editing:

#### Features
- **Action Overview**: Beautiful summary of what's being edited
- **Current Parameters**: Table of all editable fields
- **Field Editor**: Interactive editing with validation
- **Real-time Validation**: Immediate feedback on constraints
- **Edit Preview**: Before/after comparison
- **Multi-line Support**: For text fields
- **Examples & Hints**: Contextual help

#### Usage Example:
```python
from ui import InteractiveActionEditor

editor = InteractiveActionEditor()

# Edit action
edits = editor.edit_action(action)

if edits:
    # Apply edits to action
    for field, new_value in edits.items():
        action.update_field(field, new_value)
```

### 5. Notification System (`notifications.py`)

Beautiful user feedback:

#### Features
- **Toast Notifications**: Quick, non-intrusive messages
- **Typed Notifications**: Success, error, warning, info
- **Custom Notifications**: Full control over appearance
- **Progress Notifications**: For long-running operations
- **Notification History**: Track all notifications

#### Usage Example:
```python
from ui import notifications

# Show notifications
notifications.success("Operation completed!")
notifications.error("Something went wrong")
notifications.warning("Please review this action")
notifications.info("Tip: Use keyboard shortcuts")

# Custom notification
notifications.custom(
    message="Custom message",
    title="Custom Title",
    icon="🎉",
    color=ds.colors.accent_purple
)

# Progress notification
progress = notifications.progress_start("Loading...")
progress.update("Loading agents...")
progress.complete("All agents loaded!")
```

## 🎯 Design Principles

### 1. Consistency
- All components use the same design system
- Colors, typography, and spacing are consistent
- Icons and symbols have clear meanings

### 2. Visual Hierarchy
- Important information stands out
- Secondary information is muted
- Clear grouping and spacing

### 3. Feedback
- Immediate visual feedback for all actions
- Clear success/error states
- Loading states for async operations

### 4. Accessibility
- High contrast ratios
- Readable font sizes
- Clear error messages
- Keyboard navigation

### 5. Polish
- Smooth transitions
- Attention to detail
- Professional appearance

## 🎨 Color Usage Guide

### When to Use Each Color

| Color | Usage | Example |
|-------|-------|---------|
| **Primary** (`primary_500`) | Main brand color, headers, key actions | "AI Workspace Orchestrator" header |
| **Success** (`success`) | Successful operations, confirmations | "✓ Action completed" |
| **Error** (`error`) | Errors, failures, destructive actions | "✗ Failed to send message" |
| **Warning** (`warning`) | Warnings, confirmations needed | "⚠ Confirmation Required" |
| **Info** (`info`) | Informational messages, tips | "ℹ Tip: Use keyboard shortcuts" |
| **Accent Purple** (`accent_purple`) | Agents, secondary highlights | Agent names |
| **Accent Teal** (`accent_teal`) | Values, data, links | Session IDs, usernames |
| **Accent Amber** (`accent_amber`) | Edit mode, modifications | "✎ Edit parameters" |

### Text Colors

- `text_primary`: Main content
- `text_secondary`: Supporting text
- `text_tertiary`: Hints, labels
- `text_disabled`: Disabled elements

## 🚀 Quick Start

### Basic Setup

```python
from ui import EnhancedTerminalUI, ConfirmationUI, notifications, ds

# Create UI instance
ui = EnhancedTerminalUI(verbose=False)

# Show header
ui.print_header(session_id="my-session-123")

# Show notification
notifications.success("Welcome to the orchestrator!")

# User interaction
ui.print_prompt()
user_input = input()

ui.print_thinking()
# ... process user input ...

ui.print_response("Here's your response...")
```

### Advanced Usage

```python
from ui import EnhancedTerminalUI, ConfirmationUI, InteractiveActionEditor

ui = EnhancedTerminalUI()
confirmation_ui = ConfirmationUI()
editor = InteractiveActionEditor()

# Show agents
ui.print_agent_table(agents)

# Confirm actions
confirmation_ui.present_batch(pending_actions)
decisions = confirmation_ui.collect_decisions(pending_actions)

# Edit specific action
if action_id in decisions['edited']:
    edits = editor.edit_action(action)
    # Apply edits...
```

## 📊 Component Hierarchy

```
ui/
├── design_system.py       # Foundation (colors, typography, icons)
│   └── DesignSystem
│       ├── ColorPalette
│       ├── Typography
│       ├── Spacing
│       ├── Icons
│       └── Semantic
│
├── enhanced_terminal_ui.py  # Main interface
│   └── EnhancedTerminalUI
│       ├── print_header()
│       ├── print_agent_table()
│       ├── print_prompt()
│       ├── print_response()
│       ├── print_tool_call()
│       ├── print_session_stats()
│       └── print_help()
│
├── confirmation_ui.py      # Action confirmation
│   ├── ConfirmationUI
│   │   ├── present_batch()
│   │   └── collect_decisions()
│   └── ConfirmationModal
│       └── ask()
│
├── interactive_editor.py   # Parameter editing
│   └── InteractiveActionEditor
│       └── edit_action()
│
└── notifications.py        # User feedback
    ├── NotificationManager
    │   ├── success()
    │   ├── error()
    │   ├── warning()
    │   └── info()
    └── ProgressNotification
```

## 🎨 Customization

### Creating Custom Components

```python
from ui.design_system import ds
from rich.panel import Panel

# Use the design system for consistency
def create_custom_panel(content):
    return Panel(
        content,
        border_style=ds.colors.primary_500,
        box=ds.box_styles.panel_default,
        padding=ds.spacing.padding_md
    )
```

### Extending the Design System

```python
from ui.design_system import ds

# Add custom colors to your code
MY_CUSTOM_COLOR = "#FF6B9D"

# Use with design system patterns
panel = Panel(
    ...,
    border_style=MY_CUSTOM_COLOR,
    box=ds.box_styles.panel_default
)
```

## 📝 Best Practices

1. **Always use the design system**: Don't hardcode colors or styles
2. **Use semantic styles**: Use `ds.semantic.border['success']` instead of `ds.colors.success`
3. **Consistent icons**: Use the icon library for visual consistency
4. **Proper spacing**: Use `ds.spacing` for consistent padding/margins
5. **Error handling**: Always show user-friendly error messages
6. **Loading states**: Show progress for operations >1 second
7. **Accessibility**: Ensure sufficient contrast and clear messaging

## 🔧 Troubleshooting

### Common Issues

**Issue**: Colors not showing
- **Solution**: Ensure terminal supports 24-bit color. Try `export COLORTERM=truecolor`

**Issue**: Unicode icons not displaying
- **Solution**: Use a font with good Unicode support (e.g., Fira Code, JetBrains Mono)

**Issue**: Layout issues
- **Solution**: Ensure terminal width is at least 80 characters

## 📚 References

- [Rich Library Documentation](https://rich.readthedocs.io/)
- Terminal UI Best Practices
- Accessibility Guidelines for Terminal UIs

## 📄 License

Part of the AI Workspace Orchestrator project.

---

**Version**: 4.0.0
**Last Updated**: 2025-01-13
**Author**: AI System
