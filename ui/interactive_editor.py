"""
Interactive Action Editor v4.0 - Ultra Modern Edition

Beautiful, intuitive interface for editing action parameters before execution.

Features:
- Rich visual previews
- Real-time validation
- Context-aware help
- Smooth workflows

Author: AI System
Version: 4.0
"""

from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich.align import Align
from rich.padding import Padding
from rich.syntax import Syntax
from rich import box

# Import from orchestration (will be available at runtime)
try:
    from orchestration.actions import Action, ActionType, RiskLevel, FieldInfo
except ImportError:
    # Fallback for type hints
    Action = None
    ActionType = None
    RiskLevel = None
    FieldInfo = None

from ui.design_system import ds


class InteractiveActionEditor:
    """
    Ultra-modern interactive editor for action parameters.

    Features:
    - Beautiful visual hierarchy
    - Field-by-field editing
    - Real-time validation
    - Rich previews
    - Helpful hints and examples
    - Undo support
    """

    def __init__(self):
        self.console = ds.get_console()

    def edit_action(self, action) -> Optional[Dict[str, Any]]:
        """
        Main editing interface for an action.

        Shows:
        1. Action summary
        2. Current parameters
        3. Interactive editor for each field
        4. Review of changes

        Returns:
            Dict of edits, or None if cancelled
        """
        self._show_action_overview(action)

        # Collect editable fields
        editable_fields = {
            fname: finfo for fname, finfo in action.field_info.items()
            if finfo.editable
        }

        if not editable_fields:
            self._show_warning("This action has no editable fields")
            Prompt.ask(f"\n[{ds.colors.text_tertiary}]Press Enter to continue[/]", default="")
            return None

        edits = {}

        # Show current parameters
        self._show_current_parameters(action, editable_fields)

        # Ask what to do
        self.console.print()
        options_table = Table(
            show_header=False,
            border_style="dim",
            box=None,
            padding=(0, 2),
            show_edge=False
        )

        options_table.add_row(
            f"[{ds.colors.accent_amber}]e[/]",
            f"[{ds.colors.accent_amber}]{ds.icons.edit} Edit parameters[/]"
        )
        options_table.add_row(
            f"[{ds.colors.success}]a[/]",
            f"[{ds.colors.success}]{ds.icons.success} Approve as-is[/]"
        )
        options_table.add_row(
            f"[{ds.colors.error}]c[/]",
            f"[{ds.colors.error}]{ds.icons.delete} Cancel this action[/]"
        )

        self.console.print(options_table)

        choice = Prompt.ask(
            f"\n[bold {ds.colors.primary_500}]{ds.icons.chevron_right} Your choice[/]",
            choices=["e", "a", "c"],
            default="a"
        )

        if choice == 'a':
            return {}  # Approve with no edits
        elif choice == 'c':
            return None  # Cancel
        elif choice != 'e':
            return None

        # Interactive field-by-field editor
        for field_name, field_info in editable_fields.items():
            edit_result = self._edit_field(field_name, field_info, action)

            if edit_result is not None:
                edits[field_name] = edit_result

        # Show summary of changes
        if edits:
            self._show_edit_summary(action, edits)

            confirm = Prompt.ask(
                f"\n[bold {ds.colors.success}]{ds.icons.success} Apply these changes?[/]",
                choices=["y", "n"],
                default="y"
            )

            if confirm == 'y':
                return edits

        return None

    def _edit_field(
        self,
        field_name: str,
        field_info,
        action
    ) -> Optional[Any]:
        """
        Edit a single field with beautiful interface.

        Returns:
            New value, or None if skipped
        """
        self.console.clear()

        # Header
        header = Text()
        header.append(f"{ds.icons.edit} ", style=ds.colors.accent_amber)
        header.append(f"Editing: {field_info.display_label}", style=f"bold {ds.colors.accent_amber}")

        self.console.rule(header, style=ds.colors.accent_amber)
        self.console.print()

        # Build field card
        field_content = Text()

        # Description
        field_content.append(f"{ds.icons.info} Description:\n", style=f"bold {ds.semantic.component['label']}")
        field_content.append(f"  {field_info.description}\n\n", style=ds.colors.text_secondary)

        # Current value
        field_content.append(f"{ds.icons.file} Current Value:\n", style=f"bold {ds.semantic.component['label']}")

        current_val = field_info.current_value

        if field_info.field_type == 'text' and isinstance(current_val, str) and '\n' in current_val:
            # Multi-line text
            lines = current_val.split('\n')
            for i, line in enumerate(lines[:5], 1):
                field_content.append(f"  {i}. {line}\n", style=ds.colors.text_primary)

            if len(lines) > 5:
                field_content.append(f"  ... ({len(lines)-5} more lines)\n", style=f"dim {ds.colors.text_tertiary}")
        else:
            # Single value
            field_content.append(f"  {current_val}\n", style=f"bold {ds.colors.accent_teal}")

        field_content.append("\n")

        # Constraints
        if field_info.constraints.allowed_values:
            field_content.append(f"{ds.icons.star} Allowed Values:\n", style=f"bold {ds.semantic.component['label']}")
            for val in field_info.constraints.allowed_values[:5]:
                field_content.append(f"  {ds.icons.bullet} {val}\n", style=ds.colors.text_secondary)
            if len(field_info.constraints.allowed_values) > 5:
                field_content.append(f"  ... and {len(field_info.constraints.allowed_values)-5} more\n", style=f"dim {ds.colors.text_tertiary}")
            field_content.append("\n")

        if field_info.constraints.min_length or field_info.constraints.max_length:
            field_content.append(f"{ds.icons.info} Length:\n", style=f"bold {ds.semantic.component['label']}")
            if field_info.constraints.min_length:
                field_content.append(f"  Minimum: {field_info.constraints.min_length} characters\n", style=ds.colors.text_secondary)
            if field_info.constraints.max_length:
                field_content.append(f"  Maximum: {field_info.constraints.max_length} characters\n", style=ds.colors.text_secondary)
            field_content.append("\n")

        # Examples
        if field_info.examples:
            field_content.append(f"{ds.icons.sparkle} Examples:\n", style=f"bold {ds.semantic.component['label']}")
            for ex in field_info.examples[:3]:
                field_content.append(f"  {ds.icons.bullet} {ex}\n", style=f"dim {ds.colors.text_secondary}")
            field_content.append("\n")

        # Display panel
        panel = Panel(
            field_content,
            border_style=ds.colors.accent_amber,
            box=ds.box_styles.panel_default,
            padding=ds.spacing.padding_md
        )
        self.console.print(panel)

        # Get new value
        self.console.print()

        if field_info.field_type == 'text':
            help_text = Text()
            help_text.append(f"{ds.icons.info} ", style=ds.colors.info)
            help_text.append("Enter new value (type ", style=ds.colors.text_secondary)
            help_text.append("END", style=f"bold {ds.colors.accent_teal}")
            help_text.append(" on a new line when done, or ", style=ds.colors.text_secondary)
            help_text.append("SKIP", style=f"bold {ds.colors.warning}")
            help_text.append(" to keep current):", style=ds.colors.text_secondary)

            self.console.print(help_text)
            self.console.print()

            new_value = self._read_multiline()

            if new_value == 'SKIP':
                return None
        else:
            new_value = Prompt.ask(
                f"[{ds.colors.accent_amber}]New value[/] (or Enter to skip)",
                default=""
            )

            if not new_value:
                return None

        # Validate
        is_valid, error = field_info.constraints.validate(new_value)

        if not is_valid:
            self.console.print()
            error_text = Text()
            error_text.append(f"{ds.icons.error} ", style=ds.colors.error)
            error_text.append("Validation error: ", style=ds.colors.error)
            error_text.append(error, style=ds.colors.error_light)

            panel = Panel(
                error_text,
                border_style=ds.colors.error,
                box=ds.box_styles.minimal,
                padding=ds.spacing.padding_sm
            )
            self.console.print(panel)

            retry = Prompt.ask(
                f"\n[{ds.colors.warning}]Try again?[/]",
                choices=["y", "n"],
                default="y"
            )

            if retry == 'y':
                return self._edit_field(field_name, field_info, action)
            else:
                return None

        # Show validation success
        self.console.print()
        success = Text()
        success.append(f"{ds.icons.success} ", style=ds.colors.success)
        success.append("Valid!", style=ds.colors.success)
        self.console.print(success)

        # Preview change
        self.console.print()
        preview = Text()
        preview.append(f"{ds.icons.magnifying_glass} Preview:\n\n", style=f"bold {ds.colors.accent_teal}")

        preview.append("  Old: ", style=ds.colors.text_tertiary)
        preview.append(f"{str(current_val)[:80]}\n", style=f"dim {ds.colors.text_tertiary}")

        preview.append("  New: ", style=ds.colors.accent_teal)
        preview.append(f"{str(new_value)[:80]}", style=f"bold {ds.colors.accent_teal}")

        panel = Panel(
            preview,
            border_style=ds.colors.accent_teal,
            box=ds.box_styles.panel_default,
            padding=ds.spacing.padding_md
        )
        self.console.print(panel)

        # Confirm change
        confirm = Prompt.ask(
            f"\n[bold {ds.colors.success}]{ds.icons.success} Keep this change?[/]",
            choices=["y", "n"],
            default="y"
        )

        if confirm == 'y':
            return new_value
        else:
            return None

    def _read_multiline(self) -> str:
        """Read multi-line input until user types 'END'"""
        lines = []

        while True:
            try:
                line = input()
                if line == 'END':
                    break
                if line == 'SKIP':
                    return 'SKIP'
                lines.append(line)
            except EOFError:
                break

        return '\n'.join(lines)

    def _show_action_overview(self, action):
        """Show beautiful action overview"""
        self.console.clear()

        # Risk indicator
        risk_map = {
            'HIGH': (ds.icons.risk_high, ds.colors.error),
            'MEDIUM': (ds.icons.risk_medium, ds.colors.warning),
            'LOW': (ds.icons.risk_low, ds.colors.success),
        }

        risk_str = action.risk_level.value if hasattr(action.risk_level, 'value') else str(action.risk_level)
        risk_icon, risk_color = risk_map.get(risk_str.upper(), ('⚪', ds.colors.text_secondary))

        # Header
        header = Text()
        header.append(f"{risk_icon} ", style=risk_color)
        header.append(f"Action Review: {action.action_type.value.upper()}", style=f"bold {ds.colors.primary_500}")

        self.console.rule(header, style=ds.colors.primary_500)
        self.console.print()

        # Details
        details_content = Text()

        details_content.append(f"{ds.icons.agent} Agent:  ", style=ds.semantic.component['label'])
        details_content.append(f"{action.agent_name.replace('_', ' ').title()}\n", style=ds.colors.accent_purple)

        details_content.append(f"{ds.icons.gear} Type:  ", style=ds.semantic.component['label'])
        details_content.append(f"{action.action_type.value}\n", style=ds.colors.text_primary)

        details_content.append(f"{ds.icons.warning} Risk:  ", style=ds.semantic.component['label'])
        details_content.append(f"{risk_str.upper()}\n", style=risk_color)

        # Show enriched details if available
        if hasattr(action, 'details') and action.details:
            details = action.details

            if 'description' in details:
                details_content.append(f"\n\n{ds.icons.file} What Will Happen:\n", style=f"bold {ds.semantic.component['label']}")
                details_content.append(f"  {details['description']}", style=ds.colors.text_secondary)

            if 'channel' in details:
                details_content.append(f"\n\n{ds.icons.link} Channel:  ", style=ds.semantic.component['label'])
                details_content.append(details['channel'], style=ds.colors.accent_teal)

            if 'project' in details:
                details_content.append(f"\n{ds.icons.folder} Project:  ", style=ds.semantic.component['label'])
                details_content.append(details['project'], style=ds.colors.accent_teal)

        panel = Panel(
            details_content,
            border_style=risk_color,
            box=ds.box_styles.panel_emphasis,
            padding=ds.spacing.padding_lg
        )
        self.console.print(panel)
        self.console.print()

    def _show_current_parameters(self, action, editable_fields: Dict):
        """Show all current parameter values"""
        self.console.print()

        header = Text()
        header.append(f"{ds.icons.file} ", style=ds.colors.accent_teal)
        header.append("Current Parameters", style=f"bold {ds.colors.accent_teal}")

        self.console.print(header)
        self.console.print()

        # Create table
        params_table = Table(
            show_header=True,
            header_style=f"bold {ds.colors.text_primary}",
            border_style=ds.colors.border,
            box=ds.box_styles.table_default,
            padding=(0, 2)
        )

        params_table.add_column("Field", style=f"bold {ds.colors.accent_purple}")
        params_table.add_column("Value", style=ds.colors.text_secondary)

        for fname, finfo in editable_fields.items():
            value = finfo.current_value

            # Truncate long values
            if isinstance(value, str) and len(value) > 60:
                display_val = value[:57] + "..."
            else:
                display_val = str(value)

            params_table.add_row(finfo.display_label, display_val)

        self.console.print(params_table)

    def _show_edit_summary(self, action, edits: Dict[str, Any]):
        """Show beautiful summary of all changes"""
        self.console.clear()

        header = Text()
        header.append(f"{ds.icons.sparkle} ", style=ds.colors.success)
        header.append("Edit Summary", style=f"bold {ds.colors.success}")

        self.console.rule(header, style=ds.colors.success)
        self.console.print()

        # Build summary
        summary_content = Text()

        for field_name, new_value in edits.items():
            field_info = action.field_info[field_name]
            old_value = field_info.current_value

            summary_content.append(f"{ds.icons.edit} {field_info.display_label}:\n", style=f"bold {ds.colors.accent_purple}")

            summary_content.append("  Was: ", style=ds.colors.text_tertiary)
            summary_content.append(f"{str(old_value)[:70]}\n", style=f"dim {ds.colors.text_tertiary}")

            summary_content.append("  Now: ", style=ds.colors.accent_teal)
            summary_content.append(f"{str(new_value)[:70]}\n\n", style=f"bold {ds.colors.accent_teal}")

        panel = Panel(
            summary_content,
            title=f"[{ds.colors.success}]{ds.icons.success} Changes[/]",
            border_style=ds.colors.success,
            box=ds.box_styles.panel_default,
            padding=ds.spacing.padding_lg
        )

        self.console.print(panel)

    def _show_warning(self, message: str):
        """Show warning message"""
        warning = Text()
        warning.append(f"{ds.icons.warning} ", style=ds.colors.warning)
        warning.append(message, style=ds.colors.warning)

        panel = Panel(
            warning,
            border_style=ds.colors.warning,
            box=ds.box_styles.minimal,
            padding=ds.spacing.padding_sm
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()
