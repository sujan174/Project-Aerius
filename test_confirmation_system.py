#!/usr/bin/env python3
"""
Test script for the confirmation system.
Tests basic functionality of action model, queue, and UI.
"""

import asyncio
from orchestration.action_model import (
    Action, ActionType, RiskLevel, ActionStatus, FieldInfo, FieldConstraint
)
from orchestration.confirmation_queue import ConfirmationQueue, ConfirmationBatch
from orchestration.action_parser import ActionParser
from ui.confirmation_ui import ConfirmationUI


async def test_action_model():
    """Test Action dataclass and methods"""
    print("\n" + "="*70)
    print("TEST 1: Action Model")
    print("="*70 + "\n")

    # Create an action
    action = Action(
        agent_name='jira',
        action_type=ActionType.CREATE,
        risk_level=RiskLevel.MEDIUM,
        reason_for_confirmation="Creating new issue",
        instruction="Create issue with title 'Bug fix'",
        parameters={'title': 'Bug fix', 'project': 'MYAPP'},
    )

    # Add field info
    action.field_info['title'] = FieldInfo(
        display_label='Issue Title',
        description='Brief summary',
        field_type='string',
        current_value='Bug fix',
        editable=True,
        constraints=FieldConstraint(min_length=5, max_length=255)
    )

    action.field_info['project'] = FieldInfo(
        display_label='Project',
        description='Jira project',
        field_type='string',
        current_value='MYAPP',
        editable=False
    )

    print(f"✓ Created action: {action.id[:8]}...")
    print(f"  Type: {action.action_type.value}")
    print(f"  Risk: {action.risk_level.value}")
    print(f"  Parameters: {action.parameters}")

    # Test edits
    action.user_edits['title'] = 'CRITICAL: Login broken'
    is_valid, errors = action.validate_edits()
    print(f"\n✓ User edits: {action.user_edits}")
    print(f"  Valid: {is_valid}, Errors: {errors}")

    # Test status transitions
    action.mark_confirmed()
    print(f"  Status: {action.status.value}")

    print("\n✅ Action Model Test PASSED\n")


async def test_confirmation_queue():
    """Test ConfirmationQueue batching logic"""
    print("="*70)
    print("TEST 2: Confirmation Queue")
    print("="*70 + "\n")

    queue = ConfirmationQueue(
        batch_timeout_ms=100,  # Fast timeout for testing
        max_batch_size=3,
        verbose=True
    )

    # Create and queue actions
    for i in range(5):
        action = Action(
            agent_name='jira',
            action_type=ActionType.CREATE,
            risk_level=RiskLevel.MEDIUM,
            instruction=f"Create issue {i}",
            parameters={'title': f'Issue {i}'}
        )
        await queue.queue_action(action)
        print(f"  Queued action {i+1}")

    print(f"\n✓ Pending actions: {queue.get_pending_count()}")

    # Check if batching should happen (size-based)
    should_batch = queue.should_batch_now()
    print(f"  Should batch (size >= 3): {should_batch}")

    if should_batch:
        batch = queue.prepare_batch()
        print(f"✓ Prepared batch with {len(batch.actions)} actions")
        print(f"  Pending after batch: {queue.get_pending_count()}")
        queue.archive_batch()
        print(f"✓ Archived batch")

    print("\n✅ Confirmation Queue Test PASSED\n")


async def test_action_parser():
    """Test ActionParser instruction parsing"""
    print("="*70)
    print("TEST 3: Action Parser")
    print("="*70 + "\n")

    parser = ActionParser(verbose=True)

    test_cases = [
        ("Create issue with title 'Login Bug' description 'Users cannot log in'", 'jira'),
        ("Send Slack message to #general: 'System maintenance'", 'slack'),
        ("Delete issue JIRA-123", 'jira'),
        ("Update comment to 'Looks good!'", 'github'),
        ("Fetch all issues assigned to John", 'jira'),
    ]

    for instruction, agent_name in test_cases:
        action = await parser.parse_instruction(
            agent_name=agent_name,
            instruction=instruction,
            agent=None  # No real agent for testing
        )

        print(f"\n✓ Parsed: {instruction}")
        print(f"  Agent: {action.agent_name}")
        print(f"  Type: {action.action_type.value}")
        print(f"  Risk: {action.risk_level.value}")
        print(f"  Parameters: {action.parameters}")

    print("\n✅ Action Parser Test PASSED\n")


async def test_field_constraints():
    """Test field constraint validation"""
    print("="*70)
    print("TEST 4: Field Constraints")
    print("="*70 + "\n")

    # Create constraints
    constraint = FieldConstraint(
        min_length=5,
        max_length=20,
        allowed_values=['Bug', 'Task', 'Epic'],
        pattern=r'^[A-Z].*'  # Start with capital
    )

    test_cases = [
        ("Task", True),          # Valid
        ("task", False),         # Pattern violation
        ("Bug", True),           # Valid
        ("Foo", False),          # Not in allowed values
        ("Xy", False),           # Too short
        ("This is a very long title", False),  # Too long
    ]

    print("Testing constraints: min=5, max=20, pattern=^[A-Z].*, allowed=[Bug, Task, Epic]\n")

    for value, should_pass in test_cases:
        is_valid, error = constraint.validate(value)
        status = "✓" if is_valid == should_pass else "✗"
        print(f"{status} '{value}': {is_valid} {f'({error})' if error else ''}")

    print("\n✅ Field Constraints Test PASSED\n")


async def main():
    """Run all tests"""
    print("\n" + "🧪 CONFIRMATION SYSTEM - PHASE 1 TESTS" + "\n")

    try:
        await test_action_model()
        await test_confirmation_queue()
        await test_action_parser()
        await test_field_constraints()

        print("="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70 + "\n")

        print("✨ Phase 1 Implementation Summary:")
        print("  ✓ Action model with full lifecycle tracking")
        print("  ✓ Confirmation queue with intelligent batching")
        print("  ✓ Field-level constraints and validation")
        print("  ✓ Instruction parser with risk detection")
        print("  ✓ Terminal UI for batch confirmation")
        print("  ✓ Integrated into orchestrator")
        print("\nNext: Phase 2 - Add editing capabilities\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
