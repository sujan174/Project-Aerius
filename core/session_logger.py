"""
Comprehensive Session Logger - Detailed logging for every chat session

This module provides comprehensive logging where each chat session has ONE TEXT file
that contains EVERYTHING in human-readable format:
- All user messages
- All assistant responses
- All intelligence processing (hybrid classification, confidence, risk)
- All agent calls with full details
- All errors and recovery attempts
- Complete intelligence decision-making process
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class LogEntry:
    """A single log entry"""
    timestamp: float
    timestamp_iso: str
    type: str  # Many types - see below
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SessionLogger:
    """
    Comprehensive session logger that writes everything to TWO files per session:
    1. A detailed TEXT file for human readability
    2. A JSON file for programmatic access

    Logs EVERYTHING:
    - All user messages
    - All assistant responses
    - Complete intelligence processing pipeline
    - Hybrid classification decisions (fast vs LLM path)
    - Intent detection and confidence scores
    - Entity extraction
    - Risk assessment
    - Context resolution
    - Agent calls with full details
    - Errors and recovery
    - Session metadata
    """

    def __init__(self, session_id: str, log_dir: str = "logs"):
        """
        Initialize session logger

        Args:
            session_id: Unique session identifier
            log_dir: Directory to store log files
        """
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Two log files: JSON and TXT
        self.json_file = self.log_dir / f"session_{session_id}.json"
        self.text_file = self.log_dir / f"session_{session_id}.txt"

        self.entries: List[LogEntry] = []
        self.session_start = time.time()

        # Session metadata
        self.metadata = {
            'session_id': session_id,
            'started_at': self.session_start,
            'started_at_iso': datetime.fromtimestamp(self.session_start).isoformat(),
        }

        # Initialize text log file
        self._write_text_header()

        # Load existing JSON log if it exists
        if self.json_file.exists():
            self._load_existing_log()
        else:
            self._write_initial_log()

    def _write_text_header(self):
        """Write text log header"""
        try:
            with open(self.text_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write(f"SESSION LOG: {self.session_id}\n")
                f.write(f"Started: {datetime.fromtimestamp(self.session_start).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 100 + "\n\n")
        except Exception as e:
            print(f"Warning: Failed to write text log header: {e}")

    def _write_initial_log(self):
        """Write initial JSON log file structure"""
        initial_data = {
            'metadata': self.metadata,
            'entries': []
        }
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2)

    def _load_existing_log(self):
        """Load existing JSON log file"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metadata = data.get('metadata', self.metadata)
                entries_data = data.get('entries', [])
                self.entries = [
                    LogEntry(
                        timestamp=e['timestamp'],
                        timestamp_iso=e['timestamp_iso'],
                        type=e['type'],
                        data=e['data']
                    ) for e in entries_data
                ]
        except Exception:
            # If loading fails, start fresh
            self.entries = []

    def _add_entry(self, entry_type: str, data: Dict[str, Any]):
        """Add a log entry to both JSON and text logs"""
        now = time.time()
        entry = LogEntry(
            timestamp=now,
            timestamp_iso=datetime.fromtimestamp(now).isoformat(),
            type=entry_type,
            data=data
        )
        self.entries.append(entry)
        self._write_to_text_log(entry)
        self._save_json()

    def _write_to_text_log(self, entry: LogEntry):
        """Write entry to human-readable text log"""
        try:
            with open(self.text_file, 'a', encoding='utf-8') as f:
                timestamp_str = datetime.fromtimestamp(entry.timestamp).strftime('%H:%M:%S')
                f.write(f"\n[{timestamp_str}] {entry.type.upper()}\n")
                f.write("-" * 100 + "\n")

                # Format based on entry type
                if entry.type == 'user_message':
                    f.write(f"USER: {entry.data.get('message', '')}\n")

                elif entry.type == 'assistant_response':
                    f.write(f"ASSISTANT: {entry.data.get('response', '')[:500]}...\n")

                elif entry.type == 'intelligence_classification':
                    f.write(f"🧠 INTELLIGENCE CLASSIFICATION:\n")
                    f.write(f"  Path Used: {entry.data.get('path_used', 'unknown')}\n")
                    f.write(f"  Latency: {entry.data.get('latency_ms', 0):.1f}ms\n")
                    f.write(f"  Confidence: {entry.data.get('confidence', 0):.2f}\n")
                    f.write(f"  Intents: {entry.data.get('intents', [])}\n")
                    f.write(f"  Entities: {entry.data.get('entity_count', 0)} found\n")
                    if entry.data.get('reasoning'):
                        f.write(f"  Reasoning: {entry.data.get('reasoning')}\n")

                elif entry.type == 'confidence_scoring':
                    f.write(f"📊 CONFIDENCE SCORING:\n")
                    f.write(f"  Overall Score: {entry.data.get('overall_score', 0):.2f}\n")
                    f.write(f"  Intent Clarity: {entry.data.get('intent_clarity', 0):.2f}\n")
                    f.write(f"  Entity Clarity: {entry.data.get('entity_clarity', 0):.2f}\n")
                    f.write(f"  Ambiguity: {entry.data.get('ambiguity', 0):.2f}\n")
                    f.write(f"  Recommendation: {entry.data.get('recommendation', 'N/A')}\n")

                elif entry.type == 'risk_assessment':
                    f.write(f"⚠️  RISK ASSESSMENT:\n")
                    f.write(f"  Risk Level: {entry.data.get('risk_level', 'UNKNOWN')}\n")
                    f.write(f"  Needs Confirmation: {entry.data.get('needs_confirmation', False)}\n")
                    f.write(f"  Reason: {entry.data.get('reason', 'N/A')}\n")

                elif entry.type == 'context_resolution':
                    f.write(f"🔗 CONTEXT RESOLUTION:\n")
                    for resolution in entry.data.get('resolutions', []):
                        f.write(f"  '{resolution.get('reference')}' → {resolution.get('resolved_to')}\n")

                elif entry.type == 'agent_call':
                    agent = entry.data.get('agent_name', 'unknown')
                    success = entry.data.get('success', False)
                    status = "✅" if success else "❌"
                    f.write(f"{status} AGENT CALL: {agent}\n")
                    f.write(f"  Instruction: {entry.data.get('instruction', '')[:200]}...\n")
                    if entry.data.get('duration_ms'):
                        f.write(f"  Duration: {entry.data.get('duration_ms'):.1f}ms\n")
                    if entry.data.get('error'):
                        f.write(f"  Error: {entry.data.get('error')}\n")

                elif entry.type == 'error':
                    f.write(f"❌ ERROR:\n")
                    f.write(f"  {entry.data.get('error', '')}\n")
                    if entry.data.get('error_type'):
                        f.write(f"  Type: {entry.data.get('error_type')}\n")

                elif entry.type == 'system':
                    f.write(f"⚙️  SYSTEM: {entry.data.get('event', '')}\n")
                    for key, value in entry.data.items():
                        if key != 'event':
                            f.write(f"  {key}: {value}\n")

                else:
                    # Generic formatting for other types
                    f.write(json.dumps(entry.data, indent=2) + "\n")

                f.write("\n")

        except Exception as e:
            print(f"Warning: Failed to write to text log: {e}")

    def _save_json(self):
        """Save JSON log file"""
        try:
            # Update metadata
            self.metadata['last_updated'] = time.time()
            self.metadata['last_updated_iso'] = datetime.now().isoformat()
            self.metadata['entry_count'] = len(self.entries)
            self.metadata['duration_seconds'] = time.time() - self.session_start

            # Write to file
            log_data = {
                'metadata': self.metadata,
                'entries': [e.to_dict() for e in self.entries]
            }

            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            # Fail silently to avoid breaking the application
            print(f"Warning: Failed to save JSON session log: {e}")

    def log_user_message(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a user message"""
        self._add_entry('user_message', {
            'message': message,
            'length': len(message),
            **(metadata or {})
        })

    def log_assistant_response(self, response: str, metadata: Optional[Dict[str, Any]] = None):
        """Log an assistant response"""
        self._add_entry('assistant_response', {
            'response': response,
            'length': len(response),
            **(metadata or {})
        })

    def log_agent_call(
        self,
        agent_name: str,
        instruction: str,
        response: Optional[str] = None,
        duration: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an agent call"""
        data = {
            'agent_name': agent_name,
            'instruction': instruction,
            'instruction_length': len(instruction),
            'success': success,
        }

        if response:
            data['response'] = response
            data['response_length'] = len(response)

        if duration is not None:
            data['duration_ms'] = duration * 1000 if duration < 1000 else duration

        if error:
            data['error'] = error

        if metadata:
            data.update(metadata)

        self._add_entry('agent_call', data)

    def log_function_call(
        self,
        function_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        success: bool = True,
        error: Optional[str] = None,
        duration: Optional[float] = None
    ):
        """Log a function call"""
        data = {
            'function_name': function_name,
            'success': success,
        }

        if arguments:
            data['arguments'] = arguments

        if result is not None:
            data['result'] = str(result) if not isinstance(result, (dict, list, str, int, float, bool)) else result

        if error:
            data['error'] = error

        if duration is not None:
            data['duration_ms'] = duration * 1000 if duration < 1000 else duration

        self._add_entry('function_call', data)

    def log_memory_update(self, key: str, value: Any, operation: str = 'set'):
        """Log a memory update"""
        self._add_entry('memory_update', {
            'key': key,
            'value': str(value) if not isinstance(value, (dict, list, str, int, float, bool)) else value,
            'operation': operation
        })

    def log_context_update(self, context_data: Dict[str, Any]):
        """Log a context update"""
        self._add_entry('context_update', context_data)

    def log_intelligence_processing(
        self,
        stage: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None,
        success: bool = True
    ):
        """Log intelligence processing stage (generic)"""
        data = {
            'stage': stage,
            'success': success,
        }

        if input_data:
            data['input'] = input_data

        if output_data:
            data['output'] = output_data

        if duration is not None:
            data['duration_ms'] = duration * 1000 if duration < 1000 else duration

        self._add_entry('intelligence_processing', data)

    def log_intelligence_classification(
        self,
        path_used: str,
        latency_ms: float,
        confidence: float,
        intents: List[Any],
        entities: List[Any],
        reasoning: Optional[str] = None
    ):
        """Log hybrid intelligence classification"""
        self._add_entry('intelligence_classification', {
            'path_used': path_used,
            'latency_ms': latency_ms,
            'confidence': confidence,
            'intents': [str(i) for i in intents],
            'entity_count': len(entities),
            'entities': [str(e) for e in entities],
            'reasoning': reasoning
        })

    def log_confidence_scoring(
        self,
        overall_score: float,
        intent_clarity: float,
        entity_clarity: float,
        ambiguity: float,
        recommendation: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log confidence scoring details"""
        data = {
            'overall_score': overall_score,
            'intent_clarity': intent_clarity,
            'entity_clarity': entity_clarity,
            'ambiguity': ambiguity,
            'recommendation': recommendation
        }
        if details:
            data.update(details)

        self._add_entry('confidence_scoring', data)

    def log_risk_assessment(
        self,
        risk_level: str,
        needs_confirmation: bool,
        reason: str,
        intents: Optional[List[str]] = None
    ):
        """Log risk assessment"""
        self._add_entry('risk_assessment', {
            'risk_level': risk_level,
            'needs_confirmation': needs_confirmation,
            'reason': reason,
            'intents': intents or []
        })

    def log_context_resolution(
        self,
        resolutions: List[Dict[str, Any]]
    ):
        """Log context/reference resolutions"""
        self._add_entry('context_resolution', {
            'resolutions': resolutions,
            'count': len(resolutions)
        })

    def log_error(
        self,
        error: str,
        error_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        traceback: Optional[str] = None
    ):
        """Log an error"""
        data = {
            'error': error,
        }

        if error_type:
            data['error_type'] = error_type

        if context:
            data['context'] = context

        if traceback:
            data['traceback'] = traceback

        self._add_entry('error', data)

    def log_system_event(self, event: str, data: Optional[Dict[str, Any]] = None):
        """Log a system event"""
        self._add_entry('system', {
            'event': event,
            **(data or {})
        })

    def get_log_path(self) -> str:
        """Get the path to the text log file (primary/human-readable)"""
        return str(self.text_file)

    def get_json_log_path(self) -> str:
        """Get the path to the JSON log file"""
        return str(self.json_file)

    def get_entries(self, entry_type: Optional[str] = None) -> List[LogEntry]:
        """Get all entries, optionally filtered by type"""
        if entry_type:
            return [e for e in self.entries if e.type == entry_type]
        return self.entries

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the session"""
        # Count entries by type
        type_counts = {}
        for entry in self.entries:
            type_counts[entry.type] = type_counts.get(entry.type, 0) + 1

        # Count agent calls
        agent_calls = {}
        for entry in self.entries:
            if entry.type == 'agent_call':
                agent_name = entry.data.get('agent_name', 'unknown')
                agent_calls[agent_name] = agent_calls.get(agent_name, 0) + 1

        # Count errors
        error_count = type_counts.get('error', 0)

        return {
            'session_id': self.session_id,
            'duration_seconds': time.time() - self.session_start,
            'total_entries': len(self.entries),
            'entry_types': type_counts,
            'agent_calls': agent_calls,
            'error_count': error_count,
            'text_log': str(self.text_file),
            'json_log': str(self.json_file)
        }

    def close(self):
        """Close the logger and write final summary"""
        self.metadata['ended_at'] = time.time()
        self.metadata['ended_at_iso'] = datetime.now().isoformat()
        self.metadata['duration_seconds'] = time.time() - self.session_start
        self.metadata['summary'] = self.get_summary()
        self._save_json()

        # Write summary to text log
        try:
            with open(self.text_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 100 + "\n")
                f.write("SESSION SUMMARY\n")
                f.write("=" * 100 + "\n")
                summary = self.get_summary()
                f.write(f"Duration: {summary['duration_seconds']:.1f}s\n")
                f.write(f"Total Entries: {summary['total_entries']}\n")
                f.write(f"\nEntry Types:\n")
                for entry_type, count in summary['entry_types'].items():
                    f.write(f"  {entry_type}: {count}\n")
                f.write(f"\nAgent Calls:\n")
                for agent, count in summary['agent_calls'].items():
                    f.write(f"  {agent}: {count}\n")
                f.write(f"\nErrors: {summary['error_count']}\n")
                f.write("=" * 100 + "\n")
        except Exception as e:
            print(f"Warning: Failed to write text log summary: {e}")


# Global session logger instance
_global_logger: Optional[SessionLogger] = None


def get_session_logger() -> Optional[SessionLogger]:
    """Get the global session logger instance"""
    return _global_logger


def set_session_logger(logger: SessionLogger):
    """Set the global session logger instance"""
    global _global_logger
    _global_logger = logger


def init_session_logger(session_id: str, log_dir: str = "logs") -> SessionLogger:
    """Initialize and set the global session logger"""
    logger = SessionLogger(session_id, log_dir)
    set_session_logger(logger)
    return logger
