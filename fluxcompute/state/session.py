"""
Local session manager for multi-turn conversation tracking.

Stores conversation history, routing decisions, and tool outputs
in memory. Enables the classifier to see full context across turns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Session:
    """In-memory session state for a single conversation."""

    id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    routing_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_outputs: List[Any] = field(default_factory=list)
    total_queries: int = 0
    total_cost_usd: float = 0.0
    total_savings_usd: float = 0.0


class SessionManager:
    """
    Manages in-memory sessions for multi-turn state preservation.

    Sessions are keyed by session_id. Each session stores the full
    conversation history so the classifier can see context across turns.
    """

    def __init__(self, max_sessions: int = 10_000):
        self._sessions: Dict[str, Session] = {}
        self._max_sessions = max_sessions

    def get_or_create(self, session_id: str) -> Session:
        """Get existing session or create a new one."""
        if session_id not in self._sessions:
            # Evict oldest session if at capacity
            if len(self._sessions) >= self._max_sessions:
                self._evict_oldest()
            self._sessions[session_id] = Session(id=session_id)
        session = self._sessions[session_id]
        session.last_active = time.time()
        return session

    def update(
        self,
        session_id: str,
        user_message: Dict[str, str],
        assistant_message: Dict[str, str],
        model_used: str,
        cost_usd: float,
        savings_usd: float,
    ) -> None:
        """Update session after a completed query."""
        session = self.get_or_create(session_id)
        session.conversation_history.append(user_message)
        session.conversation_history.append(assistant_message)
        session.routing_history.append({
            "timestamp": time.time(),
            "model": model_used,
            "query_preview": user_message.get("content", "")[:100],
        })
        session.total_queries += 1
        session.total_cost_usd += cost_usd
        session.total_savings_usd += savings_usd

    def get_context(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.
        Used to build full context for the classifier.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return []
        return session.conversation_history.copy()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID, or None if it doesn't exist."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        """Delete a session."""
        self._sessions.pop(session_id, None)

    def _evict_oldest(self) -> None:
        """Remove the least recently active session."""
        if not self._sessions:
            return
        oldest_id = min(self._sessions, key=lambda k: self._sessions[k].last_active)
        del self._sessions[oldest_id]

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)
