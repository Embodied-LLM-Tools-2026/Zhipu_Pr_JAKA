"""
Action and primitive registry to support Planner↔Engineer workflows.

This module defines the ticket schema (planner requests), registry entries
and storage interface for high-level skills and low-level primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import time
import uuid


@dataclass
class ActionTicket:
    """Planner发出的需求单."""

    ticket_id: str
    goal: str
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class ActionEntry:
    """已注册的高层action."""

    name: str
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    code_path: str
    version: str = "0.1"
    tests: Optional[str] = None
    author: Optional[str] = None
    last_updated: float = field(default_factory=time.time)


@dataclass
class PrimitiveEntry:
    """底层原语描述."""

    name: str
    description: str
    api_signature: str
    module: str
    safety_notes: Optional[str] = None
    last_updated: float = field(default_factory=time.time)


class ActionRegistry:
    """In-memory registry (可替换为DB/文件)."""

    def __init__(self) -> None:
        self.actions: Dict[str, ActionEntry] = {}
        self.primitives: Dict[str, PrimitiveEntry] = {}
        self.tickets: Dict[str, ActionTicket] = {}

    # Ticket workflow
    def create_ticket(
        self,
        goal: str,
        description: str,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
    ) -> ActionTicket:
        ticket = ActionTicket(
            ticket_id=uuid.uuid4().hex[:12],
            goal=goal,
            description=description,
            inputs=inputs or {},
            outputs=outputs or {},
            constraints=constraints or {},
            examples=examples or [],
        )
        self.tickets[ticket.ticket_id] = ticket
        return ticket

    def list_tickets(self) -> List[Dict[str, Any]]:
        return [asdict(ticket) for ticket in self.tickets.values()]

    def get_ticket(self, ticket_id: str) -> Optional[ActionTicket]:
        return self.tickets.get(ticket_id)

    def close_ticket(self, ticket_id: str) -> None:
        self.tickets.pop(ticket_id, None)

    # Action registry
    def register_action(self, entry: ActionEntry) -> None:
        entry.last_updated = time.time()
        self.actions[entry.name] = entry

    def unregister_action(self, name: str) -> None:
        self.actions.pop(name, None)

    def list_actions(self) -> List[Dict[str, Any]]:
        return [asdict(action) for action in self.actions.values()]

    def get_action(self, name: str) -> Optional[ActionEntry]:
        return self.actions.get(name)

    # Primitive registry
    def register_primitive(self, entry: PrimitiveEntry) -> None:
        entry.last_updated = time.time()
        self.primitives[entry.name] = entry

    def list_primitives(self) -> List[Dict[str, Any]]:
        return [asdict(primitive) for primitive in self.primitives.values()]

    def get_primitive(self, name: str) -> Optional[PrimitiveEntry]:
        return self.primitives.get(name)
