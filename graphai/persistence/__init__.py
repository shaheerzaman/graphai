"""Persistence utilities for GraphAI.

This package provides in-memory state persistence implementations that can be
used to record and replay graph executions, inspired by pydantic-ai's
``SimpleStatePersistence`` and ``FullStatePersistence``.

These classes are framework-agnostic and do not modify ``Graph`` itself. They
can be wired into your execution flow (e.g., just before invoking a node and
around its execution) to capture snapshots, timings and outcomes.
"""

from .in_memory import (
    BaseStatePersistence,
    NodeStatus,
    Snapshot,
    NodeSnapshot,
    EndSnapshot,
    SimpleStatePersistence,
    FullStatePersistence,
)

__all__ = [
    "BaseStatePersistence",
    "NodeStatus",
    "Snapshot",
    "NodeSnapshot",
    "EndSnapshot",
    "SimpleStatePersistence",
    "FullStatePersistence",
]
