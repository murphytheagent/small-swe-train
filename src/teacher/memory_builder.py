"""Scaffolding helpers for trajectory memory blocks in teacher reprompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TeacherMemoryBlocks:
    compressed_memory_block: str
    critical_facts_block: str


def build_teacher_memory_blocks(
    sample: Mapping[str, Any],
    *,
    current_turn_index: int,
) -> TeacherMemoryBlocks:
    """Return placeholder memory blocks.

    Real compression/fact extraction is intentionally deferred. Keep empty
    defaults so downstream prompt assembly is stable.
    """
    _ = sample, current_turn_index
    return TeacherMemoryBlocks(
        compressed_memory_block="",
        critical_facts_block="",
    )
