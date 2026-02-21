"""Teacher prompt assembly for step-SDPO trajectories."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeacherPromptInputs:
    system_block: str
    task_block: str
    recent_raw_block: str
    compressed_memory_block: str
    critical_facts_block: str
    current_attempt_block: str
    feedback_block: str
    output_contract_block: str


def build_trajectory_block(
    *,
    recent_raw_block: str,
    compressed_memory_block: str,
    critical_facts_block: str,
) -> str:
    """Build trajectory block from raw+compressed+critical context slices."""
    return (
        "[TRAJECTORY_BLOCK]\n"
        f"[RECENT_RAW_BLOCK]\n{recent_raw_block}\n"
        f"[COMPRESSED_MEMORY_BLOCK]\n{compressed_memory_block}\n"
        f"[CRITICAL_FACTS_BLOCK]\n{critical_facts_block}\n"
    )


def build_teacher_prompt(inputs: TeacherPromptInputs) -> str:
    """Compose step-SDPO teacher prompt following v1.6 block ordering."""
    trajectory_block = build_trajectory_block(
        recent_raw_block=inputs.recent_raw_block,
        compressed_memory_block=inputs.compressed_memory_block,
        critical_facts_block=inputs.critical_facts_block,
    )

    return (
        "[SYSTEM_BLOCK]\n"
        f"{inputs.system_block}\n\n"
        "[TASK_BLOCK]\n"
        f"{inputs.task_block}\n\n"
        f"{trajectory_block}\n"
        "[CURRENT_ATTEMPT_BLOCK]\n"
        f"{inputs.current_attempt_block}\n\n"
        "[FEEDBACK_BLOCK]\n"
        f"{inputs.feedback_block}\n\n"
        "[OUTPUT_CONTRACT_BLOCK]\n"
        f"{inputs.output_contract_block}"
    )
