"""Teacher prompt assembly for step-SDPO trajectories."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeacherPromptInputs:
    initial_prompt_block: str
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
    sections: list[str] = []
    recent = str(recent_raw_block).strip()
    if recent:
        sections.append(
            "Below is a student's attempt in solving the task above, showing only recent few turns with tool response:"
        )
        sections.append(recent)
    compressed = str(compressed_memory_block).strip()
    if compressed:
        sections.append("Earlier attempt summary:")
        sections.append(compressed)
    critical = str(critical_facts_block).strip()
    if critical:
        sections.append("Key facts to keep in mind:")
        sections.append(critical)
    return "\n\n".join(sections).strip()


def build_teacher_prompt(inputs: TeacherPromptInputs) -> str:
    """Compose step-SDPO teacher prompt following v1.6 block ordering."""
    trajectory_block = build_trajectory_block(
        recent_raw_block=inputs.recent_raw_block,
        compressed_memory_block=inputs.compressed_memory_block,
        critical_facts_block=inputs.critical_facts_block,
    )
    sections: list[str] = []
    initial_prompt = str(inputs.initial_prompt_block).strip()
    if initial_prompt:
        sections.append(initial_prompt)
    if trajectory_block:
        sections.append(trajectory_block)
    current_attempt = str(inputs.current_attempt_block).strip()
    if current_attempt:
        sections.append("Below is the current turn in the student's attempt:")
        sections.append(current_attempt)
    feedback_block = str(inputs.feedback_block).strip()
    if feedback_block:
        sections.append("Additional feedback:")
        sections.append(feedback_block)
    output_contract = str(inputs.output_contract_block).strip()
    if output_contract:
        sections.append(output_contract)
    return "\n\n".join(sections).strip()
