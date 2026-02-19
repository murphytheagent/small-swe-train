from dataclasses import dataclass, field
from typing import Literal

ActionType = Literal["bash", "search", "edit", "submit"]


@dataclass
class ToolAction:
    """One model-produced action token span."""

    action_type: ActionType
    content: str
    token_start: int = 0
    token_end: int = 0


@dataclass
class Observation:
    """Environment output after running an action."""

    text: str
    reward: float
    is_terminal: bool = False


@dataclass
class TrajectoryStep:
    """One transition in a multi-turn SWE episode."""

    history: str
    action: ToolAction
    observation: Observation


@dataclass
class Trajectory:
    """An episode trajectory with final success signal."""

    task_id: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    success: bool = False


@dataclass
class DistillationTargets:
    """Per-step distillation metadata used by SDPO/SDFT."""

    student_context: str
    teacher_context: str
    action_token_mask: list[int]
