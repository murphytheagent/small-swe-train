from dataclasses import dataclass, field


@dataclass
class FormatBootstrapConfig:
    epochs: int = 1
    max_examples: int = 10000
    target_valid_action_rate: float = 0.98


@dataclass
class SdftConfig:
    epochs: int = 1
    max_examples: int = 10000
    reverse_kl_coef: float = 1.0
    demo_uplift_eval_every: int = 500


@dataclass
class StepSdpoConfig:
    epochs: int = 1
    max_episodes: int = 2000
    topk_distill: int = 64
    feedback_truncate_tokens: int = 4096


@dataclass
class TerminalHindsightConfig:
    enabled: bool = True
    last_k_steps: int = 4
    sample_rate: float = 0.5


@dataclass
class TrainConfig:
    format_bootstrap: FormatBootstrapConfig = field(default_factory=FormatBootstrapConfig)
    sdft: SdftConfig = field(default_factory=SdftConfig)
    step_sdpo: StepSdpoConfig = field(default_factory=StepSdpoConfig)
    terminal_hindsight: TerminalHindsightConfig = field(default_factory=TerminalHindsightConfig)
