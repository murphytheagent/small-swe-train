from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from small_swe_train.config import (
    FormatBootstrapConfig,
    SdftConfig,
    StepSdpoConfig,
    TerminalHindsightConfig,
)
from small_swe_train.training.metrics import MetricSink
from small_swe_train.training.stages import (
    log_stage_config,
    run_format_bootstrap,
    run_sdft,
    run_step_sdpo,
    run_terminal_hindsight,
)


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mock_format_examples():
    return []


def _mock_distill_targets():
    return []


def _mock_trajectories():
    return []


class PlaceholderModel:
    """Temporary model stub so stage wiring can run end-to-end."""

    def sample_action(self, context):
        _ = context
        raise NotImplementedError("Bind to your model runtime.")

    def train_format_step(self, context, target_action):
        _ = (context, target_action)
        return 0.0

    def train_reverse_kl_step(self, targets):
        _ = targets
        return 0.0

    def train_sdpo_step(self, targets):
        _ = targets
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    args = parser.parse_args()

    cfg_data = _load_config(args.config)
    format_cfg = FormatBootstrapConfig(**cfg_data.get("format_bootstrap", {}))
    sdft_cfg = SdftConfig(**cfg_data.get("sdft", {}))
    sdpo_cfg = StepSdpoConfig(**cfg_data.get("step_sdpo", {}))
    hindsight_cfg = TerminalHindsightConfig(**cfg_data.get("terminal_hindsight", {}))

    model = PlaceholderModel()
    metrics = MetricSink()

    print(json.dumps(log_stage_config("format_bootstrap", format_cfg)))
    run_format_bootstrap(model, _mock_format_examples(), format_cfg, metrics)

    print(json.dumps(log_stage_config("sdft", sdft_cfg)))
    run_sdft(model, _mock_distill_targets(), sdft_cfg, metrics)

    print(json.dumps(log_stage_config("step_sdpo", sdpo_cfg)))
    run_step_sdpo(model, _mock_distill_targets(), sdpo_cfg, metrics)

    print(json.dumps(log_stage_config("terminal_hindsight", hindsight_cfg)))
    run_terminal_hindsight(model, _mock_trajectories(), hindsight_cfg, metrics)

    print(json.dumps({"metrics": metrics.summary()}))
    print(json.dumps({"config": {
        "format_bootstrap": asdict(format_cfg),
        "sdft": asdict(sdft_cfg),
        "step_sdpo": asdict(sdpo_cfg),
        "terminal_hindsight": asdict(hindsight_cfg),
    }}))


if __name__ == "__main__":
    main()
