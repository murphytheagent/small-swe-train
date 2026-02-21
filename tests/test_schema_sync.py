from __future__ import annotations

import json
from pathlib import Path


def test_runtime_schemas_match_frozen_design_schemas() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    frozen_dir = (
        repo_root
        / "outputs"
        / "1771579678.414229"
        / "contracts_metrics_scaffold"
        / "schemas"
    )
    runtime_dir = repo_root / "src" / "schemas"

    for name in [
        "action_envelope.schema.json",
        "tool_args.schema.json",
        "feedback_packet.schema.json",
    ]:
        frozen = json.loads((frozen_dir / name).read_text())
        runtime = json.loads((runtime_dir / name).read_text())
        assert runtime == frozen
