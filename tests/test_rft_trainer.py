from __future__ import annotations

import json
from pathlib import Path

from config import DEFAULT_TRAINING_MODEL_NAME
from trainer.common import SDPOTrainerConfig
from trainer.rft_trainer import RFTTrainerScaffold


class _CharTokenizer:
    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        del add_special_tokens
        if isinstance(text, list):
            input_ids_batch = []
            offsets_batch = []
            for item in text:
                encoded = self(item, return_offsets_mapping=return_offsets_mapping)
                input_ids_batch.append(encoded["input_ids"])
                offsets_batch.append(encoded["offset_mapping"])
            return {"input_ids": input_ids_batch, "offset_mapping": offsets_batch}

        input_ids = [index + 1 for index, _char in enumerate(text)]
        offsets = [(index, index + 1) for index, _char in enumerate(text)]
        return {"input_ids": input_ids, "offset_mapping": offsets}


class _FakePool:
    def acquire(self, tasks):
        from env.container_pool import ContainerHandle

        return (
            ContainerHandle(
                task_id=tasks[0].task_id,
                image_name=tasks[0].image_name,
                container_id="cid-1",
                container_name="cname-1",
            ),
        )

    def release_all(self) -> None:
        return None


class _FakeCollectorExecutor:
    def run(self, request):
        from env.runtime_protocol import ToolResponse

        return ToolResponse(stdout=f"ran:{request.tool}", stderr="", exit_code=0)


def test_run_onpolicy_rft_step_from_config_uses_real_data_config_and_collector(tmp_path: Path) -> None:
    trainer = RFTTrainerScaffold(SDPOTrainerConfig(model_name=DEFAULT_TRAINING_MODEL_NAME))

    def turn_generator(**kwargs: object) -> str:
        attempt_index = int(kwargs["attempt_index"])
        turn_index = int(kwargs["turn_index"])
        if attempt_index == 0:
            if turn_index == 0:
                return '<tool_call>{"tool":"search","args":{"query":"foo"}}</tool_call>'
            return '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
        return '<tool_call>{"tool":"submit","args":{}}</tool_call>'

    artifacts = trainer.run_onpolicy_rft_step_from_config(
        total_steps=1,
        tokenizer=_CharTokenizer(),
        turn_generator=turn_generator,
        data_config_name="on_policy_swe_smith",
        runtime_overrides={
            "enabled": True,
            "rollout_only": True,
            "task_batch_size": 1,
            "attempts_per_task": 2,
            "max_turns_per_attempt": 2,
            "env_pool_size": 1,
            "tool_timeout_sec": 1,
            "container_start_timeout_sec": 1,
            "attempt_timeout_sec": 10,
            "max_tool_calls_per_turn": 3,
        },
        dataset_loader=lambda _dataset_id, _split: [
            {
                "task_id": "task-1",
                "image_name": "img:1",
                "problem_statement": "Fix bug",
                "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
                "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
            }
        ],
        pool_factory=lambda _runtime: _FakePool(),
        executor_factory=lambda _handle, _runtime: _FakeCollectorExecutor(),
        output_dir=str(tmp_path),
        checkpoint_dir=tmp_path / "checkpoints",
        global_step=10,
    )

    assert artifacts.selected_count == 1
    assert artifacts.rejected_count == 1
    assert artifacts.checkpoint_exists is True
    assert artifacts.dataproto_payload["non_tensors"]["task_id"] == ["task-1"]
    summary_payload = json.loads((tmp_path / "rollout_artifact_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["unique_task_ids"] == ["task-1"]
    assert summary_payload["unique_image_names"] == ["img:1"]
