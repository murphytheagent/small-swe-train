from __future__ import annotations

from teacher.memory_builder import build_teacher_memory_blocks


def _sample(
    *,
    trajectory_steps,
    trajectory_turn_tool_response_blocks=None,
):
    sample = {"trajectory_steps": trajectory_steps}
    if trajectory_turn_tool_response_blocks is not None:
        sample["trajectory_turn_tool_response_blocks"] = trajectory_turn_tool_response_blocks
    return sample


def test_teacher_memory_builder_enabled_populates_both_memory_blocks() -> None:
    sample = _sample(
        trajectory_steps=[
            {
                "tool": "read",
                "args": {"path": " src/alpha.py "},
                "stdout": "file body",
                "stderr": "",
                "exit_code": 0,
            },
            {
                "tool": "text_search",
                "args": {"query": "alpha"},
                "stdout": "src/beta.py:12:match\n/testbed/raw.txt:2:match",
                "stderr": "",
                "exit_code": 0,
            },
            {
                "tool": "apply_patch",
                "args": {"path": " src/alpha.py ", "patch": " - old\n + new "},
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
            },
        ],
        trajectory_turn_tool_response_blocks=[["r0", "r1"], ["r2"]],
    )

    blocks = build_teacher_memory_blocks(sample, current_turn_index=1)

    assert blocks.compressed_memory_block == (
        "Known student-discovered paths (raw):\n"
        "- src/beta.py\n"
        "- /testbed/raw.txt\n"
        "- src/alpha.py"
    )
    assert blocks.critical_facts_block == (
        "Successful apply_patch calls through current turn:\n\n"
        "[PATCH 1]\n"
        "raw_path: src/alpha.py\n"
        "raw_patch:\n"
        "- old\n"
        " + new"
    )


def test_file_search_rows_are_included_in_known_paths_block() -> None:
    sample = _sample(
        trajectory_steps=[
            {
                "tool": "file_search",
                "args": {"query": "alpha"},
                "stdout": "src/alpha.py\nsrc/alpha_test.py\n",
                "stderr": "",
                "exit_code": 0,
            }
        ]
    )

    blocks = build_teacher_memory_blocks(sample, current_turn_index=0)

    assert blocks.compressed_memory_block == (
        "Known student-discovered paths (raw):\n"
        "- src/alpha.py\n"
        "- src/alpha_test.py"
    )


def test_teacher_memory_builder_disabled_returns_empty_blocks() -> None:
    sample = _sample(
        trajectory_steps=[
            {
                "tool": "read",
                "args": {"path": "src/file.py"},
                "stdout": "body",
                "stderr": "",
                "exit_code": 0,
            }
        ],
        trajectory_turn_tool_response_blocks=[["r0"]],
    )

    blocks = build_teacher_memory_blocks(
        sample,
        current_turn_index=0,
        include_teacher_memory_blocks=False,
    )

    assert blocks.compressed_memory_block == ""
    assert blocks.critical_facts_block == ""


def test_known_paths_preserve_raw_strings_except_outer_trim() -> None:
    sample = _sample(
        trajectory_steps=[
            {
                "tool": "read",
                "args": {"path": "  /testbed/src/odd  name.py  "},
                "stdout": "body",
                "stderr": "",
                "exit_code": 0,
            },
            {
                "tool": "text_search",
                "args": {"query": "odd"},
                "stdout": "../relative/raw path.py:8:hit",
                "stderr": "",
                "exit_code": 0,
            },
        ]
    )

    blocks = build_teacher_memory_blocks(sample, current_turn_index=0)

    assert blocks.compressed_memory_block == (
        "Known student-discovered paths (raw):\n"
        "- ../relative/raw path.py\n"
        "- /testbed/src/odd  name.py"
    )


def test_known_paths_exact_string_dedupe_keeps_differently_formatted_paths_distinct() -> None:
    sample = _sample(
        trajectory_steps=[
            {
                "tool": "text_search",
                "args": {"query": "path"},
                "stdout": "src/main.py:3:hit\n./src/main.py:4:hit\nsrc/main.py:5:dup",
                "stderr": "",
                "exit_code": 0,
            },
            {
                "tool": "read",
                "args": {"path": " src/main.py "},
                "stdout": "body",
                "stderr": "",
                "exit_code": 0,
            },
        ]
    )

    blocks = build_teacher_memory_blocks(sample, current_turn_index=0)

    assert blocks.compressed_memory_block == (
        "Known student-discovered paths (raw):\n"
        "- src/main.py\n"
        "- ./src/main.py"
    )


def test_search_path_parsing_extracts_only_valid_path_line_content_rows() -> None:
    sample = _sample(
        trajectory_steps=[
            {
                "tool": "text_search",
                "args": {"query": "needle"},
                "stdout": (
                    "src/ok.py:12:match\n"
                    "src/missing_line:match\n"
                    "just text\n"
                    "  src/leading_space.py:4:skip\n"
                    "/abs/ok.py:7:match"
                ),
                "stderr": "",
                "exit_code": 0,
            }
        ]
    )

    blocks = build_teacher_memory_blocks(sample, current_turn_index=0)

    assert blocks.compressed_memory_block == (
        "Known student-discovered paths (raw):\n"
        "- src/ok.py\n"
        "- src/leading_space.py\n"
        "- /abs/ok.py"
    )


def test_known_paths_include_future_turn_discoveries_by_design() -> None:
    sample = _sample(
        trajectory_steps=[
            {
                "tool": "read",
                "args": {"path": "src/current.py"},
                "stdout": "body",
                "stderr": "",
                "exit_code": 0,
            },
            {
                "tool": "text_search",
                "args": {"query": "future"},
                "stdout": "src/future.py:9:match",
                "stderr": "",
                "exit_code": 0,
            },
        ],
        trajectory_turn_tool_response_blocks=[["r0"], ["r1"]],
    )

    blocks = build_teacher_memory_blocks(sample, current_turn_index=0)

    assert "src/future.py" in blocks.compressed_memory_block
    assert "src/current.py" in blocks.compressed_memory_block


def test_successful_prefix_patches_include_current_turn_and_exclude_later_turns() -> None:
    sample = _sample(
        trajectory_steps=[
            {
                "tool": "apply_patch",
                "args": {"path": "src/turn0.py", "patch": "- zero\n+ zero-fixed"},
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
            },
            {
                "tool": "read",
                "args": {"path": "src/context.py"},
                "stdout": "body",
                "stderr": "",
                "exit_code": 0,
            },
            {
                "tool": "apply_patch",
                "args": {"path": "src/turn1.py", "patch": "- one\n+ one-fixed"},
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
            },
            {
                "tool": "apply_patch",
                "args": {"path": "src/future.py", "patch": "- future\n+ future-fixed"},
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
            },
        ],
        trajectory_turn_tool_response_blocks=[["r0"], ["r1", "r2"], ["r3"]],
    )

    blocks = build_teacher_memory_blocks(sample, current_turn_index=1)

    assert blocks.critical_facts_block == (
        "Successful apply_patch calls through current turn:\n\n"
        "[PATCH 1]\n"
        "raw_path: src/turn0.py\n"
        "raw_patch:\n"
        "- zero\n"
        "+ zero-fixed\n\n"
        "[PATCH 2]\n"
        "raw_path: src/turn1.py\n"
        "raw_patch:\n"
        "- one\n"
        "+ one-fixed"
    )


def test_current_turn_patch_memory_is_omitted_when_student_attempt_context_is_disabled() -> None:
    sample = _sample(
        trajectory_steps=[
            {
                "tool": "apply_patch",
                "args": {"path": "src/turn0.py", "patch": "- zero\n+ zero-fixed"},
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
            },
            {
                "tool": "apply_patch",
                "args": {"path": "src/turn1.py", "patch": "- one\n+ one-fixed"},
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
            },
        ],
        trajectory_turn_tool_response_blocks=[["r0"], ["r1"]],
    )

    blocks = build_teacher_memory_blocks(
        sample,
        current_turn_index=1,
        include_student_attempt_for_teacher=False,
    )

    assert blocks.critical_facts_block == (
        "Successful apply_patch calls through current turn:\n\n"
        "[PATCH 1]\n"
        "raw_path: src/turn0.py\n"
        "raw_patch:\n"
        "- zero\n"
        "+ zero-fixed"
    )


def test_successful_pathless_codex_style_patch_uses_missing_path_placeholder() -> None:
    raw_patch = (
        "*** Begin Patch\n"
        "*** Update File: src/file.py\n"
        "@@\n"
        "-old\n"
        "+new\n"
        "*** End Patch\n"
    )
    sample = _sample(
        trajectory_steps=[
            {
                "tool": "apply_patch",
                "args": {"patch": raw_patch},
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
            }
        ],
        trajectory_turn_tool_response_blocks=[["r0"]],
    )

    blocks = build_teacher_memory_blocks(sample, current_turn_index=0)

    assert blocks.critical_facts_block == (
        "Successful apply_patch calls through current turn:\n\n"
        "[PATCH 1]\n"
        "raw_path: <missing>\n"
        "raw_patch:\n"
        "*** Begin Patch\n"
        "*** Update File: src/file.py\n"
        "@@\n"
        "-old\n"
        "+new\n"
        "*** End Patch"
    )


def test_missing_per_turn_grouping_yields_empty_patch_memory_block() -> None:
    sample = _sample(
        trajectory_steps=[
            {
                "tool": "apply_patch",
                "args": {"path": "src/file.py", "patch": "-old\n+new"},
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
            }
        ]
    )

    blocks = build_teacher_memory_blocks(sample, current_turn_index=0)

    assert blocks.critical_facts_block == ""
