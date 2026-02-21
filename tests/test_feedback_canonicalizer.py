from __future__ import annotations

from data.feedback_canonicalizer import build_feedback_packet, canonicalize_tool_feedback


def test_build_feedback_packet_derives_self_containment_checks() -> None:
    tool_output = {
        "stdout": "FAILED tests/test_math.py::test_add - AssertionError: expected 2\nsrc/math_utils.py:12: ValueError: wrong output",
        "stderr": "\x1b[31mTraceback (most recent call last):\x1b[0m",
        "exit_code": 1,
    }

    packet = build_feedback_packet(
        step_index=2,
        tool="bash",
        tool_input={"command": "pytest tests/test_math.py::test_add"},
        tool_output=tool_output,
    )

    assert packet.include_student_attempt_for_teacher is True
    assert packet.canonical_feedback.actionable_error_text is not None
    assert packet.self_containment_checks.has_failing_artifact_identity is True
    assert packet.self_containment_checks.has_actionable_error_text is True
    assert packet.self_containment_checks.has_localization_hint is True
    assert packet.is_self_contained is True


def test_head_tail_truncation_is_deterministic() -> None:
    long_stdout = " ".join(f"tok{i}" for i in range(20))

    canonical = canonicalize_tool_feedback(
        {"stdout": long_stdout, "stderr": "", "exit_code": 1},
        head_tokens=3,
        tail_tokens=2,
    )

    assert canonical.truncated is True
    assert "<...truncated...>" in canonical.normalized_text
    assert canonical.normalized_text.startswith("STDOUT: tok0 tok1")
    assert canonical.normalized_text.endswith('{"exit_code": 1}')
