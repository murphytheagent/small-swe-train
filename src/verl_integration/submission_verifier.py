"""Run SWE FAIL_TO_PASS / PASS_TO_PASS verification inside task containers."""

from __future__ import annotations

import json
import shlex
from typing import Any, Mapping, Sequence

from env.runtime_protocol import ToolRequest

_MIN_PER_TEST_TIMEOUT_SEC = 180
_MAX_PER_TEST_TIMEOUT_SEC = 1800
_GROUP_TIMEOUT_BUFFER_SEC = 120
_MAX_GROUP_TIMEOUT_SEC = 7200

_VERIFY_SCRIPT = """import json
import os
import subprocess

tests = json.loads(os.environ.get("SMALL_SWE_TESTS_JSON", "[]"))
repo_root = os.environ.get("SMALL_SWE_REPO_ROOT", ".")
pybin = os.environ.get("SMALL_SWE_PYBIN", "python3")
per_test_timeout = int(os.environ.get("SMALL_SWE_PER_TEST_TIMEOUT_SEC", "120"))

results: dict[str, bool] = {}
failures: dict[str, dict[str, object]] = {}
executed = 0

for index, raw_name in enumerate(tests):
    test_name = str(raw_name).strip()
    if not test_name:
        continue
    executed += 1
    command = [pybin, "-m", "pytest", "-q", test_name]
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=per_test_timeout,
        )
        passed = completed.returncode == 0
        results[test_name] = passed
        if not passed:
            failures[test_name] = {
                "returncode": int(completed.returncode),
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
            for remainder in tests[index + 1 :]:
                remainder_name = str(remainder).strip()
                if remainder_name and remainder_name not in results:
                    results[remainder_name] = False
            break
    except subprocess.TimeoutExpired as exc:
        failures[test_name] = {
            "returncode": 124,
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
            "timed_out": True,
        }
        results[test_name] = False
        for remainder in tests[index + 1 :]:
            remainder_name = str(remainder).strip()
            if remainder_name and remainder_name not in results:
                results[remainder_name] = False
        break

all_passed = all(bool(value) for value in results.values()) if results else True
print(
    json.dumps(
        {
            "results": results,
            "executed": int(executed),
            "all_passed": bool(all_passed),
            "failures": failures,
        },
        ensure_ascii=True,
    )
)
"""


def run_submission_verifier(
    *,
    executor: Any,
    fail_to_pass: Any,
    pass_to_pass: Any,
    verifier_timeout_sec: int,
    final_response: str,
) -> dict[str, Any]:
    """Verify terminal submission state inside the current task container."""
    fail_targets = _normalize_test_targets(fail_to_pass)
    pass_targets = _normalize_test_targets(pass_to_pass)

    requested_timeout_sec = max(int(verifier_timeout_sec), 1)
    per_test_timeout_sec = max(
        _MIN_PER_TEST_TIMEOUT_SEC,
        min(requested_timeout_sec, _MAX_PER_TEST_TIMEOUT_SEC),
    )
    fail_group = _verify_test_group(
        executor=executor,
        tests=fail_targets,
        verifier_timeout_sec=requested_timeout_sec,
        per_test_timeout_sec=per_test_timeout_sec,
    )
    pass_group = _verify_test_group(
        executor=executor,
        tests=pass_targets,
        verifier_timeout_sec=requested_timeout_sec,
        per_test_timeout_sec=per_test_timeout_sec,
    )

    fail_passed = bool(fail_group["all_passed"]) if fail_targets else True
    pass_passed = bool(pass_group["all_passed"]) if pass_targets else True
    resolved = bool(fail_passed and pass_passed)
    verification_error = "; ".join(
        error
        for error in (
            str(fail_group.get("error", "")).strip(),
            str(pass_group.get("error", "")).strip(),
        )
        if error
    )
    return {
        "submission_final_response": str(final_response),
        "fail_to_pass": list(fail_targets),
        "pass_to_pass": list(pass_targets),
        "fail_to_pass_results": dict(fail_group["results"]),
        "pass_to_pass_results": dict(pass_group["results"]),
        "fail_to_pass_all_passed": bool(fail_passed),
        "pass_to_pass_all_passed": bool(pass_passed),
        "fail_to_pass_verified": bool(fail_passed),
        "pass_to_pass_verified": bool(pass_passed),
        "verification_missing": not bool(fail_targets or pass_targets),
        "verification_error": verification_error,
        "resolved": resolved,
        "verification_feedback": _build_feedback(
            fail_group=fail_group,
            pass_group=pass_group,
            fail_to_pass=fail_targets,
            pass_to_pass=pass_targets,
            resolved=resolved,
        ),
    }


def _verify_test_group(
    *,
    executor: Any,
    tests: Sequence[str],
    verifier_timeout_sec: int,
    per_test_timeout_sec: int,
) -> dict[str, Any]:
    if not tests:
        return {
            "results": {},
            "all_passed": True,
            "executed": 0,
            "error": "",
            "stdout_tail": "",
            "stderr_tail": "",
        }

    group_timeout_sec = _resolve_group_timeout_sec(
        verifier_timeout_sec=verifier_timeout_sec,
        per_test_timeout_sec=per_test_timeout_sec,
        test_count=len(tests),
    )

    tests_json = json.dumps(list(tests), ensure_ascii=True)
    request = ToolRequest(
        tool="bash",
        args={
            "timeout_sec": int(group_timeout_sec),
            "stdin": _VERIFY_SCRIPT,
            "command": _build_verifier_shell_command(
                tests_json=tests_json,
                per_test_timeout_sec=per_test_timeout_sec,
            ),
        },
    )
    response = executor.run(request)

    stdout_text = str(response.stdout or "")
    stdout_tail = str(response.stdout or "")[-4000:]
    stderr_tail = str(response.stderr or "")[-4000:]
    if int(response.exit_code) != 0:
        return {
            "results": {name: False for name in tests},
            "all_passed": False,
            "executed": 0,
            "error": f"verifier command failed with exit code {response.exit_code}",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }

    payload = _decode_verifier_payload(stdout_text)
    if payload is None:
        return {
            "results": {name: False for name in tests},
            "all_passed": False,
            "executed": 0,
            "error": "verifier command returned invalid JSON payload",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }

    raw_results = payload.get("results")
    result_map: dict[str, bool] = {}
    if isinstance(raw_results, Mapping):
        for name in tests:
            result_map[name] = bool(raw_results.get(name, False))
    else:
        result_map = {name: False for name in tests}

    all_passed = bool(payload.get("all_passed", False))
    if result_map:
        all_passed = all(result_map.values())

    return {
        "results": result_map,
        "all_passed": bool(all_passed),
        "executed": int(payload.get("executed", 0) or 0),
        "error": "",
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "failures": payload.get("failures", {}),
    }


def _build_verifier_shell_command(*, tests_json: str, per_test_timeout_sec: int) -> str:
    return (
        "set -eu; "
        'repo_root=""; '
        'for candidate in /testbed /workspace /repo /app; do '
        'if [ -d "${candidate}/.git" ]; then repo_root="${candidate}"; break; fi; '
        "done; "
        'if [ -z "${repo_root}" ]; then '
        'for candidate in /testbed /workspace /repo /app; do '
        'if [ -d "${candidate}" ]; then repo_root="${candidate}"; break; fi; '
        "done; "
        "fi; "
        'if [ -z "${repo_root}" ]; then '
        'echo "Unable to locate task repository root." >&2; '
        "exit 2; "
        "fi; "
        'pybin=""; '
        'for candidate in python3 python; do '
        'if command -v "${candidate}" >/dev/null 2>&1; then pybin="${candidate}"; break; fi; '
        "done; "
        'if [ -z "${pybin}" ]; then '
        'echo "Python interpreter missing in task container." >&2; '
        "exit 127; "
        "fi; "
        f"SMALL_SWE_TESTS_JSON={shlex.quote(tests_json)} "
        f"SMALL_SWE_PER_TEST_TIMEOUT_SEC={int(max(per_test_timeout_sec, 1))} "
        'SMALL_SWE_REPO_ROOT="${repo_root}" '
        'SMALL_SWE_PYBIN="${pybin}" '
        '"${pybin}" -'
    )


def _resolve_group_timeout_sec(
    *,
    verifier_timeout_sec: int,
    per_test_timeout_sec: int,
    test_count: int,
) -> int:
    base_timeout = max(int(verifier_timeout_sec), 1)
    safe_test_count = max(int(test_count), 1)
    min_group_timeout = per_test_timeout_sec * safe_test_count + _GROUP_TIMEOUT_BUFFER_SEC
    return min(max(base_timeout, min_group_timeout), _MAX_GROUP_TIMEOUT_SEC)


def _decode_verifier_payload(stdout_text: str) -> dict[str, Any] | None:
    stripped = str(stdout_text or "").strip()
    if not stripped:
        return None
    for line in reversed(stripped.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return None


def _normalize_test_targets(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [name for name in (str(key).strip() for key in value.keys()) if name]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        targets: list[str] = []
        for item in value:
            name = str(item).strip()
            if name:
                targets.append(name)
        return targets
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                return _normalize_test_targets(parsed)
        if "," in stripped:
            return [chunk for chunk in (part.strip() for part in stripped.split(",")) if chunk]
        if "\n" in stripped:
            return [chunk for chunk in (part.strip() for part in stripped.splitlines()) if chunk]
        return [stripped]
    return []


def _build_feedback(
    *,
    fail_group: Mapping[str, Any],
    pass_group: Mapping[str, Any],
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    resolved: bool,
) -> str:
    if resolved:
        return "Verifier: all FAIL_TO_PASS and PASS_TO_PASS tests passed."

    lines: list[str] = []
    if fail_to_pass:
        lines.append(
            "FAIL_TO_PASS passed="
            + str(bool(fail_group.get("all_passed", False)))
            + f" tests={len(fail_to_pass)}"
        )
    if pass_to_pass:
        lines.append(
            "PASS_TO_PASS passed="
            + str(bool(pass_group.get("all_passed", False)))
            + f" tests={len(pass_to_pass)}"
        )

    fail_error = str(fail_group.get("error", "")).strip()
    pass_error = str(pass_group.get("error", "")).strip()
    if fail_error:
        lines.append(f"FAIL_TO_PASS verifier error: {fail_error}")
    if pass_error:
        lines.append(f"PASS_TO_PASS verifier error: {pass_error}")

    fail_tail = str(fail_group.get("stderr_tail", "")).strip()
    pass_tail = str(pass_group.get("stderr_tail", "")).strip()
    if fail_tail:
        lines.append("FAIL_TO_PASS stderr tail:\n" + fail_tail[-2000:])
    if pass_tail:
        lines.append("PASS_TO_PASS stderr tail:\n" + pass_tail[-2000:])
    return "\n".join(lines).strip()
