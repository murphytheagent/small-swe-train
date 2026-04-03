"""Run verifier-backed FAIL_TO_PASS / PASS_TO_PASS checks inside task containers."""

from __future__ import annotations

import json
import shlex
from typing import Any, Mapping, Sequence

from env.runtime_protocol import ToolRequest
from env.shell_helpers import (
    build_executable_resolver_shell,
    build_python_interpreter_resolver_shell,
)
from verifier_utils import normalize_verifier_kind, normalize_verifier_targets

_MIN_PER_TEST_TIMEOUT_SEC = 180
_MAX_PER_TEST_TIMEOUT_SEC = 1800
_GROUP_TIMEOUT_BUFFER_SEC = 120
_MAX_GROUP_TIMEOUT_SEC = 7200
_SUPPORTED_RUNTIME_VERIFIER_KINDS = {"pytest", "go_test"}

_VERIFY_SCRIPT = """import json
import os
import re
import subprocess
import shutil

targets = json.loads(os.environ.get("SMALL_SWE_TESTS_JSON", "[]"))
repo_root = os.environ.get("TASK_REPO_ROOT") or os.environ.get("SMALL_SWE_REPO_ROOT", ".")
verifier_kind = os.environ.get("SMALL_SWE_VERIFIER_KIND", "pytest").strip().lower()
pybin = os.environ.get("SMALL_SWE_PYBIN", "python3")
gobin = os.environ.get("SMALL_SWE_GO_BIN", "").strip()
per_test_timeout = int(os.environ.get("SMALL_SWE_PER_TEST_TIMEOUT_SEC", "120"))


def _resolve_executable(env_value: str, command_names: tuple[str, ...], fallback_paths: tuple[str, ...]) -> str:
    if env_value:
        return env_value
    for command_name in command_names:
        resolved = shutil.which(command_name)
        if resolved:
            return resolved
    for candidate in fallback_paths:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return ""

results: dict[str, bool] = {}
failures: dict[str, dict[str, object]] = {}
executed = 0
group_error = ""

for raw_name in targets:
    test_name = str(raw_name).strip()
    if not test_name:
        continue
    executed += 1
    if verifier_kind == "pytest":
        command = [pybin, "-m", "pytest", "-q", test_name]
    elif verifier_kind == "go_test":
        resolved_go = _resolve_executable(
            gobin,
            ("go",),
            ("/usr/local/go/bin/go", "/usr/lib/go/bin/go", "/opt/go/bin/go"),
        )
        if not resolved_go:
            raise FileNotFoundError("go")
        command = [resolved_go, "test", "./...", "-count=1", "-run", "^" + re.escape(test_name) + "$"]
    else:
        group_error = f"unsupported verifier kind: {verifier_kind}"
        results[test_name] = False
        failures[test_name] = {
            "returncode": 2,
            "unsupported_verifier_kind": verifier_kind,
            "command": [],
        }
        break

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
                "command": list(command),
            }
    except FileNotFoundError as exc:
        failures[test_name] = {
            "returncode": 127,
            "command": list(command),
            "error": str(exc),
        }
        results[test_name] = False
        group_error = f"verifier executable unavailable: {exc}"
        break
    except subprocess.TimeoutExpired:
        failures[test_name] = {
            "returncode": 124,
            "timed_out": True,
            "command": list(command),
        }
        results[test_name] = False

all_passed = all(bool(value) for value in results.values()) if results else True
print(
    json.dumps(
        {
            "results": results,
            "executed": int(executed),
            "all_passed": bool(all_passed),
            "failures": failures,
            "error": group_error,
            "verifier_kind": verifier_kind,
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
    verifier_kind: str = "pytest",
) -> dict[str, Any]:
    """Verify terminal submission state inside the current task container."""
    normalized_verifier_kind = normalize_verifier_kind(verifier_kind)
    fail_targets = normalize_verifier_targets(fail_to_pass)
    pass_targets = normalize_verifier_targets(pass_to_pass)

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
        verifier_kind=normalized_verifier_kind,
    )
    pass_group = _verify_test_group(
        executor=executor,
        tests=pass_targets,
        verifier_timeout_sec=requested_timeout_sec,
        per_test_timeout_sec=per_test_timeout_sec,
        verifier_kind=normalized_verifier_kind,
    )

    fail_passed = bool(fail_group["all_passed"]) if fail_targets else True
    pass_passed = bool(pass_group["all_passed"]) if pass_targets else True
    infra_invalid = bool(fail_group.get("infra_invalid", False) or pass_group.get("infra_invalid", False))
    invalid_reason = (
        str(fail_group.get("invalid_reason", "")).strip()
        or str(pass_group.get("invalid_reason", "")).strip()
    )
    resolved = bool(fail_passed and pass_passed and not infra_invalid)
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
        "verifier_kind": normalized_verifier_kind,
        "fail_to_pass": list(fail_targets),
        "pass_to_pass": list(pass_targets),
        "fail_to_pass_results": dict(fail_group["results"]),
        "pass_to_pass_results": dict(pass_group["results"]),
        "fail_to_pass_failures": dict(fail_group.get("failures", {})),
        "pass_to_pass_failures": dict(pass_group.get("failures", {})),
        "fail_to_pass_stderr_tail": str(fail_group.get("stderr_tail", "")),
        "pass_to_pass_stderr_tail": str(pass_group.get("stderr_tail", "")),
        "fail_to_pass_all_passed": bool(fail_passed),
        "pass_to_pass_all_passed": bool(pass_passed),
        "fail_to_pass_verified": bool(fail_passed),
        "pass_to_pass_verified": bool(pass_passed),
        "verification_missing": not bool(fail_targets or pass_targets),
        "infra_invalid": infra_invalid,
        "invalid_reason": invalid_reason,
        "verification_error": verification_error,
        "resolved": resolved,
        "verification_feedback": _build_feedback(
            fail_group=fail_group,
            pass_group=pass_group,
            fail_to_pass=fail_targets,
            pass_to_pass=pass_targets,
            resolved=resolved,
            verifier_kind=normalized_verifier_kind,
            infra_invalid=infra_invalid,
            invalid_reason=invalid_reason,
        ),
    }


def _verify_test_group(
    *,
    executor: Any,
    tests: Sequence[str],
    verifier_timeout_sec: int,
    per_test_timeout_sec: int,
    verifier_kind: str,
) -> dict[str, Any]:
    if not tests:
        return {
            "results": {},
            "all_passed": True,
            "executed": 0,
            "error": "",
            "failures": {},
            "stderr_tail": "",
            "infra_invalid": False,
            "invalid_reason": "",
        }
    if verifier_kind not in _SUPPORTED_RUNTIME_VERIFIER_KINDS:
        return _invalid_group_result(
            tests=tests,
            error=f"verifier kind {verifier_kind!r} is not implemented in the runtime verifier",
            invalid_reason="verifier_crash",
        )

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
                verifier_kind=verifier_kind,
            ),
        },
    )
    response = executor.run(request)

    stdout_text = str(response.stdout or "")
    stderr_tail = str(response.stderr or "")[-4000:]
    if int(response.exit_code) != 0:
        return _invalid_group_result(
            tests=tests,
            error=f"verifier command failed with exit code {response.exit_code}",
            stderr_tail=stderr_tail,
            invalid_reason="verifier_crash",
        )

    payload = _decode_verifier_payload(stdout_text)
    if payload is None:
        return _invalid_group_result(
            tests=tests,
            error="verifier command returned invalid JSON payload",
            stderr_tail=stderr_tail,
            invalid_reason="verifier_crash",
        )

    payload_error = str(payload.get("error", "")).strip()
    if payload_error:
        return _invalid_group_result(
            tests=tests,
            error=payload_error,
            stderr_tail=stderr_tail,
            invalid_reason="verifier_crash",
            failures=payload.get("failures"),
        )

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

    raw_failures = payload.get("failures")
    failure_map: dict[str, dict[str, Any]] = {}
    if isinstance(raw_failures, Mapping):
        for name in tests:
            raw_failure = raw_failures.get(name)
            if isinstance(raw_failure, Mapping):
                failure_map[name] = dict(raw_failure)

    return {
        "results": result_map,
        "all_passed": bool(all_passed),
        "executed": int(payload.get("executed", 0) or 0),
        "error": "",
        "stderr_tail": stderr_tail,
        "failures": failure_map,
        "infra_invalid": False,
        "invalid_reason": "",
    }


def _invalid_group_result(
    *,
    tests: Sequence[str],
    error: str,
    invalid_reason: str,
    stderr_tail: str = "",
    failures: Any = None,
) -> dict[str, Any]:
    failure_map: dict[str, dict[str, Any]] = {}
    if isinstance(failures, Mapping):
        for name in tests:
            raw_failure = failures.get(name)
            if isinstance(raw_failure, Mapping):
                failure_map[name] = dict(raw_failure)
    return {
        "results": {name: False for name in tests},
        "all_passed": False,
        "executed": 0,
        "error": error,
        "failures": failure_map,
        "stderr_tail": stderr_tail,
        "infra_invalid": True,
        "invalid_reason": invalid_reason,
    }


def _build_verifier_shell_command(
    *,
    tests_json: str,
    per_test_timeout_sec: int,
    verifier_kind: str,
) -> str:
    prefix = (
        "set -eu; "
        'repo_root="${TASK_REPO_ROOT:-${SMALL_SWE_REPO_ROOT:-}}"; '
        'if [ -n "$repo_root" ] && [ ! -e "$repo_root" ]; then repo_root=""; fi; '
        'if [ -z "$repo_root" ]; then '
        'for candidate in /testbed /workspace /repo /app; do '
        'if [ -d "${candidate}/.git" ]; then repo_root="${candidate}"; break; fi; '
        "done; "
        'if [ -z "${repo_root}" ]; then '
        'for candidate in /testbed /workspace /repo /app; do '
        'if [ -d "${candidate}" ]; then repo_root="${candidate}"; break; fi; '
        "done; "
        "fi; "
        "fi; "
        'if [ -z "${repo_root}" ]; then '
        'echo "Unable to locate task repository root." >&2; '
        "exit 2; "
        "fi; "
    )
    if verifier_kind == "go_test":
        prefix += build_executable_resolver_shell(
            var_name="gobin",
            command_names=("go",),
            fallback_paths=("/usr/local/go/bin/go", "/usr/lib/go/bin/go", "/opt/go/bin/go"),
            not_found_message="Go executable missing in task container.",
        )
    suffix = (
        f"SMALL_SWE_TESTS_JSON={shlex.quote(tests_json)} "
        f"SMALL_SWE_PER_TEST_TIMEOUT_SEC={int(max(per_test_timeout_sec, 1))} "
        f"SMALL_SWE_VERIFIER_KIND={shlex.quote(verifier_kind)} "
        'TASK_REPO_ROOT="${repo_root}" '
        'SMALL_SWE_PYBIN="${pybin}" '
        + ('SMALL_SWE_GO_BIN="${gobin}" ' if verifier_kind == "go_test" else "")
        + '"${pybin}" -'
    )
    return prefix + build_python_interpreter_resolver_shell(var_name="pybin") + suffix


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


def _build_feedback(
    *,
    fail_group: Mapping[str, Any],
    pass_group: Mapping[str, Any],
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    resolved: bool,
    verifier_kind: str,
    infra_invalid: bool,
    invalid_reason: str,
) -> str:
    if infra_invalid:
        lines = [
            f"Verifier invalid ({verifier_kind}): {invalid_reason or 'unknown_invalid_reason'}",
        ]
        fail_error = str(fail_group.get("error", "")).strip()
        pass_error = str(pass_group.get("error", "")).strip()
        if fail_error:
            lines.append(f"FAIL_TO_PASS verifier error: {fail_error}")
        if pass_error:
            lines.append(f"PASS_TO_PASS verifier error: {pass_error}")
        return "\n".join(lines).strip()

    if resolved:
        return f"Verifier ({verifier_kind}): all FAIL_TO_PASS and PASS_TO_PASS targets passed."

    lines: list[str] = []
    if fail_to_pass:
        lines.append(
            "FAIL_TO_PASS passed="
            + str(bool(fail_group.get("all_passed", False)))
            + f" targets={len(fail_to_pass)}"
        )
    if pass_to_pass:
        lines.append(
            "PASS_TO_PASS passed="
            + str(bool(pass_group.get("all_passed", False)))
            + f" targets={len(pass_to_pass)}"
        )

    fail_error = str(fail_group.get("error", "")).strip()
    pass_error = str(pass_group.get("error", "")).strip()
    if fail_error:
        lines.append(f"FAIL_TO_PASS verifier error: {fail_error}")
    if pass_error:
        lines.append(f"PASS_TO_PASS verifier error: {pass_error}")

    fail_failed_tests = [name for name, passed in dict(fail_group.get("results", {})).items() if not passed]
    pass_failed_tests = [name for name, passed in dict(pass_group.get("results", {})).items() if not passed]
    if fail_failed_tests:
        lines.append("FAIL_TO_PASS failing targets: " + ", ".join(fail_failed_tests[:5]))
    if pass_failed_tests:
        lines.append("PASS_TO_PASS failing targets: " + ", ".join(pass_failed_tests[:5]))

    fail_tail = str(fail_group.get("stderr_tail", "")).strip()
    pass_tail = str(pass_group.get("stderr_tail", "")).strip()
    if fail_tail:
        lines.append("FAIL_TO_PASS stderr tail:\n" + fail_tail[-2000:])
    if pass_tail:
        lines.append("PASS_TO_PASS stderr tail:\n" + pass_tail[-2000:])
    return "\n".join(lines).strip()
