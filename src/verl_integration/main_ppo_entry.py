"""Project-local SDPO entrypoint for verl PPO trainer.

This module keeps upstream verl unchanged while giving small-swe-train a stable
hook for runtime registration and process-wide patching.
"""

from __future__ import annotations


def _apply_local_runtime_bootstrap() -> None:
    from sitecustomize import apply_small_swe_runtime_patches

    apply_small_swe_runtime_patches()


_apply_local_runtime_bootstrap()

# Register local SDPO agent-loop integrations before trainer startup.
from verl_integration import swe_bridge_agent_loop  # noqa: F401,E402
from verl.trainer.main_ppo import main  # noqa: E402


if __name__ == "__main__":
    main()
