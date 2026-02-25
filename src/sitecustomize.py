"""Compatibility sitecustomize wrapper for project runtime patches.

When `PYTHONPATH` includes `src/` at interpreter startup, Python imports this
module automatically. The implementation lives in
`small_swe_runtime_patches.py` so explicit imports do not collide with system
`sitecustomize` modules.
"""

from __future__ import annotations

from small_swe_runtime_patches import apply_small_swe_runtime_patches


apply_small_swe_runtime_patches()
