"""Fallback shim for optional qwen_vl_utils dependency.

The upstream verl multiturn dataset imports `qwen_vl_utils` unconditionally for
vision helpers, even in text-only training flows. This project's RFT/SFT path
is text-only, so provide a lightweight shim that keeps imports working and
fails fast only when vision helpers are actually invoked.
"""

from __future__ import annotations


def _unsupported(*_args, **_kwargs):
    raise RuntimeError(
        "qwen_vl_utils vision helpers are unavailable in this runtime. "
        "Install and validate full vision dependencies to enable image/video data."
    )


fetch_image = _unsupported
fetch_video = _unsupported
