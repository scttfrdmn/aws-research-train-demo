"""Thin training entry point (SPEC §6).

    uv run python train.py --domain molecular --feat graph --depth deep --max-steps 5

Everything domain-neutral lives in `spine.cli`; this file just forwards. The
same file runs locally (stage 2 smoke) and as the SageMaker job's `entry_point`
(stage 3+) — nothing here is hardcoded.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from spine.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
