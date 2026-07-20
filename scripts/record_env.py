from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_env


def _strip_mode_args(argv):
    forwarded = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--mode":
            skip_next = True
            continue
        if arg.startswith("--mode="):
            continue
        forwarded.append(arg)
    return forwarded


def main():
    forwarded = _strip_mode_args(sys.argv[1:])
    run_env.main(["--mode", "record", *forwarded])


if __name__ == "__main__":
    main()
