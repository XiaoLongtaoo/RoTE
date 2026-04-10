from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    command = [
        sys.executable,
        str(ROOT / "code" / "train.py"),
        "--config",
        str(ROOT / "code" / "configs" / "rote_sasrec.yaml"),
    ]
    command.extend(sys.argv[1:])
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
