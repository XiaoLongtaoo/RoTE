from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from engine.common import dump_config, load_config, parse_command_line_args, prepare_run_dirs
from engine.registry import get_display_name, get_runner


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified training entrypoint for RoTE.")
    parser.add_argument("--config", required=True, help="Path to a YAML config under code/configs/")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved config without training.")
    args, extra = parser.parse_known_args()

    overrides = parse_command_line_args(extra) if extra else {}
    config = prepare_run_dirs(load_config(args.config, overrides), create_dirs=not args.dry_run)

    if args.dry_run:
        print(f"[DRY RUN] {get_display_name(config)}")
        print(dump_config(config))
        return 0

    runner = get_runner(config)
    runner.train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
