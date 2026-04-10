from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from engine.common import dump_config, load_config, parse_command_line_args, prepare_evaluation_run
from engine.registry import get_display_name, get_runner


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified evaluation entrypoint for RoTE.")
    parser.add_argument("--config", required=True, help="Path to a YAML config under code/configs/")
    parser.add_argument("--state-dict-path", default=None, help="Optional checkpoint path. Defaults to the latest run.")
    parser.add_argument("--run-name", default=None, help="Optional run name to evaluate from under runs/.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved config without evaluation.")
    args, extra = parser.parse_known_args()

    overrides = parse_command_line_args(extra) if extra else {}
    if args.state_dict_path is not None:
        overrides["state_dict_path"] = args.state_dict_path
    if args.run_name is not None:
        overrides["run_name"] = args.run_name

    config = prepare_evaluation_run(load_config(args.config, overrides), create_dirs=not args.dry_run)

    if args.dry_run:
        print(f"[DRY RUN] evaluate {get_display_name(config)}")
        print(dump_config(config))
        return 0

    runner = get_runner(config)
    runner.evaluate(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
