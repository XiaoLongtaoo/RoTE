from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from engine.common import dump_config, load_config, parse_command_line_args
from engine.registry import get_display_name

torch = None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_latency(case, device: torch.device, warmup_steps: int, steps: int) -> dict[str, float]:
    case.model.eval()
    timings = []
    with torch.no_grad():
        for _ in range(warmup_steps):
            case.forward()
            _synchronize(device)

        for _ in range(steps):
            _synchronize(device)
            start = time.perf_counter()
            case.forward()
            _synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)

    return {
        "latency_ms_mean": round(statistics.mean(timings), 4),
        "latency_ms_std": round(statistics.pstdev(timings) if len(timings) > 1 else 0.0, 4),
    }


def _measure_flops(case, device: torch.device) -> int | None:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    try:
        case.model.eval()
        with torch.no_grad():
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                case.forward()
                _synchronize(device)
        total_flops = int(sum(getattr(event, "flops", 0) for event in prof.key_averages()))
        return total_flops or None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark one representative forward pass for SASRec, RoTE-SASRec, RPG, or RoTE-RPG."
    )
    parser.add_argument("--config", required=True, help="Config path under code/configs/")
    parser.add_argument("--metric", default="all", choices=["latency", "flops", "all"])
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size used for the sampled benchmark batch.")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Number of warmup forward passes.")
    parser.add_argument("--steps", type=int, default=5, help="Number of timed forward passes.")
    parser.add_argument("--num-workers", type=int, default=0, help="Data-loader workers for RPG benchmark setup.")
    parser.add_argument("--device", default=None, help="Override device, for example cpu or cuda:0.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved benchmark target without running.")
    args, extra = parser.parse_known_args()

    overrides = parse_command_line_args(extra) if extra else {}
    if args.device is not None:
        overrides["device"] = args.device
    config = load_config(args.config, overrides)

    benchmark_plan = {
        "display_name": get_display_name(config),
        "metric": args.metric,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "num_workers": args.num_workers,
    }

    if args.dry_run:
        print(f"[BENCHMARK] {benchmark_plan['metric']} -> {benchmark_plan['display_name']}")
        print(dump_config(config))
        print(json.dumps(benchmark_plan, indent=2, sort_keys=True))
        return 0

    try:
        global torch
        import torch
    except Exception as exc:  # pragma: no cover - benchmarking requires torch
        raise RuntimeError("benchmark_efficiency.py requires a PyTorch environment.") from exc

    from engine.benchmark_support import build_forward_case

    case = build_forward_case(config, batch_size=args.batch_size, num_workers=args.num_workers)
    device = torch.device(config["device"])
    try:
        results = {
            "display_name": get_display_name(config),
            "backbone": config["backbone"],
            "variant": config["variant"],
            "device": str(device),
            "batch_size": args.batch_size,
            "parameter_count": sum(param.numel() for param in case.model.parameters()),
        }

        if args.metric in {"latency", "all"}:
            results.update(_measure_latency(case, device=device, warmup_steps=args.warmup_steps, steps=args.steps))
        if args.metric in {"flops", "all"}:
            results["forward_flops"] = _measure_flops(case, device=device)

        print(json.dumps(results, indent=2, sort_keys=True))
        return 0
    finally:
        case.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
