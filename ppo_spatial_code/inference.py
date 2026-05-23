from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpu_sched_experiment.simulator import PLACEMENT_POLICIES, RL_FEATURE_NAMES

if __package__:
    from .dataset import DEFAULT_2020_JOBS, DEFAULT_2023_NODES, DEFAULT_2023_PODS, load_inputs
    from .modele import STATE_FEATURE_NAMES, FairRCConfig, run_gru_episode
    from .utils import format_result, write_csv
else:  # pragma: no cover - supports direct script execution
    from dataset import DEFAULT_2020_JOBS, DEFAULT_2023_NODES, DEFAULT_2023_PODS, load_inputs
    from modele import STATE_FEATURE_NAMES, FairRCConfig, run_gru_episode
    from utils import format_result, write_csv


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Fair-RC-RL GRU scheduler.")
    parser.add_argument("--dataset", choices=["2023", "2020"], default="2023")
    parser.add_argument("--pod-csv", type=Path, default=DEFAULT_2023_PODS)
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_2023_NODES)
    parser.add_argument("--job-csv-2020", type=Path, default=DEFAULT_2020_JOBS)
    parser.add_argument("--job-offset", type=int, default=0)
    parser.add_argument("--max-jobs", type=int, default=2000)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--include-cpu-only", action="store_true")
    parser.add_argument("--placement", choices=PLACEMENT_POLICIES, default="best_fit")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--candidate-d", type=int, default=16)
    parser.add_argument("--candidate-sampling-mode", choices=["uniform", "uniform_pool"], default="uniform_pool")
    parser.add_argument("--candidate-pool-k", type=int, default=64)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--policy-model", choices=["gru"], default="gru")
    parser.add_argument("--gru-hidden-dim", type=int, default=32)
    parser.add_argument("--arrival-scale", type=float, default=1000.0)
    parser.add_argument("--sla-threshold", type=float, default=5000.0)
    parser.add_argument("--lag-target", type=float, default=100.0)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    args.model_checkpoint = args.model_checkpoint or args.model or args.output_dir / "fair_rc_rl_gru_best.pt"
    args.model = args.model_checkpoint
    args.out = args.out or args.output_dir / "fair_rc_rl_eval.csv"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    jobs, base_nodes = load_inputs(args, repo_root)
    config = FairRCConfig(
        placement=args.placement,
        top_k=args.top_k,
        candidate_d=args.candidate_d,
        candidate_sampling_mode=args.candidate_sampling_mode,
        candidate_pool_k=args.candidate_pool_k,
        sampling_temperature=args.sampling_temperature,
        policy_temperature=args.policy_temperature,
        sla_threshold=args.sla_threshold,
        lag_target=args.lag_target,
    )

    run_gru_inference(args, jobs, base_nodes, config)


def run_gru_inference(args, jobs, base_nodes, config: FairRCConfig) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional torch install
        raise RuntimeError("GRU policy requires PyTorch. Install torch to use --policy-model gru.") from exc

    if __package__:
        from .policy import GRUFairPolicy
    else:  # pragma: no cover - supports direct script execution
        from policy import GRUFairPolicy

    checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    hidden_dim = int(checkpoint.get("hidden_dim", args.gru_hidden_dim))
    policy = GRUFairPolicy(
        job_feature_dim=int(checkpoint.get("job_feature_dim", len(RL_FEATURE_NAMES))),
        state_feature_dim=int(checkpoint.get("state_feature_dim", len(STATE_FEATURE_NAMES))),
        queue_feature_dim=int(checkpoint.get("queue_feature_dim", checkpoint.get("job_feature_dim", len(RL_FEATURE_NAMES)))),
        hidden_dim=hidden_dim,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain a model state dict: {args.model_checkpoint}")
    policy.load_state_dict(state_dict)
    policy.eval()

    rows = []
    for seed in args.eval_seeds:
        with torch.no_grad():
            result, _, _ = run_gru_episode(
                jobs=jobs,
                base_nodes=base_nodes,
                policy=policy,
                seed=seed,
                config=config,
                sample=False,
            )
        result.update(
            {
                "dataset": args.dataset,
                "job_offset": args.job_offset,
                "max_jobs": args.max_jobs,
                "max_nodes": args.max_nodes,
                "top_k": args.top_k,
                "candidate_d": args.candidate_d,
                "candidate_sampling_mode": args.candidate_sampling_mode,
                "candidate_pool_k": args.candidate_pool_k,
                "arrival_scale": args.arrival_scale,
                "seed": seed,
                "policy_model": "gru",
            }
        )
        rows.append(result)
        print(format_result(result))

    write_csv(args.out, rows)
    print(f"\nWrote eval: {args.out}")


if __name__ == "__main__":
    main()
