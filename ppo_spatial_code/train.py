from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpu_sched_experiment.simulator import PLACEMENT_POLICIES, RL_FEATURE_NAMES

if __package__:
    from .dataset import DEFAULT_2020_JOBS, DEFAULT_2023_NODES, DEFAULT_2023_PODS, load_inputs
    from .modele import (
        FairRCConfig,
        run_gru_episode,
        run_gru_episode_trace,
    )
    from .training_support import build_actor, build_critic, grad_l2_norm, replay_trace, run_gru_trace_rollout_task, state_feature_names
    from .utils import format_result, write_csv
else:  # pragma: no cover - supports direct script execution
    from dataset import DEFAULT_2020_JOBS, DEFAULT_2023_NODES, DEFAULT_2023_PODS, load_inputs
    from modele import (
        FairRCConfig,
        run_gru_episode,
        run_gru_episode_trace,
    )
    from training_support import build_actor, build_critic, grad_l2_norm, replay_trace, run_gru_trace_rollout_task, state_feature_names
    from utils import format_result, write_csv


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fair-RC-RL GRU scheduler.")
    parser.add_argument("--dataset", choices=["2023", "2020"], default="2023")
    parser.add_argument("--pod-csv", type=Path, default=DEFAULT_2023_PODS)
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_2023_NODES)
    parser.add_argument("--job-csv-2020", type=Path, default=DEFAULT_2020_JOBS)
    parser.add_argument("--job-offset", type=int, default=0)
    parser.add_argument("--max-jobs", type=int, default=2000)
    parser.add_argument("--eval-job-offset", type=int, default=None)
    parser.add_argument("--eval-max-jobs", type=int, default=None)
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
    parser.add_argument("--entropy-decay", type=float, default=0.995)
    parser.add_argument("--sla-threshold", type=float, default=5000.0)
    parser.add_argument("--arrival-scale", type=float, default=1000.0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--lambda-lr", type=float, default=0.001)
    parser.add_argument("--fairness-target", type=float, default=0.25)
    parser.add_argument("--sla-target", type=float, default=0.0)
    parser.add_argument("--wait-scale", type=float, default=10000.0)
    parser.add_argument("--slowdown-weight", type=float, default=1.0)
    parser.add_argument("--wait-weight", type=float, default=0.05)
    parser.add_argument("--util-weight", type=float, default=1.0)
    parser.add_argument("--frag-weight", type=float, default=0.5)
    parser.add_argument("--debt-weight", type=float, default=0.0)
    parser.add_argument("--lag-target", type=float, default=100.0, help="Target CFS-style Normalized Service Lag AUC")
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--reward-lag-level-weight", type=float, default=0.05)
    parser.add_argument("--reward-lag-growth-weight", type=float, default=1.0)
    parser.add_argument("--reward-lag-reduction-weight", type=float, default=1.0)
    parser.add_argument("--reward-waiting-queue-weight", type=float, default=0.0)
    parser.add_argument("--reward-gate-slowdown-weight", type=float, default=1.0)
    parser.add_argument("--reward-gate-wait-weight", type=float, default=0.5)
    parser.add_argument("--reward-gate-p95-weight", type=float, default=0.5)
    parser.add_argument("--reward-gate-fairness-weight", type=float, default=1.0)
    parser.add_argument("--reward-unfinished-weight", type=float, default=10.0)
    parser.add_argument("--reward-pass-penalty-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--train-log", type=Path, default=None)
    parser.add_argument("--episode-log", type=Path, default=None)
    parser.add_argument("--model-out", type=Path, default=None)
    parser.add_argument("--model-out-final", type=Path, default=None)
    parser.add_argument("--model-out-best", type=Path, default=None)
    parser.add_argument("--best-metrics-out", type=Path, default=None)
    parser.add_argument("--ref-slowdown", type=float, default=5.668)
    parser.add_argument("--ref-wait", type=float, default=1434.8)
    parser.add_argument("--ref-p95", type=float, default=4117.0)
    parser.add_argument("--ref-fairness", type=float, default=0.722)
    parser.add_argument(
        "--selection-metric",
        choices=["cfs_normalized_service_lag_auc", "cfs_normalized_lag_auc", "normalized_service_lag_auc"],
        default="cfs_normalized_service_lag_auc",
        help="Primary checkpoint metric after efficiency-gate penalties.",
    )
    parser.add_argument("--slowdown-gate-multiplier", type=float, default=1.10)
    parser.add_argument("--wait-gate-multiplier", type=float, default=1.20)
    parser.add_argument("--p95-gate-multiplier", type=float, default=1.20)
    parser.add_argument("--fairness-floor", type=float, default=0.70)
    parser.add_argument("--unfinished-penalty", type=float, default=1_000_000.0)
    parser.add_argument("--gate-penalty-scale", type=float, default=1000.0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    args = parser.parse_args(argv)
    args.out = args.out or args.output_dir / "fair_rc_rl.csv"
    args.train_log = args.train_log or args.output_dir / "fair_rc_rl_train_log.csv"
    args.episode_log = args.episode_log or args.output_dir / "fair_rc_rl_episode_log.csv"
    model_prefix = "fair_rc_rl_gru"
    args.model_out_final = args.model_out_final or args.model_out or args.output_dir / f"{model_prefix}_final.pt"
    args.model_out_best = args.model_out_best or args.output_dir / f"{model_prefix}_best.pt"
    args.best_metrics_out = args.best_metrics_out or args.output_dir / f"{model_prefix}_best_metrics.json"
    args.model_out = args.model_out_final
    if args.num_workers < 1:
        raise SystemExit("--num-workers must be at least 1.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    progress(
        f"loading dataset={args.dataset} train_offset={args.job_offset} train_jobs={args.max_jobs} "
        f"eval_offset={args.eval_job_offset} eval_jobs={args.eval_max_jobs}"
    )
    jobs, base_nodes = load_inputs(args, repo_root)
    eval_args = build_eval_args(args)
    eval_jobs, _ = load_inputs(eval_args, repo_root)
    progress(f"loaded train_jobs={len(jobs)} eval_jobs={len(eval_jobs)} nodes={len(base_nodes)}")
    train_gru(args, jobs, base_nodes, eval_jobs, eval_args)


def train_gru(args, jobs, base_nodes, eval_jobs, eval_args) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional torch install
        raise RuntimeError("GRU policy requires PyTorch. Install torch to use --policy-model gru.") from exc

    torch.manual_seed(args.seed)
    policy = build_actor(args.policy_model, args.gru_hidden_dim)
    critic = build_critic(args.policy_model, args.gru_hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr or args.lr)
    reward_baseline = None
    train_rows = []
    episode_rows = []
    policy_temperature = args.policy_temperature
    lambda_fair = 0.0
    lambda_sla = 0.0
    lambda_lag = 0.0
    best_score_so_far = float("inf")
    best_update = None
    last_eval_result = None
    reward_component_scales = {}

    for update in range(args.episodes):
        config = build_config(args, policy_temperature)
        rewards = []
        raw_rewards = []
        fairness_costs = []
        sla_costs = []
        lag_costs = []
        debt_gaps = []

        episode_seeds = [args.seed + update * args.batch_size + batch_idx for batch_idx in range(args.batch_size)]
        progress(
            f"u={update:03d}/{args.episodes - 1:03d} rollout start "
            f"batch={args.batch_size} workers={min(args.num_workers, args.batch_size)} "
            f"seeds={episode_seeds[0]}..{episode_seeds[-1]}"
        )
        if args.num_workers > 1 and args.batch_size > 1:
            policy_state_dict = {key: value.detach().cpu().numpy() for key, value in policy.state_dict().items()}
            tasks = [
                {
                    "jobs": jobs,
                    "base_nodes": base_nodes,
                    "policy_state_dict": policy_state_dict,
                    "policy_model": args.policy_model,
                    "hidden_dim": args.gru_hidden_dim,
                    "seed": episode_seed,
                    "config": config,
                }
                for episode_seed in episode_seeds
            ]
            worker_count = min(args.num_workers, args.batch_size)
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                rollout_outputs = list(executor.map(run_gru_trace_rollout_task, tasks))
        else:
            rollout_outputs = []
            for episode_seed in episode_seeds:
                policy.eval()
                with torch.no_grad():
                    rollout_outputs.append(
                        run_gru_episode_trace(
                            jobs=jobs,
                            base_nodes=base_nodes,
                            policy=policy,
                            seed=episode_seed,
                            config=config,
                            sample=True,
                        )
                    )
        rollout_decisions = [int(item[2]) for item in rollout_outputs]
        rollout_jobs_done = [int(item[0].get("jobs_done", 0)) for item in rollout_outputs]
        progress(
            f"u={update:03d} rollout done "
            f"decisions={sum(rollout_decisions)} "
            f"jobs={sum(rollout_jobs_done)}/{sum(int(item[0].get('jobs_loaded', 0)) for item in rollout_outputs)} "
            f"avg_cfs={np.mean([float(item[0].get('cfs_normalized_service_lag_auc', 0.0)) for item in rollout_outputs]):.2f}"
        )
        for batch_idx, (result, _log_data, decisions) in enumerate(rollout_outputs):
            episode_row = build_episode_log_row(
                args,
                update,
                batch_idx,
                episode_seeds[batch_idx],
                result,
                decisions,
                phase="train_rollout",
            )
            episode_rows.append(episode_row)
            write_csv(args.episode_log, episode_rows)

        all_reward_components = []
        episode_traces = []
        progress(f"u={update:03d} replay/update start")
        for batch_idx, (result, log_data, decisions) in enumerate(rollout_outputs):
            components = attach_terminal_gate_components(
                [dict(step.get("reward_components", {})) for step in log_data],
                result,
                args,
            )
            for step, component in zip(log_data, components):
                step["reward_components"] = component
            episode_traces.append((log_data, decisions))
            all_reward_components.extend(components)
            fairness_costs.append(result["fairness_cost"])
            sla_costs.append(result["sla_cost"])
            lag_costs.append(result["lag_cost"])
            debt_gaps.append(result["avg_debt_gap"])

        update_reward_component_scales(reward_component_scales, all_reward_components)
        ppo_stats = run_ppo_update(
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            critic_optimizer=critic_optimizer,
            episode_traces=episode_traces,
            config=config,
            args=args,
            reward_component_scales=reward_component_scales,
        )
        rewards = ppo_stats["episode_returns"]
        raw_rewards = ppo_stats["raw_rewards"]
        rewards_arr = np.asarray(rewards, dtype=float)
        batch_reward_mean = float(np.mean(rewards_arr)) if len(rewards_arr) else 0.0
        reward_baseline = batch_reward_mean if reward_baseline is None else 0.9 * reward_baseline + 0.1 * batch_reward_mean
        grad_norm = ppo_stats["grad_norm"]
        critic_loss_value = ppo_stats["critic_loss"]
        policy_entropy_value = ppo_stats["policy_entropy"]
        progress(
            f"u={update:03d} replay/update done "
            f"episodes={ppo_stats['valid_episodes']} step_rewards={ppo_stats['step_rewards']} "
            f"trainR={batch_reward_mean:.3f} grad={grad_norm:.3f} critic={critic_loss_value:.3f}"
        )

        mean_fair_cost = float(np.mean(fairness_costs))
        mean_sla_cost = float(np.mean(sla_costs))
        mean_lag_cost = float(np.mean(lag_costs))
        # Lagrangian multipliers are logged but no longer drive the reward.
        # The reward is now single-objective: minimize Service Lag directly.
        policy_temperature *= args.entropy_decay

        policy.eval()
        progress(f"u={update:03d} validation start seed={args.seed} eval_jobs={len(eval_jobs)}")
        with torch.no_grad():
            eval_result, _, _ = run_gru_episode(
                jobs=eval_jobs,
                base_nodes=base_nodes,
                policy=policy,
                seed=args.seed,
                config=build_config(args, 1.0),
                sample=False,
            )
        progress(
            f"u={update:03d} validation done "
            f"slow={eval_result['avg_slowdown']:.3f} "
            f"wait={eval_result['avg_wait']:.1f} "
            f"p95={eval_result['p95_wait']:.1f} "
            f"cfs={eval_result.get('cfs_normalized_service_lag_auc', 0.0):.2f}"
        )
        last_eval_result = dict(eval_result)
        selection = selection_diagnostics(eval_result, args)
        score = selection["composite_score"]
        is_best = score < best_score_so_far
        previous_best_score = best_score_so_far
        if is_best:
            best_score_so_far = score
            best_update = update
            save_gru_checkpoint(
                path=args.model_out_best,
                policy=policy,
                critic=critic,
                args=args,
                update=update,
                eval_result=eval_result,
                score=score,
                checkpoint_kind="best",
            )
            write_best_metrics(args.best_metrics_out, args, update, score, eval_result)
            progress(f"u={update:03d} new best score={score:.4f} path={args.model_out_best}")
        save_gru_checkpoint(
            path=per_update_checkpoint_path(args, update),
            policy=policy,
            critic=critic,
            args=args,
            update=update,
            eval_result=eval_result,
            score=score,
            checkpoint_kind="update",
        )
        progress(f"u={update:03d} checkpoint saved path={per_update_checkpoint_path(args, update)}")
        if hasattr(policy, "logit_scale"):
            scale_val = torch.clamp(policy.logit_scale.exp(), min=1.0, max=20.0).item()
        else:
            scale_val = 1.0

        row = {
            "update": update,
            "train_reward_mean": batch_reward_mean,
            "train_reward_std": float(np.std(rewards_arr)),
            "raw_reward_mean": float(np.mean(raw_rewards)) if raw_rewards else 0.0,
            "reward_baseline": float(reward_baseline),
            "train_avg_cfs_lag": np.mean([float(item[0].get('cfs_normalized_service_lag_auc', 0.0)) for item in rollout_outputs]),
            "train_avg_slowdown": np.mean([float(item[0].get('avg_slowdown', 0.0)) for item in rollout_outputs]),
            "train_avg_wait": np.mean([float(item[0].get('avg_wait', 0.0)) for item in rollout_outputs]),
            "train_jobs_done": sum(rollout_jobs_done),
            "train_decisions": sum(rollout_decisions),
            "candidate_sampling_mode": args.candidate_sampling_mode,
            "candidate_pool_k": args.candidate_pool_k,
            "candidate_d": args.candidate_d,
            "train_actual_candidate_size_mean": np.mean([float(item[0].get("actual_candidate_size_mean", 0.0)) for item in rollout_outputs]),
            "train_actual_candidate_size_p10": np.mean([float(item[0].get("actual_candidate_size_p10", 0.0)) for item in rollout_outputs]),
            "train_actual_candidate_size_p50": np.mean([float(item[0].get("actual_candidate_size_p50", 0.0)) for item in rollout_outputs]),
            "train_actual_candidate_size_p90": np.mean([float(item[0].get("actual_candidate_size_p90", 0.0)) for item in rollout_outputs]),
            "train_feasible_jobs_count_mean": np.mean([float(item[0].get("feasible_jobs_count_mean", 0.0)) for item in rollout_outputs]),
            "train_feasible_jobs_count_p10": np.mean([float(item[0].get("feasible_jobs_count_p10", 0.0)) for item in rollout_outputs]),
            "train_feasible_jobs_count_p50": np.mean([float(item[0].get("feasible_jobs_count_p50", 0.0)) for item in rollout_outputs]),
            "train_feasible_jobs_count_p90": np.mean([float(item[0].get("feasible_jobs_count_p90", 0.0)) for item in rollout_outputs]),
            "train_highest_risk_selected_ratio": np.mean([float(item[0].get("highest_risk_selected_ratio", 0.0)) for item in rollout_outputs]),
            "train_selected_risk_rank_mean": np.mean([float(item[0].get("selected_risk_rank_mean", 0.0)) for item in rollout_outputs]),
            "train_pass_action_ratio": np.mean([float(item[0].get("pass_action_ratio", 0.0)) for item in rollout_outputs]),
            "train_pass_actions": sum(int(item[0].get("pass_actions", 0)) for item in rollout_outputs),
            "lambda_fair": lambda_fair,
            "lambda_sla": lambda_sla,
            "lambda_lag": lambda_lag,
            "fairness_cost_mean": mean_fair_cost,
            "sla_cost_mean": mean_sla_cost,
            "lag_cost_mean": mean_lag_cost,
            "train_avg_debt_gap_mean": float(np.mean(debt_gaps)),
            "debt_weight": args.debt_weight,
            "policy_temperature": policy_temperature,
            "policy_entropy": policy_entropy_value,
            "ppo_approx_kl": ppo_stats.get("approx_kl", 0.0),
            "ppo_clip_fraction": ppo_stats.get("clip_fraction", 0.0),
            "ppo_advantage_mean": ppo_stats.get("advantage_mean", 0.0),
            "ppo_advantage_std": ppo_stats.get("advantage_std", 0.0),
            "ppo_explained_variance": ppo_stats.get("explained_variance", 0.0),
            "grad_norm": grad_norm,
            "critic_loss": critic_loss_value,
            "scale": scale_val,
            "reward_scale_cost_cfs_log_level": reward_component_scales.get("cost_cfs_log_level", 0.0),
            "reward_scale_cost_cfs_log_growth": reward_component_scales.get("cost_cfs_log_growth", 0.0),
            "reward_scale_gain_cfs_log_reduction": reward_component_scales.get("gain_cfs_log_reduction", 0.0),
            "reward_scale_cost_pass_action": reward_component_scales.get("cost_pass_action", 0.0),
            "reward_scale_gate_slowdown": reward_component_scales.get("gate_slowdown", 0.0),
            "reward_scale_gate_p95_wait": reward_component_scales.get("gate_p95_wait", 0.0),
            "reward_component_cost_cfs_log_level_mean": reward_component_mean(all_reward_components, "cost_cfs_log_level"),
            "reward_component_cost_cfs_log_growth_mean": reward_component_mean(all_reward_components, "cost_cfs_log_growth"),
            "reward_component_gain_cfs_log_reduction_mean": reward_component_mean(all_reward_components, "gain_cfs_log_reduction"),
            "reward_component_cost_waiting_queue_size_log_mean": reward_component_mean(all_reward_components, "cost_waiting_queue_size_log"),
            "reward_component_cost_pass_action_mean": reward_component_mean(all_reward_components, "cost_pass_action"),
            "reward_weight_lag_level": args.reward_lag_level_weight,
            "reward_weight_lag_growth": args.reward_lag_growth_weight,
            "reward_weight_lag_reduction": args.reward_lag_reduction_weight,
            "reward_weight_waiting_queue": args.reward_waiting_queue_weight,
            "reward_weight_pass_penalty": args.reward_pass_penalty_weight,
            "reward_weight_unfinished": args.reward_unfinished_weight,
            "eval_avg_slowdown": eval_result["avg_slowdown"],
            "eval_avg_wait": eval_result["avg_wait"],
            "eval_p95_wait": eval_result["p95_wait"],
            "eval_gpu_utilization": eval_result["gpu_utilization"],
            "eval_fairness": eval_result["jain_inverse_slowdown"],
            "eval_fairness_cost": eval_result["fairness_cost"],
            "eval_sla_cost": eval_result["sla_cost"],
            "eval_avg_debt_gap": eval_result.get("avg_debt_gap", 0.0),
            "eval_max_debt_gap": eval_result.get("max_debt_gap", 0.0),
            "eval_avg_candidate_size": eval_result.get("avg_candidate_size", 0.0),
            "eval_actual_candidate_size_mean": eval_result.get("actual_candidate_size_mean", 0.0),
            "eval_actual_candidate_size_p10": eval_result.get("actual_candidate_size_p10", 0.0),
            "eval_actual_candidate_size_p50": eval_result.get("actual_candidate_size_p50", 0.0),
            "eval_actual_candidate_size_p90": eval_result.get("actual_candidate_size_p90", 0.0),
            "eval_p50_candidate_size": eval_result.get("p50_candidate_size", 0.0),
            "eval_p90_candidate_size": eval_result.get("p90_candidate_size", 0.0),
            "eval_single_candidate_ratio": eval_result.get("single_candidate_ratio", 0.0),
            "eval_feasible_jobs_count_mean": eval_result.get("feasible_jobs_count_mean", 0.0),
            "eval_feasible_jobs_count_p10": eval_result.get("feasible_jobs_count_p10", 0.0),
            "eval_feasible_jobs_count_p50": eval_result.get("feasible_jobs_count_p50", 0.0),
            "eval_feasible_jobs_count_p90": eval_result.get("feasible_jobs_count_p90", 0.0),
            "eval_highest_risk_selected_ratio": eval_result.get("highest_risk_selected_ratio", 0.0),
            "eval_selected_risk_rank_mean": eval_result.get("selected_risk_rank_mean", 0.0),
            "eval_pass_action_ratio": eval_result.get("pass_action_ratio", 0.0),
            "eval_pass_actions": eval_result.get("pass_actions", 0),
            "eval_avg_feasible_node_count_chosen": eval_result.get("avg_feasible_node_count_chosen", 0.0),
            "eval_avg_feasible_node_count_candidates": eval_result.get("avg_feasible_node_count_candidates", 0.0),
            "eval_avg_resource_score_chosen": eval_result.get("avg_resource_score_chosen", 0.0),
            "eval_avg_resource_score_candidates": eval_result.get("avg_resource_score_candidates", 0.0),
            "eval_avg_blocked_count_topk": eval_result.get("avg_blocked_count_topk", 0.0),
            "eval_blocked_count_topk_p95": eval_result.get("blocked_count_topk_p95", 0.0),
            "eval_feasible_count_topk_avg": eval_result.get("feasible_count_topk_avg", 0.0),
            "eval_feasible_count_topk_p10": eval_result.get("feasible_count_topk_p10", 0.0),
            "eval_feasible_count_topk_p50": eval_result.get("feasible_count_topk_p50", 0.0),
            "eval_feasible_count_topk_p90": eval_result.get("feasible_count_topk_p90", 0.0),
            "eval_median_slowdown": eval_result.get("median_slowdown", 0.0),
            "eval_p95_slowdown": eval_result.get("p95_slowdown", 0.0),
            "eval_max_slowdown": eval_result.get("max_slowdown", 0.0),
            "eval_log_slowdown_std": eval_result.get("log_slowdown_std", 0.0),
            "eval_tail_fairness_gap": eval_result.get("tail_fairness_gap", 0.0),
            "eval_large_gpu_avg_wait": eval_result.get("large_gpu_avg_wait", 0.0),
            "eval_large_gpu_p95_wait": eval_result.get("large_gpu_p95_wait", 0.0),
            "eval_large_gpu_avg_slowdown": eval_result.get("large_gpu_avg_slowdown", 0.0),
            "eval_gpu_group_wait_max_gap": eval_result.get("gpu_group_wait_max_gap", 0.0),
            "eval_gpu_group_slowdown_max_gap": eval_result.get("gpu_group_slowdown_max_gap", 0.0),
            "eval_service_share_l1_gap": eval_result.get("service_share_l1_gap", 0.0),
            "eval_service_share_l2_gap": eval_result.get("service_share_l2_gap", 0.0),
            "eval_max_service_under_share": eval_result.get("max_service_under_share", 0.0),
            "eval_max_service_lag_over_time": eval_result.get("max_service_lag_over_time", 0.0),
            "eval_avg_positive_service_lag_over_time": eval_result.get("avg_positive_service_lag_over_time", 0.0),
            "eval_service_lag_auc": eval_result.get("service_lag_auc", 0.0),
            "eval_service_lag_l2_auc": eval_result.get("service_lag_l2_auc", 0.0),
            "eval_normalized_service_lag_auc": eval_result.get("normalized_service_lag_auc", 0.0),
            "eval_service_lag_variance_avg": eval_result.get("service_lag_variance_avg", 0.0),
            "eval_service_lag_variance_max": eval_result.get("service_lag_variance_max", 0.0),
            "eval_cfs_max_service_lag_over_time": eval_result.get("cfs_max_service_lag_over_time", 0.0),
            "eval_cfs_avg_positive_service_lag_over_time": eval_result.get("cfs_avg_positive_service_lag_over_time", 0.0),
            "eval_cfs_service_lag_auc": eval_result.get("cfs_service_lag_auc", 0.0),
            "eval_cfs_service_lag_l2_auc": eval_result.get("cfs_service_lag_l2_auc", 0.0),
            "eval_cfs_service_lag_l1_auc": eval_result.get("cfs_service_lag_l1_auc", 0.0),
            "eval_cfs_normalized_service_lag_auc": eval_result.get("cfs_normalized_service_lag_auc", 0.0),
            "eval_cfs_max_normalized_service_lag": eval_result.get("cfs_max_normalized_service_lag", 0.0),
            "eval_cfs_avg_max_normalized_service_lag": eval_result.get("cfs_avg_max_normalized_service_lag", 0.0),
            "eval_cfs_normalized_lag_auc": eval_result.get("cfs_normalized_lag_auc", 0.0),
            "eval_cfs_lag_over_0p5_ratio": eval_result.get("cfs_lag_over_0p5_ratio", 0.0),
            "eval_cfs_lag_over_1p0_ratio": eval_result.get("cfs_lag_over_1p0_ratio", 0.0),
            **selection,
            "score_delta_vs_prev_best": 0.0 if not np.isfinite(previous_best_score) else score - previous_best_score,
            "best_score_so_far": best_score_so_far,
            "is_best": int(is_best),
            "best_update": best_update,
            "train_job_offset": args.job_offset,
            "train_max_jobs": args.max_jobs,
            "eval_job_offset": eval_args.job_offset,
            "eval_max_jobs": eval_args.max_jobs,
        }
        train_rows.append(row)
        write_csv(args.train_log, train_rows)
        print(format_training_progress(row), flush=True)
        if args.early_stop_patience > 0 and best_update is not None:
            if update - best_update >= args.early_stop_patience:
                print(f"Early stopping at update {update}; best update was {best_update}.")
                break

    eval_rows = []
    final_config = build_config(args, 1.0)
    policy.eval()
    for eval_idx, eval_seed in enumerate(args.eval_seeds):
        with torch.no_grad():
            result, final_trace, final_decisions = run_gru_episode_trace(
                jobs=eval_jobs,
                base_nodes=base_nodes,
                policy=policy,
                seed=eval_seed,
                config=final_config,
                sample=False,
            )
        result.update(
            {
                "dataset": args.dataset,
                "job_offset": eval_args.job_offset,
                "max_jobs": eval_args.max_jobs,
                "train_job_offset": args.job_offset,
                "train_max_jobs": args.max_jobs,
                "max_nodes": args.max_nodes,
                "top_k": args.top_k,
                "candidate_d": args.candidate_d,
                "candidate_sampling_mode": args.candidate_sampling_mode,
                "candidate_pool_k": args.candidate_pool_k,
                "arrival_scale": args.arrival_scale,
                "seed": eval_seed,
                "eval_index": eval_idx,
                "policy_model": args.policy_model,
            }
        )
        eval_rows.append(result)
        write_csv(args.out, eval_rows)
        episode_rows.append(
            build_episode_log_row(
                args,
                train_rows[-1]["update"] if train_rows else args.episodes - 1,
                eval_idx,
                eval_seed,
                result,
                final_decisions,
                phase="final_eval",
                global_episode=(train_rows[-1]["update"] + 1 if train_rows else args.episodes) * int(args.batch_size)
                + int(eval_idx),
            )
        )
        write_csv(args.episode_log, episode_rows)
        print(format_result(result), flush=True)

    save_gru_checkpoint(
        path=args.model_out_final,
        policy=policy,
        critic=critic,
        args=args,
        update=train_rows[-1]["update"] if train_rows else args.episodes - 1,
        eval_result=last_eval_result,
        score=train_rows[-1].get("composite_score") if train_rows else None,
        checkpoint_kind="final",
    )
    write_csv(args.train_log, train_rows)
    write_csv(args.episode_log, episode_rows)
    write_csv(args.out, eval_rows)
    print(f"\nWrote eval: {args.out}")
    print(f"Wrote train log: {args.train_log}")
    print(f"Wrote episode log: {args.episode_log}")
    print(f"Wrote GRU final model: {args.model_out_final}")
    print(f"Wrote GRU best model: {args.model_out_best}")
    print(f"Wrote GRU best metrics: {args.best_metrics_out}")


def build_config(args, policy_temperature: float) -> FairRCConfig:
    return FairRCConfig(
        placement=args.placement,
        top_k=args.top_k,
        candidate_d=args.candidate_d,
        candidate_sampling_mode=args.candidate_sampling_mode,
        candidate_pool_k=args.candidate_pool_k,
        sampling_temperature=args.sampling_temperature,
        policy_temperature=policy_temperature,
        sla_threshold=args.sla_threshold,
        lag_target=args.lag_target,
    )


def build_eval_args(args) -> argparse.Namespace:
    eval_args = argparse.Namespace(**vars(args))
    if args.eval_job_offset is not None:
        eval_args.job_offset = args.eval_job_offset
    if args.eval_max_jobs is not None:
        eval_args.max_jobs = args.eval_max_jobs
    return eval_args


def progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def run_ppo_update(
    policy,
    critic,
    optimizer,
    critic_optimizer,
    episode_traces: list[tuple[list[dict], int]],
    config: FairRCConfig,
    args,
    reward_component_scales: dict[str, float],
) -> dict:
    import torch

    stats = {
        "episode_returns": [],
        "raw_rewards": [],
        "valid_episodes": 0,
        "step_rewards": 0,
        "grad_norm": 0.0,
        "critic_loss": 0.0,
        "policy_entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "advantage_mean": 0.0,
        "advantage_std": 0.0,
        "explained_variance": 0.0,
    }
    epochs = max(1, int(args.ppo_epochs))
    for _epoch in range(epochs):
        policy.train()
        critic.train()
        entries = []
        flat_advantages = []
        flat_returns = []
        flat_values = []
        if _epoch == 0:
            stats["episode_returns"] = []
            stats["raw_rewards"] = []
            stats["valid_episodes"] = 0
            stats["step_rewards"] = 0
        for trace, decisions in episode_traces:
            if decisions <= 0 or not trace:
                continue
            replayed = replay_trace(policy, critic, trace, config, args.policy_model)
            if replayed.log_probs is None or replayed.values is None:
                continue
            step_rewards = scalarize_reward_components(replayed.reward_components, reward_component_scales, args)
            if not step_rewards:
                continue
            log_probs = replayed.log_probs
            values = replayed.values.squeeze(-1) if replayed.values.dim() > 1 else replayed.values
            old_log_probs = replayed.old_log_probs if replayed.old_log_probs is not None else log_probs.detach()
            length = min(len(step_rewards), values.shape[0], log_probs.shape[0], old_log_probs.shape[0])
            if length <= 0:
                continue
            log_probs = log_probs[:length]
            values = values[:length]
            old_log_probs = old_log_probs[:length].detach()
            entropies = None if replayed.entropies is None else replayed.entropies[:length]
            returns, advantages = generalized_advantage_estimates(
                step_rewards[:length],
                values.detach(),
                gamma=float(args.gamma),
                gae_lambda=float(args.gae_lambda),
            )
            entries.append((log_probs, old_log_probs, entropies, values, returns, advantages))
            flat_advantages.append(advantages.reshape(-1))
            flat_returns.append(returns.reshape(-1))
            flat_values.append(values.detach().reshape(-1))
            if _epoch == 0:
                stats["valid_episodes"] += 1
                stats["step_rewards"] += int(length)
                stats["episode_returns"].append(float(sum(step_rewards[:length])))
                stats["raw_rewards"].append(
                    float(np.mean([component.get("cost_cfs_log_level", 0.0) for component in replayed.reward_components]))
                    if replayed.reward_components
                    else 0.0
                )
        if not entries:
            break

        all_advantages = torch.cat(flat_advantages)
        adv_mean = all_advantages.mean()
        adv_std = all_advantages.std(unbiased=False)
        if float(adv_std.detach().cpu().item()) <= 1e-8:
            adv_std = torch.tensor(1.0, dtype=all_advantages.dtype, device=all_advantages.device)
        stats["advantage_mean"] = float(adv_mean.detach().cpu().item())
        stats["advantage_std"] = float(adv_std.detach().cpu().item())

        policy_losses = []
        value_losses = []
        entropy_terms = []
        approx_kls = []
        clip_fractions = []
        for log_probs, old_log_probs, entropies, values, returns, advantages in entries:
            norm_advantages = (advantages - adv_mean.to(advantages.device)) / (adv_std.to(advantages.device) + 1e-8)
            log_ratio = log_probs - old_log_probs.to(log_probs.device)
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - float(args.ppo_clip_eps), 1.0 + float(args.ppo_clip_eps))
            unclipped = ratio * norm_advantages.detach()
            clipped = clipped_ratio * norm_advantages.detach()
            policy_losses.append(-torch.minimum(unclipped, clipped).mean())
            value_losses.append(torch.nn.functional.smooth_l1_loss(values, returns.to(values.device)))
            if entropies is not None:
                entropy_terms.append(entropies.mean())
            approx_kls.append((old_log_probs.to(log_probs.device) - log_probs).mean().detach())
            clip_fractions.append((torch.abs(ratio.detach() - 1.0) > float(args.ppo_clip_eps)).float().mean())

        optimizer.zero_grad()
        critic_optimizer.zero_grad()
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropy_terms).mean() if entropy_terms else torch.tensor(0.0, device=policy_loss.device)
        loss = policy_loss + float(args.value_loss_weight) * value_loss - float(args.entropy_weight) * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(args.max_grad_norm))
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=float(args.max_grad_norm))
        stats["grad_norm"] = grad_l2_norm(policy.parameters())
        optimizer.step()
        critic_optimizer.step()

        stats["critic_loss"] = float(value_loss.detach().cpu().item())
        stats["policy_entropy"] = float(entropy.detach().cpu().item())
        stats["approx_kl"] = float(torch.stack(approx_kls).mean().cpu().item()) if approx_kls else 0.0
        stats["clip_fraction"] = float(torch.stack(clip_fractions).mean().cpu().item()) if clip_fractions else 0.0
        if flat_returns and flat_values:
            stats["explained_variance"] = explained_variance(torch.cat(flat_values), torch.cat(flat_returns))
    return stats


def generalized_advantage_estimates(step_rewards: list[float], values, gamma: float, gae_lambda: float):
    import torch

    rewards = torch.as_tensor(step_rewards, dtype=torch.float32, device=values.device)
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.tensor(0.0, dtype=torch.float32, device=values.device)
    next_value = torch.tensor(0.0, dtype=torch.float32, device=values.device)
    for idx in range(len(rewards) - 1, -1, -1):
        delta = rewards[idx] + float(gamma) * next_value - values[idx]
        last_advantage = delta + float(gamma) * float(gae_lambda) * last_advantage
        advantages[idx] = last_advantage
        next_value = values[idx]
    returns = advantages + values
    return returns.detach(), advantages.detach()


def explained_variance(values, returns) -> float:
    import torch

    if returns.numel() <= 1:
        return 0.0
    return_var = torch.var(returns)
    if float(return_var.detach().cpu().item()) <= 1e-12:
        return 0.0
    residual_var = torch.var(returns - values)
    return float((1.0 - residual_var / return_var).detach().cpu().item())


def update_reward_component_scales(scales: dict[str, float], components: list[dict[str, float]]) -> None:
    # Disabled adaptive scaling to maintain physical meaning of log differences
    if not components:
        return
    for component in components:
        for key in component.keys():
            scales[key] = 1.0


def reward_component_mean(components: list[dict[str, float]], key: str) -> float:
    values = [float(component.get(key, 0.0)) for component in components]
    return float(np.mean(values)) if values else 0.0


def scalarize_reward_components(components: list[dict[str, float]], scales: dict[str, float], args=None) -> list[float]:
    rewards = []
    weights = {
        "cost_cfs_log_level": getattr(args, "reward_lag_level_weight", 0.05),
        "cost_cfs_log_growth": getattr(args, "reward_lag_growth_weight", 1.0),
        "gain_cfs_log_reduction": getattr(args, "reward_lag_reduction_weight", 1.0),
        "cost_waiting_queue_size_log": getattr(args, "reward_waiting_queue_weight", 0.1),
        "gate_slowdown": getattr(args, "reward_gate_slowdown_weight", 1.0),
        "gate_wait": getattr(args, "reward_gate_wait_weight", 0.5),
        "gate_p95_wait": getattr(args, "reward_gate_p95_weight", 0.5),
        "gate_fairness": getattr(args, "reward_gate_fairness_weight", 1.0),
        "unfinished_jobs": getattr(args, "reward_unfinished_weight", 10.0),
        "cost_pass_action": getattr(args, "reward_pass_penalty_weight", 0.5),
    }
    
    for component in components:
        reward = 0.0
        for key, value in component.items():
            weight = weights.get(key, 0.1)
            
            if key.startswith("gain_"):
                reward += float(value) * weight
            elif key.startswith("cost_") or key.startswith("gate_") or key.startswith("unfinished"):
                reward -= float(value) * weight
        rewards.append(reward)
    return rewards


def returns_from_step_rewards(step_rewards: list[float], values_tensor):
    import torch

    device = values_tensor.device
    returns = []
    running = 0.0
    for reward in reversed(step_rewards):
        running = float(reward) + running
        returns.append(running)
    returns.reverse()
    return torch.as_tensor(returns, dtype=torch.float32, device=device)


def attach_terminal_gate_components(components: list[dict[str, float]], result: dict, args) -> list[dict[str, float]]:
    if not components:
        return []
    terminal = {
        "gate_slowdown": gate_violation(
            float(result.get("avg_slowdown", 0.0)),
            float(args.ref_slowdown) * float(args.slowdown_gate_multiplier),
        ),
        "gate_wait": gate_violation(
            float(result.get("avg_wait", 0.0)),
            float(args.ref_wait) * float(args.wait_gate_multiplier),
        ),
        "gate_p95_wait": gate_violation(
            float(result.get("p95_wait", 0.0)),
            float(args.ref_p95) * float(args.p95_gate_multiplier),
        ),
        "gate_fairness": (
            max(0.0, (float(args.fairness_floor) - float(result.get("jain_inverse_slowdown", 1.0))) / float(args.fairness_floor))
            if args.fairness_floor > 0
            else 0.0
        ),
        "unfinished_jobs": float(result.get("jobs_unfinished", 0.0)),
    }
    updated = [dict(component) for component in components]
    updated[-1].update(terminal)
    return updated


def build_episode_log_row(
    args,
    update: int,
    batch_idx: int,
    seed: int,
    result: dict,
    decisions: int,
    phase: str = "train_rollout",
    global_episode: int | None = None,
) -> dict:
    selection = selection_diagnostics(result, args)
    return {
        "phase": phase,
        "update": int(update),
        "batch_idx": int(batch_idx),
        "global_episode": int(global_episode) if global_episode is not None else int(update) * int(args.batch_size) + int(batch_idx),
        "seed": int(seed),
        "eval_index": result.get("eval_index", ""),
        "train_job_offset": args.job_offset,
        "train_max_jobs": args.max_jobs,
        "candidate_sampling_mode": args.candidate_sampling_mode,
        "candidate_pool_k": args.candidate_pool_k,
        "candidate_d": args.candidate_d,
        "jobs_done": result.get("jobs_done", 0),
        "jobs_loaded": result.get("jobs_loaded", 0),
        "jobs_unfinished": result.get("jobs_unfinished", 0),
        "decisions": int(decisions),
        "avg_slowdown": result.get("avg_slowdown", 0.0),
        "median_slowdown": result.get("median_slowdown", 0.0),
        "p95_slowdown": result.get("p95_slowdown", 0.0),
        "avg_wait": result.get("avg_wait", 0.0),
        "p95_wait": result.get("p95_wait", 0.0),
        "p99_wait": result.get("p99_wait", 0.0),
        "gpu_utilization": result.get("gpu_utilization", 0.0),
        "fairness": result.get("jain_inverse_slowdown", 0.0),
        "fairness_cost": result.get("fairness_cost", 0.0),
        "sla_cost": result.get("sla_cost", 0.0),
        "lag_cost": result.get("lag_cost", 0.0),
        "avg_debt_gap": result.get("avg_debt_gap", 0.0),
        "max_debt_gap": result.get("max_debt_gap", 0.0),
        "avg_candidate_size": result.get("avg_candidate_size", 0.0),
        "actual_candidate_size_mean": result.get("actual_candidate_size_mean", 0.0),
        "actual_candidate_size_p10": result.get("actual_candidate_size_p10", 0.0),
        "actual_candidate_size_p50": result.get("actual_candidate_size_p50", 0.0),
        "actual_candidate_size_p90": result.get("actual_candidate_size_p90", 0.0),
        "feasible_jobs_count_mean": result.get("feasible_jobs_count_mean", 0.0),
        "feasible_jobs_count_p10": result.get("feasible_jobs_count_p10", 0.0),
        "feasible_jobs_count_p50": result.get("feasible_jobs_count_p50", 0.0),
        "feasible_jobs_count_p90": result.get("feasible_jobs_count_p90", 0.0),
        "highest_risk_selected_ratio": result.get("highest_risk_selected_ratio", 0.0),
        "selected_risk_rank_mean": result.get("selected_risk_rank_mean", 0.0),
        "pass_action_ratio": result.get("pass_action_ratio", 0.0),
        "pass_actions": result.get("pass_actions", 0),
        "cfs_normalized_service_lag_auc": result.get("cfs_normalized_service_lag_auc", 0.0),
        "cfs_normalized_lag_auc": result.get("cfs_normalized_lag_auc", 0.0),
        "cfs_max_normalized_service_lag": result.get("cfs_max_normalized_service_lag", 0.0),
        "cfs_lag_over_0p5_ratio": result.get("cfs_lag_over_0p5_ratio", 0.0),
        "cfs_lag_over_1p0_ratio": result.get("cfs_lag_over_1p0_ratio", 0.0),
        **selection,
    }


def format_training_progress(row: dict) -> str:
    best_mark = "*" if row.get("is_best") else " "
    score_text = ""
    if "composite_score" in row:
        delta = row.get("score_delta_vs_prev_best", 0.0)
        delta_text = "new" if row.get("is_best") else f"{delta:+.4f}"
        score_text = (
            f"score={row['composite_score']:7.4f} "
            f"primary={row.get('selection_primary_value', 0.0):7.2f} "
            f"gatePen={row.get('selection_gate_penalty', 0.0):7.2f} "
            f"dBest={delta_text:>8s} "
            f"best={row['best_score_so_far']:7.4f}@{row['best_update']} "
        )
    gate_text = (
        f"gates[ok={int(row.get('gate_all_clear', 0))} "
        f"total={row.get('gate_violation_total', 0.0):.3f} "
        f"slow={row.get('gate_slowdown_ratio', 0.0):.2f}x/{row.get('gate_slowdown_violation', 0.0):.3f} "
        f"wait={row.get('gate_wait_ratio', 0.0):.2f}x/{row.get('gate_wait_violation', 0.0):.3f} "
        f"p95={row.get('gate_p95_wait_ratio', 0.0):.2f}x/{row.get('gate_p95_wait_violation', 0.0):.3f} "
        f"fair={row.get('gate_fairness_ratio', 0.0):.2f}x/{row.get('gate_fairness_violation', 0.0):.3f}] "
    )
    return (
        f"u={int(row['update']):03d}{best_mark} "
        f"trainR={row['train_reward_mean']:9.3f}+/-{row['train_reward_std']:.3f} "
        f"base={row['reward_baseline']:9.3f} "
        f"raw={row['raw_reward_mean']:9.3f} "
        f"train[cfs={row.get('train_avg_cfs_lag', 0.0):7.2f} "
        f"slow={row.get('train_avg_slowdown', 0.0):6.3f} "
        f"wait={row.get('train_avg_wait', 0.0):7.1f} "
        f"cand={row.get('train_actual_candidate_size_mean', 0.0):4.1f}/{row.get('candidate_d', 0)} "
        f"feas={row.get('train_feasible_jobs_count_mean', 0.0):4.1f} "
        f"pass={row.get('train_pass_action_ratio', 0.0):.2f} "
        f"jobs={int(row.get('train_jobs_done', 0))} "
        f"dec={int(row.get('train_decisions', 0))}] "
        f"{score_text}"
        f"eval[slow={row['eval_avg_slowdown']:7.3f} "
        f"cfsLag={row.get('eval_cfs_normalized_service_lag_auc', row.get('eval_normalized_service_lag_auc', 0.0)):7.1f} "
        f"cfsMax={row.get('eval_cfs_max_normalized_service_lag', 0.0):5.2f} "
        f"over1={row.get('eval_cfs_lag_over_1p0_ratio', 0.0):.2f} "
        f"wait={row['eval_avg_wait']:8.1f} "
        f"p95={row.get('eval_p95_wait', 0.0):8.1f} "
        f"fair={row['eval_fairness']:.4f}] "
        f"pass[tr={row.get('train_pass_action_ratio', 0.0):.2f} ev={row.get('eval_pass_action_ratio', 0.0):.2f}] "
        f"{gate_text}"
        f"train[fairC={row['fairness_cost_mean']:.4f} "
        f"slaC={row['sla_cost_mean']:.4f} "
        f"lagC={row.get('lag_cost_mean', 0.0):.4f}] "
        f"opt[grad={row['grad_norm']:.3f} "
        f"critic={row.get('critic_loss', 0.0):.3f} "
        f"ent={row.get('policy_entropy', 0.0):.3f} "
        f"kl={row.get('ppo_approx_kl', 0.0):.4f} "
        f"clip={row.get('ppo_clip_fraction', 0.0):.2f} "
        f"ev={row.get('ppo_explained_variance', 0.0):.2f} "
        f"T={row.get('policy_temperature', 0.0):.3f} "
        f"lambda=({row['lambda_fair']:.3f},{row['lambda_sla']:.3f},{row.get('lambda_lag', 0.0):.3f}) "
        f"scale={row.get('scale', 1.0):.3f}]"
    )


def composite_score(result: dict, args) -> float:
    # Linux-inspired model selection:
    # first stay close to the SJF efficiency envelope, then minimize CFS-style
    # virtual service lag.  This prevents a "fair" checkpoint from winning by
    # simply letting latency explode.
    metric = metric_value(result, args.selection_metric)
    return float(metric + efficiency_gate_penalty(result, args))


def selection_diagnostics(result: dict, args) -> dict:
    slowdown_limit = float(args.ref_slowdown) * float(args.slowdown_gate_multiplier)
    wait_limit = float(args.ref_wait) * float(args.wait_gate_multiplier)
    p95_wait_limit = float(args.ref_p95) * float(args.p95_gate_multiplier)
    primary = metric_value(result, args.selection_metric)
    gate_penalty = efficiency_gate_penalty(result, args)
    gate_slowdown = gate_violation(float(result.get("avg_slowdown", 0.0)), slowdown_limit)
    gate_wait = gate_violation(float(result.get("avg_wait", 0.0)), wait_limit)
    gate_p95 = gate_violation(float(result.get("p95_wait", 0.0)), p95_wait_limit)
    gate_fairness = (
        max(0.0, (float(args.fairness_floor) - float(result.get("jain_inverse_slowdown", 1.0))) / float(args.fairness_floor))
        if args.fairness_floor > 0
        else 0.0
    )
    gate_violation_total = gate_slowdown + gate_wait + gate_p95 + gate_fairness + float(result.get("jobs_unfinished", 0.0))
    return {
        "selection_metric": args.selection_metric,
        "selection_primary_value": primary,
        "selection_gate_penalty": gate_penalty,
        "composite_score": float(primary + gate_penalty),
        "gate_slowdown_limit": slowdown_limit,
        "gate_wait_limit": wait_limit,
        "gate_p95_wait_limit": p95_wait_limit,
        "gate_fairness_floor": float(args.fairness_floor),
        "gate_slowdown_ratio": ratio_or_zero(float(result.get("avg_slowdown", 0.0)), slowdown_limit),
        "gate_wait_ratio": ratio_or_zero(float(result.get("avg_wait", 0.0)), wait_limit),
        "gate_p95_wait_ratio": ratio_or_zero(float(result.get("p95_wait", 0.0)), p95_wait_limit),
        "gate_fairness_ratio": ratio_or_zero(float(result.get("jain_inverse_slowdown", 0.0)), float(args.fairness_floor)),
        "gate_slowdown_violation": gate_slowdown,
        "gate_wait_violation": gate_wait,
        "gate_p95_wait_violation": gate_p95,
        "gate_fairness_violation": gate_fairness,
        "gate_violation_total": gate_violation_total,
        "gate_all_clear": int(gate_violation_total <= 0.0),
    }


def metric_value(result: dict, metric_name: str) -> float:
    fallback = result.get("normalized_service_lag_auc", float("inf"))
    return float(result.get(metric_name, fallback))


def efficiency_gate_penalty(result: dict, args) -> float:
    penalty = 0.0
    penalty += float(args.unfinished_penalty) * float(result.get("jobs_unfinished", 0.0))
    penalty += gate_violation(
        float(result.get("avg_slowdown", 0.0)),
        float(args.ref_slowdown) * float(args.slowdown_gate_multiplier),
    )
    penalty += gate_violation(
        float(result.get("avg_wait", 0.0)),
        float(args.ref_wait) * float(args.wait_gate_multiplier),
    )
    penalty += gate_violation(
        float(result.get("p95_wait", 0.0)),
        float(args.ref_p95) * float(args.p95_gate_multiplier),
    )
    fairness = float(result.get("jain_inverse_slowdown", 1.0))
    if args.fairness_floor > 0:
        penalty += max(0.0, (float(args.fairness_floor) - fairness) / float(args.fairness_floor))
    return float(args.gate_penalty_scale) * penalty


def gate_violation(value: float, limit: float) -> float:
    if limit <= 0.0:
        return 0.0
    return max(0.0, value / limit - 1.0)


def ratio_or_zero(value: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(value) / float(denominator)


def checkpoint_config(args) -> dict:
    keys = [
        "dataset",
        "job_offset",
        "max_jobs",
        "eval_job_offset",
        "eval_max_jobs",
        "max_nodes",
        "top_k",
        "candidate_d",
        "candidate_sampling_mode",
        "candidate_pool_k",
        "arrival_scale",
        "placement",
        "policy_model",
        "gru_hidden_dim",
        "lr",
        "critic_lr",
        "value_loss_weight",
        "debt_weight",
        "lag_target",
        "entropy_weight",
        "ppo_epochs",
        "ppo_clip_eps",
        "gamma",
        "gae_lambda",
        "max_grad_norm",
        "reward_lag_level_weight",
        "reward_lag_growth_weight",
        "reward_lag_reduction_weight",
        "reward_waiting_queue_weight",
        "reward_gate_slowdown_weight",
        "reward_gate_wait_weight",
        "reward_gate_p95_weight",
        "reward_gate_fairness_weight",
        "reward_unfinished_weight",
        "reward_pass_penalty_weight",
        "episodes",
        "batch_size",
        "num_workers",
        "seed",
        "early_stop_patience",
        "ref_slowdown",
        "ref_wait",
        "ref_p95",
        "ref_fairness",
        "selection_metric",
        "slowdown_gate_multiplier",
        "wait_gate_multiplier",
        "p95_gate_multiplier",
        "fairness_floor",
        "unfinished_penalty",
        "gate_penalty_scale",
    ]
    return {key: getattr(args, key) for key in keys}


def per_update_checkpoint_path(args, update: int) -> Path:
    return args.output_dir / "checkpoints" / f"fair_rc_rl_gru_update_{int(update):03d}.pt"


def save_gru_checkpoint(path: Path, policy, args, update: int, eval_result: dict | None, score, checkpoint_kind: str, critic=None) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - train_gru imports torch first
        raise RuntimeError("GRU checkpoint saving requires PyTorch.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "state_dict": policy.state_dict(),
            "critic_state_dict": None if critic is None else critic.state_dict(),
            "checkpoint_kind": checkpoint_kind,
            "update": int(update),
            "best_score": None if score is None else float(score),
            "eval_result": dict(eval_result) if eval_result is not None else None,
            "config": checkpoint_config(args),
            "job_feature_dim": len(RL_FEATURE_NAMES),
            "state_feature_dim": len(state_feature_names(args.policy_model)),
            "queue_feature_dim": len(RL_FEATURE_NAMES),
            "hidden_dim": args.gru_hidden_dim,
            "job_feature_names": RL_FEATURE_NAMES,
            "state_feature_names": state_feature_names(args.policy_model),
            "queue_feature_names": RL_FEATURE_NAMES,
        },
        path,
    )


def write_best_metrics(path: Path, args, update: int, score: float, eval_result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_update": int(update),
        "best_score": float(score),
        "policy_model": args.policy_model,
        "placement": args.placement,
        "dataset": args.dataset,
        "job_offset": args.job_offset,
        "max_jobs": args.max_jobs,
        "eval_job_offset": args.eval_job_offset,
        "eval_max_jobs": args.eval_max_jobs,
        "max_nodes": args.max_nodes,
        "top_k": args.top_k,
        "candidate_d": args.candidate_d,
        "arrival_scale": args.arrival_scale,
        "selection_metric": args.selection_metric,
        "selection_primary_value": metric_value(eval_result, args.selection_metric),
        "selection_gate_penalty": efficiency_gate_penalty(eval_result, args),
        "references": {
            "ref_slowdown": args.ref_slowdown,
            "ref_wait": args.ref_wait,
            "ref_p95": args.ref_p95,
            "ref_fairness": args.ref_fairness,
            "slowdown_gate_multiplier": args.slowdown_gate_multiplier,
            "wait_gate_multiplier": args.wait_gate_multiplier,
            "p95_gate_multiplier": args.p95_gate_multiplier,
            "fairness_floor": args.fairness_floor,
        },
        "metrics": to_jsonable(dict(eval_result)),
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


if __name__ == "__main__":
    main()
