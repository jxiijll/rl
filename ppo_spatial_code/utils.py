from __future__ import annotations

import csv
import random
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpu_sched_experiment.simulator import NodeState


def clone_nodes(nodes):
    return [
        NodeState(
            node_id=node.node_id,
            cpu_milli=node.cpu_total,
            memory_mib=node.memory_total,
            gpu_count=len(node.gpu_free),
            model=node.gpu_model,
        )
        for node in nodes
    ]


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = np.sum(exp_values)
    if total <= 0 or not np.isfinite(total):
        return np.full_like(values, 1.0 / len(values), dtype=float)
    return exp_values / total


def sample_without_replacement(probs: np.ndarray, size: int, rng: random.Random) -> list[int]:
    probs = np.asarray(probs, dtype=float).copy()
    size = min(size, len(probs))
    chosen = []
    available = list(range(len(probs)))
    for _ in range(size):
        available_probs = np.array([probs[idx] for idx in available], dtype=float)
        total = float(np.sum(available_probs))
        if total <= 0 or not np.isfinite(total):
            pick_pos = rng.randrange(len(available))
        else:
            threshold = rng.random() * total
            cumulative = 0.0
            pick_pos = len(available) - 1
            for pos, value in enumerate(available_probs):
                cumulative += float(value)
                if cumulative >= threshold:
                    pick_pos = pos
                    break
        chosen.append(available.pop(pick_pos))
    return chosen


def fairness_cost(result: dict) -> float:
    return max(0.0, 1.0 - float(result.get("jain_inverse_slowdown", 1.0)))


def sla_cost(result: dict, sla_threshold: float) -> float:
    threshold = max(float(sla_threshold), 1.0)
    violation_ratio = max(0.0, float(result.get("p95_wait", 0.0)) - threshold) / threshold
    return float(np.log1p(violation_ratio))


def lag_cost(result: dict, lag_target: float) -> float:
    target = max(float(lag_target), 1e-9)
    lag = float(result.get("cfs_normalized_service_lag_auc", result.get("normalized_service_lag_auc", 0.0)))
    violation_ratio = max(0.0, lag - target) / target
    return float(np.log1p(violation_ratio))


def constrained_reward(
    result: dict,
    lambda_fair: float,
    lambda_sla: float,
    lambda_lag: float,
    wait_scale: float,
    slowdown_weight: float,
    wait_weight: float,
    util_weight: float,
    frag_weight: float,
    debt_weight: float = 0.0,
) -> tuple[float, float]:
    unfinished_penalty = 1000.0 * result.get("jobs_unfinished", 0)
    raw_reward = (
        -slowdown_weight * float(result["avg_slowdown"])
        -wait_weight * (float(result["avg_wait"]) / max(wait_scale, 1.0))
        +util_weight * float(result["gpu_utilization"])
        -frag_weight * float(result["avg_fragmentation"])
        -unfinished_penalty
    )
    debt_gap_penalty = float(debt_weight) * float(result.get("avg_debt_gap", 0.0))
    result["debt_gap_penalty"] = debt_gap_penalty
    lag_cost_val = float(result.get("lag_cost", 0.0))
    reward = raw_reward - lambda_fair * fairness_cost(result) - lambda_sla * result.get("sla_cost", 0.0) - lambda_lag * lag_cost_val - debt_gap_penalty
    return float(reward), float(raw_reward)


def lag_focused_reward(
    result: dict,
    lag_scale: float,
    slowdown_scale: float,
) -> tuple[float, float]:
    """Lag-primary reward with efficiency floor, fully self-calibrating.

    Both signals are normalized by running EMA scales computed from actual
    training data (passed in by the training loop).  This makes the reward
    dataset-agnostic: whether lag is 500 or 50 000, each normalised term
    stays O(1).

    Design choice (not a hyperparameter):
      - 80 % gradient from lag   → model pushes hard to reduce starvation
      - 20 % gradient from efficiency → prevents slowdown / wait blowup
    """
    lag = float(result.get("cfs_normalized_service_lag_auc", result.get("normalized_service_lag_auc", 0.0)))
    slowdown = float(result["avg_slowdown"])

    # Log1p scale suppression for extreme values
    lag_signal = -float(np.log1p(lag / max(lag_scale, 1e-9)))
    eff_signal = -float(np.log1p(slowdown / max(slowdown_scale, 1e-9)))

    raw_reward = lag_signal
    unfinished_penalty = 10.0 * float(np.log1p(result.get("jobs_unfinished", 0)))
    reward = 0.8 * lag_signal + 0.2 * eff_signal - unfinished_penalty
    return float(reward), float(raw_reward)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_weights(path: Path, feature_names: list[str], weights: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "weight"])
        for name, value in zip(feature_names, weights):
            writer.writerow([name, value])


def read_weights(path: Path, feature_names: list[str]) -> np.ndarray:
    weights_by_name = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            weights_by_name[row["feature"]] = float(row["weight"])
    return np.array([weights_by_name[name] for name in feature_names], dtype=float)


def format_result(result: dict) -> str:
    return (
        f"{result['policy']:>12s}/{result['placement']:<9s} "
        f"jobs={result['jobs_done']:4d}/{result['jobs_loaded']:<4d} "
        f"wait={result['avg_wait']:8.1f} "
        f"p95={result['p95_wait']:8.1f} "
        f"slow={result['avg_slowdown']:7.3f} "
        f"cfs_lag={result.get('cfs_normalized_service_lag_auc', result.get('normalized_service_lag_auc', 0.0)):8.2f} "
        f"util={result['gpu_utilization']:.3f} "
        f"fair={result['jain_inverse_slowdown']:.3f} "
        f"debt={result.get('avg_debt_gap', 0.0):.3f}"
    )
