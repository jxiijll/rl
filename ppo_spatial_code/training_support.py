from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpu_sched_experiment.simulator import RL_FEATURE_NAMES

if __package__:
    from .modele import (
        STATE_FEATURE_NAMES,
        FairRCConfig,
        run_gru_episode_trace,
    )
    from .policy import GRUFairPolicy, GRUValuePolicy
else:  # pragma: no cover - supports direct script execution
    from modele import (
        STATE_FEATURE_NAMES,
        FairRCConfig,
        run_gru_episode_trace,
    )
    from policy import GRUFairPolicy, GRUValuePolicy


@dataclass
class ReplayOutput:
    log_probs: object | None
    entropies: object | None
    values: object | None
    old_log_probs: object | None
    reward_components: list[dict[str, float]]
    diagnostics: list[dict[str, float]]

    @property
    def log_prob_sum(self):
        return None if self.log_probs is None else self.log_probs.sum()

    @property
    def entropy_sum(self):
        return None if self.entropies is None else self.entropies.sum()


def state_feature_names(policy_model: str) -> list[str]:
    return STATE_FEATURE_NAMES


def build_actor(policy_model: str, hidden_dim: int):
    return GRUFairPolicy(
        job_feature_dim=len(RL_FEATURE_NAMES),
        state_feature_dim=len(STATE_FEATURE_NAMES),
        queue_feature_dim=len(RL_FEATURE_NAMES),
        hidden_dim=hidden_dim,
    )


def build_critic(policy_model: str, hidden_dim: int):
    return GRUValuePolicy(
        state_feature_dim=len(state_feature_names(policy_model)),
        queue_feature_dim=len(RL_FEATURE_NAMES),
        hidden_dim=hidden_dim,
    )


def actor_kwargs(policy_model: str, hidden_dim: int) -> dict:
    return {
        "job_feature_dim": len(RL_FEATURE_NAMES),
        "state_feature_dim": len(STATE_FEATURE_NAMES),
        "queue_feature_dim": len(RL_FEATURE_NAMES),
        "hidden_dim": hidden_dim,
    }


def run_gru_trace_rollout_task(task: dict):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional torch install
        raise RuntimeError("GRU policy requires PyTorch.") from exc

    torch.set_num_threads(1)
    torch.manual_seed(task["seed"])
    policy = build_actor(task["policy_model"], task["hidden_dim"])
    policy_state_dict = {key: torch.as_tensor(value) for key, value in task["policy_state_dict"].items()}
    policy.load_state_dict(policy_state_dict)
    policy.eval()
    with torch.no_grad():
        return run_gru_episode_trace(
            jobs=task["jobs"],
            base_nodes=task["base_nodes"],
            policy=policy,
            seed=task["seed"],
            config=task["config"],
            sample=True,
        )


def replay_trace(actor, critic, trace: list[dict], config: FairRCConfig, policy_model: str) -> ReplayOutput:
    if not trace:
        return ReplayOutput(None, None, None, None, [], [])

    import torch

    return _replay_gru_trace(actor, critic, trace, config)


def _replay_gru_trace(actor, critic, trace: list[dict], config: FairRCConfig) -> ReplayOutput:
    import torch

    device = next(actor.parameters()).device
    actor_hidden = actor.initial_hidden(device=device)
    critic_hidden = critic.initial_hidden(device=device)
    log_probs = []
    entropies = []
    values = []
    for step in trace:
        job_tensor = torch.as_tensor(step["job_features"], dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(step["state_features"], dtype=torch.float32, device=device)
        queue_tensor = torch.as_tensor(step.get("queue_features", step["job_features"]), dtype=torch.float32, device=device)
        logits, actor_hidden = actor(job_tensor, state_tensor, actor_hidden, queue_tensor)
        probs = torch.softmax(logits / config.policy_temperature, dim=0)
        dist = torch.distributions.Categorical(probs=probs)
        action = torch.as_tensor(int(step["action"]), dtype=torch.long, device=device)
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        value, critic_hidden = critic(state_tensor, critic_hidden, queue_tensor)
        values.append(value)
    old_log_probs = [
        float(step["old_log_prob"])
        for step in trace
        if "old_log_prob" in step
    ]
    return ReplayOutput(
        log_probs=torch.stack(log_probs),
        entropies=torch.stack(entropies),
        values=torch.stack(values),
        old_log_probs=(
            torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
            if len(old_log_probs) == len(trace)
            else None
        ),
        reward_components=[dict(step.get("reward_components", {})) for step in trace],
        diagnostics=[
            {
                "candidate_size": float(step.get("candidate_size", 0.0)),
                "visible_pool_size": float(step.get("visible_pool_size", 0.0)),
                "feasible_jobs_count": float(step.get("feasible_jobs_count", 0.0)),
                "selected_risk_rank": float(step.get("selected_risk_rank", 0.0)),
                "selected_highest_risk": float(step.get("selected_highest_risk", 0.0)),
                "selected_pass": float(step.get("selected_pass", 0.0)),
            }
            for step in trace
        ],
    )


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)) if len(arr) else 0.0


def grad_l2_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(grad.pow(2).sum().cpu().item())
    return float(total ** 0.5)
