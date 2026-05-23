from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpu_sched_experiment.simulator import (
    ClusterSimulator,
    OnlineServiceLagTracker,
    PLACEMENT_POLICIES,
    RL_FEATURE_NAMES,
    summarize_results,
)

if __package__:
    from .utils import clone_nodes, fairness_cost, sla_cost, lag_cost
else:  # pragma: no cover - supports direct script execution
    from utils import clone_nodes, fairness_cost, sla_cost, lag_cost


STATE_FEATURE_NAMES = [
    # Original summary features.
    "log_queue_length",
    "log_topk_avg_wait",
    "log_topk_max_wait",
    "log_topk_avg_slowdown",
    "log_topk_max_slowdown",
    "high_slowdown_ratio_topk",
    "log_topk_avg_runtime",
    "short_job_ratio_topk",
    "multi_gpu_ratio_topk",
    "gpu_utilization",
    "cpu_utilization",
    "fragmentation",
    "free_gpu_total_ratio",
    "max_free_gpu_per_node_ratio",
    "mean_free_gpu_per_node_ratio",
    "feasible_count_topk_ratio",
    "blocked_count_topk_ratio",
    "max_debt_candidates",
    "avg_debt_candidates",
    "recent_avg_debt_gap",
    "candidate_avg_cfs_lag",
    "candidate_max_cfs_lag",
    # Spatial flattening: top nodes sorted by currently free GPU.
    "top1_free_gpu_ratio",
    "top2_free_gpu_ratio",
    "top3_free_gpu_ratio",
    "top4_free_gpu_ratio",
    "top5_free_gpu_ratio",
    "top6_free_gpu_ratio",
    "top7_free_gpu_ratio",
    "top8_free_gpu_ratio",
    # Trend features relative to the previous policy decision.
    "delta_gpu_utilization",
    "delta_fragmentation",
    "delta_queue_length",
]

@dataclass(frozen=True)
class FairRCConfig:
    placement: str = "best_fit"
    top_k: int = 64
    candidate_d: int = 16
    candidate_sampling_mode: str = "uniform_pool"
    candidate_pool_k: int = 64
    sampling_temperature: float = 1.0
    policy_temperature: float = 1.0
    sla_threshold: float = 5000.0
    lag_target: float = 100.0


class FairRandomizedCandidateSimulator(ClusterSimulator):
    """Non-heuristic bounded candidate exposure plus a direct GRU policy."""

    def run_fair_rc_gru_policy(
        self,
        policy,
        config: FairRCConfig,
        sample: bool = False,
        record_trace: bool = False,
    ):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional torch install
            raise RuntimeError("GRU policy requires PyTorch. Install torch to use --policy-model gru.") from exc

        if config.placement not in PLACEMENT_POLICIES:
            raise ValueError(f"Unknown placement policy: {config.placement}")
        if config.candidate_d <= 0:
            raise ValueError("candidate_d must be positive")
        if config.sampling_temperature <= 0 or config.policy_temperature <= 0:
            raise ValueError("temperatures must be positive")

        time = 0
        next_job = 0
        waiting = []
        running = []
        finished = []
        unfinished_on_exit = 0
        last_time = 0
        gpu_time = 0.0
        cpu_time = 0.0
        frag_samples = []
        run_counter = 0
        decisions = 0
        candidate_sizes = []
        visible_pool_sizes = []
        feasible_counts = []
        selected_highest_risk = []
        selected_risk_ranks = []
        pass_actions = 0
        log_probs = []
        trace = []
        debt_gaps = []
        recent_debt_gaps = []
        device = next(policy.parameters()).device
        hidden = policy.initial_hidden(device=device)
        service_lag = OnlineServiceLagTracker(self.total_gpu_milli)
        macro_trace_start = 0
        macro_potential_before = 0.0
        prev_gpu_util = 0.0
        prev_frag = 0.0
        prev_queue_len = 0.0

        while next_job < len(self.jobs) or waiting or running:
            if not waiting and not running and next_job < len(self.jobs):
                time = max(time, self.jobs[next_job].submit_time)

            elapsed = max(0, time - last_time)
            service_lag.update_interval(waiting, running, elapsed)
            gpu_time += self._gpu_used_milli() * elapsed
            cpu_time += self._cpu_used_milli() * elapsed
            last_time = time

            time = self._release_finished(time, running, finished)
            while next_job < len(self.jobs) and self.jobs[next_job].submit_time <= time:
                waiting.append(self.jobs[next_job])
                next_job += 1

            frag_samples.append(self._fragmentation())

            scheduled_any = True
            macro_trace_start = len(trace)
            macro_potential_before = self._cfs_lag_potential(service_lag, waiting, running)
            while waiting and scheduled_any:
                scheduled_any = False
                visible_jobs = self._visible_candidate_pool(waiting, config)
                top_jobs = visible_jobs
                feasible = self._feasible_feature_matrix(visible_jobs, time, config.placement, service_lag)
                if not feasible:
                    break

                sampled = self._sample_policy_candidates(feasible, config)
                sampled = self._with_pass_candidate(sampled)
                risk_rank_by_job = self._risk_rank_by_job_id(sampled, time)
                candidate_sizes.append(len(sampled))
                visible_pool_sizes.append(len(visible_jobs))
                feasible_counts.append(len(feasible))
                jobs, features = zip(*sampled)
                feature_matrix = np.vstack(features).astype(np.float32)
                state_features, (prev_gpu_util, prev_frag, prev_queue_len) = self._state_summary_features(
                    waiting=waiting,
                    top_jobs=top_jobs,
                    feasible=feasible,
                    candidates=sampled,
                    now=time,
                    recent_debt_gaps=recent_debt_gaps,
                    prev_states=(prev_gpu_util, prev_frag, prev_queue_len),
                    service_lag=service_lag,
                )
                queue_features = self._queue_set_features(
                    top_jobs=top_jobs,
                    now=time,
                    placement=config.placement,
                    service_lag=service_lag,
                )

                job_tensor = torch.as_tensor(feature_matrix, dtype=torch.float32, device=device)
                state_tensor = torch.as_tensor(state_features, dtype=torch.float32, device=device)
                queue_tensor = torch.as_tensor(queue_features, dtype=torch.float32, device=device)
                logits, hidden = policy(job_tensor, state_tensor, hidden, queue_tensor)

                if sample:
                    probs = torch.softmax(logits / config.policy_temperature, dim=0)
                    dist = torch.distributions.Categorical(probs=probs)
                    action = dist.sample()
                    chosen_idx = int(action.item())
                    log_probs.append(dist.log_prob(action))
                    action_log_prob = float(dist.log_prob(action).detach().cpu().item())
                    action_entropy = float(dist.entropy().detach().cpu().item())
                else:
                    chosen_idx = int(torch.argmax(logits).item())
                    probs = torch.softmax(logits / config.policy_temperature, dim=0)
                    dist = torch.distributions.Categorical(probs=probs)
                    action_log_prob = float(dist.log_prob(torch.as_tensor(chosen_idx, dtype=torch.long, device=device)).detach().cpu().item())
                    action_entropy = float(dist.entropy().detach().cpu().item())
                chose_pass = jobs[chosen_idx] is None
                debt_gap = self._candidate_debt_gap(jobs, chosen_idx, time)
                debt_gaps.append(debt_gap)
                recent_debt_gaps.append(debt_gap)
                if len(recent_debt_gaps) > 20:
                    recent_debt_gaps.pop(0)
                chosen_job_id = "__pass__" if chose_pass else str(jobs[chosen_idx].job_id)
                chosen_rank = int(risk_rank_by_job.get(chosen_job_id, 0 if chose_pass else len(sampled)))
                if not chose_pass:
                    selected_risk_ranks.append(chosen_rank)
                    selected_highest_risk.append(1 if chosen_rank == 1 else 0)

                job = jobs[chosen_idx]
                if record_trace:
                    trace.append(
                        {
                            "job_features": feature_matrix,
                            "state_features": state_features,
                            "queue_features": queue_features,
                            "action": chosen_idx,
                            "old_log_prob": action_log_prob,
                            "old_entropy": action_entropy,
                            "candidate_size": len(sampled),
                            "visible_pool_size": len(visible_jobs),
                            "feasible_jobs_count": len(feasible),
                            "selected_risk_rank": chosen_rank,
                            "selected_highest_risk": 1 if chosen_rank == 1 else 0,
                            "selected_pass": 1 if chose_pass else 0,
                            "reward_components": {
                                "cost_pass_action": 1.0 if chose_pass else 0.0,
                            },
                        }
                    )
                if chose_pass:
                    pass_actions += 1
                    decisions += 1
                    scheduled_any = False
                    break

                allocs = self._try_allocate(job, config.placement)
                if allocs is None:
                    break
                for alloc in allocs:
                    setattr(alloc, "start_time", time)
                waiting.remove(job)
                run_counter += 1
                decisions += 1
                heapq.heappush(running, (time + job.runtime, f"{run_counter}:{job.job_id}", allocs))
                scheduled_any = True

            next_time = None
            should_continue = False
            should_break = False
            if running:
                next_event = running[0][0]
                if next_job < len(self.jobs):
                    next_event = min(next_event, self.jobs[next_job].submit_time)
                if next_event == time and running and running[0][0] == time:
                    should_continue = True
                else:
                    next_time = next_event
            elif next_job < len(self.jobs):
                next_time = max(time + 1, self.jobs[next_job].submit_time)
            elif waiting:
                unfinished_on_exit = len(waiting)
                should_break = True

            if record_trace and macro_trace_start < len(trace):
                interval = max(0, (next_time if next_time is not None else time) - time)
                components = self._macro_reward_components(
                    service_lag=service_lag,
                    waiting=waiting,
                    running=running,
                    interval=interval,
                    potential_before=macro_potential_before,
                )
                macro_steps = max(1, len(trace) - macro_trace_start)
                per_decision_components = {key: value / macro_steps for key, value in components.items()}
                for item in trace[macro_trace_start:]:
                    reward_components = dict(item.get("reward_components", {}))
                    reward_components.update(per_decision_components)
                    reward_components["cost_pass_action"] = float(item.get("selected_pass", 0.0))
                    item["reward_components"] = {
                        **reward_components,
                    }

            if should_continue:
                continue
            if should_break:
                break
            if next_time is not None:
                time = next_time

        makespan = max((job.finish_time for job in finished), default=0)
        self._ensure_service_lag_compat(service_lag)
        result = summarize_results(
            finished=finished,
            loaded_jobs=self.jobs,
            makespan=makespan,
            gpu_time=gpu_time,
            cpu_time=cpu_time,
            total_gpu_milli=self.total_gpu_milli,
            total_cpu_milli=self.total_cpu_milli,
            fragmentation_samples=frag_samples,
            policy="fair_rc_rl_gru",
            placement=config.placement,
            online_service_metrics=service_lag.metrics(),
        )
        result["jobs_loaded"] = len(self.jobs)
        result["jobs_unfinished"] = unfinished_on_exit
        result["jobs_infeasible"] = self.infeasible_jobs
        result["candidate_d"] = config.candidate_d
        result["candidate_sampling_mode"] = config.candidate_sampling_mode
        result["candidate_pool_k"] = config.candidate_pool_k
        result.update(describe_distribution("actual_candidate_size", candidate_sizes))
        result.update(describe_distribution("feasible_jobs_count", feasible_counts))
        result.update(describe_distribution("visible_pool_size", visible_pool_sizes))
        result["avg_candidate_size"] = result["actual_candidate_size_mean"]
        result["p50_candidate_size"] = result["actual_candidate_size_p50"]
        result["p90_candidate_size"] = result["actual_candidate_size_p90"]
        result["single_candidate_ratio"] = float(np.mean(np.asarray(candidate_sizes, dtype=float) <= 1.0)) if candidate_sizes else 0.0
        result["feasible_count_topk_avg"] = result["feasible_jobs_count_mean"]
        result["feasible_count_topk_p10"] = result["feasible_jobs_count_p10"]
        result["feasible_count_topk_p50"] = result["feasible_jobs_count_p50"]
        result["feasible_count_topk_p90"] = result["feasible_jobs_count_p90"]
        result["highest_risk_selected_ratio"] = float(np.mean(selected_highest_risk)) if selected_highest_risk else 0.0
        result["selected_risk_rank_mean"] = float(np.mean(selected_risk_ranks)) if selected_risk_ranks else 0.0
        result["pass_actions"] = pass_actions
        result["pass_action_ratio"] = float(pass_actions / max(decisions, 1))
        result["avg_debt_gap"] = float(np.mean(debt_gaps)) if debt_gaps else 0.0
        result["max_debt_gap"] = float(np.max(debt_gaps)) if debt_gaps else 0.0
        result["debt_gap_samples"] = len(debt_gaps)
        result["fairness_cost"] = fairness_cost(result)
        result["sla_cost"] = sla_cost(result, config.sla_threshold)
        result["lag_cost"] = lag_cost(result, config.lag_target)
        if record_trace:
            return result, trace, decisions
        return result, log_probs, decisions

    def _visible_candidate_pool(self, waiting, config: FairRCConfig):
        mode = str(config.candidate_sampling_mode)
        if mode == "uniform":
            return list(waiting)
        if mode == "uniform_pool":
            pool_k = int(config.candidate_pool_k)
            return list(waiting[:pool_k]) if pool_k > 0 else list(waiting)
        raise ValueError(f"Unknown non-heuristic candidate sampling mode: {mode}")

    def _sample_policy_candidates(self, feasible, config: FairRCConfig):
        sample_size = min(config.candidate_d, len(feasible))
        if sample_size <= 0:
            return []
        indices = self.rng.sample(range(len(feasible)), sample_size) if sample_size < len(feasible) else list(range(len(feasible)))
        return [feasible[idx] for idx in indices]

    def _risk_rank_by_job_id(self, sampled, now: int) -> dict[str, int]:
        ranked = sorted(
            [item for item in sampled if item[0] is not None],
            key=lambda item: self._candidate_log_slowdown(item[0], now),
            reverse=True,
        )
        return {str(job.job_id): rank + 1 for rank, (job, _features) in enumerate(ranked)}

    def _with_pass_candidate(self, sampled):
        return [*sampled, (None, self._pass_features())]

    def _pass_features(self) -> np.ndarray:
        features = np.zeros(len(RL_FEATURE_NAMES), dtype=float)
        features[-1] = 1.0
        return features

    def _candidate_risks(self, feasible, now: int) -> np.ndarray:
        log_slowdowns = np.array(
            [self._candidate_log_slowdown(item[0], now) for item in feasible],
            dtype=float,
        )
        if len(log_slowdowns) == 0:
            return log_slowdowns

        anchor = float(np.median(log_slowdowns))
        q75, q25 = np.percentile(log_slowdowns, [75, 25])
        scale = max(float(q75 - q25), 1e-6)
        return np.array(
            [self._candidate_risk(item[0], now, anchor, scale) for item in feasible],
            dtype=float,
        )

    def _candidate_risk(self, job, now: int, anchor: float, scale: float) -> float:
        return (self._candidate_log_slowdown(job, now) - anchor) / scale

    def _candidate_log_slowdown(self, job, now: int) -> float:
        wait = max(0.0, float(now - job.submit_time))
        slowdown = (wait + job.runtime) / max(float(job.runtime), 1.0)
        return float(np.log1p(slowdown))

    def _candidate_debt_gap(self, jobs, chosen_idx: int, now: int) -> float:
        debts = np.array([self._candidate_log_slowdown(job, now) for job in jobs if job is not None], dtype=float)
        if len(debts) == 0:
            return 0.0
        chosen_job = jobs[chosen_idx]
        if chosen_job is None:
            return max(0.0, float(np.max(debts)))
        chosen_debt = self._candidate_log_slowdown(chosen_job, now)
        return max(0.0, float(np.max(debts) - chosen_debt))

    def _macro_reward_components(
        self,
        service_lag: OnlineServiceLagTracker,
        waiting,
        running,
        interval: int | float,
        potential_before: float,
    ) -> dict[str, float]:
        projected_lag = self._clone_service_lag(service_lag)
        projected_lag.update_interval(waiting, running, interval)
        potential_after = self._cfs_lag_potential(projected_lag, waiting, running)
        
        # Convert potentials to log1p scale for extreme value suppression (e.g. 10^6 -> 13.8)
        log_after = float(np.log1p(max(0.0, potential_after)))
        log_before = float(np.log1p(max(0.0, potential_before)))
        log_diff = log_after - log_before
        
        return {
            "cost_cfs_log_level": log_after,
            "cost_cfs_log_growth": float(max(0.0, log_diff)),
            "gain_cfs_log_reduction": float(max(0.0, -log_diff)),
            "cost_waiting_queue_size_log": float(np.log1p(len(waiting)) / 12.0),
        }

    def _clone_service_lag(self, service_lag: OnlineServiceLagTracker) -> OnlineServiceLagTracker:
        clone = OnlineServiceLagTracker(service_lag.total_gpu_milli)
        clone.expected_by_group = dict(service_lag.expected_by_group)
        clone.actual_by_group = dict(service_lag.actual_by_group)
        clone.total_capacity_time = float(service_lag.total_capacity_time)
        clone.sample_count = int(service_lag.sample_count)
        clone.max_positive_lag = float(service_lag.max_positive_lag)
        clone.positive_lag_sum = float(service_lag.positive_lag_sum)
        clone.lag_l1_auc = float(service_lag.lag_l1_auc)
        clone.lag_l2_auc = float(service_lag.lag_l2_auc)
        clone.positive_lag_auc = float(service_lag.positive_lag_auc)
        clone.variance_auc = float(service_lag.variance_auc)
        clone.variance_max = float(service_lag.variance_max)
        self._ensure_service_lag_compat(clone)
        return clone

    def _ensure_service_lag_compat(self, service_lag: OnlineServiceLagTracker) -> None:
        if not hasattr(service_lag, "cfs_positive_lag_sum"):
            service_lag.cfs_positive_lag_sum = float(service_lag.positive_lag_sum)

    def _cfs_lag_potential(self, service_lag: OnlineServiceLagTracker, waiting, running) -> float:
        groups = {int(job.num_gpu) for job in waiting}
        for _finish_time, _key, allocs in running:
            if not allocs:
                continue
            job = getattr(allocs[0], "job", None)
            if job is not None:
                groups.add(int(job.num_gpu))
        if not groups:
            return 0.0
        return float(max(service_lag.normalized_positive_lag(group) for group in groups))

    def _state_summary_features(
        self,
        waiting,
        top_jobs,
        feasible,
        candidates,
        now: int,
        recent_debt_gaps,
        prev_states: tuple[float, float, float],
        service_lag: OnlineServiceLagTracker | None = None,
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        top_jobs = list(top_jobs)
        feasible_jobs = [item[0] for item in feasible]
        candidate_jobs = [item[0] for item in candidates if item[0] is not None]
        waits = np.array([max(0.0, float(now - job.submit_time)) for job in top_jobs], dtype=float)
        slowdowns = np.array(
            [(wait + job.runtime) / max(float(job.runtime), 1.0) for wait, job in zip(waits, top_jobs)],
            dtype=float,
        )
        runtimes = np.array([float(job.runtime) for job in top_jobs], dtype=float)
        gpu_counts = np.array([float(job.num_gpu) for job in top_jobs], dtype=float)
        debts = np.array([self._candidate_log_slowdown(job, now) for job in candidate_jobs], dtype=float)
        cfs_lags = np.array(
            [
                service_lag.normalized_positive_lag(int(job.num_gpu)) if service_lag is not None else 0.0
                for job in candidate_jobs
            ],
            dtype=float,
        )
        node_free = np.array([float(node.gpu_free_milli) for node in self.nodes if node.gpu_total_milli > 0], dtype=float)
        node_total = np.array([float(node.gpu_total_milli) for node in self.nodes if node.gpu_total_milli > 0], dtype=float)

        top_count = max(len(top_jobs), 1)
        top_k = max(len(top_jobs), 1)
        free_gpu_total = float(np.sum(node_free)) if len(node_free) else 0.0
        max_node_total = float(np.max(node_total)) if len(node_total) else 1.0
        mean_node_total = float(np.mean(node_total)) if len(node_total) else 1.0
        avg_runtime = float(np.mean(runtimes)) if len(runtimes) else 0.0
        runtime_median = float(np.median(runtimes)) if len(runtimes) else 0.0

        sorted_free = np.sort(node_free)[::-1] if len(node_free) else np.array([], dtype=float)
        top8_free_ratios = np.zeros(8, dtype=np.float32)
        for idx in range(min(8, len(sorted_free))):
            top8_free_ratios[idx] = sorted_free[idx] / max_node_total if max_node_total > 0 else 0.0

        current_gpu_util = self._gpu_used_milli() / self.total_gpu_milli if self.total_gpu_milli else 0.0
        current_frag = self._fragmentation()
        current_queue_len = float(len(waiting))
        prev_gpu_util, prev_frag, prev_queue_len = prev_states
        delta_gpu_util = current_gpu_util - prev_gpu_util
        delta_frag = current_frag - prev_frag
        delta_queue = (np.log1p(current_queue_len) - np.log1p(prev_queue_len)) / 12.0
        new_prev_states = (current_gpu_util, current_frag, current_queue_len)

        features = np.array(
            [
                np.log1p(len(waiting)) / 12.0,
                np.log1p(float(np.mean(waits)) if len(waits) else 0.0) / 12.0,
                np.log1p(float(np.max(waits)) if len(waits) else 0.0) / 12.0,
                np.log1p(float(np.mean(slowdowns)) if len(slowdowns) else 0.0) / 8.0,
                np.log1p(float(np.max(slowdowns)) if len(slowdowns) else 0.0) / 8.0,
                float(np.mean(slowdowns > 10.0)) if len(slowdowns) else 0.0,
                np.log1p(avg_runtime) / 12.0,
                float(np.mean(runtimes <= runtime_median)) if len(runtimes) else 0.0,
                float(np.mean(gpu_counts > 1.0)) if len(gpu_counts) else 0.0,
                self._gpu_used_milli() / self.total_gpu_milli if self.total_gpu_milli else 0.0,
                self._cpu_used_milli() / self.total_cpu_milli if self.total_cpu_milli else 0.0,
                self._fragmentation(),
                free_gpu_total / self.total_gpu_milli if self.total_gpu_milli else 0.0,
                (float(np.max(node_free)) if len(node_free) else 0.0) / max_node_total,
                (float(np.mean(node_free)) if len(node_free) else 0.0) / mean_node_total,
                len(feasible_jobs) / top_k,
                max(0, top_count - len(feasible_jobs)) / top_k,
                float(np.max(debts)) if len(debts) else 0.0,
                float(np.mean(debts)) if len(debts) else 0.0,
                float(np.mean(recent_debt_gaps)) if recent_debt_gaps else 0.0,
                float(np.mean(cfs_lags)) if len(cfs_lags) else 0.0,
                float(np.max(cfs_lags)) if len(cfs_lags) else 0.0,
                *top8_free_ratios,
                delta_gpu_util,
                delta_frag,
                delta_queue,
            ],
            dtype=np.float32,
        )
        return features, new_prev_states

    def _queue_set_features(
        self,
        top_jobs,
        now: int,
        placement: str,
        service_lag: OnlineServiceLagTracker | None = None,
    ) -> np.ndarray:
        """Return per-job features for the visible waiting queue.

        The policy sees only already-arrived jobs from the bounded visible pool.
        These are the same feature semantics used by candidate scoring, but kept
        as individual rows so attention can preserve queue structure.
        """
        rows = []
        for job in top_jobs:
            fit_score = self._fit_score(job, placement, "leftover")
            if not np.isfinite(float(fit_score[0])):
                fit_score = (self.total_gpu_milli, 1.0)
            rows.append(self._rl_features(job, now, fit_score, service_lag))
        if not rows:
            return np.zeros((0, len(RL_FEATURE_NAMES)), dtype=np.float32)
        return np.vstack(rows).astype(np.float32)

def run_gru_episode(
    jobs,
    base_nodes,
    policy,
    seed: int,
    config: FairRCConfig,
    sample: bool,
):
    nodes = clone_nodes(base_nodes)
    sim = FairRandomizedCandidateSimulator(jobs=jobs, nodes=nodes, seed=seed)
    return sim.run_fair_rc_gru_policy(policy=policy, config=config, sample=sample)


def run_gru_episode_trace(
    jobs,
    base_nodes,
    policy,
    seed: int,
    config: FairRCConfig,
    sample: bool,
):
    nodes = clone_nodes(base_nodes)
    sim = FairRandomizedCandidateSimulator(jobs=jobs, nodes=nodes, seed=seed)
    return sim.run_fair_rc_gru_policy(policy=policy, config=config, sample=sample, record_trace=True)


def describe_distribution(prefix: str, values) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
    }
