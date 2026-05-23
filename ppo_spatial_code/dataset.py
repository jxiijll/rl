from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpu_sched_experiment.simulator import (
    load_2020_jobs,
    load_2023_jobs,
    load_2023_nodes,
    synthetic_2020_nodes,
)


DEFAULT_2023_PODS = Path("../cluster-trace-gpu-v2023/csv/openb_pod_list_gpuspec33.csv")
DEFAULT_2023_NODES = Path("../cluster-trace-gpu-v2023/csv/openb_node_list_gpu_node.csv")
DEFAULT_2020_JOBS = Path("../cluster-trace-gpu-v2020/simulator/traces/pai/pai_job_duration_estimate_100K.csv")


def load_inputs(args, repo_root: Path):
    """Load the same traces and node sets used by gpu_sched_experiment."""
    job_offset = getattr(args, "job_offset", 0)
    max_jobs = getattr(args, "max_jobs", None)
    if job_offset < 0:
        raise SystemExit("--job-offset must be non-negative.")

    if args.dataset == "2023":
        pod_csv = resolve_path(args.pod_csv, repo_root / "gpu_sched_experiment")
        node_csv = resolve_path(args.node_csv, repo_root / "gpu_sched_experiment")
        jobs = load_2023_jobs(pod_csv, max_jobs=None, gpu_only=not args.include_cpu_only)
        nodes = load_2023_nodes(node_csv, max_nodes=args.max_nodes)
    else:
        job_csv = resolve_path(args.job_csv_2020, repo_root / "gpu_sched_experiment")
        jobs = load_2020_jobs(job_csv, max_jobs=None, gpu_only=not args.include_cpu_only)
        nodes = synthetic_2020_nodes(num_nodes=args.max_nodes)

    jobs = select_job_window(jobs, job_offset=job_offset, max_jobs=max_jobs)
    if args.arrival_scale <= 0:
        raise SystemExit("--arrival-scale must be positive.")
    if args.arrival_scale != 1.0:
        jobs = [replace(job, submit_time=int(job.submit_time / args.arrival_scale)) for job in jobs]

    if not jobs:
        raise SystemExit("No jobs loaded. Check filters and input CSV.")
    if not nodes:
        raise SystemExit("No nodes loaded. Check node CSV or --max-nodes.")
    return jobs, nodes


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def select_job_window(jobs, job_offset: int = 0, max_jobs: int | None = None):
    end = None if max_jobs is None else job_offset + max_jobs
    selected = jobs[job_offset:end]
    if not selected:
        return selected
    min_submit = min(job.submit_time for job in selected)
    return [replace(job, submit_time=job.submit_time - min_submit) for job in selected]
