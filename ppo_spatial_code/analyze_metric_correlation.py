from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_METRICS = [
    "avg_wait",
    "p95_wait",
    "avg_slowdown",
    "jain_inverse_slowdown",
    "avg_fragmentation",
    "gpu_utilization",
    "avg_debt_gap",
    "avg_candidate_size",
    "avg_blocked_count_topk",
    "avg_resource_score_chosen",
    "avg_feasible_node_count_chosen",
    "median_slowdown",
    "p95_slowdown",
    "max_slowdown",
    "log_slowdown_std",
    "tail_fairness_gap",
    "service_share_l1_gap",
    "service_share_l2_gap",
    "max_service_under_share",
    "cfs_normalized_service_lag_auc",
    "cfs_normalized_lag_auc",
    "cfs_max_normalized_service_lag",
    "cfs_avg_max_normalized_service_lag",
    "cfs_lag_over_0p5_ratio",
    "cfs_lag_over_1p0_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze metric correlations across scheduler result CSVs.")
    parser.add_argument("csv", nargs="+", type=Path, help="Result CSV files to combine.")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS, help="Metric columns to analyze.")
    parser.add_argument("--out-prefix", type=Path, default=None, help="Optional prefix for output CSV files.")
    return parser.parse_args()


def load_rows(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No CSV files provided.")
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    metrics = [name for name in args.metrics if name in rows.columns]
    if not metrics:
        available = ", ".join(rows.columns)
        raise ValueError(f"None of the requested metrics exist. Available columns: {available}")

    data = rows[metrics].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(axis=1, how="all")
    metrics = list(data.columns)
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")

    rank_columns = ["source_csv"]
    for column in ("policy", "placement", "dataset", "seed", "job_offset", "max_jobs"):
        if column in rows.columns:
            rank_columns.append(column)
    rank_table = rows[rank_columns + metrics].copy()
    for metric in metrics:
        rank_table[f"{metric}_rank"] = pd.to_numeric(rank_table[metric], errors="coerce").rank(method="min")

    print("\nPearson correlation:")
    print(pearson.round(4).to_string())
    print("\nSpearman correlation:")
    print(spearman.round(4).to_string())
    print("\nMetric rank table:")
    print(rank_table.to_string(index=False))

    if args.out_prefix is not None:
        args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        pearson.to_csv(args.out_prefix.with_name(args.out_prefix.name + "_pearson.csv"))
        spearman.to_csv(args.out_prefix.with_name(args.out_prefix.name + "_spearman.csv"))
        rank_table.to_csv(args.out_prefix.with_name(args.out_prefix.name + "_ranks.csv"), index=False)


if __name__ == "__main__":
    main()
