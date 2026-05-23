# Fair-RC-RL GRU

This directory now keeps only the direct GRU scheduler.

## Files

- `train.py`: trains the direct GRU actor-critic scheduler.
- `inference.py`: evaluates a saved direct GRU checkpoint.
- `modele.py`: randomized candidate sampling plus direct GRU rollout.
- `policy.py`: GRU actor, state encoder, and value critic.
- `training_support.py`: trace replay and worker rollout helpers for GRU training.
- `utils.py`: shared metrics and CSV helpers.

## Train

```bash
python ours_model/train.py \
  --dataset 2023 \
  --job-offset 0 \
  --max-jobs 2000 \
  --eval-job-offset 2000 \
  --eval-max-jobs 1000 \
  --max-nodes 64 \
  --placement best_fit \
  --top-k 64 \
  --candidate-sampling-mode uniform_pool \
  --candidate-pool-k 64 \
  --candidate-d 16 \
  --episodes 50 \
  --batch-size 8 \
  --num-workers 8 \
  --gru-hidden-dim 64 \
  --lr 0.0003 \
  --critic-lr 0.0003 \
  --ppo-epochs 4 \
  --ppo-clip-eps 0.2 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --entropy-weight 0.01 \
  --policy-temperature 1.0 \
  --entropy-decay 0.995 \
  --output-dir ours_model/results/2023_direct_gru
```

## Outputs

With `--output-dir DIR`, training writes:

- `DIR/fair_rc_rl_train_log.csv`: one row per update.
- `DIR/fair_rc_rl_episode_log.csv`: one row per rollout episode plus one row per final evaluation seed.
- `DIR/fair_rc_rl.csv`: final evaluation rows.
- `DIR/fair_rc_rl_gru_final.pt`: final checkpoint.
- `DIR/fair_rc_rl_gru_best.pt`: best checkpoint selected by the configured metric plus gate penalties.
- `DIR/fair_rc_rl_gru_best_metrics.json`: best checkpoint metrics and selection details.
- `DIR/checkpoints/fair_rc_rl_gru_update_XXX.pt`: per-update checkpoints.

The final model uses non-heuristic bounded candidate exposure (`uniform_pool` by default), PPO/GAE actor-critic updates, candidate-set scoring context, and a Pass action appended to every non-empty feasible candidate set. Because Pass is an extra candidate, the actor sees up to `candidate_d + 1` actions. The console progress line includes the main convergence checks: train reward, candidate/feasible counts, Pass ratio, train CFS lag/slowdown/wait, validation CFS lag, selection primary value, gate penalty, distance from the previous best score, gate pass/fail ratios, PPO KL/clip fraction/explained variance, critic loss, entropy, temperature, gradient norm, and best checkpoint status.

Recommended command after enabling Pass:

```bash
python ours_model/train.py \
  --dataset 2023 \
  --job-offset 0 \
  --max-jobs 1000 \
  --eval-job-offset 2000 \
  --eval-max-jobs 1000 \
  --max-nodes 64 \
  --placement first_fit \
  --top-k 64 \
  --candidate-sampling-mode uniform_pool \
  --candidate-pool-k 64 \
  --candidate-d 32 \
  --episodes 600 \
  --batch-size 8 \
  --num-workers 8 \
  --gru-hidden-dim 128 \
  --lr 0.0002 \
  --critic-lr 0.0002 \
  --ppo-epochs 5 \
  --ppo-clip-eps 0.15 \
  --gamma 0.995 \
  --gae-lambda 0.98 \
  --entropy-weight 0.03 \
  --entropy-decay 0.999 \
  --value-loss-weight 0.5 \
  --max-grad-norm 0.5 \
  --reward-lag-level-weight 0.02 \
  --reward-lag-growth-weight 1.0 \
  --reward-lag-reduction-weight 2.0 \
  --reward-waiting-queue-weight 0.0 \
  --reward-gate-slowdown-weight 0.25 \
  --reward-gate-wait-weight 0.10 \
  --reward-gate-p95-weight 0.10 \
  --reward-gate-fairness-weight 0.25 \
  --reward-unfinished-weight 20.0 \
  --selection-metric cfs_normalized_service_lag_auc \
  --slowdown-gate-multiplier 1.35 \
  --wait-gate-multiplier 1.60 \
  --p95-gate-multiplier 1.75 \
  --fairness-floor 0.65 \
  --early-stop-patience 0 \
  --seed 0 \
  --output-dir ours_model/results/2023_ppo_pass_lag_first_fit
```

## Evaluate

```bash
python ours_model/inference.py \
  --dataset 2023 \
  --job-offset 2000 \
  --max-jobs 1000 \
  --max-nodes 64 \
  --placement best_fit \
  --top-k 64 \
  --candidate-sampling-mode uniform_pool \
  --candidate-pool-k 64 \
  --candidate-d 16 \
  --model-checkpoint ours_model/results/2023_direct_gru/fair_rc_rl_gru_best.pt \
  --out ours_model/results/2023_direct_gru/fair_rc_rl_eval.csv
```
