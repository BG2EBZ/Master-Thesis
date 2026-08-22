# Master Thesis Museum Env

## Step info schema

`MuseumEnv.step()` now returns a compact `info` structure with five top-level keys:

- `info["events"]`
- `info["episode"]`
- `info["phase"]`
- `info["robot"]`
- `info["crowd"]`

`info["events"]` contains the high-level episode markers:

- `entered_listen`
- `started_listen_wait`
- `completed_listen_wait`
- `question_started`
- `question_completed`
- `final_listen_ready`
- `callback_triggered`
- `callback_completed`
- `callback_success`
- `callback_ignored`
- `happy_triggered`
- `happy_completed`

These event fields are boolean edge markers for the current step.

`info["episode"]` contains episode-level metadata:

- `step`
- `terminated_reason`
- terminal-only on the final step:
- `duration_seconds`
- `overwhelmed_triggers`
- `impatient_triggers`
- `distracted_triggers`
- `return`
- `reward_components`

`duration_seconds` is simulated episode duration, computed from `step_count * dt`, not wall-clock runtime.

`info["phase"]` contains the current orchestration phase:

- `follow`
- `listen`

`info["robot"]` contains pose and control summary:

- `pose_xy`
- `goal_xy`
- `dist_to_goal`
- `yaw`
- `mode`
- `callback_phase`
- `emotion`
- `speaker_active`
- `action`

`info["crowd"]` contains the compact crowd snapshot:

- `pose_xy`
- `goal_xy`
- `modes`
- `profiles`
- `human_robot_distance`
- `reached_goal_indices`
- `distracted_indices`

The old debug-heavy `metrics`, per-human fuzzy dumps, timers, and callback internals are no longer returned by default.

## Reward semantics

`MuseumEnv.step()` now uses episodic reward semantics:

- intermediate steps return `0.0`
- the final scalar reward is emitted only when the episode completes or times out
- `info` keeps the same compact top-level schema shown above
- final-step reward breakdown is exposed under `info["episode"]["reward_components"]`
- each env step still represents `0.05s` of simulated decision time, even though MuJoCo may integrate smaller internal physics substeps

Timing note:

- public decision timestep: `dt = 0.05s`
- MuJoCo integrates finer internal physics substeps underneath that decision timestep
- final `info["episode"]["duration_seconds"]` is based on simulated time, not real execution time

Example:

```python
obs, reward, terminated, truncated, info = env.step(None)

print(info["phase"]["listen"])
print(reward)
print(info["crowd"]["modes"])

if terminated or truncated:
    print("Final episode reward:", reward)
```

## Run scripts

- Demo: `/home/tianci/Polimi/workspace/venv/bin/python scripts/run_env.py --mode demo [--seed 2]`
- Fast train loop: `/home/tianci/Polimi/workspace/venv/bin/python scripts/run_env.py --mode train [--seed 2]`
- Record video: `/home/tianci/Polimi/workspace/venv/bin/python scripts/run_env.py --mode record --use-timestamp-subfolder [--seed 2]`
- RWR training: `/home/tianci/Polimi/workspace/venv/bin/python scripts/train_rwr.py`
- REPS training: `/home/tianci/Polimi/workspace/venv/bin/python scripts/train_reps.py --eps 0.1`
- ePPO training: `/home/tianci/Polimi/workspace/venv/bin/python scripts/train_eppo.py --eppo-lr 0.001 --eppo-epochs 5 --eps-ppo 0.2 --ent-coeff 0.001`
- Evaluate a learned policy: `/home/tianci/Polimi/workspace/venv/bin/python scripts/eval_baseline.py --learned-params-json artifacts/runs/rwr_20260721_155908/best_params.json`

If running the evaluator from inside `scripts/`, use the parent-relative path:

- `/home/tianci/Polimi/workspace/venv/bin/python eval_baseline.py --learned-params-json ../artifacts/runs/rwr_20260721_155908/best_params.json`

Reproducibility example:

- `/home/tianci/Polimi/workspace/venv/bin/python scripts/run_env.py --mode train --seed 2 --max-steps 5 --print-every 1`

`--seed` is optional and is passed only to the initial `env.reset(...)` for that invocation.

Policy-search training uses separate CLI entrypoints:

- RWR: `scripts/train_rwr.py --beta 0.1`
- REPS: `scripts/train_reps.py --eps 0.1`
- ePPO: `scripts/train_eppo.py --eppo-lr 0.001 --eppo-epochs 5 --eppo-batch-size 10 --eps-ppo 0.2 --ent-coeff 0.001`

All policy-search scripts share the same rollout, reward, seed-plan, plotting, and artifact machinery under `train/policy_search/`. The trainers write:

- default RWR output directory: `artifacts/runs/rwr_YYYYMMDD_HHMMSS`
- default REPS output directory: `artifacts/runs/reps_YYYYMMDD_HHMMSS`
- default ePPO output directory: `artifacts/runs/eppo_YYYYMMDD_HHMMSS`
- `training_metrics.csv`
- `training_metrics.png`
- `best_params.json`: best sampled theta, final distribution-center policy params, `algorithm`, `beta`, `eps`, `eppo_config`, `final_mu`, and `final_std`

Minimal ePPO smoke test:

```bash
/home/tianci/Polimi/workspace/venv/bin/python scripts/train_eppo.py \
  --epochs 1 \
  --samples-per-epoch 2 \
  --train-seeds-per-epoch 1 \
  --n-learning-seeds 1 \
  --n-eval-seeds 1 \
  --n-humans 3 \
  --max-workers 1 \
  --eppo-lr 0.001 \
  --eppo-epochs 5 \
  --eppo-batch-size 2 \
  --eps-ppo 0.2 \
  --ent-coeff 0.001 \
  --output-dir runs/smoke_test_eppo
```

For a learning-seed by epoch dataset, run the policy-search trainer with multiple learning seeds:

```bash
/home/tianci/Polimi/workspace/venv/bin/python scripts/train_rwr.py \
  --epochs 30 \
  --samples-per-epoch 30 \
  --max-workers 8 \
  --n-learning-seeds 10 \
  --n-eval-seeds 20 \
  --output-dir artifacts/runs/rwr_dataset_10x31
```

Use `--max-workers` to cap the number of parallel worker processes used for rollout
evaluation. The actual worker count is also limited by the number of rollout tasks and
the available CPU count. Parallel rollout workers use Python's `spawn` start method so
MuJoCo and other native libraries initialize inside each worker process on server runs.

The multi-seed path writes:

- `learning_curve_raw.csv`: long-form rows for learned policy plus baseline, evaluated on fresh per-epoch seeds; `policy` can be `rwr`, `reps`, `eppo`, or `baseline`
- `learning_curve_matrix.csv`: learned-policy matrix with `learning_seed,epoch_0,epoch_1,...`
- `seed_plan.json`: shared learning seeds, training rollout seeds, and evaluation seeds for reproducible comparisons
- `learning_curve_summary.json`: aggregate returns, per-learning-seed final policy params, per-epoch eval seeds, and `seed_plan_hash`
- `learning_curve_plot.png`

For each learning seed and epoch, the learned policy and baseline are evaluated on the same fresh `--n-eval-seeds` episodes.

To guarantee RWR, REPS, and ePPO use the same seeds, train RWR first and reuse its seed plan for the other algorithms:

```bash
/home/tianci/Polimi/workspace/venv/bin/python scripts/train_reps.py \
  --eps 0.1 \
  --epochs 30 \
  --samples-per-epoch 30 \
  --max-workers 8 \
  --n-learning-seeds 10 \
  --n-eval-seeds 20 \
  --seed-plan artifacts/runs/rwr_dataset_10x31/seed_plan.json \
  --output-dir artifacts/runs/reps_dataset_10x31
```

```bash
/home/tianci/Polimi/workspace/venv/bin/python scripts/train_eppo.py \
  --epochs 30 \
  --samples-per-epoch 30 \
  --max-workers 8 \
  --n-learning-seeds 10 \
  --n-eval-seeds 20 \
  --seed-plan artifacts/runs/rwr_dataset_10x31/seed_plan.json \
  --eppo-lr 0.001 \
  --eppo-epochs 5 \
  --eppo-batch-size 10 \
  --eps-ppo 0.2 \
  --ent-coeff 0.001 \
  --output-dir artifacts/runs/eppo_dataset_10x31
```

Then create a MushroomRL-style comparison plot with baseline, RWR, REPS, and ePPO in one figure:

```bash
/home/tianci/Polimi/workspace/venv/bin/python scripts/compare_policy_search_runs.py \
  --run rwr=artifacts/runs/rwr_dataset_10x31/learning_curve_raw.csv \
  --run reps=artifacts/runs/reps_dataset_10x31/learning_curve_raw.csv \
  --run eppo=artifacts/runs/eppo_dataset_10x31/learning_curve_raw.csv \
  --output-dir artifacts/runs/policy_search_compare_10x31
```

This writes:

- `policy_search_comparison_raw.csv`
- `policy_search_comparison_summary.json`
- `policy_search_comparison_plot.png`

The comparison script verifies that input runs share the same `seed_plan_hash` before plotting.

For a quick REPS `eps` sweep, run the same command with:

```bash
scripts/train_reps.py --eps 0.05
scripts/train_reps.py --eps 0.1
scripts/train_reps.py --eps 0.5
scripts/train_reps.py --eps 1.0
```

For a quick ePPO stability sweep, keep the same seed plan and try:

```bash
scripts/train_eppo.py --eppo-lr 0.0005 --eppo-epochs 5 --eps-ppo 0.1 --ent-coeff 0.001
scripts/train_eppo.py --eppo-lr 0.001  --eppo-epochs 5 --eps-ppo 0.2 --ent-coeff 0.001
scripts/train_eppo.py --eppo-lr 0.003  --eppo-epochs 10 --eps-ppo 0.2 --ent-coeff 0.001
```

To replot the saved learning curves without retraining:

```bash
/home/tianci/Polimi/workspace/venv/bin/python scripts/plot_policy_search_learning_curve.py \
  --input-csv artifacts/runs/rwr_dataset_10x31/learning_curve_raw.csv
```

This writes both `learning_curve_plot.png` and `learning_curve_other_metrics.png`.

To write both plots to another directory:

```bash
/home/tianci/Polimi/workspace/venv/bin/python scripts/plot_policy_search_learning_curve.py \
  --input-csv artifacts/runs/rwr_dataset_10x31/learning_curve_raw.csv \
  --output-path artifacts/runs/rwr_dataset_10x31/replots
```
