# Elysium Server Training Guide

Author: Tianci Wang

Last updated: 2026-08-10

---

# 1. Connect to VPN (if not in AIRLab)

If outside the lab, connect to the Polimi VPN first.

Then test the connection:

```bash
ssh wang@10.79.7.244
```

---

# 2. Login to the server

```bash
ssh wang@10.79.7.244
```

Go to the project directory:

```bash
cd /megaverse/datasets/wang/Master-Thesis
```

---

# 3. Update the code

On the **local laptop**:

```bash
cd ~/Polimi/workspace

rsync -avh --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='runs' \
    --exclude='outputs' \
    ./Master-Thesis/ \
    wang@10.79.7.244:/megaverse/datasets/wang/Master-Thesis/
```

---

# 4. Book a GPU

Open:

https://docs.google.com/spreadsheets/d/1pMX7jAOP8vdGFNMUS9m3xLctcjIGoKxuGCWVlbWJQoU

Reserve

- GPU
- CPU slot

Do NOT use GPUs that are already booked.

---

# 5. Check server status

```bash
nvidia-smi
```

Check Docker:

```bash
docker ps
```

---

# 6. Start a tmux session

Create a new tmux session:

```bash
tmux new -s policy_search
```

List all sessions:

```bash
tmux ls
```

Reconnect to an existing session:

```bash
tmux attach -t policy_search
```

Detach without stopping the training:

```
Ctrl+b
d
```

Delete a finished session:

```bash
tmux kill-session -t policy_search
```

> **Recommendation**
>
> Always start Docker **inside a tmux session**.
> This allows the training to continue even if:
>
> - SSH disconnects
> - VPN disconnects
> - The local terminal is closed
> - The laptop is shut down

---

# 6.5 Rebuild Docker (only if dependencies changed)

If you modified

- Dockerfile
- setup.py
- requirements

rebuild:

```bash
docker build --no-cache \
    -t wang/master-thesis:py311 .
```

Otherwise skip this step.

---

# 7. Start Docker

Choose the CPU range assigned to you in the booking sheet.

Example:

- GPU: 0
- CPU: 0-31
- max policy-search workers: 32

Start Docker:

```bash
CPU_RANGE="0-31"
MAX_WORKERS="32"

docker run --rm -it \
    --cpuset-cpus="${CPU_RANGE}" \
    -u "$(id -u):$(id -g)" \
    -e HOME=/workspace/.docker-home \
    -e PYTHONNOUSERSITE=1 \
    -e MUJOCO_GL=osmesa \
    -e MAX_WORKERS="${MAX_WORKERS}" \
    -v "$PWD":/workspace \
    -w /workspace \
    wang/master-thesis:py311 \
    bash
```
Expected prompt:

```text
I have no name!@xxxx:/workspace$
```

This is normal. The container is running with your UID/GID instead of `root`, so no username exists inside the image.

Verify the assigned CPUs:

```bash
taskset -pc $$
```

Expected output:

```text
pid 1's current affinity list: 0-31
```

Use the same CPU booking to choose `--max-workers` for policy-search training.
For example, CPU range `0-31` means 32 assigned CPU cores, so use
`--max-workers 32` or a smaller number if you want to leave spare capacity.
The trainer also caps the actual worker count by the number of rollout tasks
and the CPU count visible inside Docker.

---

# 8. Verify environment

```bash
python - <<'PY'
import numpy
import scipy
import scipy.linalg
import mujoco
import gymnasium
import skfuzzy
import museum_env

print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("MuJoCo:", mujoco.__version__)
print("Everything OK")
PY
```

---

# 9. Smoke test

Before large experiments, run a tiny RWR smoke test:

```bash
python scripts/train_rwr.py \
    --epochs 1 \
    --samples-per-epoch 2 \
    --max-workers "${MAX_WORKERS}" \
    --train-seeds-per-epoch 1 \
    --n-learning-seeds 1 \
    --n-eval-seeds 1 \
    --n-humans 3 \
    --output-dir runs/smoke_test
```

Then run a tiny REPS smoke test:

```bash
python scripts/train_reps.py \
    --eps 0.1 \
    --epochs 1 \
    --samples-per-epoch 2 \
    --max-workers "${MAX_WORKERS}" \
    --train-seeds-per-epoch 1 \
    --n-learning-seeds 1 \
    --n-eval-seeds 1 \
    --n-humans 3 \
    --output-dir runs/smoke_test_reps
```

Check that

- environment runs
- artifacts are generated
- no exceptions

---

# 10. Run the real RWR experiment

Before starting a long experiment, make sure you are inside a tmux session.

Example:

```bash
python -u scripts/train_rwr.py \
    --epochs 100 \
    --samples-per-epoch 30 \
    --max-workers "${MAX_WORKERS}" \
    --seed 42 \
    --beta 0.1 \
    --train-seeds-per-epoch 1 \
    --n-learning-seeds 10 \
    --n-eval-seeds 20 \
    --n-humans 15 \
    --output-dir artifacts/runs/rwr_server_10x101
```

This writes `artifacts/runs/rwr_server_10x101/seed_plan.json`. Keep this file,
because REPS should reuse it for a fair comparison.

---

# 10.5 Run the matching REPS experiment

Use the same training budget and reuse the RWR seed plan:

```bash
python -u scripts/train_reps.py \
    --epochs 100 \
    --samples-per-epoch 30 \
    --max-workers "${MAX_WORKERS}" \
    --eps 0.1 \
    --train-seeds-per-epoch 1 \
    --n-learning-seeds 10 \
    --n-eval-seeds 20 \
    --n-humans 15 \
    --seed-plan artifacts/runs/rwr_server_10x101/seed_plan.json \
    --output-dir artifacts/runs/reps_server_10x101_eps01
```

For an `eps` sweep, repeat the REPS command with:

```bash
--eps 0.05 --output-dir artifacts/runs/reps_server_10x101_eps005
--eps 0.1  --output-dir artifacts/runs/reps_server_10x101_eps01
--eps 0.5  --output-dir artifacts/runs/reps_server_10x101_eps05
--eps 1.0  --output-dir artifacts/runs/reps_server_10x101_eps10
```

All REPS runs in the sweep should use the same RWR `seed_plan.json`.

---

# 10.6 Compare baseline, RWR, and REPS

After RWR and REPS finish, create a MushroomRL-style mean/CI comparison plot:

```bash
python scripts/compare_policy_search_runs.py \
    --run rwr=artifacts/runs/rwr_server_10x101/learning_curve_raw.csv \
    --run reps=artifacts/runs/reps_server_10x101_eps01/learning_curve_raw.csv \
    --output-dir artifacts/runs/policy_search_compare_server_10x101
```

The comparison script checks that RWR and REPS share the same `seed_plan_hash`.
If the hashes differ, the script refuses to plot because the comparison is not fair.

The comparison output includes:

- `policy_search_comparison_raw.csv`
- `policy_search_comparison_summary.json`
- `policy_search_comparison_plot.png`

The plot contains three curves:

- baseline
- RWR
- REPS

Adjust parameters as needed, but keep RWR and REPS budgets matched.

Important training parameters:

- `--epochs`: number of policy-search update epochs. Default: `100`.
- `--samples-per-epoch`: sampled policy count per epoch. Default: `30`.
- `--max-workers`: maximum parallel worker processes for rollout evaluation. Pass this explicitly from the CPU booking.
- `--seed`: master seed for policy sampling and rollout seed generation. Default: `42`.
- `--beta`: RWR reward-weight temperature. Used only by `scripts/train_rwr.py`. Default: `0.1`.
- `--eps`: REPS KL-divergence update bound. Used only by `scripts/train_reps.py`. Default: `0.1`.
- `--train-seeds-per-epoch`: rollout seeds used to evaluate each sampled theta during training. Default: `1`.
- `--n-learning-seeds`: independent learning runs. Use more than `1` to generate learning-curve data. Default: `10`.
- `--n-eval-seeds`: fresh evaluation episodes per epoch in multi-learning-seed mode. Default: `20`.
- `--n-humans`: number of humans in the simulated museum crowd. Default: `15`.
- `--seed-plan`: saved seed plan to reuse for another algorithm. Use this to make RWR and REPS share exactly the same seeds.
- `--output-dir`: result directory. If omitted, defaults are `artifacts/runs/rwr_YYYYMMDD_HHMMSS` for RWR and `artifacts/runs/reps_YYYYMMDD_HHMMSS` for REPS.

Important artifacts:

- `seed_plan.json`: learning seeds, training rollout seeds, and evaluation seeds.
- `learning_curve_raw.csv`: learned policy plus baseline rows.
- `learning_curve_summary.json`: aggregate results and `seed_plan_hash`.
- `learning_curve_plot.png`: one learned policy versus baseline.
- `policy_search_comparison_plot.png`: baseline, RWR, and REPS in one figure.

---

# 11. Monitor training

GPU usage:

```bash
watch -n 2 nvidia-smi
```

CPU usage:

```bash
htop
```

or

```bash
top
```

---

# 12. Finish training

If the training is finished:

Exit Docker:

```bash
exit
```

Detach from tmux:

```
Ctrl+b
d
```

Or delete the tmux session:

```bash
tmux kill-session -t policy_search
```

# 13. Download results

On the local laptop:

```bash
rsync -avh --progress \
    wang@10.79.7.244:/megaverse/datasets/wang/Master-Thesis/artifacts/runs/ \
    ~/Polimi/workspace/Master-Thesis/artifacts/runs/
```

---

# 14. Useful commands

Current directory

```bash
pwd
```

Check CPUs

```bash
nproc
```

Check GPUs

```bash
nvidia-smi
```

Check Docker images

```bash
docker images
```

Check running containers

```bash
docker ps
```

Stop a container

```bash
docker stop <container_name>
```

---

# Notes

- Never use `sudo docker`.
- Always store data under:

```
/megaverse/datasets/wang
```

- Never save experimental data inside the container.
- Only rebuild Docker when dependencies change.
- Python code updates do NOT require rebuilding Docker because the project directory is bind-mounted.
