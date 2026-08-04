# Elysium Server Training Guide

Author: Tianci Wang

Last updated: 2026-08-04

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
tmux new -s rwr
```

List all sessions:

```bash
tmux ls
```

Reconnect to an existing session:

```bash
tmux attach -t rwr
```

Detach without stopping the training:

```
Ctrl+b
d
```

Delete a finished session:

```bash
tmux kill-session -t rwr
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
- CPU: 0-23

Start Docker:

```bash
CPU_RANGE="0-23"

docker run --rm -it \
    --cpuset-cpus="${CPU_RANGE}" \
    -u "$(id -u):$(id -g)" \
    -e HOME=/workspace/.docker-home \
    -e PYTHONNOUSERSITE=1 \
    -e MUJOCO_GL=osmesa \
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
pid 1's current affinity list: 0-23
```


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

Before large experiments:

```bash
python scripts/train_rwr.py \
    --epochs 1 \
    --samples-per-epoch 2 \
    --train-seeds-per-epoch 1 \
    --n-learning-seeds 1 \
    --n-eval-seeds 1 \
    --n-humans 3 \
    --output-dir runs/smoke_test
```

Check that

- environment runs
- artifacts are generated
- no exceptions

---

# 10. Run the real experiment

Before starting a long experiment, make sure you are inside a tmux session.

Example:

```bash
python -u scripts/train_rwr.py \
    --epochs 100 \
    --samples-per-epoch 30 \
    --seed 42 \
    --beta 5 \
    --train-seeds-per-epoch 10 \
    --n-learning-seeds 10 \
    --n-eval-seeds 20 \
    --n-humans 15 \
    --output-dir runs/rwr_exp1
```

Adjust parameters as needed.

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
tmux kill-session -t rwr
```

# 13. Download results

On the local laptop:

```bash
rsync -avh --progress \
    wang@10.79.7.244:/megaverse/datasets/wang/Master-Thesis/runs/ \
    ~/Polimi/workspace/Master-Thesis/runs/
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