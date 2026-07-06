# AGENTS.md

## Project Overview

- Repository root: `/home/tianci/Polimi/workspace/Master-Thesis`
- Main environment ID: `MuseumEnv-v0`
- Core stack: MuJoCo + Gymnasium + NumPy, with a robot guide and a human crowd moving through a virtual museum layout.
- The main terminal entrypoint is `test_env.py`, which provides `demo`, `train`, and `record` workflows.
- Most implementation work lives under `museum_env_package/museum_env/`.

## Key Entry Points

- `museum_env_package/museum_env/env.py`: top-level environment orchestrator. Owns `reset()`, `step()`, render integration, robot/human coordination, listening flow, callback flow, and reward/info assembly.
- `museum_env_package/museum_env/robot.py`: robot finite-state controller. Handles waypoint following, stop-and-turn behavior, callback cueing, speech state, and emotion state.
- `museum_env_package/museum_env/human.py`: per-human state container plus shared movement utilities, timers, mode transitions, and profile-dependent spacing behavior.
- `museum_env_package/museum_env/human_behaviors.py`: mode-specific human motion policies for wandering, following, listening, distracted, overwhelmed, impatient, and post-explanation coordination variants.
- `museum_env_package/museum_env/env_state.py`: orchestration constants and dataclasses for listening, callback, post-explanation, cached observations, and debug state.
- `museum_env_package/museum_env/env_reporting.py`: compact `info` payload builder plus rendering helpers for labels, speaking halos, and robot visual state.
- `museum_env_package/museum_env/register_env.py`: Gymnasium registration for `MuseumEnv-v0`.

## Behavior Model

- Robot modes: `move`, `stop`, `callback`
- Human modes: `wandering`, `following`, `listening`, `distracted`, `overwhelmed`, `impatient`
- Human profiles: `normal`, `neurodivergent`
- `MuseumEnv` coordinates the multi-stage runtime flow. Listening, question handling, callback attempts, and post-explanation crowd regulation are not isolated inside one class.
- For robot/listening/callback changes, inspect `env.py`, `robot.py`, and `env_state.py` together.
- For crowd-motion or per-human behavior changes, inspect `human.py` and `human_behaviors.py` together.
- For visualization changes, treat `env_reporting.py` and `museum_env_package/museum_env/assets/museum_scene.xml` as coupled.

## Runtime Contract

- `MuseumEnv.step()` returns `obs, reward, terminated, truncated, info`.
- `info` currently uses a compact five-key schema:
  - `info["events"]`
  - `info["episode"]`
  - `info["phase"]`
  - `info["robot"]`
  - `info["crowd"]`
- Most useful debug fields:
  - `info["phase"]["follow"]`
  - `info["phase"]["listen"]`
  - `info["robot"]["mode"]`
  - `info["robot"]["callback_phase"]`
  - `info["robot"]["speaker_active"]`
  - `info["crowd"]["distracted_indices"]`
  - `info["crowd"]["human_robot_distance"]`
- `info["robot"]` is a compact pose/goal/action summary, not a full internal dump.
- `info["crowd"]` is a compact crowd snapshot with positions, goals, modes, profiles, reached-goal indices, distracted indices, and human-robot distances.
- `info["episode"]` always includes `step` and `terminated_reason`, and on terminal or truncated steps also includes duration, trigger counts, scalar return, and reward components.
- Preserve this compact contract unless the task explicitly requires an interface change. If you change it, update the dependent docs and tests in the same task.

## Runbook

These commands assume the current shared interpreter on this workstation:

```bash
cd /home/tianci/Polimi/workspace/Master-Thesis
/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode demo
/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode train --print-every 1000
/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode train --max-steps 5 --print-every 1
/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode record --use-timestamp-subfolder
/home/tianci/Polimi/workspace/venv/bin/python -m unittest /home/tianci/Polimi/workspace/Master-Thesis/museum_env_package/museum_env/test_baseline_policy.py
```

## Working Rules

- Keep `MuseumEnv-v0`, the current robot/human mode names, and the `step()` return shape stable unless the task explicitly asks for a public interface change.
- Preserve the compact `info` schema by default. If you add or remove fields, update repo docs and affected tests in the same change.
- Start behavior validation with the short train smoke command before running longer demo or record flows.
- Treat the unittest command as a regression probe. Report observed outcomes instead of assuming the full suite is green on the current branch.
- The worktree may already contain unrelated user changes. Do not reset, rewrite, or “clean up” files outside the task.
- When a bug appears in orchestration, inspect runtime state labels in `env_state.py` before changing transition logic blindly.

## Debug Notes

- `test_env.py --mode demo` uses a passive MuJoCo viewer. Press `P` to pause or resume.
- `test_env.py --mode record` writes video output under `videos/` by default and supports timestamped subfolders.
- A short non-render smoke run with `--mode train --max-steps 5 --print-every 1` is a safe first check after edits.
- On this workstation, Matplotlib may warn that `~/.config/matplotlib` is not writable and fall back to a temporary cache under `/tmp`. This warning is currently non-blocking for the runner.
- The baseline unittest entrypoint is useful, but do not claim it is fully green unless you actually ran it and reported the outcome.
