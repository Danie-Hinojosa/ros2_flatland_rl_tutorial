# Deliverable Report — CAD Tutorial + Reinforcement Learning

**Author:** Daniel Hinojosa
**Course activity:** CAD tutorial + Reinforcement Learning example extension

This document covers the two independent components of the submission. The CAD
part is summarized briefly; this fork focuses on the Reinforcement Learning
extension.

---

## Part 1 — CAD Tutorial Design

The CAD design produced during the in-class tutorial is submitted separately
(model files, screenshots / exported views). It follows the tutorial
instructions and is organized in its own folder with descriptive file names.

> *(CAD files are delivered alongside this repository, in the CAD submission
> folder. See that folder's screenshots and exported views.)*

---

## Part 2 — Reinforcement Learning Extension

### 2.1 Environment

The base project is the **ROS 2 + Flatland** SERP robot navigation tutorial
([FilipeAlmeidaFEUP/ros2_flatland_rl_tutorial](https://github.com/FilipeAlmeidaFEUP/ros2_flatland_rl_tutorial)).
A differential-drive robot equipped with a LiDAR must drive down an L-shaped
hallway and reach a target area (green beacon) while avoiding collisions with
the walls. Each time the task restarts, the start and goal positions swap, so
the agent has to learn to turn both left and right.

The control loop is wrapped as a **Gymnasium environment** (`SerpControllerEnv`,
in `serp_rl/__init__.py`):

| Element | Definition |
|---|---|
| **Observation space** | `Box(0, 2, shape=(9,), float32)` — the 90 LiDAR rays are grouped into 9 sectors and the **minimum** distance of each sector is kept. |
| **Action space** | `Discrete(3)` — `0` move forward, `1` rotate left, `2` rotate right. |
| **Reward** | `+400 + (200 − steps)` on reaching the goal, `−200` on collision, `−300` on timeout (200 steps), `+2` for each *move-forward* action, `0` otherwise. |
| **Episode end** | collision (`terminated`), goal reached (`terminated`), or 200-step timeout (`truncated`). |

### 2.2 Algorithm selected: **DQN** (instead of PPO)

The rubric asks for a valid Stable-Baselines3 algorithm **different from PPO**.
I selected **DQN (Deep Q-Network)**.

**Why DQN is the right choice for this environment:**

- The action space is **discrete** (`Discrete(3)`). DQN is a value-based method
  that learns a Q-value for *each discrete action*, so it maps directly onto
  this problem.
- This rules out **SAC, TD3 and DDPG**, which only support **continuous**
  (`Box`) action spaces — they cannot be used here without redefining the
  robot's actions as continuous velocities.
- DQN also makes the most interesting contrast with the tutorial's default:

| | **PPO** (default) | **DQN** (this submission) |
|---|---|---|
| Family | Policy-gradient (actor) | Value-based (Q-learning) |
| On/off-policy | **On-policy** | **Off-policy** |
| Experience reuse | Discards rollouts after each update | **Replay buffer** reuses past transitions |
| Exploration | Stochastic policy (entropy) | **ε-greedy** with decay |
| Action spaces | Discrete **and** continuous | **Discrete only** |

**Hyper-parameters used** (`run_rl_alg` in `serp_rl/__init__.py`):

```python
DQN("MlpPolicy", env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05)
```

`exploration_fraction=0.3` means ε decays from 1.0 → 0.05 over the first 30 % of
training: the robot acts almost randomly at first and progressively trusts its
learned Q-values.

### 2.3 What I changed from the original tutorial

1. **Migrated the environment to the Gymnasium API** (the original used the
   deprecated `gym` package): `reset()` returns `(obs, info)`, `step()` returns
   `(obs, reward, terminated, truncated, info)`, and spaces come from
   `gymnasium.spaces`.
2. **Swapped PPO → DQN** with the hyper-parameters above.
3. **Robustness fixes:** clip/replace `NaN`/`inf` LiDAR readings, add a timeout
   to `wait_lidar_reading`, non-blocking `move_model` service calls, clean node
   shutdown.
4. **Best-checkpoint selection + parallel runs** (see §2.4): snapshot
   checkpoints during training and keep the best by eval accuracy; run several
   seeded workers in parallel and keep the global best.
5. **Evidence collection:** wrapped the env in SB3's `Monitor` to log every
   episode's reward/length to CSV, save the model (`dqn_serp.zip`), write
   evaluation summaries (`dqn_results.json`, `dqn_eval_history.json`,
   `parallel_comparison.json`), and plot the learning + checkpoint-accuracy
   curves.

### 2.4 Training methodology (best checkpoint + parallel runs)

Plain "train once, keep the final model" is unreliable for DQN here (see the
journey in §2.6). Two techniques fixed it:

1. **Best-checkpoint selection.** During a single continuous `learn()` call
   (so the ε schedule is correct over the whole budget) a `CheckpointCallback`
   snapshots the model every 15 k steps. After training, **every checkpoint is
   reloaded and evaluated on 20 deterministic episodes, and the one with the
   highest finish-rate is kept** as the delivered model. This protects against
   DQN's late-training instability — the final policy is often *not* the best.

2. **Parallel independent runs.** DQN is high-variance and this task is
   asymmetric (the start/goal swap makes it two mirrored turns), so a single
   run can get stuck mastering only one direction. `scripts/run_parallel_dqn.sh`
   launches **N independent workers with different seeds**, each a full Flatland
   sim + DQN agent isolated by its own `ROS_DOMAIN_ID`. `scripts/select_best.py`
   then promotes the **global best** model across all workers. The simulation
   (not the tiny MLP) is the bottleneck, so this uses the machine's many cores
   to get N runs in roughly one run's wall-clock, then keeps the best.

### 2.5 How to reproduce

Inside the ROS 2 Humble container, from the workspace root:

```bash
pip3 install --user -r src/ros2_flatland_rl_tutorial/requirements.txt
colcon build --symlink-install
source install/setup.bash

# (a) single headless run: train + pick best checkpoint + eval + plot
RL_TIMESTEPS=100000 ./src/ros2_flatland_rl_tutorial/scripts/run_dqn_training.sh

# (b) parallel: 4 independent seeded workers, keep the global best model
RL_WORKERS=4 RL_TIMESTEPS=100000 ./src/ros2_flatland_rl_tutorial/scripts/run_parallel_dqn.sh
```

All artifacts are written to `src/ros2_flatland_rl_tutorial/results/`.

### 2.6 Evidence of results

The delivered model is the **global best of a 4-worker parallel run**
(`results/`):

- `dqn_serp.zip` — the delivered (best) trained model.
- `reward_curve.png` — learning curve of the winning worker.
- `checkpoint_accuracy.png` — eval accuracy of each checkpoint (shows *why*
  best-checkpoint selection matters).
- `dqn_results.json` — final evaluation summary of the delivered model.
- `dqn_eval_history.json` — per-checkpoint accuracy for the winning worker.
- `parallel_comparison.json` — accuracy of all 4 workers.
- `dqn_monitor.monitor.csv`, `training_log.txt` — raw logs.
- `run_100k_baseline/`, `run_200k_unstable/` — earlier runs kept as evidence.

**Final result:**

| Run | Method | Eval accuracy | Mean eval reward |
|---|---|---|---|
| 100 k single | final model | 0.45 (9/20) | 275 |
| 200 k single | final model | **0.10** (2/20) | −41 |
| **4 × 100 k parallel** | **best checkpoint, global best worker** | **0.90 (18/20)** | **595** |

All four parallel workers beat the original baseline (0.65 – 0.90).

### 2.7 Observed training behavior & key finding

**The central finding: more training made DQN *worse*, not better.** Naively
extending the single run from 100 k to 200 k steps dropped accuracy from 0.45 to
0.10 — the policy destabilized and collapsed (classic DQN Q-value
overestimation / instability). The `checkpoint_accuracy.png` plot of the winning
worker shows this cleanly: accuracy **peaks at 0.90 at 15 k steps, then decays to
~0.35 by 60–75 k**, partially recovering to 0.60 by 100 k. *Notably, all four
workers independently picked their 15 k-step checkpoint as best* — the useful
policy is learned early, and prolonged training erodes it.

**Learning curve** (`reward_curve.png`): the usual DQN signature — a flat
low-reward exploration plateau (≈ −150) while ε is high, then a sharp jump to
≈ +600 once the Q-values are good enough to exploit, followed by noisy,
unstable consolidation.

**The asymmetry was solved.** In the 0.45 baseline the deterministic policy
succeeded on *every even* eval episode and failed *every odd* one — it had
learned only one of the two mirrored turn directions (failures split between
collisions and timeout loops). The delivered 0.90 model succeeds in **both**
directions: 18/20 finishes spread across the run, with only 2 isolated
collisions and **zero timeouts**. Combining seed diversity (one worker happened
to learn both turns) with early-checkpoint selection produced a model that
generalizes to the full task.

**Possible further improvements** (future work): reward shaping using the
change in `distance_to_end` per step, adding the previous action / a short
LiDAR history to the observation, or a Double/Dueling-DQN variant for more
stable Q-values.

---
