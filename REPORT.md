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
4. **Evidence collection:** wrapped the env in SB3's `Monitor` to log every
   episode's reward/length to CSV, save the trained model (`dqn_serp.zip`),
   write an evaluation summary (`dqn_results.json`), and plot the learning curve
   (`reward_curve.png`).

### 2.4 How to reproduce

Inside the ROS 2 Humble container, from the workspace root:

```bash
pip3 install --user -r src/ros2_flatland_rl_tutorial/requirements.txt
colcon build --symlink-install
source install/setup.bash

# headless training + evaluation + plot
RL_TIMESTEPS=50000 ./src/ros2_flatland_rl_tutorial/scripts/run_dqn_training.sh
```

All artifacts are written to `src/ros2_flatland_rl_tutorial/results/`.

### 2.5 Evidence of results

See the `results/` folder:

- `reward_curve.png` — learning curve (episode reward vs. training episode).
- `dqn_monitor.monitor.csv` — raw per-episode reward/length log.
- `dqn_results.json` — final evaluation accuracy and mean reward.
- `dqn_serp.zip` — the saved trained model.
- `training_log.txt` — full ROS 2 / training console log.

### 2.6 Observed training behavior

**Run configuration:** 100 000 training timesteps, evaluated on 20 deterministic
episodes. Trained inside the ROS 2 Humble container (CPU/GPU torch 2.12, SB3 2.8.0).

**Final numbers** (`results/dqn_results.json`):

| Metric | Value |
|---|---|
| Training episodes | 1055 |
| Training outcomes | 614 finished (58 %), 420 collisions (40 %), 21 timeouts (2 %) |
| First successful episode | #193 (~17 k timesteps) |
| Mean training reward (`ep_rew_mean`) | reached **≈ +449** |
| **Deterministic eval accuracy** | **9 / 20 = 0.45** |
| Mean eval reward | 275.3 |

**What the learning curve shows** (`results/reward_curve.png`):

1. **Exploration phase (episodes 0–~190):** reward stays flat around **−150**.
   ε is still high, so the robot acts almost randomly and almost always crashes
   into a wall. No successes yet.
2. **Breakthrough (~episode 193, ~17 k steps):** as ε decays, the agent starts
   exploiting its Q-values and discovers the action sequence that rounds the
   corner and reaches the beacon. Reward **jumps sharply from −150 to ≈ +600**.
3. **Consolidation (episodes ~250–1050):** reward stabilizes in the **+400 to
   +600** band and episode length settles around ~100 steps — the robot now
   reaches the goal in most training episodes (58 % overall, and the *rate*
   over the second half of training is much higher than that average).

This is the classic DQN signature: a long low-reward exploration plateau
followed by a steep improvement once the value estimates become good enough to
act greedily.

**Why deterministic eval (45 %) is lower than the training reward (≈449):**
the start and goal positions **swap every episode**, so the task is really two
mirrored problems (turn left vs. turn right). The greedy policy solves one
turn direction reliably but, for the other, the LiDAR-only state can drive it
into a repeating loop — which is exactly why the eval failures are split
between collisions (6) and **timeouts (5)** rather than only collisions. During
training the residual ε = 0.05 randomness occasionally knocks the robot out of
those loops, inflating the training reward relative to the purely greedy eval.

**Possible improvements** (left as future work): richer reward shaping (use the
change in `distance_to_end` between steps), feeding the previous action / a
short LiDAR history into the observation to break the symmetry, longer training,
or a larger network.

---
