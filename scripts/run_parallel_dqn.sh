#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Parallel DQN training for the SERP / Flatland task.
#
# DQN is high-variance and can collapse late in training, and this task is
# asymmetric (start/goal swap → two mirrored turns). Running several
# INDEPENDENT trainings with different seeds and then keeping the single best
# model is an effective, embarrassingly-parallel way to use a multi-core
# machine: each worker is a full Flatland sim + DQN agent, isolated from the
# others by its own ROS_DOMAIN_ID.
#
# The simulation (not the tiny MLP) is the bottleneck, so this does NOT make a
# single run faster — it gives N runs in roughly the same wall-clock and we
# select the best checkpoint across all of them.
#
# Usage (inside container):
#   RL_WORKERS=4 RL_TIMESTEPS=100000 ./scripts/run_parallel_dqn.sh
# ---------------------------------------------------------------------------
# NOTE: no `set -u` — sourcing the ROS setup scripts references unbound vars.

WS=/workspace/ros2_ws
PKG=serp_rl
RESULTS_DIR="$WS/src/ros2_flatland_rl_tutorial/results"
PAR_DIR="$RESULTS_DIR/parallel"
mkdir -p "$PAR_DIR"

N="${RL_WORKERS:-4}"
export RL_TIMESTEPS="${RL_TIMESTEPS:-100000}"
export RL_CHECKPOINT_FREQ="${RL_CHECKPOINT_FREQ:-15000}"
export RL_CKPT_EVAL_EPISODES="${RL_CKPT_EVAL_EPISODES:-20}"
export RL_TEST_EPISODES="${RL_TEST_EPISODES:-20}"
BASE_DOMAIN="${RL_BASE_DOMAIN:-71}"   # avoid the compose default (42)
TIMEOUT="${RL_TIMEOUT:-7200}"

source /opt/ros/humble/setup.bash
cd "$WS"

echo "[parallel] Building $PKG ..."
colcon build --packages-select "$PKG" --symlink-install
source "$WS/install/setup.bash"

declare -a PGIDS
declare -a RUNDIRS
declare -a LOGS

echo "[parallel] Launching $N workers (RL_TIMESTEPS=$RL_TIMESTEPS each) ..."
for i in $(seq 0 $((N - 1))); do
  RUN_DIR="$PAR_DIR/run_${i}"
  rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"
  LOG="$RUN_DIR/training_log.txt"
  : > "$LOG"
  DOMAIN=$((BASE_DOMAIN + i))

  # Each worker: isolated ROS graph (ROS_DOMAIN_ID), own results dir, own seed.
  setsid env \
      ROS_DOMAIN_ID="$DOMAIN" \
      RL_RESULTS_DIR="$RUN_DIR" \
      RL_SEED="$i" \
      ros2 launch "$PKG" serp_rl.launch.py show_viz:=false >> "$LOG" 2>&1 &
  PGIDS[$i]=$!
  RUNDIRS[$i]="$RUN_DIR"
  LOGS[$i]="$LOG"
  echo "  worker $i: domain=$DOMAIN seed=$i pgid=${PGIDS[$i]} dir=$RUN_DIR"
  sleep 3   # stagger startup so the N flatland servers don't race on bring-up
done

# ---- Wait for all workers to finish (RL_RUN_DONE) or die ----
echo "[parallel] Waiting for workers (timeout ${TIMEOUT}s) ..."
elapsed=0
while true; do
  done_cnt=0
  alive_cnt=0
  for i in $(seq 0 $((N - 1))); do
    if grep -q "=== RL_RUN_DONE ===" "${LOGS[$i]}" 2>/dev/null; then
      done_cnt=$((done_cnt + 1))
    elif kill -0 "${PGIDS[$i]}" 2>/dev/null; then
      alive_cnt=$((alive_cnt + 1))
    fi
  done
  finished=$((done_cnt))
  [ "$finished" -ge "$N" ] && break
  # all remaining workers dead but not done -> stop waiting
  if [ "$alive_cnt" -eq 0 ] && [ "$finished" -lt "$N" ]; then
    echo "[parallel] all workers exited; done=$done_cnt of $N"
    break
  fi
  sleep 10
  elapsed=$((elapsed + 10))
  if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo "[parallel] timeout"; break
  fi
done

# ---- Tear down every worker tree ----
echo "[parallel] tearing down workers ..."
for i in $(seq 0 $((N - 1))); do
  kill -INT -"${PGIDS[$i]}" 2>/dev/null || true
done
sleep 3
for i in $(seq 0 $((N - 1))); do
  kill -9 -"${PGIDS[$i]}" 2>/dev/null || true
done
pkill -9 -f flatland_server 2>/dev/null || true

# ---- Select the global best model across workers ----
echo "[parallel] selecting global best model ..."
python3 "$WS/src/ros2_flatland_rl_tutorial/scripts/select_best.py"

echo "[parallel] done. Per-worker results in $PAR_DIR ; global best promoted to $RESULTS_DIR"
