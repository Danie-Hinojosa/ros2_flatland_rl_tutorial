#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Trains the SERP robot with DQN in the Flatland environment, headless.
#
# Runs INSIDE the ROS 2 Humble container. It builds the package, launches
# Flatland + the RL controller without visualization (show_viz:=false),
# waits for the training/eval to finish (detected via a log marker) and then
# tears down the launch and plots the reward curve.
#
# Usage (inside container):
#   RL_TIMESTEPS=50000 ./src/ros2_flatland_rl_tutorial/scripts/run_dqn_training.sh
# ---------------------------------------------------------------------------
set -e

WS=/workspace/ros2_ws
PKG=serp_rl
RESULTS_DIR="$WS/src/ros2_flatland_rl_tutorial/results"
mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/training_log.txt"

export RL_TIMESTEPS="${RL_TIMESTEPS:-50000}"
export RL_TEST_EPISODES="${RL_TEST_EPISODES:-20}"
export RL_RESULTS_DIR="$RESULTS_DIR"

source /opt/ros/humble/setup.bash
cd "$WS"

echo "[run_dqn_training] Building $PKG ..."
colcon build --packages-select "$PKG" --symlink-install
source "$WS/install/setup.bash"

echo "[run_dqn_training] Launching headless training (RL_TIMESTEPS=$RL_TIMESTEPS) ..."
: > "$LOG_FILE"
# setsid -> new process group so we can kill the whole launch tree at the end.
setsid ros2 launch serp_rl serp_rl.launch.py show_viz:=false >> "$LOG_FILE" 2>&1 &
LAUNCH_PID=$!
echo "[run_dqn_training] launch PGID=$LAUNCH_PID  (logs: $LOG_FILE)"

TIMEOUT="${RL_TIMEOUT:-7200}"
elapsed=0
while ! grep -q "=== RL_RUN_DONE ===" "$LOG_FILE" 2>/dev/null; do
  if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "[run_dqn_training] launch exited before completion - check $LOG_FILE"
    break
  fi
  sleep 5
  elapsed=$((elapsed + 5))
  if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo "[run_dqn_training] timeout after ${TIMEOUT}s"
    break
  fi
done

echo "[run_dqn_training] tearing down launch ..."
kill -INT -"$LAUNCH_PID" 2>/dev/null || true
sleep 3
kill -9 -"$LAUNCH_PID" 2>/dev/null || true

echo "[run_dqn_training] plotting reward curve ..."
python3 "$WS/src/ros2_flatland_rl_tutorial/scripts/plot_rewards.py" || true

echo "[run_dqn_training] done. Artifacts in $RESULTS_DIR"
