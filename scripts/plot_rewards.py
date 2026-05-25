#!/usr/bin/env python3
"""Plot the DQN training reward curve from the Stable-Baselines3 Monitor CSV.

The Monitor wrapper writes one row per training episode with columns:
    r  -> episode reward
    l  -> episode length (steps)
    t  -> wall-clock time since start (s)
The first line is a '#'-prefixed JSON header that we skip.

Output: results/reward_curve.png
"""
import os
import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt

RESULTS_DIR = os.environ.get(
    "RL_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"),
)
CSV_PATH = os.path.join(RESULTS_DIR, "dqn_monitor.monitor.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "reward_curve.png")
HISTORY_PATH = os.path.join(RESULTS_DIR, "dqn_eval_history.json")
ACC_OUT_PATH = os.path.join(RESULTS_DIR, "checkpoint_accuracy.png")


def load_monitor(path):
    rewards, lengths = [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header_seen = False
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if not header_seen:  # the "r,l,t" column header
                header_seen = True
                continue
            rewards.append(float(row[0]))
            lengths.append(float(row[1]))
    return np.array(rewards), np.array(lengths)


def moving_average(x, window):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[plot_rewards] monitor csv not found at {CSV_PATH}")
        return

    rewards, lengths = load_monitor(CSV_PATH)
    if len(rewards) == 0:
        print("[plot_rewards] no episodes logged yet.")
        return

    episodes = np.arange(1, len(rewards) + 1)
    window = max(1, min(20, len(rewards) // 5))
    ma = moving_average(rewards, window)
    ma_x = np.arange(window, len(rewards) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(episodes, rewards, alpha=0.35, label="reward/episode", color="tab:blue")
    axes[0].plot(ma_x, ma, color="tab:red", linewidth=2, label=f"moving avg ({window})")
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("DQN learning curve (SERP / Flatland)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episodes, lengths, alpha=0.35, color="tab:green")
    axes[1].plot(
        np.arange(window, len(lengths) + 1),
        moving_average(lengths, window),
        color="tab:orange",
        linewidth=2,
        label=f"moving avg ({window})",
    )
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Episode length (steps)")
    axes[1].set_title("Episode length over training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120)
    print(f"[plot_rewards] saved {OUT_PATH}  ({len(rewards)} episodes)")

    plot_checkpoint_accuracy()


def plot_checkpoint_accuracy():
    """If a checkpoint-selection history exists, plot accuracy per checkpoint."""
    if not os.path.exists(HISTORY_PATH):
        return
    with open(HISTORY_PATH) as f:
        history = json.load(f)
    if not history:
        return

    # Parse the training timestep from each checkpoint filename. The trailing
    # "dqn_final" snapshot has no number; place it just past the last numbered
    # checkpoint so the curve reads left-to-right in training order.
    def digits_of(entry):
        d = "".join(c for c in entry["checkpoint"] if c.isdigit())
        return int(d) if d else None

    numbered = [d for d in (digits_of(h) for h in history) if d is not None]
    max_step = max(numbered) if numbered else 0
    gap = (sorted(numbered)[1] - sorted(numbered)[0]) if len(numbered) >= 2 else max(1, max_step)

    def steps_of(entry):
        d = digits_of(entry)
        return d if d is not None else max_step + gap  # "final" goes last

    history = sorted(history, key=steps_of)
    xs = [steps_of(h) for h in history]
    accs = [h["accuracy"] for h in history]

    best_i = int(np.argmax(accs))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, accs, "o-", color="tab:purple", label="checkpoint accuracy")
    ax.scatter([xs[best_i]], [accs[best_i]], color="red", zorder=5, s=90,
               label=f"best = {accs[best_i]:.2f} @ {xs[best_i]} steps")
    ax.set_xlabel("Training timesteps (checkpoint)")
    ax.set_ylabel("Eval accuracy (finish rate)")
    ax.set_title("DQN checkpoint accuracy — best-model selection")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ACC_OUT_PATH, dpi=120)
    print(f"[plot_rewards] saved {ACC_OUT_PATH}  ({len(history)} checkpoints)")


if __name__ == "__main__":
    main()
