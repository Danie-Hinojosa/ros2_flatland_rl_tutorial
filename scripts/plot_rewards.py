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


if __name__ == "__main__":
    main()
