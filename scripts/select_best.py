#!/usr/bin/env python3
"""Select the global best model across parallel DQN workers.

Each worker wrote a results dir (results/parallel/run_i/) containing its own
best-checkpoint model (dqn_serp.zip) and dqn_results.json. This script reads
every worker's summary, picks the one with the highest eval accuracy (ties
broken by mean reward), and promotes that worker's artifacts to the top-level
results/ folder so it becomes the delivered model. It also writes a comparison
summary and regenerates the plots for the winning run.
"""
import os
import json
import glob
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
PAR_DIR = os.path.join(RESULTS_DIR, "parallel")

PROMOTE_FILES = [
    "dqn_serp.zip",
    "dqn_results.json",
    "dqn_monitor.monitor.csv",
    "dqn_eval_history.json",
    "training_log.txt",
]


def load_runs():
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(PAR_DIR, "run_*"))):
        summary_path = os.path.join(run_dir, "dqn_results.json")
        if not os.path.exists(summary_path):
            print(f"[select_best] {run_dir}: no dqn_results.json (run failed?) — skipping")
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        runs.append((run_dir, summary))
    return runs


def main():
    runs = load_runs()
    if not runs:
        print("[select_best] no completed worker runs found.")
        return

    comparison = []
    best = None
    for run_dir, s in runs:
        worker = os.path.basename(run_dir)
        acc = s.get("accuracy", -1.0)
        mean_r = s.get("mean_eval_reward", float("-inf"))
        comparison.append({
            "worker": worker,
            "accuracy": acc,
            "mean_eval_reward": mean_r,
            "best_checkpoint": s.get("best_checkpoint"),
            "successful_episodes": s.get("successful_episodes"),
            "test_episodes": s.get("test_episodes"),
        })
        print(f"[select_best] {worker}: accuracy={acc:.3f} mean_reward={mean_r:.1f} "
              f"({s.get('best_checkpoint')})")
        key = (acc, mean_r)
        if best is None or key > (best[1].get("accuracy", -1.0),
                                  best[1].get("mean_eval_reward", float("-inf"))):
            best = (run_dir, s)

    best_dir, best_summary = best
    best_worker = os.path.basename(best_dir)
    print(f"[select_best] GLOBAL BEST = {best_worker}  accuracy={best_summary.get('accuracy'):.3f}")

    # Promote winning artifacts to the top-level results/ dir.
    for fname in PROMOTE_FILES:
        src = os.path.join(best_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(RESULTS_DIR, fname))

    # Record which worker won and the full comparison.
    comparison_sorted = sorted(comparison, key=lambda c: (c["accuracy"], c["mean_eval_reward"]),
                               reverse=True)
    with open(os.path.join(RESULTS_DIR, "parallel_comparison.json"), "w") as f:
        json.dump({
            "n_workers": len(runs),
            "winner": best_worker,
            "winner_accuracy": best_summary.get("accuracy"),
            "runs": comparison_sorted,
        }, f, indent=2)

    # Regenerate plots for the promoted (winning) run.
    env = dict(os.environ, RL_RESULTS_DIR=os.path.abspath(RESULTS_DIR))
    subprocess.run(["python3", os.path.join(HERE, "plot_rewards.py")], env=env, check=False)

    print(f"[select_best] promoted {best_worker} -> {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
