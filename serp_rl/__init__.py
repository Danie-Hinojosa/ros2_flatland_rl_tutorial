#!/usr/bin/env python3
import os
import json
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher

from geometry_msgs.msg import Twist, Pose2D
from sensor_msgs.msg import LaserScan
from flatland_msgs.srv import MoveModel
from flatland_msgs.msg import Collisions

import gymnasium as gym
from gymnasium import spaces

# RL algorithm: DQN (Deep Q-Network) instead of the tutorial's default PPO.
# DQN is an off-policy, value-based algorithm that fits a discrete action
# space (here Discrete(3)), in contrast to PPO which is on-policy/policy-gradient.
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback


class SerpControllerEnv(Node, gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        Node.__init__(self, "SerpControllerEnv")
        gym.Env.__init__(self)

        # Predefined speed for the robot
        linear_speed = 0.5
        angular_speed = 1.57079632679

        # Set of actions: (linear, angular)
        self.actions = [
            (linear_speed, 0.0),     # move forward
            (0.0, angular_speed),    # rotate left
            (0.0, -angular_speed),   # rotate right
        ]

        # How close the robot needs to be to the target to finish the task
        self.end_range = 0.2

        # Number of divisions of the LiDAR
        self.n_lidar_sections = 9
        self.lidar_sample = []

        # Variables that track a possible end state
        self.distance_to_end = 10.0
        self.collision = False

        # Possible starting positions
        self.start_positions = [
            (0.0, 0.0, 1.57079632679),
            (1.6, 1.6, 3.14159265359),
        ]
        self.position = 0

        self.step_number = 0
        self.max_steps = 200
        self.previous_action = -1

        # Used for data collection during training
        self.total_step_cnt = 0
        self.total_episode_cnt = 0
        self.training = False

        # ROS publishers/subscribers
        self.pub: Publisher = self.create_publisher(Twist, "/cmd_vel", 1)

        self.create_subscription(LaserScan, "/static_laser", self.process_lidar, 1)
        self.create_subscription(LaserScan, "/end_beacon_laser", self.process_end_lidar, 1)
        self.create_subscription(Collisions, "/collisions", self.process_collisions, 1)

        # Gymnasium spaces
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0.0,
            high=2.0,
            shape=(self.n_lidar_sections,),
            dtype=np.float32,
        )

        self.state = np.zeros((self.n_lidar_sections,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Stop robot
        self.change_robot_speeds(0.0, 0.0)

        if self.total_step_cnt != 0:
            self.total_episode_cnt += 1

        # Move robot and end beacon
        start_pos = self.start_positions[self.position]
        self.position = 1 - self.position
        end_pos = self.start_positions[self.position]

        self.move_model("serp", start_pos[0], start_pos[1], start_pos[2])
        self.move_model("end_beacon", end_pos[0], end_pos[1], 0.0)

        # Reset variables
        self.lidar_sample = []
        self.wait_lidar_reading()

        # Flatland can sometimes send several collision messages
        time.sleep(0.1)

        self.distance_to_end = 10.0
        self.collision = False
        self.step_number = 0
        self.previous_action = -1

        self.state = np.array(self.lidar_sample, dtype=np.float32)
        info = {}

        return self.state, info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)

        # Perform action
        self.change_robot_speeds(self.actions[action][0], self.actions[action][1])

        self.lidar_sample = []
        self.wait_lidar_reading()
        self.change_robot_speeds(0.0, 0.0)

        self.state = np.array(self.lidar_sample, dtype=np.float32)

        self.step_number += 1
        self.total_step_cnt += 1

        terminated = False
        truncated = False
        end_state = ""

        if self.collision:
            end_state = "collision"
            reward = -200.0
            terminated = True
        elif self.distance_to_end < self.end_range:
            end_state = "finished"
            reward = 400.0 + (200 - self.step_number)
            terminated = True
        elif self.step_number >= self.max_steps:
            end_state = "timeout"
            reward = -300.0
            truncated = True
        elif action == 0:
            reward = 2.0
        else:
            reward = 0.0

        info = {"end_state": end_state}

        if (terminated or truncated) and self.training:
            self.get_logger().info(
                f"Training - Episode {self.total_episode_cnt} end state: {end_state}"
            )
            self.get_logger().info(f"Total steps: {self.total_step_cnt}")

        self.previous_action = action

        return self.state, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.change_robot_speeds(0.0, 0.0)

    def reset_counters(self):
        self.total_step_cnt = 0
        self.total_episode_cnt = 0

    def change_robot_speeds(self, linear, angular):
        twist_msg = Twist()
        twist_msg.linear.x = float(linear)
        twist_msg.angular.z = float(angular)
        self.pub.publish(twist_msg)

    def wait_lidar_reading(self, timeout_sec=2.0):
        start = time.time()
        while len(self.lidar_sample) != self.n_lidar_sections:
            if time.time() - start > timeout_sec:
                self.get_logger().warn("Timed out waiting for LiDAR reading.")
                self.lidar_sample = [2.0] * self.n_lidar_sections
                break
            time.sleep(0.001)

    def move_model(self, model_name, x, y, theta):
        client = self.create_client(MoveModel, "/move_model")
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("/move_model service not available, waiting again...")

        request = MoveModel.Request()
        request.name = model_name
        request.pose = Pose2D()
        request.pose.x = float(x)
        request.pose.y = float(y)
        request.pose.theta = float(theta)

        future = client.call_async(request)
        return future

    def process_lidar(self, data):
        self.lidar_sample = []

        rays = list(data.ranges)
        if len(rays) == 0:
            return

        clean_rays = []
        for r in rays:
            if np.isnan(r) or np.isinf(r):
                clean_rays.append(2.0)
            else:
                clean_rays.append(float(np.clip(r, 0.0, 2.0)))

        rays_per_section = max(1, len(clean_rays) // self.n_lidar_sections)

        for i in range(self.n_lidar_sections - 1):
            section = clean_rays[rays_per_section * i : rays_per_section * (i + 1)]
            self.lidar_sample.append(min(section) if section else 2.0)

        last_section = clean_rays[(self.n_lidar_sections - 1) * rays_per_section :]
        self.lidar_sample.append(min(last_section) if last_section else 2.0)

    def process_end_lidar(self, data):
        clean_data = []
        for x in data.ranges:
            if not np.isnan(x) and not np.isinf(x):
                clean_data.append(float(x))

        if not clean_data:
            return

        self.distance_to_end = min(clean_data)

    def process_collisions(self, data):
        if len(data.collisions) > 0:
            self.collision = True

    def evaluate_model(self, agent, n_episodes, label=""):
        """Run `n_episodes` deterministic episodes and return metrics.

        Returns (accuracy, mean_reward, end_states) where accuracy is the
        fraction of episodes that ended in 'finished'. Evaluation always runs
        with self.training=False so it never pollutes the training log.
        """
        was_training = self.training
        self.training = False

        successful = 0
        rewards = []
        end_states = []
        for i in range(n_episodes):
            obs, info = self.reset()
            terminated = truncated = False
            ep_reward = 0.0
            while not (terminated or truncated):
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.step(action)
                ep_reward += reward
            rewards.append(ep_reward)
            end_states.append(info.get("end_state"))
            if info.get("end_state") == "finished":
                successful += 1
            self.get_logger().info(
                f"  [eval{(' ' + label) if label else ''}] episode {i + 1}/{n_episodes}: "
                f"{info.get('end_state')} reward={ep_reward:.1f}"
            )

        self.training = was_training
        accuracy = successful / n_episodes
        return accuracy, float(np.mean(rewards)), end_states

    def run_rl_alg(self):
        # ---- Configuration (overridable via environment variables) ----
        # Total training timesteps. Use a small value (e.g. RL_TIMESTEPS=2000)
        # for a quick smoke test, larger for a real training run.
        total_timesteps = int(os.environ.get("RL_TIMESTEPS", "150000"))
        n_test_episodes = int(os.environ.get("RL_TEST_EPISODES", "20"))
        # How often (in timesteps) to snapshot a checkpoint during training.
        checkpoint_freq = int(os.environ.get("RL_CHECKPOINT_FREQ", "15000"))
        # Episodes used to score each checkpoint when picking the best one.
        ckpt_eval_episodes = int(os.environ.get("RL_CKPT_EVAL_EPISODES", "20"))
        # Random seed. Distinct seeds across parallel workers give independent
        # runs whose best checkpoints we compare to pick a global best model.
        seed_env = os.environ.get("RL_SEED")
        seed = int(seed_env) if seed_env not in (None, "") else None
        results_dir = os.environ.get(
            "RL_RESULTS_DIR",
            "/workspace/ros2_ws/src/ros2_flatland_rl_tutorial/results",
        )
        os.makedirs(results_dir, exist_ok=True)
        ckpt_dir = os.path.join(results_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Wait until at least one lidar reading exists
        self.wait_lidar_reading()

        # Check environment compatibility (Gymnasium API)
        check_env(self, warn=True)

        # Monitor wraps the env so every training episode's reward/length is
        # logged to a CSV that we later turn into the learning (reward) curve.
        monitor_path = os.path.join(results_dir, "dqn_monitor")
        monitored_env = Monitor(self, filename=monitor_path)

        # ---- Create the DQN agent ----
        # Hyper-parameters tuned for this small discrete task. DQN keeps a
        # replay buffer (off-policy) and uses epsilon-greedy exploration that
        # decays over the first `exploration_fraction` of training.
        agent = DQN(
            "MlpPolicy",
            monitored_env,
            verbose=1,
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
            exploration_final_eps=0.05,
            seed=seed,
        )

        # ---- Train (single continuous run so the exploration schedule is
        # correct over the whole budget) while snapshotting checkpoints.
        # DQN can be unstable: the *final* policy is not necessarily the best,
        # so we keep periodic checkpoints and pick the best one afterwards.
        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=ckpt_dir,
            name_prefix="dqn",
        )

        self.get_logger().info(
            f"Starting DQN training for {total_timesteps} timesteps "
            f"(checkpoint every {checkpoint_freq}, seed={seed})"
        )
        self.training = True
        self.reset_counters()

        agent.learn(total_timesteps=total_timesteps, log_interval=10, callback=checkpoint_cb)

        self.training = False

        # Always keep the final policy too, for reference.
        final_path = os.path.join(ckpt_dir, "dqn_final")
        agent.save(final_path)

        # ---- Pick the best checkpoint by evaluation accuracy ----
        # Each checkpoint is loaded and scored on ckpt_eval_episodes; the model
        # with the highest finish-rate is selected (ties broken by reward).
        candidates = sorted(
            f for f in os.listdir(ckpt_dir) if f.startswith("dqn") and f.endswith(".zip")
        )
        self.get_logger().info(f"Selecting best of {len(candidates)} checkpoints by accuracy")

        history = []
        best = {"accuracy": -1.0, "mean_reward": float("-inf"), "path": None, "name": None}
        for name in candidates:
            path = os.path.join(ckpt_dir, name[:-4])  # strip .zip for DQN.load
            model = DQN.load(path, env=monitored_env)
            acc, mean_r, _ = self.evaluate_model(model, ckpt_eval_episodes, label=name)
            history.append({"checkpoint": name, "accuracy": acc, "mean_reward": mean_r})
            self.get_logger().info(f"  checkpoint {name}: accuracy={acc:.2f} mean_reward={mean_r:.1f}")
            if (acc, mean_r) > (best["accuracy"], best["mean_reward"]):
                best = {"accuracy": acc, "mean_reward": mean_r, "path": path, "name": name}

        with open(os.path.join(results_dir, "dqn_eval_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # ---- Final evaluation of the best checkpoint (reported number) ----
        best_agent = DQN.load(best["path"], env=monitored_env)
        self.get_logger().info(
            f"Best checkpoint: {best['name']} (selection accuracy {best['accuracy']:.2f}). "
            f"Final evaluation over {n_test_episodes} episodes:"
        )
        accuracy, mean_reward, end_states = self.evaluate_model(
            best_agent, n_test_episodes, label="final"
        )

        # ---- Save the best model as the delivered model ----
        model_path = os.path.join(results_dir, "dqn_serp")
        best_agent.save(model_path)
        self.get_logger().info(f"Best model saved to {model_path}.zip")

        # ---- Persist evaluation summary ----
        successful_episodes = sum(1 for s in end_states if s == "finished")
        results = {
            "algorithm": "DQN",
            "policy": "MlpPolicy",
            "total_timesteps": total_timesteps,
            "model_selection": "best checkpoint by eval accuracy",
            "best_checkpoint": best["name"],
            "test_episodes": n_test_episodes,
            "successful_episodes": successful_episodes,
            "accuracy": accuracy,
            "mean_eval_reward": mean_reward,
            "eval_end_states": {
                "finished": end_states.count("finished"),
                "collision": end_states.count("collision"),
                "timeout": end_states.count("timeout"),
            },
        }
        with open(os.path.join(results_dir, "dqn_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        self.get_logger().info(
            f"Training Finished. Best checkpoint {best['name']}  "
            f"Accuracy: {accuracy} ({successful_episodes}/{n_test_episodes})  "
            f"mean eval reward: {mean_reward:.1f}"
        )
        # Marker used by the run script to detect completion and tear down.
        self.get_logger().info("=== RL_RUN_DONE ===")

    def destroy_node(self):
        self.change_robot_speeds(0.0, 0.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    serp = SerpControllerEnv()

    thread = threading.Thread(target=serp.run_rl_alg, daemon=True)
    thread.start()

    try:
        rclpy.spin(serp)
    except KeyboardInterrupt:
        pass
    finally:
        serp.close()
        serp.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()