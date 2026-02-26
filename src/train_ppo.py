"""
PPO (Proximal Policy Optimization) Training Script for Snake RL.
Adds a policy-gradient method to compare against value-based methods (Q-Learning, DQN).
"""
import argparse
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from snake_env import SnakeEnv
from _paths import MODELS_DIR, LOGS_DIR


def train_ppo(timesteps=300000, lr=3e-4, gamma=0.99, n_steps=2048,
              batch_size=64, n_epochs=10, reward_shaping=False):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    env = Monitor(SnakeEnv(reward_shaping=reward_shaping), LOGS_DIR)
    eval_env = Monitor(SnakeEnv(reward_shaping=reward_shaping))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODELS_DIR, "best_ppo_model"),
        log_path=LOGS_DIR,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        clip_range=0.2,
        ent_coef=0.01,      # Encourage exploration
        verbose=1,
        tensorboard_log=os.path.join(LOGS_DIR, "ppo_tensorboard"),
        device=device,
    )

    print("Starting PPO training...")
    model.learn(total_timesteps=timesteps, callback=eval_callback, tb_log_name="PPO")
    model.save(os.path.join(MODELS_DIR, "ppo_snake_final"))
    print("PPO training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Snake with PPO")
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--reward_shaping", action="store_true")
    args = parser.parse_args()

    train_ppo(args.timesteps, args.lr, args.gamma,
              args.n_steps, args.batch_size, args.n_epochs, args.reward_shaping)
