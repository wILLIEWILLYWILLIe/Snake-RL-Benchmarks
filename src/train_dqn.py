import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import os
from snake_env import SnakeEnv
from _paths import MODELS_DIR, LOGS_DIR

def train_dqn(timesteps=100000, lr=1e-4, gamma=0.99, buffer_size=100000, reward_shaping=False):
    # Setup MPS if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    env = SnakeEnv(reward_shaping=reward_shaping)
    env = Monitor(env, LOGS_DIR)  # For tensorboard and evaluation
    
    eval_env = SnakeEnv(reward_shaping=reward_shaping)
    eval_env = Monitor(eval_env)
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=os.path.join(MODELS_DIR, 'best_dqn_model'),
        log_path=LOGS_DIR, 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=lr,
        gamma=gamma,
        buffer_size=buffer_size,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=os.path.join(LOGS_DIR, "dqn_tensorboard"),
        device=device
    )
    
    print("Starting training...")
    model.learn(total_timesteps=timesteps, callback=eval_callback, tb_log_name="DQN")
    model.save(os.path.join(MODELS_DIR, "dqn_snake_final"))
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--reward_shaping", action="store_true")
    args = parser.parse_args()
    
    train_dqn(args.timesteps, args.lr, args.gamma, args.buffer_size, args.reward_shaping)
