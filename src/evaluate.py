"""
Evaluate any trained Snake RL agent with Pygame visualization.
Supports: tabular (Q-Learning), sarsa, dqn, ppo
"""
import argparse
import os
import time
import numpy as np
from snake_env import SnakeEnv
from tabular_q import TabularQ, state_to_int
from _paths import MODELS_DIR


def evaluate(algo="dqn", model_path="", num_games=1):
    env = SnakeEnv(render_mode="human")
    total_score = 0

    for game in range(num_games):
        obs_vec, _ = env.reset()
        done, score = False, 0

        if algo in ("dqn", "ppo"):
            if algo == "dqn":
                from stable_baselines3 import DQN
                path = model_path or os.path.join(MODELS_DIR, "best_dqn_model", "best_model.zip")
                model = DQN.load(path)
            else:
                from stable_baselines3 import PPO
                path = model_path or os.path.join(MODELS_DIR, "best_ppo_model", "best_model.zip")
                model = PPO.load(path)

            print(f"[{algo.upper()}] Loading model from {path}")
            while not done:
                action, _ = model.predict(obs_vec, deterministic=True)
                obs_vec, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if reward == 10:
                    score += 1
                time.sleep(0.05)

        elif algo in ("tabular", "sarsa"):
            if algo == "tabular":
                path = model_path or os.path.join(MODELS_DIR, "tabular_q_model.pkl")
                agent = TabularQ(2**11, env.action_space.n, epsilon=0)
            else:
                from tabular_sarsa import TabularSARSA
                path = model_path or os.path.join(MODELS_DIR, "sarsa_model.pkl")
                agent = TabularSARSA(2**11, env.action_space.n, epsilon=0)

            agent.load(path)
            print(f"[{algo.upper()}] Loading model from {path}")
            while not done:
                state = state_to_int(obs_vec)
                action = agent.choose_action(state)
                obs_vec, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if reward == 10:
                    score += 1
                time.sleep(0.05)

        total_score += score
        print(f"  Game {game+1}/{num_games} — Score: {score}")

    if num_games > 1:
        print(f"\n  Average score over {num_games} games: "
              f"{total_score/num_games:.1f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake agent")
    parser.add_argument("--algo", type=str,
                        choices=["dqn", "ppo", "tabular", "sarsa"],
                        default="tabular")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--num_games", type=int, default=1,
                        help="Number of games to play (default 1)")
    args = parser.parse_args()
    evaluate(args.algo, args.model_path, args.num_games)
