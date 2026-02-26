"""
Automated hyperparameter sweep for Tabular Q-Learning.
Sweeps over learning rate (α), discount factor (γ), and epsilon decay rate,
as specified in the project proposal.  Results are saved as a CSV and a
heat-map style summary plot.
"""
import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from snake_env import SnakeEnv
from tabular_q import TabularQ, state_to_int
from _paths import ROOT, FIGURES_DIR


def run_single(lr, gamma, eps_decay, episodes, reward_shaping):
    env = SnakeEnv(reward_shaping=reward_shaping)
    agent = TabularQ(2 ** 11, env.action_space.n,
                     lr=lr, gamma=gamma, epsilon_decay=eps_decay)
    scores = []
    for _ in range(episodes):
        state_vec, _ = env.reset()
        state = state_to_int(state_vec)
        done, score = False, 0
        while not done:
            action = agent.choose_action(state)
            ns_vec, reward, done, _, _ = env.step(action)
            next_state = state_to_int(ns_vec)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            if reward == 10:
                score += 1
        scores.append(score)
    last_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    return last_100, scores


def sweep(episodes=1000, reward_shaping=False):
    lrs = [0.01, 0.05, 0.1, 0.2]
    gammas = [0.9, 0.95, 0.99]
    eps_decays = [0.99, 0.995, 0.999]

    results = []
    total = len(lrs) * len(gammas) * len(eps_decays)
    print(f"Starting hyperparameter sweep: {total} combinations × {episodes} episodes each")
    print("=" * 60)

    for i, (lr, gamma, eps_decay) in enumerate(itertools.product(lrs, gammas, eps_decays)):
        avg, _ = run_single(lr, gamma, eps_decay, episodes, reward_shaping)
        results.append({
            "lr": lr, "gamma": gamma, "eps_decay": eps_decay,
            "avg_score_last_100": round(avg, 2)
        })
        print(f"[{i+1}/{total}] lr={lr}, γ={gamma}, ε_decay={eps_decay} -> avg={avg:.2f}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(ROOT, "results", "hyperparameter_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ---- Best result ----
    best = df.loc[df["avg_score_last_100"].idxmax()]
    print(f"\n★ Best: lr={best['lr']}, γ={best['gamma']}, "
          f"ε_decay={best['eps_decay']} → avg score={best['avg_score_last_100']}")

    # ---- Heatmap: lr vs gamma (best eps_decay) ----
    best_eps = best["eps_decay"]
    subset = df[df["eps_decay"] == best_eps].pivot(
        index="lr", columns="gamma", values="avg_score_last_100"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(subset.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(subset.columns)))
    ax.set_xticklabels(subset.columns)
    ax.set_yticks(range(len(subset.index)))
    ax.set_yticklabels(subset.index)
    ax.set_xlabel("Discount Factor (γ)")
    ax.set_ylabel("Learning Rate (α)")
    ax.set_title(f"Avg Score (last 100 ep) — ε_decay={best_eps}")

    for i in range(len(subset.index)):
        for j in range(len(subset.columns)):
            ax.text(j, i, f"{subset.values[i, j]:.1f}",
                    ha="center", va="center", fontsize=11)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    heatmap_path = os.path.join(FIGURES_DIR, "hyperparameter_heatmap.png")
    fig.savefig(heatmap_path, dpi=150)
    print(f"Heatmap saved to {heatmap_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for Tabular Q")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Episodes per experiment (default 1000)")
    parser.add_argument("--reward_shaping", action="store_true")
    args = parser.parse_args()
    sweep(args.episodes, args.reward_shaping)
