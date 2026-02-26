"""
Multi-algorithm comparison script.
Trains all four algorithms (Tabular Q, SARSA, DQN, PPO) and produces a
publication-quality figure with:
  1. Learning curves (score vs episode/timestep)  ← each algorithm
  2. Reward-shaping ablation (with vs without)
  3. Summary statistics table printed to console

This is the centerpiece plot for the final report.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from snake_env import SnakeEnv
from tabular_q import TabularQ, state_to_int
from tabular_sarsa import TabularSARSA
from _paths import FIGURES_DIR

matplotlib.rcParams.update({
    "font.size": 12,
    "figure.figsize": (14, 10),
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# --------------- helpers ---------------
def smooth(values, window=50):
    """Simple moving average for smooth learning curves."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def train_tabular(AgentClass, name, episodes, reward_shaping):
    """Train a tabular agent (Q or SARSA) and return per-episode scores."""
    env = SnakeEnv(reward_shaping=reward_shaping)
    agent = AgentClass(2 ** 11, env.action_space.n)
    scores = []

    for ep in range(episodes):
        state_vec, _ = env.reset()
        state = state_to_int(state_vec)
        done, score = False, 0

        if isinstance(agent, TabularSARSA):
            action = agent.choose_action(state)
            while not done:
                ns_vec, reward, done, _, _ = env.step(action)
                ns = state_to_int(ns_vec)
                na = agent.choose_action(ns)
                agent.learn(state, action, reward, ns, na, done)
                state, action = ns, na
                if reward == 10:
                    score += 1
        else:
            while not done:
                action = agent.choose_action(state)
                ns_vec, reward, done, _, _ = env.step(action)
                ns = state_to_int(ns_vec)
                agent.learn(state, action, reward, ns, done)
                state = ns
                if reward == 10:
                    score += 1

        scores.append(score)

        if (ep + 1) % 200 == 0:
            print(f"  [{name}] Episode {ep+1}/{episodes} — avg(last 100): "
                  f"{np.mean(scores[-100:]):.1f}")

    return scores


def train_sb3(AlgoClass, name, timesteps, reward_shaping):
    """Train an SB3 agent and return per-episode scores via Monitor."""
    import torch
    from stable_baselines3.common.monitor import Monitor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    env = Monitor(SnakeEnv(reward_shaping=reward_shaping))

    model = AlgoClass("MlpPolicy", env, verbose=0, device=device)
    print(f"  [{name}] Training for {timesteps} timesteps on {device} ...")
    model.learn(total_timesteps=timesteps)

    # Extract episode rewards from the Monitor wrapper
    ep_rewards = env.get_episode_rewards()
    # Convert cumulative reward to approximate score (food count)
    scores = [max(0, int((r + 10) / 10)) for r in ep_rewards]
    return scores


# --------------- main ---------------
def main():
    EPISODES_TAB = 2000      # for tabular methods
    TIMESTEPS_DL = 200000    # for DQN / PPO

    os.makedirs(FIGURES_DIR, exist_ok=True)

    all_curves = {}

    # 1. Tabular Q-Learning
    print("▶ Training Tabular Q-Learning ...")
    all_curves["Q-Learning"] = train_tabular(TabularQ, "Q-Learning",
                                             EPISODES_TAB, False)

    # 2. SARSA
    print("▶ Training SARSA ...")
    all_curves["SARSA"] = train_tabular(TabularSARSA, "SARSA",
                                        EPISODES_TAB, False)

    # 3. Tabular Q-Learning WITH reward shaping
    print("▶ Training Q-Learning + Reward Shaping ...")
    all_curves["Q-Learning + RS"] = train_tabular(TabularQ, "Q-Learning+RS",
                                                   EPISODES_TAB, True)

    # 4. DQN
    print("▶ Training DQN ...")
    from stable_baselines3 import DQN
    all_curves["DQN"] = train_sb3(DQN, "DQN", TIMESTEPS_DL, False)

    # 5. PPO
    print("▶ Training PPO ...")
    from stable_baselines3 import PPO
    all_curves["PPO"] = train_sb3(PPO, "PPO", TIMESTEPS_DL, False)

    # ================ Plot ================
    colors = {
        "Q-Learning":       "#2196F3",
        "SARSA":            "#FF9800",
        "Q-Learning + RS":  "#4CAF50",
        "DQN":              "#E91E63",
        "PPO":              "#9C27B0",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ---- (a) All learning curves (smoothed) ----
    ax = axes[0, 0]
    for name, scores in all_curves.items():
        s = smooth(scores, 50)
        ax.plot(s, label=name, color=colors[name], linewidth=1.8)
    ax.set_title("(a) Learning Curves — All Algorithms", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score (smoothed, window=50)")
    ax.legend(fontsize=10)

    # ---- (b) Bar chart: final performance ----
    ax = axes[0, 1]
    names = list(all_curves.keys())
    final_avgs = [np.mean(all_curves[n][-100:]) for n in names]
    bars = ax.bar(names, final_avgs, color=[colors[n] for n in names],
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, final_avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("(b) Avg Score (Last 100 Episodes)", fontweight="bold")
    ax.set_ylabel("Average Score")
    ax.tick_params(axis="x", rotation=15)

    # ---- (c) Reward Shaping Ablation (Q-Learning) ----
    ax = axes[1, 0]
    s_no = smooth(all_curves["Q-Learning"], 50)
    s_rs = smooth(all_curves["Q-Learning + RS"], 50)
    ax.plot(s_no, label="Without Reward Shaping", color="#2196F3", linewidth=1.8)
    ax.plot(s_rs, label="With Reward Shaping",    color="#4CAF50", linewidth=1.8)
    ax.set_title("(c) Reward Shaping Ablation (Q-Learning)", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score (smoothed)")
    ax.legend(fontsize=10)

    # ---- (d) Score distribution box plot ----
    ax = axes[1, 1]
    # Use the last 200 episodes for each
    data = [all_curves[n][-200:] for n in names]
    bp = ax.boxplot(data, labels=names, patch_artist=True)
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.7)
    ax.set_title("(d) Score Distribution (Last 200 Episodes)", fontweight="bold")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Snake RL — Multi-Algorithm Comparison",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGURES_DIR, "algorithm_comparison.png"), dpi=200)
    print(f"\n✅ Comparison figure saved to {os.path.join(FIGURES_DIR, 'algorithm_comparison.png')}")

    # ================ Summary Table ================
    print("\n" + "=" * 65)
    print(f"{'Algorithm':<22} {'Avg(last 100)':<16} {'Max':<8} {'Std':<8}")
    print("-" * 65)
    for n in names:
        s = all_curves[n]
        avg = np.mean(s[-100:])
        mx = np.max(s)
        std = np.std(s[-100:])
        print(f"{n:<22} {avg:<16.2f} {mx:<8} {std:<8.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
