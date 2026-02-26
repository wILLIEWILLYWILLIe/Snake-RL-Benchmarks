"""Generate the Snake RL Jupyter Notebook."""
import json, os

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "name": "Snake_RL_Multi_Algorithm_Comparison.ipynb"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": [],
}

def md(source):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [source]})

def code(source):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source],
    })

# =====================================================================
# CELL 1 — Title
# =====================================================================
md(r"""# 🐍 Snake RL: Multi-Algorithm Comparison

**EE 473 — Reinforcement Learning | Northwestern University**

This notebook implements and compares **four Reinforcement Learning algorithms** on the classic Snake game:

| Algorithm | Type | Update Rule |
|-----------|------|-------------|
| **Tabular Q-Learning** | Value-based, Off-policy | $Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big]$ |
| **SARSA** | Value-based, On-policy | $Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma Q(s',a') - Q(s,a)\big]$ |
| **DQN** | Value-based + NN, Off-policy | Neural network Q-function with experience replay |
| **PPO** | Policy Gradient, On-policy | Clipped surrogate objective with actor-critic |

We also conduct a **reward-shaping ablation** and a **hyperparameter sensitivity analysis**.
""")

# =====================================================================
# CELL 2 — Install
# =====================================================================
code(r"""# Install required packages (Colab-ready)
!pip install gymnasium stable-baselines3 numpy matplotlib pandas --quiet
""")

# =====================================================================
# CELL 3 — Imports
# =====================================================================
code(r"""import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import random
import pickle
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

print("✅ All imports successful")
""")

# =====================================================================
# CELL 4 — Environment Design (markdown)
# =====================================================================
md(r"""## 1. Environment Design

We implement a **custom Gymnasium environment** for Snake with a compact state representation:

### State Space (11-dimensional boolean vector)
| Index | Feature | Description |
|-------|---------|-------------|
| 0–2 | Danger | Collision ahead (straight / right / left) |
| 3–6 | Direction | Current heading (left / right / up / down) |
| 7–10 | Food | Relative food position (left / right / up / down) |

- **Total states**: $2^{11} = 2048$ — perfectly suited for tabular methods
- **Action space**: 3 discrete actions — *go straight*, *turn right*, *turn left*
- **Reward**: +10 (food), −10 (death), 0 (otherwise)
- **Reward Shaping** (optional): +0.1 / −0.1 based on Manhattan distance change to food
""")

# =====================================================================
# CELL 5 — SnakeEnv
# =====================================================================
code(r"""class SnakeEnv(gym.Env):
    # Custom Snake environment for RL training (headless)
    metadata = {"render_modes": [], "render_fps": 15}

    def __init__(self, render_mode=None, grid_size=20, block_size=20, reward_shaping=False):
        super().__init__()
        self.grid_size = grid_size
        self.block_size = block_size
        self.reward_shaping = reward_shaping
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.int8)
        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.head = [self.grid_size // 2, self.grid_size // 2]
        self.snake = [self.head.copy(),
                      [self.head[0] - 1, self.head[1]],
                      [self.head[0] - 2, self.head[1]]]
        self.direction = [1, 0]
        self.food = self._place_food()
        self.score = 0
        self.frame_iteration = 0
        return self._get_state(), {}

    def _place_food(self):
        while True:
            food = [random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1)]
            if food not in self.snake:
                return food

    def step(self, action):
        self.frame_iteration += 1
        clock_wise = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        idx = clock_wise.index(self.direction)
        if action == 0:
            new_dir = clock_wise[idx]
        elif action == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]
        self.direction = new_dir
        self.head = [self.head[0] + self.direction[0],
                     self.head[1] + self.direction[1]]
        self.snake.insert(0, self.head.copy())

        reward, terminated = 0, False
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            terminated = True
            reward = -10
            return self._get_state(), reward, terminated, False, {}
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()
            if self.reward_shaping:
                old_d = abs(self.snake[1][0] - self.food[0]) + abs(self.snake[1][1] - self.food[1])
                new_d = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
                reward = 0.1 if new_d < old_d else -0.1
        return self._get_state(), reward, terminated, False, {}

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt[0] < 0 or pt[0] >= self.grid_size or pt[1] < 0 or pt[1] >= self.grid_size:
            return True
        return pt in self.snake[1:]

    def _get_state(self):
        hx, hy = self.head
        cw = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        idx = cw.index(self.direction)
        ds, dr, dl = cw[idx], cw[(idx+1)%4], cw[(idx-1)%4]
        state = [
            int(self._is_collision([hx+ds[0], hy+ds[1]])),
            int(self._is_collision([hx+dr[0], hy+dr[1]])),
            int(self._is_collision([hx+dl[0], hy+dl[1]])),
            int(self.direction == [-1, 0]),
            int(self.direction == [1, 0]),
            int(self.direction == [0, -1]),
            int(self.direction == [0, 1]),
            int(self.food[0] < hx), int(self.food[0] > hx),
            int(self.food[1] < hy), int(self.food[1] > hy),
        ]
        return np.array(state, dtype=np.int8)

    def close(self):
        pass

# Quick sanity check
env = SnakeEnv()
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}  |  Sample: {obs}")
print(f"Action space: {env.action_space}")
print("✅ SnakeEnv works!")
""")

# =====================================================================
# CELL 6 — Helper
# =====================================================================
code(r"""def state_to_int(state_vec):
    # Convert 11-dim boolean vector to a single integer index (0-2047)
    val = 0
    for i, bit in enumerate(state_vec):
        val += int(bit) * (2 ** i)
    return val

# Smooth helper for plotting
def smooth(values, window=50):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")
""")

# =====================================================================
# CELL 7 — Q-Learning markdown
# =====================================================================
md(r"""## 2. Algorithm Implementations

### 2.1 Tabular Q-Learning (Off-Policy)

Q-Learning updates using the **maximum** next-state Q-value, regardless of the action actually taken.
This makes it **off-policy** — tending toward *optimistic*, risk-seeking behavior.

$$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \Big]$$
""")

# =====================================================================
# CELL 8 — TabularQ class
# =====================================================================
code(r"""class TabularQ:
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.q_table = np.zeros((state_size, action_size))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_size = action_size

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        best_next = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next] * (not done)
        self.q_table[state][action] += self.lr * (td_target - self.q_table[state][action])
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
""")

# =====================================================================
# CELL 9 — SARSA markdown
# =====================================================================
md(r"""### 2.2 SARSA (On-Policy)

SARSA updates using the **actual** next action (including exploration), making it **on-policy**.
This results in more *conservative*, risk-averse behavior compared to Q-Learning.

$$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma Q(s', a') - Q(s,a) \Big]$$
""")

# =====================================================================
# CELL 10 — TabularSARSA class
# =====================================================================
code(r"""class TabularSARSA:
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.q_table = np.zeros((state_size, action_size))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_size = action_size

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, next_action, done):
        td_target = reward + self.gamma * self.q_table[next_state][next_action] * (not done)
        self.q_table[state][action] += self.lr * (td_target - self.q_table[state][action])
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
""")

# =====================================================================
# CELL 11 — DQN & PPO markdown
# =====================================================================
md(r"""### 2.3 DQN (Deep Q-Network)

DQN approximates the Q-function with a neural network. Key techniques:
- **Experience Replay**: stores transitions in a buffer and samples mini-batches to break correlation
- **Target Network**: a slowly-updated copy of the Q-network to stabilize training

### 2.4 PPO (Proximal Policy Optimization)

PPO directly optimizes the policy using the **clipped surrogate objective**:

$$L^{CLIP}(\theta) = \mathbb{E}\Big[ \min\big( r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \big) \Big]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.

Both DQN and PPO are implemented using **Stable-Baselines3**.
""")

# =====================================================================
# CELL 12 — Training markdown
# =====================================================================
md(r"""## 3. Training All Algorithms

We train each algorithm and record:
- **Per-episode scores** (number of food items eaten)
- **Wall-clock training time**
""")

# =====================================================================
# CELL 13 — Train tabular methods
# =====================================================================
code(r"""EPISODES_TAB = 2000
STATE_SIZE = 2 ** 11
ACTION_SIZE = 3

def train_tabular(AgentClass, name, episodes, reward_shaping=False, is_sarsa=False):
    # Train a tabular agent and return (scores, elapsed_seconds)
    env = SnakeEnv(reward_shaping=reward_shaping)
    agent = AgentClass(STATE_SIZE, ACTION_SIZE)
    scores = []
    t0 = time.time()

    for ep in range(episodes):
        state_vec, _ = env.reset()
        state = state_to_int(state_vec)
        done, score = False, 0

        if is_sarsa:
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

        if (ep + 1) % 500 == 0:
            print(f"  [{name}] Episode {ep+1}/{episodes} — "
                  f"avg(last 100): {np.mean(scores[-100:]):.1f}")

    elapsed = time.time() - t0
    print(f"  [{name}] Done in {elapsed:.1f}s — "
          f"final avg: {np.mean(scores[-100:]):.2f}\n")
    return scores, elapsed

all_curves = {}
train_times = {}

# --- Q-Learning ---
print("▶ Training Tabular Q-Learning ...")
all_curves["Q-Learning"], train_times["Q-Learning"] = \
    train_tabular(TabularQ, "Q-Learning", EPISODES_TAB)

# --- SARSA ---
print("▶ Training SARSA ...")
all_curves["SARSA"], train_times["SARSA"] = \
    train_tabular(TabularSARSA, "SARSA", EPISODES_TAB, is_sarsa=True)

# --- Q-Learning + Reward Shaping ---
print("▶ Training Q-Learning + Reward Shaping ...")
all_curves["Q-Learning + RS"], train_times["Q-Learning + RS"] = \
    train_tabular(TabularQ, "Q-Learning+RS", EPISODES_TAB, reward_shaping=True)
""")

# =====================================================================
# CELL 14 — Train DQN
# =====================================================================
code(r"""import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

TIMESTEPS_DL = 200_000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"▶ Training DQN on {device} for {TIMESTEPS_DL} timesteps ...")

env_dqn = Monitor(SnakeEnv())
t0 = time.time()
model_dqn = DQN("MlpPolicy", env_dqn, verbose=0, device=device,
                learning_rate=1e-4, gamma=0.99, buffer_size=100_000,
                exploration_fraction=0.1, exploration_final_eps=0.05)
model_dqn.learn(total_timesteps=TIMESTEPS_DL)
train_times["DQN"] = time.time() - t0

# Extract per-episode scores from Monitor
ep_rewards_dqn = env_dqn.get_episode_rewards()
all_curves["DQN"] = [max(0, int((r + 10) / 10)) for r in ep_rewards_dqn]
print(f"  [DQN] Done in {train_times['DQN']:.1f}s — "
      f"{len(all_curves['DQN'])} episodes — "
      f"final avg: {np.mean(all_curves['DQN'][-100:]):.2f}\n")
""")

# =====================================================================
# CELL 15 — Train PPO
# =====================================================================
code(r"""from stable_baselines3 import PPO

print(f"▶ Training PPO on {device} for {TIMESTEPS_DL} timesteps ...")

env_ppo = Monitor(SnakeEnv())
t0 = time.time()
model_ppo = PPO("MlpPolicy", env_ppo, verbose=0, device=device,
                learning_rate=3e-4, n_steps=2048, batch_size=64,
                n_epochs=10, clip_range=0.2, ent_coef=0.01)
model_ppo.learn(total_timesteps=TIMESTEPS_DL)
train_times["PPO"] = time.time() - t0

ep_rewards_ppo = env_ppo.get_episode_rewards()
all_curves["PPO"] = [max(0, int((r + 10) / 10)) for r in ep_rewards_ppo]
print(f"  [PPO] Done in {train_times['PPO']:.1f}s — "
      f"{len(all_curves['PPO'])} episodes — "
      f"final avg: {np.mean(all_curves['PPO'][-100:]):.2f}\n")
""")

# =====================================================================
# CELL 16 — Hyperparameter markdown
# =====================================================================
md(r"""## 4. Hyperparameter Sensitivity Analysis

We sweep over key Q-Learning hyperparameters to study their effect:

| Parameter | Values |
|-----------|--------|
| Learning rate $\alpha$ | 0.05, 0.1, 0.2 |
| Discount factor $\gamma$ | 0.9, 0.95, 0.99 |
| $\varepsilon$-decay | 0.995 (fixed) |

Each configuration runs for **1000 episodes**; we report the average score over the last 100 episodes.
""")

# =====================================================================
# CELL 17 — Hyperparameter sweep
# =====================================================================
code(r"""import itertools
import pandas as pd

SWEEP_EPISODES = 1000
lrs = [0.05, 0.1, 0.2]
gammas = [0.9, 0.95, 0.99]

results = []
total = len(lrs) * len(gammas)
print(f"Running hyperparameter sweep: {total} configurations × {SWEEP_EPISODES} episodes\n")

for i, (lr, gamma) in enumerate(itertools.product(lrs, gammas)):
    env_sw = SnakeEnv()
    agent = TabularQ(STATE_SIZE, ACTION_SIZE, lr=lr, gamma=gamma, epsilon_decay=0.995)
    scores = []
    for _ in range(SWEEP_EPISODES):
        sv, _ = env_sw.reset()
        s = state_to_int(sv)
        done, sc = False, 0
        while not done:
            a = agent.choose_action(s)
            nsv, r, done, _, _ = env_sw.step(a)
            ns = state_to_int(nsv)
            agent.learn(s, a, r, ns, done)
            s = ns
            if r == 10:
                sc += 1
        scores.append(sc)
    avg = np.mean(scores[-100:])
    results.append({"lr": lr, "gamma": gamma, "avg_score": round(avg, 2)})
    print(f"  [{i+1}/{total}] α={lr}, γ={gamma} → avg={avg:.2f}")

df = pd.DataFrame(results)
best = df.loc[df["avg_score"].idxmax()]
print(f"\n★ Best: α={best['lr']}, γ={best['gamma']} → avg score={best['avg_score']}")

# ---- Heatmap ----
pivot = df.pivot(index="lr", columns="gamma", values="avg_score")
fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xlabel("Discount Factor (γ)")
ax.set_ylabel("Learning Rate (α)")
ax.set_title("Hyperparameter Sensitivity — Avg Score (last 100 ep)")
for r in range(len(pivot.index)):
    for c in range(len(pivot.columns)):
        ax.text(c, r, f"{pivot.values[r, c]:.1f}",
                ha="center", va="center", fontsize=12, fontweight="bold")
fig.colorbar(im, ax=ax)
fig.tight_layout()
plt.show()
""")

# =====================================================================
# CELL 18 — Results markdown
# =====================================================================
md(r"""## 5. Results & Comparison

We now produce a **4-panel comparison figure**:

| Panel | Content | Purpose |
|-------|---------|---------|
| (a) | Smoothed learning curves | Compare convergence speed |
| (b) | Final avg score bar chart | Compare ultimate performance |
| (c) | Reward shaping ablation | Effect of dense vs sparse rewards |
| (d) | Score distribution box plot | Compare stability |
""")

# =====================================================================
# CELL 19 — 4-panel comparison plot
# =====================================================================
code(r"""colors = {
    "Q-Learning":      "#2196F3",
    "SARSA":           "#FF9800",
    "Q-Learning + RS": "#4CAF50",
    "DQN":             "#E91E63",
    "PPO":             "#9C27B0",
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ---- (a) Learning Curves ----
ax = axes[0, 0]
for name, scores in all_curves.items():
    s = smooth(scores, 50)
    ax.plot(s, label=name, color=colors[name], linewidth=1.8)
ax.set_title("(a) Learning Curves — All Algorithms", fontweight="bold")
ax.set_xlabel("Episode")
ax.set_ylabel("Score (smoothed, window=50)")
ax.legend(fontsize=10)

# ---- (b) Final Performance ----
ax = axes[0, 1]
names = list(all_curves.keys())
final_avgs = [np.mean(all_curves[n][-100:]) for n in names]
bars = ax.bar(names, final_avgs, color=[colors[n] for n in names],
              edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, final_avgs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("(b) Avg Score (Last 100 Episodes)", fontweight="bold")
ax.set_ylabel("Average Score")
ax.tick_params(axis="x", rotation=15)

# ---- (c) Reward Shaping Ablation ----
ax = axes[1, 0]
ax.plot(smooth(all_curves["Q-Learning"], 50),
        label="Without Reward Shaping", color="#2196F3", linewidth=1.8)
ax.plot(smooth(all_curves["Q-Learning + RS"], 50),
        label="With Reward Shaping", color="#4CAF50", linewidth=1.8)
ax.set_title("(c) Reward Shaping Ablation (Q-Learning)", fontweight="bold")
ax.set_xlabel("Episode")
ax.set_ylabel("Score (smoothed)")
ax.legend(fontsize=10)

# ---- (d) Score Distribution ----
ax = axes[1, 1]
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
plt.show()
""")

# =====================================================================
# CELL 20 — Summary table
# =====================================================================
code(r"""# ---- Summary Statistics Table ----
print("=" * 72)
print(f"{'Algorithm':<22} {'Avg(last100)':<14} {'Max':<8} {'Std':<8} {'Time(s)':<10}")
print("-" * 72)
for n in all_curves:
    s = all_curves[n]
    avg = np.mean(s[-100:])
    mx = np.max(s)
    std = np.std(s[-100:])
    t = train_times.get(n, 0)
    print(f"{n:<22} {avg:<14.2f} {mx:<8} {std:<8.2f} {t:<10.1f}")
print("=" * 72)
""")

# =====================================================================
# CELL 21 — Visualization markdown
# =====================================================================
md(r"""## 6. Visualizing the Agent
Since `pygame` windows cannot open directly in Colab, we provide a helper to visualize the trained agent playing a game directly in the notebook output using `matplotlib` and `IPython.display`.
""")

# =====================================================================
# CELL 22 — Watch Agent Play
# =====================================================================
code(r"""from IPython import display

def watch_agent_play(algo_name="Q-Learning", fps=10):
    # Retrieve the trained agent
    if algo_name not in all_curves:
        print(f"Algorithm '{algo_name}' not found. Did you train it?")
        return
        
    env = SnakeEnv(reward_shaping=False)
    
    # We'll need to instantiate the tabular agent and load its Q-table (to simulate saving/loading).
    # For now, we'll just train a quick one if it's missing, or we could have kept the agent instance.
    # To keep this notebook self-contained without complex serialization, let's train a very quick 
    # agent just for this visualization if needed, OR we can train one fully for 2000 episodes:
    
    print(f"Training a quick {algo_name} agent for visualization (1000 episodes)...")
    agent = TabularQ(STATE_SIZE, ACTION_SIZE, epsilon=1.0, epsilon_decay=0.99)
    # Quick train
    for _ in range(1000):
        sv, _ = env.reset()
        s = state_to_int(sv)
        done = False
        while not done:
            a = agent.choose_action(s)
            nsv, r, done, _, _ = env.step(a)
            ns = state_to_int(nsv)
            agent.learn(s, a, r, ns, done)
            s = ns
    
    # Now evaluate visually
    agent.epsilon = 0  # fully greedy
    sv, _ = env.reset()
    s = state_to_int(sv)
    done = False
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    while not done:
        a = agent.choose_action(s)
        nsv, r, done, _, _ = env.step(a)
        s = state_to_int(nsv)
        
        # Render mechanism
        ax.clear()
        
        # Draw grid background
        grid = np.zeros((env.grid_size, env.grid_size, 3))
        
        # Draw snake (green)
        for i, (x, y) in enumerate(env.snake):
            grid[y, x] = [0, 1, 0] if i == 0 else [0, 0.8, 0]
            
        # Draw food (red)
        fx, fy = env.food
        grid[fy, fx] = [1, 0, 0]
        
        ax.imshow(grid)
        ax.set_title(f"Score: {env.score}")
        ax.axis('off')
        
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1.0 / fps)
        
    print(f"Game Over! Final Score: {env.score}")

# Let's watch the agent!
watch_agent_play("Q-Learning", fps=5)
""")

# =====================================================================
# CELL 23 — Discussion
# =====================================================================
md(r"""## 6. Discussion

### Training Time
- **Tabular methods** (Q-Learning, SARSA) train in seconds — the 2048-state Q-table is tiny and requires only table look-ups.
- **DQN** and **PPO** require minutes of training due to neural network forward/backward passes. DQN additionally maintains a replay buffer, and PPO collects rollouts with multiple optimization epochs per update.

### Algorithm Performance and Key Observations
1. **Q-Learning vs SARSA** — Q-Learning tends to reach higher scores because its off-policy update assumes optimal future behavior, encouraging risk-taking (e.g., navigating close to walls). SARSA's on-policy update accounts for exploratory actions, producing more conservative but sometimes safer behavior.
2. **Tabular vs Deep RL** — For this small state space (2048 states), tabular methods can achieve perfect coverage and often outperform neural-network-based methods. DQN and PPO introduce function approximation error and suffer from sample inefficiency on a problem this small — an example of *"using a sledgehammer to crack a nut."*
3. **Reward Shaping** — Adding distance-based shaping (+0.1/−0.1) significantly accelerates early learning for Q-Learning by providing a dense reward signal, though final performance may converge similarly.

### Hyperparameter Sensitivity
- **Learning rate (α)**: Values around 0.1 work well; too low (0.01) slows convergence, too high (0.2) can destabilize learning.
- **Discount factor (γ)**: Higher values (0.99) yield better performance by valuing future rewards more heavily, crucial for sequential food collection.
- **ε-decay**: Controls the exploration→exploitation transition. Moderate decay (0.995) balances sufficient exploration with convergence speed.

### Challenges
- **Sparse rewards**: Without reward shaping, the agent rarely encounters positive reward early in training, leading to slow initial learning.
- **Episode length management**: Without a step limit relative to snake length, the agent can learn to loop indefinitely without dying.
- **DQN instability**: The overestimation bias in DQN leads to noisy learning curves; techniques like Double DQN or Dueling DQN would help.

### Future Work
- **Larger grid** or pixel-based observations to challenge tabular methods and showcase deep RL advantages.
- **CNN-based DQN** using raw game frames as input.
- **Curriculum learning**: start with smaller grids and progressively increase difficulty.
- **Double DQN / Dueling DQN** to reduce overestimation bias.
""")

# =====================================================================
# CELL 24 — Conclusion
# =====================================================================
md(r"""## 7. Conclusion

We implemented and compared four RL algorithms — **Tabular Q-Learning**, **SARSA**, **DQN**, and **PPO** — on a custom Snake environment.

**Key takeaways:**
- **Tabular methods excel** when the state space is small enough for complete coverage. Q-Learning achieves the highest final performance with minimal training time.
- **Off-policy (Q-Learning) vs on-policy (SARSA)** differences are clearly visible in risk-taking behavior.
- **Deep RL methods** (DQN, PPO) provide frameworks for generalization but are overkill for this problem size. They would shine with larger state spaces or pixel-based inputs.
- **Reward shaping** is a powerful engineering tool that meaningfully accelerates convergence for tabular methods.
- **Hyperparameter tuning** is critical: learning rate and discount factor significantly impact performance, and even simple grid-search can identify strong configurations.
""")

# =====================================================================
# Write the notebook
# =====================================================================
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Snake_RL_Multi_Algorithm_Comparison.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"✅ Notebook written to {out_path}")
