"""
SARSA (State-Action-Reward-State-Action) — an on-policy tabular method.
Comparing on-policy SARSA with off-policy Q-Learning highlights their
different exploration / exploitation characteristics.
"""
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
from snake_env import SnakeEnv
from tabular_q import state_to_int
from _paths import MODELS_DIR, FIGURES_DIR


class TabularSARSA:
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
        # SARSA update: uses the ACTUAL next action (on-policy)
        td_target = reward + self.gamma * self.q_table[next_state][next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)


def train(episodes=2000, lr=0.1, gamma=0.99, eps_decay=0.995,
          reward_shaping=False):
    env = SnakeEnv(reward_shaping=reward_shaping)
    state_size = 2 ** 11
    action_size = env.action_space.n
    agent = TabularSARSA(state_size, action_size, lr=lr, gamma=gamma,
                         epsilon_decay=eps_decay)
    scores = []

    for e in range(episodes):
        state_vec, _ = env.reset()
        state = state_to_int(state_vec)
        action = agent.choose_action(state)
        done = False
        score = 0

        while not done:
            next_state_vec, reward, done, _, _ = env.step(action)
            next_state = state_to_int(next_state_vec)
            next_action = agent.choose_action(next_state)

            agent.learn(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            if reward == 10:
                score += 1

        scores.append(score)

        if (e + 1) % 100 == 0:
            avg = np.mean(scores[-100:])
            print(f"Episode: {e+1}/{episodes}, Epsilon: {agent.epsilon:.3f}, "
                  f"Avg Score (last 100): {avg:.2f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    agent.save(os.path.join(MODELS_DIR, "sarsa_model.pkl"))

    plt.figure()
    plt.plot(scores)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.title("SARSA Training")
    plt.savefig(os.path.join(FIGURES_DIR, "sarsa_scores.png"))
    print("Saved SARSA model and learning curve.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Snake with SARSA")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    parser.add_argument("--reward_shaping", action="store_true")
    args = parser.parse_args()

    train(args.episodes, args.lr, args.gamma, args.eps_decay,
          args.reward_shaping)
