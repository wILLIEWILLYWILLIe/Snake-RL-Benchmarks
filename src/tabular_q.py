import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
from snake_env import SnakeEnv
from _paths import MODELS_DIR, FIGURES_DIR

def state_to_int(state_vec):
    # Convert the 11-dimensional boolean/int array to a single integer
    # State represents: [danger_s, danger_r, danger_l, dir_l, dir_r, dir_u, dir_d, f_l, f_r, f_u, f_d]
    val = 0
    for i, bit in enumerate(state_vec):
        val += int(bit) * (2 ** i)
    return val

class TabularQ:
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
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
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error
        
        if done:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

def train(episodes=1000, lr=0.1, gamma=0.99, eps_decay=0.995, reward_shaping=False):
    env = SnakeEnv(reward_shaping=reward_shaping)
    state_size = 2 ** 11  # 2048 states
    action_size = env.action_space.n
    
    agent = TabularQ(state_size, action_size, lr=lr, gamma=gamma, epsilon_decay=eps_decay)
    
    scores = []
    
    for e in range(episodes):
        state_vec, _ = env.reset()
        state = state_to_int(state_vec)
        done = False
        score = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state_vec, reward, done, _, _ = env.step(action)
            next_state = state_to_int(next_state_vec)
            
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            if reward == 10:  # Food eaten
                score += 1
                
        scores.append(score)
        
        if (e + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode: {e+1}/{episodes}, Epsilon: {agent.epsilon:.3f}, Avg Score (last 100): {avg_score:.2f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    agent.save(os.path.join(MODELS_DIR, 'tabular_q_model.pkl'))
    
    plt.plot(scores)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Tabular Q-Learning Training')
    plt.savefig(os.path.join(FIGURES_DIR, 'tabular_q_scores.png'))
    print("Saved model and learning curve.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    parser.add_argument("--reward_shaping", action="store_true")
    args = parser.parse_args()
    
    train(args.episodes, args.lr, args.gamma, args.eps_decay, args.reward_shaping)
