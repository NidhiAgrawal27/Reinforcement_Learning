import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self, state_size, action_size, q_agent_params):
        self.state_size = state_size  # Number of state features
        self.action_size = action_size  # Number of actions
        self.q_table = {}  # Q-table: {(state): [Q(action 1), ..., Q(action N)]}
        self.alpha = q_agent_params['learning_rate']  # Learning rate
        self.gamma = q_agent_params['discount_factor']  # Discount factor
        self.epsilon = q_agent_params['epsilon']  # Exploration rate
        self.epsilon_decay = q_agent_params['epsilon_decay']  # Decay rate for exploration
        self.min_epsilon = q_agent_params['min_epsilon']  # Minimum exploration rate

    def get_state_key(self, state):
        """Convert state to a hashable key for Q-table."""
        return tuple(state.round(1))  # Round values for discretization

    def act(self, state):
        """Choose an action using an ϵ-greedy policy."""
        state_key = self.get_state_key(state)
        if random.uniform(0, 1) < self.epsilon:  # Exploration
            return random.randint(0, self.action_size - 1)
        else:  # Exploitation
            return np.argmax(self.q_table.get(state_key, [0] * self.action_size)) # default value of [0, 0] if state_key not in q_table

    def update(self, state, action, reward, next_state, done):
        """Update Q-value using the Bellman equation."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Initialize Q-values for unseen states
        if state_key not in self.q_table:
            self.q_table[state_key] = [0] * self.action_size
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0] * self.action_size

        # Q-learning update rule
        target = reward
        if not done:
            target += self.gamma * max(self.q_table[next_state_key]) # gamma is the discount factor, representing the importance of future rewards
        self.q_table[state_key][action] += self.alpha * (target - self.q_table[state_key][action]) # alpha value: controls how much the Q-value is adjusted

        # Decay epsilon to gradually shift the agent from exploring (choosing random actions) to exploiting (choosing actions based on the Q-table)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filename):
        """Save the Q-table to a file."""
        with open(filename, 'wb') as file:
            pickle.dump(self.q_table, file)
        print(f"Q-table saved to {filename}.")

    def load_q_table(self, filename):
        """Load the Q-table from a file."""
        with open(filename, 'rb') as file:
            self.q_table = pickle.load(file)
        print(f"Q-table loaded from {filename}.")
