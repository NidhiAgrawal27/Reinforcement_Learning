import numpy as np
import random

class EnvPredictiveMaintenance:
    def __init__(self, initial_states_df, env_params):
        # Store the DataFrame with initial states
        self.initial_states_df = initial_states_df
        self.env_params = env_params
        self.max_wear_level = self.env_params['max_wear_level']  # Maximum wear before failure
        self.max_error_rate = self.env_params['max_error_rate']  # Maximum error rate before failure
        self.max_op_hrs = self.env_params['max_op_hrs']  # Maximum operational hours before failure
        self.maintenance_cost = self.env_params['maintenance_cost']  # Penalty for maintenance
        self.failure_cost = self.env_params['failure_cost']  # Penalty for failure
        self.operation_reward = self.env_params['operation_reward']  # Reward for successful operation
        self.correct_maintenance_reward = self.env_params['correct_maintenance_reward']  # Reward for maintenance at the right time
        self.incorrect_operation_penalty = self.env_params['incorrect_operation_penalty']  # Penalty for maintenance at the wrong time
        self.maintain_every_n_months = self.env_params['maintain_every_n_months'] # Ideal maintenance interval
        self.state_index = 0
        self.maintenance_instances = 0
        self.operation_instances = 0

    def reset(self, randomness = False):
        """Reset the environment to a random initial state sampled from the DataFrame."""
        if randomness == True: 
            initial_state = self.initial_states_df.sample(n=1).iloc[0]
        else: 
            initial_state = self.initial_states_df.iloc[self.state_index]
            self.state_index += 1
            if self.state_index >= len(self.initial_states_df):
                self.state_index = 0  # Reset to the start if all states have been used
        self.state = {
            "Time Since Last Maintenance(in months)": initial_state["Time Since Last Maintenance(in months)"],
            "Operational Hours": initial_state["Operational Hours"],
            "Wear Level": initial_state["Wear Level"],
            "Error Rate": initial_state["Error Rate"]
        }
        return self._get_state_vector()

    def _get_state_vector(self):
        """Return the state as a vector."""
        return np.array([
            self.state["Time Since Last Maintenance(in months)"],
            self.state["Operational Hours"],
            self.state["Wear Level"],
            self.state["Error Rate"]
        ])

    def step(self, action):
        """Apply an action and return the next state, reward, and done flag."""
        wear = self.state["Wear Level"]
        error_rate = self.state["Error Rate"]

        # Ranges to reduce after maintenance
        reduce_wear_level = tuple(self.env_params['reduce_wear_level'])
        reduce_error_rate = tuple(self.env_params['reduce_error_rate'])
        
        # Ranges to increase after operation
        add_op_hrs = tuple(self.env_params['add_op_hrs'])
        add_wear_level = tuple(self.env_params['add_wear_level'])
        add_error_rate = tuple(self.env_params['add_error_rate'])

        done = False
        if action == 1:  # Perform maintenance
            self.state["Wear Level"] = max(0, wear - random.randint(*reduce_wear_level)) # reduce wear level after maintenance
            self.state["Error Rate"] = max(0, error_rate - random.randint(*reduce_error_rate)) # reduce error rate after maintenance
            if self.state["Time Since Last Maintenance(in months)"] == self.maintain_every_n_months: 
                reward = self.correct_maintenance_reward
            elif self.maintenance_instances > self.operation_instances:
                reward = 5*self.maintenance_cost
            else: 
                reward = self.maintenance_cost
            self.maintenance_instances += 1
            self.state["Time Since Last Maintenance(in months)"] = 0
        else:  # Continue operation
            self.state["Operational Hours"] += random.randint(*add_op_hrs)
            self.state["Wear Level"] = min(self.max_wear_level, wear + random.randint(*add_wear_level))
            self.state["Error Rate"] = min(self.max_error_rate, error_rate + random.randint(*add_error_rate))
            if self.state["Wear Level"] >= self.max_wear_level or self.state["Error Rate"] >= self.max_error_rate or self.state["Operational Hours"] >= self.max_op_hrs:
                reward = self.failure_cost
                done = True
            elif self.state["Time Since Last Maintenance(in months)"] == self.maintain_every_n_months: 
                reward = self.incorrect_operation_penalty
            else:
                reward = self.operation_reward
                self.operation_instances += 1
            self.state["Time Since Last Maintenance(in months)"] += 1

        return self._get_state_vector(), reward, done


    def render(self):
        """Print the current state."""
        print(f"State: {self.state}")
