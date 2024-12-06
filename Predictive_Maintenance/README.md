# Predictive Maintenance for Machines using Reinforcement Learning

## Overview

This repository contains the source code for application of reinforcement learning to schedule maintenance for a laboratory equipment. The goal is to minimize machine downtime and optimize operational efficiency by predicting when maintenance is required based on sensor data.

### Objective

The RL agent learns to:

- Predict Maintenance Needs: Decide whether to schedule maintenance or continue operation based on the machine's current state.
- Optimize Rewards:
    - Positive Rewards: Ensuring the machine remains operational without unnecessary maintenance.
    - Negative Rewards: Avoiding breakdowns or unnecessary maintenance.

### Environment Details

- State: Machine condition indicators such as operational hours, wear levels, and error rates.
- Actions: Two possible actions:
    - 0: Continue operation.
    - 1: Perform maintenance.
- Rewards:
    - High reward for keeping the machine operational without excessive maintenance.
    - Penalties for breakdowns or unnecessary maintenance.

### Components

#### Configuration

- config.yml: Defines key parameters such as data generation, environment settings, and Q-learning hyperparameters.

#### Data Generation

- DataGenerator.py: Generates synthetic datasets for training and testing based on configurable parameters such as wear level and operational hours.

#### Reinforcement Learning Framework

- EnvPredictiveMaintenance.py: Simulates the environment where the agent operates, providing states, actions, rewards, and transitions.
- QLearningAgent.py: Implements the Q-learning algorithm to train the agent:
    - Exploration vs. exploitation (epsilon-greedy policy).
    - Q-value updates using the Bellman equation.
    - Saving/loading the Q-table.

#### Training and Testing

- main_train.py:
    - Loads configurations.
    - Generates training data (if configured).
    - Trains the agent across episodes.
    - Saves the trained Q-table.
- main_test.py:
    - Loads configurations.
    - Generates/Loads test data (as configured).
    - Loads trained Q-table.
    - Evaluates the agent's performance on unseen data.
    - Generates visualizations of rewards, state progression, and action distributions.

#### Visualization
- visualization.py: Provides plotting functions to analyze the agent's performance:
    - Reward trends across episodes.
    - Action distributions (e.g., maintenance vs. operation).
    - Wear level and error rate progression.

### Execution

- Generate Data:
    - Update the generate_data parameter in config.yml to 1 to create a new dataset.
- Train the Agent:
    - Execute main_train.py to generate/load train data (as configured) and train the agent.
    - The Q-table will be saved automatically for future use.
- Test the Agent:
    - Run main_test.py to generate/load test data (as configured) and to evaluate the agent on test data.
    - Analyze the generated plots to understand the agent's performance.

### Key Metrics

- Total Rewards: Cumulative reward across all episodes.
- Action Distribution: The balance between maintenance actions and continued operations.
- Failure Rate: Percentage of machine failures during testing.
- Reward Trends: Visual trends showing agent performance over episodes.

### Future Improvements

- Deep Reinforcement Learning: Extend Q-learning to Deep Q-Networks for more complex state representations.
- Real-World Data Integration: Replace synthetic data with actual sensor readings.
- Hyperparameter Optimization: Automate tuning of RL hyperparameters.

### References

Siraskar, Rajesh, Satish Kumar, Shruti Patil, Arunkumar Bongale, and Ketan Kotecha. "Reinforcement learning for predictive maintenance: A systematic technical review." Artificial Intelligence Review 56, no. 11 (2023): 12885-12947.

https://www.researchgate.net/publication/369530421_Reinforcement_learning_for_predictive_maintenance_a_systematic_technical_review

