# Predictive Maintenance for Machines using Reinforcement Learning

## Overview: 

This repository contains the source code for Predictive Maintenance for Machines using Reinforcement Learning. This repository demonstrates the application of reinforcement learning (RL) to schedule predictive maintenance for laboratory equipment. The goal is to minimize machine downtime and optimize operational efficiency by predicting when maintenance is required based on sensor data.


Use Reinforcement Learning to predict and schedule maintenance for lab instruments to avoid downtime.

Environment: Laboratory equipment with sensors monitoring usage and wear.
State: Current machine condition (e.g., usage time, error rates).
Actions: Schedule maintenance or continue operation.
Rewards:
- Positive: Machine is operational.
- Negative: Unnecessary maintenance or equipment breakdowns.

## Data: Generate data for training and testing

config.yml: All variables are defined here.
DataGenerator.py: Class to generate training and test data.
EnvPredictiveMaintenance.py: Creates environment which takes action, provides reward and next state
QLearningAgent.py: Explores or exploits based on epsilon (exploration rate) in config yml. Qvalues are initialized and updated for various states in many iterations. Final q table is saved.
main_train.py: 



## Execution:
1. In order to generate new set of data, change variable value for generate_data as 1
2. run main_train to train agent
3 Run main_test to test agent and generate plots.

