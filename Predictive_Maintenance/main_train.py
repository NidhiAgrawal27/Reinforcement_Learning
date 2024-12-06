import pandas as pd
import pathlib
import yaml
from DataGenerator import DataGenerator
from EnvPredictiveMaintenance import EnvPredictiveMaintenance
from QLearningAgent import QLearningAgent
from visualization import *

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':

    config = load_config('config.yml')
    data_params = config['data_parameters']
    env_params = config['env_params']
    q_agent_params = config['q_agent_params']
    dir_params = config['dir_params']
    filenames = config['filenames']
    train_params = config['train_params']
    # test_params = config['test_params']

    train_file_name = f"{dir_params['data_dir']}{filenames['train_file_name']}"
    test_file_name = f"{dir_params['data_dir']}{filenames['test_file_name']}"

    pathlib.Path(dir_params['data_dir']).mkdir(parents=True, exist_ok=True)
    pathlib.Path(dir_params['logs_dir']).mkdir(parents=True, exist_ok=True)

    # ***** Generate and save data *****
    if data_params['generate_data'] == 1:
        train_data_gen = DataGenerator()
        test_data_gen = DataGenerator()
        train_data_gen.generate_samples(
                                    num_samples=data_params['num_train_samples'], 
                                    months_since_last_maintenance = data_params['months_since_last_maintenance'],
                                    max_operational_hours = data_params['max_operational_hours'],
                                    max_wear_level = data_params['max_wear_level'],
                                    max_error_rate = data_params['max_error_rate'], 
                                    savefilename = train_file_name)

    # ******************** TRAIN ********************

    initial_states_df = pd.read_csv(train_file_name)
    env = EnvPredictiveMaintenance(initial_states_df=initial_states_df, env_params=env_params)

    state_size = len(env.reset())  # State vector length
    action_size = train_params['action_size']
    agent = QLearningAgent(state_size, action_size, q_agent_params)

    episodes = initial_states_df.shape[0] # train_params['episodes']  # Number of episodes for training
    max_steps = train_params['max_steps']  # Max steps per episode

    total_rewards = []
    for episode in range(episodes):

        if episode % 4 == 0: randomness = True # Random reset
        else: randomness = False # Sequential reset

        state = env.reset(randomness)
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    # plot_rewards(total_rewards, f"{dir_params['logs_dir']}train_rewards.png")

    agent.save_q_table(f"{dir_params['logs_dir']}q_table.pkl")
    print("Training complete.")
