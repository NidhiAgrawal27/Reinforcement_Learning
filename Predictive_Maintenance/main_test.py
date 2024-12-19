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

    test_file_name = f"{dir_params['data_dir']}{filenames['test_file_name']}"

    pathlib.Path(dir_params['data_dir']).mkdir(parents=True, exist_ok=True)
    pathlib.Path(dir_params['logs_dir']).mkdir(parents=True, exist_ok=True)

    # ***** Generate and save data *****
    if data_params['generate_data'] == 1:
        test_data_gen = DataGenerator()
        test_data_gen.generate_samples(
                                    num_samples=data_params['num_test_samples'], 
                                    months_since_last_maintenance = data_params['months_since_last_maintenance'],
                                    max_operational_hours = data_params['max_operational_hours'],
                                    max_wear_level = data_params['max_wear_level'],
                                    max_error_rate = data_params['max_error_rate'],
                                    savefilename = test_file_name)

    test_states_df = pd.read_csv(test_file_name)
    env = EnvPredictiveMaintenance(initial_states_df=test_states_df, env_params=env_params)

    state_size = len(env.reset())  # State vector length
    action_size = train_params['action_size']

    agent = QLearningAgent(state_size, action_size, q_agent_params)
    agent.load_q_table(f"{dir_params['logs_dir']}q_table.pkl")

    num_episodes = test_states_df.shape[0]
    num_steps = env_params['max_months_for_failure']
    
    action_dict, steps_dict, failed_dict, state_dict = {}, {}, {}, {}
    total_rewards, wear_levels, error_rates = [], [], []
    failure_count = 0
    count_total_steps = 0

    for episode in range(num_episodes):
        
        action_list, step_list, failed_list = [], [], []
        state_dict[episode]= {}

        state = env.reset()
        
        state_dict[episode]['time_since_last_maintenance_in_months'] = state[0]
        state_dict[episode]['operational_hours'] = state[1]
        state_dict[episode]['wear_level'] = state[2]
        state_dict[episode]['error_rate'] = state[3]

        total_reward = 0
        failed = 0

        for step in range(state[0], num_steps):
            count_total_steps += 1
            if failed == 2: action = 1
            else: action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            wear_levels.append(env.state["Wear Level"])
            error_rates.append(env.state["Error Rate"])
            
            state = next_state
            step_list.append(step)
            action_list.append(action)

            if done:
                if reward == env.failure_cost:  # Check for failure
                    failure_count += 1
                    failed = 2 # to plot on the graph
                    # break
            else: failed = 0
            failed_list.append(failed)

        action_dict[episode] = action_list
        steps_dict[episode] = step_list
        failed_dict[episode] = failed_list
        total_rewards.append(total_reward)

    # Metrics
    total_num_maintenance = sum([action_list.count(1) for action_list in action_dict.values()])
    total_maintenance_cost = total_num_maintenance * env.maintenance_cost
    total_num_operations = sum([action_list.count(0) for action_list in action_dict.values()])
    print(f"Num of Test data: {num_episodes}")
    print(f"Total Num of Maintenance: {total_num_maintenance}, Total Maintenance Cost: {total_maintenance_cost}")
    print(f"Total Num of Operations: {total_num_operations}")
    print(f"Total rewards: {sum(total_rewards)}")
    print(f"Average Reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"Failure Count: {failure_count}, Num of total steps: {count_total_steps}")
    print(f"Failure Rate: {(failure_count / count_total_steps) * 100:.2f}%")

    # Visualizations
    labels = ['Continue Operation', 'Perform Maintenance', 'Failure']
    action_counts = [total_num_operations, total_num_maintenance, failure_count]
    colours = ['blue', 'orange', 'green']
    plot_action_distribution(action_counts, labels, colours, f"{dir_params['logs_dir']}action_distribution.png")
    plot_rewards(total_rewards, f"{dir_params['logs_dir']}rewards.png")
    plot_state_progression(wear_levels, error_rates, f"{dir_params['logs_dir']}state_progression.png")

    ytick_labels = ['Operational', 'Maintenance', 'Failure']
    plot_actions(action_dict, steps_dict, failed_dict, state_dict, ytick_labels,f"{dir_params['logs_dir']}episode_actions.png")
