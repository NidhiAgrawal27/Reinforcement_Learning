import matplotlib.pyplot as plt

# Plot the cumulative reward for each episode during testing to see trends.
def plot_rewards(rewards, savefilename):
    # plt.scatter(range(1,len(rewards)+1), rewards, label="Total Reward")
    plt.plot(rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total Reward Over Episodes")
    plt.legend()
    plt.savefig(savefilename)
    plt.close()

# Visualize the distribution of actions (e.g., maintenance vs. continue operation) to understand the agent's decision-making strategy.
def plot_action_distribution(action_counts, labels, colours, savefilename):
    bars = plt.bar(labels, action_counts, color=colours, width=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')
    plt.title("Action Distribution")
    plt.savefig(savefilename)
    plt.close()

# Wear Level and Error Rate Progression
def plot_state_progression(wear_levels, error_rates, savefilename):
    plt.plot(wear_levels, label="Wear Level", color='red')
    plt.plot(error_rates, label="Error Rate", color='green')
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.title("Wear Level and Error Rate Over Time")
    plt.legend()
    plt.savefig(savefilename)
    plt.close()

def plot_loss(losses, savefilename):
    plt.plot(losses, label="Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Loss Over Training")
    plt.legend()
    plt.savefig(savefilename)
    plt.close()

# def plot_actions(action_dict, steps_dict):
#     plt.figure(figsize=(12, 6))
#     for episode, actions in action_dict.items():
#         time_steps = steps_dict[episode]
#         plt.plot(time_steps, actions, marker='o', label=f'Episode {episode}')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Actions')
#     plt.title('Actions over Time Steps for Each Episode')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def plot_actions(action_dict, steps_dict, failed_dict, state_dict, ytick_labels, savefilename):
    num_episodes = len(action_dict)
    fig, axes = plt.subplots(num_episodes, 1, figsize=(12, 2 * num_episodes))
    if num_episodes == 1:
        axes = [axes]
    colors = plt.cm.get_cmap('tab10', num_episodes)
    for episode, actions in action_dict.items():
        time_steps = steps_dict[episode]
        axes[episode].plot(time_steps, actions, marker='o', color=colors(episode))
        
        # axes[episode].scatter(time_steps, failed_dict[episode], marker='o', color=colors(episode))
        failed_time_steps = [t+0.5 for t, f in zip(time_steps, failed_dict[episode]) if f != 0]
        failed_values = [f for f in failed_dict[episode] if f != 0]
        axes[episode].scatter(failed_time_steps, failed_values, marker='o', color=colors(episode))

        axes[episode].set_yticks([0, 1, 2])
        axes[episode].set_yticklabels(ytick_labels)
        axes[episode].grid(True)
        axes[episode].set_xticks(range(1, max(time_steps)+1))
        axes[episode].set_xlabel('Time Steps')
        axes[episode].set_title(f"Initial state\nLast Maintenance Month: {state_dict[episode]['time_since_last_maintenance_in_months']} \
            Operational Hours: {state_dict[episode]['operational_hours']} \
            Wear Level: {state_dict[episode]['wear_level']} \
            Error Rate: {state_dict[episode]['error_rate']}")
    plt.suptitle('Actions over Time Steps for Each Episode')
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(savefilename)
