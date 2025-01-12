import random
from tic_tac_toe_env import TicTacToe
import pandas as pd
import matplotlib.pyplot as plt

# Constants
N_GAMES = 10000  # Number of games to simulate
ALPHA = 0.1  # Learning rate for Q-learning updates
GAMMA = 0.9  # Discount factor for future rewards
EPSILON_START = 0.1  # Initial exploration rate for epsilon-greedy strategy
EPSILON_END = 0.001  # Minimum exploration rate
DECAY_RATE = 0.05  # Decay factor for epsilon after each game

# Initialize Q-tables for both players (X and O)
Q_X = {}  # Q-table for player X
Q_O = {}  # Q-table for player O


# Helper functions

def select_action(Q, state, available_actions, epsilon):
    """
    Select an action using the epsilon-greedy strategy.
    :param Q: Q-table for the current player
    :param state: Current game state
    :param available_actions: List of valid actions
    :param epsilon: Current exploration rate
    :return: Chosen action
    """
    if random.uniform(0, 1) < epsilon:  # Exploration
        return random.choice(available_actions)
    # Exploitation
    if state in Q:
        return max(Q[state], key=Q[state].get)  # Best action based on Q-values
    else:
        return random.choice(available_actions)  # Default to random action


def initialize_q(Q, state, available_actions):
    """
    Initialize the Q-values for a state-action pair if not already present.
    :param Q: Q-table
    :param state: Current game state
    :param available_actions: List of valid actions
    """
    if state not in Q:
        Q[state] = {action: 0 for action in available_actions}


def update_q_value(Q, state, action, reward, new_state, alpha, gamma):
    """
    Update the Q-value using the Bellman equation.
    :param Q: Q-table
    :param state: Current state
    :param action: Action taken
    :param reward: Reward received
    :param new_state: Resulting state
    :param alpha: Learning rate
    :param gamma: Discount factor
    """
    if state not in Q:
        Q[state] = {}
    if action not in Q[state]:
        Q[state][action] = 0

    max_future_q = max(Q[new_state].values()) if new_state in Q else 0
    Q[state][action] += alpha * (reward + gamma * max_future_q - Q[state][action])


# Main training loop
draw_games = []  # Track whether each game ends in a draw
env = TicTacToe()  # Initialize the Tic-Tac-Toe environment
epsilon = EPSILON_START  # Start with the initial exploration rate

for game in range(N_GAMES):
    state, _ = env.reset()  # Reset the environment for a new game
    terminated = False
    previous_state = None
    previous_action = None

    while not terminated:
        player_turn = env.get_player_turn()  # Get the current player (1 = X, -1 = O)
        available_actions = env.get_available_actions()  # Get valid actions for the current state

        if player_turn == 1:  # Player X's turn
            initialize_q(Q_X, state, available_actions)
            action = select_action(Q_X, state, available_actions, epsilon)
        else:  # Player O's turn
            initialize_q(Q_O, state, available_actions)
            action = select_action(Q_O, state, available_actions, epsilon)

        # Perform the chosen action
        new_state, reward, terminated, _, _ = env.step(action)

        # Update Q-values for the previous player's perspective
        if previous_state is not None and previous_action is not None:
            if player_turn == 1:  # O was the last player
                update_q_value(Q_O, previous_state, previous_action, -reward, new_state, ALPHA, GAMMA)
            else:  # X was the last player
                update_q_value(Q_X, previous_state, previous_action, reward, new_state, ALPHA, GAMMA)

        # Update the current state and action for the next iteration
        previous_state = state
        previous_action = action

        if terminated:
            # Final Q-value update for the last action
            if player_turn == 1:  # X's last move
                update_q_value(Q_X, state, action, reward, new_state, ALPHA, GAMMA)
            else:  # O's last move
                update_q_value(Q_O, state, action, -reward, new_state, ALPHA, GAMMA)

            # Record if the game was a draw
            draw_games.append(reward == 0)
        else:
            state = new_state

    # Decay epsilon to encourage exploitation over time
    epsilon = max(EPSILON_END, epsilon * DECAY_RATE)

# Save and analyze results
pd.Series(name='draw_games', data=draw_games).to_csv('draw_games.csv')

# Load the draw games data from the CSV file
draw_games_data = pd.read_csv('draw_games.csv', index_col=0)

# Calculate the cumulative percentage of draw games over time
draw_games_data['Cumulative_Draw_Percentage'] = draw_games_data['draw_games'].expanding().mean()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(draw_games_data.index, draw_games_data['Cumulative_Draw_Percentage'], label='Cumulative Draw Percentage')
plt.xlabel('Game Number')
plt.ylabel('Cumulative Draw Percentage')
plt.title('Cumulative Draw Percentage Over Games')
plt.legend()
plt.grid(True)
plt.show()

# Print the total number of drawn games
n_drawn = draw_games.count(True)
print(f"Number of drawn games: {n_drawn} out of {N_GAMES} ({n_drawn / N_GAMES:.2%})")