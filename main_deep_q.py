from environment import KDimConnectN
from models import DeepQ, SimpleFeedForward, RandomPlayer
import numpy as np
import tensorflow as tf
from utils import plot_running_average

def get_action_coords(action, game_board):
    return np.array(np.where(action.reshape(game_board[1:]) == 1)).flatten()

def perform_player_action(state, game_board, player, model, epsilon):

    # We have to to check for valid moves here because the agent might try to put a token into a completely full position

    valid_move = False
    punished_for_invalid_move = False
    while not valid_move:
        next_action = get_action_coords(model.sample_action(state, epsilon), game_board)
        next_state, reward, done, valid_move = env.step(np.atleast_1d(next_action), player)

        next_state = next_state.flatten()
        # We pubish for a bad state, but we should keep looping until a valid state is chosen
        if not punished_for_invalid_move and not valid_move:
            model.add_experience([state, next_action, reward, next_state, done])
            model.train()
            punished_for_invalid_move = True

    model.add_experience([state, next_action, reward, next_state, done])
    cost = model.train()

    return next_state, reward, cost, done


def run_episode_Q(env, player_1_model, player_2_model, epsilon):
    done = False
    steps = 0

    total_cost_p1 = 0
    total_reward_p1 = 0

    total_cost_p2 = 0
    total_reward_p2 = 0

    state = env.reset()
    state = state.flatten()

    epsilon = max(0.001, epsilon-0.001)

    while not done:

        player = 1 if steps % 2 == 0 else -1

        if player == 1:
            state, reward, cost, done = perform_player_action(state, env.dimension_configuration, player, player_1_model, epsilon)

            total_reward_p1 += reward
            total_cost_p1 += cost
        else:
            state, reward, cost, done = perform_player_action(state, env.dimension_configuration, player, player_2_model, epsilon)

            total_reward_p2 += reward
            total_cost_p2 += cost

        steps += 1

    return steps, total_reward_p1, total_reward_p2, total_cost_p1, total_cost_p2, epsilon





if __name__ == '__main__':
    gamma = .99
    epsilon = 1.0

    # Traditional connect-4 game
    game_board = [4,4,4] # 3 dimensional
    to_win = 4 # connect 4

    env = KDimConnectN(game_board, to_win)

    n_inputs = np.prod(game_board)

    # We will flatten the state
    n_outputs = np.prod(game_board[1:])

    # layer configuration
    layers = [
        [256, tf.nn.tanh],
        [128, tf.nn.tanh],
        [64, tf.nn.tanh],
        [n_outputs, tf.nn.tanh]
    ]

    learning_rate = 1e-2

    iterations = 4000

    # Set up our model and target networks for Deep Q learning
    model = SimpleFeedForward(n_inputs, n_outputs, layers, learning_rate, use_bias=True, use_l2 = False, name="model")
    target = SimpleFeedForward(n_inputs, n_outputs, layers, learning_rate, use_bias=True, use_l2 = False, name="target")

    # Our Deep Q network
    deep_q_network = DeepQ(gamma, model, target, episode_state_history_max=10000, episode_state_history_min=100, batch_sz=32, update_period=100)

    # Our initial random player
    random_player = RandomPlayer(env)

    init = tf.global_variables_initializer()
    session = tf.Session()

    deep_q_network.set_session(session)

    session.run(init)


    total_steps = np.zeros(iterations)

    total_rewards_p1 = np.zeros(iterations)
    total_costs_p1 = np.zeros(iterations)
    total_rewards_p2 = np.zeros(iterations)
    total_costs_p2 = np.zeros(iterations)

    player1_wins = np.zeros(iterations)
    player2_wins = np.zeros(iterations)

    # Play against random player
    for i in range(0, iterations):
        steps, total_reward_p1, total_reward_p2, total_cost_p1, total_cost_p2, epsilon = run_episode_Q(env, deep_q_network, random_player, epsilon)


        total_rewards_p1[i] = total_reward_p1
        total_costs_p1[i] = total_cost_p1
        player1_wins[i] = 1 if total_reward_p1 > total_reward_p2 else 0

        total_rewards_p2[i] = total_reward_p2
        total_costs_p2[i] = total_cost_p2
        player2_wins[i] = 1 if total_reward_p2 > total_reward_p1 else 0

        total_steps[i] = steps

        if i % 10 == 0:
            print("P1 Iteration %d, mean steps:%d, mean reward (last 100): %.3f, mean cost (last 100): %.3f, Epsilon: %.3f" % (
            i, total_steps[max(0, i - 100):i].mean(), total_rewards_p1[max(0, i - 100):i].mean(), total_costs_p1[max(0, i - 100):i].mean(), epsilon))

            print("P2 Iteration %d, mean steps:%d, mean reward (last 100): %.3f, mean cost (last 100): %.3f, Epsilon: %.3f" % (
                    i, total_steps[max(0, i - 100):i].mean(), total_rewards_p2[max(0, i - 100):i].mean(),
                    total_costs_p2[max(0, i - 100):i].mean(), epsilon))

    plot_running_average(total_rewards_p1, title="Running average reward P1", filename="deepq_vs_random_reward_p1.png", bin=100)
    plot_running_average(total_rewards_p2, title="Running average reward P2", filename="deepq_vs_random_reward_p2.png", bin=100)
    plot_running_average(total_costs_p1, title="Running average costs P1", filename="deepq_vs_random_cost_p1.png", bin=100)
    plot_running_average(total_costs_p2, title="Running average costs P2", filename="deepq_vs_random_cos_p2.png", bin=100)

    plot_running_average(player1_wins, title="Running prob wins P1", filename="deepq_vs_random_wins_p1.png", bin=100)
    plot_running_average(player2_wins, title="Running prob wins P2", filename="deepq_vs_random_wins_p2.png", bin=100)

    plot_running_average(total_steps, title="Running average steps", filename="deepq_vs_random_steps.png", bin=100)

    # Play against itself
    epsilon = 1.0
    deep_q_network.reset_experience()
    for i in range(0, iterations):
        steps, total_reward_p1, total_reward_p2, total_cost_p1, total_cost_p2, epsilon = run_episode_Q(env, deep_q_network, deep_q_network, epsilon)


        total_rewards_p1[i] = total_reward_p1
        total_costs_p1[i] = total_cost_p1
        player1_wins[i] = 1 if total_reward_p1 > total_reward_p2 else 0

        total_rewards_p2[i] = total_reward_p2
        total_costs_p2[i] = total_cost_p2
        player2_wins[i] = 1 if total_reward_p2 > total_reward_p1 else 0

        total_steps[i] = steps

        if i % 10 == 0:
            print("P1 Iteration %d, steps:%d mean reward (last 100): %.3f, mean cost (last 100): %.3f, Epsilon: %.3f" % (
            i, steps, total_rewards_p1[max(0, i - 100):i].mean(), total_costs_p1[max(0, i - 100):i].mean(), epsilon))

            print("P2 Iteration %d, steps:%d mean reward (last 100): %.3f, mean cost (last 100): %.3f, Epsilon: %.3f" % (
                    i, steps, total_rewards_p2[max(0, i - 100):i].mean(),
                    total_costs_p2[max(0, i - 100):i].mean(), epsilon))

    plot_running_average(total_rewards_p1, title="Running average reward P1", filename="deepq_vs_deepq_reward_p1.png", bin=100)
    plot_running_average(total_rewards_p2, title="Running average reward P2", filename="deepq_vs_deepq_reward_p2.png", bin=100)
    plot_running_average(total_costs_p1, title="Running average costs P1", filename="deepq_vs_deepq_cost_p1.png", bin=100)
    plot_running_average(total_costs_p2, title="Running average costs P2", filename="deepq_vs_deepq_cost_p2.png", bin=100)

    plot_running_average(player1_wins, title="Running average wins P1", filename="deepq_vs_random_wins_p1.png", bin=100)
    plot_running_average(player2_wins, title="Running average wins P2", filename="deepq_vs_random_wins_p2.png", bin=100)

    plot_running_average(total_steps, title="Running average steps", filename="deepq_vs_deepq_steps.png", bin=100)



