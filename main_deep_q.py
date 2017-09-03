from environment import KDimConnectN
from models import DeepQ, SimpleFeedForward
import numpy as np
import tensorflow as tf
from utils import plot_running_average

def run_episode_Q(env, model, epsilon):
    done = False
    steps = 0
    total_cost = 0
    total_reward = 0
    state = env.reset()
    state = state.flatten()

    epsilon = max(0.01, epsilon-0.001)

    while not done:

        player = 1 if steps % 2 == 0 else -1

        valid_move = False
        while not valid_move:
            next_action = model.sample_action(state, epsilon) if player == 1 else env.sample_action()
            next_state, reward, done, valid_move = env.step(np.atleast_1d(next_action), player)

            next_state = next_state.flatten()

        if player == 1:
            model.add_experience([state.flatten(), next_action, reward, next_state.flatten(), done])
            cost = model.train()


            total_reward += reward
            total_cost += cost

        steps += 1

    return steps, total_reward, total_cost, epsilon


if __name__ == '__main__':
    gamma = .99
    epsilon = 1.0

    # Traditional connect-4 game
    game_board = [60,70] # 2 dimensional
    to_win = 4 # connect 4

    n_inputs = np.prod(game_board)

    # We will flatten the state
    n_outputs = np.prod(game_board[1:])

    # layer configuration
    layers = [
        [64, tf.nn.sigmoid],
        [32, tf.nn.sigmoid],
        [16, tf.nn.sigmoid],
        [n_outputs, tf.nn.sigmoid]
    ]

    learning_rate = 1e-2

    iterations = 1000

    model = SimpleFeedForward(n_inputs, n_outputs, layers, learning_rate, use_bias=True, use_l2 = False, name="model")
    target = SimpleFeedForward(n_inputs, n_outputs, layers, learning_rate, use_bias=True, use_l2 = False, name="target")

    deep_q_network = DeepQ(gamma, model, target, episode_state_history_max=10000, episode_state_history_min=100, batch_sz=32, update_period=100)

    init = tf.global_variables_initializer()
    session = tf.Session()

    deep_q_network.set_session(session)

    session.run(init)

    env = KDimConnectN(game_board, to_win)

    total_rewards = np.zeros(iterations)
    total_steps = np.zeros(iterations)
    total_errors = np.zeros(iterations)


    for i in range(0, iterations):
        steps, total_reward, total_error, epsilon = run_episode_Q(env, deep_q_network, epsilon)

        total_steps[i] = steps
        total_rewards[i] = total_reward
        total_errors[i] = total_error

        if i % 10 == 0:
            print("Iteration %d, reward:%.2f steps:%d mean reward (last 100): %.3f, mean error (last 100): %.3f, Epsilon: %.3f" % (
            i, total_reward, steps, total_rewards[max(0, i - 100):i].mean(), total_errors[max(0, i - 100):i].mean(), epsilon))

    plot_running_average(total_rewards, 100)
    plot_running_average(total_steps, 100)