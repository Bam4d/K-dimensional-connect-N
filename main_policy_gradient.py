from environment import KDimConnectN
import numpy as np

def run_episode_policy_gradient(env, model):
    done = False
    steps = 0
    total_error = []
    total_reward = []
    state = env.reset()
    while not done:

        valid_move = False
        while not valid_move:
            next_action = model.sample_action([state])
            next_state, reward, done, valid_move = env.step(next_action)

        model.add_experience([state, next_action, reward, next_state, done])
        error, _ = model.train()

        steps+=1
        total_reward += reward
        total_error += error

    return steps, total_reward, total_error


if __name__ == '__main__':
    gamma = .99
    epsilon = 0.05

    # Traditional connect-4 game
    game_board = (6,7) # 2 dimensional
    to_win = 4 # connect 4

    env = KDimConnectN(game_board, to_win)