from pytest import fixture
from environment import KDimConnectN
import numpy as np

iterations = 10


def test_sample_from_envs():
    """
    Play a game completely randomly
    :return:
    """
    env = KDimConnectN(dimension_configuration=(6, 7,8,9,10), N=4)

    for i in range(0, iterations):

        env.reset()
        steps = 0
        done = False

        while not done:

            if steps % 2:
                player = -1
            else:
                player = 1

            valid_move = False
            while not valid_move:
                action = env.sample_action()
                state, reward, done, valid_move = env.step(action, player)

            steps += 1
            if done:
                print("player %d, won after %d moves." % (player, steps))
                #print(state)

