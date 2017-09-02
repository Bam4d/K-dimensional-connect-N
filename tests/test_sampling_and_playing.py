from pytest import fixture
from environment import Environment
import numpy as np

test_samples = 1000


def test_sample_from_envs():
    """
    Play a game completely randomly
    :return:
    """
    env = Environment(dimension_configuration=(6, 7), N=4)

    for i in range(0,test_samples):
        if i % 2:
            player = -1
        else:
            player = 1

        action = env.sample_action()
        state, reward, done, _ = env.step(action, player)

        print(state)

        if done:
            print("player %d, won the game!", player)
            env.reset()