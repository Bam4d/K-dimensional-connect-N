from pytest import fixture
from environment import Environment
import numpy as np

def test_check_for_win_2d_dim1():
    env = Environment(dimension_configuration=(7,6), N=4)

    # gravity goes "up" with respect to how this array looks
    env.state = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])

    assert env.check_for_win(1, (0, 0)) == True
    assert env.check_for_win(1, (0, 1)) == True
    assert env.check_for_win(1, (0, 2)) == True
    assert env.check_for_win(1, (0, 3)) == True

    assert env.check_for_win(1, (0, 4)) == False

    assert env.check_for_win(-1, (0, 0)) == False
    assert env.check_for_win(-1, (0, 1)) == False
    assert env.check_for_win(-1, (0, 2)) == False
    assert env.check_for_win(-1, (0, 3)) == False

    assert env.check_for_win(-1, (1, 0)) == False
    assert env.check_for_win(-1, (1, 1)) == False
    assert env.check_for_win(-1, (1, 2)) == False
    assert env.check_for_win(-1, (1, 3)) == False


def test_check_for_win_2d_dim2():
    env = Environment(dimension_configuration=(7,6), N=4)

    # gravity goes "up" with respect to how this array looks
    env.state = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])

    assert env.check_for_win(1, (0, 3)) == True
    assert env.check_for_win(1, (1, 3)) == True
    assert env.check_for_win(1, (2, 3)) == True
    assert env.check_for_win(1, (3, 3)) == True

    assert env.check_for_win( 1, (4, 3)) == False
    assert env.check_for_win(-1, (0, 3)) == False
    assert env.check_for_win(-1, (1, 3)) == False
    assert env.check_for_win(-1, (2, 3)) == False
    assert env.check_for_win(-1, (3, 3)) == False

def test_check_for_win_2d_diag1():
    env = Environment(dimension_configuration=(7, 6), N=4)

    # gravity goes "up" with respect to how this array looks
    env.state = np.array([
        [1,-1,-1,-1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1,-1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])

    assert env.check_for_win(1, (0, 0)) == True
    assert env.check_for_win(1, (1, 1)) == True
    assert env.check_for_win(1, (2, 2)) == True
    assert env.check_for_win(1, (3, 3)) == True

    assert env.check_for_win(1, (4, 4)) == False
    assert env.check_for_win(-1, (0, 0)) == False
    assert env.check_for_win(-1, (0, 3)) == False
    assert env.check_for_win(-1, (0, 3)) == False

def test_check_for_win_2d_diag2():
    env = Environment(dimension_configuration=(7, 6), N=4)

    # gravity goes "up" with respect to how this array looks
    env.state = np.array([
        [ 1,-1,-1,-1, 0, 0, 0],
        [ 1, 1,-1, 1, 0, 0, 0],
        [ 1,-1, 1,-1, 0, 0, 0],
        [-1, 0, 0,-1, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0]
    ])

    assert env.check_for_win(-1, (0, 3)) == True
    assert env.check_for_win(-1, (1, 2)) == True
    assert env.check_for_win(-1, (2, 1)) == True
    assert env.check_for_win(-1, (3, 0)) == True

    assert env.check_for_win(1, (0, 0)) == False
    assert env.check_for_win(1, (1, 1)) == False
    assert env.check_for_win(1, (2, 2)) == False
    assert env.check_for_win(1, (3, 3)) == False

    assert env.check_for_win(1, (4, 4)) == False
    assert env.check_for_win(-1, (0, 0)) == False

def test_check_for_win_3d_dim1():
    env = Environment(dimension_configuration=(3, 3, 3), N=3)
    env.state = np.array([
        [
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ],[
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],[
            [0, 0, 0],
            [0, 0, 0],
            [-1, -1, -1]
        ]
    ])

    assert env.check_for_win(1, (0, 0, 0)) == True
    assert env.check_for_win(1, (1, 1, 0)) == True
    assert env.check_for_win(-1, (2, 2, 0)) == True

    assert env.check_for_win(1, (0, 0, 2)) == True
    assert env.check_for_win(1, (1, 1, 2)) == True
    assert env.check_for_win(-1, (2, 2, 1)) == True

    assert env.check_for_win(1, (2, 2, 1)) == False
    assert env.check_for_win(-1, (0, 0, 0)) == False

def test_check_for_win_3d_dim2():
    env = Environment(dimension_configuration=(3, 3, 3), N=3)
    env.state = np.array([
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0]
        ], [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ], [
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1]
        ]
    ])

    assert env.check_for_win(1, (0, 0, 0)) == True
    assert env.check_for_win(1, (1, 1, 1)) == True
    assert env.check_for_win(-1, (2, 2, 2)) == True

    assert env.check_for_win(1, (0, 1, 0)) == True
    assert env.check_for_win(1, (1, 0, 1)) == True
    assert env.check_for_win(-1, (2, 1, 2)) == True

    assert env.check_for_win(-1, (2, 2, 1)) == False
    assert env.check_for_win(-1, (0, 0, 0)) == False

def test_check_for_win_3d_dim3():
    env = Environment(dimension_configuration=(3, 3, 3), N=3)
    env.state = np.array([
        [
            [1, 0, 0],
            [0, 0,-1],
            [0, 1, 0]
        ], [
            [1, 0, 0],
            [0, 0,-1],
            [0, 1, 0]
        ], [
            [1, 0, 0],
            [0, 0,-1],
            [0, 1, 0]
        ]
    ])

    assert env.check_for_win(1, (0, 0, 0)) == True
    assert env.check_for_win(-1, (0, 1, 2)) == True
    assert env.check_for_win(1, (0, 2, 1)) == True

    assert env.check_for_win(1, (1, 0, 0)) == True
    assert env.check_for_win(-1, (1, 1, 2)) == True
    assert env.check_for_win(1, (1, 2, 1)) == True

    assert env.check_for_win(1, (2, 0, 0)) == True
    assert env.check_for_win(-1, (2, 1, 2)) == True
    assert env.check_for_win(1, (2, 2, 1)) == True

    assert env.check_for_win(-1, (2, 2, 1)) == False
    assert env.check_for_win(-1, (0, 0, 0)) == False

def test_check_for_win_3d_diag1():
    env = Environment(dimension_configuration=(3, 3, 3), N=3)
    env.state = np.array([
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1]
        ]
    ])

    assert env.check_for_win(1, (0, 0, 0)) == True
    assert env.check_for_win(1, (0, 1, 1)) == True
    assert env.check_for_win(1, (0, 2, 2)) == True

    assert env.check_for_win(1, (1, 1, 1)) == True

    assert env.check_for_win(1, (2, 0, 0)) == True
    assert env.check_for_win(1, (2, 1, 1)) == True
    assert env.check_for_win(1, (2, 2, 2)) == True

    assert env.check_for_win(-1, (2, 0, 2)) == False
    assert env.check_for_win(-1, (1, 1, 1)) == False
    assert env.check_for_win(-1, (1, 1, 1)) == False


def test_check_for_win_3d_diag2():
    env = Environment(dimension_configuration=(3, 3, 3), N=3)
    env.state = np.array([
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ], [
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 0]
        ], [
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0]
        ]
    ])

    assert env.check_for_win(1, (0, 0, 0)) == True
    assert env.check_for_win(1, (1, 1, 0)) == True
    assert env.check_for_win(1, (2, 2, 0)) == True

    assert env.check_for_win(1, (0, 2, 2)) == True
    assert env.check_for_win(1, (1, 1, 2)) == True
    assert env.check_for_win(1, (2, 0, 2)) == True

    assert env.check_for_win(-1, (2, 0, 2)) == False
    assert env.check_for_win(-1, (1, 1, 1)) == False
    assert env.check_for_win(-1, (1, 1, 1)) == False

def test_check_for_win_3d_diag3():
    env = Environment(dimension_configuration=(3, 3, 3), N=3)
    env.state = np.array([
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, -1]
        ], [
            [0, 1, 0],
            [0, 0, 0],
            [0, -1, 0]
        ], [
            [0, 0, 1],
            [0, 0, 0],
            [-1, 0, 0]
        ]
    ])

    assert env.check_for_win(1, (0, 0, 0)) == True
    assert env.check_for_win(1, (1, 0, 1)) == True
    assert env.check_for_win(1, (2, 0, 2)) == True

    assert env.check_for_win(-1, (0, 2, 2)) == True
    assert env.check_for_win(-1, (1, 2, 1)) == True
    assert env.check_for_win(-1, (2, 2, 0)) == True

    assert env.check_for_win(-1, (2, 0, 2)) == False
    assert env.check_for_win(-1, (1, 1, 1)) == False
    assert env.check_for_win(-1, (1, 1, 1)) == False

def test_check_for_win_3d_diag3():
    env = Environment(dimension_configuration=(3, 3, 3), N=3)
    env.state = np.array([
        [
            [1, 0, 0],
            [0, 0, 0],
            [1, 0, 0]
        ], [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], [
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 1]
        ]
    ])

    assert env.check_for_win(1, (0, 0, 0)) == True
    assert env.check_for_win(1, (1, 1, 1)) == True
    assert env.check_for_win(1, (2, 2, 2)) == True

    assert env.check_for_win(1, (0, 2, 0)) == True
    assert env.check_for_win(1, (2, 0, 2)) == True

    assert env.check_for_win(-1, (2, 0, 2)) == False
    assert env.check_for_win(-1, (1, 1, 1)) == False
    assert env.check_for_win(-1, (1, 1, 1)) == False

def test_check_for_win_3d_4x4():
    env = Environment(dimension_configuration=(4, 4, 4), N=4)
    env.state = np.array([
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, 0, 0, 0]
        ],[
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0,-1, 0, 0],
            [0, 0, 0, 0]
        ],[
            [0, 0, 0, 0],
            [0, 0,-1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ],[
            [0, 0, 0,-1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ]
    ])

    assert env.check_for_win(1, (0, 0, 0)) == True
    assert env.check_for_win(1, (1, 1, 1)) == True
    assert env.check_for_win(1, (2, 2, 2)) == True
    assert env.check_for_win(1, (3, 3, 3)) == True

    assert env.check_for_win(-1, (0, 3, 0)) == True
    assert env.check_for_win(-1, (1, 2, 1)) == True
    assert env.check_for_win(-1, (2, 1, 2)) == True
    assert env.check_for_win(-1, (3, 0, 3)) == True





