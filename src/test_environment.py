from chess import C2, C4, Move
from torch import Size, tensor

from .environment import ChessEnvironment


def test_environment_state():
    env = ChessEnvironment()
    env.state().shape == Size([12, 8, 8])


def test_play_environment_state():
    env = ChessEnvironment()
    reward, _ = env.play(Move(C2, C4))

    assert reward == 0
