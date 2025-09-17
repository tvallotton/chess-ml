from dataclasses import dataclass, field
from re import I

import chess
import torch
from chess import A1, A8, B1, B8, C1, C8, D1, D8, E1, E8, F1, F8, G1, G8, H1, H8, Color

PIECE_VALUE = {
    chess.QUEEN: 9,
    chess.PAWN: 1,
    chess.BISHOP: 3,
    chess.KNIGHT: 3,
    chess.ROOK: 5,
    chess.KING: 0,
}


@dataclass
class VectorEmbeddings:
    _board: chess.Board

    def legal_moves_mask(self):
        legal_moves = torch.zeros((64, 64), dtype=bool)
        for move in self._board.legal_moves:
            legal_moves[move.from_square, move.to_square] = True
        return legal_moves

    def board_tensor(self):
        board = torch.zeros((12, 8, 8))

        for square in range(0, 64):
            piece = self._board.piece_at(square)
            if piece is None:
                continue

            column = square % 8
            row = square // 8
            channel = 6 * (piece.color != self._board.turn) + piece.piece_type - 1
            board[channel, row, column] = 1.0

        return board


@dataclass
class GameInfo:
    board_state: torch.Tensor
    legal_move_mask: torch.Tensor
    black_reward: float
    white_reward: float
    turn_reward: float


@dataclass
class ChessEnvironment:
    board: chess.Board = field(default_factory=chess.Board)
    move_count: int = 0
    max_move_count: int = 50

    def state(self):
        return VectorEmbeddings(self.board).board_tensor()

    def legal_move_mask(self):
        return VectorEmbeddings(self.board).legal_moves_mask()

    def reset(self):
        self.move_count = 0
        self.board.reset()

    def play(self, move: chess.Move):
        turn = self.board.turn

        starting_material = self.material_balance(turn)
        self.move_count += 1
        self.push_move(move)

        if self.board.is_game_over():
            r = self.final_reward()
            return r, True

        ending_material = self.material_balance(turn)
        reward = ending_material - starting_material
        return reward, self.move_count >= self.max_move_count

    def material_balance(self, turn: Color):
        out = 0.0
        for piece in self.board.piece_map().values():
            sign = 2 * (piece.color == turn) - 1
            out += sign * PIECE_VALUE[piece.piece_type]
        return out

    def final_reward(self):
        if self.board.result() == "1/2-1/2":
            return 0.0
        return 1000.0

    def push_move(self, move: chess.Move):
        if self.is_promotion(move):
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

        assert self.board.is_legal(move)

        self.board.push(move)

    def is_promotion(self, move: chess.Move):
        is_pawn = self.board.piece_at(move.from_square).piece_type == chess.PAWN

        black_promo = {A1, B1, C1, D1, E1, F1, G1, H1}
        white_promo = {A8, B8, C8, D8, E8, F8, G8, H8}

        promotion_rank = white_promo if self.board.turn else black_promo

        return is_pawn and move.to_square in promotion_rank


class BatchedChessEnvironment:
    def __init__(self, batch_size, max_move_count=50):
        self.environments = [
            ChessEnvironment(max_move_count) for _ in range(batch_size)
        ]
        self.reward = torch.zeros(batch_size, max_move_count)

    def reset(self):
        for env in self.environments:
            env.reset()
        self.reward.zero_()

    def state(self):
        return torch.stack([env.state() for env in self.environments])

    def legal_move_mask(self):
        return torch.stack([env.lega_move_mask() for env in self.environments])

    def play(self, moves: torch.Tensor):

        for i, (env, move) in enumerate(zip(self.environments, moves)):
            if env.board.is_game_over():
                continue

            from_sq = int((move // 64).item())
            to_sq = int((move % 64).item())
            move = chess.Move(from_sq, to_sq)
            reward, _ = env.play(move)
            self.reward[i, env.move_count] = reward
