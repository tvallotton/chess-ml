from collections import deque
from dataclasses import dataclass, field
from re import I

import chess
import numpy as np
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
    device: str = "cpu"

    def legal_moves_mask(self):
        legal_moves = torch.zeros((64, 64), dtype=bool, device=self.device)
        for move in self._board.legal_moves:

            from_sq = self.to_relative(move.from_square)
            to_sq = self.to_relative(move.to_square)

            legal_moves[from_sq, to_sq] = True

        return legal_moves

    def into_move(self, move: torch.Tensor):

        from_r, from_c, to_row, to_col = torch.unravel_index(move, (8, 8, 8, 8))

        if not self._board.turn:
            from_r = 7 - from_r
            from_c = 7 - from_c
            to_row = 7 - to_row
            to_col = 7 - to_col

        from_r = 7 - from_r
        to_row = 7 - to_row

        from_square = chess.square(rank_index=int(from_r), file_index=int(from_c))
        to_square = chess.square(rank_index=int(to_row), file_index=int(to_col))

        return chess.Move(from_square, to_square)

    def to_absolute(self, square: torch.Tensor) -> chess.Square:
        row, col = torch.unravel_index(square, (8, 8))
        if not self._board.turn:
            row = 7 - col
            col = 7 - col
        row = 7 - row
        return chess.square(rank_index=int(row), file_index=int(col))

    def to_relative(self, square: chess.Square) -> torch.Tensor:
        row = chess.square_rank(square)
        col = chess.square_file(square)

        row = 7 - row

        if not self._board.turn:
            row = 7 - row
            col = 7 - col
        return torch.tensor(
            np.ravel_multi_index((row, col), (8, 8)), device=self.device
        )

    def board_tensor(self):
        board = torch.zeros((12, 8, 8))
        movable_squares = torch.ones(1, 8, 8)

        for square in range(0, 64):
            piece = self._board.piece_at(square)
            if piece is None:
                continue

            column = 7 - square % 8
            row = square // 8
            channel = 6 * (piece.color != self._board.turn) + piece.piece_type - 1
            board[channel, row, column] = 1.0

        if self._board.turn:
            board = torch.flip(board, (-2, -1))

        return torch.cat((board, movable_squares), dim=0)


@dataclass
class ChessEnvironment:
    board: chess.Board = field(default_factory=chess.Board)
    move_count: int = 0
    max_move_count: int = 50
    device: str = "cpu"

    def into_move(self, move: torch.Tensor):
        return VectorEmbeddings(self.board, self.device).into_move(move)

    def state(self):
        return VectorEmbeddings(self.board, self.device).board_tensor()

    def legal_move_mask(self):
        return VectorEmbeddings(self.board, self.device).legal_moves_mask()

    def reset(self):
        self.move_count = 0
        self.board.reset()
        self.opponent_material_difference = 0.0

    def play(self, move: chess.Move):
        "returns reward relative to the CURRENT PLAYER, not white."
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

    def material_balance(self, turn):
        out = 0.0
        for piece in self.board.piece_map().values():
            sign = 2 * (piece.color == turn) - 1
            out += sign * PIECE_VALUE[piece.piece_type]
        return out

    def final_reward(self):
        if self.board.result() == "1/2-1/2":
            return 0.0
        # A player CANNOT LOSE duing its own turn, so it must have one
        # rewards are RELATIVE.
        return 20.0

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
