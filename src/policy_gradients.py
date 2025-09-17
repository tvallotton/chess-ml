import chess
import lightning as L
import torch
from torch import nn, optim

from src.environment import BatchedChessEnvironment, ChessEnvironment


class ResidualConv(nn.Module):
    def __init__(self, channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor):
        y = self.dropout(self.elu(self.conv(x)))
        return y + x


class ChessPolicy(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.from_square = nn.Sequential(
            nn.Conv2d(12, d, 3, padding=1),
            ResidualConv(d, 3, padding=1),
            ResidualConv(d, 3, padding=1),
            ResidualConv(d, 3, padding=1),
            ResidualConv(d, 3, padding=1),
            nn.Flatten(-2),
        )

        self.to_square = nn.Sequential(
            nn.Conv2d(12, d, 3, padding=1),
            ResidualConv(d, 3, padding=1),
            ResidualConv(d, 3, padding=1),
            ResidualConv(d, 3, padding=1),
            ResidualConv(d, 3, padding=1),
            nn.Flatten(-2),
        )

    def forward(self, board: torch.Tensor, move_mask: torch.Tensor):
        from_square = self.from_square(board)
        to_square = self.to_square(board)

        # normalize
        from_square = from_square / torch.norm(
            from_square, dim=-2, keepdim=True
        ).clamp_min(1e-8)
        to_square = to_square / torch.norm(to_square, dim=-2, keepdim=True).clamp_min(
            1e-8
        )

        # generate probs
        logits = torch.bmm(from_square.transpose(-1, -2), to_square)
        masked_logits = logits.masked_fill(~move_mask, -torch.inf)
        return masked_logits.view(-1, 64**2)


class TrainingLoop:

    def __init__(self, model: ChessPolicy, max_move_count=50):
        self.model = model
        self.optimizer = optim.Adam(model.parameters())

        self.environment = BatchedChessEnvironment()

        self.gamma = 0.98
        self.max_move_count = max_move_count
        self.games = 0

    def train_once(self):

        self.optimizer.zero_grad()

        rewards = torch.zeros(self.max_move_count)
        log_probs = torch.zeros(self.max_move_count)
        finished = False
        i = 0

        while not finished:
            board = self.environment.state()
            mask = self.environment.legal_move_mask()
            logits = self.model(torch.stack([board]), torch.stack([mask]))

            dist = torch.distributions.Categorical(logits=logits)
            sample = dist.sample()
            log_probs[i] = dist.log_prob(sample)

            from_sq = int((sample // 64).item())
            to_sq = int((sample % 64).item())
            move = chess.Move(from_sq, to_sq)

            reward, finished = self.environment.play(move)

            rewards[i] = reward

            i += 1

        t = torch.arange(0, i)
        discount = self.gamma**t
        returns = rewards[:i] * discount[:i]

        white_returns = returns[::2].flip(-1).cumsum(-1).flip(-1)
        black_returns = returns[1::2].flip(-1).cumsum(-1).flip(-1)

        # subtract baseline
        white_returns = white_returns - white_returns.mean()
        black_returns = black_returns - black_returns.mean()

        white_loss = -(white_returns * log_probs[:i:2]).mean()
        black_loss = -(black_returns * log_probs[1:i:2]).mean()

        loss = white_loss + black_loss

        self.environment.reset()
        loss.backward()
        self.optimizer.step()
        self.games += 1


def backward_cumsum(x):

    torch.flip(x, (-1,)).cumsum(-1).flip()
