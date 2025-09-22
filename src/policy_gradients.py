import chess
import torch
from torch import logit, nn, optim

from src.environment import ChessEnvironment
from src.residual_conv import ResidualConv


class ChessPolicy(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.from_square = nn.Sequential(
            nn.Conv2d(13, d, 3, padding=1),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
            nn.Flatten(-2),
        )

        self.to_square = nn.Sequential(
            nn.Conv2d(13, d, 3, padding=1),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
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
        assert not logits.isnan().any()
        masked_logits = logits.masked_fill(~move_mask, -torch.inf)
        return masked_logits.view(-1, 64**2)


class TrainingLoop:

    def __init__(self, model: ChessPolicy, max_move_count=200):
        self.model = model
        self.optimizer = optim.Adam(model.parameters())

        self.environment = ChessEnvironment()

        self.gamma = 0.99
        self.max_move_count = max_move_count
        self.games = 0

    def train_once(self):

        self.optimizer.zero_grad()

        rewards = torch.zeros(self.max_move_count)
        log_probs = torch.zeros(self.max_move_count)
        # entropy = torch.zeros(self.max_move_count)
        finished = False
        i = 0

        while not finished:
            board = self.environment.state()
            mask = self.environment.legal_move_mask()
            logits = self.model(torch.stack([board]), torch.stack([mask]))

            dist = torch.distributions.Categorical(logits=logits)
            sample = dist.sample()
            log_probs[i] = dist.log_prob(sample)

            move = self.environment.into_move(sample)

            reward, finished = self.environment.play(move)

            rewards[i] = reward

            i += 1

        t = torch.arange(0, i)
        discount = self.gamma**t

        returns = rewards[:i] * discount[:i]

        reward_sign = torch.where((t % 2) == 0, 1.0, -1.0)
        returns = returns * reward_sign

        G = reward_sign * returns.flip(-1).cumsum(-1).flip(-1) / discount[:i]

        G = (G - G.mean()) / (G.std().clamp_min(1e-6))

        loss = -(G * log_probs[:i]).mean()

        self.environment.reset()
        loss.backward()
        self.optimizer.step()
        self.games += 1
