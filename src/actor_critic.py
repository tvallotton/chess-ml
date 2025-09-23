import os
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from email.policy import Policy
from math import isnan
from typing import TypedDict

import numpy as np
import torch
from IPython import embed
from matplotlib.collections import CircleCollection
from torch import device, nn
from torch.nn import functional as F
from torchrl.data import LazyMemmapStorage, ReplayBuffer

from src.environment import ChessEnvironment

infinity = 1e8


class ResidualConv(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()

        self.conv = nn.Sequential(
            # nn.BatchNorm2d(channels),
            # nn.LayerNorm(channels),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.LeakyReLU(),
            # nn.LayerNorm(channels),
            # nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor):
        y = self.conv(x)
        return y + x


class Actor(nn.Module):
    def __init__(self, embed_dim, n_heads, head_dim=2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.conv = nn.Sequential(
            nn.Conv2d(14, embed_dim, 3, padding=1),
            ResidualConv(embed_dim),
            ResidualConv(embed_dim),
            ResidualConv(embed_dim),
            ResidualConv(embed_dim),
            ResidualConv(embed_dim),
            ResidualConv(embed_dim),
            ResidualConv(embed_dim),
            ResidualConv(embed_dim),
            ResidualConv(embed_dim),
        )

        self.from_square = nn.Linear(embed_dim, head_dim * n_heads)
        self.to_square = nn.Linear(embed_dim, head_dim * n_heads)
        self.join_heads = nn.Linear(n_heads, 1)

    def forward(self, board: torch.Tensor, move_mask: torch.Tensor):
        board = self.conv(board)

        B, C, _, _ = board.shape
        H = self.n_heads

        board = board.view(B, C, 64).transpose(-1, -2)  # (B, 64, C)

        # reshape into heads
        from_square = self.from_square(board).view(B, 64, H, self.head_dim)
        to_square = self.to_square(board).view(B, 64, H, self.head_dim)

        # normalize heads
        from_square = from_square / torch.norm(from_square, dim=-1, keepdim=True)
        to_square = to_square / torch.norm(to_square, dim=-1, keepdim=True)

        # create multiple heads
        from_square = from_square.permute(0, 2, 1, 3).reshape(B * H, 64, self.head_dim)
        to_square = to_square.permute(0, 2, 1, 3).reshape(B * H, 64, self.head_dim)

        # comute multi head cosine similarity
        multi_head = torch.bmm(from_square, to_square.transpose(-1, -2)).view(
            B, H, 64, 64
        )

        multi_head = multi_head.permute(0, 2, 3, 1)  # (B, 64, 64, H)

        logits = self.join_heads(multi_head)
        logits = logits.view(B, 64, 64)
        masked_logits = logits.masked_fill(~move_mask, -infinity)

        return masked_logits.view(B, 64**2)


class Critic(nn.Module):
    def __init__(self, embed_dim, heads, head_dim):
        super().__init__()

        self.actor = Actor(embed_dim, heads, head_dim)

    def forward(self, board: torch.Tensor, move_mask: torch.Tensor):
        return self.actor(board, move_mask)


class ReplayMemory(TypedDict):
    reward: torch.Tensor
    from_state: torch.Tensor
    from_move_mask: torch.Tensor
    to_move_mask_as_opponent: torch.Tensor
    to_state_as_opponent: torch.Tensor
    to_move_mask_as_player: torch.Tensor
    to_state_as_player: torch.Tensor
    move: torch.Tensor
    done: torch.Tensor


class ReplayBufferInfo(TypedDict):
    _weight: torch.Tensor
    index: torch.Tensor


class TrainingLoop:
    def __init__(self, batchsize, device, max_move_count=150):
        self.device = device
        self.batchsize = batchsize
        self.environments = [
            ChessEnvironment(max_move_count=max_move_count, device=device)
            for _ in range(batchsize)
        ]
        self.actor = Actor(32, 20, 3).to(device)
        self.critics = nn.ModuleList([Critic(32, 16, 3), Critic(32, 16, 3)]).to(device)
        self.target_critics = deepcopy(self.critics).to(device)
        self.update_target(1.0)

        self.replay_buffer = ReplayBuffer(
            storage=LazyMemmapStorage(500_000),
        )

        self.tau = 0.001

        self.log_alpha = nn.Parameter(
            torch.tensor(0.0, requires_grad=True, device=device)
        )
        self.gamma = 0.99
        self.episodes = 0
        self.games = 0

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters())
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha])

    def alpha(self):
        return self.log_alpha.exp()

    def train(self, batches: int, save=None):
        for _ in range(batches):
            print(_)
            self.play_one_batch()
            self.learn()
            self.update_target(self.tau)
            if save is not None:
                save(self, self.episodes)

    def learn(self):
        if len(self.replay_buffer) < 10_000:
            return

        memory, info = self.sample_batch()

        logits = self.actor(memory["from_state"], memory["from_move_mask"])
        q_value = self.critic(memory["from_state"], memory["from_move_mask"]).view(
            -1, 64, 64
        )

        self.train_alpha(logits, memory)
        self.train_actor(logits, q_value)
        self.train_critic(memory, info)
        self.episodes += 1

    def sample_batch(self) -> tuple[ReplayMemory, ReplayBufferInfo]:
        batch, info = self.replay_buffer.sample(self.batchsize, return_info=True)

        for k in batch:
            batch[k] = batch[k].to(self.device)

        info["_weight"] = info["_weight"].to(self.device)
        info["index"] = info["index"].to(self.device)

        return batch, info

    def soft_value(self, logits: torch.Tensor, q_value: torch.Tensor):
        q_value = q_value.view(-1, 64**2)
        logpi = F.log_softmax(logits, dim=-1)
        pi = logpi.exp()
        soft_value = pi * (q_value - self.alpha().detach() * logpi)
        return soft_value.sum(-1)

    def actor_loss(self, logits: torch.Tensor, q_value: torch.Tensor):
        return -self.soft_value(logits, q_value.detach())

    def train_alpha(self, logits: torch.Tensor, memory: ReplayMemory):
        max_entropy = memory["from_move_mask"].sum((-1, -2)).log()
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy().detach()
        loss = self.log_alpha * (entropy - 0.8 * max_entropy.detach())
        self.alpha_optimizer.zero_grad()
        loss.mean().backward()
        self.alpha_optimizer.step()

    def train_critic(self, memory: ReplayMemory, info: ReplayBufferInfo):

        with torch.no_grad():
            # use player's coordinates to evaluate game
            next_q_value = self.target_critic(
                memory["to_state_as_player"], memory["to_move_mask_as_player"]
            )
            # use opponent's coordinates on actor
            next_logits = self.actor(
                memory["to_state_as_opponent"], memory["to_move_mask_as_opponent"]
            )
            # change to player coordinates
            next_logits = (
                next_logits.view(-1, 64, 64).flip(-1, -2).view(-1, 64**2).detach()
            )
            next_soft_value = self.soft_value(next_logits, next_q_value)
            bootstrap = torch.where(memory["done"], 0.0, next_soft_value)
            y = memory["reward"] + self.gamma * bootstrap

        q_value0 = self.critics[0](memory["from_state"], memory["from_move_mask"])
        q_value1 = self.critics[1](memory["from_state"], memory["from_move_mask"])
        q_value = (q_value0 + q_value1) / 2
        q_value = torch.gather(q_value, 1, memory["move"].view(-1, 1)).flatten()

        mse = (q_value - y) ** 2

        loss = mse.mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        print(f"critic loss: {loss.item():+.4f}")
        self.critic_optimizer.step()

        td_error = (q_value - y).detach()
        priority = td_error.abs() + 1e-6
        self.replay_buffer.update_priority(info["index"], priority)

    def train_actor(self, logits: torch.Tensor, q_value: torch.Tensor):

        self.actor_optimizer.zero_grad()
        loss = self.actor_loss(logits, q_value).mean(0)
        assert not loss.isnan().any()
        loss.backward()
        print(f"actor loss: {loss.item():+.4f}")
        self.actor_optimizer.step()

    def critic(self, state: torch.Tensor, move_mask: torch.Tensor):

        return torch.minimum(*(critic(state, move_mask) for critic in self.critics))

    def target_critic(self, state: torch.Tensor, move_mask: torch.Tensor):
        return torch.minimum(
            *(critic(state, move_mask) for critic in self.target_critics)
        )

    def update_target(self, tau):
        with torch.no_grad():

            parameters = zip(
                self.target_critics.parameters(), self.critics.parameters()
            )
            buffers = zip(self.target_critics.buffers(), self.critics.buffers())

            for target, online in parameters:
                if not target.dtype.is_floating_point:
                    target.copy_(online)
                    continue
                target.mul_(1 - tau).add_(tau * online)

            for target, online in buffers:
                target.copy_(online)

        self.target_critics.eval()

    def play_one_batch(self):

        from_state = self.state()
        from_mask = self.legal_move_mask()

        logits = self.actor(from_state, from_mask)

        dist = torch.distributions.Categorical(logits=logits)
        moves = dist.sample()

        self.push_moves(moves)

    def push_moves(self, moves: torch.Tensor):
        for env, move in zip(self.environments, moves):
            player = env.board.turn
            from_move_mask = env.legal_move_mask()
            from_state = env.state()

            reward, finished = env.play(env.into_move(move))

            memory = ReplayMemory(
                reward=torch.tensor(reward),
                from_state=from_state,
                from_move_mask=from_move_mask,
                to_state_as_player=env.state(player),
                to_state_as_opponent=env.state(not player),
                to_move_mask_as_player=env.legal_move_mask(player),
                to_move_mask_as_opponent=env.legal_move_mask(not player),
                move=move,
                done=torch.tensor(finished),
            )

            self.replay_buffer.add(memory)

            if finished:
                self.games += 1
                env.reset()

    def state(self, perspective=None) -> torch.Tensor:
        return torch.stack(
            [env.state(perspective) for env in self.environments],
        ).to(self.device)

    def legal_move_mask(self) -> torch.Tensor:
        return torch.stack(
            [env.legal_move_mask() for env in self.environments],
        ).to(self.device)

    def save_checkpoint(self, path: str):
        ckpt = {
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "tau": self.tau,
            "log_alpha": self.log_alpha,
            "gamma": self.gamma,
            "episodes": self.episodes,
            "games": self.games,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
        }

        os.makedirs(path, exist_ok=True)

        torch.save(ckpt, os.path.join(path, "parameters.ckpt"))
        # self.replay_buffer.dumps(os.path.join(path, "replay_buffer.ckpt"))

    def load_checkpoint(self, dir: str, map_location=None, strict: bool = True):
        param_path = os.path.join(dir, "parameters.ckpt")

        ckpt = torch.load(
            param_path,
            map_location=map_location,
            weights_only=False,
        )
        self.games = ckpt.get("games", self.games)
        self.actor.load_state_dict(ckpt["actor"], strict=strict)
        self.critics.load_state_dict(ckpt["critics"], strict=strict)

        self.update_target(1.0)

        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])

        self.tau = ckpt.get("tau", self.tau)

        self.log_alpha = ckpt.get("log_alpha", self.log_alpha)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=0.001)

        self.gamma = ckpt.get("gamma", self.gamma)
        self.episodes = ckpt.get("episodes", self.episodes)

        rng = ckpt["rng"]

        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
