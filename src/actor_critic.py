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
from matplotlib.collections import CircleCollection
from torch import device, nn
from torch.nn import functional as F
from torchrl.data import LazyMemmapStorage, PrioritizedReplayBuffer

from src.environment import ChessEnvironment

infinity = 1e8


class ResidualConv(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor):
        y = self.conv(x)
        return y + x


class Actor(nn.Module):
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
            1 / infinity
        )

        # generate probs
        logits = torch.bmm(from_square.transpose(-1, -2), to_square)

        masked_logits = logits.masked_fill(~move_mask, -infinity)
        return masked_logits.view(-1, 64**2)


class Critic(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d

        self.from_square = nn.Sequential(
            nn.Conv2d(13, d, 3, padding=1),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
            nn.Conv2d(d, d, 3, padding=1),
            nn.Flatten(-2),
        )

        self.to_square = nn.Sequential(
            nn.Conv2d(13, d, 3, padding=1),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
            ResidualConv(d),
            nn.Conv2d(d, d, 3, padding=1),
            nn.Flatten(-2),
        )

    def forward(self, board, move_mask: torch.Tensor):
        from_square = self.from_square(board)
        to_square = self.to_square(board)
        is_move_legal = torch.where(move_mask, 1.0, 0.0)
        return torch.bmm(from_square.transpose(-2, -1), to_square) * is_move_legal


class ReplayMemory(TypedDict):
    reward: torch.Tensor
    from_state: torch.Tensor
    to_state: torch.Tensor
    from_move_mask: torch.Tensor
    to_move_mask: torch.Tensor
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
        self.actor = Actor(16).to(device)
        self.critics = nn.ModuleList([Critic(16), Critic(16)]).to(device)
        self.target_critics = deepcopy(self.critics).to(device)
        self.update_target(1.0)

        self.replay_buffer = PrioritizedReplayBuffer(
            alpha=0.2,
            beta=1,
            storage=LazyMemmapStorage(20_000),
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters())

        self.tau = 0.001
        self.alpha = 1.0  # fixed for now
        self.gamma = 0.99
        self.episodes = 0

    def train(self, batches: int, save=None):
        for _ in range(batches):
            print(_)
            self.play_one_batch()
            self.learn()
            self.update_target(self.tau)
            self.episodes += 1
            if save is not None:
                save(self, self.episodes)

    def learn(self):
        if len(self.replay_buffer) < 10_000:
            return

        memory, info = self.sample_batch()

        logits = self.actor(memory["from_state"], memory["from_move_mask"])
        q_value = self.critic(memory["from_state"], memory["from_move_mask"])
        assert q_value.size(-1) == 64
        self.train_actor(logits, q_value)
        assert q_value.size(-1) == 64
        self.train_critic(memory, info, q_value)

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
        soft_value = pi * (q_value - self.alpha * logpi)
        return soft_value.sum(-1)

    def actor_loss(self, logits: torch.Tensor, q_value: torch.Tensor):
        return -self.soft_value(logits, q_value.detach())

    def train_critic(
        self, memory: ReplayMemory, info: ReplayBufferInfo, q_value: torch.Tensor
    ):
        assert q_value.size(-1) == 64

        with torch.no_grad():
            next_q_value = self.target_critic(
                memory["to_state"], memory["to_move_mask"]
            )
            next_logits = self.actor(memory["to_state"], memory["to_move_mask"])
            next_soft_value = self.soft_value(next_logits, next_q_value)
            bootstrap = torch.where(memory["done"], 0.0, next_soft_value)
            y = memory["reward"] + self.gamma * bootstrap

        q_value = torch.gather(
            q_value.flatten(-2), 1, memory["move"].view(-1, 1)
        ).flatten()

        mse = (q_value - y) ** 2
        weight = info["_weight"]
        loss = (weight * mse).sum() / weight.mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        td_error = (q_value - y).detach()
        priority = td_error.abs() + 1e-6
        self.replay_buffer.update_priority(info["index"], priority)

    def train_actor(self, logits: torch.Tensor, q_value: torch.Tensor):

        self.actor_optimizer.zero_grad()
        loss = self.actor_loss(logits, q_value)
        assert not loss.isnan().any()
        loss.mean(0).backward()
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

        reward, next_masks, next_states, done = self.push_moves(moves)

        for i in range(self.batchsize):
            entry = ReplayMemory(
                move=moves[i],
                reward=reward[i],
                from_state=from_state[i],
                to_state=next_states[i],
                from_move_mask=from_mask[i],
                to_move_mask=next_masks[i],
                done=done[i],
            )
            self.replay_buffer.add(entry)

    def push_moves(self, moves: torch.Tensor):

        rewards = []
        next_masks = []
        next_states = []
        done = []

        for env, move in zip(self.environments, moves):
            move = env.into_move(move)
            reward, finished = env.play(move)

            next_masks.append(env.legal_move_mask())
            rewards.append(reward)
            next_states.append(env.state())
            done.append(finished)

            if finished:
                env.reset()

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_masks = torch.stack(next_masks).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        done = torch.tensor(done).to(self.device)

        return (rewards, next_masks, next_states, done)

    def state(self) -> torch.Tensor:
        return torch.stack(
            [env.state() for env in self.environments],
        )

    def legal_move_mask(self) -> torch.Tensor:
        return torch.stack(
            [env.legal_move_mask() for env in self.environments],
        )

    def save_checkpoint(self, path: str):
        ckpt = {
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "tau": self.tau,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "episodes": self.episodes,
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

        self.actor.load_state_dict(ckpt["actor"], strict=strict)
        self.critics.load_state_dict(ckpt["critics"], strict=strict)

        self.update_target(1.0)

        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])

        self.tau = ckpt.get("tau", self.tau)

        self.alpha = ckpt.get("alpha", self.alpha)
        self.gamma = ckpt.get("gamma", self.gamma)
        self.episodes = ckpt.get("episodes", self.episodes)

        rng = ckpt["rng"]

        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])

        # rbuffer_path = os.path.join(dir, "replay_buffer.ckpt")
        # print(rbuffer_path)
        # self.replay_buffer.add(ckpt["replay_state"])
        # self.replay_buffer.loads(rbuffer_path)
