"""
Afterstate value agent for placement-based Tetris.

Instead of Q(s,a) → 80 values, evaluates V(afterstate) for each valid
placement. For each candidate action, simulates the result on the board,
builds an afterstate observation, and evaluates it with a value network.
Picks the action with highest: immediate_reward + gamma * V(afterstate).

Training uses TD(0) V-learning with a target network (polyak averaging).
No Double DQN needed — there's no max over actions in the Bellman target.
"""

from __future__ import annotations

import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.ai.model import AfterstateNet
from src.ai.replay_buffer import AfterstateReplayBuffer


class AfterstateAgent:
    """Afterstate value agent with TD(0) V-learning."""

    def __init__(
        self,
        input_channels: int = 2,
        board_height: int = 20,
        board_width: int = 10,
        gamma: float = 0.97,
        learning_rate: float = 2.5e-4,
        replay_buffer_size: int = 200_000,
        tau: float = 0.002,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau

        self.policy_net = AfterstateNet(
            input_channels, board_height, board_width
        ).to(self.device)

        self.target_net = AfterstateNet(
            input_channels, board_height, board_width
        ).to(self.device)

        obs_shape = (input_channels, board_height, board_width)
        self.replay_buffer = AfterstateReplayBuffer(replay_buffer_size, obs_shape)

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.loss_fn = nn.SmoothL1Loss()

        self.sync_target_net(hard=True)
        self.target_net.eval()

    def select_action(
        self,
        afterstates_info: list[tuple[int, np.ndarray, float]],
        epsilon: float,
    ) -> tuple[int, np.ndarray]:
        """Select action from afterstate evaluations.

        Args:
            afterstates_info: List of (action_idx, afterstate_obs, immediate_reward)
                from env.get_afterstates().
            epsilon: Exploration probability.

        Returns:
            (action_idx, chosen_afterstate_obs).
        """
        if not afterstates_info:
            return 0, np.zeros((2, 20, 10), dtype=np.float32)

        if random.random() < epsilon:
            idx = random.randrange(len(afterstates_info))
            action_idx, obs, _ = afterstates_info[idx]
            return action_idx, obs

        # Batch evaluate all afterstates
        obs_batch = np.stack([info[1] for info in afterstates_info])
        rewards_batch = np.array([info[2] for info in afterstates_info])

        self.policy_net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)
            values = self.policy_net(obs_t).squeeze(-1)  # (N,)
            rewards_t = torch.tensor(rewards_batch, dtype=torch.float32).to(self.device)
            scores = rewards_t + self.gamma * values
        self.policy_net.train()

        best_idx = scores.argmax().item()
        action_idx, obs, _ = afterstates_info[best_idx]
        return action_idx, obs

    def train_step(self, batch_size: int) -> tuple[float, float, float, float]:
        """V-learning update: V(AS) -> r + gamma * V_target(AS').

        Returns:
            (loss, grad_norm, mean_v, max_v) for logging.
        """
        afterstates, rewards, next_afterstates, dones = self.replay_buffer.sample(
            batch_size, self.device
        )

        # Current V-values
        v_values = self.policy_net(afterstates)  # (batch, 1)

        with torch.no_grad():
            next_v = self.target_net(next_afterstates)  # (batch, 1)
            target_v = rewards + self.gamma * next_v * (1.0 - dones)

        loss = self.loss_fn(v_values, target_v)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=1.0
        )
        self.optimizer.step()

        self.sync_target_net(hard=False)

        mean_v = v_values.mean().item()
        max_v = v_values.max().item()

        return loss.item(), grad_norm.item(), mean_v, max_v

    def sync_target_net(self, hard: bool = False) -> None:
        """Sync target via hard copy or polyak averaging (includes BN buffers)."""
        if hard:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            target_sd = self.target_net.state_dict()
            policy_sd = self.policy_net.state_dict()
            for key in target_sd:
                target_sd[key] = (
                    self.tau * policy_sd[key] + (1.0 - self.tau) * target_sd[key]
                )
            self.target_net.load_state_dict(target_sd)

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | pathlib.Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
