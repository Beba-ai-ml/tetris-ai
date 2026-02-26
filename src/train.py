"""
Training loop for afterstate V-learning Tetris agent.

Features:
  - Afterstate evaluation: V(s') for each valid placement
  - TD(0) V-learning with target network (polyak averaging)
  - Cosine annealing LR with warm restarts
  - CSV + TensorBoard logging
  - Session-based checkpoint/log organization
  - Hold usage tracking for monitoring
"""

from __future__ import annotations

import csv
import pathlib
import threading
import queue as queue_mod
from typing import Any

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment, misc]

from src.env import TetrisEnv
from src.ai.agent import AfterstateAgent


# ── CSV Logger ───────────────────────────────────────────────────────────────

CSV_FIELDNAMES = [
    "episode",
    "reward",
    "lines",
    "steps",
    "epsilon",
    "loss",
    "grad_norm",
    "mean_v",
    "max_v",
    "lr",
    "best_reward",
    "hold_rate",
]


class CSVLogger:
    """Thread-safe CSV logger that writes rows in a background thread."""

    def __init__(self, path: str | pathlib.Path, fieldnames: list[str]) -> None:
        self._path = pathlib.Path(path)
        self._fieldnames = fieldnames
        self._queue: queue_mod.Queue = queue_mod.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def _writer(self) -> None:
        header_written = self._path.exists() and self._path.stat().st_size > 0
        while True:
            try:
                row = self._queue.get()
                if row is None:
                    break
                with open(self._path, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=self._fieldnames)
                    if not header_written:
                        w.writeheader()
                        header_written = True
                    w.writerow(row)
                    f.flush()
            except Exception as e:
                print(f"CSVLogger error: {e}", flush=True)

    def write(self, row: dict) -> None:
        self._queue.put_nowait(row)

    def shutdown(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)


# ── Training ─────────────────────────────────────────────────────────────────

def get_epsilon(episode: int, config: dict[str, Any]) -> float:
    """Compute epsilon with linear decay."""
    eps_start = config["epsilon_start"]
    eps_end = config["epsilon_end"]
    decay_episodes = config["epsilon_decay_episodes"]
    return max(
        eps_end,
        eps_start - (eps_start - eps_end) * episode / decay_episodes,
    )


def train(
    config: dict[str, Any],
    resume_path: str | None = None,
    session_id: str = "default",
) -> None:
    """Run the full training loop (runs indefinitely until Ctrl+C).

    Uses afterstate V-learning: for each step, enumerate all valid placements,
    simulate each, evaluate V(afterstate) with the network, and pick the action
    with highest immediate_reward + gamma * V(afterstate).

    Replay buffer stores (afterstate, reward, next_afterstate, done) using
    a delayed push pattern:
      V(AS_t) = E[r_{t+1} + gamma * V(AS_{t+1})]
    The reward stored with AS_t is from the NEXT placement (the one that
    created AS_{t+1}), not from the placement that created AS_t.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Session: {session_id}")

    # Session directory structure
    base_dir = pathlib.Path(config.get("checkpoint_dir", "checkpoints"))
    session_dir = base_dir / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    env = TetrisEnv(
        board_width=config.get("board_width", 10),
        board_height=config.get("board_height", 30),
        holes_weight=config.get("holes_weight", 0.3),
        height_weight=config.get("height_weight", 0.03),
        bumpiness_weight=config.get("bumpiness_weight", 0.01),
        game_over_penalty=config.get("game_over_penalty", 25.0),
    )

    agent = _create_agent(config, device)

    # Cosine annealing LR with restarts
    lr_cycle = config.get("lr_cycle_episodes", 50000)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        agent.optimizer,
        T_0=lr_cycle,
        eta_min=config.get("min_learning_rate", 1e-5),
    )

    # TensorBoard — session-specific log dir
    log_dir = pathlib.Path(config.get("log_dir", "runs")) / f"session_{session_id}"
    writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter is not None else None

    # CSV logger
    csv_path = session_dir / f"session_{session_id}.csv"
    csv_logger = CSVLogger(csv_path, CSV_FIELDNAMES)
    print(f"CSV log: {csv_path}")

    best_reward = float("-inf")
    global_step = 0
    start_episode = 0
    batch_size = config.get("batch_size", 256)
    min_replay_size = config.get("min_replay_size", 1000)
    save_freq = config.get("save_freq", 500)
    train_every_n_steps = config.get("train_every_n_steps", 4)
    max_steps = config.get("max_steps_per_episode", 5000)

    # Warmup: collect min_replay_size transitions before training
    warmup_done = False
    print(f"Warming up replay buffer ({min_replay_size} transitions)...")

    # Resume from checkpoint
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        agent.policy_net.load_state_dict(checkpoint["policy_net"])
        agent.target_net.load_state_dict(checkpoint["target_net"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        start_episode = checkpoint.get("episode", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        best_reward = checkpoint.get("best_reward", float("-inf"))
        print(f"Resumed from episode {start_episode}, global_step={global_step}, best_reward={best_reward:.1f}")

    episode = start_episode
    try:
        while True:
            state = env.reset()
            episode_reward = 0.0
            episode_lines = 0
            episode_steps = 0
            episode_holds = 0
            episode_losses: list[float] = []
            episode_grad_norms: list[float] = []
            episode_mean_vs: list[float] = []
            episode_max_vs: list[float] = []
            done = False

            epsilon = get_epsilon(episode, config)
            if not warmup_done:
                epsilon = 1.0

            # Initial afterstate: empty board from reset (before any placement)
            prev_afterstate = env.get_current_afterstate_obs()

            while not done and episode_steps < max_steps:
                # Enumerate and evaluate all valid afterstates
                afterstates_info = env.get_afterstates()

                if not afterstates_info:
                    # No valid actions — game over
                    done = True
                    break

                # Select action
                action, chosen_afterstate = agent.select_action(
                    afterstates_info, epsilon
                )

                # Track hold usage
                if action >= env._place_actions:
                    episode_holds += 1

                # Execute action
                next_state, reward, done, info = env.step(action)

                # Afterstate TD(0) delayed push:
                # V(prev_AS) → reward + gamma * V(chosen_AS)
                # The reward here is from THIS placement (the one that transitions
                # us from prev_afterstate to chosen_afterstate).
                agent.replay_buffer.push(
                    prev_afterstate, reward, chosen_afterstate, done
                )

                prev_afterstate = chosen_afterstate
                state = next_state
                episode_reward += reward
                episode_lines += info.get("lines_cleared", 0)
                episode_steps += 1

                if not warmup_done and len(agent.replay_buffer) >= min_replay_size:
                    warmup_done = True
                    print(f"Warmup complete ({len(agent.replay_buffer)} transitions). Training starts.")

                if warmup_done and global_step % train_every_n_steps == 0:
                    loss, grad_norm, mean_v, max_v = agent.train_step(batch_size)
                    episode_losses.append(loss)
                    episode_grad_norms.append(grad_norm)
                    episode_mean_vs.append(mean_v)
                    episode_max_vs.append(max_v)

                global_step += 1

            if warmup_done:
                scheduler.step(episode % lr_cycle)

            # Compute averages
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            avg_grad_norm = float(np.mean(episode_grad_norms)) if episode_grad_norms else 0.0
            avg_mean_v = float(np.mean(episode_mean_vs)) if episode_mean_vs else 0.0
            avg_max_v = float(np.mean(episode_max_vs)) if episode_max_vs else 0.0
            current_lr = scheduler.get_last_lr()[0]
            hold_rate = episode_holds / max(episode_steps, 1)

            # TensorBoard
            _log_tb(writer, episode, episode_reward, episode_lines,
                    epsilon, avg_loss, episode_steps, avg_grad_norm, current_lr,
                    avg_mean_v, avg_max_v, hold_rate)

            # CSV
            csv_logger.write({
                "episode": episode,
                "reward": f"{episode_reward:.2f}",
                "lines": episode_lines,
                "steps": episode_steps,
                "epsilon": f"{epsilon:.4f}",
                "loss": f"{avg_loss:.6f}",
                "grad_norm": f"{avg_grad_norm:.4f}",
                "mean_v": f"{avg_mean_v:.4f}",
                "max_v": f"{avg_max_v:.4f}",
                "lr": f"{current_lr:.2e}",
                "best_reward": f"{best_reward:.2f}",
                "hold_rate": f"{hold_rate:.4f}",
            })

            # Checkpoints — saved inside session dir
            if warmup_done and episode % save_freq == 0:
                _save_checkpoint(agent, session_dir, episode, global_step, best_reward, is_best=False)

            if warmup_done and episode_reward > best_reward:
                best_reward = episode_reward
                _save_checkpoint(agent, session_dir, episode, global_step, best_reward, is_best=True)

            if episode % 10 == 0:
                phase = "WARMUP" if not warmup_done else ""
                print(
                    f"Episode {episode} | "
                    f"Reward: {episode_reward:.1f} | Lines: {episode_lines} | "
                    f"Epsilon: {epsilon:.3f} | Steps: {episode_steps} | "
                    f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                    f"Hold: {hold_rate:.2f}"
                    + (f" | {phase}" if phase else "")
                )

            episode += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        _save_checkpoint(agent, session_dir, episode, global_step, best_reward, is_best=False)

    csv_logger.shutdown()
    if writer is not None:
        writer.close()
    print(f"Training stopped at episode {episode}. Best reward: {best_reward:.1f}")


def _create_agent(config: dict[str, Any], device: str) -> AfterstateAgent:
    """Instantiate afterstate agent from config."""
    return AfterstateAgent(
        input_channels=2,
        board_height=20,
        board_width=config.get("board_width", 10),
        gamma=config.get("gamma", 0.97),
        learning_rate=config.get("learning_rate", 2.5e-4),
        replay_buffer_size=config.get("replay_buffer_size", 200_000),
        tau=config.get("tau", 0.002),
        device=device,
    )


def _log_tb(
    writer: Any,
    episode: int,
    reward: float,
    lines: int,
    epsilon: float,
    loss: float,
    length: int,
    grad_norm: float = 0.0,
    lr: float = 0.0,
    mean_v: float = 0.0,
    max_v: float = 0.0,
    hold_rate: float = 0.0,
) -> None:
    """Log episode metrics to TensorBoard."""
    if writer is None:
        return
    writer.add_scalar("Reward/episode", reward, episode)
    writer.add_scalar("Lines/episode", lines, episode)
    writer.add_scalar("Epsilon", epsilon, episode)
    writer.add_scalar("Loss/episode", loss, episode)
    writer.add_scalar("Steps/episode", length, episode)
    writer.add_scalar("GradNorm/episode", grad_norm, episode)
    writer.add_scalar("LearningRate", lr, episode)
    writer.add_scalar("VValues/mean", mean_v, episode)
    writer.add_scalar("VValues/max", max_v, episode)
    writer.add_scalar("HoldRate/episode", hold_rate, episode)


def _save_checkpoint(
    agent: AfterstateAgent,
    session_dir: str | pathlib.Path,
    episode: int,
    global_step: int = 0,
    best_reward: float = float("-inf"),
    is_best: bool = False,
) -> None:
    """Save an agent checkpoint with training state."""
    session_dir = pathlib.Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "policy_net": agent.policy_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "episode": episode,
        "global_step": global_step,
        "best_reward": best_reward,
    }

    torch.save(data, session_dir / f"model_ep{episode}.pt")

    if is_best:
        torch.save(data, session_dir / "best_model.pt")
