"""
Entry point for the Tetris AI project.

Supports three modes:
  - train: Train the Double DQN agent.
  - play:  Play Tetris manually with keyboard controls.
  - watch: Watch a trained AI agent play.

Usage:
    python main.py --mode train
    python main.py --mode train --config config/hyperparams.yaml
    python main.py --mode play
    python main.py --mode watch --model checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml


def load_config(config_path: str | pathlib.Path) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dict of configuration key-value pairs.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = pathlib.Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with mode, config, and model attributes.
    """
    parser = argparse.ArgumentParser(
        description="Tetris AI — Train a Double DQN agent or play Tetris.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "play", "watch"],
        default="train",
        help="Run mode: 'train' (train the agent), 'play' (manual play), 'watch' (AI plays).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/hyperparams.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a trained model checkpoint (required for 'watch' mode).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (for 'train' mode).",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for organizing checkpoints/logs (default: auto-generated timestamp).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to play in 'watch' mode (0 = infinite).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for 'watch' mode rendering (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point: parse args, load config, and dispatch to the selected mode."""
    args = parse_args()
    config = load_config(args.config)

    if args.mode == "train":
        from src.train import train
        import datetime
        session_id = args.session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        train(config, resume_path=args.resume, session_id=session_id)

    elif args.mode == "play":
        from src.play import play_manual
        play_manual(config)

    elif args.mode == "watch":
        if args.model is None:
            print("Error: --model is required for 'watch' mode.", file=sys.stderr)
            sys.exit(1)
        from src.play import play_ai
        play_ai(config, args.model, max_episodes=args.episodes, watch_fps=args.fps)

    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
