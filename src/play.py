"""
Manual play and AI watch modes.

Provides two modes:
  - play_manual: Human plays Tetris with keyboard controls.
  - play_ai: A trained AI agent plays while the user watches.

Both modes use the TetrisRenderer for visualization.
"""

from __future__ import annotations

import pathlib
from typing import Any

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]

from src.game.tetris import TetrisGame, Action
from src.renderer import TetrisRenderer


# ── Keyboard mapping for manual play ─────────────────────────────────────
# Arrow keys for movement, Z/X for rotation, Space for hard drop, C for hold
KEY_MAP: dict[int, Action] = {}
if pygame is not None:
    KEY_MAP = {
        pygame.K_LEFT: Action.LEFT,
        pygame.K_RIGHT: Action.RIGHT,
        pygame.K_DOWN: Action.SOFT_DROP,
        pygame.K_SPACE: Action.HARD_DROP,
        pygame.K_z: Action.ROTATE_CCW,
        pygame.K_x: Action.ROTATE_CW,
        pygame.K_UP: Action.ROTATE_CW,
        pygame.K_c: Action.HOLD,
    }


def play_manual(config: dict[str, Any]) -> None:
    """Run the game in manual (human) play mode.

    The player uses keyboard controls:
      - Left/Right arrow: move piece
      - Down arrow: soft drop
      - Space: hard drop
      - Z: rotate counter-clockwise
      - X / Up arrow: rotate clockwise
      - C: hold piece
      - Escape / close window: quit

    The game runs at the configured FPS with gravity and scoring.

    Args:
        config: Config dict loaded from hyperparams.yaml.
    """
    if pygame is None:
        raise ImportError("pygame is required for play mode. Install it: pip install pygame")

    board_width = config.get("board_width", 10)
    board_height = config.get("board_height", 30)
    cell_size = config.get("cell_size", 30)
    fps = config.get("fps", 60)
    visible_height = config.get("visible_height", 20)

    game = TetrisGame(board_width, board_height)
    renderer = TetrisRenderer(game, cell_size=cell_size, visible_height=visible_height)
    # Force renderer init before event loop (pygame must be initialized for event.get())
    renderer.render(fps)

    game.reset()
    running = True

    # DAS (Delayed Auto Shift) parameters for smooth movement
    das_delay = 10  # frames before auto-repeat starts
    das_rate = 3    # frames between auto-repeats
    das_keys = {pygame.K_LEFT: 0, pygame.K_RIGHT: 0, pygame.K_DOWN: 0}
    das_active = {pygame.K_LEFT: False, pygame.K_RIGHT: False, pygame.K_DOWN: False}

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

                if game.game_over:
                    if event.key == pygame.K_r:
                        game.reset()
                    continue

                if event.key in KEY_MAP:
                    action = KEY_MAP[event.key]
                    game.step(action)
                    # Start DAS tracking for held keys
                    if event.key in das_keys:
                        das_keys[event.key] = 0
                        das_active[event.key] = False

            elif event.type == pygame.KEYUP:
                if event.key in das_keys:
                    das_keys[event.key] = 0
                    das_active[event.key] = False

        if not running:
            break

        # Handle held keys with DAS for movement
        if not game.game_over:
            keys = pygame.key.get_pressed()
            for key in das_keys:
                if keys[key]:
                    das_keys[key] += 1
                    if das_keys[key] >= das_delay:
                        if not das_active[key]:
                            das_active[key] = True
                            das_keys[key] = das_delay  # reset counter at activation
                        elif (das_keys[key] - das_delay) % das_rate == 0:
                            if key in KEY_MAP:
                                game.step(KEY_MAP[key])

            # Apply gravity tick (even without action, game needs to tick)
            # We pass a "no-op" by just doing a gravity-only step
            # Actually, step already applies gravity with each call.
            # But we need gravity to apply even when player does nothing.
            # The step() function applies gravity each call, so we just
            # need to call step with some neutral action periodically.
            # Since there's no NOOP action, we handle gravity separately.
            game._apply_gravity()

            # Check lock delay even without action
            if game.current_piece is not None:
                can_move_down = game.board.is_valid_position(
                    game.current_piece,
                    game.current_x,
                    game.current_y + 1,
                    game.current_rotation,
                )
                if not can_move_down:
                    game._lock_delay_counter += 1
                    if game._lock_delay_counter >= 30:
                        lines = game._lock_piece()
                        game._lines_cleared_last = lines
                        score_gained = game._calculate_score(lines)
                        game.score += score_gained
                        if lines > 0:
                            game.total_lines += lines
                            game._update_level()
                        if not game.game_over:
                            if not game._spawn_piece():
                                game.game_over = True

        # Render
        renderer.render(fps)

        # Draw game over overlay
        if game.game_over:
            _draw_game_over_overlay(renderer)

    renderer.close()


def _draw_game_over_overlay(renderer: TetrisRenderer) -> None:
    """Draw a semi-transparent game over overlay with restart instructions."""
    overlay = pygame.Surface(
        (renderer.board_pixel_width, renderer.board_pixel_height), pygame.SRCALPHA
    )
    overlay.fill((0, 0, 0, 150))
    renderer.screen.blit(overlay, (0, 0))

    font_large = pygame.font.SysFont("monospace", 36, bold=True)
    font_small = pygame.font.SysFont("monospace", 18)

    text_go = font_large.render("GAME OVER", True, (255, 50, 50))
    text_restart = font_small.render("Press R to restart", True, (255, 255, 255))
    text_quit = font_small.render("Press ESC to quit", True, (200, 200, 200))

    cx = renderer.board_pixel_width // 2
    cy = renderer.board_pixel_height // 2

    renderer.screen.blit(text_go, (cx - text_go.get_width() // 2, cy - 40))
    renderer.screen.blit(text_restart, (cx - text_restart.get_width() // 2, cy + 10))
    renderer.screen.blit(text_quit, (cx - text_quit.get_width() // 2, cy + 40))

    pygame.display.flip()


def play_ai(
    config: dict[str, Any],
    model_path: str | pathlib.Path,
    max_episodes: int = 0,
    watch_fps: int = 30,
) -> None:
    """Run the game with a trained AI agent playing (afterstate-based).

    Args:
        config: Config dict loaded from hyperparams.yaml.
        model_path: Path to a saved model checkpoint (.pt file).
        max_episodes: Stop after N episodes (0 = infinite).
        watch_fps: Frames per second for rendering.
    """
    if pygame is None:
        raise ImportError("pygame is required for watch mode. Install it: pip install pygame")

    from src.env import TetrisEnv
    from src.ai.agent import AfterstateAgent

    board_width = config.get("board_width", 10)
    board_height = config.get("board_height", 30)
    cell_size = config.get("cell_size", 30)
    visible_height = config.get("visible_height", 20)

    env = TetrisEnv(board_width, board_height)
    game = env.game
    renderer = TetrisRenderer(game, cell_size=cell_size, visible_height=visible_height)
    renderer.render(watch_fps)

    agent = AfterstateAgent(
        input_channels=2,
        board_height=20,
        board_width=board_width,
    )
    agent.load(model_path)

    running = True
    ep_num = 0
    while running:
        env.reset()
        done = False
        ep_num += 1
        ep_reward = 0.0
        ep_lines = 0
        ep_steps = 0

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break

            if not running:
                break

            afterstates_info = env.get_afterstates()
            if not afterstates_info:
                done = True
                break

            action, _ = agent.select_action(afterstates_info, epsilon=0.0)
            _, reward, done, info = env.step(action)
            ep_reward += reward
            ep_lines += info.get("lines_cleared", 0)
            ep_steps += 1
            renderer.render(watch_fps)

        if done:
            print(
                f"Episode {ep_num}"
                + (f"/{max_episodes}" if max_episodes else "")
                + f" | Lines: {ep_lines} | Reward: {ep_reward:.1f} | Steps: {ep_steps}"
            )

        if max_episodes and ep_num >= max_episodes:
            if done:
                _draw_game_over_overlay(renderer)
                pygame.time.wait(2000)
            break

        if running and done:
            _draw_game_over_overlay(renderer)
            wait_start = pygame.time.get_ticks()
            waiting = True
            while waiting and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False
                        else:
                            waiting = False
                if pygame.time.get_ticks() - wait_start > 2000:
                    waiting = False

    renderer.close()
