"""
Record a demo GIF of the Tetris AI playing.

Renders the board to images using PIL (no pygame needed), then saves as GIF.
Usage: python scripts/record_gif.py
"""

import sys
import pathlib

# Ensure project root is on path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from src.env import TetrisEnv
from src.ai.agent import DoubleDQNAgent
from src.game.pieces import PIECE_TYPES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "session_phase2b" / "best_model.pt"
OUTPUT_PATH = PROJECT_ROOT / "assets" / "demo.gif"
MIN_TOTAL_LINES = 80
TARGET_FPS = 24
FRAME_DURATION_MS = 1000 // TARGET_FPS  # ~41ms per frame

# Visual settings
CELL_SIZE = 24
BOARD_COLS = 10
BOARD_ROWS = 20
SIDEBAR_WIDTH = 160
MARGIN = 2

BOARD_PX_W = BOARD_COLS * CELL_SIZE
BOARD_PX_H = BOARD_ROWS * CELL_SIZE
IMG_W = BOARD_PX_W + SIDEBAR_WIDTH
IMG_H = BOARD_PX_H

# Colors (RGB)
BG_COLOR = (18, 18, 24)
GRID_COLOR = (40, 40, 50)
GRID_LINE_COLOR = (30, 30, 40)
SIDEBAR_BG = (14, 14, 20)
BORDER_COLOR = (80, 80, 100)
TEXT_COLOR = (220, 220, 230)
LABEL_COLOR = (140, 140, 160)
ACCENT_COLOR = (100, 200, 255)

# Tetris-style piece colors (slightly desaturated for polish)
PIECE_COLORS = {
    1: (0, 220, 220),     # I - cyan
    2: (220, 220, 0),     # O - yellow
    3: (160, 0, 200),     # T - purple
    4: (0, 220, 80),      # S - green
    5: (220, 30, 30),     # Z - red
    6: (30, 60, 220),     # J - blue
    7: (220, 140, 0),     # L - orange
}

# Piece name by ID
PIECE_NAMES = {1: "I", 2: "O", 3: "T", 4: "S", 5: "Z", 6: "J", 7: "L"}


def darken(color, amount=50):
    return tuple(max(0, c - amount) for c in color)


def lighten(color, amount=40):
    return tuple(min(255, c + amount) for c in color)


def try_load_font(size):
    """Try to load a monospace font, fall back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def try_load_bold_font(size):
    """Try to load a bold monospace font."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return try_load_font(size)


# Fonts
FONT_SMALL = try_load_font(12)
FONT_MEDIUM = try_load_font(16)
FONT_LARGE = try_load_bold_font(20)
FONT_TITLE = try_load_bold_font(14)


def draw_cell(draw, x, y, color, size=CELL_SIZE):
    """Draw a single filled cell with 3D-style shading."""
    # Main fill
    draw.rectangle([x, y, x + size - 1, y + size - 1], fill=color)
    # Highlight (top-left edges)
    highlight = lighten(color, 60)
    draw.line([(x, y), (x + size - 2, y)], fill=highlight, width=1)
    draw.line([(x, y), (x, y + size - 2)], fill=highlight, width=1)
    # Shadow (bottom-right edges)
    shadow = darken(color, 60)
    draw.line([(x + 1, y + size - 1), (x + size - 1, y + size - 1)], fill=shadow, width=1)
    draw.line([(x + size - 1, y + 1), (x + size - 1, y + size - 1)], fill=shadow, width=1)
    # Inner highlight
    inner = lighten(color, 25)
    draw.rectangle([x + 2, y + 2, x + size - 3, y + size - 3], fill=inner)
    # Dark inner border
    inner_dark = darken(color, 20)
    draw.rectangle([x + 2, y + 2, x + size - 3, y + size - 3], outline=inner_dark)


def draw_piece_preview(draw, piece, x_start, y_start, preview_cell=16):
    """Draw a small piece preview at given position."""
    if piece is None:
        return
    shape = piece["rotations"][0]
    color = PIECE_COLORS.get(piece["id"], (128, 128, 128))
    rows, cols = shape.shape

    # Center in a 4x4 area
    ox = x_start + (4 * preview_cell - cols * preview_cell) // 2
    oy = y_start + (4 * preview_cell - rows * preview_cell) // 2

    for r in range(rows):
        for c in range(cols):
            if shape[r, c] != 0:
                px = ox + c * preview_cell
                py = oy + r * preview_cell
                draw_cell(draw, px, py, color, size=preview_cell)


def render_frame(env, total_lines, episode, episode_lines):
    """Render a single frame as a PIL Image."""
    img = Image.new("RGB", (IMG_W, IMG_H), BG_COLOR)
    draw = ImageDraw.Draw(img)

    game = env.game
    grid = game.board.get_grid()
    buffer_rows = game.board.height - BOARD_ROWS

    # --- Draw board ---
    # Empty cells with subtle grid
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            cell_val = grid[buffer_rows + row, col]

            if cell_val != 0:
                color = PIECE_COLORS.get(int(cell_val), (128, 128, 128))
                draw_cell(draw, x, y, color)
            else:
                # Empty cell
                draw.rectangle([x, y, x + CELL_SIZE - 1, y + CELL_SIZE - 1], fill=GRID_COLOR)
                draw.rectangle([x, y, x + CELL_SIZE - 1, y + CELL_SIZE - 1], outline=GRID_LINE_COLOR)

    # Board border
    draw.rectangle([0, 0, BOARD_PX_W - 1, BOARD_PX_H - 1], outline=BORDER_COLOR, width=2)

    # --- Draw sidebar ---
    sx = BOARD_PX_W
    draw.rectangle([sx, 0, IMG_W - 1, IMG_H - 1], fill=SIDEBAR_BG)
    draw.line([(sx, 0), (sx, IMG_H)], fill=BORDER_COLOR, width=2)

    pad = 12
    cx = sx + pad
    cy = 12

    # NEXT piece
    draw.text((cx, cy), "NEXT", fill=LABEL_COLOR, font=FONT_TITLE)
    cy += 20
    # Preview box background
    box_w = 4 * 16 + 8
    box_h = 4 * 16 + 8
    draw.rectangle([cx, cy, cx + box_w, cy + box_h], fill=darken(SIDEBAR_BG, 5), outline=BORDER_COLOR)
    draw_piece_preview(draw, game.next_piece, cx + 4, cy + 4, preview_cell=16)
    cy += box_h + 16

    # HOLD piece
    hold_label = "HOLD"
    draw.text((cx, cy), hold_label, fill=LABEL_COLOR, font=FONT_TITLE)
    cy += 20
    draw.rectangle([cx, cy, cx + box_w, cy + box_h], fill=darken(SIDEBAR_BG, 5), outline=BORDER_COLOR)
    draw_piece_preview(draw, game.held_piece, cx + 4, cy + 4, preview_cell=16)
    cy += box_h + 20

    # Stats
    stats = [
        ("SCORE", str(game.score)),
        ("LINES", str(total_lines)),
        ("EP LINES", str(episode_lines)),
        ("EPISODE", str(episode)),
        ("LEVEL", str(game.level)),
    ]

    for label, value in stats:
        draw.text((cx, cy), label, fill=LABEL_COLOR, font=FONT_SMALL)
        cy += 15
        draw.text((cx, cy), value, fill=ACCENT_COLOR, font=FONT_LARGE)
        cy += 28

    return img


def main():
    print(f"Recording demo GIF...")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Target: {MIN_TOTAL_LINES}+ total lines cleared")
    print(f"  FPS: {TARGET_FPS}")
    print()

    # Setup
    env = TetrisEnv(board_width=10, board_height=30)
    agent = DoubleDQNAgent(
        input_channels=4, board_height=20, board_width=10,
        num_actions=80, device="cpu"
    )
    agent.load(str(MODEL_PATH))
    agent.policy_net.eval()
    print("Model loaded successfully.")

    frames = []
    total_lines = 0
    episode = 0
    max_episodes = 50  # Safety cap

    while total_lines < MIN_TOTAL_LINES and episode < max_episodes:
        obs = env.reset()
        mask = env.get_valid_mask()
        done = False
        episode += 1
        ep_lines = 0

        # Capture the initial frame
        frames.append(render_frame(env, total_lines, episode, ep_lines))

        step_count = 0
        while not done:
            action = agent.select_action(obs, epsilon=0.0, valid_mask=mask)
            obs, reward, done, info = env.step(action)
            mask = info["valid_mask"]
            lines_this_step = info["lines_cleared"]
            ep_lines += lines_this_step
            total_lines += lines_this_step
            step_count += 1

            # Capture every step (each step = one piece placed)
            frames.append(render_frame(env, total_lines, episode, ep_lines))

            # If we've hit our target mid-game, keep playing until game over
            # for a clean ending (but cap at a reasonable amount)
            if total_lines >= MIN_TOTAL_LINES and step_count > 50:
                break

        print(f"  Episode {episode}: {ep_lines} lines (total: {total_lines})")

        # Add a few freeze frames at game over
        if done:
            for _ in range(TARGET_FPS):  # 1 second pause
                frames.append(frames[-1].copy())

    print(f"\nRecording complete: {len(frames)} frames across {episode} episodes")
    print(f"Total lines cleared: {total_lines}")

    # Save as GIF
    print(f"Saving GIF to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Quantize to 128 colors for smaller file size
    quantized_frames = []
    for f in frames:
        quantized_frames.append(f.quantize(colors=128, method=Image.Quantize.MEDIANCUT))

    quantized_frames[0].save(
        str(OUTPUT_PATH),
        save_all=True,
        append_images=quantized_frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
    )

    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"GIF saved: {OUTPUT_PATH} ({file_size_mb:.1f} MB)")

    # If too large, reduce further
    if file_size_mb > 10:
        print("GIF too large (>10MB), reducing to 64 colors...")
        quantized_frames = []
        for f in frames:
            quantized_frames.append(f.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
        quantized_frames[0].save(
            str(OUTPUT_PATH),
            save_all=True,
            append_images=quantized_frames[1:],
            duration=FRAME_DURATION_MS,
            loop=0,
            optimize=True,
        )
        file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        print(f"Reduced GIF: {file_size_mb:.1f} MB")

    if file_size_mb > 10:
        print("Still too large. Skipping every other frame...")
        reduced = frames[::2]
        quantized_frames = []
        for f in reduced:
            quantized_frames.append(f.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
        quantized_frames[0].save(
            str(OUTPUT_PATH),
            save_all=True,
            append_images=quantized_frames[1:],
            duration=FRAME_DURATION_MS * 2,
            loop=0,
            optimize=True,
        )
        file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        print(f"Final GIF: {file_size_mb:.1f} MB")

    print("Done!")


if __name__ == "__main__":
    main()
