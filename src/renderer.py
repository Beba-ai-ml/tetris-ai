"""
Pygame renderer for the Tetris game.

Draws the board grid, active piece, ghost piece (optional), next piece
preview, held piece preview, and a sidebar with score / level / lines
information.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]

from src.game.tetris import TetrisGame
from src.game.pieces import PIECE_TYPES


# ── Color constants ───────────────────────────────────────────────────────
BACKGROUND_COLOR = (30, 30, 30)
GRID_LINE_COLOR = (60, 60, 60)
BORDER_COLOR = (200, 200, 200)
TEXT_COLOR = (255, 255, 255)
GHOST_ALPHA = 80  # transparency for ghost piece (0-255)
SIDEBAR_BG_COLOR = (20, 20, 20)
EMPTY_CELL_COLOR = (40, 40, 40)

# ── Piece ID -> RGB color mapping (built from PIECE_TYPES at import) ─────
PIECE_COLORS: dict[int, tuple[int, int, int]] = {
    piece["id"]: piece["color"] for piece in PIECE_TYPES
}


class TetrisRenderer:
    """Pygame-based renderer for visualizing a Tetris game.

    The window is divided into:
      - Left: main board area (cell_size * board_width) x (cell_size * visible_height)
      - Right: sidebar with next piece, held piece, score, level, lines

    Attributes:
        game: Reference to the TetrisGame being rendered.
        cell_size: Pixel size of each grid cell.
        visible_height: Number of visible rows (total height minus buffer).
        board_pixel_width: Pixel width of the board area.
        board_pixel_height: Pixel height of the board area.
        sidebar_width: Pixel width of the sidebar.
        window_width: Total window width.
        window_height: Total window height.
        screen: Pygame display surface (created on first render).
    """

    # Sidebar dimensions
    SIDEBAR_WIDTH_CELLS: int = 7  # sidebar width in cell units

    def __init__(
        self,
        game: TetrisGame,
        cell_size: int = 30,
        visible_height: int = 20,
    ) -> None:
        """Initialize the renderer.

        Does NOT create the Pygame window yet — that happens on the first
        call to render(), so headless environments don't open a window.

        Args:
            game: The TetrisGame instance to render.
            cell_size: Size of each grid cell in pixels.
            visible_height: Number of visible rows to display (excludes buffer zone).
        """
        if pygame is None:
            raise ImportError("pygame is required for rendering. Install it: pip install pygame")

        self.game = game
        self.cell_size = cell_size
        self.visible_height = visible_height

        self.board_pixel_width = cell_size * game.board.width
        self.board_pixel_height = cell_size * visible_height
        self.sidebar_width = cell_size * self.SIDEBAR_WIDTH_CELLS
        self.window_width = self.board_pixel_width + self.sidebar_width
        self.window_height = self.board_pixel_height

        self.screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._initialized: bool = False

    def render(self, fps: int = 60) -> None:
        """Draw the current game state to the screen.

        Initializes Pygame on the first call. Draws the board, active piece,
        ghost piece, next/held piece previews, and score sidebar.

        Args:
            fps: Target frames per second for the display clock.
        """
        if not self._initialized:
            self._init_pygame()

        self.screen.fill(BACKGROUND_COLOR)
        self._draw_board()
        self._draw_ghost_piece()
        self._draw_current_piece()
        self._draw_sidebar()

        # Draw border around the board
        pygame.draw.rect(
            self.screen,
            BORDER_COLOR,
            (0, 0, self.board_pixel_width, self.board_pixel_height),
            2,
        )

        pygame.display.flip()
        self._clock.tick(fps)

    def _init_pygame(self) -> None:
        """Initialize Pygame display, clock, and font.

        Called once on the first render() invocation.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Tetris AI")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 20)
        self._small_font = pygame.font.SysFont("monospace", 14)
        self._initialized = True

    def _draw_board(self) -> None:
        """Draw the board grid with filled cells and grid lines.

        Only draws the visible rows (buffer zone is hidden). Each filled
        cell is colored according to its piece ID.
        """
        grid = self.game.board.grid
        buffer_rows = self.game.board.height - self.visible_height

        for row in range(self.visible_height):
            for col in range(self.game.board.width):
                cell_value = grid[buffer_rows + row, col]
                x = col * self.cell_size
                y = row * self.cell_size

                if cell_value != 0:
                    color = PIECE_COLORS.get(int(cell_value), (128, 128, 128))
                    pygame.draw.rect(
                        self.screen, color, (x, y, self.cell_size, self.cell_size)
                    )
                    # Draw a slightly darker border for 3D effect
                    darker = tuple(max(0, c - 40) for c in color)
                    pygame.draw.rect(
                        self.screen, darker, (x, y, self.cell_size, self.cell_size), 1
                    )
                else:
                    pygame.draw.rect(
                        self.screen, EMPTY_CELL_COLOR, (x, y, self.cell_size, self.cell_size)
                    )

                # Grid lines
                pygame.draw.rect(
                    self.screen, GRID_LINE_COLOR, (x, y, self.cell_size, self.cell_size), 1
                )

    def _draw_current_piece(self) -> None:
        """Draw the currently active piece at its position on the board."""
        piece = self.game.current_piece
        if piece is None:
            return

        shape = piece["rotations"][self.game.current_rotation]
        color = piece["color"]
        buffer_rows = self.game.board.height - self.visible_height
        rows, cols = shape.shape

        for r in range(rows):
            for c in range(cols):
                if shape[r, c] != 0:
                    board_row = self.game.current_y + r
                    board_col = self.game.current_x + c
                    screen_row = board_row - buffer_rows
                    if 0 <= screen_row < self.visible_height and 0 <= board_col < self.game.board.width:
                        x = board_col * self.cell_size
                        y = screen_row * self.cell_size
                        pygame.draw.rect(
                            self.screen, color, (x, y, self.cell_size, self.cell_size)
                        )
                        darker = tuple(max(0, cv - 40) for cv in color)
                        pygame.draw.rect(
                            self.screen, darker, (x, y, self.cell_size, self.cell_size), 1
                        )

    def _draw_ghost_piece(self) -> None:
        """Draw the ghost piece (drop preview) with transparency.

        Shows where the current piece would land if hard-dropped.
        """
        piece = self.game.current_piece
        if piece is None:
            return

        # Find the ghost position by dropping the piece as far as possible
        ghost_y = self.game.current_y
        while self.game.board.is_valid_position(
            piece, self.game.current_x, ghost_y + 1, self.game.current_rotation
        ):
            ghost_y += 1

        # Don't draw ghost if it's at the same position as the piece
        if ghost_y == self.game.current_y:
            return

        shape = piece["rotations"][self.game.current_rotation]
        color = piece["color"]
        buffer_rows = self.game.board.height - self.visible_height
        rows, cols = shape.shape

        # Create a transparent surface for the ghost
        ghost_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        ghost_color = (*color, GHOST_ALPHA)

        for r in range(rows):
            for c in range(cols):
                if shape[r, c] != 0:
                    board_row = ghost_y + r
                    board_col = self.game.current_x + c
                    screen_row = board_row - buffer_rows
                    if 0 <= screen_row < self.visible_height and 0 <= board_col < self.game.board.width:
                        x = board_col * self.cell_size
                        y = screen_row * self.cell_size
                        ghost_surface.fill((0, 0, 0, 0))
                        ghost_surface.fill(ghost_color)
                        self.screen.blit(ghost_surface, (x, y))
                        # Draw outline
                        pygame.draw.rect(
                            self.screen, color, (x, y, self.cell_size, self.cell_size), 1
                        )

    def _draw_sidebar(self) -> None:
        """Draw the sidebar with next piece, held piece, score, level, and lines."""
        sidebar_x = self.board_pixel_width
        # Fill sidebar background
        pygame.draw.rect(
            self.screen,
            SIDEBAR_BG_COLOR,
            (sidebar_x, 0, self.sidebar_width, self.window_height),
        )
        # Draw border between board and sidebar
        pygame.draw.line(
            self.screen,
            BORDER_COLOR,
            (sidebar_x, 0),
            (sidebar_x, self.window_height),
            2,
        )

        margin = 15
        x_center = sidebar_x + margin

        # Next piece preview
        self._draw_piece_preview(
            self.game.next_piece,
            x_center,
            20,
            "NEXT",
        )

        # Held piece preview
        hold_label = "HOLD"
        if not self.game.can_hold:
            hold_label = "HOLD (used)"
        self._draw_piece_preview(
            self.game.held_piece,
            x_center,
            160,
            hold_label,
        )

        # Score, level, lines text
        text_x = x_center
        text_y = 310

        self._draw_text("SCORE", text_x, text_y)
        self._draw_text(str(self.game.score), text_x, text_y + 25)

        text_y += 65
        self._draw_text("LEVEL", text_x, text_y)
        self._draw_text(str(self.game.level), text_x, text_y + 25)

        text_y += 65
        self._draw_text("LINES", text_x, text_y)
        self._draw_text(str(self.game.total_lines), text_x, text_y + 25)

    def _draw_piece_preview(
        self,
        piece: dict | None,
        x_offset: int,
        y_offset: int,
        label: str,
    ) -> None:
        """Draw a small piece preview (for next/held piece display).

        Args:
            piece: Piece dict to preview, or None (draws empty box).
            x_offset: Pixel X position for the preview box.
            y_offset: Pixel Y position for the preview box.
            label: Text label to display above the preview (e.g., 'NEXT').
        """
        preview_cell = self.cell_size * 2 // 3  # smaller cells for preview
        box_size = preview_cell * 5

        # Label
        self._draw_text(label, x_offset, y_offset)

        # Preview box background
        box_y = y_offset + 25
        pygame.draw.rect(
            self.screen,
            EMPTY_CELL_COLOR,
            (x_offset, box_y, box_size, box_size),
        )
        pygame.draw.rect(
            self.screen,
            BORDER_COLOR,
            (x_offset, box_y, box_size, box_size),
            1,
        )

        if piece is None:
            return

        shape = piece["rotations"][0]
        color = piece["color"]
        rows, cols = shape.shape

        # Center the piece in the preview box
        piece_pixel_w = cols * preview_cell
        piece_pixel_h = rows * preview_cell
        offset_x = x_offset + (box_size - piece_pixel_w) // 2
        offset_y = box_y + (box_size - piece_pixel_h) // 2

        for r in range(rows):
            for c in range(cols):
                if shape[r, c] != 0:
                    px = offset_x + c * preview_cell
                    py = offset_y + r * preview_cell
                    pygame.draw.rect(
                        self.screen, color, (px, py, preview_cell, preview_cell)
                    )
                    darker = tuple(max(0, cv - 40) for cv in color)
                    pygame.draw.rect(
                        self.screen, darker, (px, py, preview_cell, preview_cell), 1
                    )

    def _draw_text(self, text: str, x: int, y: int, color: tuple[int, int, int] = TEXT_COLOR) -> None:
        """Render text onto the screen.

        Args:
            text: String to display.
            x: Pixel X position.
            y: Pixel Y position.
            color: RGB color tuple for the text.
        """
        surface = self._font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def close(self) -> None:
        """Shut down Pygame and close the window."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
