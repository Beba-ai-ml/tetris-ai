# Tetris AI — Technical Documentation

> Target audience: ML engineer who wants to understand or modify the codebase.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Module Reference](#3-module-reference)
   - [main.py](#31-mainpy)
   - [src/env.py](#32-srcenvpy)
   - [src/train.py](#33-srctrainpy)
   - [src/play.py](#34-srcplaypy)
   - [src/renderer.py](#35-srcrendererpy)
   - [src/ai/agent.py](#36-srcaiagentpy)
   - [src/ai/model.py](#37-srcaimodelpy)
   - [src/ai/replay_buffer.py](#38-srcaireplay_bufferpy)
   - [src/game/tetris.py](#39-srcgametetrispy)
   - [src/game/board.py](#310-srcgameboardpy)
   - [src/game/pieces.py](#311-srcgamepiecespy)
4. [Environment](#4-environment)
5. [Model Architecture](#5-model-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [Configuration Reference](#7-configuration-reference)
8. [Session System](#8-session-system)
9. [Game Engine](#9-game-engine)

---

## 1. Project Overview

Tetris AI is a placement-based Tetris agent trained with **Afterstate V-learning** (evolved from Dueling Double DQN in Phase 2b). Instead of frame-by-frame actions, the agent simulates every valid piece placement, evaluates the resulting board state with a CNN value network, and picks the best move: `argmax_a [ reward(a) + gamma * V(afterstate(a)) ]`.

The agent operates over an **80-action discrete space**: 40 direct placements and 40 hold-then-place actions. Only valid placements generate afterstates, so invalid actions are naturally excluded without masking.

**Current results (Phase 3, 101k+ episodes):** Best single game: **1,766 lines** (ep 99,826), best reward: **35,402**, avg lines (last 1K): **117**, 6 games over 1,000 lines. Training ongoing — no plateau observed.

Key design choices:
- **Afterstate V-learning** — evaluates V(board_after_placement) instead of Q(state, action)
- **CNN with asymmetric stride** (preserves column resolution, 2-channel input: board + held piece)
- **TD(0) with target network** and soft (polyak) updates — no Double DQN needed
- **Cosine annealing LR with warm restarts** for continued learning at later episodes
- **Session-based** checkpoint and log organization for multi-run experiments

---

## 2. Architecture

### Component Diagram

```
main.py
  |
  |-- (train mode) --> src/train.py
  |                        |
  |                        |-- src/env.py (TetrisEnv + afterstate generation)
  |                        |       |
  |                        |       |-- src/game/tetris.py (TetrisGame)
  |                        |               |-- src/game/board.py (Board)
  |                        |               |-- src/game/pieces.py (PIECE_TYPES, kick tables)
  |                        |
  |                        |-- src/ai/agent.py (AfterstateAgent)
  |                                |-- src/ai/model.py (AfterstateNet)  x2 (policy + target)
  |                                |-- src/ai/replay_buffer.py (AfterstateReplayBuffer)
  |
  |-- (play mode)  --> src/play.py --> src/game/tetris.py + src/renderer.py
  |
  |-- (watch mode) --> src/play.py --> src/env.py + src/ai/agent.py + src/renderer.py
```

### Data Flow (one training step)

```
env.reset()
    |
    v
env.get_afterstates()
    |  Simulates every valid placement (all rotations x columns x hold/no-hold)
    |  Returns list of (action, afterstate_obs(2,20,10), immediate_reward)
    v
agent.select_action(afterstates_info, epsilon)
    |  epsilon-greedy: random valid action OR argmax [ reward + gamma * V(afterstate) ]
    v
action: int (0-79)
    |
    v
env.step(action)
    |  Execute the chosen placement, clear lines, spawn next piece
    v
(next_obs, reward, done, info)
    |
    v
replay_buffer.push(afterstate, reward, next_afterstate, done)
    |
    v  (every train_every_n_steps global steps, after warmup)
agent.train_step(batch_size)
    |  TD(0) V-learning: V(s) -> r + gamma * V_target(s')
    |  Smooth L1 loss, grad clip 1.0, polyak target update (tau=0.002)
    v
(loss, grad_norm, mean_q, max_q) --> CSV + TensorBoard
```

---

## 3. Module Reference

### 3.1 `main.py`

**Purpose:** Entry point. Parses CLI arguments, loads YAML config, and dispatches to train, play, or watch mode.

#### Functions

| Function | Arguments | Returns | Description |
|---|---|---|---|
| `load_config` | `config_path: str \| Path` | `dict` | Reads and parses the YAML config file with `yaml.safe_load`. Raises `FileNotFoundError` if the path does not exist. |
| `parse_args` | — | `argparse.Namespace` | Defines and parses all CLI flags (see table below). |
| `main` | — | `None` | Calls `load_config`, dispatches to `train()`, `play_manual()`, or `play_ai()` based on `--mode`. Auto-generates a timestamp `session_id` if `--session-id` is not provided. |

#### CLI Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--mode` | `str` | `"train"` | Run mode: `train`, `play`, or `watch`. |
| `--config` | `str` | `"config/hyperparams.yaml"` | Path to the YAML config file. |
| `--model` | `str` | `None` | Path to a `.pt` checkpoint (required for `watch` mode). |
| `--resume` | `str` | `None` | Path to a checkpoint to resume training from. |
| `--session-id` | `str` | `None` | Session identifier. Defaults to `YYYYMMDD_HHMMSS` timestamp. |
| `--episodes` | `int` | `0` | Number of episodes in `watch` mode (0 = infinite). |
| `--fps` | `int` | `30` | Rendering FPS for `watch` mode. |

---

### 3.2 `src/env.py`

**Purpose:** Placement-based Tetris environment for RL. Wraps `TetrisGame`, builds observations, computes shaped rewards, manages the hold action, and exposes a valid action mask. One `step()` = one piece placed.

#### Module-Level Constants

| Name | Value | Description |
|---|---|---|
| `LINE_REWARDS` | `{0: 0.0, 1: 5.0, 2: 15.0, 3: 40.0, 4: 150.0}` | Reward values for clearing 0-4 lines per placement. |

#### Module-Level Functions

| Function | Arguments | Returns | Description |
|---|---|---|---|
| `calculate_reward` | `lines_cleared, holes_delta, height_delta, bumpiness_delta, game_over, piece_locked=False, new_holes_from_lock=0, holes_weight=0.3, height_weight=0.03, bumpiness_weight=0.01, game_over_penalty=35.0` | `float` | Computes the shaped reward for a single placement. Returns `-game_over_penalty` immediately on game over. Otherwise: `LINE_REWARDS[lines_cleared] - holes_weight*holes_delta - height_weight*height_delta - bumpiness_weight*bumpiness_delta + 0.5 (clean placement if new_holes==0) + 0.1 (survival bonus)`. |

#### Class: `TetrisEnv`

```python
TetrisEnv(board_width=10, board_height=30, **kwargs)
```

**Constructor kwargs** (passed through to `calculate_reward`):

| kwarg | Default | Description |
|---|---|---|
| `holes_weight` | `0.3` | Weight for holes delta penalty. |
| `height_weight` | `0.03` | Weight for aggregate height delta penalty. |
| `bumpiness_weight` | `0.01` | Weight for bumpiness delta penalty. |
| `game_over_penalty` | `35.0` | Flat penalty applied when the game ends. |

**Class constants:**

| Attribute | Value | Description |
|---|---|---|
| `VISIBLE_HEIGHT` | `20` | Number of rows in the observation (the visible play area). |
| `NUM_CHANNELS` | `4` | Number of observation channels. |
| `NUM_ROTATIONS` | `4` | Rotation states per piece. |

**Instance attributes (post-init):**

| Attribute | Type | Description |
|---|---|---|
| `game` | `TetrisGame` | Underlying game instance. |
| `action_space_n` | `int` | Total actions: `80` (= `4 rotations × 10 columns × 2`). |
| `observation_shape` | `tuple` | `(4, 20, 10)`. |

**Methods:**

| Method | Arguments | Returns | Description |
|---|---|---|---|
| `reset` | — | `ndarray(4,20,10)` | Resets the game, recomputes baseline board metrics (`_prev_holes`, `_prev_height`, `_prev_bumpiness`), and returns the initial observation. |
| `get_valid_mask` | — | `ndarray(80,) bool` | Returns a boolean mask where `True` = valid action. Actions 0-39 check if the current piece fits at `(rotation, column)`. Actions 40-79 additionally require `can_hold=True` and check the post-hold piece (peeked without mutating state). |
| `step` | `action: int` | `(ndarray, float, bool, dict)` | Executes one placement. If `action >= 40`, performs hold first (via `_perform_hold`), then decodes `rotation = (action % 40) // 10`, `column = (action % 40) % 10`, hard-drops the piece, locks it, spawns the next piece, computes reward, and builds the next observation. Returns `(obs, reward, done, info)`. `info` contains `score`, `level`, `lines_cleared`, `total_lines`, `holes`, `height`, `bumpiness`, `valid_mask`. |
| `_check_piece_fits` | `piece, rotation, column` | `bool` | Checks if the piece fits at the spawn row for a given `(rotation, column)` without mutating state. |
| `_perform_hold` | — | `None` | Executes the hold swap, mutating `game.current_piece`, `game.held_piece`, `game.next_piece`, and setting `game.can_hold = False`. |
| `_build_observation` | — | `ndarray(4,20,10) float32` | Constructs the 4-channel observation cropped to visible rows. Ch0 = board grid (binarized). Ch1 = current piece shape (spawn rotation, top-left). Ch2 = next piece shape. Ch3 = held piece shape (zeros if None). |
| `_render_piece_shape` | `channel: ndarray, shape: ndarray` | `None` | Writes a piece's binarized shape into the top-left corner of a channel array. |
| `_get_board_metrics` | — | `tuple(int,int,int)` | Returns `(holes, aggregate_height, bumpiness)` from the board. |
| `_build_info` | `lines_cleared, valid_mask` | `dict` | Constructs the `info` dict returned from `step()`. |

---

### 3.3 `src/train.py`

**Purpose:** Full training loop for the Dueling Double DQN agent. Handles warmup, epsilon-greedy exploration, cosine annealing LR, soft target updates, and session-organized CSV/TensorBoard logging and checkpointing.

#### Module-Level Constants

| Name | Value | Description |
|---|---|---|
| `CSV_FIELDNAMES` | `["episode", "reward", "lines", "steps", "epsilon", "loss", "grad_norm", "mean_q", "max_q", "lr", "best_reward"]` | Column names written to the per-session CSV log. |

#### Functions

| Function | Arguments | Returns | Description |
|---|---|---|---|
| `get_epsilon` | `episode: int, config: dict` | `float` | Linear epsilon decay: `max(eps_end, eps_start - (eps_start - eps_end) * episode / decay_episodes)`. |
| `train` | `config: dict, resume_path: str \| None, session_id: str` | `None` | Main training loop (runs indefinitely until `KeyboardInterrupt`). See [Training Pipeline](#6-training-pipeline). |
| `_create_agent` | `config: dict, device: str` | `DoubleDQNAgent` | Instantiates a `DoubleDQNAgent` from config values. Computes `num_actions = 4 * board_width * 2 = 80`. |
| `_log_tb` | `writer, episode, reward, lines, epsilon, loss, length, grad_norm, lr, mean_q, max_q` | `None` | Writes 9 scalar metrics to TensorBoard (no-op if `SummaryWriter` is unavailable). |
| `_save_checkpoint` | `agent, session_dir, episode, global_step, best_reward, is_best` | `None` | Saves a `dict` with `policy_net`, `target_net`, `optimizer`, `episode`, `global_step`, `best_reward` to `session_dir/model_ep{episode}.pt`. If `is_best=True`, also saves to `session_dir/best_model.pt`. |

#### Class: `CSVLogger`

```python
CSVLogger(path: str | Path, fieldnames: list[str])
```

Thread-safe CSV logger. Writes rows in a background daemon thread via a `queue.Queue` to avoid blocking the training loop.

| Method | Arguments | Returns | Description |
|---|---|---|---|
| `write` | `row: dict` | `None` | Enqueues a row dict (non-blocking). |
| `shutdown` | — | `None` | Sends sentinel `None` to the queue and joins the thread (timeout: 5 seconds). |

---

### 3.4 `src/play.py`

**Purpose:** Provides `play_manual` (human keyboard play) and `play_ai` (AI agent watch mode), both using `TetrisRenderer` for visualization via pygame.

#### Module-Level Constants

| Name | Value | Description |
|---|---|---|
| `KEY_MAP` | `{K_LEFT: LEFT, K_RIGHT: RIGHT, K_DOWN: SOFT_DROP, K_SPACE: HARD_DROP, K_z: ROTATE_CCW, K_x: ROTATE_CW, K_UP: ROTATE_CW, K_c: HOLD}` | Mapping of pygame key codes to `Action` enum values. Empty dict if pygame is not installed. |

#### Functions

| Function | Arguments | Returns | Description |
|---|---|---|---|
| `play_manual` | `config: dict` | `None` | Opens a pygame window and runs an interactive game. Implements Delayed Auto Shift (DAS): 10-frame delay, 3-frame repeat rate for held movement keys. Press R to restart after game over, Escape to quit. |
| `play_ai` | `config: dict, model_path: str \| Path, max_episodes: int = 0, watch_fps: int = 30` | `None` | Loads a checkpoint into `DoubleDQNAgent`, runs `env.step()` with `epsilon=0.0` (pure greedy), renders via `TetrisRenderer`. Waits 2 seconds (or a keypress) between episodes. Exits after `max_episodes` if nonzero. |
| `_draw_game_over_overlay` | `renderer: TetrisRenderer` | `None` | Draws a semi-transparent black overlay with "GAME OVER", "Press R to restart", and "Press ESC to quit" text. |

**DAS parameters in `play_manual`:**

| Parameter | Value | Description |
|---|---|---|
| `das_delay` | `10` | Frames before auto-repeat activates for held movement keys. |
| `das_rate` | `3` | Frames between auto-repeat events once DAS is active. |

---

### 3.5 `src/renderer.py`

**Purpose:** Pygame-based renderer for visualizing a `TetrisGame`. Draws the board grid, active piece, ghost piece (semi-transparent drop preview), sidebar with next/held piece previews, score, level, and lines.

#### Module-Level Constants

| Name | Value | Description |
|---|---|---|
| `BACKGROUND_COLOR` | `(30, 30, 30)` | Main window background (dark gray). |
| `GRID_LINE_COLOR` | `(60, 60, 60)` | Color of grid cell borders. |
| `BORDER_COLOR` | `(200, 200, 200)` | Board and sidebar border color. |
| `TEXT_COLOR` | `(255, 255, 255)` | Sidebar text color. |
| `GHOST_ALPHA` | `80` | Alpha (0-255) for the ghost piece surface. |
| `SIDEBAR_BG_COLOR` | `(20, 20, 20)` | Sidebar background (near-black). |
| `EMPTY_CELL_COLOR` | `(40, 40, 40)` | Color for empty board cells. |
| `PIECE_COLORS` | `dict[int, tuple]` | Piece ID (1-7) to RGB color, built from `PIECE_TYPES` at import time. |

#### Class: `TetrisRenderer`

```python
TetrisRenderer(game: TetrisGame, cell_size: int = 30, visible_height: int = 20)
```

Pygame is initialized lazily on the first call to `render()` so that importing the class in headless/training environments does not open a window.

**Class constants:**

| Attribute | Value | Description |
|---|---|---|
| `SIDEBAR_WIDTH_CELLS` | `7` | Sidebar width expressed in cell units. |

**Instance attributes (post-init):**

| Attribute | Type | Description |
|---|---|---|
| `board_pixel_width` | `int` | `cell_size * board_width` (300 px at defaults). |
| `board_pixel_height` | `int` | `cell_size * visible_height` (600 px at defaults). |
| `sidebar_width` | `int` | `cell_size * SIDEBAR_WIDTH_CELLS` (210 px at defaults). |
| `window_width` | `int` | `board_pixel_width + sidebar_width` (510 px at defaults). |
| `window_height` | `int` | Same as `board_pixel_height`. |

**Methods:**

| Method | Arguments | Returns | Description |
|---|---|---|---|
| `render` | `fps: int = 60` | `None` | Draws one frame: board, ghost, active piece, sidebar. Calls `_init_pygame()` on first invocation. Calls `pygame.display.flip()` and `clock.tick(fps)`. |
| `close` | — | `None` | Calls `pygame.quit()` and resets `_initialized`. |
| `_init_pygame` | — | `None` | Creates the pygame window, clock, and font. Window caption: `"Tetris AI"`. Fonts: `monospace 20` (main), `monospace 14` (small). |
| `_draw_board` | — | `None` | Iterates visible rows; draws filled cells with piece color and a darker (−40) 3D border edge; draws empty cells in `EMPTY_CELL_COLOR`; overlays `GRID_LINE_COLOR` grid lines. |
| `_draw_current_piece` | — | `None` | Draws the active piece at its current `(current_x, current_y, current_rotation)` position, offset by buffer rows. |
| `_draw_ghost_piece` | — | `None` | Finds the lowest valid row by simulating a hard drop, then draws a transparent (`GHOST_ALPHA=80`) overlay at that position. Skips if ghost position equals current position. |
| `_draw_sidebar` | — | `None` | Draws "NEXT" piece preview (y=20), "HOLD" piece preview (y=160, labeled "HOLD (used)" when `can_hold=False`), and score/level/lines text (starting y=310). |
| `_draw_piece_preview` | `piece, x_offset, y_offset, label` | `None` | Draws a labeled preview box using cells at `2/3` of `cell_size`. Centers the piece shape within a `5×preview_cell` square box. |
| `_draw_text` | `text, x, y, color=TEXT_COLOR` | `None` | Renders a string using the main font and blits it at `(x, y)`. |

---

### 3.6 `src/ai/agent.py`

**Purpose:** Afterstate value agent with TD(0) V-learning. Owns two `AfterstateNet` networks (policy and target), the `AfterstateReplayBuffer`, and the Adam optimizer. Evaluates V(afterstate) for each valid placement and picks the best.

#### Class: `AfterstateAgent`

```python
AfterstateAgent(
    input_channels=2,
    board_height=20,
    board_width=10,
    gamma=0.97,
    learning_rate=2.5e-4,
    replay_buffer_size=200_000,
    tau=0.002,
    device="cpu",
)
```

**Instance attributes:**

| Attribute | Type | Description |
|---|---|---|
| `policy_net` | `AfterstateNet` | Online value network for afterstate evaluation and gradient updates. |
| `target_net` | `AfterstateNet` | Frozen target network updated via polyak averaging. Always in `eval()` mode. |
| `replay_buffer` | `AfterstateReplayBuffer` | Afterstate transition storage. |
| `optimizer` | `Adam` | `lr=learning_rate`, `weight_decay=1e-4`. |
| `loss_fn` | `SmoothL1Loss` | Huber loss for V-value regression. |
| `gamma` | `float` | Discount factor. |
| `tau` | `float` | Polyak averaging coefficient for target net updates. |

**Methods:**

| Method | Arguments | Returns | Description |
|---|---|---|---|
| `select_action` | `afterstates_info: list[(action, obs, reward)], epsilon: float` | `(int, afterstate_obs)` | Epsilon-greedy over afterstate values. If `random() < epsilon`, returns a random valid action. Otherwise, evaluates V(afterstate) for each candidate with `policy_net.eval()`, picks `argmax [ reward + gamma * V(afterstate) ]`. Returns chosen action and its afterstate obs. |
| `train_step` | `batch_size: int` | `(float, float, float, float)` | Samples a batch. Computes V(afterstate) with `policy_net`. Target: `r + gamma * V_target(next_afterstate) * (1 - done)`. Backpropagates `SmoothL1Loss`, clips gradients at `max_norm=1.0`, steps optimizer, calls `sync_target_net(hard=False)`. Returns `(loss, grad_norm, mean_v, max_v)`. |
| `sync_target_net` | `hard: bool = False` | `None` | Hard copy if `hard=True`. Polyak averaging over all `state_dict` keys (including BatchNorm buffers) if `hard=False`. |
| `save` | `path: str \| Path` | `None` | Saves `{"policy_net": ..., "target_net": ..., "optimizer": ...}` to a `.pt` file. |
| `load` | `path: str \| Path` | `None` | Loads state dicts from a checkpoint. Uses `weights_only=True`. |

---

### 3.7 `src/ai/model.py`

**Purpose:** Afterstate value network. Accepts a `(batch, 2, 20, 10)` tensor (board after placement + held piece) and outputs a scalar V(afterstate).

#### Class: `AfterstateNet`

```python
AfterstateNet(input_channels=2, board_height=20, board_width=10)
```

Inherits from `nn.Module`.

**Methods:**

| Method | Arguments | Returns | Description |
|---|---|---|---|
| `__init__` | `input_channels, board_height, board_width` | — | Defines `conv_layers` (3x Conv+BN+ReLU with asymmetric stride) and `value_head` (FC→ReLU→FC→scalar). Flat size = `64 * (board_height // 4) * board_width = 3200`. |
| `forward` | `x: Tensor(batch, 2, 20, 10)` | `Tensor(batch, 1)` | Passes input through `conv_layers`, flattens, returns `value_head(flat)` — a scalar V(afterstate). |

---

### 3.8 `src/ai/replay_buffer.py`

**Purpose:** Pre-allocated circular replay buffer storing afterstate transitions `(afterstate, reward, next_afterstate, done)`. Uses `uint8` numpy arrays for memory efficiency (afterstates are binarized float32 observations cast to `uint8`).

#### Class: `AfterstateReplayBuffer`

```python
AfterstateReplayBuffer(capacity: int, obs_shape: tuple = (2, 20, 10))
```

**Instance attributes:**

| Attribute | dtype | Shape | Description |
|---|---|---|---|
| `states` | `uint8` | `(capacity, 2, 20, 10)` | Afterstate observations (board after placement + held piece). |
| `next_states` | `uint8` | `(capacity, 2, 20, 10)` | Next afterstate observations. |
| `rewards` | `float32` | `(capacity,)` | Immediate rewards. |
| `dones` | `float32` | `(capacity,)` | Episode termination flags (0.0 or 1.0). |
| `pos` | `int` | — | Current write position (circular). |
| `size` | `int` | — | Current number of stored transitions (capped at `capacity`). |

**Methods:**

| Method | Arguments | Returns | Description |
|---|---|---|---|
| `push` | `state, reward, next_state, done` | `None` | Writes one transition at `pos`, advances `pos` modulo `capacity`, increments `size` up to `capacity`. Casts states to `uint8`. |
| `sample` | `batch_size: int, device: str \| device` | `tuple[Tensor, ...]` | Samples `batch_size` transitions without replacement. Converts `uint8` to `float32` tensors. Returns `(states, rewards, next_states, dones)` on the specified device. |
| `__len__` | — | `int` | Returns `self.size`. |

---

### 3.9 `src/game/tetris.py`

**Purpose:** Game orchestrator. Ties together `Board` and `PIECE_TYPES` into a full Tetris game with SRS rotation, wall kicks, NES-style scoring, hold mechanics, 7-bag randomizer, gravity, and lock delay. Exposes both a frame-by-frame `step(action)` interface (used in manual play) and lower-level methods used directly by `TetrisEnv` for placement-based control.

#### Enum: `Action`

`IntEnum` with values: `LEFT=0, RIGHT=1, ROTATE_CW=2, ROTATE_CCW=3, SOFT_DROP=4, HARD_DROP=5, HOLD=6, NOOP=7`.

#### Module-Level Constants

| Name | Value | Description |
|---|---|---|
| `SCORE_TABLE` | `{1:40, 2:100, 3:300, 4:1200}` | NES-style base scores per line clear count. Multiplied by `(level+1)`. |
| `GRAVITY_TABLE` | `[48, 43, 38, ..., 1]` | Frames per gravity drop at each level (0-29). Level 0 = 48 frames/drop, level 29 = 1 frame/drop. Levels beyond the table use 1 frame. |

#### Class: `TetrisGame`

```python
TetrisGame(board_width: int = 10, board_height: int = 30)
```

**Public attributes:**

| Attribute | Type | Description |
|---|---|---|
| `board` | `Board` | The game board (10 × 30 grid). |
| `score` | `int` | Current NES-style score. |
| `level` | `int` | Current level (`total_lines // 10`). |
| `total_lines` | `int` | Cumulative lines cleared since last reset. |
| `current_piece` | `dict \| None` | Active piece dict (keys: `id`, `name`, `color`, `rotations`). |
| `current_x` | `int` | Column of piece top-left corner. |
| `current_y` | `int` | Row of piece top-left corner. |
| `current_rotation` | `int` | Active rotation state (0-3). |
| `next_piece` | `dict \| None` | Next piece to be spawned. |
| `held_piece` | `dict \| None` | Currently held piece, or `None`. |
| `can_hold` | `bool` | Whether hold is currently allowed (resets to `True` on each new piece). |
| `game_over` | `bool` | Set to `True` when spawn fails. |

**Methods:**

| Method | Arguments | Returns | Description |
|---|---|---|---|
| `reset` | — | `dict` | Clears board, resets all state, refills 7-bag, sets `next_piece`, calls `_spawn_piece()`. Returns `get_state()`. |
| `step` | `action: int` | `(dict, float, bool)` | Frame-by-frame game step for the per-frame `Action` space. Processes action (move/rotate/drop/hold), applies gravity, handles lock delay (30-frame threshold), clears lines, updates score/level. Returns `(state_info, reward, done)`. |
| `get_state` | — | `dict` | Returns a snapshot dict with keys: `board_grid`, `current_piece`, `current_x`, `current_y`, `current_rotation`, `next_piece`, `held_piece`, `can_hold`, `score`, `level`, `lines_cleared`, `total_lines`. |
| `_fill_bag` | — | `None` | Shuffles all 7 `PIECE_TYPES` into `self._bag`. |
| `_next_from_bag` | — | `dict` | Pops a piece from `_bag`, refilling if empty. |
| `_spawn_piece` | — | `bool` | Sets `current_piece = next_piece`, draws next from bag, centers horizontally, positions at spawn row (`buffer_rows - piece_height + 1`), resets `can_hold`. Returns `False` (sets `game_over`) if spawn position is invalid. |
| `_move` | `dx: int, dy: int` | `bool` | Attempts to translate the current piece by `(dx, dy)`. Returns `True` if successful. |
| `_rotate` | `direction: int` | `bool` | Attempts SRS rotation (+1 CW, -1 CCW). Tries each kick offset from the appropriate kick table. Returns `True` if any kick succeeds. |
| `_hard_drop` | — | `int` | Instantly drops piece to its lowest valid row. Returns number of rows dropped. |
| `_hold` | — | `bool` | Swaps current and held piece (or stores current and spawns next if no held piece). Sets `can_hold = False`. Returns `False` if `can_hold` is already `False`. |
| `_lock_piece` | — | `int` | Calls `board.place_piece()`, then `board.clear_lines()`. Clears `current_piece`. Returns lines cleared. |
| `_calculate_score` | `lines_cleared: int` | `int` | Returns `SCORE_TABLE[lines_cleared] * (level + 1)`. Returns `0` for 0 lines. |
| `_update_level` | — | `None` | Sets `level = total_lines // 10`. |
| `_get_gravity_frames` | — | `int` | Returns `GRAVITY_TABLE[level]` or `1` for levels beyond the table. |
| `_apply_gravity` | — | `None` | Increments `_gravity_counter`. When it reaches `_get_gravity_frames()`, resets counter and calls `_move(0, 1)`. |

---

### 3.10 `src/game/board.py`

**Purpose:** 10×30 Tetris grid (int8 numpy array). Handles collision detection, piece placement, line clearing, and board metric computation (holes, aggregate height, bumpiness). Row 0 is the top of the board; rows 0-9 are a hidden buffer zone; rows 10-29 are the visible play area.

#### Class: `Board`

```python
Board(width: int = 10, height: int = 30)
```

**Instance attributes:**

| Attribute | Type | Description |
|---|---|---|
| `width` | `int` | Number of columns (default 10). |
| `height` | `int` | Total rows including buffer zone (default 30). |
| `grid` | `ndarray(30, 10) int8` | `0` = empty, `1-7` = piece type ID. |

**Methods:**

| Method | Arguments | Returns | Description |
|---|---|---|---|
| `is_valid_position` | `piece, x, y, rotation` | `bool` | Checks every filled cell of the piece shape against board boundaries and existing blocks. Returns `False` if any cell is out of bounds or overlaps a nonzero grid cell. |
| `place_piece` | `piece, x, y, rotation` | `None` | Writes `piece["id"]` into `grid` at each filled cell. Does not validate position. |
| `clear_lines` | — | `int` | Identifies fully filled rows (`np.all(row != 0)`), removes them with a boolean mask, prepends empty rows at the top, and returns the count of lines cleared (0-4). |
| `get_grid` | — | `ndarray(30,10) int8` | Returns `grid.copy()`. |
| `get_holes` | — | `int` | Counts holes: empty cells with at least one filled cell above them in the same column. Iterates column by column, top to bottom. |
| `get_aggregate_height` | — | `int` | Returns sum of `get_column_heights()`. |
| `get_bumpiness` | — | `int` | Returns sum of absolute differences between adjacent column heights: `sum(|h[i] - h[i+1]|)`. |
| `get_column_heights` | — | `list[int]` | For each column, finds the topmost filled row and returns `height - row_index`. Empty columns return `0`. |
| `is_game_over` | — | `bool` | Returns `True` if any cell in rows `buffer_rows` to `buffer_rows+1` (the visible spawn area) is nonzero. Not called by `TetrisEnv`; game over is determined by `_spawn_piece()` failure. |
| `reset` | — | `None` | Fills `grid` with zeros. |

---

### 3.11 `src/game/pieces.py`

**Purpose:** All tetromino definitions and SRS wall-kick tables. Each piece is a `dict` with keys `id` (int 1-7), `name` (str), `color` (RGB tuple), and `rotations` (list of 4 numpy arrays). Rotation order: `[0=spawn, 1=CW, 2=180°, 3=CCW]`.

#### Piece Definitions

| Constant | `id` | Name | Color |
|---|---|---|---|
| `I_PIECE` | 1 | I | Cyan `(0,255,255)` |
| `O_PIECE` | 2 | O | Yellow `(255,255,0)` |
| `T_PIECE` | 3 | T | Purple `(128,0,128)` |
| `S_PIECE` | 4 | S | Green `(0,255,0)` |
| `Z_PIECE` | 5 | Z | Red `(255,0,0)` |
| `J_PIECE` | 6 | J | Blue `(0,0,255)` |
| `L_PIECE` | 7 | L | Orange `(255,165,0)` |

`PIECE_TYPES: list[dict]` — ordered list of all 7 pieces, used by the 7-bag randomizer.

#### Bounding Box Dimensions

| Piece | Bounding box (rows × cols) |
|---|---|
| I | 4 × 4 |
| O | 2 × 2 (all 4 rotations identical) |
| T, S, Z, J, L | 3 × 3 |

#### SRS Kick Tables

| Constant | Applies to |
|---|---|
| `KICK_TABLE_JLSTZ` | J, L, S, T, Z pieces |
| `KICK_TABLE_I` | I piece |

Each table is `dict[int, dict[int, list[tuple[int,int]]]]` indexed by `[from_rotation][to_rotation]`. Each entry is a list of up to 5 `(dx, dy)` offsets tried in order. `dy` follows the SRS convention: positive = up on screen = subtract from row index.

#### Functions

| Function | Arguments | Returns | Description |
|---|---|---|---|
| `get_kick_offsets` | `piece: dict, from_rot: int, to_rot: int` | `list[tuple[int,int]]` | Returns the kick list for the given piece and rotation transition. O-piece always returns `[(0, 0)]`. I-piece uses `KICK_TABLE_I`; all others use `KICK_TABLE_JLSTZ`. |

---

## 4. Environment

### Observation Space

**Shape:** `(4, 20, 10)`, `dtype=float32`

The observation is cropped to the 20 visible rows (buffer rows 0-9 are excluded). All values are `0.0` or `1.0`.

| Channel | Content | Construction |
|---|---|---|
| `Ch0` (Board) | Locked pieces on the board, binarized. Active piece is NOT included. | `grid[buffer:buffer+20] != 0` cast to float32 |
| `Ch1` (Current piece) | Spawn-rotation shape of the current piece, written to the top-left corner. | `piece["rotations"][0]` binarized |
| `Ch2` (Next piece) | Spawn-rotation shape of the next piece, written to the top-left corner. | `next_piece["rotations"][0]` binarized |
| `Ch3` (Held piece) | Spawn-rotation shape of the held piece, written to the top-left corner. All zeros if no piece is held. | `held_piece["rotations"][0]` binarized or zeros |

The current piece shape is always rendered at spawn rotation (rotation 0), regardless of the piece's actual rotation on the board, because the agent selects rotations through the action encoding — the observation only needs to identify which piece type is active.

### Action Space

**Size:** 80 discrete actions

| Range | Meaning | Rotation decoding | Column decoding |
|---|---|---|---|
| `0 – 39` | Place current piece | `action // 10` (0-3) | `action % 10` (0-9) |
| `40 – 79` | Hold first, then place | `(action - 40) // 10` (0-3) | `(action - 40) % 10` (0-9) |

**Piece is instantly hard-dropped.** The agent does not move pieces frame by frame; it selects the final column and rotation directly.

**Valid action masking:** `TetrisEnv.get_valid_mask()` returns a `bool` ndarray of shape `(80,)`. During action selection and target computation, invalid actions are set to `-inf` in the Q-value tensor so they cannot be selected by `argmax`.

Hold actions (40-79) are masked entirely when `game.can_hold == False` (already held once this piece). When computing hold masks, `get_valid_mask()` peeks at the post-hold piece without mutating game state: if `held_piece` is not `None`, the post-hold piece is `held_piece`; if `held_piece` is `None`, the post-hold piece is `next_piece`.

### Reward Function

The reward for each step is computed by `calculate_reward()` in `src/env.py`. All terms are applied symmetrically: worsening the board is penalized; improving it (e.g., clearing holes) is rewarded.

| Term | Formula | Notes |
|---|---|---|
| Line clear | `LINE_REWARDS[lines_cleared]` | `{0:0.0, 1:5.0, 2:15.0, 3:40.0, 4:150.0}` |
| Game over penalty | `-35.0` | Applied instead of all other terms; no further computation. |
| Hole delta | `-0.3 × (post_holes - pre_holes)` | Negative if holes decreased (reward). |
| Height delta | `-0.03 × (post_height - pre_height)` | Negative if aggregate height decreased (reward). |
| Bumpiness delta | `-0.01 × (post_bumpiness - pre_bumpiness)` | Negative if surface became smoother (reward). |
| Clean placement bonus | `+0.5` | Added when `new_holes_from_lock == 0` (piece locked without creating any new holes). |
| Survival bonus | `+0.1` | Added for every piece successfully placed (not game over). |

`pre_*` values are the board metrics from before the placement. `post_*` are read after the piece is locked and lines are cleared. Deltas are therefore signed: worsening the board yields a positive delta (negative contribution to reward); improving it yields a negative delta (positive contribution).

### Valid Action Masking

Masking is applied in two places:

1. **Action selection** (`agent.select_action`): Invalid Q-values are set to `-inf` before `argmax`. Random exploration samples only from valid indices.
2. **Target computation** (`agent.train_step`): The `next_masks` stored in the replay buffer are applied to `policy_net(next_states)` before the Double DQN action selection step. Invalid next-state actions never contribute to bootstrap targets.

---

## 5. Model Architecture

**Class:** `AfterstateNet` in `src/ai/model.py`

**Input:** `(batch, 2, 20, 10)` float32 tensor — board after placement + held piece

### Convolutional Backbone

| Layer | Type | In channels | Out channels | Kernel | Padding | Stride | Output shape |
|---|---|---|---|---|---|---|---|
| Conv1 | `Conv2d` + `BatchNorm2d` + `ReLU` | 2 | 32 | 3×3 | 1 | (1,1) | (batch, 32, 20, 10) |
| Conv2 | `Conv2d` + `BatchNorm2d` + `ReLU` | 32 | 64 | 3×3 | 1 | **(2,1)** | (batch, 64, 10, 10) |
| Conv3 | `Conv2d` + `BatchNorm2d` + `ReLU` | 64 | 64 | 3×3 | 1 | **(2,1)** | (batch, 64, 5, 10) |

**Asymmetric stride `(2,1)`:** Height is compressed by 4× across Conv2+Conv3 (from 20 to 5), while width (columns) is preserved at 10. This is intentional: column resolution matters for placement decisions.

**Flatten:** `64 × 5 × 10 = 3200`

### Value Head

Single scalar output — no dueling decomposition (meaningless with one output).

| Head | Layers | Output |
|---|---|---|
| Value | `Linear(3200, 512)` → `ReLU` → `Linear(512, 1)` | Scalar `V(afterstate)` |

**Action selection:** The agent doesn't use the network for action selection directly. Instead, it simulates every valid placement via `env.get_afterstates()`, evaluates each resulting board state, and picks: `argmax_a [ reward(a) + gamma * V(afterstate(a)) ]`.

**No Dropout, no masked advantage.** Weight decay (L2=1e-4) provides regularization.

### Parameter Count

Approximate:
- Conv backbone: ~370K parameters (2-channel input vs 4-channel saves ~600 params in Conv1)
- Value head: ~1.6M parameters
- **Total: ~2.0M parameters** (down from ~3.7M in the old Dueling DQN)

---

## 6. Training Pipeline

### Initialization

1. Device is set to CUDA if available, otherwise CPU.
2. Session directory is created at `checkpoints/session_{session_id}/`.
3. `TetrisEnv` is instantiated with all reward weights from config.
4. `AfterstateAgent` is instantiated. `target_net` is hard-synced from `policy_net` at startup and set to `eval()`.
5. `CosineAnnealingWarmRestarts` scheduler is created with `T_0=lr_cycle_episodes` (50000) and `eta_min=min_learning_rate` (1e-5).
6. `SummaryWriter` is created at `runs/session_{session_id}/` (skipped if TensorBoard is not installed).
7. `CSVLogger` is started at `checkpoints/session_{session_id}/session_{session_id}.csv`.

### Warmup Phase

Before any gradient updates, the loop runs with `epsilon=1.0` (pure random exploration from valid actions) until `len(replay_buffer) >= min_replay_size` (default: 1000 transitions). Training does not start until warmup is complete.

### Episode Loop

For each episode:

1. `env.reset()` → initial board state.
2. `epsilon = get_epsilon(episode, config)` (linear decay, overridden to `1.0` during warmup).
3. Inner loop (up to `max_steps_per_episode=5000`):
   - `env.get_afterstates()` → list of `(action, afterstate_obs, immediate_reward)` for all valid placements
   - `agent.select_action(afterstates_info, epsilon)` → `action` (epsilon-greedy over afterstate values)
   - `env.step(action)` → `(next_obs, reward, done, info)`
   - `replay_buffer.push(afterstate, reward, next_afterstate, done)`
   - Every `train_every_n_steps=4` global steps (after warmup): `agent.train_step(batch_size=256)`
   - `global_step += 1`
4. After episode: `scheduler.step(episode % lr_cycle)`.
5. Log to TensorBoard and CSV (averages of loss, grad_norm, mean_v, max_v over the episode).
6. Checkpoint if `episode % save_freq == 0` (500) or if `episode_reward > best_reward`.

### Epsilon Decay

```
epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * episode / epsilon_decay_episodes)
```

With current config: linear decay from `1.0` to `0.01` over `100,000` episodes.

### Learning Rate Schedule

`CosineAnnealingWarmRestarts` from PyTorch. LR oscillates between `learning_rate` (2.5e-4) and `min_learning_rate` (1e-5), restarting every `lr_cycle_episodes` (50,000) episodes. `scheduler.step(episode % lr_cycle)` is called once per episode.

### Checkpointing

Each checkpoint `.pt` file contains:
```python
{
    "policy_net": state_dict,
    "target_net": state_dict,
    "optimizer": state_dict,
    "episode": int,
    "global_step": int,
    "best_reward": float,
}
```

**Periodic checkpoints:** `checkpoints/session_{id}/model_ep{episode}.pt` (every 500 episodes).

**Best checkpoint:** `checkpoints/session_{id}/best_model.pt` (updated whenever `episode_reward > best_reward`).

On `KeyboardInterrupt`, a final checkpoint is saved before the training loop exits.

### Resuming Training

Pass `--resume path/to/checkpoint.pt`. The loader restores `policy_net`, `target_net`, `optimizer` state dicts and restores `start_episode`, `global_step`, and `best_reward` from the checkpoint dict.

---

## 7. Configuration Reference

File: `config/hyperparams.yaml`

### Game Settings

| Key | Value | Description |
|---|---|---|
| `board_width` | `10` | Board width in columns. |
| `board_height` | `30` | Total board height including 10-row buffer zone. |
| `visible_height` | `20` | Visible rows in the observation (board_height - buffer). |

### Training

| Key | Value | Description |
|---|---|---|
| `gamma` | `0.97` | Discount factor for future rewards. Higher value increases planning horizon. Phase 1a used `0.95`. |
| `learning_rate` | `0.00025` | Initial learning rate for Adam optimizer. |
| `min_learning_rate` | `0.00001` | Cosine annealing LR floor (eta_min). |
| `lr_cycle_episodes` | `50000` | Episodes per cosine annealing warm restart cycle (`T_0`). |
| `batch_size` | `256` | Number of transitions sampled per `train_step`. |
| `replay_buffer_size` | `200000` | Maximum capacity of the replay buffer. Phase 1a used `100000`. |
| `min_replay_size` | `1000` | Minimum transitions collected before training starts (warmup). |
| `tau` | `0.002` | Polyak averaging coefficient for target network. Phase 1a used `0.005`. Lower = more stable target. |
| `epsilon_start` | `1.0` | Initial epsilon for epsilon-greedy exploration. |
| `epsilon_end` | `0.01` | Minimum (final) epsilon. |
| `epsilon_decay_episodes` | `100000` | Number of episodes over which epsilon decays linearly. Phase 1a used `10000`. |
| `train_every_n_steps` | `4` | Train once per this many environment steps. |
| `max_steps_per_episode` | `5000` | Hard cap on steps per episode (safety measure). |

### Reward Shaping

| Key | Value | Description |
|---|---|---|
| `holes_weight` | `0.3` | Multiplier for the holes delta penalty term. |
| `height_weight` | `0.03` | Multiplier for the aggregate height delta penalty term. |
| `bumpiness_weight` | `0.01` | Multiplier for the bumpiness delta penalty term. |
| `game_over_penalty` | `35.0` | Flat penalty applied on game over. Phase 1a used `25.0`. |

Line clear rewards are hardcoded in `src/env.py` as `LINE_REWARDS = {0:0.0, 1:5.0, 2:15.0, 3:40.0, 4:150.0}` and are not configurable via YAML.

### Rendering

| Key | Value | Description |
|---|---|---|
| `cell_size` | `30` | Pixel size of each board cell in the pygame window. |
| `fps` | `60` | Target FPS for manual play mode. |
| `render_during_training` | `false` | Unused flag (training never renders). |

### Checkpoints and Logging

| Key | Value | Description |
|---|---|---|
| `save_freq` | `500` | Save a periodic checkpoint every N episodes. |
| `checkpoint_dir` | `"checkpoints"` | Root directory for checkpoint storage. |
| `log_dir` | `"runs"` | Root directory for TensorBoard event files. |

---

## 8. Session System

The `--session-id` flag organizes all outputs for a single training run into isolated subdirectories.

### Directory Structure

```
checkpoints/
  session_{id}/
    model_ep0.pt
    model_ep500.pt
    model_ep1000.pt
    ...
    best_model.pt
    session_{id}.csv

runs/
  session_{id}/
    events.out.tfevents.*   (TensorBoard)
```

If `--session-id` is not provided, a timestamp string `YYYYMMDD_HHMMSS` is auto-generated at startup.

### CSV Log Format

The CSV file at `checkpoints/session_{id}/session_{id}.csv` has one row per episode with these columns:

| Column | Type | Description |
|---|---|---|
| `episode` | `int` | Episode number (0-indexed). |
| `reward` | `float` | Total episode reward (2 decimal places). |
| `lines` | `int` | Total lines cleared in the episode. |
| `steps` | `int` | Steps taken in the episode (pieces placed). |
| `epsilon` | `float` | Epsilon value at the start of the episode (4 decimal places). |
| `loss` | `float` | Mean Smooth L1 loss over training steps in the episode (6 decimal places). |
| `grad_norm` | `float` | Mean gradient norm over training steps in the episode (4 decimal places). |
| `mean_q` | `float` | Mean of all Q-values output by policy net over training steps (4 decimal places). |
| `max_q` | `float` | Max Q-value output by policy net over training steps (4 decimal places). |
| `lr` | `float` | Current learning rate from scheduler at episode end (scientific notation). |
| `best_reward` | `float` | Best episode reward seen so far (2 decimal places). |

The CSV is written by a background daemon thread (`CSVLogger`) using a `queue.Queue`, so disk I/O does not block the training loop. A header row is written on first creation; subsequent runs append rows if the file already exists.

### Resuming a Session

To continue training under the same session ID (preserving the CSV and reusing the same checkpoint directory):

```bash
python main.py --mode train \
    --session-id phase2 \
    --resume checkpoints/session_phase2/model_ep5000.pt
```

The `start_episode`, `global_step`, and `best_reward` are restored from the checkpoint. New rows are appended to the existing CSV.

---

## 9. Game Engine

The game engine consists of three modules: `pieces.py`, `board.py`, and `tetris.py`. `TetrisEnv` bypasses `TetrisGame.step()` entirely for placement-based control — it directly manipulates `game.current_rotation`, `game.current_x`, `game.current_y`, calls `game._lock_piece()`, and calls `game._spawn_piece()`.

### `pieces.py` — Tetromino Definitions

Defines all 7 tetrominoes as `dict` objects. Each has:
- `"id"`: integer 1-7 (written into the board grid on placement)
- `"name"`: single character string (`"I"`, `"O"`, `"T"`, `"S"`, `"Z"`, `"J"`, `"L"`)
- `"color"`: RGB tuple following the Tetris Guideline color scheme
- `"rotations"`: list of 4 numpy `int8` arrays, one per rotation state

Rotation states: `0=spawn`, `1=CW (R)`, `2=180°`, `3=CCW (L)`.

Also defines `KICK_TABLE_JLSTZ` and `KICK_TABLE_I` — the official SRS wall kick offset tables. `get_kick_offsets(piece, from_rot, to_rot)` selects the correct table and returns the list of `(dx, dy)` offsets to try. The O-piece always returns `[(0,0)]` (no kicks).

### `board.py` — Grid and Metrics

A 10×30 numpy `int8` grid (`0=empty`, `1-7=piece ID`). The top 10 rows (indices 0-9) are a hidden buffer zone; the visible area is rows 10-29.

Key computed metrics (used in reward shaping):
- **Holes:** Empty cells with at least one filled cell above them in the same column.
- **Aggregate height:** Sum of the height of the tallest filled cell in each column (measured from the bottom).
- **Bumpiness:** Sum of absolute differences between adjacent column heights.

Line clearing removes all fully filled rows (using a boolean mask) and prepends empty rows at the top, preserving the existing blocks above cleared lines.

### `tetris.py` — Game Orchestrator

**7-bag randomizer:** `_fill_bag()` shuffles all 7 piece types; `_next_from_bag()` pops one and refills when empty. This guarantees each piece appears exactly once per group of 7, bounding I-piece droughts to 12 pieces maximum.

**SRS rotation:** `_rotate(direction)` tries each `get_kick_offsets()` offset in sequence. The first valid position is used. If none succeed, the rotation is rejected. `dy` in kick offsets follows SRS convention (positive = up on screen = subtract from row index).

**Lock delay:** In frame-based mode (`step()`), a piece resting on the board increments `_lock_delay_counter` each frame. A successful move or rotation resets the counter. The piece locks after 30 frames of inactivity. Hard drop bypasses lock delay entirely.

**NES-style scoring:**
```
score += SCORE_TABLE[lines] * (level + 1)
```
Where `SCORE_TABLE = {1:40, 2:100, 3:300, 4:1200}`. Level = `total_lines // 10`.

**Gravity:** `GRAVITY_TABLE` maps level to frames-per-drop (level 0 = 48 frames, level 29+ = 1 frame). Used only in `play_manual` frame-based mode; entirely irrelevant in placement-based training (`TetrisEnv` bypasses gravity).

**Hold mechanic:** One hold allowed per piece (enforced by `can_hold`, reset to `True` on each spawn). Swapping with an empty hold stores the current piece and spawns `next_piece`. Swapping with a held piece exchanges current and held, re-centering the incoming piece at spawn position.

**Game over conditions:**
1. `_spawn_piece()` returns `False` (new piece overlaps existing blocks at spawn position).
2. `TetrisEnv.step()` detects no valid placements after spawning the next piece (`not np.any(next_mask)`).
