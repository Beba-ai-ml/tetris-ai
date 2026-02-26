# Tetris AI

A placement-based Tetris agent trained with **Afterstate V-learning** (evolved from Dueling Double DQN). The agent evaluates every possible piece placement by simulating the resulting board state and picking the best one. One decision per piece — no frame-by-frame controls.

![Demo](assets/demo.gif)

---

## Key Features

- **Afterstate V-learning** -- evaluates V(board_after_placement) for every valid move, picks the best
- **Placement-based action space** -- 80 discrete actions (4 rotations x 10 columns x 2 modes: place / hold-then-place)
- **CNN value network** with BatchNorm and asymmetric stride (preserves column resolution)
- **Hold mechanic** -- agent strategically swaps pieces to optimize placement
- **SRS rotation system** with full wall-kick tables (standard Tetris Guideline)
- **7-bag randomizer** for fair piece distribution
- **Reward shaping** -- line clears, hole/height/bumpiness penalties, clean placement bonus
- **Cosine annealing LR** with warm restarts

---

## Architecture

```
main.py                         CLI entry point (train / play / watch)
    |
src/env.py                      Placement-based RL environment (80 actions, afterstate generation)
    |
src/game/tetris.py              Game engine (scoring, gravity, SRS, 7-bag)
src/game/board.py               Board logic (collision, line clearing, metrics)
src/game/pieces.py              Tetromino definitions + SRS kick tables
    |
src/ai/model.py                 AfterstateNet CNN (2ch input -> scalar V(s'))
src/ai/agent.py                 Afterstate value agent (TD(0) V-learning, polyak target updates)
src/ai/replay_buffer.py         Pre-allocated circular replay buffer
    |
src/train.py                    Training loop (CSV + TensorBoard logging)
src/play.py                     Manual play + AI watch modes (pygame)
src/renderer.py                 Pygame renderer (board, ghost piece, sidebar)
```

### Network Architecture (Afterstate V-Network)

```
Input: (batch, 2, 20, 10)
  Ch0: Board grid AFTER placement + line clear (binary)
  Ch1: Held piece shape (zeros if None)

  -> Conv2d(2->32, 3x3) + BN + ReLU
  -> Conv2d(32->64, 3x3, stride=(2,1)) + BN + ReLU   # compress height, keep width
  -> Conv2d(64->64, 3x3, stride=(2,1)) + BN + ReLU
  -> Flatten (3200)
  -> Linear(3200->512) -> ReLU -> Linear(512->1)

Output: scalar V(afterstate)
```

The agent simulates every valid placement, evaluates V(resulting board), and picks:
`argmax_a [ reward(a) + gamma * V(afterstate(a)) ]`

---

## Quick Start

```bash
git clone https://github.com/Beba-ai-ml/tetris-ai.git
cd tetris-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training

```bash
python3 main.py --mode train --session-id my_run
```

Training runs indefinitely until `Ctrl+C`. Checkpoints and CSV logs are saved to `checkpoints/session_<id>/`. TensorBoard logs go to `runs/session_<id>/`.

Resume from a checkpoint:

```bash
python3 main.py --mode train --session-id my_run --resume checkpoints/session_my_run/best_model.pt
```

Monitor with TensorBoard:

```bash
tensorboard --logdir runs/
```

---

## Watch AI Play

```bash
python3 main.py --mode watch --model checkpoints/session_phase3_afterstate/best_model.pt
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--fps` | 30 | Rendering speed |
| `--episodes` | 0 | Number of episodes (0 = infinite) |

---

## Play Manually

```bash
python3 main.py --mode play
```

Controls: Arrow keys to move, Z/X to rotate, Space to hard drop, C to hold.

---

## Project Structure

```
tetris-ai/
  main.py                     Entry point (train / play / watch)
  requirements.txt            Python dependencies
  config/
    hyperparams.yaml           All training hyperparameters
  src/
    env.py                     Placement-based RL environment
    train.py                   Training loop with logging
    play.py                    Manual play + AI watch modes
    renderer.py                Pygame renderer
    game/
      tetris.py                Game engine (scoring, SRS, 7-bag)
      board.py                 Board logic (collision, line clearing)
      pieces.py                Tetromino shapes + SRS kick tables
    ai/
      model.py                 AfterstateNet CNN (V-learning)
      agent.py                 Afterstate value agent (TD(0))
      replay_buffer.py         Circular replay buffer
  scripts/
    record_gif.py              Record demo GIF (PIL-based, no pygame)
  checkpoints/                 Saved models + training CSV logs
  runs/                        TensorBoard logs
  assets/
    demo.gif                   Demo recording
```

---

## How It Works

### Placement-Based Action Space

Instead of per-frame controls (left, right, rotate, drop), the agent makes one decision per piece: **where to place it**. Each action encodes `(rotation, column, hold?)`:

- Actions **0-39**: Place current piece -- `rotation = action // 10`, `column = action % 10`
- Actions **40-79**: Hold first, then place the resulting piece with the same encoding

The piece is instantly hard-dropped to the chosen position. This eliminates the credit assignment problem of multi-step placement.

### Afterstate V-Learning

- **Afterstate evaluation**: For each valid placement, simulate the result (lock piece, clear lines), then evaluate V(resulting board) with a CNN
- **Action selection**: Pick the action maximizing `reward(a) + gamma * V(afterstate(a))` -- no Q-value overestimation problem
- **TD(0) V-learning**: Train the value network to predict `V(s) = r + gamma * V(s')` using target network for stability
- **Soft target updates**: Polyak averaging (`tau=0.002`) for stable learning
- **No action masking needed**: Only valid placements generate afterstates, so invalid actions are naturally excluded

### Reward Shaping

| Component | Value |
|-----------|-------|
| 1 line clear | +5 |
| 2 lines (double) | +15 |
| 3 lines (triple) | +40 |
| 4 lines (Tetris) | +150 |
| Hole creation | -0.3 per new hole |
| Height increase | -0.03 per row |
| Bumpiness increase | -0.01 per unit |
| Clean placement (0 new holes) | +0.5 |
| Survival bonus (per piece) | +0.1 |
| Game over | -35 |

### Training Pipeline

1. **Warmup**: Collect 1,000 random transitions to fill replay buffer
2. **Epsilon decay**: Linear from 1.0 to 0.01 over 100,000 episodes
3. **Training frequency**: One gradient step every 4 environment steps
4. **Learning rate**: Cosine annealing with warm restarts (cycle = 50,000 episodes)
5. **Batch size**: 256 from a 200,000-transition replay buffer

---

## Configuration

All hyperparameters live in [`config/hyperparams.yaml`](config/hyperparams.yaml):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `board_width` / `board_height` | 10 / 30 | Standard Tetris board (20 visible + 10 buffer) |
| `learning_rate` | 0.00025 | Adam optimizer LR |
| `gamma` | 0.97 | Discount factor |
| `tau` | 0.002 | Soft target update rate |
| `batch_size` | 256 | Training batch size |
| `replay_buffer_size` | 200,000 | Experience replay capacity |
| `epsilon_decay_episodes` | 100,000 | Episodes for epsilon to reach minimum |
| `holes_weight` | 0.3 | Penalty weight for hole creation |
| `game_over_penalty` | 35.0 | Negative reward on death |

---

## Results

### Phase 3 — Afterstate V-Learning (100k episodes)

| Metric | Value |
|--------|-------|
| Best single game | **1,766 lines** |
| Best reward | **35,401** |
| Avg lines (last 1K eps) | **140** |
| Avg lines (last 10K eps) | 38.7 |
| Hold rate (learned) | ~32% (down from 50% random) |

The agent cleared **501 lines in a single game** during the demo recording above.

### Training Evolution

| Phase | Architecture | Best Lines | Avg Lines | Episodes |
|-------|-------------|-----------|-----------|----------|
| Phase 2b | Dueling Double DQN | 100 | 47 | 119k |
| **Phase 3** | **Afterstate V-learning** | **1,766** | **140** | **100k** |

The afterstate architecture delivered a **17.7x improvement** in best game performance and **3x improvement** in average lines compared to the previous Dueling Double DQN approach. Training was still accelerating at session end — no plateau observed.

---

## License

MIT
