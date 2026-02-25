"""
Comprehensive evaluation of the trained Tetris AI model.
Plays 100 episodes with epsilon=0.0 (pure greedy) and collects detailed statistics.
"""

import sys
import time
import numpy as np
import torch

from src.env import TetrisEnv
from src.ai.agent import DoubleDQNAgent


def evaluate(checkpoint_path: str, num_episodes: int = 100, device: str = "cpu"):
    """Run evaluation and collect per-episode + aggregate statistics."""

    # --- Setup ---
    env = TetrisEnv(board_width=10, board_height=30)
    agent = DoubleDQNAgent(
        input_channels=4,
        board_height=20,
        board_width=10,
        num_actions=80,
        device=device,
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)
    agent.policy_net.eval()  # Set to eval mode (affects BatchNorm and Dropout)

    # --- Per-episode storage ---
    episode_data = []

    print(f"\nRunning {num_episodes} episodes with epsilon=0.0 (pure greedy)...\n")
    start_time = time.time()

    for ep in range(num_episodes):
        obs = env.reset()
        mask = env.get_valid_mask()
        done = False

        ep_reward = 0.0
        ep_steps = 0
        ep_lines = 0
        ep_hold_actions = 0
        ep_line_clears = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # track each clear type

        while not done:
            action = agent.select_action(obs, epsilon=0.0, valid_mask=mask)

            # Track hold usage
            if action >= 40:
                ep_hold_actions += 1

            obs, reward, done, info = env.step(action)
            mask = info["valid_mask"]

            ep_reward += reward
            ep_steps += 1

            lines_this_step = info["lines_cleared"]
            ep_lines += lines_this_step
            ep_line_clears[lines_this_step] += 1

        # Board state at death
        holes_at_death = info["holes"]
        height_at_death = info["height"]
        bumpiness_at_death = info["bumpiness"]

        # Max column height at death (from aggregate height we can estimate avg,
        # but let's also get column-level detail)
        col_heights = env.game.board.get_column_heights()
        max_col_height = max(col_heights) if col_heights else 0

        ep_record = {
            "episode": ep,
            "lines": ep_lines,
            "steps": ep_steps,
            "reward": ep_reward,
            "hold_actions": ep_hold_actions,
            "tetrises": ep_line_clears[4],
            "triples": ep_line_clears[3],
            "doubles": ep_line_clears[2],
            "singles": ep_line_clears[1],
            "no_clear": ep_line_clears[0],
            "holes_at_death": holes_at_death,
            "height_at_death": height_at_death,
            "max_col_height": max_col_height,
            "bumpiness_at_death": bumpiness_at_death,
            "col_heights_at_death": col_heights,
        }
        episode_data.append(ep_record)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes} done | "
                  f"Lines: {ep_lines}, Steps: {ep_steps}, Reward: {ep_reward:.1f}")

    elapsed = time.time() - start_time
    print(f"\nEvaluation complete in {elapsed:.1f}s "
          f"({elapsed / num_episodes:.2f}s per episode)\n")

    # --- Aggregate Stats ---
    lines_arr = np.array([d["lines"] for d in episode_data])
    steps_arr = np.array([d["steps"] for d in episode_data])
    reward_arr = np.array([d["reward"] for d in episode_data])
    hold_arr = np.array([d["hold_actions"] for d in episode_data])
    holes_arr = np.array([d["holes_at_death"] for d in episode_data])
    height_arr = np.array([d["height_at_death"] for d in episode_data])
    max_height_arr = np.array([d["max_col_height"] for d in episode_data])
    bumpiness_arr = np.array([d["bumpiness_at_death"] for d in episode_data])

    total_singles = sum(d["singles"] for d in episode_data)
    total_doubles = sum(d["doubles"] for d in episode_data)
    total_triples = sum(d["triples"] for d in episode_data)
    total_tetrises = sum(d["tetrises"] for d in episode_data)
    total_line_clear_events = total_singles + total_doubles + total_triples + total_tetrises
    total_lines_from_clears = total_singles * 1 + total_doubles * 2 + total_triples * 3 + total_tetrises * 4

    total_steps_all = int(steps_arr.sum())
    total_hold_all = int(hold_arr.sum())

    print("=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)
    for name, arr in [("Lines cleared", lines_arr), ("Steps survived", steps_arr),
                       ("Total reward", reward_arr)]:
        print(f"{name:<25} {arr.mean():>8.1f} {np.median(arr):>8.1f} "
              f"{arr.std():>8.1f} {arr.min():>8.1f} {arr.max():>8.1f}")

    print(f"\n--- Hold Usage ---")
    print(f"  Total hold actions:    {total_hold_all} / {total_steps_all} total actions")
    print(f"  Hold usage rate:       {100 * total_hold_all / max(total_steps_all, 1):.1f}%")
    print(f"  Hold actions per game: {hold_arr.mean():.1f} (median {np.median(hold_arr):.0f})")

    print(f"\n--- Line Clear Distribution ---")
    print(f"  Total line clear events: {total_line_clear_events}")
    print(f"  Total lines cleared:     {total_lines_from_clears}")
    if total_line_clear_events > 0:
        print(f"  Singles (1-line):  {total_singles:>5} ({100*total_singles/total_line_clear_events:.1f}%)")
        print(f"  Doubles (2-line):  {total_doubles:>5} ({100*total_doubles/total_line_clear_events:.1f}%)")
        print(f"  Triples (3-line):  {total_triples:>5} ({100*total_triples/total_line_clear_events:.1f}%)")
        print(f"  Tetrises (4-line): {total_tetrises:>5} ({100*total_tetrises/total_line_clear_events:.1f}%)")
    else:
        print(f"  No line clears at all!")

    tetris_rate = 100 * total_tetrises / max(total_line_clear_events, 1)
    print(f"  Tetris rate: {tetris_rate:.1f}% of line clear events")
    if total_lines_from_clears > 0:
        tetris_line_pct = 100 * (total_tetrises * 4) / total_lines_from_clears
        print(f"  Lines from Tetrises: {tetris_line_pct:.1f}% of all lines cleared")

    print(f"\n--- Board State at Death ---")
    print(f"{'Metric':<25} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)
    for name, arr in [("Holes at death", holes_arr),
                       ("Agg. height at death", height_arr),
                       ("Max col height", max_height_arr),
                       ("Bumpiness at death", bumpiness_arr)]:
        print(f"{name:<25} {arr.mean():>8.1f} {np.median(arr):>8.1f} "
              f"{arr.std():>8.1f} {arr.min():>8.1f} {arr.max():>8.1f}")

    # --- Failure Pattern Analysis ---
    print(f"\n{'=' * 70}")
    print("FAILURE PATTERN ANALYSIS")
    print("=" * 70)

    # 1. Early vs late game deaths
    step_percentiles = np.percentile(steps_arr, [10, 25, 50, 75, 90])
    print(f"\n--- Game Length Distribution ---")
    print(f"  10th percentile: {step_percentiles[0]:.0f} steps")
    print(f"  25th percentile: {step_percentiles[1]:.0f} steps")
    print(f"  50th percentile: {step_percentiles[2]:.0f} steps (median)")
    print(f"  75th percentile: {step_percentiles[3]:.0f} steps")
    print(f"  90th percentile: {step_percentiles[4]:.0f} steps")

    early_deaths = int(np.sum(steps_arr < 20))
    mid_deaths = int(np.sum((steps_arr >= 20) & (steps_arr < 50)))
    late_deaths = int(np.sum((steps_arr >= 50) & (steps_arr < 100)))
    long_games = int(np.sum(steps_arr >= 100))
    print(f"\n  Early deaths (<20 steps):   {early_deaths} ({100*early_deaths/num_episodes:.0f}%)")
    print(f"  Mid deaths (20-49 steps):   {mid_deaths} ({100*mid_deaths/num_episodes:.0f}%)")
    print(f"  Late deaths (50-99 steps):  {late_deaths} ({100*late_deaths/num_episodes:.0f}%)")
    print(f"  Long games (100+ steps):    {long_games} ({100*long_games/num_episodes:.0f}%)")

    # 2. Correlation: holes vs game length
    if len(steps_arr) > 5:
        corr_holes_steps = np.corrcoef(holes_arr, steps_arr)[0, 1]
        corr_height_steps = np.corrcoef(height_arr, steps_arr)[0, 1]
        corr_bumpy_steps = np.corrcoef(bumpiness_arr, steps_arr)[0, 1]
        print(f"\n--- Correlations with Game Length ---")
        print(f"  Holes at death vs steps:      r={corr_holes_steps:.3f}")
        print(f"  Height at death vs steps:      r={corr_height_steps:.3f}")
        print(f"  Bumpiness at death vs steps:   r={corr_bumpy_steps:.3f}")

    # 3. Hold usage in short vs long games
    short_mask = steps_arr < np.median(steps_arr)
    long_mask = ~short_mask
    if short_mask.sum() > 0 and long_mask.sum() > 0:
        hold_rate_short = hold_arr[short_mask].sum() / max(steps_arr[short_mask].sum(), 1)
        hold_rate_long = hold_arr[long_mask].sum() / max(steps_arr[long_mask].sum(), 1)
        print(f"\n--- Hold Usage: Short vs Long Games ---")
        print(f"  Short games (below median): {100*hold_rate_short:.1f}% hold rate")
        print(f"  Long games (above median):  {100*hold_rate_long:.1f}% hold rate")

    # 4. Lines efficiency: lines per step
    lines_per_step = lines_arr / np.maximum(steps_arr, 1)
    print(f"\n--- Efficiency ---")
    print(f"  Lines per step (mean): {lines_per_step.mean():.3f}")
    print(f"  Lines per step (max):  {lines_per_step.max():.3f}")
    print(f"  Theoretical max (Tetris every step): 4.0")
    print(f"  Efficiency vs max: {100*lines_per_step.mean()/4.0:.2f}%")

    # 5. Column height profile at death (average across episodes)
    all_col_heights = np.array([d["col_heights_at_death"] for d in episode_data])
    avg_col_heights = all_col_heights.mean(axis=0)
    print(f"\n--- Average Column Height Profile at Death ---")
    print(f"  Column:  ", end="")
    for c in range(10):
        print(f"  {c:>3}", end="")
    print()
    print(f"  Height:  ", end="")
    for c in range(10):
        print(f"  {avg_col_heights[c]:>5.1f}", end="")
    print()

    # Identify if any columns are systematically higher
    min_col_h = avg_col_heights.min()
    max_col_h = avg_col_heights.max()
    tallest_col = int(avg_col_heights.argmax())
    shortest_col = int(avg_col_heights.argmin())
    print(f"\n  Tallest column: {tallest_col} (avg height {max_col_h:.1f})")
    print(f"  Shortest column: {shortest_col} (avg height {min_col_h:.1f})")
    print(f"  Height spread: {max_col_h - min_col_h:.1f}")

    # 6. Zero-line games
    zero_line_games = int(np.sum(lines_arr == 0))
    print(f"\n--- Edge Cases ---")
    print(f"  Games with 0 lines cleared: {zero_line_games} ({100*zero_line_games/num_episodes:.0f}%)")
    print(f"  Games with 10+ lines:       {int(np.sum(lines_arr >= 10))} ({100*int(np.sum(lines_arr >= 10))/num_episodes:.0f}%)")
    print(f"  Games with 20+ lines:       {int(np.sum(lines_arr >= 20))} ({100*int(np.sum(lines_arr >= 20))/num_episodes:.0f}%)")

    # 7. Best and worst episodes
    best_ep = episode_data[int(np.argmax(lines_arr))]
    worst_ep = episode_data[int(np.argmin(lines_arr))]
    print(f"\n--- Best Episode ---")
    print(f"  Episode {best_ep['episode']}: {best_ep['lines']} lines, "
          f"{best_ep['steps']} steps, reward={best_ep['reward']:.1f}, "
          f"holds={best_ep['hold_actions']}, "
          f"tetrises={best_ep['tetrises']}")

    print(f"\n--- Worst Episode ---")
    print(f"  Episode {worst_ep['episode']}: {worst_ep['lines']} lines, "
          f"{worst_ep['steps']} steps, reward={worst_ep['reward']:.1f}, "
          f"holds={worst_ep['hold_actions']}, "
          f"holes_at_death={worst_ep['holes_at_death']}")

    # 8. Performance degradation analysis - bucket by game phase
    print(f"\n--- Performance by Game Phase ---")
    # For episodes long enough, compare early/late line clearing rate
    phase_thresholds = [10, 20, 30, 50, 75, 100]
    print(f"  {'Phase':<20} {'Episodes reaching':>18} {'Avg total lines at phase':>25}")
    for thresh in phase_thresholds:
        reaching = int(np.sum(steps_arr >= thresh))
        if reaching > 0:
            # Average lines cleared by episodes that lasted at least this long
            # (can't slice mid-episode, so just report how many survive)
            avg_lines = lines_arr[steps_arr >= thresh].mean()
            print(f"  Step {thresh:>3}+{'':<15} {reaching:>8} ({100*reaching/num_episodes:>5.1f}%) "
                  f"  {avg_lines:>10.1f}")

    print(f"\n{'=' * 70}")
    print("SUMMARY & TOP FAILURE PATTERNS")
    print("=" * 70)

    # Determine top failure patterns based on data
    patterns = []

    # Pattern: Hole accumulation
    if holes_arr.mean() > 5:
        patterns.append(
            f"HOLE ACCUMULATION: Agent dies with avg {holes_arr.mean():.1f} holes. "
            f"It creates holes faster than it clears them, leading to unrecoverable board states."
        )
    elif holes_arr.mean() > 2:
        patterns.append(
            f"MODERATE HOLE CREATION: Agent dies with avg {holes_arr.mean():.1f} holes. "
            f"Holes slowly accumulate and eventually block line clears."
        )

    # Pattern: Short game length
    if np.median(steps_arr) < 50:
        patterns.append(
            f"SHORT GAMES: Median game is only {np.median(steps_arr):.0f} steps (~{np.median(steps_arr):.0f} pieces). "
            f"Agent cannot sustain play beyond ~{np.percentile(steps_arr, 75):.0f} pieces consistently."
        )

    # Pattern: No Tetrises
    if total_tetrises == 0:
        patterns.append(
            f"NO TETRISES: Agent never achieves a 4-line clear. "
            f"It does not build wells or stack efficiently for Tetris opportunities."
        )
    elif tetris_rate < 5:
        patterns.append(
            f"VERY LOW TETRIS RATE: Only {tetris_rate:.1f}% of clears are Tetrises. "
            f"Agent rarely builds wells or plans for 4-line clears."
        )

    # Pattern: Low hold usage
    hold_rate_pct = 100 * total_hold_all / max(total_steps_all, 1)
    if hold_rate_pct < 5:
        patterns.append(
            f"UNDERUSE OF HOLD: Only {hold_rate_pct:.1f}% of actions use hold. "
            f"Agent has not learned to strategically save I-pieces or swap unfavorable pieces."
        )

    # Pattern: Height explosion
    if max_height_arr.mean() > 25:
        patterns.append(
            f"HEIGHT EXPLOSION: Average max column height at death is {max_height_arr.mean():.1f}. "
            f"Board fills from bottom to top without enough clearing."
        )

    # Pattern: Bumpiness
    if bumpiness_arr.mean() > 15:
        patterns.append(
            f"HIGH BUMPINESS: Average bumpiness at death is {bumpiness_arr.mean():.1f}. "
            f"Surface is very uneven, making line clears difficult."
        )

    # Pattern: Mostly singles
    if total_line_clear_events > 0 and (total_singles / total_line_clear_events) > 0.8:
        patterns.append(
            f"SINGLE-LINE BIAS: {100*total_singles/total_line_clear_events:.0f}% of clears are singles. "
            f"Agent clears one line at a time instead of building for multi-line clears."
        )

    # Pattern: Uneven column heights
    if (max_col_h - min_col_h) > 8:
        patterns.append(
            f"UNEVEN STACKING: Column height spread is {max_col_h - min_col_h:.1f}. "
            f"Tallest col {tallest_col} (avg {max_col_h:.1f}) vs shortest col {shortest_col} (avg {min_col_h:.1f}). "
            f"Agent builds unevenly, creating cliffs."
        )

    print()
    for i, pattern in enumerate(patterns[:5], 1):
        print(f"  {i}. {pattern}")
        print()

    print(f"\n--- Improvement Suggestions ---")
    suggestions = []
    if holes_arr.mean() > 3:
        suggestions.append("Increase hole penalty weight (currently 0.3) to make the agent more hole-averse")
    if total_tetrises == 0 or tetris_rate < 5:
        suggestions.append("Add a well-building reward to incentivize leaving a column open for I-pieces")
    if hold_rate_pct < 10:
        suggestions.append("Add explicit hold-for-I-piece bonus or increase exploration of hold actions during training")
    if np.median(steps_arr) < 50:
        suggestions.append("Increase gamma (currently 0.97) to 0.99 for longer planning horizon")
    if bumpiness_arr.mean() > 15:
        suggestions.append("Increase bumpiness penalty weight (currently 0.01) to encourage flatter surfaces")
    if (total_singles / max(total_line_clear_events, 1)) > 0.7:
        suggestions.append("Increase rewards for multi-line clears further (doubles/triples) to break single-line habit")

    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}")

    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_model.pt"
    evaluate(checkpoint, num_episodes=100, device="cpu")
