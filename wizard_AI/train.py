"""Training entrypoint for Wizard AI using External Sampling MCCFR.

Usage:
    python train.py                                    # full training
    python train.py --debug                            # fast debug run
    python train.py --num-players 4 --rounds 1,2,3     # custom config
    python train.py --iterations 50000 --rounds 1,2    # custom iterations
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import pyspiel

# Register our game before importing the solver
import wizard_game  # noqa: F401
from open_spiel.python.algorithms.external_sampling_mccfr import (
    ExternalSamplingSolver,
)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("wizard_train")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def checkpoint_path(output_dir: str, round_num: int, iteration: int) -> str:
    return os.path.join(output_dir, f"round_{round_num}_iter_{iteration}.pkl")


def latest_checkpoint_path(output_dir: str, round_num: int) -> str:
    return os.path.join(output_dir, f"round_{round_num}_latest.pkl")


def save_checkpoint(solver, round_num: int, iteration: int, num_players: int,
                    output_dir: str):
    """Pickle solver infostates + metadata."""
    data = {
        "infostates": solver._infostates,
        "iteration": iteration,
        "round": round_num,
        "num_players": num_players,
    }
    os.makedirs(output_dir, exist_ok=True)
    path = latest_checkpoint_path(output_dir, round_num)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Also save a versioned copy at milestones
    versioned = checkpoint_path(output_dir, round_num, iteration)
    with open(versioned, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.debug("Saved checkpoint: %s", path)


def load_checkpoint(output_dir: str, round_num: int):
    """Load latest checkpoint for a round, or return None."""
    path = latest_checkpoint_path(output_dir, round_num)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Resumed from checkpoint: %s (iteration %d)",
                path, data["iteration"])
    return data


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_random_vs_policy(game, policy, num_episodes=200, seed=42):
    """Estimate average returns of the trained policy vs random play.

    All players use the trained policy. Returns the mean score per player
    over num_episodes random deals.
    """
    rng = np.random.RandomState(seed)
    total_returns = np.zeros(game.num_players())
    for _ in range(num_episodes):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                action = rng.choice(actions, p=probs)
                state.apply_action(action)
            else:
                action_probs = policy.action_probabilities(state)
                actions = list(action_probs.keys())
                probs = list(action_probs.values())
                # Normalize in case of small numerical issues
                probs = np.array(probs, dtype=np.float64)
                probs = np.maximum(probs, 0)
                s = probs.sum()
                if s > 0:
                    probs /= s
                else:
                    probs = np.ones(len(probs)) / len(probs)
                action = rng.choice(actions, p=probs)
                state.apply_action(action)
        total_returns += np.array(state.returns())
    return total_returns / num_episodes


# ---------------------------------------------------------------------------
# Training loop for a single round
# ---------------------------------------------------------------------------

def train_round(num_players: int, num_cards: int, iterations: int,
                checkpoint_every: int, eval_every: int, output_dir: str):
    """Train MCCFR for a single round size. Returns summary dict."""
    logger.info("=== Training round %d (%d players, %d iterations) ===",
                num_cards, num_players, iterations)

    game = pyspiel.load_game("python_wizard", {
        "num_players": num_players,
        "num_cards": num_cards,
    })

    solver = ExternalSamplingSolver(game)
    start_iteration = 0

    # Try to resume from checkpoint
    ckpt = load_checkpoint(output_dir, num_cards)
    if ckpt and ckpt["num_players"] == num_players:
        solver._infostates = ckpt["infostates"]
        start_iteration = ckpt["iteration"]
        logger.info("Resuming round %d from iteration %d", num_cards, start_iteration)

    summary = {
        "round": num_cards,
        "num_players": num_players,
        "iterations": iterations,
        "checkpoints": [],
        "evaluations": [],
    }

    t_start = time.time()
    for i in range(start_iteration, iterations):
        solver.iteration()

        iter_num = i + 1
        elapsed = time.time() - t_start
        num_infostates = len(solver._infostates)

        if iter_num % max(1, iterations // 20) == 0 or iter_num == iterations:
            logger.info(
                "Round %d | iter %d/%d | infostates: %d | elapsed: %.1fs",
                num_cards, iter_num, iterations, num_infostates, elapsed,
            )

        # Periodic evaluation
        if eval_every > 0 and (iter_num % eval_every == 0 or iter_num == iterations):
            avg_policy = solver.average_policy()
            mean_returns = evaluate_random_vs_policy(game, avg_policy,
                                                     num_episodes=100)
            eval_entry = {
                "iteration": iter_num,
                "mean_returns": mean_returns.tolist(),
                "elapsed_s": round(elapsed, 2),
            }
            summary["evaluations"].append(eval_entry)
            logger.info("Round %d | eval @ iter %d | mean returns: %s",
                        num_cards, iter_num,
                        [round(r, 2) for r in mean_returns])

        # Checkpoint
        if checkpoint_every > 0 and (
            iter_num % checkpoint_every == 0 or iter_num == iterations
        ):
            save_checkpoint(solver, num_cards, iter_num, num_players, output_dir)
            summary["checkpoints"].append(iter_num)

    elapsed_total = time.time() - t_start
    summary["total_time_s"] = round(elapsed_total, 2)
    summary["final_infostates"] = len(solver._infostates)

    # Final evaluation
    avg_policy = solver.average_policy()
    mean_returns = evaluate_random_vs_policy(game, avg_policy, num_episodes=200)
    summary["final_mean_returns"] = mean_returns.tolist()
    logger.info("Round %d complete | %d infostates | %.1fs | returns: %s",
                num_cards, len(solver._infostates), elapsed_total,
                [round(r, 2) for r in mean_returns])
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Wizard AI with MCCFR")
    parser.add_argument("--debug", action="store_true",
                        help="Fast debug run (100 iters, rounds 1-3)")
    parser.add_argument("--num-players", type=int, default=3,
                        help="Number of players (3-6, default: 3)")
    parser.add_argument("--rounds", type=str, default=None,
                        help="Comma-separated round numbers (default: all)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="MCCFR iterations per round")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate policy every N iterations")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints and results")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"],
                        help="Logging level")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)

    num_players = args.num_players
    max_round = 60 // num_players

    # Determine rounds to train
    if args.debug:
        rounds = list(range(1, min(4, max_round + 1)))
        iterations = args.iterations or 100
        checkpoint_every = args.checkpoint_every or 50
        eval_every = args.eval_every or 50
        logger.info("DEBUG MODE: rounds=%s, iterations=%d", rounds, iterations)
    else:
        if args.rounds:
            rounds = [int(r) for r in args.rounds.split(",")]
        else:
            rounds = list(range(1, max_round + 1))
        iterations = args.iterations or 100_000
        checkpoint_every = args.checkpoint_every or 10_000
        eval_every = args.eval_every or 10_000

    output_dir = args.output_dir
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Training config: %d players, rounds %s, %d iterations/round",
                num_players, rounds, iterations)

    all_summaries = {}
    for round_num in rounds:
        if round_num > max_round:
            logger.warning("Skipping round %d (max %d for %d players)",
                           round_num, max_round, num_players)
            continue
        summary = train_round(
            num_players=num_players,
            num_cards=round_num,
            iterations=iterations,
            checkpoint_every=checkpoint_every,
            eval_every=eval_every,
            output_dir=output_dir,
        )
        all_summaries[round_num] = summary

        # Write per-round summary
        summary_path = os.path.join(results_dir, f"round_{round_num}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Wrote summary: %s", summary_path)

    # Write combined summary
    combined_path = os.path.join(results_dir, "training_summary.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info("Training complete. Combined summary: %s", combined_path)


if __name__ == "__main__":
    main()
