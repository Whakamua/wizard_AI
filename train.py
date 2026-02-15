"""Training entrypoint for Wizard AI (MCCFR or Deep CFR).

Usage:
    python train.py                                            # MCCFR (default)
    python train.py --debug                                    # fast debug
    python train.py --solver deep_cfr --rounds 1,2,3           # Deep CFR
    python train.py --solver deep_cfr --debug                  # Deep CFR debug
    python train.py --num-players 4 --rounds 1,2,3             # custom config
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
# Deep CFR
# ---------------------------------------------------------------------------

def _import_deep_cfr():
    """Lazy import of PyTorch Deep CFR to avoid hard dependency."""
    import torch
    from open_spiel.python.pytorch.deep_cfr import DeepCFRSolver
    return torch, DeepCFRSolver


def deep_cfr_checkpoint_path(output_dir: str, round_num: int) -> str:
    return os.path.join(output_dir, f"round_{round_num}_deep_cfr.pt")


def save_deep_cfr_checkpoint(solver, round_num: int, num_players: int,
                             output_dir: str):
    """Save Deep CFR networks to a torch checkpoint."""
    torch, _ = _import_deep_cfr()
    os.makedirs(output_dir, exist_ok=True)
    data = {
        "policy_network": solver._policy_network.state_dict(),
        "advantage_networks": [
            net.state_dict() for net in solver._advantage_networks
        ],
        "iteration": solver._iteration,
        "round": round_num,
        "num_players": num_players,
        "solver_type": "deep_cfr",
    }
    path = deep_cfr_checkpoint_path(output_dir, round_num)
    torch.save(data, path)
    logger.info("Saved Deep CFR checkpoint: %s", path)


def load_deep_cfr_checkpoint(output_dir: str, round_num: int):
    """Load Deep CFR checkpoint, or return None."""
    torch, _ = _import_deep_cfr()
    path = deep_cfr_checkpoint_path(output_dir, round_num)
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    logger.info("Loaded Deep CFR checkpoint: %s", path)
    return data


def train_round_deep_cfr(num_players: int, num_cards: int, cfg: dict,
                         output_dir: str):
    """Train Deep CFR for a single round. Returns summary dict."""
    torch, DeepCFRSolver = _import_deep_cfr()

    logger.info("=== Training round %d with Deep CFR (%d players) ===",
                num_cards, num_players)

    game = pyspiel.load_game("python_wizard", {
        "num_players": num_players,
        "num_cards": num_cards,
    })

    solver = DeepCFRSolver(
        game,
        policy_network_layers=cfg["policy_layers"],
        advantage_network_layers=cfg["advantage_layers"],
        num_iterations=cfg["iterations"],
        num_traversals=cfg["traversals"],
        learning_rate=cfg["lr"],
        batch_size_advantage=cfg["batch_size"],
        batch_size_strategy=cfg["batch_size"],
        memory_capacity=cfg["memory_capacity"],
        policy_network_train_steps=cfg["policy_train_steps"],
        advantage_network_train_steps=cfg["advantage_train_steps"],
        reinitialize_advantage_networks=True,
    )

    t_start = time.time()
    logger.info("Deep CFR: %d iterations x %d traversals, lr=%.1e, "
                "batch=%d, memory=%d",
                cfg["iterations"], cfg["traversals"], cfg["lr"],
                cfg["batch_size"], cfg["memory_capacity"])

    policy_net, advantage_losses, policy_loss = solver.solve()

    elapsed = time.time() - t_start
    pl_val = float(policy_loss) if policy_loss is not None else 0.0
    logger.info("Round %d Deep CFR complete | %.1fs | policy_loss=%.6f",
                num_cards, elapsed, pl_val)

    # Log per-player advantage losses (final iteration)
    for p, losses in advantage_losses.items():
        valid = [l for l in losses if l is not None]
        if valid:
            logger.info("  Player %d advantage losses (last 3): %s", p,
                        [round(float(l), 6) for l in valid[-3:]])

    # Save checkpoint
    save_deep_cfr_checkpoint(solver, num_cards, num_players, output_dir)

    # Evaluate
    mean_returns = evaluate_random_vs_policy(game, solver, num_episodes=200)
    logger.info("Round %d Deep CFR eval | returns: %s",
                num_cards, [round(r, 2) for r in mean_returns])

    summary = {
        "round": num_cards,
        "num_players": num_players,
        "solver": "deep_cfr",
        "iterations": cfg["iterations"],
        "traversals": cfg["traversals"],
        "total_time_s": round(elapsed, 2),
        "policy_loss": float(policy_loss) if policy_loss is not None else None,
        "advantage_losses_final": {
            str(p): float(losses[-1]) if losses and losses[-1] is not None else None
            for p, losses in advantage_losses.items()
        },
        "final_mean_returns": mean_returns.tolist(),
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Wizard AI")
    parser.add_argument("--debug", action="store_true",
                        help="Fast debug run")
    parser.add_argument("--solver", type=str, default="mccfr",
                        choices=["mccfr", "deep_cfr"],
                        help="Algorithm: mccfr (tabular) or deep_cfr")
    parser.add_argument("--num-players", type=int, default=3,
                        help="Number of players (3-6, default: 3)")
    parser.add_argument("--rounds", type=str, default=None,
                        help="Comma-separated round numbers (default: all)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Iterations per round")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save checkpoint every N iterations (MCCFR)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N iterations (MCCFR)")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints and results")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"],
                        help="Logging level")
    # Deep CFR specific
    parser.add_argument("--traversals", type=int, default=None,
                        help="Deep CFR: traversals per iteration")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Deep CFR: learning rate")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Deep CFR: batch size")
    parser.add_argument("--memory-capacity", type=int, default=2_000_000,
                        help="Deep CFR: reservoir buffer capacity")
    parser.add_argument("--policy-layers", type=str, default="256,256",
                        help="Deep CFR: policy network layers")
    parser.add_argument("--advantage-layers", type=str, default="128,128",
                        help="Deep CFR: advantage network layers")
    parser.add_argument("--policy-train-steps", type=int, default=5000,
                        help="Deep CFR: policy net training steps")
    parser.add_argument("--advantage-train-steps", type=int, default=750,
                        help="Deep CFR: advantage net training steps")
    return parser.parse_args(argv)


def _build_deep_cfr_config(args, debug: bool) -> dict:
    """Build Deep CFR hyperparameter dict from CLI args."""
    if debug:
        return {
            "iterations": args.iterations or 5,
            "traversals": args.traversals or 10,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "memory_capacity": args.memory_capacity,
            "policy_layers": tuple(int(x) for x in args.policy_layers.split(",")),
            "advantage_layers": tuple(int(x) for x in args.advantage_layers.split(",")),
            "policy_train_steps": 100,
            "advantage_train_steps": 50,
        }
    return {
        "iterations": args.iterations or 100,
        "traversals": args.traversals or 200,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "memory_capacity": args.memory_capacity,
        "policy_layers": tuple(int(x) for x in args.policy_layers.split(",")),
        "advantage_layers": tuple(int(x) for x in args.advantage_layers.split(",")),
        "policy_train_steps": args.policy_train_steps,
        "advantage_train_steps": args.advantage_train_steps,
    }


def main(argv=None):
    args = parse_args(argv)

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)

    num_players = args.num_players
    max_round = 60 // num_players
    solver_name = args.solver

    # Determine rounds to train
    if args.debug:
        rounds = list(range(1, min(4, max_round + 1)))
    else:
        if args.rounds:
            rounds = [int(r) for r in args.rounds.split(",")]
        else:
            rounds = list(range(1, max_round + 1))

    output_dir = args.output_dir
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Training config: solver=%s, %d players, rounds %s",
                solver_name, num_players, rounds)

    all_summaries = {}
    for round_num in rounds:
        if round_num > max_round:
            logger.warning("Skipping round %d (max %d for %d players)",
                           round_num, max_round, num_players)
            continue

        if solver_name == "deep_cfr":
            cfg = _build_deep_cfr_config(args, args.debug)
            summary = train_round_deep_cfr(
                num_players=num_players,
                num_cards=round_num,
                cfg=cfg,
                output_dir=output_dir,
            )
        else:
            # MCCFR defaults
            if args.debug:
                iterations = args.iterations or 100
                checkpoint_every = args.checkpoint_every or 50
                eval_every = args.eval_every or 50
            else:
                iterations = args.iterations or 100_000
                checkpoint_every = args.checkpoint_every or 10_000
                eval_every = args.eval_every or 10_000
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
