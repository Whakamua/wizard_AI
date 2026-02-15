"""Tests for the Wizard training pipeline."""

import os
import sys
import pickle
import shutil
import tempfile
import time

import numpy as np
import pytest
import pyspiel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wizard_game  # noqa: F401  (registers the game)
from open_spiel.python.algorithms.external_sampling_mccfr import (
    ExternalSamplingSolver,
)
from train import (
    save_checkpoint,
    load_checkpoint,
    evaluate_random_vs_policy,
    train_round,
    main as train_main,
)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def game_1c():
    return pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 1})


@pytest.fixture
def game_2c():
    return pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 2})


# ---------------------------------------------------------------------------
# Solver initialization
# ---------------------------------------------------------------------------

class TestSolverInit:

    def test_solver_creates(self, game_1c):
        solver = ExternalSamplingSolver(game_1c)
        assert solver is not None
        assert len(solver._infostates) == 0

    def test_solver_creates_2card(self, game_2c):
        solver = ExternalSamplingSolver(game_2c)
        assert solver is not None


# ---------------------------------------------------------------------------
# Iteration runs
# ---------------------------------------------------------------------------

class TestIterationRuns:

    def test_single_iteration(self, game_1c):
        solver = ExternalSamplingSolver(game_1c)
        solver.iteration()
        assert len(solver._infostates) > 0

    def test_multiple_iterations_grows_infostates(self, game_1c):
        solver = ExternalSamplingSolver(game_1c)
        for _ in range(10):
            solver.iteration()
        n10 = len(solver._infostates)
        for _ in range(40):
            solver.iteration()
        n50 = len(solver._infostates)
        assert n50 >= n10

    def test_average_policy_valid(self, game_1c):
        """Average policy should produce valid probability distributions."""
        solver = ExternalSamplingSolver(game_1c)
        for _ in range(50):
            solver.iteration()
        avg_policy = solver.average_policy()

        # Play a few states and check probabilities
        import random
        rng = random.Random(42)
        for _ in range(20):
            state = game_1c.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    actions, probs = zip(*outcomes)
                    state.apply_action(rng.choices(actions, weights=probs, k=1)[0])
                else:
                    action_probs = avg_policy.action_probabilities(state)
                    probs = list(action_probs.values())
                    assert all(p >= 0 for p in probs), "Negative probability"
                    total = sum(probs)
                    assert abs(total - 1.0) < 1e-6, f"Probs sum to {total}"
                    # Pick action
                    action = rng.choices(list(action_probs.keys()),
                                         weights=probs, k=1)[0]
                    state.apply_action(action)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

class TestCheckpointing:

    def test_save_and_load(self, game_1c, tmp_dir):
        solver = ExternalSamplingSolver(game_1c)
        for _ in range(20):
            solver.iteration()
        original_infostates = {
            k: [v[0].copy(), v[1].copy()]
            for k, v in solver._infostates.items()
        }
        save_checkpoint(solver, round_num=1, iteration=20,
                        num_players=3, output_dir=tmp_dir)
        loaded = load_checkpoint(tmp_dir, round_num=1)
        assert loaded is not None
        assert loaded["iteration"] == 20
        assert loaded["round"] == 1
        assert loaded["num_players"] == 3
        # Verify infostates match
        for key in original_infostates:
            assert key in loaded["infostates"]
            np.testing.assert_array_equal(
                loaded["infostates"][key][0], original_infostates[key][0]
            )
            np.testing.assert_array_equal(
                loaded["infostates"][key][1], original_infostates[key][1]
            )

    def test_load_nonexistent(self, tmp_dir):
        result = load_checkpoint(tmp_dir, round_num=99)
        assert result is None

    def test_resume_preserves_policy(self, game_1c, tmp_dir):
        """Training, saving, and resuming should produce identical policies."""
        solver = ExternalSamplingSolver(game_1c)
        for _ in range(30):
            solver.iteration()
        save_checkpoint(solver, round_num=1, iteration=30,
                        num_players=3, output_dir=tmp_dir)

        # Create new solver and load checkpoint
        solver2 = ExternalSamplingSolver(game_1c)
        ckpt = load_checkpoint(tmp_dir, round_num=1)
        solver2._infostates = ckpt["infostates"]

        # Policies should match
        pol1 = solver.average_policy()
        pol2 = solver2.average_policy()

        import random
        rng = random.Random(123)
        state = game_1c.new_initial_state()
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(rng.choices(actions, weights=probs, k=1)[0])
        # At first player decision, compare
        ap1 = pol1.action_probabilities(state)
        ap2 = pol2.action_probabilities(state)
        assert ap1.keys() == ap2.keys()
        for a in ap1:
            assert abs(ap1[a] - ap2[a]) < 1e-10


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

class TestEvaluation:

    def test_evaluate_returns_correct_shape(self, game_1c):
        solver = ExternalSamplingSolver(game_1c)
        for _ in range(20):
            solver.iteration()
        avg_policy = solver.average_policy()
        returns = evaluate_random_vs_policy(game_1c, avg_policy, num_episodes=50)
        assert returns.shape == (3,)

    def test_evaluate_returns_finite(self, game_1c):
        solver = ExternalSamplingSolver(game_1c)
        for _ in range(20):
            solver.iteration()
        avg_policy = solver.average_policy()
        returns = evaluate_random_vs_policy(game_1c, avg_policy, num_episodes=50)
        assert np.all(np.isfinite(returns))


# ---------------------------------------------------------------------------
# train_round integration
# ---------------------------------------------------------------------------

class TestTrainRound:

    def test_train_round_produces_summary(self, tmp_dir):
        summary = train_round(
            num_players=3, num_cards=1, iterations=20,
            checkpoint_every=10, eval_every=10, output_dir=tmp_dir,
        )
        assert summary["round"] == 1
        assert summary["final_infostates"] > 0
        assert "final_mean_returns" in summary
        assert len(summary["final_mean_returns"]) == 3
        # Check checkpoint files exist
        assert os.path.exists(os.path.join(tmp_dir, "round_1_latest.pkl"))


# ---------------------------------------------------------------------------
# Debug mode timing
# ---------------------------------------------------------------------------

class TestDebugMode:

    def test_debug_completes_under_60s(self, tmp_dir):
        t0 = time.time()
        train_main(["--debug", "--output-dir", tmp_dir])
        elapsed = time.time() - t0
        assert elapsed < 60, f"Debug mode took {elapsed:.1f}s (limit: 60s)"


# ---------------------------------------------------------------------------
# Deep CFR
# ---------------------------------------------------------------------------

class TestDeepCFRInit:

    def test_solver_creates(self, game_1c):
        from open_spiel.python.pytorch.deep_cfr import DeepCFRSolver
        solver = DeepCFRSolver(game_1c, num_iterations=1, num_traversals=1)
        assert solver is not None

    def test_solver_creates_2card(self, game_2c):
        from open_spiel.python.pytorch.deep_cfr import DeepCFRSolver
        solver = DeepCFRSolver(game_2c, num_iterations=1, num_traversals=1)
        assert solver is not None


class TestDeepCFRSolve:

    def test_solve_completes(self, game_1c):
        from open_spiel.python.pytorch.deep_cfr import DeepCFRSolver
        solver = DeepCFRSolver(
            game_1c,
            num_iterations=2,
            num_traversals=5,
            policy_network_train_steps=10,
            advantage_network_train_steps=10,
        )
        policy_net, adv_losses, policy_loss = solver.solve()
        assert policy_net is not None
        assert float(policy_loss) >= 0  # may be numpy scalar

    def test_action_probs_valid(self, game_1c):
        """After solve, action_probabilities should return a dict over legal actions."""
        import random
        from open_spiel.python.pytorch.deep_cfr import DeepCFRSolver
        solver = DeepCFRSolver(
            game_1c,
            num_iterations=3,
            num_traversals=10,
            policy_network_train_steps=100,
            advantage_network_train_steps=50,
        )
        solver.solve()

        rng = random.Random(42)
        for _ in range(10):
            state = game_1c.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    actions, probs = zip(*outcomes)
                    state.apply_action(rng.choices(actions, weights=probs, k=1)[0])
                else:
                    ap = solver.action_probabilities(state)
                    assert len(ap) > 0, "No actions returned"
                    probs = np.array(list(ap.values()), dtype=np.float64)
                    probs = np.maximum(probs, 0)
                    s = probs.sum()
                    if s > 0:
                        probs /= s
                    else:
                        probs = np.ones(len(probs)) / len(probs)
                    action = rng.choices(list(ap.keys()),
                                         weights=probs.tolist(), k=1)[0]
                    state.apply_action(action)


class TestDeepCFRCheckpoint:

    def test_checkpoint_round_trip(self, game_1c, tmp_dir):
        import torch
        from open_spiel.python.pytorch.deep_cfr import DeepCFRSolver
        from train import save_deep_cfr_checkpoint, load_deep_cfr_checkpoint

        solver = DeepCFRSolver(
            game_1c,
            num_iterations=2,
            num_traversals=3,
            policy_network_train_steps=10,
            advantage_network_train_steps=10,
        )
        solver.solve()

        # Save
        save_deep_cfr_checkpoint(solver, round_num=1, num_players=3,
                                 output_dir=tmp_dir)

        # Load
        data = load_deep_cfr_checkpoint(tmp_dir, round_num=1)
        assert data is not None
        assert data["solver_type"] == "deep_cfr"
        assert data["round"] == 1
        assert data["num_players"] == 3

        # Reconstruct and compare weights
        solver2 = DeepCFRSolver(game_1c, num_iterations=1, num_traversals=1)
        solver2._policy_network.load_state_dict(data["policy_network"])

        # Check that weights are the same
        for p1, p2 in zip(solver._policy_network.parameters(),
                          solver2._policy_network.parameters()):
            torch.testing.assert_close(p1, p2)


class TestDeepCFRTrainRound:

    def test_train_round_deep_cfr_produces_summary(self, tmp_dir):
        from train import train_round_deep_cfr
        cfg = {
            "iterations": 2,
            "traversals": 5,
            "lr": 1e-3,
            "batch_size": 32,
            "memory_capacity": 1000,
            "policy_layers": (64, 64),
            "advantage_layers": (64, 64),
            "policy_train_steps": 20,
            "advantage_train_steps": 10,
        }
        summary = train_round_deep_cfr(
            num_players=3, num_cards=1, cfg=cfg, output_dir=tmp_dir,
        )
        assert summary["solver"] == "deep_cfr"
        assert summary["round"] == 1
        assert "policy_loss" in summary
        assert "final_mean_returns" in summary
        assert len(summary["final_mean_returns"]) == 3
        # Check checkpoint file exists
        assert os.path.exists(os.path.join(tmp_dir, "round_1_deep_cfr.pt"))

    def test_debug_deep_cfr_completes(self, tmp_dir):
        """Deep CFR debug mode should complete quickly."""
        t0 = time.time()
        train_main(["--debug", "--solver", "deep_cfr", "--output-dir", tmp_dir])
        elapsed = time.time() - t0
        assert elapsed < 120, f"Deep CFR debug took {elapsed:.1f}s (limit: 120s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
