"""Tests for the Wizard OpenSpiel game environment."""

import sys
import os
import random
import pytest
import numpy as np

# Ensure the wizard_AI package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pyspiel
import wizard_game  # registers the game on import
from wizard_game import (
    NUM_CARDS_TOTAL, NUM_COLORS, NUM_VALUES, WIZARD_START, JESTER_START,
    BID_OFFSET, TRUMP_CHOICE_OFFSET, NO_TRUMP, Phase,
    card_color, card_value, is_wizard, is_jester, card_name,
)


# ---------------------------------------------------------------------------
# Registration & loading
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_game_registered(self):
        assert "python_wizard" in pyspiel.registered_names()

    def test_load_default(self):
        game = pyspiel.load_game("python_wizard")
        assert game.num_players() == 3

    @pytest.mark.parametrize("np_,nc", [(3, 1), (3, 5), (4, 3), (5, 2), (6, 1)])
    def test_load_parameterized(self, np_, nc):
        game = pyspiel.load_game("python_wizard", {"num_players": np_, "num_cards": nc})
        assert game.num_players() == np_


# ---------------------------------------------------------------------------
# Random playthroughs
# ---------------------------------------------------------------------------

def _play_random_game(num_players=3, num_cards=1, seed=None):
    """Play a full random game and return the terminal state."""
    rng = random.Random(seed)
    game = pyspiel.load_game(
        "python_wizard", {"num_players": num_players, "num_cards": num_cards}
    )
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = rng.choices(actions, weights=probs, k=1)[0]
        else:
            legal = state.legal_actions()
            assert len(legal) > 0, f"No legal actions in phase {state._phase}"
            action = rng.choice(legal)
        state.apply_action(action)
    return state


class TestRandomPlaythroughs:

    @pytest.mark.parametrize("seed", range(100))
    def test_random_game_3p_1c(self, seed):
        state = _play_random_game(3, 1, seed)
        assert state.is_terminal()
        returns = state.returns()
        assert len(returns) == 3

    @pytest.mark.parametrize("seed", range(50))
    def test_random_game_3p_3c(self, seed):
        state = _play_random_game(3, 3, seed)
        assert state.is_terminal()

    @pytest.mark.parametrize("seed", range(20))
    def test_random_game_4p_2c(self, seed):
        state = _play_random_game(4, 2, seed)
        assert state.is_terminal()

    @pytest.mark.parametrize("seed", range(20))
    def test_random_game_5p_2c(self, seed):
        state = _play_random_game(5, 2, seed)
        assert state.is_terminal()

    @pytest.mark.parametrize("seed", range(10))
    def test_random_game_6p_1c(self, seed):
        state = _play_random_game(6, 1, seed)
        assert state.is_terminal()

    def test_random_game_last_round_3p(self):
        """Last round: 20 cards each for 3 players, no trump."""
        state = _play_random_game(3, 20, seed=42)
        assert state.is_terminal()

    @pytest.mark.parametrize("seed", range(200))
    def test_tricks_won_sum(self, seed):
        """Total tricks won must equal num_cards."""
        nc = random.Random(seed).choice([1, 2, 3, 4, 5])
        state = _play_random_game(3, nc, seed)
        assert sum(state._tricks_won) == nc


# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

class TestCardHelpers:

    def test_number_card_colors(self):
        for c in range(4):
            for v in range(13):
                card = c * 13 + v
                assert card_color(card) == c
                assert card_value(card) == v + 1

    def test_wizards(self):
        for i in range(4):
            card = WIZARD_START + i
            assert is_wizard(card)
            assert not is_jester(card)
            assert card_color(card) == -1

    def test_jesters(self):
        for i in range(4):
            card = JESTER_START + i
            assert is_jester(card)
            assert not is_wizard(card)
            assert card_color(card) == -1

    def test_card_names(self):
        assert card_name(0) == "Y1"  # Yellow 1
        assert card_name(12) == "Y13"
        assert card_name(13) == "R1"  # Red 1
        assert card_name(52) == "W1"  # Wizard 1
        assert card_name(56) == "J1"  # Jester 1


# ---------------------------------------------------------------------------
# Trick winner logic
# ---------------------------------------------------------------------------

class TestTrickWinner:

    def _make_state_with_trick(self, trick_cards, trump_color=NO_TRUMP):
        """Create a state with a pre-set current trick and trump color."""
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 5})
        state = game.new_initial_state()
        # Directly set internal state for testing
        state._trump_color = trump_color
        state._current_trick = trick_cards
        return state

    def test_wizard_wins(self):
        # Player 0: Y1, Player 1: W1, Player 2: R13
        trick = [(0, 0), (1, 52), (2, 25)]  # Y1, W1, R13
        state = self._make_state_with_trick(trick)
        assert state._trick_winner() == 1

    def test_first_wizard_wins(self):
        # Two wizards: first one wins
        trick = [(0, 0), (1, 52), (2, 53)]  # Y1, W1, W2
        state = self._make_state_with_trick(trick)
        assert state._trick_winner() == 1

    def test_trump_beats_non_trump(self):
        # Trump is Red (1). Player 0: Y13, Player 1: R1, Player 2: Y12
        trick = [(0, 12), (1, 13), (2, 11)]  # Y13, R1, Y12
        state = self._make_state_with_trick(trick, trump_color=1)
        assert state._trick_winner() == 1  # Red 1 is trump

    def test_highest_trump_wins(self):
        # Trump is Red (1). Multiple trumps.
        trick = [(0, 13), (1, 14), (2, 15)]  # R1, R2, R3
        state = self._make_state_with_trick(trick, trump_color=1)
        assert state._trick_winner() == 2  # R3 highest

    def test_highest_lead_suit_wins(self):
        # No trump. Lead suit is Yellow (0).
        trick = [(0, 0), (1, 5), (2, 12)]  # Y1, Y6, Y13
        state = self._make_state_with_trick(trick)
        assert state._trick_winner() == 2  # Y13

    def test_off_suit_loses(self):
        # No trump. Lead Yellow, one player plays Red.
        trick = [(0, 0), (1, 13), (2, 5)]  # Y1, R1, Y6
        state = self._make_state_with_trick(trick)
        assert state._trick_winner() == 2  # Y6 beats Y1, R1 doesn't count

    def test_all_jesters_first_wins(self):
        trick = [(0, 56), (1, 57), (2, 58)]  # J1, J2, J3
        state = self._make_state_with_trick(trick)
        assert state._trick_winner() == 0

    def test_jester_loses_to_number(self):
        trick = [(0, 56), (1, 0), (2, 57)]  # J1, Y1, J2
        state = self._make_state_with_trick(trick)
        assert state._trick_winner() == 1


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:

    def test_correct_bid_zero(self):
        """Bid 0, won 0 -> 20 points."""
        state = _play_random_game(3, 1, seed=0)
        # Manually test scoring logic
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 2})
        s = game.new_initial_state()
        s._game_over = True
        s._bids = [(0, 0), (1, 1), (2, 0)]
        s._tricks_won = [0, 1, 1]
        scores = s.returns()
        assert scores[0] == 20.0  # correct: 20 + 10*0 = 20
        assert scores[1] == 30.0  # correct: 20 + 10*1 = 30
        assert scores[2] == -10.0  # wrong by 1: -10

    def test_correct_bid_nonzero(self):
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 3})
        s = game.new_initial_state()
        s._game_over = True
        s._bids = [(0, 2), (1, 0), (2, 1)]
        s._tricks_won = [2, 0, 1]
        scores = s.returns()
        assert scores[0] == 40.0  # 20 + 10*2
        assert scores[1] == 20.0  # 20 + 10*0
        assert scores[2] == 30.0  # 20 + 10*1

    def test_wrong_bid_penalty(self):
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 3})
        s = game.new_initial_state()
        s._game_over = True
        s._bids = [(0, 3), (1, 3), (2, 3)]
        s._tricks_won = [1, 1, 1]
        scores = s.returns()
        # Each bid 3 but won 1: off by 2 -> -20
        assert scores[0] == -20.0
        assert scores[1] == -20.0
        assert scores[2] == -20.0


# ---------------------------------------------------------------------------
# Legal actions
# ---------------------------------------------------------------------------

class TestLegalActions:

    def test_bid_range(self):
        """Bid actions should be BID_OFFSET..BID_OFFSET+num_cards."""
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 5})
        state = game.new_initial_state()
        # Fast-forward through dealing by playing random chance actions
        rng = random.Random(42)
        while state._phase in (Phase.DEAL, Phase.REVEAL_TRUMP, Phase.CHOOSE_TRUMP):
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                state.apply_action(rng.choices(actions, weights=probs, k=1)[0])
            else:
                legal = state.legal_actions()
                state.apply_action(rng.choice(legal))
        assert state._phase == Phase.BID
        legal = state.legal_actions()
        assert legal == list(range(BID_OFFSET, BID_OFFSET + 6))  # bid 0..5

    def test_suit_following(self):
        """When a suit is led, must follow suit (or play wizard/jester)."""
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 5})
        state = game.new_initial_state()

        # Manually set up a play scenario
        state._phase = Phase.PLAY
        state._num_cards = 5
        state._hands = [{0, 1, 13, 52, 56}, {2, 14, 15, 26, 39}, {3, 27, 40, 53, 57}]
        state._trick_leader = 0
        state._play_index = 0

        # Player 0 leads: can play anything
        legal_p0 = state._legal_play_actions(0)
        assert legal_p0 == sorted(state._hands[0])

        # Simulate player 0 plays Y1 (card 0, Yellow)
        state._current_trick = [(0, 0)]
        state._play_index = 1

        # Player 1 must follow Yellow. Has Y3(2), R2(14), R3(15), G1(26), B1(39)
        # Yellow cards: Y3(2). Specials: none.
        legal_p1 = state._legal_play_actions(1)
        assert legal_p1 == [2]  # only Y3

    def test_no_suit_can_play_anything(self):
        """If you don't have the led suit, play anything."""
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 3})
        state = game.new_initial_state()
        state._phase = Phase.PLAY
        state._hands = [set(), {14, 15, 53}, set()]  # Player 1: R2, R3, W2
        state._current_trick = [(0, 0)]  # Lead Yellow
        state._play_index = 1
        state._trick_leader = 0

        # Player 1 has no Yellow -> can play anything
        legal = state._legal_play_actions(1)
        assert legal == sorted({14, 15, 53})

    def test_wizard_jester_always_playable(self):
        """Wizards and Jesters can always be played even when following suit."""
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 3})
        state = game.new_initial_state()
        state._phase = Phase.PLAY
        # Player 1 has: Y3(2), W1(52), J1(56)
        state._hands = [set(), {2, 52, 56}, set()]
        state._current_trick = [(0, 0)]  # Lead Yellow
        state._play_index = 1
        state._trick_leader = 0

        legal = state._legal_play_actions(1)
        # Has Yellow (Y3), plus Wizard and Jester are always legal
        assert 2 in legal   # Y3
        assert 52 in legal  # W1
        assert 56 in legal  # J1

    def test_wizard_lead_no_suit_restriction(self):
        """Leading with a Wizard means others can play anything."""
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 3})
        state = game.new_initial_state()
        state._phase = Phase.PLAY
        state._hands = [set(), {2, 14, 52}, set()]
        state._current_trick = [(0, 52)]  # Lead Wizard
        state._play_index = 1
        state._trick_leader = 0

        legal = state._legal_play_actions(1)
        assert legal == sorted({2, 14, 52})

    def test_jester_lead_then_suit_set_by_first_number(self):
        """Jester lead: suit is set by the first number card played."""
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 3})
        state = game.new_initial_state()
        state._phase = Phase.PLAY
        # Player 0 leads Jester, player 1 plays R1
        state._hands = [set(), set(), {2, 14, 56}]  # P2: Y3, R2, J1
        state._current_trick = [(0, 56), (1, 13)]  # Jester, then R1
        state._play_index = 2
        state._trick_leader = 0

        legal = state._legal_play_actions(2)
        # Must follow Red. Has R2(14) + J1(56) is special
        assert 14 in legal  # R2
        assert 56 in legal  # J1
        assert 2 not in legal  # Y3 can't play


# ---------------------------------------------------------------------------
# Trump determination
# ---------------------------------------------------------------------------

class TestTrumpDetermination:

    def test_number_card_sets_trump(self):
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 1})
        state = game.new_initial_state()
        # Deal 3 cards
        state.apply_action(0)   # P0: Y1
        state.apply_action(13)  # P1: R1
        state.apply_action(26)  # P2: G1
        # Now reveal trump
        assert state._phase == Phase.REVEAL_TRUMP
        state.apply_action(14)  # R2 -> trump is Red
        assert state._trump_color == 1
        assert state._phase == Phase.BID

    def test_jester_no_trump(self):
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 1})
        state = game.new_initial_state()
        state.apply_action(0)
        state.apply_action(13)
        state.apply_action(26)
        state.apply_action(56)  # Jester -> no trump
        assert state._trump_color == NO_TRUMP
        assert state._phase == Phase.BID

    def test_wizard_dealer_chooses(self):
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 1})
        state = game.new_initial_state()
        state.apply_action(0)
        state.apply_action(13)
        state.apply_action(26)
        state.apply_action(52)  # Wizard -> dealer chooses
        assert state._phase == Phase.CHOOSE_TRUMP
        assert state.current_player() == 0  # dealer
        legal = state.legal_actions()
        assert legal == list(range(TRUMP_CHOICE_OFFSET, TRUMP_CHOICE_OFFSET + 4))
        state.apply_action(TRUMP_CHOICE_OFFSET + 2)  # choose Green
        assert state._trump_color == 2
        assert state._phase == Phase.BID

    def test_last_round_no_trump(self):
        """Last round (all cards dealt): no trump reveal."""
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 20})
        state = game.new_initial_state()
        rng = random.Random(99)
        while state._phase == Phase.DEAL:
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(rng.choices(actions, weights=probs, k=1)[0])
        # Should skip directly to BID
        assert state._phase == Phase.BID


# ---------------------------------------------------------------------------
# Observer / information state
# ---------------------------------------------------------------------------

class TestObserver:

    def test_info_state_string_unique(self):
        """Different hands should produce different info state strings."""
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 1})
        state = game.new_initial_state()
        state.apply_action(0)   # P0: Y1
        state.apply_action(13)  # P1: R1
        state.apply_action(26)  # P2: G1
        state.apply_action(14)  # Trump: R2

        s0 = state.information_state_string(0)
        s1 = state.information_state_string(1)
        s2 = state.information_state_string(2)
        assert s0 != s1
        assert s1 != s2
        assert "Y1" in s0
        assert "R1" in s1
        assert "G1" in s2

    def test_info_state_tensor_shape(self):
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 1})
        state = game.new_initial_state()
        state.apply_action(0)
        state.apply_action(13)
        state.apply_action(26)
        state.apply_action(14)

        tensor = state.information_state_tensor(0)
        assert isinstance(tensor, list) or isinstance(tensor, np.ndarray)
        assert len(tensor) > 0

    def test_observation_string(self):
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 1})
        state = game.new_initial_state()
        state.apply_action(0)
        state.apply_action(13)
        state.apply_action(26)
        state.apply_action(14)

        obs = state.observation_string(0)
        assert isinstance(obs, str)
        assert len(obs) > 0


# ---------------------------------------------------------------------------
# Full game flow integration
# ---------------------------------------------------------------------------

class TestFullGameFlow:

    def test_complete_1card_game(self):
        """Manually play a complete 1-card round."""
        game = pyspiel.load_game("python_wizard", {"num_players": 3, "num_cards": 1})
        state = game.new_initial_state()

        # Deal: P0=Y1(0), P1=R1(13), P2=G1(26)
        state.apply_action(0)
        state.apply_action(13)
        state.apply_action(26)

        # Trump: B1(39) -> Blue trump
        state.apply_action(39)
        assert state._trump_color == 3  # Blue

        # Bids: P1 bids 0, P2 bids 0, P0 bids 1
        assert state.current_player() == 1
        state.apply_action(BID_OFFSET + 0)  # P1 bids 0
        assert state.current_player() == 2
        state.apply_action(BID_OFFSET + 0)  # P2 bids 0
        assert state.current_player() == 0
        state.apply_action(BID_OFFSET + 1)  # P0 bids 1

        # Play: P1 starts (left of dealer)
        assert state._phase == Phase.PLAY
        assert state.current_player() == 1
        state.apply_action(13)  # P1 plays R1
        # P2 has G1, no Red -> can play anything
        state.apply_action(26)  # P2 plays G1
        # P0 has Y1, no Red -> can play anything
        state.apply_action(0)   # P0 plays Y1

        assert state.is_terminal()
        # R1 wins (no trump cards, lead suit Red, only Red card)
        assert state._tricks_won[1] == 1
        scores = state.returns()
        assert scores[0] == -10.0  # bid 1, won 0
        assert scores[1] == -10.0  # bid 0, won 1
        assert scores[2] == 20.0   # bid 0, won 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
