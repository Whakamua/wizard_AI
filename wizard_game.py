"""Wizard card game implemented as an OpenSpiel Python game.

A single round of Wizard: deal cards, reveal trump, bid tricks, play tricks,
score. Parameterized by num_players (3-6) and num_cards (round number, 1-20).

Card encoding (60 cards):
  Number cards: color * 13 + (value - 1), colors 0-3, values 1-13  -> 0..51
  Wizards: 52..55
  Jesters: 56..59

Action encoding (85 actions):
  0..59   : card (deal / play)
  60..80  : bid (predict 0..20 tricks)
  81..84  : choose trump color (yellow/red/green/blue)
"""

import enum
import numpy as np
import pyspiel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CARDS_TOTAL = 60
NUM_COLORS = 4
NUM_VALUES = 13  # 1..13
NUM_WIZARDS = 4
NUM_JESTERS = 4

COLOR_NAMES = ["Yellow", "Red", "Green", "Blue"]
WIZARD_START = NUM_COLORS * NUM_VALUES  # 52
JESTER_START = WIZARD_START + NUM_WIZARDS  # 56

MAX_BID = 20
BID_OFFSET = NUM_CARDS_TOTAL  # bids are actions 60..80
TRUMP_CHOICE_OFFSET = BID_OFFSET + MAX_BID + 1  # 81..84
NUM_DISTINCT_ACTIONS = TRUMP_CHOICE_OFFSET + NUM_COLORS  # 85

NO_TRUMP = -1


class Phase(enum.IntEnum):
    DEAL = 0
    REVEAL_TRUMP = 1
    CHOOSE_TRUMP = 2
    BID = 3
    PLAY = 4


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def card_color(card: int) -> int:
    """Return color index (0-3) for number cards, or -1 for wizard/jester."""
    if card < WIZARD_START:
        return card // NUM_VALUES
    return -1


def card_value(card: int) -> int:
    """Return value (1-13) for number cards, or -1."""
    if card < WIZARD_START:
        return (card % NUM_VALUES) + 1
    return -1


def is_wizard(card: int) -> bool:
    return WIZARD_START <= card < JESTER_START


def is_jester(card: int) -> bool:
    return JESTER_START <= card < NUM_CARDS_TOTAL


def card_name(card: int) -> str:
    if is_wizard(card):
        return f"W{card - WIZARD_START + 1}"
    if is_jester(card):
        return f"J{card - JESTER_START + 1}"
    c = COLOR_NAMES[card_color(card)]
    return f"{c[0]}{card_value(card)}"


# ---------------------------------------------------------------------------
# Suit isomorphism helpers
# ---------------------------------------------------------------------------

# Canonical color labels: index 0 = trump (T), 1-3 = non-trump (A, B, C)
_CANONICAL_NAMES = ["T", "A", "B", "C"]


def canonical_suit_map(trump_color: int, hand_cards, play_history_cards=()) -> dict:
    """Map original color indices to canonical indices.

    Trump always maps to 0.  Non-trump suits map to 1, 2, 3 based on the
    order they first appear in the player's sorted hand, then the play
    history.  Unseen suits are filled in ascending original-index order.
    """
    mapping = {}
    if trump_color >= 0:
        mapping[trump_color] = 0
    next_id = 1 if trump_color >= 0 else 0
    for card in list(sorted(hand_cards)) + list(play_history_cards):
        c = card_color(card)
        if c >= 0 and c not in mapping:
            mapping[c] = next_id
            next_id += 1
    # Fill remaining unseen suits deterministically
    for c in range(NUM_COLORS):
        if c not in mapping:
            mapping[c] = next_id
            next_id += 1
    return mapping


def remap_card(card: int, suit_map: dict) -> int:
    """Return the card id with its suit replaced according to *suit_map*.

    Wizards and Jesters are unchanged.
    """
    if is_wizard(card) or is_jester(card):
        return card
    return suit_map[card_color(card)] * NUM_VALUES + (card_value(card) - 1)


def canonical_card_name(card: int, suit_map: dict) -> str:
    """Human-readable name using canonical suit labels."""
    if is_wizard(card):
        return f"W{card - WIZARD_START + 1}"
    if is_jester(card):
        return f"J{card - JESTER_START + 1}"
    canon_idx = suit_map[card_color(card)]
    return f"{_CANONICAL_NAMES[canon_idx]}{card_value(card)}"


def _get_play_history(state) -> list:
    """Extract the list of card-play actions from the full history."""
    hist = state.history()
    deal_actions = state._total_to_deal
    trump_actions = 0 if state._is_last_round else 1
    choose_actions = (
        1 if (state._trump_card >= 0 and is_wizard(state._trump_card))
        else 0
    )
    bid_actions = len(state._bids)
    play_start = deal_actions + trump_actions + choose_actions + bid_actions
    return hist[play_start:]


# ---------------------------------------------------------------------------
# OpenSpiel game registration
# ---------------------------------------------------------------------------

_GAME_TYPE = pyspiel.GameType(
    short_name="python_wizard",
    long_name="Python Wizard",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=6,
    min_num_players=3,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
    parameter_specification={
        "num_players": 3,
        "num_cards": 1,
    },
)

_MAX_GAME_LENGTH = (
    60  # deal (up to 60 chance nodes)
    + 1  # reveal trump
    + 1  # choose trump
    + 6  # bids (up to 6 players)
    + 60  # card plays (up to 20 tricks * 6 players, but capped by 60 cards)
)


def _make_game_info(num_players: int, num_cards: int) -> pyspiel.GameInfo:
    # Scoring: best case = correct bid on num_cards tricks = 20 + 10*num_cards
    # worst case = bid num_cards, win 0 = -10*num_cards
    max_util = 20.0 + 10.0 * num_cards
    min_util = -10.0 * num_cards
    return pyspiel.GameInfo(
        num_distinct_actions=NUM_DISTINCT_ACTIONS,
        max_chance_outcomes=NUM_CARDS_TOTAL,
        num_players=num_players,
        min_utility=min_util,
        max_utility=max_util,
        max_game_length=_MAX_GAME_LENGTH,
    )


# ---------------------------------------------------------------------------
# WizardGame
# ---------------------------------------------------------------------------

class WizardGame(pyspiel.Game):

    def __init__(self, params=None):
        params = params or {}
        self._num_players = int(params.get("num_players", 3))
        self._num_cards = int(params.get("num_cards", 1))
        assert 3 <= self._num_players <= 6, "num_players must be 3-6"
        max_cards = NUM_CARDS_TOTAL // self._num_players
        assert 1 <= self._num_cards <= max_cards, (
            f"num_cards must be 1-{max_cards} for {self._num_players} players"
        )
        game_info = _make_game_info(self._num_players, self._num_cards)
        super().__init__(_GAME_TYPE, game_info, params)

    def new_initial_state(self):
        return WizardState(self)

    def num_players(self):
        return self._num_players

    def make_py_observer(self, iig_obs_type=None, params=None):
        return WizardObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True),
            self._num_players,
            self._num_cards,
            params,
        )


# ---------------------------------------------------------------------------
# WizardState
# ---------------------------------------------------------------------------

class WizardState(pyspiel.State):

    def __init__(self, game: WizardGame):
        super().__init__(game)
        self._num_players = game._num_players
        self._num_cards = game._num_cards
        self._is_last_round = (
            self._num_cards * self._num_players == NUM_CARDS_TOTAL
        )

        # Card tracking: sets for speed
        self._deck = set(range(NUM_CARDS_TOTAL))  # remaining in deck
        self._hands = [set() for _ in range(self._num_players)]  # per-player
        self._deal_count = 0  # cards dealt so far
        self._total_to_deal = self._num_cards * self._num_players

        # Trump
        self._trump_card = -1  # revealed card
        self._trump_color = NO_TRUMP  # resolved color

        # Bidding
        self._bids = []  # list of (player, bid) in order
        self._bid_index = 0  # how many players have bid

        # Playing
        self._tricks_won = [0] * self._num_players
        self._current_trick = []  # list of (player, card)
        self._trick_leader = 1 % self._num_players  # left of dealer (0)
        self._tricks_played = 0
        self._play_index = 0  # within current trick

        # Phase
        self._phase = Phase.DEAL
        self._game_over = False

    # --- Core API ---

    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        if self._phase in (Phase.DEAL, Phase.REVEAL_TRUMP):
            return pyspiel.PlayerId.CHANCE
        if self._phase == Phase.CHOOSE_TRUMP:
            return 0  # dealer is player 0
        if self._phase == Phase.BID:
            # bid order: left of dealer, clockwise
            return (1 + self._bid_index) % self._num_players
        # Phase.PLAY
        return (self._trick_leader + self._play_index) % self._num_players

    def _legal_actions(self, player):
        """Legal actions for a non-chance player."""
        if self._phase == Phase.CHOOSE_TRUMP:
            return list(range(TRUMP_CHOICE_OFFSET, TRUMP_CHOICE_OFFSET + NUM_COLORS))
        if self._phase == Phase.BID:
            return list(range(BID_OFFSET, BID_OFFSET + self._num_cards + 1))
        if self._phase == Phase.PLAY:
            return self._legal_play_actions(player)
        return []

    def _legal_play_actions(self, player):
        hand = self._hands[player]
        if not hand:
            return []
        # First card in trick or trick is all jesters so far -> any card
        if self._play_index == 0:
            return sorted(hand)
        # Determine the suit to follow
        lead_suit = self._trick_lead_suit()
        if lead_suit == -1:
            # Lead was wizard or only jesters so far -> any card
            return sorted(hand)
        # Must follow suit if possible (wizards/jesters always allowed)
        suited = [c for c in hand if card_color(c) == lead_suit]
        specials = [c for c in hand if is_wizard(c) or is_jester(c)]
        if suited:
            return sorted(suited + specials)
        # No cards of that suit -> play anything
        return sorted(hand)

    def _trick_lead_suit(self):
        """Return the suit that must be followed, or -1 if none."""
        for _, card in self._current_trick:
            if is_wizard(card):
                return -1  # wizard lead: no suit to follow
            if is_jester(card):
                continue  # skip jesters to find the actual lead suit
            return card_color(card)
        return -1  # all jesters so far

    def chance_outcomes(self):
        assert self.is_chance_node()
        remaining = sorted(self._deck)
        p = 1.0 / len(remaining)
        return [(c, p) for c in remaining]

    def _apply_action(self, action):
        if self._phase == Phase.DEAL:
            self._apply_deal(action)
        elif self._phase == Phase.REVEAL_TRUMP:
            self._apply_reveal_trump(action)
        elif self._phase == Phase.CHOOSE_TRUMP:
            self._apply_choose_trump(action)
        elif self._phase == Phase.BID:
            self._apply_bid(action)
        elif self._phase == Phase.PLAY:
            self._apply_play(action)

    def _apply_deal(self, card):
        player = self._deal_count % self._num_players
        self._hands[player].add(card)
        self._deck.discard(card)
        self._deal_count += 1
        if self._deal_count == self._total_to_deal:
            if self._is_last_round:
                self._phase = Phase.BID
            else:
                self._phase = Phase.REVEAL_TRUMP

    def _apply_reveal_trump(self, card):
        self._deck.discard(card)
        self._trump_card = card
        if is_wizard(card):
            self._phase = Phase.CHOOSE_TRUMP
        elif is_jester(card):
            self._trump_color = NO_TRUMP
            self._phase = Phase.BID
        else:
            self._trump_color = card_color(card)
            self._phase = Phase.BID

    def _apply_choose_trump(self, action):
        self._trump_color = action - TRUMP_CHOICE_OFFSET
        self._phase = Phase.BID

    def _apply_bid(self, action):
        bid = action - BID_OFFSET
        player = (1 + self._bid_index) % self._num_players
        self._bids.append((player, bid))
        self._bid_index += 1
        if self._bid_index == self._num_players:
            self._phase = Phase.PLAY

    def _apply_play(self, card):
        player = (self._trick_leader + self._play_index) % self._num_players
        self._hands[player].discard(card)
        self._current_trick.append((player, card))
        self._play_index += 1
        if self._play_index == self._num_players:
            self._resolve_trick()

    def _resolve_trick(self):
        winner = self._trick_winner()
        self._tricks_won[winner] += 1
        self._tricks_played += 1
        self._current_trick = []
        self._play_index = 0
        self._trick_leader = winner
        if self._tricks_played == self._num_cards:
            self._game_over = True

    def _trick_winner(self):
        """Determine winner of the current trick."""
        # First wizard wins
        for player, card in self._current_trick:
            if is_wizard(card):
                return player
        # No wizards: highest trump wins
        best_player, best_val = -1, -1
        for player, card in self._current_trick:
            if not is_jester(card) and card_color(card) == self._trump_color:
                v = card_value(card)
                if v > best_val:
                    best_val = v
                    best_player = player
        if best_player >= 0:
            return best_player
        # No trumps: highest in led suit wins
        lead_suit = -1
        for _, card in self._current_trick:
            if not is_jester(card):
                lead_suit = card_color(card)
                break
        if lead_suit >= 0:
            for player, card in self._current_trick:
                if not is_jester(card) and card_color(card) == lead_suit:
                    v = card_value(card)
                    if v > best_val:
                        best_val = v
                        best_player = player
            if best_player >= 0:
                return best_player
        # All jesters: first jester wins
        return self._current_trick[0][0]

    def is_terminal(self):
        return self._game_over

    def returns(self):
        if not self._game_over:
            return [0.0] * self._num_players
        scores = [0.0] * self._num_players
        bid_map = {p: b for p, b in self._bids}
        for p in range(self._num_players):
            bid = bid_map[p]
            won = self._tricks_won[p]
            if bid == won:
                scores[p] = 20.0 + 10.0 * won
            else:
                scores[p] = -10.0 * abs(won - bid)
        return scores

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            return f"Deal:{card_name(action)}"
        if BID_OFFSET <= action <= BID_OFFSET + MAX_BID:
            return f"Bid:{action - BID_OFFSET}"
        if TRUMP_CHOICE_OFFSET <= action < TRUMP_CHOICE_OFFSET + NUM_COLORS:
            return f"Trump:{COLOR_NAMES[action - TRUMP_CHOICE_OFFSET]}"
        return f"Play:{card_name(action)}"

    def __str__(self):
        parts = [f"Phase:{self._phase.name}"]
        if self._trump_color != NO_TRUMP:
            parts.append(f"Trump:{COLOR_NAMES[self._trump_color]}")
        elif self._trump_card >= 0 and is_jester(self._trump_card):
            parts.append("Trump:None")
        for p, b in self._bids:
            parts.append(f"P{p}bid{b}")
        parts.append(f"Tricks:{self._tricks_won}")
        if self._current_trick:
            trick_str = " ".join(
                f"P{p}:{card_name(c)}" for p, c in self._current_trick
            )
            parts.append(f"Current:[{trick_str}]")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# WizardObserver
# ---------------------------------------------------------------------------

# Tensor layout:
#   player one-hot         : num_players
#   hand bitmap            : 60
#   trump color one-hot    : 5  (4 colors + no-trump)
#   bids                   : num_players (normalized by num_cards)
#   bid made flags         : num_players (1 if player has bid)
#   cards played history   : num_cards * num_players * 60  (one-hot per slot)
#   tricks won             : num_players (normalized by num_cards)

def _tensor_size(num_players, num_cards):
    return (
        num_players  # player
        + NUM_CARDS_TOTAL  # hand
        + 5  # trump color
        + num_players  # bids
        + num_players  # bid flags
        + num_cards * num_players * NUM_CARDS_TOTAL  # play history
        + num_players  # tricks won
    )


class WizardObserver:
    """Observer for Wizard, conforming to the PyObserver interface."""

    def __init__(self, iig_obs_type, num_players, num_cards, params):
        if params:
            raise ValueError(f"Observation parameters not supported; got {params}")

        self._num_players = num_players
        self._num_cards = num_cards
        self._iig_obs_type = iig_obs_type

        # Build named pieces → views into a single flat tensor
        pieces = [("player", num_players, (num_players,))]
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("hand", NUM_CARDS_TOTAL, (NUM_CARDS_TOTAL,)))
        if iig_obs_type.public_info:
            pieces.append(("trump_color", 5, (5,)))
            pieces.append(("bids", num_players, (num_players,)))
            pieces.append(("bid_flags", num_players, (num_players,)))
            pieces.append(("tricks_won", num_players, (num_players,)))
            if iig_obs_type.perfect_recall:
                play_size = num_cards * num_players * NUM_CARDS_TOTAL
                pieces.append(("play_history", play_size,
                               (num_cards * num_players, NUM_CARDS_TOTAL)))

        total_size = sum(s for _, s, _ in pieces)
        self.tensor = np.zeros(total_size, np.float32)
        self.dict = {}
        idx = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[idx:idx + size].reshape(shape)
            idx += size

    def set_from(self, state, player):
        """Populate tensor from state for the given player."""
        self.tensor.fill(0)
        nc = self._num_cards
        np_ = self._num_players

        # Build canonical suit mapping
        play_hist = _get_play_history(state)
        suit_map = canonical_suit_map(
            state._trump_color, state._hands[player], play_hist,
        )

        # Player one-hot
        self.dict["player"][player] = 1

        # Hand bitmap (private info) — remapped suits
        if "hand" in self.dict:
            for card in state._hands[player]:
                self.dict["hand"][remap_card(card, suit_map)] = 1

        # Trump color one-hot (public) — canonical: trump is always slot 0
        if "trump_color" in self.dict:
            if state._trump_color == NO_TRUMP:
                self.dict["trump_color"][4] = 1
            else:
                self.dict["trump_color"][0] = 1  # trump → canonical 0

        # Bids (public)
        if "bids" in self.dict:
            bid_map = {p: b for p, b in state._bids}
            for p in range(np_):
                if p in bid_map:
                    self.dict["bids"][p] = bid_map[p] / max(nc, 1)
                    self.dict["bid_flags"][p] = 1

        # Tricks won (public)
        if "tricks_won" in self.dict:
            for p in range(np_):
                self.dict["tricks_won"][p] = state._tricks_won[p] / max(nc, 1)

        # Play history (perfect recall) — remapped suits
        if "play_history" in self.dict:
            for slot, card in enumerate(play_hist):
                if slot < self.dict["play_history"].shape[0]:
                    self.dict["play_history"][slot, remap_card(card, suit_map)] = 1

    def string_from(self, state, player):
        """Information state string for CFR: must be unique per info set.

        Uses canonical suit labels so that strategically equivalent states
        (differing only by non-trump suit identity) map to the same string.
        """
        play_hist = _get_play_history(state)
        suit_map = canonical_suit_map(
            state._trump_color, state._hands[player], play_hist,
        )
        _cn = lambda c: canonical_card_name(c, suit_map)  # noqa: E731

        parts = []
        parts.append(f"p{player}")
        # Private: own hand (sorted by remapped card id for consistency)
        hand = sorted(state._hands[player], key=lambda c: remap_card(c, suit_map))
        parts.append("h:" + ",".join(_cn(c) for c in hand))
        # Public: trump — canonical label is always "T" (or "N" for none)
        if state._trump_color >= 0:
            parts.append("t:T")
        elif state._phase.value >= Phase.BID.value:
            parts.append("t:N")
        # Public: all bids
        if state._bids:
            bids_str = ",".join(f"{p}:{b}" for p, b in state._bids)
            parts.append(f"b:[{bids_str}]")
        # Public: play history
        if state._phase == Phase.PLAY or state._game_over:
            parts.append(f"tw:{state._tricks_won}")
            parts.append(f"tp:{state._tricks_played}")
            if state._current_trick:
                ct = ",".join(
                    f"{p}:{_cn(c)}" for p, c in state._current_trick
                )
                parts.append(f"ct:[{ct}]")
            if play_hist:
                parts.append("ph:" + ",".join(_cn(a) for a in play_hist))
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

pyspiel.register_game(_GAME_TYPE, WizardGame)
