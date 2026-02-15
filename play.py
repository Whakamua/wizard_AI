"""Interactive CLI for playing Wizard with a trained AI.

An operator feeds in the real-world game state (hands, trump, opponent plays)
and the AI recommends bids and card plays with full probability distributions.

Usage:
    python play.py --checkpoint-dir checkpoints --num-players 3
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pyspiel

import wizard_game  # noqa: F401
from wizard_game import (
    NUM_CARDS_TOTAL, NUM_COLORS, NUM_VALUES, WIZARD_START, JESTER_START,
    BID_OFFSET, TRUMP_CHOICE_OFFSET, NO_TRUMP, COLOR_NAMES, Phase,
    card_color, card_name, is_wizard, is_jester,
)
from open_spiel.python.algorithms.external_sampling_mccfr import (
    ExternalSamplingSolver,
)
from open_spiel.python.algorithms.mccfr import AveragePolicy


# ---------------------------------------------------------------------------
# Card parsing
# ---------------------------------------------------------------------------

def parse_card(token: str) -> int:
    """Parse a card token like 'R7', 'W1', 'J2', 'Y13' into a card ID.

    Formats accepted:
      Number cards: {Color initial}{Value}  e.g. Y1, R13, G7, B3
      Wizards:      W or W1..W4
      Jesters:      J or J1..J4
    """
    token = token.strip().upper()
    if not token:
        raise ValueError("Empty card token")

    color_map = {"Y": 0, "R": 1, "G": 2, "B": 3}

    if token[0] == "W":
        idx = int(token[1:]) - 1 if len(token) > 1 else 0
        if not 0 <= idx < 4:
            raise ValueError(f"Wizard index must be 1-4, got {token}")
        return WIZARD_START + idx

    if token[0] == "J":
        idx = int(token[1:]) - 1 if len(token) > 1 else 0
        if not 0 <= idx < 4:
            raise ValueError(f"Jester index must be 1-4, got {token}")
        return JESTER_START + idx

    if token[0] in color_map:
        color = color_map[token[0]]
        value = int(token[1:])
        if not 1 <= value <= NUM_VALUES:
            raise ValueError(f"Card value must be 1-{NUM_VALUES}, got {value}")
        return color * NUM_VALUES + (value - 1)

    raise ValueError(f"Cannot parse card: '{token}'. "
                     f"Use format: Y1-Y13, R1-R13, G1-G13, B1-B13, W1-W4, J1-J4")


def parse_card_list(text: str) -> list[int]:
    """Parse card tokens separated by spaces, commas, or both."""
    # Accept "W1, W2, W3" or "W1 W2 W3" or "W1,W2,W3"
    text = text.replace(",", " ")
    tokens = text.strip().split()
    return [parse_card(t) for t in tokens]


def format_probs(action_probs: dict, top_n: int = 10) -> str:
    """Format action probabilities as a readable string."""
    sorted_actions = sorted(action_probs.items(), key=lambda x: -x[1])
    lines = []
    for action, prob in sorted_actions[:top_n]:
        if BID_OFFSET <= action <= BID_OFFSET + 20:
            label = f"Bid {action - BID_OFFSET}"
        elif TRUMP_CHOICE_OFFSET <= action < TRUMP_CHOICE_OFFSET + NUM_COLORS:
            label = f"Trump: {COLOR_NAMES[action - TRUMP_CHOICE_OFFSET]}"
        elif 0 <= action < NUM_CARDS_TOTAL:
            label = f"Play {card_name(action)}"
        else:
            label = f"Action {action}"
        bar = "#" * int(prob * 40)
        lines.append(f"  {label:20s} {prob:6.1%}  {bar}")
    if len(sorted_actions) > top_n:
        lines.append(f"  ... and {len(sorted_actions) - top_n} more actions")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load policy
# ---------------------------------------------------------------------------

def _load_deep_cfr_policy(checkpoint_path: str, num_players: int, num_cards: int):
    """Load a Deep CFR checkpoint and return a solver that provides
    action_probabilities(state)."""
    import torch
    from open_spiel.python.pytorch.deep_cfr import DeepCFRSolver

    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if data.get("num_players") != num_players:
        print(f"ERROR: Checkpoint has {data['num_players']} players, "
              f"but you specified {num_players}")
        return None

    game = pyspiel.load_game("python_wizard", {
        "num_players": num_players,
        "num_cards": num_cards,
    })
    # Create a solver with minimal settings just to reconstruct networks
    solver = DeepCFRSolver(game, num_iterations=1, num_traversals=1)
    solver._policy_network.load_state_dict(data["policy_network"])
    for i, net_state in enumerate(data["advantage_networks"]):
        solver._advantage_networks[i].load_state_dict(net_state)
    print(f"Loaded Deep CFR policy: round {num_cards}, "
          f"iteration {data.get('iteration', '?')}")
    return solver


def load_policy(checkpoint_dir: str, num_players: int, num_cards: int):
    """Load the trained policy for a specific round (MCCFR or Deep CFR).

    Returns an object with action_probabilities(state) method.
    """
    # Try Deep CFR checkpoint first
    deep_path = os.path.join(checkpoint_dir, f"round_{num_cards}_deep_cfr.pt")
    mccfr_path = os.path.join(checkpoint_dir, f"round_{num_cards}_latest.pkl")

    if os.path.exists(deep_path):
        return _load_deep_cfr_policy(deep_path, num_players, num_cards)

    if not os.path.exists(mccfr_path):
        print(f"ERROR: No checkpoint found for round {num_cards}")
        print(f"  Looked for: {deep_path}")
        print(f"              {mccfr_path}")
        print(f"Train first: python train.py --num-players {num_players} "
              f"--rounds {num_cards}")
        return None

    with open(mccfr_path, "rb") as f:
        data = pickle.load(f)

    if data["num_players"] != num_players:
        print(f"ERROR: Checkpoint has {data['num_players']} players, "
              f"but you specified {num_players}")
        return None

    game = pyspiel.load_game("python_wizard", {
        "num_players": num_players,
        "num_cards": num_cards,
    })
    policy = AveragePolicy(game, list(range(num_players)), data["infostates"])
    print(f"Loaded MCCFR policy: round {num_cards}, {data['iteration']} iterations, "
          f"{len(data['infostates'])} information sets")
    return policy


# ---------------------------------------------------------------------------
# Interactive game session
# ---------------------------------------------------------------------------

def prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        sys.exit(0)


def prompt_int(msg: str, lo: int, hi: int) -> int:
    """Prompt for an integer in [lo, hi], retrying on bad input."""
    while True:
        raw = prompt(msg)
        try:
            val = int(raw)
            if lo <= val <= hi:
                return val
            print(f"  Must be between {lo} and {hi}. Try again.")
        except ValueError:
            print(f"  Invalid number: '{raw}'. Try again.")


def prompt_card(msg: str) -> int:
    """Prompt for a single card, retrying on bad input."""
    while True:
        raw = prompt(msg)
        try:
            return parse_card(raw)
        except ValueError as e:
            print(f"  {e}. Try again.")


def prompt_card_list(msg: str, expected_count: int) -> list[int]:
    """Prompt for a list of cards, retrying until count matches."""
    while True:
        raw = prompt(msg)
        try:
            cards = parse_card_list(raw)
        except ValueError as e:
            print(f"  {e}. Try again.")
            continue
        if len(cards) != expected_count:
            print(f"  Expected {expected_count} card(s), got {len(cards)}. Try again.")
            continue
        if len(set(cards)) != len(cards):
            print(f"  Duplicate cards detected. Try again.")
            continue
        return cards


def prompt_color(msg: str) -> int:
    """Prompt for a color choice (Y/R/G/B), retrying on bad input."""
    color_map = {"Y": 0, "R": 1, "G": 2, "B": 3}
    while True:
        raw = prompt(msg).upper()
        if raw in color_map:
            return color_map[raw]
        print(f"  Invalid color: '{raw}'. Enter Y, R, G, or B.")


def run_session(checkpoint_dir: str, num_players: int):
    """Run an interactive play session."""
    max_round = 60 // num_players

    print("=" * 60)
    print("  WIZARD AI - Interactive Play")
    print("=" * 60)
    print()
    print("Card notation (separate with spaces or commas):")
    print()
    print("  Number cards : {Color}{Value}")
    print("                 Y = Yellow, R = Red, G = Green, B = Blue")
    print("                 Value = 1 to 13")
    print("                 Examples: Y1  R13  G7  B3")
    print()
    print("  Wizards      : W1  W2  W3  W4")
    print("  Jesters      : J1  J2  J3  J4")
    print()
    print("  Example hand : R7 B13 W1 G3   (or: R7, B13, W1, G3)")
    print()

    while True:
        # --- Setup ---
        raw = prompt(f"Round number (1-{max_round}, or 'q' to quit): ")
        if raw.lower() == "q":
            break
        try:
            num_cards = int(raw)
            if not 1 <= num_cards <= max_round:
                print(f"  Must be between 1 and {max_round}. Try again.")
                continue
        except ValueError:
            print(f"  Invalid number: '{raw}'. Try again.")
            continue

        policy = load_policy(checkpoint_dir, num_players, num_cards)
        if policy is None:
            continue

        ai_player = prompt_int(
            f"AI player position (0-{num_players - 1}): ",
            0, num_players - 1,
        )

        # --- Create game state ---
        game = pyspiel.load_game("python_wizard", {
            "num_players": num_players,
            "num_cards": num_cards,
        })
        state = game.new_initial_state()

        # --- Deal phase: input AI's hand ---
        print(f"\nEnter the {num_cards} card(s) in the AI's hand.")
        print(f"  Example: {' '.join(['R7', 'B13', 'W1', 'G3', 'J1'][:num_cards])}")
        ai_hand = prompt_card_list("AI hand: ", num_cards)

        # Deal cards through chance nodes to build a valid state.
        # AI's cards go to ai_player; other players get arbitrary hidden cards.
        used_cards = set(ai_hand)
        remaining = [c for c in range(NUM_CARDS_TOTAL) if c not in used_cards]
        np.random.shuffle(remaining)

        other_card_idx = 0
        for deal_num in range(num_cards * num_players):
            player_for_card = deal_num % num_players
            if player_for_card == ai_player:
                card = ai_hand[deal_num // num_players]
            else:
                card = remaining[other_card_idx]
                other_card_idx += 1
            state.apply_action(card)

        # --- Trump phase ---
        is_last_round = (num_cards * num_players == NUM_CARDS_TOTAL)
        if not is_last_round:
            print("\nEnter the trump card that was revealed from the deck.")
            print("  Examples: R8  W1  J1")
            trump_card = prompt_card("Trump card: ")

            # Make sure trump card isn't already dealt
            if trump_card in used_cards:
                print(f"  WARNING: {card_name(trump_card)} is already in a hand, "
                      f"proceeding anyway.")
            state.apply_action(trump_card)

            if state._phase == Phase.CHOOSE_TRUMP:
                if ai_player == 0:  # dealer chooses
                    print("\nWizard revealed! AI is dealer and must choose trump color.")
                    action_probs = policy.action_probabilities(state, ai_player)
                    print(format_probs(action_probs))
                    best = max(action_probs, key=action_probs.get)
                    color_idx = best - TRUMP_CHOICE_OFFSET
                    print(f"\n>> AI recommends: {COLOR_NAMES[color_idx]}")
                    chosen = prompt_color("Confirm color choice (Y/R/G/B): ")
                    state.apply_action(TRUMP_CHOICE_OFFSET + chosen)
                else:
                    print("\nWizard revealed! Dealer chooses trump color.")
                    chosen = prompt_color("Dealer chose trump color (Y/R/G/B): ")
                    state.apply_action(TRUMP_CHOICE_OFFSET + chosen)

        trump_disp = (COLOR_NAMES[state._trump_color]
                      if state._trump_color != NO_TRUMP else "None")
        print(f"\nTrump color: {trump_disp}")

        # --- Bid phase ---
        print(f"\nBid phase ({num_players} players, starting left of dealer):")
        for bid_idx in range(num_players):
            bidder = (1 + bid_idx) % num_players
            if bidder == ai_player:
                action_probs = policy.action_probabilities(state, ai_player)
                print(f"\n  Player {bidder} (AI) bid recommendation:")
                print(format_probs(action_probs))
                best = max(action_probs, key=action_probs.get)
                print(f"\n  >> AI recommends: Bid {best - BID_OFFSET}")
                bid_val = prompt_int(
                    f"  Enter AI's actual bid (0-{num_cards}): ", 0, num_cards
                )
            else:
                bid_val = prompt_int(f"  Player {bidder} bid (0-{num_cards}): ",
                                     0, num_cards)
            state.apply_action(BID_OFFSET + bid_val)

        bids_display = ", ".join(f"P{p}: {b}" for p, b in state._bids)
        print(f"\nBids: {bids_display}")

        # --- Play phase ---
        print(f"\nPlay phase ({num_cards} trick(s)):")
        for trick_num in range(num_cards):
            print(f"\n--- Trick {trick_num + 1}/{num_cards} ---")
            for play_idx in range(num_players):
                player = (state._trick_leader + play_idx) % num_players
                if player == ai_player:
                    action_probs = policy.action_probabilities(state, ai_player)
                    legal = state.legal_actions()
                    legal_probs = {a: action_probs.get(a, 0) for a in legal}
                    total = sum(legal_probs.values())
                    if total > 0:
                        legal_probs = {a: p / total for a, p in legal_probs.items()}
                    hand_str = " ".join(card_name(c) for c in sorted(state._hands[ai_player]))
                    print(f"\n  Player {player} (AI) - hand: {hand_str}")
                    print(format_probs(legal_probs))
                    best = max(legal_probs, key=legal_probs.get)
                    print(f"\n  >> AI recommends: Play {card_name(best)}")
                    card = prompt_card("  Enter AI's actual play: ")
                else:
                    card = prompt_card(f"  Player {player} plays: ")

                    # Opponent cards: swap into their simulated hand if needed
                    if card not in state._hands[player]:
                        if state._hands[player]:
                            old_card = next(iter(state._hands[player]))
                            state._hands[player].discard(old_card)
                        state._hands[player].add(card)

                state.apply_action(card)

            print(f"  Trick winner: Player {state._trick_leader}")
            print(f"  Tricks won: {state._tricks_won}")

        # --- Results ---
        print("\n" + "=" * 40)
        print("  ROUND RESULTS")
        print("=" * 40)
        scores = state.returns()
        for p in range(num_players):
            bid = dict(state._bids).get(p, "?")
            marker = " (AI)" if p == ai_player else ""
            print(f"  Player {p}{marker}: bid {bid}, "
                  f"won {state._tricks_won[p]}, score {scores[p]:+.0f}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Play Wizard with trained AI")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory containing trained checkpoints")
    parser.add_argument("--num-players", type=int, default=3,
                        help="Number of players (3-6)")
    args = parser.parse_args()
    run_session(args.checkpoint_dir, args.num_players)


if __name__ == "__main__":
    main()
