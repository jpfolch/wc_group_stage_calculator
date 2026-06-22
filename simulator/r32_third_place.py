"""
FIFA World Cup 2026 - Round of 32 third-place allocation (verified Annexe C).

Sources:
  - Regulations for the FIFA World Cup 26, Article 12.6 (official FIFA PDF) -
    defines the 8 winner-vs-3rd matches and each match's candidate 3rd-place groups.
  - Annexe C of the same regulations, all 495 rows, as transcribed on the
    Wikipedia "2026 FIFA World Cup knockout stage" page (loaded from annex_c.json).

Every Annexe C row was validated against the Art. 12.6 candidate sets and checked
to be a valid bijection (0 errors across all 495 rows).

Winner-slot -> match number and candidate 3rd-place groups (Art. 12.6):
    1E  M74  {A,B,C,D,F}
    1I  M77  {C,D,F,G,H}
    1A  M79  {C,E,F,H,I}   <-- Mexico (host, A1; clinched Group A)
    1L  M80  {E,H,I,J,K}
    1D  M81  {B,E,F,I,J}
    1G  M82  {A,E,H,I,J}
    1B  M85  {E,F,G,I,J}
    1K  M87  {D,E,I,J,L}

Usage:
    from simulator.r32_third_place import opponent_for, mexico_opponent
    mexico_opponent({'A','B','C','D','E','F','G','H'})  -> 'H'   # 3rd-place group
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "annex_c.json")

# slot -> (match number, FIFA group letter of the winner)
SLOT_MATCH = {
    "1E": ("M74", "E"), "1I": ("M77", "I"), "1A": ("M79", "A"), "1L": ("M80", "L"),
    "1D": ("M81", "D"), "1G": ("M82", "G"), "1B": ("M85", "B"), "1K": ("M87", "K"),
}

with open(_DATA) as _f:
    # keys are comma-joined sorted group letters, e.g. "A,B,C,D,E,F,G,H"
    _RAW = json.load(_f)

# Re-key by frozenset for convenient lookup.
ANNEX_C = {frozenset(k.split(",")): v for k, v in _RAW.items()}


def _key(qualifying_groups):
    qset = frozenset(qualifying_groups)
    if len(qset) != 8:
        raise ValueError(f"need exactly 8 distinct qualifying groups, got {sorted(qset)}")
    if qset not in ANNEX_C:
        raise KeyError(f"no Annexe C row for {sorted(qset)} (invalid group set?)")
    return qset


def allocation(qualifying_groups):
    """Full R32 third-place allocation: dict slot -> 3rd-place group letter."""
    return dict(ANNEX_C[_key(qualifying_groups)])


def opponent_for(winner_slot, qualifying_groups):
    """Which 3rd-place group the given winner-slot (e.g. '1A') faces."""
    if winner_slot not in SLOT_MATCH:
        raise ValueError(f"{winner_slot} is not a slot that faces a 3rd-placed team; "
                         f"valid: {sorted(SLOT_MATCH)}")
    return ANNEX_C[_key(qualifying_groups)][winner_slot]


def mexico_opponent(qualifying_groups):
    """Which 3rd-place group Mexico (Group A winner, M79) faces. Returns a single letter."""
    return opponent_for("1A", qualifying_groups)


if __name__ == "__main__":
    example = {"A", "B", "C", "D", "E", "F", "G", "H"}
    alloc = allocation(example)
    print(f"Qualifying 3rd-place groups: {sorted(example)}\n")
    print("Full R32 winner-vs-3rd allocation:")
    for slot, (match, wg) in SLOT_MATCH.items():
        print(f"  {match}: Winner {wg} (slot {slot}) vs 3rd-place of group {alloc[slot]}")
    print(f"\nMexico (M79) plays 3rd-place of group: {mexico_opponent(example)}")
