"""R32 bracket determination and probability helpers."""
import json

from simulator.models import SimResult, Team
from simulator.paths import CONFIG_DIR
from simulator.r32_third_place import SLOT_MATCH

_BRACKET_PATH = CONFIG_DIR / "bracket.json"

# Annexe C slot -> group letter of that slot's winner
_SLOT_TO_WINNER_GROUP = {slot: info[1] for slot, info in SLOT_MATCH.items()}


def load_bracket() -> dict:
    with open(_BRACKET_PATH) as f:
        return json.load(f)


def has_unverified(bracket: dict) -> bool:
    return any(not m.get("verified", True) for m in bracket.get("r32_fixed", []))


def resolve_r32_matchups(
    bracket: dict,
    sim_result: SimResult,
) -> dict[str, dict[str, float]]:
    """Return {team_id: {opponent_team_id: probability}} for R32 matchups.

    For fixed bracket matches, opponent is deterministic given each simulation run's
    group finish. We compute this from the group_finish_counts.

    For Annexe C matches, we use the annexe_c_opponent_counts already computed.
    """
    n = sim_result.n_simulations
    teams = sim_result.teams
    groups = sim_result.groups
    finish_counts = sim_result.group_finish_counts

    # Build helper: (group, position) -> list of team_ids (by finish count order)
    # For easy slot resolution.

    # For Annexe C matches, we know:
    #   Slot 1A → Winner of Group A  (whoever finished 1st in group A)
    #   vs       → 3rd-place team from a specific group (from annexe_c_opponent_counts)
    # We want: for each team, what's their probability of facing each possible opponent?

    matchup_probs: dict[str, dict[str, float]] = {tid: {} for tid in teams}

    # --- Annexe C matches ---
    for slot, winner_group in _SLOT_TO_WINNER_GROUP.items():
        # Winner of winner_group faces a 3rd-place team
        opp_group_counts = sim_result.annexe_c_opponent_counts.get(slot, {})

        # For each team in winner_group, if they finished 1st:
        winner_group_teams = groups.get(winner_group, [])
        for tid in winner_group_teams:
            times_first = finish_counts.get(tid, {}).get(1, 0)
            if times_first == 0:
                continue
            # In all simulations where tid finished 1st, who was the 3rd-place opponent?
            # We don't have the exact joint distribution, so we approximate:
            # P(tid wins group) * P(opp_group = X | winner_group has a 1st-place team)
            # This is a good approximation when P(tid wins) is independent of Annexe C outcome
            for opp_group, count in opp_group_counts.items():
                # 3rd-place teams from opp_group
                opp_group_teams = groups.get(opp_group, [])
                if not opp_group_teams:
                    continue
                # Probability this is the 3rd-place team in opp_group
                for opp_tid in opp_group_teams:
                    times_third = finish_counts.get(opp_tid, {}).get(3, 0)
                    # Joint probability (approximated)
                    p = (times_first / n) * (count / n) * (times_third / max(
                        sum(finish_counts.get(t, {}).get(3, 0) for t in opp_group_teams), 1
                    ))
                    if p > 0:
                        matchup_probs[tid][opp_tid] = matchup_probs[tid].get(opp_tid, 0) + p
                        matchup_probs[opp_tid][tid] = matchup_probs[opp_tid].get(tid, 0) + p

    # --- Fixed bracket matches ---
    for match in bracket.get("r32_fixed", []):
        slot_1 = match["slot_1"]  # e.g. "1C" or "2F"
        slot_2 = match["slot_2"]

        def slot_to_teams(slot):
            pos = int(slot[0])  # 1 or 2
            group = slot[1].upper()
            return [
                (tid, finish_counts.get(tid, {}).get(pos, 0))
                for tid in groups.get(group, [])
            ]

        teams_1 = slot_to_teams(slot_1)
        teams_2 = slot_to_teams(slot_2)

        total_1 = sum(c for _, c in teams_1)
        total_2 = sum(c for _, c in teams_2)
        if total_1 == 0 or total_2 == 0:
            continue

        for tid1, c1 in teams_1:
            for tid2, c2 in teams_2:
                p = (c1 / n) * (c2 / n)
                if p > 0:
                    matchup_probs[tid1][tid2] = matchup_probs[tid1].get(tid2, 0) + p
                    matchup_probs[tid2][tid1] = matchup_probs[tid2].get(tid1, 0) + p

    return matchup_probs


def most_likely_r32_opponent(matchup_probs: dict, team_id: str) -> tuple[str, float]:
    """Return (most_likely_opponent_id, probability)."""
    opps = matchup_probs.get(team_id, {})
    if not opps:
        return ("", 0.0)
    best = max(opps.items(), key=lambda x: x[1])
    return best
