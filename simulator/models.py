from dataclasses import dataclass, field

@dataclass
class Team:
    id: str
    name: str
    abbreviation: str
    group: str


@dataclass
class MatchResult:
    match_id: str
    home_team_id: str
    away_team_id: str
    home_goals: int
    away_goals: int
    group: str
    is_completed: bool = True


@dataclass
class MatchFixture:
    match_id: str
    home_team_id: str
    away_team_id: str
    group: str
    utc_date: str
    prob_home: float = 1 / 3
    prob_draw: float = 1 / 3
    prob_away: float = 1 / 3
    # Per-fixture Poisson goal rates derived from the betting markets.
    # None until odds are loaded; simulate.py falls back to solve_lambdas() when absent.
    lambda_home: float | None = None
    lambda_away: float | None = None
    # True when lambda_home/away came from a real market (API or manually entered).
    # False when falling back to the default λ_total split.
    has_market_odds: bool = False


@dataclass
class GroupStanding:
    team_id: str
    group: str
    played: int
    wins: int
    draws: int
    losses: int
    gf: int
    ga: int
    points: int

    @property
    def gd(self) -> int:
        return self.gf - self.ga


@dataclass
class SimResult:
    n_simulations: int
    teams: dict  # team_id -> Team
    groups: dict  # group_letter -> list[team_id]
    # team_id -> {1: count, 2: count, 3: count, 4: count}
    group_finish_counts: dict
    # team_id -> count reaching R32
    r32_counts: dict
    # group_letter -> count having 3rd-place team among best-8
    third_qualified_counts: dict
    # (group_letter, slot) -> {opponent_group: count}  for Annexe C matches
    annexe_c_opponent_counts: dict
    # match_id -> (avg_home_goals, avg_away_goals) across all simulation runs.
    # Fixed/manual scores appear as exact floats; simulated as means.
    fixture_avg_goals: dict = field(default_factory=dict)
