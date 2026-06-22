"""Derive per-match Poisson goal rates (λ_home, λ_away) from betting markets.

Given:
  - P(home win) from the H2H market
  - Optionally: over/under line + P(over) from the totals market

We solve for (λ_home, λ_away) such that:
  1. λ_home + λ_away = λ_total   (anchored to the totals market)
  2. P(Pois(λ_home) > Pois(λ_away)) ≈ P(home win)   (matches H2H market)

All goal scoring is modelled as independent Poisson processes.
No external dependencies beyond the standard library.
"""
from __future__ import annotations
import math

_MAX_GOALS = 12           # truncation for PMF sums — P(X > 12) < 0.1 % for λ ≤ 5
DEFAULT_LAMBDA_TOTAL = 2.6  # WC group-stage historical average


# ---- Poisson helpers --------------------------------------------------------

def _pmf(k: int, lam: float) -> float:
    """Poisson PMF: P(X = k) for X ~ Pois(lam)."""
    if k < 0 or lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def _cdf(k: int, lam: float) -> float:
    """Poisson CDF: P(X ≤ k)."""
    return sum(_pmf(j, lam) for j in range(k + 1))


def _p_home_wins(lh: float, la: float) -> float:
    """P(Pois(lh) > Pois(la)) — probability home team scores strictly more goals."""
    h = [_pmf(k, lh) for k in range(_MAX_GOALS + 1)]
    a = [_pmf(k, la) for k in range(_MAX_GOALS + 1)]
    return sum(
        h[g_h] * a[g_a]
        for g_a in range(_MAX_GOALS + 1)
        for g_h in range(g_a + 1, _MAX_GOALS + 1)
    )


# ---- Bisection --------------------------------------------------------------

def _bisect(f, lo: float, hi: float, tol: float = 1e-4, max_iter: int = 60) -> float:
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ---- Public API -------------------------------------------------------------

def lambda_total_from_over_under(line: float, prob_over: float) -> float:
    """Find λ_total such that P(Pois(λ) ≥ ⌈line⌉) = prob_over.

    Typical football markets: line = 2.5, so we solve P(goals ≥ 3) = prob_over.
    Uses bisection on the Poisson CDF.
    """
    k = math.ceil(line)  # goals needed for "over" to win (e.g. 3 for a 2.5 line)

    def _residual(lam: float) -> float:
        return (1.0 - _cdf(k - 1, lam)) - prob_over

    try:
        return _bisect(_residual, 0.01, 15.0)
    except Exception:
        return line  # fallback: use line value directly


def solve_lambdas(
    prob_home: float,
    lambda_total: float = DEFAULT_LAMBDA_TOTAL,
) -> tuple[float, float]:
    """Find (λ_home, λ_away) that reproduce the H2H win probability.

    Constraint: λ_home + λ_away = lambda_total (from totals market or default).
    Strategy:   bisect on λ_home so that P(Pois(λ_h) > Pois(λ_total - λ_h)) = prob_home.

    Edge cases:
      - prob_home ≈ 1/3  →  symmetric split (λ/2, λ/2)
      - prob_home outside the achievable Poisson range  →  clamped to boundary
    """
    if lambda_total <= 0.05:
        half = DEFAULT_LAMBDA_TOTAL / 2
        return half, half

    if abs(prob_home - 1 / 3) < 1e-3:
        half = lambda_total / 2
        return half, half

    eps = 0.02                         # keep both λ values above zero
    lo, hi = eps, lambda_total - eps

    def _residual(lh: float) -> float:
        return _p_home_wins(lh, lambda_total - lh) - prob_home

    f_lo, f_hi = _residual(lo), _residual(hi)

    if f_lo >= 0 and f_hi >= 0:        # prob_home below achievable min → clamp
        return lo, lambda_total - lo
    if f_lo <= 0 and f_hi <= 0:        # prob_home above achievable max → clamp
        return hi, lambda_total - hi

    lh = _bisect(_residual, lo, hi)
    return lh, lambda_total - lh
