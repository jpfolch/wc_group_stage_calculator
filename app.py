"""WC 2026 R32 Matchup Predictor — Streamlit frontend."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="WC 2026 R32 Predictor",
    page_icon="⚽",
    layout="wide",
)

# ---- Config ----
_ROOT = Path(__file__).parent
_SETTINGS = json.loads((_ROOT / "config" / "settings.json").read_text())
_POISSON = _SETTINGS.get("poisson_lambdas")
_START = str(_SETTINGS.get("group_stage_start", "20260611"))
_END = str(_SETTINGS.get("group_stage_end", "20260627"))
_CACHE_PATH = _ROOT / "config" / "data_cache.json"

# ---- Session state ----
for _k, _v in {
    "loaded": False,
    "team_registry": {},
    "group_standings": {},
    "completed": [],
    "fixtures": [],
    "sim_result": None,
    "matchup_probs": None,
    "last_updated": "",
    "odds_note": "",
    "cache_loaded_at": "",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---- Helpers ----
def _team_name(team_registry: dict, tid: str) -> str:
    t = team_registry.get(tid)
    return t.name if t else tid


def _best_in_slot(slot: str, result) -> tuple:
    """(team_id, %) for the most likely team to fill a slot like '1A' or '2H'."""
    pos = int(slot[0])
    group = slot[1].upper()
    tids = result.groups.get(group, [])
    if not tids:
        return None, 0.0
    best = max(tids, key=lambda t: result.group_finish_counts.get(t, {}).get(pos, 0))
    pct = result.group_finish_counts.get(best, {}).get(pos, 0) / result.n_simulations * 100
    return best, pct


def _best_third_in_group(group: str, result) -> tuple:
    """(team_id, %) for the most likely 3rd-place finisher in a group."""
    tids = result.groups.get(group, [])
    if not tids:
        return None, 0.0
    best = max(tids, key=lambda t: result.group_finish_counts.get(t, {}).get(3, 0))
    pct = result.group_finish_counts.get(best, {}).get(3, 0) / result.n_simulations * 100
    return best, pct


def _compute_group_lineups(result) -> dict[str, dict[int, str]]:
    """Greedy dedup: assign each team in a group to exactly one position.

    For each group, position 1 gets the team with the highest P(1st); position 2 gets
    the team with the highest P(2nd) from the remaining teams; etc.  This prevents the
    same team appearing in two bracket slots (e.g. Spain at both 1E and 2E).

    Returns {group_letter: {1: team_id, 2: team_id, 3: team_id, 4: team_id}}
    """
    lineups: dict[str, dict[int, str]] = {}
    for group, tids in result.groups.items():
        remaining = set(tids)
        assignment: dict[int, str] = {}
        for pos in range(1, 5):
            if not remaining:
                break
            _pos = pos  # local alias avoids lambda-capture gotcha
            best = max(remaining,
                       key=lambda t: result.group_finish_counts.get(t, {}).get(_pos, 0))
            assignment[pos] = best
            remaining.discard(best)
        lineups[group] = assignment
    return lineups


def _load_bracket_topology():
    """Parse world_cup_2026_bracket.json into ordered pair lists for the bracket table.

    Returns (r32_order, r16_pairs, qf_pairs, sf_pairs, final_id, r32_slots) where:
    - r32_order  : list of R32 match IDs in bracket-display order (left→right, top→bottom)
    - *_pairs    : list of (match_id, feeder_home, feeder_away) in that display order
    - final_id   : match ID string of the Final (e.g. "M104")
    - r32_slots  : dict match_id → (slot1, slot2) e.g. ("2A","2B") or ("1E",None) for Annexe C
    """
    wc = json.loads((_ROOT / "config" / "world_cup_2026_bracket.json").read_text())

    def _pm(s):
        return f"M{s.split()[-1]}"

    def _parse_slot(s):
        p = s.split()
        if p[0] == "Winner" and p[1] == "Group":
            return f"1{p[2]}"
        if p[0] == "Runner-up" and p[1] == "Group":
            return f"2{p[2]}"
        return None  # "Best 3rd place …" → Annexe C

    # Feed map: match_id → (feeder_home, feeder_away)
    _feed: dict = {}
    for rnd in ("round_of_16", "quarter_finals", "semi_finals", "final"):
        for m in wc["rounds"][rnd]:
            mid = f"M{m['match']}"
            if "Winner Match" in m["home"] and "Winner Match" in m["away"]:
                _feed[mid] = (_pm(m["home"]), _pm(m["away"]))

    _r32_ids = {f"M{m['match']}" for m in wc["rounds"]["round_of_32"]}
    _r16_ids = {f"M{m['match']}" for m in wc["rounds"]["round_of_16"]}
    _qf_ids  = {f"M{m['match']}" for m in wc["rounds"]["quarter_finals"]}
    _sf_ids  = {f"M{m['match']}" for m in wc["rounds"]["semi_finals"]}
    final_id = f"M{wc['rounds']['final'][0]['match']}"

    def _ordered(mid, target):
        if mid in target:
            return [mid]
        if mid not in _feed:
            return []
        h, a = _feed[mid]
        return _ordered(h, target) + _ordered(a, target)

    r32_order = _ordered(final_id, _r32_ids)
    r16_pairs = [(m,) + _feed[m] for m in _ordered(final_id, _r16_ids)]
    qf_pairs  = [(m,) + _feed[m] for m in _ordered(final_id, _qf_ids)]
    sf_pairs  = [(m,) + _feed[m] for m in _ordered(final_id, _sf_ids)]

    r32_slots = {
        f"M{m['match']}": (_parse_slot(m["home"]), _parse_slot(m["away"]))
        for m in wc["rounds"]["round_of_32"]
    }
    return r32_order, r16_pairs, qf_pairs, sf_pairs, final_id, r32_slots


_BRACKET_TOPOLOGY = _load_bracket_topology()


def _fetch_data() -> None:
    import datetime
    import os
    from dotenv import load_dotenv
    from simulator import espn_client, odds_client
    from simulator.paths import ENV_FILE

    load_dotenv(ENV_FILE)

    tr, gs, comp, fix = espn_client.fetch_all(start_date=_START, end_date=_END)

    # Enrich fixtures with per-match Poisson λ values from the odds API
    odds_note = ""
    api_key = os.environ.get("ODDS_API_KEY", "")
    if api_key:
        try:
            raw_odds = odds_client.fetch_odds(api_key)
            fix = odds_client.merge_into_fixtures(fix, raw_odds, tr)
            n_with = sum(1 for f in fix if f.lambda_home is not None)
            n_with_totals = sum(
                1 for f in fix
                if f.lambda_home is not None
                and abs(f.lambda_home + (f.lambda_away or 0) - 2.6) > 0.05
            )
            odds_note = (
                f"odds for {n_with}/{len(fix)} fixtures "
                f"({n_with_totals} with totals market)"
            )
        except Exception as exc:
            odds_note = f"odds fetch failed: {exc}"
    else:
        odds_note = "no ODDS_API_KEY — using default λ"

    # Pre-fill manual odds from existing cache for fixtures still lacking API odds
    _try_prefill_manual_odds_from_cache(fix)

    st.session_state.update(
        loaded=True,
        team_registry=tr,
        group_standings=gs,
        completed=comp,
        fixtures=fix,
        odds_note=odds_note,
        sim_result=None,
        matchup_probs=None,
        last_updated=datetime.datetime.now().strftime("%H:%M:%S"),
        cache_loaded_at="",
    )

    # Auto-save to cache so the data is available offline next time
    try:
        from simulator import cache as _cache
        _cache.save(tr, gs, comp, fix, _CACHE_PATH)
    except Exception:
        pass  # cache write failure is non-fatal


def _load_from_cache() -> None:
    from simulator import cache as _cache
    tr, gs, comp, fix, cached_at = _cache.load(_CACHE_PATH)
    st.session_state.update(
        loaded=True,
        team_registry=tr,
        group_standings=gs,
        completed=comp,
        fixtures=fix,
        sim_result=None,
        matchup_probs=None,
        last_updated="",
        cache_loaded_at=cached_at,
        odds_note=f"{sum(1 for f in fix if f.has_market_odds)}/{len(fix)} fixtures with market odds (cached)",
    )


def _apply_custom_odds(fixtures: list) -> list:
    """Return a copy of fixtures with any manually entered decimal odds applied."""
    import copy
    from simulator.match_model import solve_lambdas, DEFAULT_LAMBDA_TOTAL

    updated = []
    for fix in fixtures:
        oh = st.session_state.get(f"oh_{fix.match_id}")
        od_val = st.session_state.get(f"od_{fix.match_id}")
        oa = st.session_state.get(f"oa_{fix.match_id}")
        if oh and od_val and oa and oh > 1.0 and od_val > 1.0 and oa > 1.0:
            fix = copy.copy(fix)
            raw_h, raw_d, raw_a = 1 / oh, 1 / od_val, 1 / oa
            total = raw_h + raw_d + raw_a
            fix.prob_home = raw_h / total
            fix.prob_draw = raw_d / total
            fix.prob_away = raw_a / total
            lh, la = solve_lambdas(fix.prob_home, DEFAULT_LAMBDA_TOTAL)
            fix.lambda_home = lh
            fix.lambda_away = la
            fix.has_market_odds = True
        updated.append(fix)
    return updated


def _try_prefill_manual_odds_from_cache(fresh_fixtures: list) -> None:
    """For fixtures that didn't get API odds, pre-fill oh_/od_/oa_ session-state keys
    from any manually-saved odds in the cache so the user can re-deploy unchanged odds
    without re-typing.  Only fills keys that are currently None (won't overwrite user input)."""
    if not _CACHE_PATH.exists():
        return
    try:
        from simulator import cache as _cache
        _, _, _, cached_fix, _ = _cache.load(_CACHE_PATH)
    except Exception:
        return
    cached_by_id = {f.match_id: f for f in cached_fix}
    prefilled = 0
    for fix in fresh_fixtures:
        if fix.has_market_odds:
            continue  # already has live API odds
        cf = cached_by_id.get(fix.match_id)
        if cf is None or not cf.has_market_odds:
            continue  # no cached manual odds for this fixture
        ph, pd_, pa = cf.prob_home, cf.prob_draw, cf.prob_away
        if ph <= 0 or pd_ <= 0 or pa <= 0:
            continue
        k_h = f"oh_{fix.match_id}"
        k_d = f"od_{fix.match_id}"
        k_a = f"oa_{fix.match_id}"
        # Only pre-fill if the user hasn't typed anything yet this session
        if (st.session_state.get(k_h) is None
                and st.session_state.get(k_d) is None
                and st.session_state.get(k_a) is None):
            st.session_state[k_h] = round(1.0 / ph, 2)
            st.session_state[k_d] = round(1.0 / pd_, 2)
            st.session_state[k_a] = round(1.0 / pa, 2)
            prefilled += 1
    if prefilled:
        st.session_state["_prefill_note"] = (
            f"✏️ Pre-filled odds from cache for {prefilled} fixture(s) — "
            "check the Missing Odds section below to review."
        )


def _collect_manual_results(fixtures: list) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for fix in fixtures:
        hg = st.session_state.get(f"h_{fix.match_id}")
        ag = st.session_state.get(f"a_{fix.match_id}")
        if hg is not None and ag is not None:
            out[fix.match_id] = (int(hg), int(ag))
    return out


def _run_sim(manual_results: dict, n_sims: int) -> None:
    from simulator import simulate
    from simulator.bracket import load_bracket, resolve_r32_matchups
    fixtures = _apply_custom_odds(st.session_state.fixtures)
    result = simulate.run(
        team_registry=st.session_state.team_registry,
        group_standings=st.session_state.group_standings,
        completed=st.session_state.completed,
        fixtures=fixtures,
        n_simulations=n_sims,
        poisson_params=_POISSON,
        manual_results=manual_results,
    )
    bracket = load_bracket()
    st.session_state.sim_result = result
    st.session_state.matchup_probs = resolve_r32_matchups(bracket, result)


# ---- Sidebar ----
with st.sidebar:
    st.title("⚽ WC 2026")
    if st.button("🔄  Update Data from ESPN", use_container_width=True, type="primary"):
        with st.spinner("Fetching live data…"):
            try:
                _fetch_data()
                teams = st.session_state.team_registry
                completed = st.session_state.completed
                fixtures = st.session_state.fixtures
                st.success(
                    f"Loaded {len(teams)} teams · "
                    f"{len(completed)} completed · "
                    f"{len(fixtures)} remaining"
                )
            except Exception as exc:
                st.error(f"Fetch failed: {exc}")

    if st.session_state.last_updated:
        st.caption(f"Last updated: {st.session_state.last_updated}")
    if st.session_state.cache_loaded_at:
        st.caption(f"📂 From cache: {st.session_state.cache_loaded_at}")
    if st.session_state.get("odds_note"):
        st.caption(f"📈 {st.session_state.odds_note}")
    if st.session_state.get("_prefill_note"):
        st.info(st.session_state["_prefill_note"])
        st.session_state["_prefill_note"] = ""

    # Offline / cache button
    from simulator import cache as _cache_mod
    _cache_ts = _cache_mod.timestamp(_CACHE_PATH)
    if _cache_ts:
        _cache_label = f"📂  Load from Cache\n{_cache_ts}"
        if st.button(_cache_label, use_container_width=True):
            try:
                _load_from_cache()
                st.success("Loaded from cache.")
            except Exception as exc:
                st.error(f"Cache load failed: {exc}")
    else:
        st.button("📂  Load from Cache", use_container_width=True, disabled=True,
                  help="No cache yet — fetch live data first.")

    st.divider()
    n_sims = st.select_slider(
        "Simulations",
        options=[1_000, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
        format_func=lambda v: f"{v:,}",
    )


# ---- Main ----
st.title("WC 2026 — R32 Matchup Predictor")
st.caption(
    "Set known upcoming scores, then run the Monte Carlo simulation to see who Mexico plays."
)

if not st.session_state.loaded:
    st.info("Press **🔄 Update Data from ESPN** in the sidebar to load live standings and fixtures.")
    st.stop()

teams = st.session_state.team_registry
group_standings = st.session_state.group_standings
fixtures: list = st.session_state.fixtures

# ---- Current Standings ----
with st.expander("📊 Current Group Standings", expanded=False):
    gl_list = sorted(group_standings.keys())
    n_cols = min(len(gl_list), 4)
    cols = st.columns(n_cols)
    for idx, gl in enumerate(gl_list):
        with cols[idx % n_cols]:
            entries = sorted(
                group_standings[gl],
                key=lambda s: (-s.points, -(s.gf - s.ga), -s.gf),
            )
            df_stand = pd.DataFrame([
                {
                    "#": i + 1,
                    "Team": _team_name(teams, s.team_id),
                    "Pts": s.points,
                    "GD": s.gd,
                    "GF": s.gf,
                    "P": s.played,
                }
                for i, s in enumerate(entries)
            ])
            st.markdown(f"**Group {gl}**")
            st.dataframe(df_stand, hide_index=True, use_container_width=True, height=175)

# ---- Remaining Fixtures ----
st.subheader("Remaining Fixtures")
if not fixtures:
    st.success("All group stage matches are complete — run the simulation to see standings.")
else:
    st.caption(
        "Enter both goals to fix a score; leave either blank to let it be simulated.  "
        "Odds badge: ✅ market odds  ✏️ custom odds entered  ⚠️ default λ (no odds)"
    )
    by_group: dict[str, list] = {}
    for fix in fixtures:
        by_group.setdefault(fix.group, []).append(fix)

    for gl in sorted(by_group):
        with st.expander(f"Group {gl}  —  {len(by_group[gl])} remaining", expanded=True):
            for fix in by_group[gl]:
                home = _team_name(teams, fix.home_team_id)
                away = _team_name(teams, fix.away_team_id)

                # Determine odds badge from current widget state
                oh = st.session_state.get(f"oh_{fix.match_id}")
                od_v = st.session_state.get(f"od_{fix.match_id}")
                oa = st.session_state.get(f"oa_{fix.match_id}")
                has_custom = bool(oh and od_v and oa and oh > 1.0 and od_v > 1.0 and oa > 1.0)
                if fix.has_market_odds:
                    badge = "✅"
                elif has_custom:
                    badge = "✏️"
                else:
                    badge = "⚠️"

                c1, c2, c3, c4, c5, c6 = st.columns([3.5, 1, 0.4, 1, 3.5, 0.6])
                with c1:
                    st.markdown(
                        f"<p style='text-align:right;padding-top:6px'><b>{home}</b></p>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.number_input(
                        "Home goals", min_value=0, max_value=20, value=None,
                        key=f"h_{fix.match_id}", label_visibility="collapsed",
                    )
                with c3:
                    st.markdown(
                        "<p style='text-align:center;padding-top:6px;color:#888'>–</p>",
                        unsafe_allow_html=True,
                    )
                with c4:
                    st.number_input(
                        "Away goals", min_value=0, max_value=20, value=None,
                        key=f"a_{fix.match_id}", label_visibility="collapsed",
                    )
                with c5:
                    st.markdown(
                        f"<p style='padding-top:6px'><b>{away}</b></p>",
                        unsafe_allow_html=True,
                    )
                with c6:
                    st.markdown(
                        f"<p style='text-align:center;padding-top:6px;font-size:1.1em'>{badge}</p>",
                        unsafe_allow_html=True,
                    )

# ---- Missing Odds section ----
missing_odds_fixtures = [f for f in fixtures if not f.has_market_odds]
if missing_odds_fixtures:
    with st.expander(
        f"⚠️  Enter custom odds for {len(missing_odds_fixtures)} fixture(s) without market data",
        expanded=False,
    ):
        st.caption(
            "Enter decimal odds (e.g. 1.80 / 3.50 / 4.20). "
            "Leave all blank to use the default λ = 2.6 symmetric split.  "
            "Values marked ✏️ were pre-filled from the cache — review and adjust as needed."
        )
        miss_by_group: dict[str, list] = {}
        for fix in missing_odds_fixtures:
            miss_by_group.setdefault(fix.group, []).append(fix)

        for gl in sorted(miss_by_group):
            st.markdown(f"**Group {gl}**")
            for fix in miss_by_group[gl]:
                home = _team_name(teams, fix.home_team_id)
                away = _team_name(teams, fix.away_team_id)
                st.markdown(
                    f"<span style='color:#555'>{home} vs {away}</span>",
                    unsafe_allow_html=True,
                )
                oc1, oc2, oc3 = st.columns(3)
                with oc1:
                    st.number_input(
                        f"Home ({home})", min_value=1.01, max_value=100.0,
                        value=None, step=0.05, format="%.2f",
                        key=f"oh_{fix.match_id}", label_visibility="visible",
                    )
                with oc2:
                    st.number_input(
                        "Draw", min_value=1.01, max_value=100.0,
                        value=None, step=0.05, format="%.2f",
                        key=f"od_{fix.match_id}", label_visibility="visible",
                    )
                with oc3:
                    st.number_input(
                        f"Away ({away})", min_value=1.01, max_value=100.0,
                        value=None, step=0.05, format="%.2f",
                        key=f"oa_{fix.match_id}", label_visibility="visible",
                    )

        st.divider()
        if st.button(
            "💾  Add to cache",
            key="save_odds_cache",
            help="Bake the entered odds into the cache file so they load automatically next time",
        ):
            from simulator import cache as _cache_so
            _upd = _apply_custom_odds(st.session_state.fixtures)
            _n_new = (
                sum(1 for f in _upd if f.has_market_odds)
                - sum(1 for f in st.session_state.fixtures if f.has_market_odds)
            )
            st.session_state.fixtures = _upd
            _cache_so.save(
                st.session_state.team_registry,
                st.session_state.group_standings,
                st.session_state.completed,
                _upd,
                _CACHE_PATH,
            )
            st.success(
                f"Saved odds for {_n_new} fixture(s) to cache. "
                "They will be pre-loaded next time you click **Load from Cache**."
            )

# ---- Run simulation ----
manual = _collect_manual_results(fixtures)
n_fixed = len(manual)
n_free = len(fixtures) - n_fixed
btn_label = f"▶  Run Simulation  ({n_fixed} fixed · {n_free} simulated)"

if st.button(btn_label, type="primary", use_container_width=True):
    with st.spinner(f"Running {n_sims:,} Monte Carlo simulations…"):
        try:
            _run_sim(manual, n_sims)
        except Exception as exc:
            st.error(f"Simulation error: {exc}")
            st.stop()

if st.session_state.sim_result is None:
    st.stop()

# ---- Results ----
result = st.session_state.sim_result
matchup_probs = st.session_state.matchup_probs
n = result.n_simulations

# Dedup group lineups: each team assigned to exactly one position per group
_group_lineups = _compute_group_lineups(result)

# Locate Mexico
mexico_id: str | None = next(
    (
        tid for tid, t in result.teams.items()
        if "mexico" in t.name.lower() or t.abbreviation.upper() == "MEX"
    ),
    None,
)

tab_mex, tab_groups, tab_third, tab_annexe, tab_bracket, tab_scores, tab_how = st.tabs(
    ["🇲🇽  Mexico's R32", "📊  Group Finish", "🏅  3rd Place", "📋  Annexe C", "🏆  Predicted Bracket", "⚽  Avg Scores", "ℹ️  How It Works"]
)

# ---- Mexico tab ----
with tab_mex:
    st.header(f"Mexico's R32 Opponent Probabilities  ({n:,} sims)")
    st.caption(
        "Mexico play match **M79** as Group A winners. Their opponent is the best 3rd-place "
        "team from groups C/E/F/H/I, determined by Annexe C."
    )

    if mexico_id is None:
        st.warning("Could not identify Mexico in simulation results.")
    else:
        opps = matchup_probs.get(mexico_id, {})
        if not opps:
            st.info("No R32 matchup data available for Mexico.")
        else:
            sorted_opps = [(oid, p) for oid, p in sorted(opps.items(), key=lambda x: -x[1]) if p > 0.001]
            max_p = sorted_opps[0][1] if sorted_opps else 1.0

            for rank, (opp_id, prob) in enumerate(sorted_opps[:15]):
                opp = result.teams.get(opp_id)
                if opp is None:
                    continue
                pct = prob * 100
                bar_w = prob / max_p * 100
                color = "#1b5e20" if rank == 0 else ("#388e3c" if rank < 3 else "#78909c")
                weight = "bold" if rank < 3 else "normal"
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:12px;margin:4px 0'>"
                    f"<div style='min-width:230px;font-weight:{weight}'>"
                    f"{opp.name}"
                    f"<span style='color:#aaa;font-size:0.85em'> (Grp {opp.group})</span>"
                    f"</div>"
                    f"<div style='flex:1;background:#e8e8e8;border-radius:4px;height:20px'>"
                    f"<div style='background:{color};width:{bar_w:.1f}%;height:100%;border-radius:4px'></div>"
                    f"</div>"
                    f"<div style='min-width:48px;text-align:right;font-weight:{weight}'>"
                    f"{pct:.1f}%"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ---- Group finish tab ----
with tab_groups:
    st.header(f"Group Finish Probabilities  ({n:,} simulations)")
    rows = []
    for gl in sorted(result.groups):
        for tid in result.groups[gl]:
            fc = result.group_finish_counts.get(tid, {})
            team = result.teams.get(tid)
            rows.append({
                "Group": gl,
                "Team": team.name if team else tid,
                "1st %": round(fc.get(1, 0) / n * 100, 1),
                "2nd %": round(fc.get(2, 0) / n * 100, 1),
                "3rd %": round(fc.get(3, 0) / n * 100, 1),
                "Out %": round(fc.get(4, 0) / n * 100, 1),
                "R32 %": round(result.r32_counts.get(tid, 0) / n * 100, 1),
            })
    df_groups = pd.DataFrame(rows)
    st.dataframe(
        df_groups.style.background_gradient(subset=["R32 %"], cmap="Greens"),
        hide_index=True,
        use_container_width=True,
        height=700,
    )

# ---- 3rd place tab ----
with tab_third:
    st.header("3rd-Place Qualification Probability")
    rows = []
    for gl in sorted(result.groups):
        q_pct = result.third_qualified_counts.get(gl, 0) / n * 100
        thirds = [
            (
                (result.teams.get(tid).name if result.teams.get(tid) else tid),
                result.group_finish_counts.get(tid, {}).get(3, 0) / n * 100,
            )
            for tid in result.groups[gl]
        ]
        thirds = sorted([(nm, p) for nm, p in thirds if p > 0.5], key=lambda x: -x[1])
        candidates = ", ".join(f"{nm} {p:.0f}%" for nm, p in thirds[:3])
        rows.append({
            "Group": gl,
            "3rd Qualifies %": round(q_pct, 1),
            "Most Likely 3rd-Place Teams": candidates,
        })
    df_third = pd.DataFrame(rows)
    st.dataframe(
        df_third.style.background_gradient(subset=["3rd Qualifies %"], cmap="Greens"),
        hide_index=True,
        use_container_width=True,
    )

# ---- Annexe C tab ----
with tab_annexe:
    from simulator.r32_third_place import SLOT_MATCH
    st.header("Annexe C — Winner vs 3rd-Place Matchup Probabilities")
    st.caption(
        "The 8 group winners who face a best 3rd-place team in the R32. "
        "**Most Likely Winner** shows the team most likely to win that group (✓ = virtually guaranteed). "
        "Each opponent group also shows the most likely 3rd-place finisher from that group."
    )
    rows = []
    _winner_guaranteed_flags: dict[str, bool] = {}
    for slot, (match_num, winner_group) in sorted(SLOT_MATCH.items(), key=lambda x: x[1][0]):
        opp_counts = result.annexe_c_opponent_counts.get(slot, {})
        total = sum(opp_counts.values())
        if total == 0:
            continue
        parts = sorted(opp_counts.items(), key=lambda x: -x[1])

        # Most likely group winner
        winner_tid, winner_pct = _best_in_slot(f"1{winner_group}", result)
        winner_name = result.teams[winner_tid].name if winner_tid else "?"
        guaranteed = winner_pct >= 99.5
        _winner_guaranteed_flags[match_num] = guaranteed
        winner_str = f"{winner_name} ({winner_pct:.0f}%)" + (" ✓" if guaranteed else "")

        # Opponent groups with most likely 3rd-place team per group
        parts_desc = []
        for g, c in parts:
            if c / total > 0.02:
                third_tid, _ = _best_third_in_group(g, result)
                third_name = result.teams[third_tid].name if third_tid else "?"
                parts_desc.append(f"Grp {g}: {c / total * 100:.0f}%  ({third_name})")
        desc = "  ·  ".join(parts_desc)

        rows.append({
            "Match": match_num,
            "Slot": f"1{winner_group}",
            "Most Likely Winner": winner_str,
            "Possible 3rd-Place Opponent (% of sims)": desc,
        })
    df_annexe = pd.DataFrame(rows)

    def _style_annexe_df(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for idx, row in df.iterrows():
            if _winner_guaranteed_flags.get(row["Match"], False):
                styles.at[idx, "Most Likely Winner"] = (
                    "background-color:#e8f5e9; color:#1b5e20; font-weight:bold"
                )
        return styles

    st.dataframe(
        df_annexe.style.apply(_style_annexe_df, axis=None),
        hide_index=True,
        use_container_width=True,
    )

# ---- Predicted Bracket tab ----
with tab_bracket:
    from simulator.r32_third_place import SLOT_MATCH as _SM_B

    st.header(f"Full Tournament Bracket — Predicted Lineup  ({n:,} simulations)")
    st.caption(
        "R32 slots: most likely team from the group-stage simulation. "
        "R16 onward: the higher-probability R32 team is projected forward. "
        "Mexico 🇲🇽 highlighted."
    )

    # ── Bracket topology (from world_cup_2026_bracket.json) ─────────────
    _R32_ORDER, _R16_PAIRS, _QF_PAIRS, _SF_PAIRS, _final_id, _bkt_r32_slots = _BRACKET_TOPOLOGY

    # ── Annexe C slot map: match_id -> (slot_key, winner_group) ─────────
    _bkt_annexe = {_mn: (_sk, _wg) for _sk, (_mn, _wg) in _SM_B.items()}
    _mex_name_b = result.teams[mexico_id].name if mexico_id else ""

    def _bkt_esc(s): return s.replace("&", "&amp;").replace("<", "&lt;")
    def _bkt_mex(name): return bool(_mex_name_b and _mex_name_b.lower() in name.lower())

    def _slot_info(mid, slot_num):
        """Return {"name":…, "slot":…, "pct":…} for the given R32 match + slot position.
        Uses _group_lineups for dedup: no team appears in two slots of the same group."""
        _s1, _s2 = _bkt_r32_slots[mid]
        _sl = _s1 if slot_num == 1 else _s2
        if _sl is not None:
            _pos = int(_sl[0]); _grp = _sl[1].upper()
            _t = _group_lineups.get(_grp, {}).get(_pos)
            _p = (result.group_finish_counts.get(_t, {}).get(_pos, 0) / result.n_simulations * 100
                  if _t else 0.0)
            return {"name": result.teams[_t].name if _t and result.teams.get(_t) else "?",
                    "slot": _sl, "pct": _p}
        # Annexe C slot (None = 3rd-place side)
        _sk, _wg = _bkt_annexe[mid]
        if slot_num == 1:
            _t = _group_lineups.get(_wg, {}).get(1)
            _p = (result.group_finish_counts.get(_t, {}).get(1, 0) / result.n_simulations * 100
                  if _t else 0.0)
            return {"name": result.teams[_t].name if _t and result.teams.get(_t) else "?",
                    "slot": f"1{_wg}", "pct": _p}
        _oc = result.annexe_c_opponent_counts.get(_sk, {})
        if _oc:
            _ot = sum(_oc.values())
            _bg = max(_oc, key=_oc.get)
            _t2 = _group_lineups.get(_bg, {}).get(3)
            return {"name": result.teams[_t2].name if _t2 and result.teams.get(_t2) else "?",
                    "slot": f"3rd Grp {_bg}", "pct": _oc[_bg] / _ot * 100}
        return {"name": "?", "slot": "3rd", "pct": 0.0}

    # ── 32 team-row list ─────────────────────────────────────────────────
    _bkt_rows = []
    for _mid in _R32_ORDER:
        _bkt_rows.append({**_slot_info(_mid, 1), "match": _mid})
        _bkt_rows.append({**_slot_info(_mid, 2), "match": _mid})

    # ── R32 predicted winner per match (higher slot pct) ─────────────────
    _r32_w = {_bkt_rows[i]["match"]:
              (_bkt_rows[i] if _bkt_rows[i]["pct"] >= _bkt_rows[i+1]["pct"] else _bkt_rows[i+1])
              for i in range(0, 32, 2)}

    # ── Project forward through bracket ──────────────────────────────────
    def _project(pairs, src):
        t1 = {m: src.get(a, {"name":"?","pct":0}) for m,a,b in pairs}
        t2 = {m: src.get(b, {"name":"?","pct":0}) for m,a,b in pairs}
        w  = {m: (t1[m] if t1[m]["pct"] >= t2[m]["pct"] else t2[m]) for m,_,__ in pairs}
        return t1, t2, w

    _r16_t1, _r16_t2, _r16_w = _project(_R16_PAIRS, _r32_w)
    _qf_t1,  _qf_t2,  _qf_w  = _project(_QF_PAIRS,  _r16_w)
    _sf_t1,  _sf_t2,  _sf_w  = _project(_SF_PAIRS,   _qf_w)
    _sf1_id, _sf2_id = _SF_PAIRS[0][0], _SF_PAIRS[1][0]
    _fin_t1 = _sf_w.get(_sf1_id, {"name":"?","pct":0})
    _fin_t2 = _sf_w.get(_sf2_id, {"name":"?","pct":0})
    _fin_w  = _fin_t1 if _fin_t1["pct"] >= _fin_t2["pct"] else _fin_t2

    # ── HTML bracket table (32 rows, 5 columns with rowspan) ─────────────
    _brows = []
    for _ri in range(32):
        _rd   = _bkt_rows[_ri]
        _is_m = _bkt_mex(_rd["name"])
        _is_last_in_match = (_ri % 2 == 1)
        _is_r16_boundary  = (_ri % 4 == 3) and _ri < 31
        _bdr_b = ("2px solid #bbb" if _is_r16_boundary
                  else ("1px solid #eee" if _is_last_in_match else "none"))

        _r32_td = (
            f'<td style="padding:3px 6px; font-size:0.78em; vertical-align:middle;'
            f' max-width:155px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'
            f' border-bottom:{_bdr_b};'
            f' {"background:#edf7ed; border-left:3px solid #1b5e20;" if _is_m else "border-left:2px solid #e8e8e8;"}'
            f'">'
            f'{"🇲🇽 " if _is_m else ""}<b>{_bkt_esc(_rd["name"])}</b>'
            f' <span style="color:#999; font-size:0.82em;">{_rd["slot"]} {_rd["pct"]:.0f}%</span>'
            f'</td>'
        )
        _row = f'<tr style="height:22px;">{_r32_td}'

        if _ri % 4 == 0:
            _r16m, _r32a, _r32b = _R16_PAIRS[_ri // 4]
            _a, _b = _r16_t1[_r16m], _r16_t2[_r16m]
            _mm = _bkt_mex(_a["name"]) or _bkt_mex(_b["name"])
            _row += (
                f'<td rowspan="4" style="vertical-align:middle; text-align:center;'
                f' padding:4px 6px; font-size:0.78em; min-width:105px;'
                f' {"background:#edf7ed; border:1.5px solid #1b5e20;" if _mm else "background:#f8f8f8; border:1px solid #e0e0e0;"}'
                f' border-radius:4px;">'
                f'<div style="color:#aaa; font-size:0.72em;">{_r16m}</div>'
                f'<div style="font-weight:bold;">{_bkt_esc(_a["name"])}</div>'
                f'<div style="color:#bbb; font-size:0.72em;">vs</div>'
                f'<div style="font-weight:bold;">{_bkt_esc(_b["name"])}</div>'
                f'</td>'
            )

        if _ri % 8 == 0:
            _qfm, _r16a, _r16b = _QF_PAIRS[_ri // 8]
            _a, _b = _qf_t1[_qfm], _qf_t2[_qfm]
            _mm = _bkt_mex(_a["name"]) or _bkt_mex(_b["name"])
            _row += (
                f'<td rowspan="8" style="vertical-align:middle; text-align:center;'
                f' padding:4px 6px; font-size:0.78em; min-width:105px;'
                f' {"background:#edf7ed; border:1.5px solid #1b5e20;" if _mm else "background:#f8f8f8; border:1px solid #e0e0e0;"}'
                f' border-radius:4px;">'
                f'<div style="color:#aaa; font-size:0.72em;">{_qfm}</div>'
                f'<div style="font-weight:bold;">{_bkt_esc(_a["name"])}</div>'
                f'<div style="color:#bbb; font-size:0.72em;">vs</div>'
                f'<div style="font-weight:bold;">{_bkt_esc(_b["name"])}</div>'
                f'</td>'
            )

        if _ri % 16 == 0:
            _sfm, _qfa, _qfb = _SF_PAIRS[_ri // 16]
            _a, _b = _sf_t1[_sfm], _sf_t2[_sfm]
            _mm = _bkt_mex(_a["name"]) or _bkt_mex(_b["name"])
            _row += (
                f'<td rowspan="16" style="vertical-align:middle; text-align:center;'
                f' padding:4px 6px; font-size:0.78em; min-width:105px;'
                f' {"background:#edf7ed; border:1.5px solid #1b5e20;" if _mm else "background:#f8f8f8; border:1px solid #e0e0e0;"}'
                f' border-radius:4px;">'
                f'<div style="color:#aaa; font-size:0.72em;">{_sfm}</div>'
                f'<div style="font-weight:bold;">{_bkt_esc(_a["name"])}</div>'
                f'<div style="color:#bbb; font-size:0.72em;">vs</div>'
                f'<div style="font-weight:bold;">{_bkt_esc(_b["name"])}</div>'
                f'</td>'
            )

        if _ri == 0:
            _row += (
                f'<td rowspan="32" style="vertical-align:middle; text-align:center;'
                f' padding:10px; font-size:0.8em; min-width:115px;'
                f' background:#fffde7; border:2px solid #ffd700; border-radius:6px;">'
                f'<div style="color:#aaa; font-size:0.72em; margin-bottom:4px;">{_final_id} · Final</div>'
                f'<div style="font-weight:bold; font-size:0.9em;">{_bkt_esc(_fin_t1["name"])}</div>'
                f'<div style="color:#bbb; margin:3px 0; font-size:0.76em;">vs</div>'
                f'<div style="font-weight:bold; font-size:0.9em;">{_bkt_esc(_fin_t2["name"])}</div>'
                f'<div style="margin-top:10px; padding-top:6px; border-top:1px solid #e8d000;">'
                f'<div style="color:#888; font-size:0.72em;">Predicted winner</div>'
                f'<div style="font-weight:bold; font-size:0.95em; color:#1b5e20;">🏆 {_bkt_esc(_fin_w["name"])}</div>'
                f'</div></td>'
            )

        _row += "</tr>"
        _brows.append(_row)

    _bkt_html = (
        '<div style="overflow-x:auto; padding:6px 0;">'
        '<table style="border-collapse:separate; border-spacing:3px 1px;'
        ' font-family:sans-serif; width:100%;">'
        "<thead><tr style=\"font-size:0.78em; color:#555;\">"
        "<th style=\"text-align:left; padding:4px 8px; border-bottom:2px solid #bbb;\">R32</th>"
        "<th style=\"text-align:center; padding:4px 8px; border-bottom:2px solid #bbb;\">R16</th>"
        "<th style=\"text-align:center; padding:4px 8px; border-bottom:2px solid #bbb;\">QF</th>"
        "<th style=\"text-align:center; padding:4px 8px; border-bottom:2px solid #bbb;\">SF</th>"
        "<th style=\"text-align:center; padding:4px 8px; border-bottom:2px solid #bbb;\">Final 🏆</th>"
        "</tr></thead><tbody>"
        + "".join(_brows)
        + "</tbody></table></div>"
    )
    st.markdown(_bkt_html, unsafe_allow_html=True)


# ---- Avg Scores tab ----
with tab_scores:
    st.header(f"Simulated Match Scores  ({n:,} simulations)")
    st.caption(
        "**Final** = already played · **Fixed** = score you set · **Simulated** = mean over all runs"
    )

    completed_list = st.session_state.completed
    all_fixtures = st.session_state.fixtures
    avg_goals = result.fixture_avg_goals

    # Build a lookup: match_id -> MatchFixture for remaining fixtures
    fixture_by_id = {f.match_id: f for f in all_fixtures}

    # Collect all matches into rows
    rows: list[dict] = []

    for m in completed_list:
        rows.append({
            "group": m.group,
            "home": _team_name(teams, m.home_team_id),
            "score": f"{m.home_goals} – {m.away_goals}",
            "away": _team_name(teams, m.away_team_id),
            "status": "Final",
            "_sort": 0,
        })

    for fix in all_fixtures:
        if fix.match_id in manual and fix.match_id in avg_goals:
            hg, ag = avg_goals[fix.match_id]
            score_str = f"{int(hg)} – {int(ag)}"
            status = "Fixed"
        elif fix.match_id in avg_goals:
            hg, ag = avg_goals[fix.match_id]
            score_str = f"{hg:.2f} – {ag:.2f}"
            status = "Simulated"
        else:
            continue
        rows.append({
            "group": fix.group,
            "home": _team_name(teams, fix.home_team_id),
            "score": score_str,
            "away": _team_name(teams, fix.away_team_id),
            "status": status,
            "_sort": 1 if status == "Fixed" else 2,
        })

    if not rows:
        st.info("No match data to display.")
    else:
        # Display one table per group
        all_groups = sorted({r["group"] for r in rows})
        for gl in all_groups:
            group_rows = sorted(
                [r for r in rows if r["group"] == gl],
                key=lambda r: r["_sort"],
            )
            df = pd.DataFrame(
                [
                    {
                        "Home": r["home"],
                        "Score": r["score"],
                        "Away": r["away"],
                        "Status": r["status"],
                    }
                    for r in group_rows
                ]
            )

            def _row_style(row):
                if row["Status"] == "Final":
                    return ["color: #888"] * len(row)
                if row["Status"] == "Fixed":
                    return ["font-weight: bold; color: #1565c0"] * len(row)
                return ["color: #2e7d32"] * len(row)

            st.markdown(f"**Group {gl}**")
            st.dataframe(
                df.style.apply(_row_style, axis=1),
                hide_index=True,
                use_container_width=True,
                height=35 * len(df) + 38,
            )
            st.write("")


# ---- How It Works tab ----
with tab_how:
    st.header("How It Works")

    st.subheader("1 · H2H market → win probabilities")
    st.markdown("""
Bookmakers publish **decimal odds** for each outcome (home win / draw / away win).
For example, odds of **2.50** on a home win imply a probability of 1 ÷ 2.50 = 40 %.

Because bookmakers build in a profit margin (the *overround*), the three raw implied
probabilities sum to more than 100 %. We remove that margin by **normalising**:

```
raw_home  = 1 / odds_home        # e.g. 1/2.50 = 0.400
raw_draw  = 1 / odds_draw        # e.g. 1/3.40 = 0.294
raw_away  = 1 / odds_away        # e.g. 1/3.00 = 0.333
                                 # sum = 1.027  (2.7 % overround)
prob_home = raw_home / 1.027     # 38.9 %
prob_draw = raw_draw / 1.027     # 28.6 %
prob_away = raw_away / 1.027     # 32.4 %
```

Odds from **all available bookmakers** are averaged before conversion, so no single
bookmaker dominates.
""")

    st.subheader("2 · Totals market → expected goals")
    st.markdown("""
The **over/under (totals) market** prices whether the total goals in a match will
exceed a line — usually **2.5** for football. We use this to anchor the expected
total goals λ_total for each match.

Given a 2.5-goal line with prices P_over and P_under, we:

1. **Normalise** to get the fair P(over): strip the bookmaker margin the same way
   as step 1.
2. **Invert the Poisson CDF**: find λ_total such that
   P(Pois(λ_total) ≥ 3) = P(over). This is solved numerically with bisection.

```
# Example: line = 2.5, fair P(over) = 0.52
# Find λ such that 1 − CDF(2, λ) = 0.52
# → CDF(2, λ) = 0.48
# Bisection → λ_total ≈ 2.75 goals expected
```

If no totals market is available, λ_total falls back to **2.6** (WC group-stage
historical average).
""")

    st.subheader("3 · Solving for per-team goal rates")
    st.markdown("""
Given λ_total (how many total goals the market expects) and P(home win) (from the
H2H market), we solve for the individual rates **λ_home** and **λ_away**:

**Constraint 1** — total goals:  `λ_home + λ_away = λ_total`

**Constraint 2** — win probability:
  `P(Pois(λ_home) > Pois(λ_away)) = P(home win)`

Because constraint 1 lets us write `λ_away = λ_total − λ_home`, this reduces to
a one-dimensional bisection on λ_home. The home-win probability is a strictly
monotone function of λ_home, so the bisection always converges.

```
# Example: P(home win) = 0.45, λ_total = 2.75
# Bisect λ_home in (0, 2.75) until
#   P(Pois(λ_h) > Pois(2.75 − λ_h)) ≈ 0.45
# → λ_home ≈ 1.55,  λ_away ≈ 1.20
```

A match between roughly equal teams (P(home win) ≈ 33 %) gets a symmetric split
`λ_home = λ_away = λ_total / 2`.  A heavy favourite gets a much higher λ, producing
more goals on average and bigger realistic winning margins.
""")

    st.subheader("4 · Monte Carlo simulation")
    st.markdown("""
The simulation is run **N times** (default 10 000) independently. In each run:

1. For every remaining fixture, **two independent Poisson draws** are made:
   ```
   home_goals ~ Pois(λ_home)
   away_goals ~ Pois(λ_away)
   ```
   The result (win/draw/loss) follows naturally from the scores — no ad-hoc
   outcome sampling or goal-forcing needed.  The H2H win probability is reproduced
   *implicitly* because that is exactly what the λ values were calibrated to.

2. **Group standings** (points, GF, GA, H2H records) are updated after each match.

3. Groups are **ranked** using the FIFA 2026 tiebreaker rules (see below).

4. The best 8 third-place teams are selected and **Annexe C** is looked up to
   determine each winner-vs-3rd matchup.

After all N runs, every outcome is expressed as a **percentage of simulations**.
""")

    st.subheader("5 · FIFA 2026 group-stage tiebreaker rules")
    st.markdown("""
When two or more teams finish level on points, FIFA 2026 applies these criteria
**in order**:

| Priority | Criterion |
|---|---|
| 1 | Points in **all** group matches |
| 2 | Points in **head-to-head** (H2H) matches among the tied teams |
| 3 | Goal difference in H2H matches |
| 4 | Goals scored in H2H matches |
| 5 | Goal difference in **all** group matches |
| 6 | Goals scored in all group matches |
| 7 | Drawing of lots |

> **Key change from 2022:** H2H record (criteria 2–4) now comes *before* overall
> goal difference (5–6). This is why Mexico's first-place finish is already
> confirmed — their H2H record guarantees it regardless of their final match result.

When **you fix a score** in this app, that result is applied deterministically
before the Monte Carlo runs: the team standings and H2H records are updated, and
only the remaining unfixed matches are simulated.
""")

    st.subheader("6 · Round-of-32 matchup probabilities")
    st.markdown("""
Mexico (**Group A winner, slot 1A**) plays in **match M79**. Their opponent is
the best 3rd-place team from groups **C, E, F, H or I**, determined by Annexe C
of the FIFA regulations.

- **Fixed bracket matches**: probability = fraction of sims where each team
  finished in the required position.
- **Annexe C matches**: each simulation independently determines which eight
  3rd-place groups qualify and looks up the exact Annexe C allocation, so all
  uncertainty in the 3rd-place standings is propagated correctly.
""")

# ---- Share Summary section ----
st.divider()
with st.expander("📋  Share Summary — copy-paste text for any tab"):
    st.caption(
        "Select all text in a box below (Ctrl+A / Cmd+A inside the box) and copy to share."
    )

    from simulator.r32_third_place import SLOT_MATCH as _SM_SHARE

    _share_tabs = st.tabs(
        ["🇲🇽 Mexico", "📊 Groups", "🏅 3rd Place", "📋 Annexe C", "🏆 Bracket"]
    )

    # ── Mexico summary ───────────────────────────────────────────────────
    with _share_tabs[0]:
        _lines = [f"Mexico R32 Opponent Probabilities ({n:,} sims)\n"]
        if mexico_id:
            _opps = matchup_probs.get(mexico_id, {})
            for _rank, (_oid, _p) in enumerate(
                sorted(_opps.items(), key=lambda x: -x[1])[:12], 1
            ):
                if _p < 0.001:
                    break
                _ot = result.teams.get(_oid)
                _lines.append(
                    f"  {_rank:2d}. {(_ot.name if _ot else _oid):<22s} {_p*100:5.1f}%"
                    + (f"  (Grp {_ot.group})" if _ot else "")
                )
        st.text_area("Mexico R32", "\n".join(_lines), height=280, label_visibility="collapsed")

    # ── Group Finish summary ─────────────────────────────────────────────
    with _share_tabs[1]:
        _lines = [f"Group Finish Probabilities ({n:,} sims)\n"]
        for _gl in sorted(result.groups):
            _lines.append(f"Group {_gl}:")
            _rows_g = []
            for _tid in result.groups[_gl]:
                _fc = result.group_finish_counts.get(_tid, {})
                _tm = result.teams.get(_tid)
                _rows_g.append((_tm.name if _tm else _tid,
                                _fc.get(1,0)/n*100, _fc.get(2,0)/n*100,
                                _fc.get(3,0)/n*100, result.r32_counts.get(_tid,0)/n*100))
            _rows_g.sort(key=lambda x: -x[4])
            for _nm, _p1, _p2, _p3, _pr in _rows_g:
                _lines.append(f"  {_nm:<22s}  1st:{_p1:5.1f}%  2nd:{_p2:5.1f}%  3rd:{_p3:5.1f}%  R32:{_pr:5.1f}%")
            _lines.append("")
        st.text_area("Groups", "\n".join(_lines), height=320, label_visibility="collapsed")

    # ── 3rd Place summary ────────────────────────────────────────────────
    with _share_tabs[2]:
        _lines = [f"3rd-Place Qualification Probabilities ({n:,} sims)\n"]
        for _gl in sorted(result.groups):
            _qp = result.third_qualified_counts.get(_gl, 0) / n * 100
            _thirds_sh = sorted(
                [(result.teams[_t].name if result.teams.get(_t) else _t,
                  result.group_finish_counts.get(_t,{}).get(3,0)/n*100)
                 for _t in result.groups[_gl]],
                key=lambda x: -x[1]
            )
            _cands = ", ".join(f"{_nm} {_p:.0f}%" for _nm, _p in _thirds_sh[:2] if _p > 1)
            _lines.append(f"Grp {_gl}: qualifies {_qp:5.1f}%   candidates: {_cands}")
        st.text_area("3rd Place", "\n".join(_lines), height=260, label_visibility="collapsed")

    # ── Annexe C summary ─────────────────────────────────────────────────
    with _share_tabs[3]:
        _lines = [f"Annexe C — Winner vs 3rd-Place Matchup ({n:,} sims)\n"]
        for _slot_s, (_mn_s, _wg_s) in sorted(_SM_SHARE.items(), key=lambda x: x[1][0]):
            _oc_s = result.annexe_c_opponent_counts.get(_slot_s, {})
            _tot_s = sum(_oc_s.values())
            if _tot_s == 0:
                continue
            _wt, _wp = _best_in_slot(f"1{_wg_s}", result)
            _wn = result.teams[_wt].name if _wt else "?"
            _guar = " (guaranteed)" if _wp >= 99.5 else f" ({_wp:.0f}%)"
            _lines.append(f"{_mn_s}  Slot 1{_wg_s}: {_wn}{_guar}")
            for _g_s, _c_s in sorted(_oc_s.items(), key=lambda x: -x[1]):
                if _c_s / _tot_s < 0.02:
                    continue
                _t3, _ = _best_third_in_group(_g_s, result)
                _n3 = result.teams[_t3].name if _t3 else "?"
                _lines.append(f"      Grp {_g_s}: {_c_s/_tot_s*100:.0f}%  ({_n3})")
            _lines.append("")
        st.text_area("Annexe C", "\n".join(_lines), height=320, label_visibility="collapsed")

    # ── Bracket summary ──────────────────────────────────────────────────
    with _share_tabs[4]:
        _lines = [f"Predicted Bracket ({n:,} sims — R16+ projections)\n"]
        _R32_ORD_S, _R16_S, _QF_S, _SF_S, _fin_id_s, _r32sl_s = _BRACKET_TOPOLOGY
        _bkt_ax_s = {_mn: (_sk, _wg) for _sk, (_mn, _wg) in _SM_SHARE.items()}

        def _si_s(mid, sn):
            _s1, _s2 = _r32sl_s[mid]
            _sl = _s1 if sn == 1 else _s2
            if _sl is not None:
                _pos = int(_sl[0]); _grp = _sl[1].upper()
                _t = _group_lineups.get(_grp, {}).get(_pos)
                _p = (result.group_finish_counts.get(_t, {}).get(_pos, 0) / result.n_simulations * 100
                      if _t else 0.0)
                return result.teams[_t].name if _t and result.teams.get(_t) else "?", _sl, _p
            _sk2, _wg2 = _bkt_ax_s[mid]
            if sn == 1:
                _t = _group_lineups.get(_wg2, {}).get(1)
                _p = (result.group_finish_counts.get(_t, {}).get(1, 0) / result.n_simulations * 100
                      if _t else 0.0)
                return result.teams[_t].name if _t and result.teams.get(_t) else "?", f"1{_wg2}", _p
            _oc2 = result.annexe_c_opponent_counts.get(_sk2, {})
            if _oc2:
                _ot2 = sum(_oc2.values())
                _bg2 = max(_oc2, key=_oc2.get)
                _t2 = _group_lineups.get(_bg2, {}).get(3)
                return (result.teams[_t2].name if _t2 and result.teams.get(_t2) else "?",
                        f"3rd Grp {_bg2}", _oc2[_bg2]/_ot2*100)
            return "?", "3rd", 0.0

        _r32w_s = {}
        for _mid_s in _R32_ORD_S:
            _n1s, _sl1s, _p1s = _si_s(_mid_s, 1)
            _n2s, _sl2s, _p2s = _si_s(_mid_s, 2)
            _lines.append(f"R32  {_mid_s}: {_n1s} ({_sl1s} {_p1s:.0f}%)  vs  {_n2s} ({_sl2s} {_p2s:.0f}%)")
            _r32w_s[_mid_s] = (_n1s, _p1s) if _p1s >= _p2s else (_n2s, _p2s)
        _lines.append("")

        def _proj_s(pairs, src):
            res = {}
            for _m, _a, _b in pairs:
                _wa, _pa = src[_a]
                _wb, _pb = src[_b]
                _rnd = "R16" if len(pairs)==8 else ("QF" if len(pairs)==4 else "SF")
                _lines.append(f"{_rnd}  {_m}: {_wa} ({_pa:.0f}%)  vs  {_wb} ({_pb:.0f}%)")
                res[_m] = (_wa, _pa) if _pa >= _pb else (_wb, _pb)
            _lines.append("")
            return res

        _r16w_s = _proj_s(_R16_S, _r32w_s)
        _qfw_s  = _proj_s(_QF_S,  _r16w_s)
        _sfw_s  = _proj_s(_SF_S,  _qfw_s)
        _sf1s, _sf2s = _SF_S[0][0], _SF_S[1][0]
        _ft1s, _fp1s = _sfw_s.get(_sf1s, ("?", 0))
        _ft2s, _fp2s = _sfw_s.get(_sf2s, ("?", 0))
        _lines.append(f"Final {_fin_id_s}: {_ft1s} ({_fp1s:.0f}%)  vs  {_ft2s} ({_fp2s:.0f}%)")
        _fw_s = _ft1s if _fp1s >= _fp2s else _ft2s
        _lines.append(f"\nPredicted winner: {_fw_s}")

        st.text_area("Bracket", "\n".join(_lines), height=360, label_visibility="collapsed")
