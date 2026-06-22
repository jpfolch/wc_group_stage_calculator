"""WC 2026 Group Stage Simulator — draws live odds and runs Monte Carlo simulation."""
import argparse
import json
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from simulator import espn_client, odds_client, report, simulate
from simulator.bracket import has_unverified, load_bracket
from simulator.paths import CONFIG_DIR, ENV_FILE


def main():
    load_dotenv(ENV_FILE)

    parser = argparse.ArgumentParser(
        description="WC 2026 Group Stage Simulator — uses The Odds API + ESPN live data"
    )
    parser.add_argument("--sims", type=int, default=None, help="Number of simulations (default from settings.json)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--no-odds", action="store_true", help="Use uniform 1/3 probabilities (no Odds API call)")
    parser.add_argument("--fallback", action="store_true", help="Use config/standings.json instead of ESPN API")
    parser.add_argument("--odds-key", default=None, help="The Odds API key (or set ODDS_API_KEY env var)")
    parser.add_argument("--no-sweepstake", action="store_true", help="Skip the sweepstake participant report")
    args = parser.parse_args()

    console = Console()

    settings_path = CONFIG_DIR / "settings.json"
    with open(settings_path) as f:
        settings = json.load(f)

    n_sims = args.sims or settings.get("n_simulations", 10_000)
    sport_key = settings.get("sport_key", "soccer_fifa_world_cup")
    regions = settings.get("regions", "uk")
    poisson_params = settings.get("poisson_lambdas")
    start_date = settings.get("group_stage_start", "20260611")
    end_date = settings.get("group_stage_end", "20260627")

    bracket = load_bracket()
    if has_unverified(bracket):
        console.print(
            Panel(
                "[bold yellow]WARNING:[/bold yellow] Some R32 bracket pairings in "
                "[bold]config/bracket.json[/bold] are marked [bold yellow]UNVERIFIED[/bold yellow].\n"
                "Please cross-check against the official FIFA 2026 bracket and set "
                "[bold]\"verified\": true[/bold] when confirmed.",
                title="Bracket Warning",
                border_style="yellow",
            )
        )

    console.print("[cyan]Fetching current WC 2026 standings and fixtures from ESPN…[/cyan]")
    team_registry, group_standings, completed, fixtures = espn_client.fetch_all(
        start_date=start_date,
        end_date=end_date,
        force_fallback=args.fallback,
    )

    if not group_standings:
        console.print(
            "[bold red]No group standings found. "
            "Try --fallback and provide config/standings.json.[/bold red]"
        )
        sys.exit(1)

    console.print(
        f"[green]Loaded {len(team_registry)} teams across {len(group_standings)} groups. "
        f"{len(completed)} completed matches, {len(fixtures)} remaining.[/green]"
    )

    if not args.no_odds:
        api_key = args.odds_key or os.environ.get("ODDS_API_KEY", "")
        if api_key:
            console.print(f"[cyan]Fetching odds from The Odds API ({sport_key})…[/cyan]")
            try:
                raw_odds = odds_client.fetch_odds(api_key, sport_key=sport_key, regions=regions)
                fixtures = odds_client.merge_into_fixtures(fixtures, raw_odds, team_registry)
                console.print(f"[green]Loaded odds for {len(raw_odds)} matches.[/green]")
            except Exception as exc:
                console.print(f"[yellow]Odds API failed: {exc}. Using uniform 1/3 probabilities.[/yellow]")
        else:
            console.print(
                "[yellow]No ODDS_API_KEY set. Using uniform 1/3 probabilities. "
                "Set ODDS_API_KEY in .env or pass --odds-key.[/yellow]"
            )

    console.print(f"[cyan]Running {n_sims:,} simulations…[/cyan]")
    result = simulate.run(
        team_registry=team_registry,
        group_standings=group_standings,
        completed=completed,
        fixtures=fixtures,
        n_simulations=n_sims,
        poisson_params=poisson_params,
        seed=args.seed,
    )
    console.print("[green]Simulation complete.[/green]\n")

    report.print_live_standings(group_standings, team_registry, console)
    console.print()
    report.print_simulation_summary(result, console)
    console.print()
    report.print_third_place_summary(result, console)
    console.print()
    report.print_annexe_c_summary(result, console)

    if not args.no_sweepstake:
        try:
            assignments = _load_sweepstake_assignments()
            console.print()
            report.print_sweepstake_report(result, assignments, console)
        except Exception as exc:
            console.print(f"[yellow]Sweepstake report skipped: {exc}[/yellow]")


def _load_sweepstake_assignments() -> dict:
    cached = CONFIG_DIR / "sweepstake.json"
    if cached.exists():
        with open(cached) as f:
            return json.load(f)

    from scripts.sweepstake_choice import assign

    players = [
        "Nathan", "Alex", "Jose", "Sarah", "Kallum", "Francesca",
        "Linden", "Rory", "Vishal", "Andrew", "Matt", "Rasa",
    ]
    raw = assign(players, mode="snake")
    return {player: teams for player, teams in raw.items()}


if __name__ == "__main__":
    main()
