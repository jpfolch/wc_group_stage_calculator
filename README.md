# solve_sweepstake

WC 2026 group stage Monte Carlo simulator with live ESPN data, optional odds from The Odds API, and sweepstake reporting.

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
uv sync
cp .env.example .env   # add your ODDS_API_KEY (optional)
```

If you see `ModuleNotFoundError: No module named 'simulator'`, recreate the environment:

```bash
rm -rf .venv
uv sync
```

On macOS with the project in `~/Documents`, Python 3.12+ may skip hidden `.pth` files inside `.venv`, breaking editable installs. If problems persist after re-syncing, either run `chflags -R nohidden .venv` or use a `venv/` directory instead: `UV_PROJECT_ENVIRONMENT=venv uv sync`.

## Usage

Run the simulator:

```bash
uv run wc-sim
```

Common flags:

| Flag | Description |
|------|-------------|
| `--sims N` | Number of Monte Carlo runs (default from `config/settings.json`) |
| `--seed N` | RNG seed for reproducibility |
| `--no-odds` | Skip The Odds API; use uniform 1/3 match probabilities |
| `--fallback` | Use `config/standings.json` instead of ESPN API |
| `--no-sweepstake` | Skip the sweepstake participant report |

Example smoke test without API keys:

```bash
uv run wc-sim --no-odds --fallback --sims 100
```

Preview sweepstake team assignments:

```bash
uv run python scripts/sweepstake_choice.py
```

## Configuration

- `config/settings.json` — simulation count, date range, Poisson parameters
- `config/bracket.json` — R32 fixed pairings
- `config/standings.json` — fallback standings (see `config/standings_template.json`)
- `config/sweepstake.json` — optional cached sweepstake assignments
