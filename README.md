<div align="center">

# WC 2026 Forecaster

**Probabilistic match predictions for FIFA World Cup 2026 — locked before kickoff, never touched after.**

[![daily-fetch](https://github.com/VirajMishra1/worldcup-forecaster/actions/workflows/daily-fetch.yml/badge.svg)](https://github.com/VirajMishra1/worldcup-forecaster/actions/workflows/daily-fetch.yml) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

[Live Dashboard](https://virajmishra1.github.io/worldcup-forecaster/) · [Track Record](#live-track-record)

</div>

---

## What is this?

Most football predictions are gut-feel opinion dressed up as analysis. This is a proper statistical model: Dixon-Coles score model fit on 13,779 international matches from 2010–2024, with time-decay weighting, squad-value adjustments, and calibration tuned on a 5,500-match walk-forward backtest.

Before every World Cup game, it produces a full probability distribution over every possible scoreline (0-0, 1-0, 2-1, ...). From that distribution it derives win/draw/loss odds, over/under 2.5, BTTS, and top-3 most likely exact scores. All of that gets committed to GitHub 60 minutes before kickoff and is never edited. After the match, actual outcomes are compared and the track record updates automatically.

The point isn't to beat bookmakers. It's to build something that's actually calibrated — and to prove it on live data with a verifiable track record, not cherry-picked examples.

---

## WC 2026 — Winner Odds

10,000 Monte Carlo simulations · bracket-aware · updated daily after every result

| Team | Win probability |
|------|----------------|
| 🇦🇷 Argentina | 15.8% |
| 🇧🇷 Brazil | 12.2% |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 10.2% |
| 🇺🇸 United States | 10.2% |
| 🇩🇪 Germany | 9.3% |
| 🇫🇷 France | 7.7% |
| 🇪🇸 Spain | 5.9% |
| 🇧🇪 Belgium | 5.2% |
| 🇵🇹 Portugal | 4.8% |
| 🇲🇦 Morocco | 4.1% |
| 🇲🇽 Mexico | 3.9% |
| 🇨🇴 Colombia | 2.6% |

_32 completed WC 2026 results included in the refit · updated 2026-06-20_

---

<!-- TRACK_RECORD_START -->
## Live Track Record (8 matches)

| Metric | Value | Random baseline |
|--------|-------|-----------------|
| W/D/L accuracy | 62.5% | 33.3% |
| Log-loss | 0.8687 | 1.0986 |
| Brier score | 0.5136 | 0.6667 |

### Per-match predictions

| Date | Match | H% / D% / A% | Result | LL | ✓ |
|------|-------|--------------|--------|----|---|
| 2026-06-18 | Czech Republic vs South Africa | 56%/27%/17% | Draw (1-1) | 1.297 | ✗ |
| 2026-06-18 | Switzerland vs Bosnia and Herzegovina | 69%/20%/11% | Switzerland (4-1) | 0.369 | ✓ |
| 2026-06-18 | Canada vs Qatar | 44%/31%/26% | Canada (6-0) | 0.823 | ✓ |
| 2026-06-19 | Mexico vs South Korea | 39%/33%/28% | Mexico (1-0) | 0.935 | ✓ |
| 2026-06-19 | United States vs Australia | 33%/33%/34% | United States (2-0) | 1.101 | ✗ |
| 2026-06-19 | Scotland vs Morocco | 16%/29%/55% | Morocco (0-1) | 0.605 | ✓ |
| 2026-06-20 | Brazil vs Haiti | 95%/4%/1% | Brazil (3-0) | 0.053 | ✓ |
| 2026-06-20 | Turkey vs Paraguay | 56%/27%/17% | Paraguay (0-1) | 1.767 | ✗ |
<!-- TRACK_RECORD_END -->

Log-loss is the main metric here — it rewards calibration, not just picking winners. Lower is better. A random 1/3-1/3-1/3 guess scores 1.0986. The model is currently at 0.8687.

---

<!-- ODDS_COMPARISON_START -->
## Model vs Polymarket · market updated Jun 19 10:12 UTC

| Team | Model | Market | Edge |
|------|-------|--------|------|

_Edge = Model% − Market%. Positive = model thinks team is underpriced._

<!-- ODDS_COMPARISON_END -->

---

## How it works

### Scoring model

Every prediction starts from expected goals — one per team. The model learns these from historical results: each team has an attack rating and a defense rating, and those combine to give expected goals for any matchup. From expected goals you get a probability distribution over all possible scorelines, and from that distribution every market falls out — win/draw/loss, over/under, BTTS, exact score.

One thing worth knowing: standard Poisson models underestimate how often low-scoring games happen in football. Dixon-Coles adds a correction term (τ) that inflates the probability on 0-0, 1-0, 0-1, and 1-1 results to match what the data actually shows.

<details>
<summary><strong>The math</strong></summary>

Each team `i` has an attack parameter `α_i` and defense parameter `δ_i`. For a match between home team `i` and away team `j`:

```
λ_home = exp(α_i + δ_j + γ·is_home)
λ_away = exp(α_j + δ_i)

P(X=x, Y=y) = τ(x, y, λ_home, λ_away, ρ) · Poisson(x; λ_home) · Poisson(y; λ_away)
```

`is_home = 0` for all WC matches (neutral venues). The Dixon-Coles τ correction applies only when `x + y ≤ 1`, parameterised by `ρ ≈ −0.065` (fitted from data). Parameters are found by MLE with L-BFGS-B.

</details>

### Training data

13,779 international matches from 2010–2024:

- **Time decay** — exponential weighting with a 1.5-year half-life, so recent form matters more than historical results
- **Friendly downweight** — friendly matches carry 15% of the weight of competitive fixtures
- **WC tournament boosts** — WC 2022 matches weighted 3×, 2018 × 2.0, 2014 × 1.5, 2010 × 1.2 (same competition, strongest prior)
- **xG substitution** — for 63 WC 2022 matches, actual scorelines are replaced with StatsBomb expected goals, since xG is a cleaner signal of team quality than luck-adjusted results

The model is refitted daily as WC 2026 results come in (weighted 3×).

### Adjustments at prediction time

**Squad values** — Transfermarkt market values anchor predictions for teams with thin historical data. Haiti at €18M vs France at €1.2B shouldn't require hundreds of historical matches to get right. Applied conservatively (exponent 0.375, capped effect).

**Recent form** — last 5 competitive matches, weighted by opponent quality. Beating Argentina counts more than beating a bottom-ranked side. Clipped to ±15% on expected goals.

**Rest days** — small ±3% adjustment per day relative to a 4-day baseline.

**Calibration** — isotonic regression calibration, fitted on 5,518 backtest matches, adjusts the raw model outputs to better match empirical win rates.

### How predictions stay honest

Everything in `data/predictions.parquet` is append-only. A GitHub Action locks predictions 60 minutes before kickoff and commits them — there's no mechanism to overwrite a locked row. After the match, results are fetched and the track record updates. The git history is the audit trail.

---

## Model performance

Walk-forward backtest — refitted every 30 days, no lookahead, 5,518 matches (2018–2023):

| Metric | Model | Random baseline |
|--------|-------|-----------------|
| Log-loss | 0.8961 | 1.0986 |
| Brier score | 0.5265 | 0.6667 |
| W/D/L accuracy | 59.0% | 33.3% |

![Calibration](reports/calibration.png)

The calibration plot shows how often predicted probabilities match actual outcomes — when the model says 60%, the team wins roughly 60% of the time. A perfectly calibrated model sits on the diagonal.

![Log-loss curve](reports/log_loss_curve.png)

---

## Run locally

```bash
git clone https://github.com/VirajMishra1/worldcup-forecaster.git
cd worldcup-forecaster
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Get a free API key from [football-data.org](https://www.football-data.org/) and add it to `.env`:

```
FOOTBALL_DATA_API_KEY=your_key_here
```

Then:

```bash
python -m scripts.fetch_fixtures      # fetch WC schedule
python -m scripts.fetch_results       # fetch completed results
python -m scripts.refit_params        # fit the model
python -m scripts.predict_all_fixtures  # generate predictions
python -m scripts.simulate_tournament   # 10k tournament simulation

python -m cli.predict --home France --away Argentina  # single match
python -m cli.backtest --start 2018-01-01 --end 2024-12-31  # historical backtest
```

---

## Stack

- Python 3.11, numpy, scipy, pandas, pyarrow
- scikit-learn for isotonic calibration
- httpx for data fetching
- GitHub Actions for the full pipeline (zero infra cost)
- GitHub Pages for the dashboard

---

## References

- Dixon, M. and Coles, S. (1997). "Modelling Association Football Scores and Inefficiencies in the Football Betting Market." *Applied Statistics*, 46(2), 265–280.
- Karlis, D. and Ntzoufras, I. (2003). "Analysis of sports data by using bivariate Poisson models." *The Statistician*, 52(3), 381–393.
- StatsBomb open data (WC 2022 xG): [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)
