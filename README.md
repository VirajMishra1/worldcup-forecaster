# worldcup-forecaster

Calibrated probabilistic match predictor for FIFA World Cup 2026. Dixon-Coles score model (independent Poisson + low-score tau correction), time-decayed MLE fit on 30 years of international results, lineup-aware adjustment via Transfermarkt squad values. Predictions are locked at kickoff and published to a public track record that auto-updates after every match.

This is not a "pick the winner" tool. It is a forecasting system that estimates the full probability distribution over scorelines, then derives every market (W/D/L, exact score, totals, BTTS) from that distribution. Success is measured by **calibration and log-loss vs a uniform 1/3-1/3-1/3 random baseline**, not by accuracy.

---

## WC 2026 — Model Winner Odds

10,000 Monte Carlo simulations · updated daily · [model details](#model)

| Team | Win% | Implied odds |
|------|------|-------------|
| 🇦🇷 Argentina | 21.3% | 3.7:1 |
| 🇪🇸 Spain | 12.3% | 7.1:1 |
| 🇧🇷 Brazil | 10.8% | 8.3:1 |
| 🇫🇷 France | 9.7% | 9.3:1 |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 7.5% | 12.3:1 |
| 🇺🇸 United States | 7.1% | 13.0:1 |
| 🇲🇦 Morocco | 5.7% | 16.5:1 |
| 🇵🇹 Portugal | 5.3% | 17.9:1 |
| 🇩🇪 Germany | 4.5% | 21.4:1 |
| 🇲🇽 Mexico | 3.5% | 27.7:1 |

_Squad-value adjustment applied (Transfermarkt €M, exponent 0.375). 32 completed WC results included in refit. Updated 2026-06-20._

---

## The bet

| | |
|---|---|
| What it is | A Dixon-Coles score model (independent Poisson + low-score tau correction), fit on 30 years of historical international matches via MLE with time-decay weighting. Lineup adjustments derived from Transfermarkt market-value deltas. Predictions logged before kickoff, never edited. |
| What it isn't | A "guaranteed winner" tool. A high-accuracy exact-score predictor. A replacement for bookmaker odds. |
| Honest target | Log-loss below 1.09 (uniform baseline) on the live track record; reliability diagram within five percentage points of the diagonal on the historical walk-forward backtest. |
| Honest range on remaining ~48 WC matches | W/D/L accuracy 50-58%. Log-loss 0.98-1.05. Exact-score accuracy 10-13%. Over/Under 2.5 60-63%. |

---

## Realistic per-metric ceilings

Numbers below are realistic best cases over the remaining WC matches. Variance dominates at this sample size. The historical backtest is the credibility anchor; the live tournament is the consistency demo.

| Metric | Random baseline | Realistic model best | Realistic model worst |
|---|---|---|---|
| W/D/L accuracy | 33% | 55-58% | 45-50% |
| Log-loss (W/D/L) | 1.099 | 0.98-1.02 | 1.05-1.10 |
| Brier score (3-way) | 0.667 | 0.58-0.61 | 0.62-0.65 |
| Exact scoreline accuracy | ~7% | 11-13% | 8-10% |
| Over/Under 2.5 goals | 50% | 60-63% | 55-58% |
| BTTS | 50% | 57-60% | 53-56% |

---

## Model

### Core: Dixon-Coles score model

Each team has time-evolving attack strength `alpha_i` and defense strength `delta_i`. For a match between home team `i` and away team `j`:

```
lambda_home = exp(alpha_i + delta_j + gamma * is_home + epsilon_lineup_i)
lambda_away = exp(alpha_j + delta_i + epsilon_lineup_j)

P(X = x, Y = y) = tau(x, y, lambda_home, lambda_away)
                  * Poisson(x; lambda_home)
                  * Poisson(y; lambda_away)
```

The Dixon-Coles `tau` correction inflates probabilities at low-score outcomes (0-0, 1-0, 0-1, 1-1), which Poisson otherwise systematically under-predicts in football.

`is_home = 0` for World Cup neutral venues. Do not import club-football home advantage onto international neutrals.

### Parameter fitting

Maximum likelihood estimation on historical match data (13,779 matches, 2010–2024), with exponential time-decay weighting (half-life 1.5 years). Friendly matches are downweighted at 0.15× relative to competitive fixtures; UEFA Nations League and CONCACAF Nations League matches carry full competitive weight.

```
weight(match) = exp(-ln(2) * age_years / 1.5)
```

Per-team strengths are pooled toward a country-prior mean (Brazil's mean is not Saudi Arabia's). Regularize via L2 on `alpha_i` and `delta_i` deviations from the pooled prior.

### Squad strength adjustment (Phase 3)

Every prediction applies a squad-value prior derived from Transfermarkt squad market values. This corrects for teams whose historical ratings are noisy (few competitive games, easy qualifier opponents) and anchors minnow vs elite mismatches.

```
squad_ratio  = transfermarkt_squad_value / mean_wc_squad_value
epsilon_squad = 0.4 * log(clip(squad_ratio^0.375, 0.5, 1.5))
              = 0.15 * log(squad_ratio)   (effective coefficient)
lambda_h *= exp(epsilon_squad_h)
lambda_a *= exp(epsilon_squad_a)
```

Examples: France €1.2B vs mean €282M → +9% lambda. Haiti €18M → −16% lambda. The 0.375 exponent (effective coefficient 0.15) keeps the adjustment conservative — match-history ratings remain dominant.

### Red card handling (live extension, v2 only)

Standard literature adjustment. On red card to team A at minute `t`:

```
remaining_fraction = (90 - t) / 90
lambda_A_adjusted   = lambda_A * exp(-0.4 * remaining_fraction)
lambda_OPP_adjusted = lambda_OPP * exp( 0.3 * remaining_fraction)
```

Coefficients fitted on historical matches with red cards (Football-Data.co.uk has red-card columns).

### Substitution handling

Most substitutions are like-for-like and signal is below noise. Ignore in v1. v2: subtract leaving player rating, add entering player rating, apply small lambda adjustment.

---

## Data sources

| Source | Use | Cost | Notes |
|---|---|---|---|
| [Football-Data.co.uk](https://www.football-data.co.uk/) | 30 years of historical match results plus closing odds | Free | Backtest anchor. CSV per league/season. |
| [ClubElo](http://clubelo.com/) | Pre-computed Elo for sanity checking own ratings | Free API | Reference, not source |
| [Football-Data.org](https://www.football-data.org/) | Live fixtures, lineups, events | Free tier | Pre-kickoff lineup feed |
| [FBref](https://fbref.com/) | Advanced stats, xG/90, per-player numbers | Scrape, free | Optional v2 input |
| [understat](https://understat.com/) | Shot-level xG | Scrape, free | Optional v2 input |
| [StatsBomb open data](https://github.com/statsbomb/open-data) | Event-level data including past WCs | Free | v2 feature extraction |
| [Transfermarkt](https://www.transfermarkt.com/) | Market values per player | Scrape, free | Lineup strength delta |
| [SofaScore](https://www.sofascore.com/) | Live in-play events | Scrape, free | v2 in-play only |
| [Pinnacle closing odds](https://www.pinnacle.com/) | External benchmark (not yet integrated) | Free via scrape | Future calibration target |
| [Polymarket](https://polymarket.com/) | Thinner book for ROI comparison | Free API | Edge candidate |

---

## Architecture

```
worldcup-forecaster/
  data/
    raw/                          # CSV dumps, never edited
    historical_matches.parquet    # Cleaned, joined, deduped
    teams.parquet                 # Country prior strengths
    fixtures_2026.parquet         # WC schedule
    predictions.parquet           # Locked predictions, append-only
    results.parquet               # Realized outcomes
  scripts/
    fetch_historical.py           # One-shot, pulls 30 years
    fetch_lineups.py              # Cron, 90min before kickoff
    fetch_results.py              # Cron, 30min after final whistle
    fetch_odds.py                 # Cron, T-60min: Pinnacle + Polymarket
  model/
    elo.py                        # Time-decay Elo computation
    poisson.py                    # Dixon-Coles MLE fit
    features.py                   # Rest days, travel, lineup delta
    predict.py                    # Distribution over scorelines
    markets.py                    # W/D/L, exact, totals, BTTS from joint
  backtest/
    walk_forward.py               # Re-fit weekly, no lookahead
    metrics.py                    # Log-loss, Brier, calibration, ROI
    plots.py                      # Reliability diagram, log-loss curve
  cli/
    predict.py                    # `python -m cli.predict ENG vs USA`
    backtest.py                   # `python -m cli.backtest --years 2015 2024`
  reports/
    track_record.md               # Auto-regenerated, appears in README
    calibration.png               # Auto-regenerated
    log_loss_curve.png            # Auto-regenerated
  .github/workflows/
    daily-fetch.yml               # Cron pulls fixtures, results, regenerates README
    pre-match-lock.yml            # Lineup poll + prediction lock
  pyproject.toml
  README.md
```

Stack:

- Python 3.11+
- numpy, scipy, statsmodels, pandas, pyarrow
- requests + httpx for fetchers
- matplotlib for static reports
- pytest for the few unit tests that matter (MLE convergence, market integration sanity)

No frontend framework. No ORM. No service mesh. The public artifact is the README, regenerated daily by GitHub Actions.

---

## Pipeline (per fixture)

```
T-24h:   Cron computes pre-lineup probabilistic forecast using squad-strength only.
         Stored as "preliminary" prediction.
T-90m:   Lineup poll starts, every 5 min.
T-60m:   Lineups locked. Recompute with XI-strength delta.
         Pull Pinnacle + Polymarket closing odds for benchmarking.
         Write "official" locked prediction to predictions.parquet.
T+0:     Match kicks off. Prediction frozen.
T+90:    Result fetched. Written to results.parquet.
T+90+5:  Metrics recomputed across all matches to date.
         Reports regenerated, README pushed.
```

The README on the repo's main page always shows the current track record. Anyone clicking the link sees live calibration vs the bookmaker.

---

## Evaluation

### What to measure

The right metric for a probabilistic forecaster is **log-loss**, not accuracy. Log-loss penalizes overconfidence and rewards calibration. A model that says "60% home win" and is right 60% of the time has perfect calibration even with 60% accuracy.

Tracked per match and aggregated:

- Log-loss vs realized outcome (W/D/L, totals, BTTS)
- Brier score (three-outcome generalization)
- Reliability diagram bins
- Implied vs realized probability scatter
- ROI per market vs Pinnacle and vs Polymarket

### What to publish

- Reliability diagram from the 10K-match historical backtest. This is the single most credible artifact.
- Running log-loss curve across WC matches, with uniform-random and always-home baselines overlaid.
- Per-market scorecard at end of tournament.
- Honest write-up of one market where the model beat the book and one where it lost badly.

### Walk-forward backtest, no lookahead

Refit the model after each "week" (or each WC matchday) using only data available before kickoff of the next match. No future data leaks into past predictions. Implementation in `backtest/walk_forward.py`.

---

## Public artifact strategy

No web frontend. The repository README is the artifact.

GitHub Actions on cron:

1. **Daily fetch** pulls fixtures + any completed results.
2. **Pre-match lock** runs 60 minutes before each fixture, polls lineups, locks a prediction.
3. **Post-match update** fetches the result and regenerates:
   - `reports/track_record.md` (table appended to README)
   - `reports/calibration.png` (committed and embedded in README)
   - `reports/log_loss_curve.png` (committed and embedded in README)
4. Commits regenerated reports back to the repo.

The README always shows the current state. Anyone landing on the GitHub page sees the live scorecard. No deployment. No infra cost beyond GitHub Actions free tier.

---

## Roadmap

### Phase 1 — MVP, 72 hours

- [x] Pull historical match data (JamshadAli18 GitHub mirror, 13,779 matches 2010–2024)
- [x] Clean and join into one parquet
- [x] Time-decay weighting (half-life 1.5yr), friendly downweight (0.15)
- [x] Dixon-Coles MLE fit (scipy L-BFGS-B, vectorised; independent Poisson + tau low-score correction)
- [x] CLI: `predict` outputs scoreline grid plus all markets (W/D/L, O/U 2.5, BTTS, exact scores)
- [x] Walk-forward backtest on 2018-2024, generate calibration plot
- [x] Predict every remaining WC fixture — append-only predictions.parquet, top-3 scorelines stored

Goal: be predicting before tomorrow's matchday.

### Phase 2 — Automation, 1 week

- [x] Monte Carlo tournament simulator (10,000 simulations, bracket-aware)
- [x] Football-Data.org integration for live fixtures + results (daily CI)
- [x] Transfermarkt squad-value adjustment via `model/lineup.py` + `data/squad_values.json`
- [x] Polymarket odds fetch + model-vs-market edge table (`cli/odds.py`)
- [x] GitHub Action: daily fetch + pre-match prediction lock (`daily-fetch.yml`)
- [x] GitHub Action: post-match track record update + README regeneration
- [x] Auto-generated reliability diagram + log-loss curve in README (see below)

Goal: hands-off operation for the rest of WC.

### Phase 3 — Refinement, ongoing

- [x] Country-prior pooling via adaptive per-team L2 regularization (thin-data nations 3× shrinkage)
- [x] Recent-form term separate from time-decay (last-5 competitive matches goal ratio)
- [x] Rest-days feature (±3% per day vs 4-day baseline; travel-km skipped — WC venues within same country)
- [x] Cross-validated regularization tuning (grid: 0.003–0.030, best found reported in reports/reg_tuning.json)
- [x] Per-market backtest comparisons (O/U 2.5, BTTS calibration curves) — `uv run python3 -m cli.market_stats`

### Phase 4 — In-play, v2

- [x] SofaScore polling for live events (`scripts/fetch_live_events.py`, no API key needed)
- [x] Real-time lambda updates (`model/inplay.py` — remaining-time Poisson + red card adjustment)
- [x] Live win-probability dashboard at [virajmishra1.github.io/worldcup-forecaster](https://virajmishra1.github.io/worldcup-forecaster/) — updates every 5 min during matches, auto-refreshes in browser
- [ ] Snapshot every 30 seconds for post-match in-play accuracy analysis

### Phase 5 — Edge hunting, v3

- [x] Polymarket arbitrage scanner using model probabilities (`cli/odds.py`)
- [x] Kelly-criterion sizing simulator (`cli/kelly.py`)
- [x] Cross-promotion with [polymath](https://github.com/VirajMishra1/polymath) — Polymarket analytics terminal by the same author

---

## Honest constraints and risks

| Risk | Reality | Mitigation |
|---|---|---|
| Sample size | 48 remaining matches is too small to prove skill | Anchor credibility on 10K-match backtest, frame WC as demo not proof |
| Variance | A two-match upset streak makes a good model look bad | Publish log-loss and calibration, not accuracy |
| Lineup unreliability | Sometimes leaked, sometimes wrong | Fall back to preliminary forecast if no lineup by T-30m |
| Scraping fragility | Transfermarkt, SofaScore TOS-gray, layout can change | Wrap fetchers in try/except, fall back gracefully, never block on a single source |
| International data sparsity | Some nations have few competitive matches | Country-prior pooling, regularization toward continental confederation means |
| Overfitting | Easy to add features and look good on backtest | Walk-forward CV is the only metric that counts; in-sample fit is a lie |
| Public bad streak | First three picks could all flop | Predictions are locked and published anyway; honest losses are the credibility play |

---

## What NOT to do

- Do not promise exact-score accuracy as a headline metric.
- Do not optimize for W/D/L accuracy at the cost of log-loss.
- Do not use future data in any backtest, ever.
- Do not hide bad predictions. The track record is append-only.
- Do not import club-football home advantage onto neutral WC venues.
- Do not train a neural network on 5,000 matches and expect it to beat Poisson. It will not.
- Do not introduce a frontend before the model is calibrated.
- Do not chase ROI vs Pinnacle on main markets. You will lose.

---

## Setup

```
git init
git remote add origin git@github.com:VirajMishra1/worldcup-forecaster.git

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python scripts/fetch_historical.py
python -m cli.backtest --start 2018-01-01 --end 2024-12-31
python -m cli.predict --home France --away Argentina
```

---

## References

- Dixon, M. and Coles, S. (1997). "Modelling Association Football Scores and Inefficiencies in the Football Betting Market." *Applied Statistics*, 46(2), 265-280.
- Karlis, D. and Ntzoufras, I. (2003). "Analysis of sports data by using bivariate Poisson models." *The Statistician*, 52(3), 381-393.
- Constantinou, A. and Fenton, N. (2013). "Determining the level of ability of football teams by dynamic ratings based on the relative discrepancies in scores between adjacent divisions." *Journal of Quantitative Analysis in Sports*, 9(1), 37-50.
- Hvattum, L. and Arntzen, H. (2010). "Using ELO ratings for match result prediction in association football." *International Journal of Forecasting*, 26(3), 460-470.
- FiveThirtyEight Soccer Predictions methodology (archived).

## Model Calibration

A reliability diagram answers: when the model says 60%, does the outcome happen 60% of the time? Perfect calibration sits on the diagonal.

Historical backtest (walk-forward, no lookahead, form + rest factors included): **log_loss=0.8961 · brier=0.5265 · accuracy=59.0% · n=5,518 matches** (2018–2023, refit every 30 days)

![Calibration](reports/calibration.png)

![Log-loss curve](reports/log_loss_curve.png)

---

<!-- TRACK_RECORD_START -->
## Live Track Record (9 matches)

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

<!-- ODDS_COMPARISON_START -->
## Model vs Polymarket · market updated Jun 19 10:12 UTC

| Team | Model | Market | Edge |
|------|-------|--------|------|

_Edge = Model% − Market%. Positive = model thinks team is underpriced on Polymarket._

<!-- ODDS_COMPARISON_END -->
