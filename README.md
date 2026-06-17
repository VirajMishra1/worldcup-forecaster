# worldcup-forecaster

Calibrated probabilistic match predictor for FIFA World Cup 2026. Bivariate Poisson with Dixon-Coles correction, time-decayed Elo strength, lineup-aware adjustment via market values. Predictions are locked at kickoff and published to a public track record that auto-updates after every match.

This is not a "pick the winner" tool. It is a forecasting system that estimates the full probability distribution over scorelines, then derives every market (W/D/L, exact score, totals, BTTS) from that distribution. Success is measured by **calibration and log-loss vs Pinnacle closing odds**, not by accuracy.

---

## The bet

| | |
|---|---|
| What it is | A bivariate Poisson model with Dixon-Coles low-score correction, fit on 30 years of historical international matches via MLE with time-decay weighting. Lineup adjustments derived from Transfermarkt market-value deltas. Predictions logged before kickoff, never edited. |
| What it isn't | A "guaranteed winner" tool. A high-accuracy exact-score predictor. A betting system that beats Pinnacle on main markets. |
| Honest target | Log-loss within 0.03 of Pinnacle closing odds, reliability diagram within five percentage points of the diagonal on a 10,000-match historical backtest. |
| Honest range on remaining ~48 WC matches | W/D/L accuracy 50-58%. Log-loss 0.98-1.05. Exact-score accuracy 10-13%. Over/Under 2.5 60-63%. |

---

## Realistic per-metric ceilings

Numbers below are realistic best cases over the remaining WC matches. Variance dominates at this sample size. The historical backtest is the credibility anchor; the live tournament is the consistency demo.

| Metric | Random | Bookmaker (Pinnacle) | Realistic best | Realistic worst |
|---|---|---|---|---|
| W/D/L accuracy | 33% | 55-58% | 55-58% | 45-50% |
| Log-loss (W/D/L) | 1.099 | 0.95-0.97 | 0.98-1.02 | 1.05-1.10 |
| Brier score (3-way) | 0.667 | 0.55-0.57 | 0.58-0.61 | 0.62-0.65 |
| Exact scoreline accuracy | ~7% | 13-15% | 11-13% | 8-10% |
| Over/Under 2.5 goals | 50% | 60-64% | 60-63% | 55-58% |
| BTTS | 50% | 58-62% | 57-60% | 53-56% |
| ROI vs Pinnacle | -2.5% (vig) | 0% by definition | ~0% | -3% |
| ROI vs Polymarket | n/a | n/a | +2-5% plausible | -5% |

---

## Model

### Core: bivariate Poisson with Dixon-Coles correction

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

Maximum likelihood estimation on historical match data, with exponential time-decay weighting (half-life two years):

```
weight(match) = exp(-ln(2) * age_years / 2.0)
```

Per-team strengths are pooled toward a country-prior mean (Brazil's mean is not Saudi Arabia's). Regularize via L2 on `alpha_i` and `delta_i` deviations from the pooled prior.

### Lineup adjustment

One hour before kickoff, lineups are fetched. For each XI, compute the sum of Transfermarkt market values. Compare to the team's rolling 12-month average XI market value. The delta becomes a small adjustment to lambda:

```
xi_strength_ratio = sum_market_value(announced_xi) / mean_market_value(team_xi_12mo)
epsilon_lineup = 0.4 * log(xi_strength_ratio)
```

The 0.4 coefficient is fit empirically on backtest data. Constrain `xi_strength_ratio` to `[0.5, 1.5]` to prevent outliers from blowing up lambda.

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
| [Pinnacle closing odds](https://www.pinnacle.com/) | Benchmark to beat | Free via scrape | Calibration target |
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
    poisson.py                    # Bivariate Poisson + Dixon-Coles MLE
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
- Running log-loss curve across WC matches, with Pinnacle and always-home baselines overlaid.
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

- [ ] Pull Football-Data.co.uk historical CSVs (1995-2024)
- [ ] Clean and join into one parquet
- [ ] Time-decay Elo computation, sanity-check vs ClubElo
- [ ] Bivariate Poisson + Dixon-Coles MLE fit
- [ ] CLI: `predict ENG vs USA` outputs scoreline grid plus all markets
- [ ] Walk-forward backtest on 2018-2024, generate calibration plot
- [ ] Manual run: predict every remaining WC fixture, commit results

Goal: be predicting before tomorrow's matchday.

### Phase 2 — Automation, 1 week

- [ ] Football-Data.org integration for live fixtures + lineups
- [ ] Transfermarkt market-value scrape + lineup-delta adjustment
- [ ] Pinnacle + Polymarket odds scraping for benchmarking
- [ ] GitHub Action: daily fetch + pre-match prediction lock
- [ ] GitHub Action: post-match update + README regeneration
- [ ] Auto-generated reliability diagram + log-loss curve in README

Goal: hands-off operation for the rest of WC.

### Phase 3 — Refinement, ongoing

- [ ] Country-prior pooling for small-sample national teams
- [ ] Recent-form term separate from Elo
- [ ] Rest-days and travel-km features
- [ ] Cross-validated regularization tuning
- [ ] Per-market backtest comparisons

### Phase 4 — In-play, v2

- [ ] SofaScore polling for live events (goals, red cards)
- [ ] Real-time lambda updates and remaining-time integration
- [ ] Live win-probability chart on a single static HTML page
- [ ] Snapshot every 30 seconds for post-match in-play accuracy analysis

### Phase 5 — Edge hunting, v3

- [ ] Polymarket arbitrage scanner using model probabilities
- [ ] Kelly-criterion sizing simulator on backtest
- [ ] Cross-promotion with [polymath](https://github.com/VirajMishra1/polymath)

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
python -m cli.predict --home ENG --away USA
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

![Calibration](reports/calibration.png)

![Log-loss curve](reports/log_loss_curve.png)

Historical backtest (walk-forward, no lookahead): **log_loss=0.8420 · brier=0.4950 · accuracy=61.2% · n=3,807 matches**

---

<!-- TRACK_RECORD_START -->
_No completed matches yet._

<!-- TRACK_RECORD_END -->
