# FPL Model

A mathematical model for picking an optimal Fantasy Premier League squad, built
around modelling each scoring rule the way the rule actually works rather than
approximating it.

Two stages: an expected-points model that projects every player, and an integer
program that picks the best legal 15 under budget. The optimiser maximises
expected **rank**, not raw points.

```
ingest.py  →  xp_model.py  →  optimizer.py  →  app.py
 database      projections      squad           web UI
```

---

## Why this exists

Most public FPL solvers get two of the biggest scoring routes wrong. This was
measured against 2025-26 data, not assumed:

| Component | Common approach | Reality | Error |
|---|---|---|---|
| Clean sheets | sigmoid `1/(1+e^(gc−1))` → 0.408 | Poisson `e^(−λ)` → 0.254 | actual was **0.257** |
| Defensive contributions | linear per-90 rate → 1.490 xP | threshold probability → 0.539 xP | **2.8× overvalued** |

The clean-sheet error inflates every defender and goalkeeper by roughly 59%.
The DefCon error is worse because it is largest exactly where a solver looks: a
defender recording 19 CBIT scores **2 points**, the same as one recording 10,
but a linear model awards 3.8.

---

## The model

Expected points are built from eight components, each modelled to match the
scoring rule:

| Component | Method | Why not the naive version |
|---|---|---|
| Minutes | Multinomial P(0) / P(1–59) / P(60+) | A binary 60' cutoff throws away the largest source of uncertainty |
| Clean sheet | Poisson `P(CS) = e^(−λ)` | The sigmoid approximation overstates by ~59% |
| Goals / assists | xG90, xA90 × expected minutes × fixture | Raw goals/90 overreacts to finishing variance |
| DefCon | Negative binomial `P(actions ≥ threshold)` | Points are **capped** at 2 — it is a threshold, not a rate |
| Saves | `E[⌊saves/3⌋]` over a Poisson | Save points are also a threshold |
| Goals conceded | `E[⌊conceded/2⌋]` over a Poisson | Linear approximation misprices the −1 per 2 |
| Bonus | Within-fixture BPS rank | Bonus is top-3 per match, not linear in BPS |
| Appearance | `P(60+)×2 + P(1–59)×1` | — |

Team strength comes from a Poisson regression on **ClubElo** ratings and home
advantage, fitted on 380 fixtures — not from FPL's own `strength_*` fields,
which are all zero at season start and would cause a divide-by-zero.

### Optimising for rank, not points

Expected points alone is the wrong objective for a league. What moves rank is
your holding of a player relative to the field's **effective ownership**:

```
Δᵢ = (mᵢ − EOᵢ) · pointsᵢ
```

where `mᵢ = startᵢ + captainᵢ` (0 unowned, 1 starting, 2 captained) and
`EOᵢ = ownership + captaincy share`.

In expectation the `EOᵢ` term is constant and cancels — so a naive "ownership
bonus" does nothing. EO bites through **variance**: being far from the field on
a volatile player is what actually costs rank. The penalty is
`σᵢ²(mᵢ − EOᵢ)²`.

That looks quadratic but linearises exactly. Since `captainᵢ ≤ startᵢ` and both
are binary, `mᵢ² = startᵢ + 3·captainᵢ`, giving:

```
(mᵢ − EOᵢ)² = startᵢ(1 − 2EOᵢ) + captainᵢ(3 − 2EOᵢ) + const
```

so the whole thing stays a mixed-integer **linear** program and solves in
milliseconds.

**Worked example.** Haaland: 73.5% owned, ~77% captained → EO ≈ 1.51, with the
highest points standard deviation in the game (5.21). The start coefficient
goes strongly negative, turning the penalty into a large reward for owning him.
He enters the squad at any `λ ≥ 0.2`.

The trade-off is explicit and tunable:

| λ (template adherence) | Squad | XI expected points |
|---|---|---|
| 0.0 | no Haaland | 87.3 |
| 0.2 | Haaland, captained | 84.6 |
| 0.55 (default) | Haaland, captained | 82.6 |

About 4.7 expected points bought for a large reduction in rank variance.

---

## Handling unknown players

28% of players at GW1 had no Premier League history — promoted clubs are
90–100% blank. Every player still gets an estimate, tagged by how much is
actually known:

| Tier | Meaning | Players | Avg confidence |
|---|---|---|---|
| A | 450+ PL minutes last season | 316 | 0.57 |
| A− | Some PL minutes, under 450 | 89 | 0.16 |
| C | No PL history, pre-season minutes only | 102 | 0.10 |
| D | Nothing at all | 92 | 0.00 |

Rates shrink toward a positional prior with weight `minutes / (minutes + 900)`
— continuous, rather than hardcoded confidence buckets. Low-confidence players
are discounted in selection and marked visually in the UI, so the model can
say *I don't know* instead of bluffing a positional average.

**Pre-season is a minutes source only.** Friendlies have 95% minutes coverage
but 4% xG coverage, and the opposition (York City, Bromley, Konyaspor) makes
rate estimation meaningless. It is used for `p_start_est` and nothing else.

**Transfers.** Per-90 rates don't survive a move — 0.45 xG/90 in a weak side
and in a strong side are different players. The model stores `share_team_xg`
and `share_team_xa` (fraction of the old club's output) and projects a mover as
`share × new team's expected xG`.

---

## Data

[olbauday/FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights),
refreshed twice daily. Per-player per-match defensive action counts (tackles,
interceptions, blocks, clearances, recoveries) are what make threshold-based
DefCon modelling possible; xG/xA and ClubElo ratings come from the same source,
aligned to official FPL IDs.

### Four data traps, all handled and asserted in `verify.py`

1. **`matches` joins teams on `code`, not `id`** — despite the upstream docs.
   Both are small integers, so joining on `id` yields a fully populated table
   with the wrong club on every fixture and no error anywhere.
2. **`matches.csv` mixes competitions** — Champions League, Europa, EFL Cup and
   friendlies sit alongside league games, with NULL team references for non-PL
   opponents. Fitting team strength unfiltered blends a Real Madrid tie into a
   club's league record.
3. **The league's slug is `prem`**, not `premier-league`. The obvious parser
   classifies all 380 league fixtures as "other".
4. **`GROUP BY` over a `UNION ALL` view returns wrong results in DuckDB 1.5.5** —
   the group key resolves to the first branch for both, so every team collected
   only its home fixtures counted twice and `goals_for` equalled
   `goals_against` for all 20 clubs. Filtering to a single team masks it via
   predicate pushdown. `team_match` is therefore a materialised table.

### Leakage

Two traps, both guarded:

- In `player_gw_stats`, `bps`, `influence`, `creativity`, `threat`,
  `ict_index` and `starts` are **calculated after the match**. Used raw, `bps`
  "predicts" points at ρ=0.94 because BPS determines bonus. They are lagged one
  gameweek. Only `now_cost`, `selected_by_percent`, `form` and set-piece order
  are genuine deadline snapshots.
- Every rolling feature uses `shift(1)` before the expanding mean, so a
  gameweek-*k* feature sees only gameweeks 1…*k*−1.

---

## Usage

```bash
pip install duckdb pandas requests lightgbm scikit-learn scipy pulp streamlit

python ingest.py              # build the database (~2 min)
python verify.py              # 10 integrity checks — run after every ingest
python rank_features.py --raw-only
python apply_preferences.py   # blend priors with fitted weights
streamlit run app.py          # web UI
python export_html.py         # static snapshot → best_team.html
```

### Files

| File | Purpose |
|---|---|
| `ingest.py` | Download, clean, load to DuckDB |
| `verify.py` | Integrity checks — the data traps above |
| `rank_features.py` | Fit stat importance per position (LightGBM + permutation) |
| `preferences.json` | **Your priorities.** Edit this to change the model |
| `apply_preferences.py` | Blend stated priorities with fitted weights |
| `xp_model.py` | Expected points |
| `optimizer.py` | MILP squad selection |
| `app.py` / `export_html.py` | Web UI and static export |

### Tuning

`preferences.json` holds everything subjective: per-position weights for each
scoring route, set-piece multipliers, minutes-risk thresholds, template
adherence, and how sceptical to be about unknown players. `alpha` controls how
much your judgement counts against the fitted data (0 = data only, 1 =
judgement only).

Defender weights are **conditional on team strength**: for weak sides (Elo
< 1800) DefCon outranks clean sheets, since clean sheets are unlikely and a
high-CBIT defender grinding a 0-0 collects clean sheet, DefCon and bonus
simultaneously. For strong sides it flips.

Two numbers are kept deliberately separate: `xp` is the honest rules-based
projection; `selection_score` is `xp` adjusted by preferences, confidence and
template risk, and is what the optimiser maximises.

---

## Status and limitations

**Not backtested.** This is the honest headline. The weights are asserted —
some fitted on a single season, some stated by judgement. Nothing has been
tested against naive baselines (the most-owned XI, last season's top scorers).
Until that exists, nobody knows whether it beats guessing. That is the next
piece of work, and it should come before anyone trusts it with real transfers.

Other known gaps:

- **One season of DefCon data.** The rule is new, so those priors are the
  shakiest numbers in the model.
- **Ownership is not effective ownership.** Captaincy share isn't published, so
  it is modelled with a softmax whose temperature was set by judgement. If too
  concentrated, it overstates Haaland's EO.
- **No live FPL API.** Prices and injury news go stale between twice-daily data
  refreshes. A `bootstrap-static` fetch before each deadline would fix it.
- **Single gameweek.** No multi-week transfer planning, hits, or chip strategy.
- **Set-piece threat is weakly measured** — only team-level set-play xG and
  aerial duels exist as proxies, which is why its weight is deliberately low.

## Licence

MIT. Data belongs to its respective sources; see FPL-Core-Insights for terms.
