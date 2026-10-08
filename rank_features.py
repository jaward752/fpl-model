"""
Stat importance ranking for FPL points, per position.

Answers: of every stat available, which ones actually predict next-gameweek
FPL points, and by how much?

    python rank_features.py                 # rank everything, all positions
    python rank_features.py --pos DEF       # one position
    python rank_features.py --top 20        # show top N

Outputs feature_importance.csv and weights.json (editable — weights.json is
the file you tune by hand if you want to override what the data says).

CRITICAL DESIGN POINT — no lookahead.
Every feature is built ONLY from matches strictly BEFORE the one being
predicted, using an expanding mean shifted by one match. If you build features
from the same match you are predicting, tackles will "predict" points with
absurd accuracy because both describe the same 90 minutes. That is the single
easiest way to build an FPL model that backtests brilliantly and loses money.
This is also why vaastav's `xP` column must never be used as a feature: it is
scraped after the gameweek ends and can contain post-deadline information.
"""

from __future__ import annotations

import argparse
import json

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

import lightgbm as lgb

import config as C

SEASON = "2025-2026"

# Per-match stats to turn into rolling form features.
# VIEW_STATS come from v_player_match, BASE_STATS from player_match_stats.
VIEW_STATS = [
    "minutes", "goals", "assists", "xg", "xa", "shots", "chances_created",
    "saves", "xgot_faced", "goals_prevented", "team_goals_conceded",
    "tackles", "interceptions", "blocks", "clearances", "recoveries",
    "cbit", "cbirt",
]
BASE_STATS = [
    "xgot", "shots_on_target", "successful_dribbles", "big_chances_missed",
    "touches_opposition_box", "touches", "accurate_passes",
    "final_third_passes", "accurate_crosses", "accurate_long_balls",
    "tackles_won", "headed_clearances", "dribbled_past", "duels_won",
    "duels_lost", "ground_duels_won", "aerial_duels_won", "was_fouled",
    "fouls_committed", "sweeper_actions", "high_claim", "saves_inside_box",
    "dispossessed", "offsides", "penalties_scored",
    # physical / workrate
    "top_speed", "distance_covered", "sprinting_distance",
    "number_of_sprints", "running_distance",
]
MATCH_STATS = VIEW_STATS + BASE_STATS

# FPL-side fields. CRITICAL SPLIT.
#
# DEADLINE_SNAPSHOT are genuinely known before kickoff — the upstream docs
# describe now_cost, selected_by_percent and form as "the player's status at
# the deadline of that gameweek". Safe to use contemporaneously.
#
# POST_MATCH are described as "calculated by subtracting the previous
# gameweek's total from the current one" — i.e. they describe the match that
# has already been played. Using them raw is LEAKAGE: bps determines bonus
# points, which are part of the points being predicted, so bps "predicts"
# points at rho=0.94 and buries every real signal. These are lagged by one
# gameweek like any other form feature.
DEADLINE_SNAPSHOT = [
    "now_cost", "selected_by_percent", "form",
    "penalties_order", "corners_and_indirect_freekicks_order",
    "direct_freekicks_order",
]
POST_MATCH = [
    "bps", "influence", "creativity", "threat", "ict_index", "starts",
    "transfers_in_event", "transfers_out_event",
]
SNAPSHOT_STATS = DEADLINE_SNAPSHOT + POST_MATCH

HUMAN = {
    "minutes": "Minutes played", "xg": "Expected goals (xG)",
    "xa": "Expected assists (xA)", "xgot": "Expected goals on target",
    "cbit": "CBIT (defensive actions, DEF)",
    "cbirt": "CBIRT (defensive actions incl. recoveries)",
    "p60_rate": "Rate of 60+ minute appearances",
    "defcon_rate": "DefCon threshold hit rate",
    "elo_diff": "Team Elo minus opponent Elo",
    "opponent_elo": "Opponent strength (Elo)",
    "is_home": "Playing at home",
    "now_cost": "Price", "selected_by_percent": "Ownership %",
    "bps": "Bonus Points System score", "form": "FPL form",
    "penalties_order": "Penalty taker order",
    "team_goals_conceded": "Goals conceded while on pitch",
}


def load_panel(con) -> pd.DataFrame:
    """One row per player per league gameweek, with the points target."""
    vcols = ", ".join(f"pm.{c}" for c in VIEW_STATS)
    bcols = ", ".join(f"s.{c}" for c in BASE_STATS)
    snap = ", ".join(f"g.{c}" for c in SNAPSHOT_STATS)
    df = con.execute(f"""
        SELECT pm.player_id, pm.web_name, pm.pos, pm.gameweek AS gw,
               pm.is_home, pm.team_elo, pm.opponent_elo, pm.hit_defcon,
               {vcols}, {bcols}, {snap},
               g.event_points AS points
        FROM v_player_match pm
        JOIN player_match_stats s
          ON s.player_id = pm.player_id AND s.match_id = pm.match_id
         AND s.season = pm.season
        JOIN player_gw_stats g
          ON g.id = pm.player_id AND g.season = pm.season AND g.gw = pm.gameweek
        WHERE pm.is_pl AND pm.season = '{SEASON}'
        ORDER BY pm.player_id, pm.gameweek
    """).df()
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Expanding means over PRIOR matches only. shift(1) is what makes this
    honest: at gameweek k a player's feature is the mean of gameweeks 1..k-1.
    """
    df = df.sort_values(["player_id", "gw"]).copy()
    g = df.groupby("player_id", sort=False)
    feats: list[str] = []

    for c in MATCH_STATS:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[f"{c}_avg"] = g[c].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean())
        feats.append(f"{c}_avg")
        # short-term form: last 4 matches
        df[f"{c}_r4"] = g[c].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean())
        feats.append(f"{c}_r4")

    # derived rate features
    df["_p60"] = (pd.to_numeric(df["minutes"], errors="coerce") >= 60).astype(float)
    df["p60_rate"] = g["_p60"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    feats.append("p60_rate")

    df["_dc"] = df["hit_defcon"].astype(float)
    df["defcon_rate"] = g["_dc"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    feats.append("defcon_rate")

    df["_pts"] = pd.to_numeric(df["points"], errors="coerce")
    df["points_avg"] = g["_pts"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["points_r4"] = g["_pts"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    feats += ["points_avg", "points_r4"]

    # fixture context — known before kickoff, no shift needed
    df["elo_diff"] = df["team_elo"] - df["opponent_elo"]
    df["is_home"] = df["is_home"].astype(float)
    feats += ["elo_diff", "opponent_elo", "team_elo", "is_home"]

    # deadline snapshots — genuinely point-in-time, safe as-is
    for c in DEADLINE_SNAPSHOT:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            feats.append(c)

    # post-match fields — MUST be lagged or they leak the target
    for c in POST_MATCH:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[f"{c}_prev_avg"] = g[c].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean())
        df[f"{c}_prev_r4"] = g[c].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean())
        feats += [f"{c}_prev_avg", f"{c}_prev_r4"]

    return df, feats


def rank_position(df: pd.DataFrame, feats: list[str], pos: str) -> pd.DataFrame:
    d = df[(df["pos"] == pos) & (df["gw"] >= 5)].copy()
    d = d.dropna(subset=["points"])
    if len(d) < 200:
        return pd.DataFrame()

    X = d[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = d["points"].astype(float).values
    groups = d["player_id"].values

    # keep only features with signal-bearing variance
    keep = [c for c in feats if X[c].std() > 1e-9]
    X = X[keep]

    # 1) rank correlation — simple, monotonic, no model assumptions
    rho = {c: abs(spearmanr(X[c], y, nan_policy="omit").statistic) for c in keep}

    # 2) model-based: grouped CV so the same player never spans train/test
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=0, verbose=-1,
    )
    cv = GroupKFold(n_splits=4)
    gains = np.zeros(len(keep))
    perms = np.zeros(len(keep))
    for tr, te in cv.split(X, y, groups):
        model.fit(X.iloc[tr], y[tr])
        gains += model.booster_.feature_importance(importance_type="gain")
        pi = permutation_importance(model, X.iloc[te], y[te],
                                    n_repeats=3, random_state=0, n_jobs=1)
        perms += np.clip(pi.importances_mean, 0, None)
    gains /= gains.sum() or 1
    perms /= perms.sum() or 1

    out = pd.DataFrame({
        "feature": keep,
        "spearman": [rho[c] for c in keep],
        "gain": gains,
        "permutation": perms,
    })
    # blended score, normalised to 100
    out["score"] = (
        0.25 * out.spearman / (out.spearman.max() or 1)
        + 0.35 * out.gain / (out.gain.max() or 1)
        + 0.40 * out.permutation / (out.permutation.max() or 1)
    )
    out["score"] = 100 * out.score / out.score.max()
    out["pos"] = pos
    out["label"] = out.feature.map(
        lambda f: HUMAN.get(f, HUMAN.get(f.replace("_avg", "").replace("_r4", ""),
                                         f.replace("_", " "))))
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", nargs="*", default=["GKP", "DEF", "MID", "FWD"])
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--raw-only", action="store_true",
                    help="drop composite/aggregate features (form, points, bps, "
                         "ICT) so underlying match stats rank on their own merit")
    args = ap.parse_args()

    con = duckdb.connect(str(C.DB_PATH), read_only=True)
    raw = load_panel(con)
    con.close()
    print(f"panel: {len(raw):,} player-gameweeks")

    df, feats = build_features(raw)
    if args.raw_only:
        drop = ("form", "points_avg", "points_r4", "bps_", "influence_",
                "creativity_", "threat_", "ict_index_", "selected_by_percent",
                "transfers_in_event", "transfers_out_event", "starts_")
        feats = [f for f in feats if not f.startswith(drop)]
        print("raw-only mode: composite features excluded")
    print(f"features engineered: {len(feats)}\n")

    all_out = []
    for pos in args.pos:
        r = rank_position(df, feats, pos)
        if r.empty:
            print(f"{pos}: insufficient data")
            continue
        all_out.append(r)
        print(f"===== {pos} — top {args.top} predictors of FPL points =====")
        show = r.head(args.top)[["feature", "label", "score", "spearman",
                                 "permutation"]]
        show = show.assign(score=show.score.round(1),
                           spearman=show.spearman.round(3),
                           permutation=show.permutation.round(4))
        print(show.to_string(index=False))
        print()

    if all_out:
        full = pd.concat(all_out, ignore_index=True)
        full.to_csv("feature_importance.csv", index=False)

        weights = {
            pos: {r.feature: round(float(r.score), 2)
                  for r in full[full.pos == pos].head(25).itertuples()}
            for pos in full.pos.unique()
        }
        with open("weights.json", "w") as f:
            json.dump(weights, f, indent=2)
        print("wrote feature_importance.csv and weights.json")
        print("weights.json is yours to edit — it overrides the fitted scores.")


if __name__ == "__main__":
    main()
