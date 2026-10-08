"""
Expected points model.

Two numbers are produced per player per gameweek, and the distinction matters:

  xp              honest rules-based expected points. No preferences applied.
  selection_score xp adjusted by your preference weights and template risk.
                  This is what the optimiser maximises.

Keeping them separate means you can always see what the model actually thinks
before your priors tilt it.

Components, and why each is modelled the way it is:
  minutes   multinomial P(0)/P(1-59)/P(60+), not a binary 60' cutoff
  clean sheet  Poisson exp(-lambda), NOT a sigmoid (sigmoid overstates by ~59%)
  goals/assists  xG90/xA90 scaled by expected minutes and fixture
  defcon    P(actions >= threshold) from a negative binomial, NOT a linear rate
  saves     E[floor(saves/3)] over a Poisson, since save points are a threshold
  conceded  E[floor(goals_conceded/2)] over a Poisson
  bonus     P(top-3 BPS) approximated from a BPS-proxy rank within the fixture
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor

import config as C

ROOT = Path(__file__).resolve().parent
SEASON_HIST = "2025-2026"
SEASON_CUR = "2026-2027"

POINTS = {
    "goal": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "assist": 3,
    "cs": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "defcon": 2,
}


def fit_goal_model(con) -> PoissonRegressor:
    """log(expected goals) ~ elo_diff + home. Replaces arbitrary ratio models."""
    d = con.execute(f"""
        SELECT goals_for, elo - opponent_elo AS elo_diff,
               CASE WHEN is_home THEN 1 ELSE 0 END AS home
        FROM team_match
        WHERE is_pl AND season = '{SEASON_HIST}'
          AND goals_for IS NOT NULL AND elo IS NOT NULL
    """).df()
    X = d[["elo_diff", "home"]].astype(float).values / np.array([100.0, 1.0])
    m = PoissonRegressor(alpha=1e-6, max_iter=500).fit(X, d.goals_for.values)
    return m


def expected_goals(model, elo_diff, home):
    X = np.column_stack([np.asarray(elo_diff, float) / 100.0,
                         np.asarray(home, float)])
    return model.predict(X)


def defcon_probability(rate_per90, exp_minutes, threshold, dispersion=4.0):
    """
    P(defensive actions >= threshold). Negative binomial, not linear scaling.

    A defender averaging 19 CBIT still scores exactly 2 points, so multiplying
    a per-90 rate by points/threshold (as fpl-solver does) overvalues high
    volume defenders by up to 3x. Overdispersion is real in count data, so a
    negative binomial fits better than a Poisson.
    """
    mu = np.clip(np.asarray(rate_per90, float) * np.asarray(exp_minutes, float) / 90.0, 1e-6, None)
    r = dispersion
    p = r / (r + mu)
    return 1.0 - nbinom.cdf(threshold - 1, r, p)


def e_floor_div(lmbda, divisor, cap=15):
    """E[floor(X / divisor)] for X ~ Poisson(lambda)."""
    lmbda = np.asarray(lmbda, float)
    out = np.zeros_like(lmbda)
    for k in range(1, cap + 1):
        out += poisson.sf(k * divisor - 1, lmbda)
    return out


def build(gw: int = 1) -> pd.DataFrame:
    con = duckdb.connect(str(C.DB_PATH), read_only=True)
    gmodel = fit_goal_model(con)

    fixtures = con.execute(f"""
        SELECT match_id, gameweek, home_team, away_team,
               home_team_elo, away_team_elo
        FROM v_matches
        WHERE season = '{SEASON_CUR}' AND is_pl AND gameweek = {gw}
    """).df()

    players = con.execute(f"""
        SELECT pp.*, g.now_cost, g.selected_by_percent,
               g.penalties_order, g.corners_and_indirect_freekicks_order,
               g.direct_freekicks_order
        FROM player_prior pp
        LEFT JOIN player_gw_stats g
          ON g.id = pp.player_id AND g.season = pp.season
    """).df()
    con.close()

    # map each player to his fixture
    long = []
    for _, f in fixtures.iterrows():
        for team, opp, home, elo, oelo in (
            (f.home_team, f.away_team, 1, f.home_team_elo, f.away_team_elo),
            (f.away_team, f.home_team, 0, f.away_team_elo, f.home_team_elo),
        ):
            sub = players[players.team == team].copy()
            sub["opponent"] = opp
            sub["is_home"] = home
            sub["elo_diff"] = elo - oelo
            long.append(sub)
    df = pd.concat(long, ignore_index=True)

    # ---- team-level expectations -------------------------------------------
    df["team_xg"] = expected_goals(gmodel, df.elo_diff, df.is_home)
    df["team_xgc"] = expected_goals(gmodel, -df.elo_diff, 1 - df.is_home)
    df["p_cs"] = np.exp(-df.team_xgc)          # Poisson, not sigmoid

    # ---- minutes: multinomial ----------------------------------------------
    # p_start_est is already the observed rate of 60+ minute appearances, so
    # it needs no extra haircut. Conditional means are CALIBRATED from 2025-26
    # rather than guessed: previously a flat 82 minutes with a 0.92 multiplier
    # capped every player at ~75 expected minutes, so no one could be modelled
    # as a nailed 90-minute starter. That systematically understated players
    # whose value is concentrated in per-90 rates - i.e. premium attackers.
    MEAN_IF_60 = {"GKP": 89.9, "DEF": 87.5, "MID": 83.2, "FWD": 81.1}
    MEAN_IF_CAMEO = {"GKP": 32.2, "DEF": 21.9, "MID": 22.9, "FWD": 20.7}
    p_start = df.p_start_est.clip(0, 1).fillna(0.0)
    df["p60"] = p_start
    df["p1_59"] = (1 - p_start) * 0.35
    df["p0"] = (1 - df.p60 - df.p1_59).clip(0, 1)
    df["exp_minutes"] = (df.p60 * df.pos.map(MEAN_IF_60).fillna(84)
                         + df.p1_59 * df.pos.map(MEAN_IF_CAMEO).fillna(22))

    # ---- set-piece multipliers ---------------------------------------------
    prefs = json.loads((ROOT / "preferences.json").read_text())
    sp = prefs["set_pieces"]
    pen = pd.to_numeric(df.penalties_order, errors="coerce")
    cor = pd.to_numeric(df.corners_and_indirect_freekicks_order, errors="coerce")
    fk = pd.to_numeric(df.direct_freekicks_order, errors="coerce")
    df["pen_mult"] = np.where(pen == 1, sp["penalty_taker_1st"],
                       np.where(pen == 2, sp["penalty_taker_2nd"], 1.0))
    df["cor_mult"] = np.where(cor == 1, sp["corner_taker_1st"], 1.0)
    df["fk_mult"] = np.where(fk == 1, sp["freekick_taker_1st"], 1.0)

    mins90 = df.exp_minutes / 90.0

    # ---- attacking returns --------------------------------------------------
    df["e_goals"] = df.xg90.fillna(0) * mins90 * df.pen_mult * df.fk_mult
    df["e_assists"] = df.xa90.fillna(0) * mins90 * df.cor_mult

    goal_pts = df.pos.map(POINTS["goal"]).fillna(4).astype(float)
    cs_pts = df.pos.map(POINTS["cs"]).fillna(0).astype(float)

    df["pts_goals"] = df.e_goals * goal_pts
    df["pts_assists"] = df.e_assists * POINTS["assist"]

    # ---- clean sheets (needs 60+ minutes) ----------------------------------
    df["pts_cs"] = df.p_cs * df.p60 * cs_pts

    # ---- appearance ---------------------------------------------------------
    df["pts_appearance"] = df.p60 * 2 + df.p1_59 * 1

    # ---- defcon: threshold, capped ------------------------------------------
    thr = df.pos.map({"DEF": 10, "MID": 12, "FWD": 12}).astype(float)
    rate = np.where(df.pos == "DEF", df.cbit90.fillna(0), df.cbirt90.fillna(0))
    p_hit = defcon_probability(rate, df.exp_minutes, thr.fillna(99))
    df["p_defcon"] = np.where(df.pos == "GKP", 0.0, p_hit * df.p60)
    df["pts_defcon"] = df.p_defcon * POINTS["defcon"]

    # ---- goalkeeper saves & conceded ---------------------------------------
    exp_saves = df.saves90.fillna(0) * mins90
    df["pts_saves"] = np.where(df.pos == "GKP", e_floor_div(exp_saves, 3), 0.0)
    conceded_pen = e_floor_div(df.team_xgc * mins90, 2)
    df["pts_conceded"] = np.where(df.pos.isin(["GKP", "DEF"]), -conceded_pen, 0.0)

    # ---- bonus: rank-based approximation ------------------------------------
    bps_proxy = (df.e_goals * 24 + df.e_assists * 18 + df.p_cs * df.p60 * 12
                 + df.p_defcon * 6 + mins90 * 6)
    df["_bps"] = bps_proxy
    df["bps_rank"] = df.groupby("opponent")["_bps"].rank(ascending=False, pct=True)
    df["pts_bonus"] = np.clip((1 - df.bps_rank) ** 6 * 9, 0, 3) * df.p60

    comp = ["pts_appearance", "pts_goals", "pts_assists", "pts_cs",
            "pts_defcon", "pts_saves", "pts_conceded", "pts_bonus"]
    df["xp"] = df[comp].sum(axis=1)

    # ---- preference-adjusted selection score --------------------------------
    bw = json.loads((ROOT / "blended_weights.json").read_text())
    routes = bw["blended_route_weights"]
    split = prefs["defender_team_split"]

    def route_w(pos, route, team_elo):
        if pos == "DEF" and route in split["strong_team"]:
            band = "strong_team" if team_elo >= split["elo_threshold"] else "weak_team"
            return split[band][route] / 10.0
        return routes.get(pos, {}).get(route, 5.0) / 10.0

    mapping = {
        "pts_goals": ("goals", "attacking_returns"),
        "pts_assists": ("assists", "attacking_returns"),
        "pts_cs": ("clean_sheet", "clean_sheet"),
        "pts_defcon": ("defcon", "defcon"),
        "pts_saves": ("saves", "saves"),
        "pts_bonus": ("bonus", "bonus"),
    }
    sel = df.pts_appearance + df.pts_conceded
    for col, (rname, dname) in mapping.items():
        w = np.array([
            route_w(p, dname if p == "DEF" else rname, e if pd.notna(e) else 1800)
            for p, e in zip(df.pos, df.team_elo)
        ])
        sel = sel + df[col] * (0.5 + w)     # weight 0..10 -> multiplier 0.5..1.5
    # ---- unknown-player caution --------------------------------------------
    # A tier-D player's xP comes from a positional average, not evidence. Let
    # it compete on equal terms and the optimiser fills the squad with unknown
    # cheap players from promoted clubs, which is exactly the failure mode a
    # confidence score exists to prevent.
    uc = prefs["unknown_player_caution"]
    floor = uc["confidence_floor"]
    conf = df.confidence.fillna(0).clip(0, 1)
    df["selection_score"] = sel * (floor + (1 - floor) * conf)

    # ---- minutes-risk gate ---------------------------------------------------
    mrt = prefs["minutes_risk_tolerance"]
    df["min_p_start"] = df.pos.map(lambda p: mrt.get(p, {}).get("min_p_start", 0.5))
    df["eligible"] = ((df.p_start_est >= df.min_p_start)
                      & (df.confidence >= uc["hard_min_confidence"]))
    # exceptional-data override for MID/FWD
    med = df.groupby("pos").xp.transform("median")
    override_ok = df.pos.map(lambda p: mrt.get(p, {}).get("exceptional_override", False))
    override_mult = df.pos.map(lambda p: mrt.get(p, {}).get("override_xp_multiple", 99))
    # The override lets a MID/FWD with exceptional projections through a weak
    # minutes record. It must NOT also bypass the confidence floor: a tier-D
    # player's "exceptional" xP is a positional average, not evidence, so
    # without this guard the bench refills with unknown promoted-club players.
    df.loc[override_ok & (df.xp >= med * override_mult)
           & (df.confidence >= uc["hard_min_confidence"]), "eligible"] = True

    df["gw"] = gw
    return df


if __name__ == "__main__":
    d = build(1)
    print(f"{len(d)} player-fixtures for GW1")
    cols = ["web_name", "pos", "team", "now_cost", "xp", "selection_score",
            "p_cs", "p_defcon", "p_start_est", "confidence", "eligible"]
    print(d.nlargest(15, "xp")[cols].round(3).to_string(index=False))
