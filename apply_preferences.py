"""
Blend your stated preferences with the weights fitted from 2025-26 data.

    python apply_preferences.py            # show blended weights
    python apply_preferences.py --alpha 0  # ignore preferences, pure data
    python apply_preferences.py --alpha 1  # ignore data, pure preferences

Writes blended_weights.json, which the xP model reads.

    final = alpha * your_weight + (1 - alpha) * fitted_weight

Fitted weights come from rank_features.py (--raw-only), mapped from individual
stats onto the scoring routes you ranked. Route mapping is explicit below so
you can see exactly which stats back each route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

import config as C

ROOT = Path(__file__).resolve().parent

# Which fitted features evidence each scoring route.
ROUTE_STATS = {
    "clean_sheet":       ["opponent_elo", "elo_diff", "team_elo",
                          "team_goals_conceded_avg", "team_goals_conceded_r4"],
    "defcon":            ["cbit_avg", "cbit_r4", "cbirt_avg", "cbirt_r4",
                          "defcon_rate", "tackles_avg", "interceptions_avg",
                          "clearances_avg", "recoveries_avg", "blocks_avg"],
    "attacking_returns": ["xg_avg", "xg_r4", "xa_avg", "xa_r4", "shots_avg",
                          "shots_on_target_avg", "chances_created_avg"],
    "goals":             ["xg_avg", "xg_r4", "shots_avg", "shots_on_target_avg",
                          "shots_on_target_r4", "touches_opposition_box_avg",
                          "big_chances_missed_avg"],
    "assists":           ["xa_avg", "xa_r4", "chances_created_avg",
                          "accurate_crosses_avg", "successful_dribbles_avg"],
    "saves":             ["saves_avg", "saves_r4", "xgot_faced_avg",
                          "xgot_faced_r4", "goals_prevented_avg",
                          "saves_inside_box_avg"],
    "bonus":             ["bps_prev_avg", "bps_prev_r4", "influence_prev_avg",
                          "duels_won_avg", "touches_avg"],
    "minutes_security":  ["minutes_avg", "minutes_r4", "p60_rate",
                          "starts_prev_avg", "starts_prev_r4"],
    "set_piece_threat":  ["aerial_duels_won_avg", "headed_clearances_avg",
                          "accurate_crosses_avg"],
    "consistency":       ["goals_avg", "assists_avg", "xg_r4", "xa_r4"],
    "penalty_saves":     [],
}


def fitted_route_weights(csv: Path) -> dict:
    """Collapse per-stat fitted importance onto routes, per position."""
    if not csv.exists():
        print(f"! {csv.name} not found — run rank_features.py --raw-only first")
        return {}
    df = pd.read_csv(csv)
    out: dict[str, dict[str, float]] = {}
    for pos, g in df.groupby("pos"):
        lookup = dict(zip(g.feature, g.score))
        routes = {}
        for route, stats in ROUTE_STATS.items():
            vals = [lookup[s] for s in stats if s in lookup]
            routes[route] = float(sum(sorted(vals, reverse=True)[:4]) / 4) if vals else 0.0
        mx = max(routes.values()) or 1.0
        out[pos] = {k: round(10 * v / mx, 2) for k, v in routes.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()

    prefs = json.loads((ROOT / "preferences.json").read_text())
    alpha = args.alpha if args.alpha is not None else prefs["alpha"]
    fitted = fitted_route_weights(ROOT / "feature_importance.csv")

    print(f"alpha = {alpha}  "
          f"({int(alpha*100)}% your judgement, {int((1-alpha)*100)}% fitted data)\n")

    blended = {}
    for pos, user_w in prefs["positions"].items():
        user_w = {k: v for k, v in user_w.items() if not k.startswith("_")}
        fit_w = fitted.get(pos, {})
        rows = []
        out = {}
        for route, uw in user_w.items():
            fw = fit_w.get(route, 0.0)
            final = alpha * uw + (1 - alpha) * fw
            out[route] = round(final, 2)
            rows.append((route, uw, round(fw, 2), round(final, 2)))
        blended[pos] = out
        print(f"===== {pos} =====")
        t = pd.DataFrame(rows, columns=["route", "yours", "fitted", "blended"])
        print(t.sort_values("blended", ascending=False).to_string(index=False))
        print()

    payload = {
        "alpha": alpha,
        "blended_route_weights": blended,
        "fitted_route_weights": fitted,
        "passthrough": {k: prefs[k] for k in (
            "defender_team_split", "hard_constraints", "minutes_risk_tolerance",
            "set_pieces", "midfielder_archetypes", "forward_logic",
            "template_and_rank") if k in prefs},
    }
    (ROOT / "blended_weights.json").write_text(json.dumps(payload, indent=2))
    print("wrote blended_weights.json")


if __name__ == "__main__":
    main()
