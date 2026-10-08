"""
Squad optimiser. Mixed-integer linear program over the 15-man squad,
starting XI, captain and vice.

Objective, per your stated priority of rank over raw points:

    maximise  sum(selection_score) - lambda * ownership_risk

ownership_risk is asymmetric on purpose. NOT owning a highly-owned player is
penalised much harder than owning an unowned one, because that is where rank
is actually lost: if 60% of your rivals own a player who hauls and you don't,
you drop. Owning an unpopular player who blanks costs you almost nothing.

Captaincy uses the same logic. A differential captain must beat the template
captain by `captain_differential_xp_edge_required` xP before it is taken.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

import xp_model

ROOT = Path(__file__).resolve().parent

SQUAD = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET = 100.0
MAX_PER_CLUB = 3
BENCH_WEIGHT = 0.12   # bench still has value; stops the solver buying 4.0 fodder


def add_effective_ownership(d: pd.DataFrame) -> pd.DataFrame:
    """
    Effective ownership = ownership + captaincy share.

    Captaincy share isn't published in this dataset, so it is modelled: the
    field captains the highest-projected premium assets, concentrated heavily
    on one or two. A softmax over xP, weighted by ownership and restricted to
    realistic candidates, reproduces that shape. Shares sum to 1.0 because
    every manager captains exactly one player.

    This is the single biggest missing piece in the earlier version. A player
    on 73% ownership who is also captained by half the field has an effective
    ownership above 120% - not owning him is a large, one-sided rank risk, and
    raw ownership alone never expressed that.
    """
    d = d.copy()
    cand = (d.own >= 0.05) & (d.pos != "GKP")
    score = np.where(cand, d.xp.clip(lower=0) * (0.4 + d.own), 0.0)
    # temperature: lower = more concentrated on the obvious captain
    z = np.exp((score - score.max()) / 0.85) * cand
    d["capt_share"] = z / z.sum() if z.sum() > 0 else 0.0
    d["eo"] = d.own + d.capt_share

    # per-player points volatility; positional fallback for thin histories
    pos_sd = {"GKP": 2.75, "DEF": 3.17, "MID": 2.88, "FWD": 3.15}
    base = d.pos.map(pos_sd).fillna(3.0)
    # volatile players are those whose returns are lumpy: scale with xP
    d["sigma"] = (base * (0.6 + 0.5 * d.xp / max(d.xp.max(), 1e-6))).clip(1.0, 6.0)
    return d


def optimise(df: pd.DataFrame, budget: float = BUDGET,
             lam: float | None = None) -> dict:
    prefs = json.loads((ROOT / "preferences.json").read_text())
    tr = prefs["template_and_rank"]
    lam = tr["lambda_ownership_risk"] if lam is None else lam
    cap_edge = tr["captain_differential_xp_edge_required"]

    d = df[df.eligible].copy().reset_index(drop=True)
    d["now_cost"] = pd.to_numeric(d.now_cost, errors="coerce").fillna(99)
    d["own"] = pd.to_numeric(d.selected_by_percent, errors="coerce").fillna(0) / 100.0
    d = d[d.now_cost < 90].reset_index(drop=True)

    d = add_effective_ownership(d)

    n = len(d)
    idx = range(n)

    prob = pulp.LpProblem("fpl", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", idx, cat="Binary")
    start = pulp.LpVariable.dicts("start", idx, cat="Binary")
    capt = pulp.LpVariable.dicts("capt", idx, cat="Binary")

    # ---- objective ---------------------------------------------------------
    # Rank, not raw points. What matters is your holding of a player RELATIVE
    # to the field's effective ownership:
    #
    #     delta_i = (m_i - EO_i) * points_i
    #
    # where m_i = start_i + capt_i (0 benched/unowned, 1 starting, 2 captained)
    # and EO_i = ownership + captaincy share.
    #
    # In expectation the EO term is a constant and drops out - which is why the
    # old "ownership bonus" did almost nothing. EO matters through VARIANCE:
    # being far from the field on a volatile player is what actually moves rank.
    # Penalty is sigma_i^2 * (m_i - EO_i)^2.
    #
    # That looks quadratic but linearises exactly. capt_i <= start_i and both
    # are binary, so m_i^2 = start_i + 3*capt_i, giving
    #
    #     (m_i - EO_i)^2 = start_i(1 - 2EO_i) + capt_i(3 - 2EO_i) + EO_i^2
    #
    # and the EO_i^2 term is constant. For Haaland (EO ~1.2) the start
    # coefficient goes strongly negative, so the penalty turns into a large
    # reward for owning him - which is the intended behaviour.
    var_pen = []
    for i in idx:
        s2 = d.sigma[i] ** 2
        eo = d.eo[i]
        var_pen.append((s2 * (1 - 2 * eo), s2 * (3 - 2 * eo)))

    prob += (
        pulp.lpSum(d.selection_score[i] * start[i] for i in idx)
        + pulp.lpSum(BENCH_WEIGHT * d.selection_score[i] * (squad[i] - start[i])
                     for i in idx)
        + pulp.lpSum(d.selection_score[i] * capt[i] for i in idx)
        - lam * pulp.lpSum(var_pen[i][0] * start[i] + var_pen[i][1] * capt[i]
                           for i in idx)
    )

    # ---- structure ---------------------------------------------------------
    prob += pulp.lpSum(squad[i] for i in idx) == 15
    prob += pulp.lpSum(start[i] for i in idx) == 11
    prob += pulp.lpSum(capt[i] for i in idx) == 1
    prob += pulp.lpSum(d.now_cost[i] * squad[i] for i in idx) <= budget

    for pos, k in SQUAD.items():
        prob += pulp.lpSum(squad[i] for i in idx if d.pos[i] == pos) == k
    for pos in XI_MIN:
        sel = [i for i in idx if d.pos[i] == pos]
        prob += pulp.lpSum(start[i] for i in sel) >= XI_MIN[pos]
        prob += pulp.lpSum(start[i] for i in sel) <= XI_MAX[pos]

    for team in d.team.unique():
        sel = [i for i in idx if d.team[i] == team]
        prob += pulp.lpSum(squad[i] for i in sel) <= MAX_PER_CLUB

    # ---- template proximity -------------------------------------------------
    # Your priority: stay close to template, take risk sparingly. Without an
    # explicit constraint the budget alone pushes premium high-ownership assets
    # out, which is the opposite of template-following. The template is the 15
    # most-owned eligible players; the squad must overlap with all but
    # `max_template_deviation_players` of them.
    max_dev = tr.get("max_template_deviation_players")
    if max_dev is not None:
        template = d.nlargest(15, "own").index.tolist()
        prob += pulp.lpSum(squad[i] for i in template) >= 15 - max_dev

    for i in idx:
        prob += start[i] <= squad[i]
        prob += capt[i] <= start[i]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"solver status: {pulp.LpStatus[prob.status]}")

    d["in_squad"] = [squad[i].value() > 0.5 for i in idx]
    d["starting"] = [start[i].value() > 0.5 for i in idx]
    d["captain"] = [capt[i].value() > 0.5 for i in idx]

    sq = d[d.in_squad].copy()

    # vice = best non-captain starter
    starters = sq[sq.starting & ~sq.captain]
    vice_id = starters.nlargest(1, "xp").index[0] if len(starters) else None
    sq["vice"] = sq.index == vice_id

    # template captain check
    tmpl = sq[sq.starting].assign(
        popularity=lambda x: pd.to_numeric(x.selected_by_percent, errors="coerce").fillna(0))
    template_cap = tmpl.nlargest(1, "popularity")
    chosen_cap = sq[sq.captain]
    note = None
    if len(template_cap) and len(chosen_cap):
        tc, cc = template_cap.iloc[0], chosen_cap.iloc[0]
        if tc.web_name != cc.web_name and (cc.xp - tc.xp) < cap_edge:
            note = (f"captain switched to template pick {tc.web_name} "
                    f"({tc.popularity:.1f}% owned): {cc.web_name} led by only "
                    f"{cc.xp - tc.xp:.2f} xP, under the {cap_edge} threshold")
            sq["captain"] = sq.index == tc.name

    return {
        "squad": sq.sort_values(["starting", "pos", "xp"], ascending=[False, True, False]),
        "cost": float(sq.now_cost.sum()),
        "xi_xp": float(sq[sq.starting].xp.sum()
                       + sq[sq.captain].xp.sum()),
        "captain_note": note,
        "lambda": lam,
    }


if __name__ == "__main__":
    df = xp_model.build(1)
    r = optimise(df)
    s = r["squad"]
    print(f"cost {r['cost']:.1f}m   XI xP (capt doubled) {r['xi_xp']:.2f}")
    if r["captain_note"]:
        print("NOTE:", r["captain_note"])
    print()
    cols = ["web_name", "pos", "team", "now_cost", "xp", "selected_by_percent",
            "confidence", "starting", "captain"]
    print(s[cols].round(2).to_string(index=False))
