"""
FPL optimiser — web UI.

    pip install streamlit
    streamlit run app.py

Every dial in the sidebar maps to a value in preferences.json. Move one and
the squad re-solves live.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import optimizer
import xp_model

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="FPL Optimiser", layout="wide")

PITCH_CSS = """
<style>
.pitch{background:linear-gradient(180deg,#1b7a3d 0%,#166b34 100%);
 border-radius:14px;padding:18px 10px;margin-top:6px}
.row{display:flex;justify-content:center;gap:10px;margin:14px 0;flex-wrap:wrap}
.card{background:#fff;border-radius:9px;padding:7px 9px;min-width:104px;
 text-align:center;box-shadow:0 2px 5px rgba(0,0,0,.25)}
.card .nm{font-weight:700;font-size:12.5px;line-height:1.2}
.card .tm{color:#666;font-size:10.5px}
.card .xp{color:#0b6b2f;font-weight:700;font-size:12px;margin-top:2px}
.card .own{color:#888;font-size:10px}
.badge{display:inline-block;background:#111;color:#fff;border-radius:4px;
 padding:0 4px;font-size:10px;margin-left:3px}
.lowconf{border:2px solid #d9822b}
.bench{background:#eceff1;border-radius:10px;padding:10px;margin-top:10px}
</style>
"""


@st.cache_data(show_spinner=False)
def get_projection(gw: int) -> pd.DataFrame:
    return xp_model.build(gw)


def card(r) -> str:
    cls = "card lowconf" if r.confidence < 0.35 else "card"
    b = ""
    if getattr(r, "captain", False):
        b = '<span class="badge">C</span>'
    elif getattr(r, "vice", False):
        b = '<span class="badge">V</span>'
    own = pd.to_numeric(pd.Series([r.selected_by_percent]),
                        errors="coerce").fillna(0).iloc[0]
    return (f'<div class="{cls}"><div class="nm">{r.web_name}{b}</div>'
            f'<div class="tm">{r.team} · £{r.now_cost:.1f}m</div>'
            f'<div class="xp">{r.xp:.2f} xP</div>'
            f'<div class="own">{own:.0f}% owned</div></div>')


def main() -> None:
    st.title("FPL Optimiser")
    st.caption("Component xP model + integer programming, tuned to your stated "
               "priorities. Optimises expected rank, not raw points.")

    prefs = json.loads((ROOT / "preferences.json").read_text())

    with st.sidebar:
        st.header("Settings")
        gw = st.number_input("Gameweek", 1, 38, 1)
        budget = st.slider("Budget (£m)", 90.0, 105.0, 100.0, 0.5)

        st.subheader("Risk appetite")
        lam = st.slider(
            "Template adherence", 0.0, 1.5,
            float(prefs["template_and_rank"]["lambda_ownership_risk"]), 0.05,
            help="Higher = stay closer to the template. Protects rank, lowers "
                 "expected points. Lower = chase differentials.")
        max_dev = st.slider(
            "Max deviation from template (players)", 0, 15,
            int(prefs["template_and_rank"]["max_template_deviation_players"]),
            help="How many of the 15 most-owned players you're willing to skip.")

        st.subheader("Caution")
        floor = st.slider(
            "Trust in unknown players", 0.0, 1.0,
            float(prefs["unknown_player_caution"]["confidence_floor"]), 0.05,
            help="Lower = more sceptical of players with little data "
                 "(promoted clubs, new signings).")
        st.caption("Orange border = low confidence")

    df = get_projection(int(gw)).copy()

    # live overrides
    conf = df.confidence.fillna(0).clip(0, 1)
    base = df.selection_score / (
        prefs["unknown_player_caution"]["confidence_floor"]
        + (1 - prefs["unknown_player_caution"]["confidence_floor"]) * conf)
    df["selection_score"] = base * (floor + (1 - floor) * conf)

    prefs["template_and_rank"]["max_template_deviation_players"] = int(max_dev)
    (ROOT / "_runtime_prefs.json").write_text(json.dumps(prefs))

    try:
        res = optimizer.optimise(df, budget=float(budget), lam=float(lam))
    except Exception as e:  # noqa: BLE001
        st.error(f"No valid squad found: {e}")
        return

    sq = res["squad"]
    xi = sq[sq.starting]
    bench = sq[~sq.starting]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Squad cost", f"£{res['cost']:.1f}m")
    c2.metric("XI expected points", f"{res['xi_xp']:.1f}")
    c3.metric("Avg confidence", f"{sq.confidence.mean():.2f}")
    own = pd.to_numeric(sq.selected_by_percent, errors="coerce").fillna(0)
    c4.metric("Avg ownership", f"{own.mean():.0f}%")

    if res["captain_note"]:
        st.info(res["captain_note"])

    st.markdown(PITCH_CSS, unsafe_allow_html=True)
    html = ['<div class="pitch">']
    for pos in ("GKP", "DEF", "MID", "FWD"):
        row = xi[xi.pos == pos].sort_values("xp", ascending=False)
        if len(row):
            html.append('<div class="row">'
                        + "".join(card(r) for r in row.itertuples()) + "</div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    st.markdown("**Bench** (in order)")
    st.markdown('<div class="bench"><div class="row">'
                + "".join(card(r) for r in
                          bench.sort_values("xp", ascending=False).itertuples())
                + "</div></div>", unsafe_allow_html=True)

    with st.expander("Full squad detail"):
        cols = ["web_name", "pos", "team", "now_cost", "xp", "selection_score",
                "p_cs", "p_defcon", "p_start_est", "selected_by_percent",
                "confidence", "starting", "captain"]
        st.dataframe(sq[cols].round(3), use_container_width=True)

    with st.expander("Top projected players (all)"):
        top = df[df.eligible].nlargest(40, "xp")[
            ["web_name", "pos", "team", "now_cost", "xp", "p_cs", "p_defcon",
             "p_start_est", "selected_by_percent", "confidence"]]
        st.dataframe(top.round(3), use_container_width=True)

    st.caption("xP is the honest rules-based projection. The optimiser "
               "maximises a preference-adjusted score, so the highest-xP "
               "players are not always picked.")


if __name__ == "__main__":
    main()
