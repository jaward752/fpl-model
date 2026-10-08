"""Render the current optimal squad to a standalone HTML file."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

import optimizer
import xp_model

OUT = Path(__file__).resolve().parent / "best_team.html"

CSS = """
*{box-sizing:border-box}
body{margin:0;padding:28px;background:#f4f6f8;
 font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#16202b}
.wrap{max-width:1000px;margin:0 auto}
h1{margin:0 0 4px;font-size:26px}
.sub{color:#5a6b7c;font-size:13.5px;margin-bottom:18px}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.stat{background:#fff;border-radius:10px;padding:12px 16px;flex:1;min-width:150px;
 box-shadow:0 1px 3px rgba(0,0,0,.08)}
.stat .k{color:#6b7c8d;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.stat .v{font-size:21px;font-weight:700;margin-top:3px}
.pitch{background:linear-gradient(180deg,#1b7a3d,#13592a);border-radius:14px;padding:20px 12px}
.row{display:flex;justify-content:center;gap:10px;margin:16px 0;flex-wrap:wrap}
.card{background:#fff;border-radius:9px;padding:8px 10px;min-width:112px;text-align:center;
 box-shadow:0 2px 6px rgba(0,0,0,.28)}
.card .nm{font-weight:700;font-size:13px}
.card .tm{color:#66757f;font-size:10.5px;margin-top:1px}
.card .xp{color:#0b6b2f;font-weight:700;font-size:12.5px;margin-top:3px}
.card .own{color:#8a97a2;font-size:10px}
.badge{display:inline-block;background:#111;color:#fff;border-radius:4px;padding:0 5px;
 font-size:10px;margin-left:4px;vertical-align:middle}
.lowconf{border:2px solid #d9822b}
.bench{background:#e7ebee;border-radius:12px;padding:8px 12px;margin-top:14px}
.bench h3{margin:8px 0 0;font-size:13px;color:#55636f;text-align:center}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;
 overflow:hidden;margin-top:18px;font-size:13px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #eef1f4}
th{background:#f0f3f6;font-size:11px;text-transform:uppercase;color:#5a6b7c;letter-spacing:.4px}
tr:last-child td{border-bottom:none}
.note{background:#fff6e5;border-left:4px solid #d9822b;padding:11px 14px;
 border-radius:6px;margin:16px 0;font-size:13px}
.foot{color:#7b8a97;font-size:12px;margin-top:20px;line-height:1.55}
"""


def card(r) -> str:
    cls = "card lowconf" if r.confidence < 0.35 else "card"
    b = ('<span class="badge">C</span>' if r.captain else
         '<span class="badge">V</span>' if getattr(r, "vice", False) else "")
    own = pd.to_numeric(pd.Series([r.selected_by_percent]),
                        errors="coerce").fillna(0).iloc[0]
    return (f'<div class="{cls}"><div class="nm">{r.web_name}{b}</div>'
            f'<div class="tm">{r.team} &middot; &pound;{r.now_cost:.1f}m</div>'
            f'<div class="xp">{r.xp:.2f} xP</div>'
            f'<div class="own">{own:.0f}% owned</div></div>')


def main(gw: int = 1) -> None:
    df = xp_model.build(gw)
    res = optimizer.optimise(df)
    sq = res["squad"]
    xi, bench = sq[sq.starting], sq[~sq.starting]
    own = pd.to_numeric(sq.selected_by_percent, errors="coerce").fillna(0)

    rows = []
    for pos in ("GKP", "DEF", "MID", "FWD"):
        sel = xi[xi.pos == pos].sort_values("xp", ascending=False)
        if len(sel):
            rows.append('<div class="row">'
                        + "".join(card(r) for r in sel.itertuples()) + "</div>")

    tbl = ["<tr><th>Player</th><th>Pos</th><th>Team</th><th>Price</th><th>xP</th>"
           "<th>Clean sheet</th><th>DefCon</th><th>Start</th><th>Owned</th>"
           "<th>Confidence</th></tr>"]
    for r in sq.itertuples():
        o = pd.to_numeric(pd.Series([r.selected_by_percent]),
                          errors="coerce").fillna(0).iloc[0]
        tbl.append(
            f"<tr><td>{r.web_name}{' (C)' if r.captain else ''}</td><td>{r.pos}</td>"
            f"<td>{r.team}</td><td>&pound;{r.now_cost:.1f}m</td><td>{r.xp:.2f}</td>"
            f"<td>{r.p_cs:.0%}</td><td>{r.p_defcon:.0%}</td>"
            f"<td>{r.p_start_est:.0%}</td><td>{o:.0f}%</td>"
            f"<td>{r.confidence:.2f}</td></tr>")

    note = f'<div class="note">{res["captain_note"]}</div>' if res["captain_note"] else ""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>FPL Optimal Squad &mdash; GW{gw}</title><style>{CSS}</style></head><body>
<div class="wrap">
<h1>Optimal squad &mdash; Gameweek {gw}</h1>
<div class="sub">Component xP model with integer programming, tuned to your stated
priorities. Generated {dt.datetime.now():%d %b %Y %H:%M}.</div>
<div class="stats">
<div class="stat"><div class="k">Squad cost</div><div class="v">&pound;{res['cost']:.1f}m</div></div>
<div class="stat"><div class="k">XI expected points</div><div class="v">{res['xi_xp']:.1f}</div></div>
<div class="stat"><div class="k">Avg ownership</div><div class="v">{own.mean():.0f}%</div></div>
<div class="stat"><div class="k">Avg confidence</div><div class="v">{sq.confidence.mean():.2f}</div></div>
</div>
{note}
<div class="pitch">{''.join(rows)}</div>
<div class="bench"><h3>BENCH</h3><div class="row">
{''.join(card(r) for r in bench.sort_values('xp', ascending=False).itertuples())}
</div></div>
<table>{''.join(tbl)}</table>
<div class="foot">
<b>xP</b> is the honest rules-based projection: appearance, goals, assists,
clean sheets (Poisson), DefCon (threshold), saves, bonus and goals conceded.
The optimiser maximises a <i>preference-adjusted</i> score that also penalises
deviation from the template, so the highest-xP players are not always picked.<br>
An orange border marks low confidence &mdash; players with little or no
Premier League history, whose projections come largely from positional averages.
</div>
</div></body></html>"""
    OUT.write_text(html)
    print(f"wrote {OUT}  (cost {res['cost']:.1f}m, xP {res['xi_xp']:.1f})")


if __name__ == "__main__":
    main(1)
