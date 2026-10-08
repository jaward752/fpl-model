"""
FPL data ingestion pipeline.

Downloads the FPL-Core-Insights dataset, corrects two documented data traps,
and loads everything into a single DuckDB file ready for modelling.

    python ingest.py              # incremental (skips cached files)
    python ingest.py --refresh    # re-download everything

Data traps handled here (see README.md for detail):
  1. matches.home_team / away_team join on teams.CODE, not teams.ID, despite
     what the upstream README says. Joining on id silently mislabels teams.
  2. matches.csv mixes competitions. Non-Premier-League opponents have a NULL
     team reference. Rows are tagged with a competition so PL-only modelling
     is a filter rather than a guess.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import duckdb
import pandas as pd
import requests

import config as C


# ------------------------------------------------------------------ download

def _fetch(url: str, dest) -> bool:
    """Download url to dest. Returns False on 404 (expected for future GWs)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for attempt in range(C.RETRIES):
        try:
            r = requests.get(url, timeout=C.TIMEOUT)
            if r.status_code == 404:
                return False
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            return True
        except requests.RequestException:
            if attempt == C.RETRIES - 1:
                print(f"  ! failed: {url}", file=sys.stderr)
                return False
            time.sleep(2 ** attempt)
    return False


def _jobs(season: str):
    """(url, local_path) pairs for one season."""
    out = []
    for name in ("players", "teams", "playerstats", "gameweek_summaries"):
        out.append((
            f"{C.BASE_URL}/{season}/{name}.csv",
            C.RAW_DIR / season / f"{name}.csv",
        ))
    for gw in range(1, C.MAX_GAMEWEEK + 1):
        folder = quote(f"By Gameweek/GW{gw}")
        for name in ("playermatchstats", "matches", "player_gameweek_stats"):
            out.append((
                f"{C.BASE_URL}/{season}/{folder}/{name}.csv",
                C.RAW_DIR / season / f"gw{gw:02d}" / f"{name}.csv",
            ))
    # Pre-season friendlies. NB: these live under "By Tournament/Friendlies/GW0",
    # NOT "By Gameweek/GW0" (which 404s). Minutes coverage is ~95% but xG
    # coverage is ~4%, so this is a MINUTES source only - see README.
    fr = quote("By Tournament/Friendlies/GW0")
    for name in ("playermatchstats", "matches"):
        out.append((
            f"{C.BASE_URL}/{season}/{fr}/{name}.csv",
            C.RAW_DIR / season / "friendlies" / f"{name}.csv",
        ))
    return out


def download(seasons) -> None:
    for season in seasons:
        jobs = _jobs(season)
        print(f"[download] {season}: {len(jobs)} files")
        with ThreadPoolExecutor(max_workers=C.MAX_WORKERS) as pool:
            got = list(pool.map(lambda j: _fetch(*j), jobs))
        print(f"[download] {season}: {sum(got)}/{len(jobs)} available")


# ------------------------------------------------------------------- loading

def _read_all(season: str, filename: str, gw_col: bool = True) -> pd.DataFrame:
    """Concatenate one filename across every gameweek folder of a season."""
    frames = []
    for gw in range(1, C.MAX_GAMEWEEK + 1):
        p = C.RAW_DIR / season / f"gw{gw:02d}" / f"{filename}.csv"
        if not p.exists() or p.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        if df.empty:
            continue
        if gw_col:
            df["source_gw"] = gw
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_master(season: str, name: str) -> pd.DataFrame:
    p = C.RAW_DIR / season / f"{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def _competition(match_id) -> str:
    """Derive competition from the match_id slug."""
    if not isinstance(match_id, str):
        return "unknown"
    s = match_id.lower()
    for key, label in (
        # NB: the upstream slug for the league is "prem", not "premier-league".
        # Check the longer competition names first so "prem" cannot shadow them.
        ("champions-league", "Champions League"),
        ("europa-league", "Europa League"),
        ("conference-league", "Conference League"),
        ("fa-cup", "FA Cup"),
        ("efl-cup", "EFL Cup"),
        ("league-cup", "EFL Cup"),
        ("friendly", "Friendly"),
        ("club-friendlies", "Friendly"),
        ("prem", "Premier League"),
    ):
        if key in s:
            return label
    return "other"


def build(seasons) -> None:
    C.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(C.DB_PATH))

    teams, players, matches, pms, pgs, gws = [], [], [], [], [], []
    friendlies = []

    for season in seasons:
        print(f"[load] {season}")

        t = _read_master(season, "teams")
        if not t.empty:
            t["season"] = season
            teams.append(t)

        p = _read_master(season, "players")
        if not p.empty:
            p["season"] = season
            p["pos"] = p["position"].map(C.POSITION_MAP).fillna("UNK")
            players.append(p)

        g = _read_master(season, "gameweek_summaries")
        if not g.empty:
            g["season"] = season
            gws.append(g)

        m = _read_all(season, "matches")
        if not m.empty:
            m["season"] = season
            m["competition"] = m["match_id"].map(_competition)
            # TRAP 2: a NULL team reference means a non-PL opponent.
            m["is_pl"] = (
                (m["competition"] == "Premier League")
                & m["home_team"].notna()
                & m["away_team"].notna()
            )
            m = m.drop_duplicates(subset=["match_id"], keep="last")
            matches.append(m)

        s = _read_all(season, "playermatchstats")
        if not s.empty:
            s["season"] = season
            s = s.drop_duplicates(subset=["player_id", "match_id"], keep="last")
            pms.append(s)

        w = _read_all(season, "player_gameweek_stats", gw_col=False)
        if not w.empty:
            w["season"] = season
            if "gw" in w.columns:
                w = w.drop_duplicates(subset=["id", "gw"], keep="last")
            pgs.append(w)

        # pre-season friendlies (minutes signal only)
        fp = C.RAW_DIR / season / "friendlies" / "playermatchstats.csv"
        if fp.exists() and fp.stat().st_size > 0:
            fr = pd.read_csv(fp, low_memory=False)
            if not fr.empty:
                fr["season"] = season
                fr = fr.drop_duplicates(subset=["player_id", "match_id"], keep="last")
                friendlies.append(fr)

    def _reg(name: str, frames):
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        con.register(f"_{name}", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _{name}")
        print(f"  {name:22s} {len(df):>7,} rows")
        return df

    _reg("teams", teams)
    _reg("players", players)
    _reg("matches", matches)
    _reg("player_match_stats", pms)
    _reg("player_gw_stats", pgs)
    _reg("gameweeks", gws)
    _reg("friendly_stats", friendlies)

    _create_views(con)
    con.close()


# --------------------------------------------------------------------- views

def _create_views(con) -> None:
    """Analysis-ready views with DefCon fields and correct team joins."""

    # Drop the legacy UNION ALL view. It produced wrong GROUP BY results (see
    # the team_match comment below) and was replaced by a materialised table.
    # CREATE OR REPLACE TABLE does not remove a same-named view, so a database
    # built by an older revision keeps the broken view alongside the fixed
    # table unless it is explicitly dropped here.
    con.execute("DROP VIEW IF EXISTS v_team_match")

    # TRAP 1: matches join teams on CODE, not ID.
    con.execute("""
        CREATE OR REPLACE VIEW v_matches AS
        SELECT
            m.match_id, m.season, m.gameweek, m.kickoff_time,
            m.competition, m.is_pl, m.finished,
            m.home_team AS home_code, m.away_team AS away_code,
            th.id   AS home_team_id, th.name AS home_team,
            ta.id   AS away_team_id, ta.name AS away_team,
            m.home_score, m.away_score,
            m.home_team_elo, m.away_team_elo,
            m.home_expected_goals_xg AS home_xg,
            m.away_expected_goals_xg AS away_xg
        FROM matches m
        LEFT JOIN teams th ON th.code = m.home_team AND th.season = m.season
        LEFT JOIN teams ta ON ta.code = m.away_team AND ta.season = m.season
    """)

    # The modelling table: one row per player per match, with CBIT/CBIRT
    # computed and the DefCon threshold evaluated as a boolean outcome.
    con.execute(f"""
        CREATE OR REPLACE VIEW v_player_match AS
        SELECT
            s.player_id, s.match_id, s.season,
            p.web_name, p.pos, p.team_code,
            m.gameweek, m.competition, m.is_pl, m.kickoff_time,
            CASE WHEN m.home_code = p.team_code THEN true ELSE false END AS is_home,
            CASE WHEN m.home_code = p.team_code
                 THEN m.away_team_elo ELSE m.home_team_elo END AS opponent_elo,
            CASE WHEN m.home_code = p.team_code
                 THEN m.home_team_elo ELSE m.away_team_elo END AS team_elo,

            COALESCE(s.minutes_played, 0)                       AS minutes,
            COALESCE(s.goals, 0)                                AS goals,
            COALESCE(s.assists, 0)                              AS assists,
            COALESCE(s.xg, 0)                                   AS xg,
            COALESCE(s.xa, 0)                                   AS xa,
            COALESCE(s.total_shots, 0)                          AS shots,
            COALESCE(s.chances_created, 0)                      AS chances_created,
            COALESCE(s.saves, 0)                                AS saves,
            COALESCE(s.xgot_faced, 0)                           AS xgot_faced,
            COALESCE(s.goals_prevented, 0)                      AS goals_prevented,
            COALESCE(s.team_goals_conceded, 0)                  AS team_goals_conceded,

            COALESCE(s.tackles, 0)        AS tackles,
            COALESCE(s.interceptions, 0)  AS interceptions,
            COALESCE(s.blocks, 0)         AS blocks,
            COALESCE(s.clearances, 0)     AS clearances,
            COALESCE(s.recoveries, 0)     AS recoveries,

            COALESCE(s.tackles,0) + COALESCE(s.interceptions,0)
              + COALESCE(s.blocks,0) + COALESCE(s.clearances,0)      AS cbit,
            COALESCE(s.tackles,0) + COALESCE(s.interceptions,0)
              + COALESCE(s.blocks,0) + COALESCE(s.clearances,0)
              + COALESCE(s.recoveries,0)                             AS cbirt,

            -- DefCon is a CAPPED THRESHOLD event, not a linear rate.
            CASE
              WHEN p.pos = 'DEF' AND (COALESCE(s.tackles,0)+COALESCE(s.interceptions,0)
                   +COALESCE(s.blocks,0)+COALESCE(s.clearances,0)) >= {C.DEFCON_THRESHOLD['DEF']}
                THEN true
              WHEN p.pos IN ('MID','FWD') AND (COALESCE(s.tackles,0)+COALESCE(s.interceptions,0)
                   +COALESCE(s.blocks,0)+COALESCE(s.clearances,0)+COALESCE(s.recoveries,0))
                   >= {C.DEFCON_THRESHOLD['MID']}
                THEN true
              ELSE false
            END AS hit_defcon,

            CASE
              WHEN p.pos = 'DEF' AND (COALESCE(s.tackles,0)+COALESCE(s.interceptions,0)
                   +COALESCE(s.blocks,0)+COALESCE(s.clearances,0)) >= {C.DEFCON_THRESHOLD['DEF']}
                THEN {C.DEFCON_POINTS}
              WHEN p.pos IN ('MID','FWD') AND (COALESCE(s.tackles,0)+COALESCE(s.interceptions,0)
                   +COALESCE(s.blocks,0)+COALESCE(s.clearances,0)+COALESCE(s.recoveries,0))
                   >= {C.DEFCON_THRESHOLD['MID']}
                THEN {C.DEFCON_POINTS}
              ELSE 0
            END AS defcon_points
        FROM player_match_stats s
        JOIN players p ON p.player_id = s.player_id AND p.season = s.season
        LEFT JOIN v_matches m ON m.match_id = s.match_id AND m.season = s.season
    """)

    # Per-team, per-match attack/defence record. Feeds the Poisson clean-sheet
    # model that replaces the sigmoid approximation used by other solvers.
    #
    # MATERIALISED AS A TABLE, NOT A VIEW, DELIBERATELY.
    # DuckDB 1.5.5 mis-resolves the grouping key when you GROUP BY over a
    # UNION ALL *view*: the key is taken from the first branch for both
    # branches, so grouping by team silently collects only that team's home
    # fixtures (counted twice), and goals_for comes out equal to goals_against
    # for every team. Filtering to a single team masks the bug via predicate
    # pushdown, which makes it very easy to miss. Materialising sidesteps it
    # entirely and is faster for repeated model fitting.
    con.execute("""
        CREATE OR REPLACE TABLE team_match AS
        SELECT match_id, season, gameweek, competition, is_pl, kickoff_time,
               home_team_id AS team_id, home_team AS team, away_team AS opponent,
               true AS is_home, home_score AS goals_for, away_score AS goals_against,
               home_xg AS xg_for, away_xg AS xg_against,
               home_team_elo AS elo, away_team_elo AS opponent_elo,
               CASE WHEN away_score = 0 THEN 1 ELSE 0 END AS clean_sheet
        FROM v_matches
        UNION ALL
        SELECT match_id, season, gameweek, competition, is_pl, kickoff_time,
               away_team_id AS team_id, away_team AS team, home_team AS opponent,
               false AS is_home, away_score AS goals_for, home_score AS goals_against,
               away_xg AS xg_for, home_xg AS xg_against,
               away_team_elo AS elo, home_team_elo AS opponent_elo,
               CASE WHEN home_score = 0 THEN 1 ELSE 0 END AS clean_sheet
        FROM v_matches
    """)

    # Per-player DefCon base rate — the empirical prior for the threshold model.
    con.execute("""
        CREATE OR REPLACE VIEW v_defcon_rates AS
        SELECT season, player_id, web_name, pos,
               COUNT(*)                              AS matches,
               SUM(minutes)                          AS minutes,
               AVG(CASE WHEN pos='DEF' THEN cbit ELSE cbirt END) AS mean_actions,
               AVG(CASE WHEN hit_defcon THEN 1.0 ELSE 0.0 END)   AS defcon_hit_rate,
               SUM(defcon_points)                    AS defcon_points_total
        FROM v_player_match
        WHERE is_pl AND minutes >= 60 AND pos <> 'GKP'
        GROUP BY 1,2,3,4
        HAVING COUNT(*) >= 3
    """)

    _build_player_prior(con)
    print("  views: v_matches, v_player_match, v_defcon_rates, v_pl_history, "
          "v_preseason | tables: team_match, player_prior")


def _build_player_prior(con) -> None:
    """
    Give EVERY player in the current season a usable estimate, tagged by how
    much we actually know about them.

    Tiers, best first:
      A  >=450 Premier League minutes last season  -> own rates, high confidence
      A- <450  Premier League minutes last season  -> own rates, shrunk hard
      C  no PL history but pre-season minutes      -> prior rates, real minutes
      D  nothing at all                            -> prior rates, prior minutes

    Tier B (previous non-PL club via FBref) is not implemented: the source data
    does not exist in this dataset. Players who would be tier B currently fall
    through to C or D. See README.

    Rates are stored as SHARE OF TEAM OUTPUT as well as per-90, because per-90
    does not survive a transfer: 0.45 xG/90 in a weak side and 0.45 xG/90 in a
    strong side are different players. Project a mover as
    share x new_team_expected_xG, not as his old per-90.
    """

    # ---- last season's Premier League record, keyed by player_code ----------
    con.execute("""
        CREATE OR REPLACE VIEW v_pl_history AS
        WITH team_tot AS (
            SELECT pm.season, p.team_code,
                   SUM(pm.xg) AS team_xg, SUM(pm.xa) AS team_xa
            FROM v_player_match pm
            JOIN players p ON p.player_id = pm.player_id AND p.season = pm.season
            WHERE pm.is_pl
            GROUP BY 1,2
        )
        SELECT
            p.player_code, pm.season, p.web_name, pm.pos, p.team_code,
            COUNT(*)                                   AS apps,
            SUM(pm.minutes)                            AS minutes,
            SUM(CASE WHEN pm.minutes >= 60 THEN 1 ELSE 0 END) AS starts_60,
            90.0 * SUM(pm.xg)  / NULLIF(SUM(pm.minutes),0) AS xg90,
            90.0 * SUM(pm.xa)  / NULLIF(SUM(pm.minutes),0) AS xa90,
            90.0 * SUM(pm.cbit)  / NULLIF(SUM(pm.minutes),0) AS cbit90,
            90.0 * SUM(pm.cbirt) / NULLIF(SUM(pm.minutes),0) AS cbirt90,
            90.0 * SUM(pm.saves) / NULLIF(SUM(pm.minutes),0) AS saves90,
            AVG(CASE WHEN pm.minutes >= 60 AND pm.hit_defcon THEN 1.0
                     WHEN pm.minutes >= 60 THEN 0.0 END)     AS defcon_hit_rate,
            -- transfer-safe: fraction of the old club's output he accounted for
            SUM(pm.xg) / NULLIF(MAX(t.team_xg),0)            AS share_team_xg,
            SUM(pm.xa) / NULLIF(MAX(t.team_xa),0)            AS share_team_xa
        FROM v_player_match pm
        JOIN players p ON p.player_id = pm.player_id AND p.season = pm.season
        LEFT JOIN team_tot t ON t.season = pm.season AND t.team_code = p.team_code
        WHERE pm.is_pl
        GROUP BY 1,2,3,4,5
    """)

    # ---- pre-season minutes signal -----------------------------------------
    con.execute("""
        CREATE OR REPLACE VIEW v_preseason AS
        SELECT p.player_code, f.season,
               COUNT(*)                                            AS friendly_apps,
               SUM(COALESCE(f.minutes_played,0))                    AS friendly_mins,
               SUM(CASE WHEN COALESCE(f.minutes_played,0) >= 60
                        THEN 1 ELSE 0 END)                          AS friendly_starts,
               MAX(COALESCE(f.minutes_played,0))                    AS friendly_max_mins
        FROM friendly_stats f
        JOIN players p ON p.player_id = f.player_id AND p.season = f.season
        GROUP BY 1,2
    """)

    # ---- positional priors, conditioned on nothing (league-wide averages) ---
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pos_prior AS
        SELECT pos,
               90.0*SUM(xg)/NULLIF(SUM(minutes),0)    AS xg90,
               90.0*SUM(xa)/NULLIF(SUM(minutes),0)    AS xa90,
               90.0*SUM(cbit)/NULLIF(SUM(minutes),0)  AS cbit90,
               90.0*SUM(cbirt)/NULLIF(SUM(minutes),0) AS cbirt90,
               90.0*SUM(saves)/NULLIF(SUM(minutes),0) AS saves90,
               AVG(CASE WHEN minutes>=60 AND hit_defcon THEN 1.0
                        WHEN minutes>=60 THEN 0.0 END) AS defcon_hit_rate,
               AVG(CASE WHEN minutes>=60 THEN 1.0 ELSE 0.0 END) AS p60
        FROM v_player_match WHERE is_pl GROUP BY 1
    """)

    # ---- one row per player in the current season --------------------------
    con.execute(f"""
        CREATE OR REPLACE TABLE player_prior AS
        WITH cur AS (
            SELECT p.season, p.player_id, p.player_code, p.web_name, p.pos,
                   t.name AS team, t.id AS team_id
            FROM players p
            JOIN teams t ON t.code = p.team_code AND t.season = p.season
            WHERE p.season = '{C.SEASONS[-1]}'
        ),
        elo AS (
            SELECT team, AVG(elo) AS team_elo
            FROM team_match WHERE season = '{C.SEASONS[-1]}' AND is_pl
            GROUP BY 1
        ),
        h AS (SELECT * FROM v_pl_history WHERE season = '{C.SEASONS[0]}'),
        f AS (SELECT * FROM v_preseason  WHERE season = '{C.SEASONS[-1]}')
        SELECT
            c.season, c.player_id, c.player_code, c.web_name, c.pos, c.team,
            e.team_elo,
            COALESCE(h.minutes, 0)        AS pl_minutes_prev,
            COALESCE(h.apps, 0)           AS pl_apps_prev,
            h.team_code                   AS prev_team_code,
            COALESCE(f.friendly_mins, 0)  AS preseason_mins,
            COALESCE(f.friendly_starts,0) AS preseason_starts,

            CASE
              WHEN COALESCE(h.minutes,0) >= 450 THEN 'A'
              WHEN COALESCE(h.minutes,0) > 0    THEN 'A-'
              WHEN COALESCE(f.friendly_mins,0) > 0 THEN 'C'
              ELSE 'D'
            END AS tier,

            -- shrinkage weight on the player's own data: 0 at no minutes,
            -- ->1 as minutes accumulate. 900 mins = half weight.
            COALESCE(h.minutes,0) / (COALESCE(h.minutes,0) + 900.0) AS w_own,

            -- blended rate estimates (own data shrunk toward positional prior)
            (COALESCE(h.xg90, pp.xg90) * (COALESCE(h.minutes,0)/(COALESCE(h.minutes,0)+900.0))
              + pp.xg90 * (1 - COALESCE(h.minutes,0)/(COALESCE(h.minutes,0)+900.0))) AS xg90,
            (COALESCE(h.xa90, pp.xa90) * (COALESCE(h.minutes,0)/(COALESCE(h.minutes,0)+900.0))
              + pp.xa90 * (1 - COALESCE(h.minutes,0)/(COALESCE(h.minutes,0)+900.0))) AS xa90,
            (COALESCE(h.cbit90, pp.cbit90) * (COALESCE(h.minutes,0)/(COALESCE(h.minutes,0)+900.0))
              + pp.cbit90 * (1 - COALESCE(h.minutes,0)/(COALESCE(h.minutes,0)+900.0))) AS cbit90,
            (COALESCE(h.cbirt90, pp.cbirt90) * (COALESCE(h.minutes,0)/(COALESCE(h.minutes,0)+900.0))
              + pp.cbirt90 * (1 - COALESCE(h.minutes,0)/(COALESCE(h.minutes,0)+900.0))) AS cbirt90,
            (COALESCE(h.saves90, pp.saves90) * (COALESCE(h.minutes,0)/(COALESCE(h.minutes,0)+900.0))
              + pp.saves90 * (1 - COALESCE(h.minutes,0)/(COALESCE(h.minutes,0)+900.0))) AS saves90,
            (COALESCE(h.defcon_hit_rate, pp.defcon_hit_rate)
                * (COALESCE(h.minutes,0)/(COALESCE(h.minutes,0)+900.0))
              + pp.defcon_hit_rate * (1 - COALESCE(h.minutes,0)/(COALESCE(h.minutes,0)+900.0)))
              AS defcon_hit_rate,

            -- transfer-safe shares (NULL when he has no PL history)
            h.share_team_xg, h.share_team_xa,
            CASE WHEN h.team_code IS NOT NULL AND h.team_code <> (
                     SELECT team_code FROM players p2
                     WHERE p2.player_code = c.player_code AND p2.season = c.season LIMIT 1)
                 THEN true ELSE false END AS changed_club,

            -- crude starting-minutes signal, best available source
            CASE
              WHEN COALESCE(h.apps,0) > 0
                THEN 1.0 * h.starts_60 / h.apps
              WHEN COALESCE(f.friendly_mins,0) > 0 AND COALESCE(f.friendly_apps,0) > 0
                THEN 1.0 * f.friendly_starts / f.friendly_apps
              ELSE pp.p60
            END AS p_start_est,

            CASE
              WHEN COALESCE(h.apps,0) > 0 THEN 'pl_history'
              WHEN COALESCE(f.friendly_mins,0) > 0 THEN 'preseason'
              ELSE 'positional_prior'
            END AS minutes_source,

            -- 0..1 confidence, for surfacing uncertainty in the UI
            ROUND(LEAST(1.0,
                  0.75 * (COALESCE(h.minutes,0)/(COALESCE(h.minutes,0)+900.0))
                + 0.25 * LEAST(1.0, COALESCE(f.friendly_mins,0)/270.0)), 3) AS confidence
        FROM cur c
        LEFT JOIN h  ON h.player_code = c.player_code
        LEFT JOIN f  ON f.player_code = c.player_code
        LEFT JOIN elo e ON e.team = c.team
        LEFT JOIN pos_prior pp ON pp.pos = c.pos
    """)


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="delete cached CSVs and re-download")
    ap.add_argument("--seasons", nargs="*", default=C.SEASONS)
    args = ap.parse_args()

    if args.refresh and C.RAW_DIR.exists():
        import shutil
        shutil.rmtree(C.RAW_DIR)

    download(args.seasons)
    build(args.seasons)
    print(f"\nDone -> {C.DB_PATH}")


if __name__ == "__main__":
    main()
