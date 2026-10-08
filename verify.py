"""
Sanity checks on the ingested database.

Every check here exists because it catches a specific, silent failure mode.
Run after each ingest:  python verify.py
"""

import duckdb
import config as C

con = duckdb.connect(str(C.DB_PATH), read_only=True)


def q(sql):
    return con.execute(sql).df()


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    return ok


print("\n=== 1. Table row counts ===")
print(q("""
    SELECT 'teams' t, season, COUNT(*) n FROM teams GROUP BY 2
    UNION ALL SELECT 'players', season, COUNT(*) FROM players GROUP BY 2
    UNION ALL SELECT 'matches', season, COUNT(*) FROM matches GROUP BY 2
    UNION ALL SELECT 'player_match_stats', season, COUNT(*) FROM player_match_stats GROUP BY 2
    ORDER BY 1,2
""").to_string(index=False))

print("\n=== 2. Team join integrity (TRAP 1) ===")
# If matches were joined on id instead of code, PL fixtures would show NULL names.
r = q("""
    SELECT COUNT(*) total,
           SUM(CASE WHEN home_team IS NULL OR away_team IS NULL THEN 1 ELSE 0 END) unresolved
    FROM v_matches WHERE is_pl
""").iloc[0]
check("all Premier League fixtures resolve to named teams",
      r.unresolved == 0, f"{int(r.unresolved)}/{int(r.total)} unresolved")

print("\n=== 3. Competition split (TRAP 2) ===")
print(q("""
    SELECT season, competition, COUNT(*) matches
    FROM v_matches GROUP BY 1,2 ORDER BY 1, 3 DESC
""").to_string(index=False))

print("\n=== 4. Premier League fixture count ===")
r = q("SELECT COUNT(*) n FROM v_matches WHERE is_pl AND season='2025-2026'").iloc[0]
check("2025-26 has 380 league fixtures", r.n == 380, f"got {int(r.n)}")

print("\n=== 4b. Team-match view integrity (UNION branch aliasing) ===")
# If the second UNION branch is unaliased, GROUP BY resolves the key to the
# first branch's column and every team's goals_for equals its goals_against.
t = q("""
    SELECT team, COUNT(*) AS n, COUNT(DISTINCT match_id) AS uniq,
           SUM(goals_for) AS gf, SUM(goals_against) AS ga,
           SUM(CASE WHEN is_home THEN 1 ELSE 0 END) AS home_n
    FROM team_match WHERE is_pl AND season='2025-2026'
    GROUP BY team
""")
check("20 teams present", len(t) == 20, f"got {len(t)}")
check("each team has 38 matches", (t.n == 38).all())
check("each team has 38 distinct fixtures", (t.uniq == 38).all(),
      "collapsed to 19 => UNION branch aliasing bug")
check("each team has 19 home fixtures", (t.home_n == 19).all())
check("goals_for != goals_against for most teams",
      (t.gf != t.ga).sum() >= 18, "all-equal => UNION branch aliasing bug")
check("league goals_for total == goals_against total",
      t.gf.sum() == t.ga.sum(), f"{t.gf.sum():.0f} vs {t.ga.sum():.0f}")
print(t.sort_values("gf", ascending=False).head(5).to_string(index=False))

print("\n=== 5. DefCon threshold model vs linear approximation ===")
print(q("""
    SELECT pos,
           COUNT(*)                                  AS player_matches,
           ROUND(AVG(CASE WHEN pos='DEF' THEN cbit ELSE cbirt END), 2) AS mean_actions,
           ROUND(AVG(CASE WHEN hit_defcon THEN 1.0 ELSE 0 END), 3)     AS p_hit,
           ROUND(AVG(defcon_points), 3)              AS true_xpts,
           ROUND(AVG(CASE WHEN pos='DEF' THEN cbit/10.0*2
                          ELSE cbirt/12.0*2 END), 3) AS linear_xpts
    FROM v_player_match
    WHERE is_pl AND minutes >= 60 AND pos <> 'GKP' AND season='2025-2026'
    GROUP BY 1 ORDER BY 1
""").to_string(index=False))
print("  -> linear_xpts is the fpl-solver approach; the gap is its overvaluation.")

print("\n=== 6. Top DefCon players 2025-26 (validation against known names) ===")
print(q("""
    SELECT web_name, pos, matches, ROUND(defcon_hit_rate,2) hit_rate,
           ROUND(mean_actions,1) mean_actions, defcon_points_total pts
    FROM v_defcon_rates
    WHERE season='2025-2026'
    ORDER BY defcon_points_total DESC LIMIT 10
""").to_string(index=False))

print("\n=== 7. Clean sheet base rate (Poisson sanity) ===")
r = q("""
    SELECT ROUND(AVG(clean_sheet),3) cs_rate,
           ROUND(AVG(goals_against),3) mean_ga,
           ROUND(EXP(-AVG(goals_against)),3) poisson_pred
    FROM team_match WHERE is_pl AND season='2025-2026' AND goals_against IS NOT NULL
""").iloc[0]
print(f"  actual CS rate {r.cs_rate}   mean GA {r.mean_ga}   Poisson exp(-lambda) {r.poisson_pred}")
sig = 1 / (1 + pow(2.718281828, r.mean_ga - 1.0))
print(f"  fpl-solver sigmoid would predict {sig:.3f}  <- error vs actual: {sig - r.cs_rate:+.3f}")

print("\n=== 8. Minutes distribution (for the multinomial minutes model) ===")
print(q("""
    SELECT pos,
           ROUND(AVG(CASE WHEN minutes=0 THEN 1.0 ELSE 0 END),3)               AS p_zero,
           ROUND(AVG(CASE WHEN minutes BETWEEN 1 AND 59 THEN 1.0 ELSE 0 END),3) AS p_1_59,
           ROUND(AVG(CASE WHEN minutes>=60 THEN 1.0 ELSE 0 END),3)              AS p_60plus
    FROM v_player_match WHERE is_pl AND season='2025-2026'
    GROUP BY 1 ORDER BY 1
""").to_string(index=False))

print("\n=== 9. Null coverage on key modelling columns ===")
print(q("""
    SELECT
      ROUND(100.0*SUM(CASE WHEN xg IS NULL THEN 1 ELSE 0 END)/COUNT(*),2)          AS pct_null_xg,
      ROUND(100.0*SUM(CASE WHEN opponent_elo IS NULL THEN 1 ELSE 0 END)/COUNT(*),2) AS pct_null_elo,
      ROUND(100.0*SUM(CASE WHEN minutes IS NULL THEN 1 ELSE 0 END)/COUNT(*),2)      AS pct_null_min
    FROM v_player_match WHERE is_pl AND season='2025-2026'
""").to_string(index=False))

print("\n=== 10. Player coverage: every player must have an estimate ===")
cov = q("""
    SELECT
      (SELECT COUNT(*) FROM players WHERE season = (SELECT MAX(season) FROM players)) AS in_game,
      (SELECT COUNT(*) FROM player_prior) AS with_prior,
      (SELECT COUNT(*) FROM player_prior WHERE xg90 IS NULL OR cbit90 IS NULL
              OR defcon_hit_rate IS NULL OR p_start_est IS NULL
              OR confidence IS NULL) AS incomplete
""").iloc[0]
check("every player in the game has a prior row", cov.in_game == cov.with_prior,
      f"{int(cov.with_prior)}/{int(cov.in_game)}")
check("no player has a NULL estimate", cov.incomplete == 0,
      f"{int(cov.incomplete)} incomplete")

print("\n=== 11. Data tier breakdown ===")
print(q("""
    SELECT tier, COUNT(*) AS players,
           ROUND(AVG(confidence),3) AS avg_confidence,
           ROUND(AVG(pl_minutes_prev)) AS avg_pl_mins,
           ROUND(AVG(preseason_mins)) AS avg_preseason_mins,
           MAX(minutes_source) AS example_minutes_source
    FROM player_prior GROUP BY tier ORDER BY tier
""").to_string(index=False))

print("\n=== 12. Pre-season fills the promoted-club gap ===")
print(q("""
    SELECT team, COUNT(*) AS squad,
           SUM(CASE WHEN tier IN ('A','A-') THEN 1 ELSE 0 END) AS has_pl_data,
           SUM(CASE WHEN tier='C' THEN 1 ELSE 0 END) AS preseason_only,
           SUM(CASE WHEN tier='D' THEN 1 ELSE 0 END) AS no_data_at_all
    FROM player_prior
    WHERE team IN ('Hull City','Coventry City','Ipswich Town')
    GROUP BY team ORDER BY team
""").to_string(index=False))

print("\n=== 13. Transfers detected ===")
t = q("SELECT COUNT(*) n FROM player_prior WHERE changed_club").iloc[0]
print(f"  {int(t.n)} players changed Premier League club; share_team_xg/xa "
      f"populated for transfer-safe projection")

con.close()
print()
