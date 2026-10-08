"""Configuration for the FPL data ingestion pipeline."""

from pathlib import Path

# ---------------------------------------------------------------- data source
# olbauday/FPL-Core-Insights. Chosen over vaastav/Fantasy-Premier-League as the
# primary source because it exposes per-player, per-match defensive action
# counts (tackles / interceptions / blocks / clearances / recoveries), which are
# required to model DefCon as a threshold event rather than a linear rate.
BASE_URL = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"

# Seasons to ingest. 2025-2026 is the first season with DefCon scoring AND
# per-match action counts, so it is the earliest usable season for that
# component. Earlier seasons are still fine for goals/assists/CS modelling but
# are not published by this repo.
SEASONS = ["2025-2026", "2026-2027"]

MAX_GAMEWEEK = 38

# ------------------------------------------------------------------- storage
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"          # cached CSVs, safe to delete
DB_PATH = ROOT / "fpl.duckdb"   # the database

# ------------------------------------------------------- download behaviour
TIMEOUT = 60
MAX_WORKERS = 8
RETRIES = 3

# ---------------------------------------------------------- FPL scoring rules
# 2026/27 rules. Kept here (not in model code) so weights can be re-tuned
# without touching the pipeline.
DEFCON_THRESHOLD = {
    "DEF": 10,   # CBIT  = clearances + blocks + interceptions + tackles
    "MID": 12,   # CBIRT = CBIT + recoveries
    "FWD": 12,
    "GKP": None,  # goalkeepers are not eligible
}
DEFCON_POINTS = 2  # capped: hitting 2x the threshold still scores 2

POSITION_MAP = {
    "Goalkeeper": "GKP",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD",
}
