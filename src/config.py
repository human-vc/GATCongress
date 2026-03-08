from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SEED = 42

CONGRESSES = list(range(100, 119))

THRESHOLD_TAU = 0.5
MIN_VOTES = 50
MIN_SHARED_VOTES = 20
DEFECTION_THRESHOLD = 0.10

DEM_COLOR = "#2166ac"
REP_COLOR = "#b2182b"
CROSS_COLOR = "#d95f02"

VOTEVIEW_URLS = {
    "HSall_members.csv": "https://voteview.com/static/data/out/members/HSall_members.csv",
    "HSall_votes.csv": "https://voteview.com/static/data/out/votes/HSall_votes.csv",
    "HSall_rollcalls.csv": "https://voteview.com/static/data/out/rollcalls/HSall_rollcalls.csv",
}
