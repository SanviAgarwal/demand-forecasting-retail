from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

HORIZON = 42          # forecast horizon in days (same 6-week window as the Rossmann test set)
N_EVAL_STORES = 30    # stores used for the head-to-head benchmark (SARIMA is fit per store)
SEED = 42
