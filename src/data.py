import numpy as np
import pandas as pd

from .config import RAW, HORIZON, SEED

ZERO_FILL = ["Sales", "Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday"]


def load_rossmann() -> pd.DataFrame:
    """Load train.csv + store.csv into a complete Store x Date grid.

    Days missing for a store (e.g. refurbishment closures) are filled as closed days
    (Open=0, Sales=0), so every store has the same continuous daily index.
    """
    train = pd.read_csv(RAW / "train.csv", parse_dates=["Date"], dtype={"StateHoliday": str})
    store = pd.read_csv(RAW / "store.csv")

    train["StateHoliday"] = (train["StateHoliday"] != "0").astype(int)

    grid = pd.MultiIndex.from_product(
        [np.sort(train["Store"].unique()), pd.date_range(train["Date"].min(), train["Date"].max())],
        names=["Store", "Date"],
    )
    df = train.set_index(["Store", "Date"]).reindex(grid)
    df[ZERO_FILL] = df[ZERO_FILL].fillna(0)
    df = df.reset_index()
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    df = df.merge(store, on="Store", how="left")
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


def split_train_val(df: pd.DataFrame, horizon: int = HORIZON):
    """Time-based split: the last `horizon` days are the validation window."""
    cutoff = df["Date"].max() - pd.Timedelta(days=horizon)
    return df[df["Date"] <= cutoff], df[df["Date"] > cutoff], cutoff


def pick_eval_stores(df: pd.DataFrame, n: int, seed: int = SEED) -> list:
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(df["Store"].unique(), size=n, replace=False).tolist())
