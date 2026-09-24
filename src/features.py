import numpy as np
import pandas as pd

from .config import HORIZON

FEATURES = [
    "Store", "DayOfWeek", "Day", "Month", "WeekOfYear",
    "Open", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance", "Promo2",
    "lag_h", "lag_h7", "lag_h14", "lag_364",
    "roll_mean_7", "roll_mean_28", "roll_mean_91",
]


def make_features(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Leak-free features for a `horizon`-day-ahead forecast.

    Every lag / rolling feature is shifted by at least `horizon` days, so a row in the
    validation window only uses information that existed at the forecast origin.
    Calendar, Open, Promo and holiday flags are known in advance and used as-is.
    """
    df = df.sort_values(["Store", "Date"]).copy()
    df["y"] = np.log1p(df["Sales"])
    df["y_open"] = df["y"].where(df["Open"] == 1)
    g = df.groupby("Store")["y_open"]

    df["lag_h"] = g.shift(horizon)
    df["lag_h7"] = g.shift(horizon + 7)
    df["lag_h14"] = g.shift(horizon + 14)
    df["lag_364"] = g.shift(364)
    for w in (7, 28, 91):
        df[f"roll_mean_{w}"] = g.transform(lambda s: s.shift(horizon).rolling(w, min_periods=1).mean())

    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    for c in ("StoreType", "Assortment"):
        df[c] = df[c].astype("category").cat.codes
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    return df
