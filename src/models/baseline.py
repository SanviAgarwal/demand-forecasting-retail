import pandas as pd


def seasonal_naive(train: pd.DataFrame, val: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Repeat the last observed week: pred(t) = sales on the same weekday in the final 7 train days."""
    last_week = train[train["Date"] > cutoff - pd.Timedelta(days=7)]
    lookup = last_week.groupby(["Store", "DayOfWeek"])["Sales"].last().rename("pred").reset_index()
    out = val[["Store", "Date", "DayOfWeek", "Open"]].merge(lookup, on=["Store", "DayOfWeek"], how="left")
    out.loc[out["Open"] == 0, "pred"] = 0
    return out[["Store", "Date", "pred"]]
