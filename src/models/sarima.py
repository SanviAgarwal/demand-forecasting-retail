import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.sarimax import SARIMAX

EXOG = ["Open", "Promo", "StateHoliday", "SchoolHoliday"]
TRAIN_DAYS = 540  # fit on the most recent ~1.5 years to keep per-store fits fast


def _forecast_one(store, tr: pd.DataFrame, va: pd.DataFrame) -> pd.DataFrame:
    tr = tr.tail(TRAIN_DAYS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = SARIMAX(
                np.log1p(tr["Sales"].to_numpy()),
                exog=tr[EXOG].to_numpy(dtype=float),
                order=(1, 0, 1),
                seasonal_order=(1, 1, 1, 7),   # weekly seasonality
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False, maxiter=50)
            pred = np.expm1(res.forecast(steps=len(va), exog=va[EXOG].to_numpy(dtype=float))).clip(0)
        except Exception:
            pred = np.full(len(va), np.nan)
    pred = np.where(va["Open"].to_numpy() == 0, 0.0, pred)
    return pd.DataFrame({"Store": store, "Date": va["Date"].to_numpy(), "pred": pred})


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, stores: list, n_jobs: int = -1) -> pd.DataFrame:
    """Per-store SARIMAX(1,0,1)(1,1,1,7) on log1p(Sales) with Open/Promo/holiday regressors."""
    tr_g = {s: d for s, d in train[train["Store"].isin(stores)].groupby("Store")}
    va_g = {s: d for s, d in val[val["Store"].isin(stores)].groupby("Store")}
    out = Parallel(n_jobs=n_jobs)(delayed(_forecast_one)(s, tr_g[s], va_g[s]) for s in stores)
    return pd.concat(out, ignore_index=True)
