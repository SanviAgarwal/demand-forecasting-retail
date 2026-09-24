import numpy as np
import pandas as pd
import xgboost as xgb

from ..config import SEED
from ..features import FEATURES


def fit_predict(feat: pd.DataFrame, cutoff, n_estimators: int = 800) -> pd.DataFrame:
    """Global XGBoost model across all stores, trained on open days, target = log1p(Sales)."""
    train = feat[(feat["Date"] <= cutoff) & (feat["Open"] == 1)]
    val = feat[feat["Date"] > cutoff]

    model = xgb.XGBRegressor(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        n_jobs=-1, random_state=SEED,
    )
    model.fit(train[FEATURES], train["y"])

    pred = np.expm1(model.predict(val[FEATURES])).clip(0)
    pred = np.where(val["Open"].to_numpy() == 0, 0.0, pred)
    return pd.DataFrame({"Store": val["Store"].to_numpy(), "Date": val["Date"].to_numpy(), "pred": pred})
