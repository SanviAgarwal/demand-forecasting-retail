import numpy as np
import pandas as pd


def mape(y, p):
    return float(np.mean(np.abs(y - p) / y) * 100)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def rmspe(y, p):
    """Root mean squared percentage error - the official Rossmann Kaggle metric."""
    return float(np.sqrt(np.mean(((y - p) / y) ** 2)) * 100)


def benchmark_table(actual: pd.DataFrame, preds: dict, baseline: str = "SARIMA") -> pd.DataFrame:
    """actual: Store, Date, Sales (open days only). preds: {model: DataFrame[Store, Date, pred]}."""
    rows = []
    for name, p in preds.items():
        m = actual.merge(p, on=["Store", "Date"], how="inner").dropna(subset=["pred"])
        rows.append({
            "Model": name,
            "MAPE (%)": mape(m["Sales"], m["pred"]),
            "RMSE": rmse(m["Sales"], m["pred"]),
            "RMSPE (%)": rmspe(m["Sales"], m["pred"]),
            "n_points": len(m),
        })
    t = pd.DataFrame(rows).set_index("Model")
    if baseline in t.index:
        t[f"MAPE improvement vs {baseline} (%)"] = (t.loc[baseline, "MAPE (%)"] - t["MAPE (%)"]) / t.loc[baseline, "MAPE (%)"] * 100
        t[f"RMSE improvement vs {baseline} (%)"] = (t.loc[baseline, "RMSE"] - t["RMSE"]) / t.loc[baseline, "RMSE"] * 100
    return t.round(2)
