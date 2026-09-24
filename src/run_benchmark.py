import argparse

from . import data, features, metrics
from .config import HORIZON, N_EVAL_STORES, RESULTS
from .models import baseline, sarima, xgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stores", type=int, default=N_EVAL_STORES)
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--lstm-epochs", type=int, default=10)
    ap.add_argument("--xgb-trees", type=int, default=800)
    args = ap.parse_args()

    df = data.load_rossmann()
    train, val, cutoff = data.split_train_val(df, HORIZON)
    eval_stores = data.pick_eval_stores(df, args.n_stores)
    print(f"Train up to {cutoff.date()} | validate {val['Date'].min().date()} to {val['Date'].max().date()} "
          f"| {len(eval_stores)} eval stores")

    # ground truth: open days in the validation window for the evaluation stores
    actual = val[(val["Store"].isin(eval_stores)) & (val["Open"] == 1) & (val["Sales"] > 0)][["Store", "Date", "Sales"]]

    preds = {"Seasonal naive": baseline.seasonal_naive(train, val, cutoff)}

    print("Fitting SARIMA (per store)...")
    preds["SARIMA"] = sarima.fit_predict(train, val, eval_stores)

    print("Fitting XGBoost (global)...")
    feat = features.make_features(df, HORIZON)
    preds["XGBoost"] = xgb.fit_predict(feat, cutoff, n_estimators=args.xgb_trees)

    if not args.skip_lstm:
        from .models import lstm  # imported lazily so --skip-lstm works without torch
        print("Training LSTM (global)...")
        preds["LSTM"] = lstm.fit_predict(df, HORIZON, epochs=args.lstm_epochs)

    table = metrics.benchmark_table(actual, preds, baseline="SARIMA")
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "benchmark.csv")
    (RESULTS / "benchmark.md").write_text(table.to_markdown())
    print("\n" + table.to_markdown())


if __name__ == "__main__":
    main()
