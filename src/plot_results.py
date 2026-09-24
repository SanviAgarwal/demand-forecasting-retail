import argparse

import matplotlib.pyplot as plt

from . import data, features
from .config import HORIZON
from .models import baseline, sarima, xgb

SCREENSHOTS = data.RAW.parent.parent / "screenshots"


def plot_store(store_id, val, preds: dict, out_dir):
    actual = val[(val["Store"] == store_id) & (val["Open"] == 1)][["Date", "Sales"]]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(actual["Date"], actual["Sales"], label="Actual", color="black", linewidth=2)

    for name, p in preds.items():
        line = p[p["Store"] == store_id].merge(actual[["Date"]], on="Date", how="inner")
        ax.plot(line["Date"], line["pred"], label=name, linewidth=1.5, alpha=0.85)

    ax.set_title(f"Store {store_id} — actual vs. predicted sales (42-day holdout)")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"forecast_store_{store_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", type=int, nargs="+", default=None,
                     help="Store IDs to plot. Defaults to 3 stores sampled with the project seed.")
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--lstm-epochs", type=int, default=10)
    ap.add_argument("--xgb-trees", type=int, default=800)
    args = ap.parse_args()

    df = data.load_rossmann()
    train, val, cutoff = data.split_train_val(df, HORIZON)
    stores = args.stores or data.pick_eval_stores(df, 3)

    preds = {"Seasonal naive": baseline.seasonal_naive(train, val, cutoff)}

    print("Fitting SARIMA...")
    preds["SARIMA"] = sarima.fit_predict(train, val, stores)

    print("Fitting XGBoost...")
    feat = features.make_features(df, HORIZON)
    preds["XGBoost"] = xgb.fit_predict(feat, cutoff, n_estimators=args.xgb_trees)

    if not args.skip_lstm:
        from .models import lstm
        print("Training LSTM...")
        preds["LSTM"] = lstm.fit_predict(df, HORIZON, epochs=args.lstm_epochs)

    for s in stores:
        plot_store(s, val, preds, SCREENSHOTS)


if __name__ == "__main__":
    main()