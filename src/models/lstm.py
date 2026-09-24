import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import HORIZON, SEED

LOOKBACK = 56
COV = ["Open", "Promo", "SchoolHoliday", "StateHoliday"]


class Net(nn.Module):
    """LSTM encoder over the last LOOKBACK days + MLP head that sees the known future covariates."""

    def __init__(self, n_stores, n_cov, horizon, hidden=64, emb=8):
        super().__init__()
        self.emb = nn.Embedding(n_stores, emb)
        self.lstm = nn.LSTM(1 + n_cov, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden + emb + horizon * n_cov, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, horizon),
        )

    def forward(self, hist, fut, sid):
        _, (h, _) = self.lstm(hist)
        z = torch.cat([h[-1], self.emb(sid), fut.flatten(1)], dim=1)
        return self.head(z)


def _prepare(df: pd.DataFrame, horizon: int):
    """df must be a complete, sorted Store x Date grid. Returns arrays shaped (stores, days, ...)."""
    stores = np.sort(df["Store"].unique())
    S, T = len(stores), df["Date"].nunique()
    train_T = T - horizon

    y = np.log1p(df["Sales"].to_numpy(np.float32)).reshape(S, T)
    opn = df["Open"].to_numpy(np.float32).reshape(S, T)
    cov = df[COV].to_numpy(np.float32)
    dow = np.eye(7, dtype=np.float32)[df["Date"].dt.dayofweek.to_numpy()]
    C = np.concatenate([cov, dow], axis=1).reshape(S, T, -1)

    # per-store normalisation using open days in the training window only
    w = opn[:, :train_T]
    mu = (y[:, :train_T] * w).sum(1) / w.sum(1).clip(1)
    var = (((y[:, :train_T] - mu[:, None]) ** 2) * w).sum(1) / w.sum(1).clip(1)
    sd = np.sqrt(var).clip(0.1)
    yn = np.where(opn == 1, (y - mu[:, None]) / sd[:, None], 0.0).astype(np.float32)

    X = np.concatenate([yn[..., None], C], axis=2)  # (S, T, 1 + n_cov)
    return stores, S, T, train_T, yn, opn, C, X, mu, sd


def _batch(X, C, yn, opn, s, o, horizon):
    hist_idx = o[:, None] + np.arange(-LOOKBACK, 0)
    fut_idx = o[:, None] + np.arange(horizon)
    hist = torch.from_numpy(X[s[:, None], hist_idx])
    fut = torch.from_numpy(C[s[:, None], fut_idx])
    tgt = torch.from_numpy(yn[s[:, None], fut_idx])
    msk = torch.from_numpy(opn[s[:, None], fut_idx])
    return hist, fut, tgt, msk, torch.from_numpy(s.astype(np.int64))


def fit_predict(df, horizon=HORIZON, epochs=10, stride=3, batch_size=512, lr=2e-3, device=None) -> pd.DataFrame:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    stores, S, T, train_T, yn, opn, C, X, mu, sd = _prepare(df.sort_values(["Store", "Date"]), horizon)

    # training origins: forecast window [o, o+horizon) must lie fully inside the train period
    origins = np.arange(LOOKBACK, train_T - horizon + 1, stride)
    s_all = np.repeat(np.arange(S), len(origins))
    o_all = np.tile(origins, S)

    net = Net(S, C.shape[2], horizon).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(s_all))
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            hist, fut, tgt, msk, sid = [t.to(device) for t in _batch(X, C, yn, opn, s_all[idx], o_all[idx], horizon)]
            loss = (((net(hist, fut, sid) - tgt) ** 2) * msk).sum() / msk.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        print(f"[LSTM] epoch {ep + 1}/{epochs}  train loss {total / len(perm):.4f}")

    # forecast the validation window from origin = train_T
    net.eval()
    s = np.arange(S)
    o = np.full(S, train_T)
    with torch.no_grad():
        hist, fut, _, _, sid = [t.to(device) for t in _batch(X, C, yn, opn, s, o, horizon)]
        out = net(hist, fut, sid).cpu().numpy()  # (S, horizon), normalised log-sales

    pred = np.expm1(out * sd[:, None] + mu[:, None]).clip(0)
    pred = np.where(opn[:, train_T:] == 0, 0.0, pred)

    dates = np.sort(df["Date"].unique())[train_T:]
    return pd.DataFrame({
        "Store": np.repeat(stores, horizon),
        "Date": np.tile(dates, S),
        "pred": pred.ravel(),
    })
