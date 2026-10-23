"""Net-of-cost backtest of XGBoost vs LSTM on the daily data (paper Table 9).

Also produces the inputs of Supplementary Table S8.

Covers the 9 daily datasets of Table table_dataset (DAX, DJIA, S&P 500 x three
stress episodes). The intraday datasets are not available, so they are not
included; this is a re-implementation and not the program that produced the
gross results of the XGBoost--LSTM comparison tables.

Design (fixed a priori, no tuning on test data):
  * target r_{t+1} = log(p_{t+1}/p_t); input = last w daily log returns
    (w in {5, 10, 30}); LSTM sees them as a sequence, XGBoost as a vector;
  * each model is estimated once per episode on all pairs (x_t, r_{t+1}) with
    t+1 before the test start; standardisation uses training data only;
  * forecasts for origins t with test_start <= date_t and date_{t+1} <= test_end;
  * position s_t = sign(forecast) in {-1, +1}, held over (t, t+1];
    net return s_t R_{t+1} - c |s_t - s_{t-1}|, R = simple return, s before
    the first test origin = 0 (entry cost counted); c in {0, 1, 5} bp;
  * buy-and-hold: s_t = +1, no trading cost;
  * per model/window: turnover and cumulative return averaged across the 9
    datasets; 95% CI from a stationary bootstrap (mean block length 5,
    2000 replications) applied to each dataset's net return series.
Outputs: results/csv/daily_backtest/*.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from koopman_core import build_xgb_model, set_seed

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "dataset"
RESULTS = HERE / "results"
SEED = 42
OUT = RESULTS / "csv" / "daily_backtest"

FILES = {"DAX": "dax_1day_final2.xlsx", "DJIA": "u30_1day.xlsx", "S&P 500": "u500_1day.xlsx"}
EPISODES = {  # Table table_dataset (daily rows)
    "COVID-19": ("2019-12-05", "2020-03-31"),
    "Russia-Ukraine": ("2021-10-30", "2022-02-24"),
    "Chinese correction": ("2023-07-05", "2023-10-27"),
}
WINDOWS = [5, 10, 30]
COSTS_BP = [0, 1, 5]


def load_daily(fname):
    df = pd.read_excel(DATA_DIR / fname)
    df.columns = [str(c).strip() for c in df.columns]
    close = "cp" if "cp" in df.columns else "closed"
    d = df["date"].astype(int)
    year = d // 10000
    year = np.where(year > 2400, year - 543, year)  # DAX file uses Buddhist-era years
    dates = pd.to_datetime(pd.DataFrame({"year": year, "month": (d // 100) % 100, "day": d % 100}))
    out = pd.DataFrame({"date": dates, "p": df[close].astype(float)}).sort_values("date")
    return out.drop_duplicates("date").reset_index(drop=True)


class LSTMReg(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1]).squeeze(-1)


def fit_predict_lstm(xtr, ytr, xte, epochs=100, batch=64, lr=1e-3):
    set_seed(SEED)
    model = LSTMReg()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(xtr[..., None], dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            loss = nn.functional.mse_loss(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(xte[..., None], dtype=torch.float32)).numpy()


def windows(r, w):
    """X[t] = (r_{t-w+1},...,r_t); valid for t >= w (r[0] is NaN)."""
    X = np.full((len(r), w), np.nan)
    for t in range(w, len(r)):
        X[t] = r[t - w + 1: t + 1]
    return X


def stationary_bootstrap_idx(n, mean_block, rng):
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for i in range(1, n):
        idx[i] = rng.integers(n) if rng.random() < 1.0 / mean_block else (idx[i - 1] + 1) % n
    return idx


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=str(OUT))
    OUT = Path(ap.parse_args().output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, series = [], []
    for name, fname in FILES.items():
        df = load_daily(fname)
        p = df["p"].values
        r = np.r_[np.nan, np.diff(np.log(p))]  # r[t] = log(p_t/p_{t-1})
        R = np.r_[np.nan, p[1:] / p[:-1] - 1]
        y = np.r_[r[1:], np.nan]               # target of origin t: r_{t+1}
        dates = df["date"].values
        for ep, (start, end) in EPISODES.items():
            s0, e0 = np.datetime64(start), np.datetime64(end)
            test = np.array([t for t in range(len(p) - 1) if dates[t] >= s0 and dates[t + 1] <= e0])
            first = test[0]
            for w in WINDOWS:
                X = windows(r, w)
                tr = np.array([t for t in range(w, first) if t + 1 < first])  # target observed before test
                mu, sd = r[1:first].mean(), r[1:first].std()
                Xs = (X - mu) / sd
                ys = y / sd
                preds = {
                    "XGBoost": build_xgb_model(seed=SEED).fit(Xs[tr], ys[tr]).predict(Xs[test]),
                    "LSTM": fit_predict_lstm(Xs[tr], ys[tr], Xs[test]),
                }
                for model, f in preds.items():
                    s = np.where(f >= 0, 1.0, -1.0)
                    ds = np.abs(np.diff(np.r_[0.0, s]))
                    gross = s * R[test + 1]
                    rec = {"index": name, "episode": ep, "model": model, "window": w,
                           "n_test": len(test), "test_first": str(dates[first])[:10],
                           "test_last_target": str(dates[test[-1] + 1])[:10],
                           "turnover": ds.mean(), "hit_rate": float(np.mean(np.sign(f) == np.sign(y[test])))}
                    for c in COSTS_BP:
                        net = gross - c / 1e4 * ds
                        rec[f"cum_c{c}"] = 100 * (np.prod(1 + net) - 1)
                        series.append({"key": (model, w, c), "dataset": (name, ep), "net": net})
                    rows.append(rec)
            bh = R[test + 1]
            rec = {"index": name, "episode": ep, "model": "Buy-and-hold", "window": 0,
                   "n_test": len(test), "test_first": str(dates[first])[:10],
                   "test_last_target": str(dates[test[-1] + 1])[:10], "turnover": 0.0, "hit_rate": np.nan}
            for c in COSTS_BP:
                rec[f"cum_c{c}"] = 100 * (np.prod(1 + bh) - 1)
                series.append({"key": ("Buy-and-hold", 0, c), "dataset": (name, ep), "net": bh})
            rows.append(rec)
            print(f"{name} {ep}: {len(test)} test days {rec['test_first']}..{rec['test_last_target']}", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_dataset.csv", index=False)

    rng = np.random.default_rng(SEED)
    n_boot = 2000
    keys = sorted({s["key"] for s in series}, key=str)
    ci = {}
    by_key = {k: [s["net"] for s in series if s["key"] == k] for k in keys}
    for k, nets in by_key.items():
        boots = np.empty(n_boot)
        for b in range(n_boot):
            vals = [100 * (np.prod(1 + x[stationary_bootstrap_idx(len(x), 5, rng)]) - 1) for x in nets]
            boots[b] = np.mean(vals)
        ci[k] = np.quantile(boots, [0.025, 0.975])

    summ = []
    for (model, w), g in per.groupby(["model", "window"], sort=False):
        row = {"model": model, "window": w, "n_datasets": len(g), "turnover": g["turnover"].mean()}
        for c in COSTS_BP:
            lo, hi = ci[(model, w, c)]
            row.update({f"cum_c{c}": g[f"cum_c{c}"].mean(), f"ci_lo_c{c}": lo, f"ci_hi_c{c}": hi})
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "table_netcost_daily.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
