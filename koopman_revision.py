"""Comparison of predictor sets for next-day return forecasts (paper Section 5.3).

Produces Table 4 (predictor counts), Table 10 (RMSE and Diebold-Mariano tests),
Figure 4 (gain importance) and the inputs of Supplementary Tables S1-S4.

Implements Algorithm 1 of the paper:
  * expanding window: initial training window = first floor(0.8 T) usable rows,
    the scaler, the Koopman autoencoder and XGBoost are re-estimated every R
    forecast origins using only pairs (x_t, y_t) with t+1 <= tau;
  * MinMaxScaler fitted on the training window only;
  * lagged Koopman residual e_t = ||z_t - K z_{t-1}|| (a forward residual using
    z_{t+1}, i.e. x_{t+1}, would contain the target);
  * source-file columns whose timing uses information after t are excluded.

Feature sets: F0 = ret_lag_1..15, F1 = all conventional predictors,
F2 = 16 latent z + 16 eigen-coordinates Q^T z + lagged residual, F3 = F1 + F2.
Benchmarks: zero forecast, expanding historical mean.
Outputs: results/csv/predictor_sets/*.csv and results/figures/feature_score_koopman_rev.{eps,jpg}.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from koopman_core import (
    PAPER_CMAP,
    KoopmanAE,
    build_technical_features,
    build_xgb_model,
    koopman_loss,
    resolve_close_col,
    set_seed,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "dataset"
RESULTS = HERE / "results"
SEED = 42

DATASETS = {
    # key: (file, panel label)
    "dax": ("dax_1day_final2.xlsx", "DAX"),
    "u30": ("u30_1day.xlsx", "DJIA"),
    "u500": ("u500_1day.xlsx", "S&P 500"),
}

# Identifiers / non-predictors present in the source files.
ID_COLS = {"Unnamed: 0", "date", "time", "no."}

# Source columns excluded because their value at t uses information after t, or
# because their timing cannot be reconstructed from the data (conservative).
#   drawdown (DAX): a new value is stamped on day t exactly when r_t < 0 and
#       r_{t+1} >= 0 (557/557 run ends), i.e. it reveals the sign of r_{t+1}.
#   Crash Indicator u30/U500 (1%): not a function of r_t (e.g. -4.6% days have 0,
#       9% of flagged days have r_t > -1%); timing unverifiable -> excluded.
SOURCE_AUDIT = {
    "dax": {
        "return": ("keep", "p_t/p_{t-1}-1 (matches close, |diff|<1e-7); known at t"),
        "direction of dax": ("keep", "(r_t-r_{t-1})/r_{t-1}; known at t; inf/65535 sentinels set to NaN"),
        "Crash Indicator dax (10%)": ("keep", "equals 1{direction_t>0} on all rows; known at t"),
        "drawdown": ("exclude", "set on day t iff r_t<0 and r_{t+1}>=0: uses r_{t+1} (look-ahead)"),
    },
    "u30": {
        "Return of u30": ("keep", "p_t/p_{t-1}-1 (exact); known at t"),
        "direction of u30": ("keep", "equals 1{r_t>0} on all rows; known at t"),
        "Crash Indicator u30 (1%)": ("exclude", "not reproducible from r_{<=t}; timing unverifiable"),
    },
    "u500": {
        "Return of U500": ("keep", "p_t/p_{t-1}-1 (exact); known at t"),
        "direction of U500": ("keep", "equals 1{r_t>0} on all rows; known at t"),
        "Crash Indicator U500 (1%)": ("exclude", "not reproducible from r_{<=t}; timing unverifiable"),
    },
}

F0_COLS = [f"ret_lag_{k}" for k in range(1, 16)]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(key: str):
    fname, _ = DATASETS[key]
    df = pd.read_excel(DATA_DIR / fname)
    df.columns = [str(c).strip() for c in df.columns]
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    close_col = resolve_close_col(df, preferred="cp")
    audit = SOURCE_AUDIT[key]
    numeric_src = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in ID_COLS and c != close_col]
    unknown = [c for c in numeric_src if c not in audit]
    if unknown:
        raise ValueError(f"{key}: unaudited numeric source columns {unknown}")
    drop_src = [c for c in numeric_src if audit[c][0] == "exclude"]
    keep_src = [c for c in numeric_src if audit[c][0] == "keep"]

    # Object (text / model-output) columns are dropped as in koopman.py.
    base = df[["date", close_col] + keep_src].copy()
    feats = build_technical_features(base, close_col=close_col)
    # Uniform names across datasets.
    feats = feats.rename(columns={f"{close_col}_lag_{k}": f"close_lag_{k}" for k in range(1, 16)})

    feats["log_ret"] = np.log(feats[close_col].astype(float)).diff()
    feats["target"] = feats["log_ret"].shift(-1)  # y_t = log(p_{t+1}/p_t)

    feat_cols = [c for c in feats.select_dtypes(include=[np.number]).columns
                 if c not in {"date", close_col, "log_ret", "target"}]
    x = feats[feat_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    for c in keep_src:
        if c.startswith("direction"):
            x.loc[x[c].abs() >= 65535, c] = np.nan
    feats[feat_cols] = x

    # Warm-up: drop leading rows where trailing-window predictors are undefined;
    # drop the last row (target unobserved).
    close_based = [c for c in feat_cols if c not in keep_src]
    ok = feats[close_based].notna().all(axis=1) & feats["target"].notna()
    first = int(np.argmax(ok.values))
    feats = feats.iloc[first:].reset_index(drop=True)
    feats = feats[feats["target"].notna()].reset_index(drop=True)

    audit_rows = [{"dataset": key, "column": c, "decision": audit[c][0], "timing": audit[c][1]}
                  for c in numeric_src]
    return feats, feat_cols, keep_src, drop_src, audit_rows


# ---------------------------------------------------------------------------
# Koopman autoencoder (training on the current window only)
# ---------------------------------------------------------------------------

def train_koopman_window(x_train, latent_dim=16, hidden_dim=64, n_epochs=200,
                         batch_size=64, lr=1e-3, seed=SEED):
    """Train KoopmanAE on all pairs (x_t, x_{t+1}) inside the training window."""
    set_seed(seed)
    x_t = torch.tensor(x_train[:-1], dtype=torch.float32)
    x_tp1 = torch.tensor(x_train[1:], dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t, x_tp1), batch_size=batch_size, shuffle=False)

    model = KoopmanAE(x_train.shape[1], latent_dim, hidden_dim=hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss, best_state = float("inf"), None
    for _ in range(n_epochs):
        model.train()
        losses = []
        for bx, bxp in loader:
            out = model(bx, bxp)
            loss, _ = koopman_loss(out, bx, bxp)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        avg = float(np.mean(losses))
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


def koopman_features(model, x_all):
    """z_t, phi_t = Q^T z_t (eigenvalues sorted desc.), e_t = ||z_t - K z_{t-1}||.

    Every quantity at t depends on x_t and x_{t-1} only.
    """
    with torch.no_grad():
        z = model.encode(torch.tensor(x_all, dtype=torch.float32)).numpy()
        K, mags, basis = model.get_K()
        K, mags, basis = K.numpy(), mags.numpy(), basis.numpy()
    order = np.argsort(mags)[::-1]
    mu = mags[order]
    phi = z @ basis[:, order]
    e = np.full((len(z), 1), np.nan)
    e[1:, 0] = np.linalg.norm(z[1:] - z[:-1] @ K.T, axis=1)
    return np.hstack([z, phi, e]), mu


F2_NAMES = [f"z_{i}" for i in range(1, 17)] + [f"phi_{i}" for i in range(1, 17)] + ["koop_resid_lag"]


# ---------------------------------------------------------------------------
# Expanding-window run for one dataset
# ---------------------------------------------------------------------------

def gain_importance(model, names):
    score = model.get_booster().get_score(importance_type="gain")
    vals = np.array([score.get(f"f{i}", 0.0) for i in range(len(names))])
    vals = vals / vals.sum() if vals.sum() > 0 else vals
    return pd.DataFrame({"feature": names, "importance": vals}).sort_values(
        "importance", ascending=False).reset_index(drop=True)


def run_dataset(key, R=20, epochs=200, max_refits=None, torch_threads=4):
    torch.set_num_threads(torch_threads)
    t_start = time.time()
    feats, feat_cols, keep_src, drop_src, audit_rows = load_dataset(key)
    T = len(feats)
    T0 = int(np.floor(0.8 * T))
    y = feats["target"].values.astype(float)
    X_raw = feats[feat_cols].values.astype(float)

    origins = np.arange(T0, T)  # forecast y_tau = r_{tau+1} from info at tau
    refit_at = origins[(origins - T0) % R == 0]
    if max_refits is not None:
        refit_at = refit_at[-max_refits:]
        origins = origins[origins >= refit_at[0]]

    preds = {s: np.full(len(origins), np.nan) for s in ["F0", "F1", "F2", "F3"]}
    hist_mean = np.array([y[:tau].mean() for tau in origins])  # y_t, t+1<=tau
    log_rows, last = [], {}

    for j, tau in enumerate(refit_at):
        tr = np.arange(0, tau)  # {t : t+1 <= tau}
        # 50%-missing screen on the training window only.
        miss = np.isnan(X_raw[tr]).mean(axis=0)
        cols = [c for c, m in zip(feat_cols, miss) if m <= 0.5]
        ci = [feat_cols.index(c) for c in cols]
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_raw[tr][:, ci])
        Xs = scaler.transform(X_raw[:, ci])  # applied to t<=tau and later origins; fitted on tr only

        # Koopman AE on scaled pairs (x_t, x_{t+1}), t+1 <= tau  -> rows 0..tau.
        x_ae = np.nan_to_num(Xs, nan=0.0).astype(np.float32)
        ae, ae_loss = train_koopman_window(x_ae[: tau + 1], n_epochs=epochs)
        F2, mu = koopman_features(ae, x_ae)

        f0_idx = [cols.index(c) for c in F0_COLS]
        sets = {
            "F0": (Xs[:, f0_idx], F0_COLS),
            "F1": (Xs, cols),
            "F2": (F2, F2_NAMES),
            "F3": (np.hstack([Xs, F2]), cols + F2_NAMES),
        }
        block = (origins >= tau) & (origins < (refit_at[j + 1] if j + 1 < len(refit_at) else T))
        for s, (M, names) in sets.items():
            m = build_xgb_model(seed=SEED)
            m.fit(M[tr], y[tr])
            preds[s][block] = m.predict(M[origins[block]])
            last[s] = (m, names)
        log_rows.append({"dataset": key, "refit_origin": int(tau), "n_train": len(tr),
                         "n_F1": len(cols), "ae_train_loss": ae_loss,
                         "mu_max": float(mu[0]), "mu_min": float(mu[-1])})
        print(f"[{key}] refit {j + 1}/{len(refit_at)} tau={tau} n_train={len(tr)} "
              f"n_F1={len(cols)} ae_loss={ae_loss:.5f} ({time.time() - t_start:.0f}s)", flush=True)

    out = pd.DataFrame({"dataset": key, "t": origins, "date": feats["date"].values[origins],
                        "y": y[origins], "zero": 0.0, "hist_mean": hist_mean,
                        **{s: preds[s] for s in preds}})
    imp = gain_importance(*last["F3"])
    eig = pd.DataFrame({"dataset": key, "j": np.arange(1, 17), "mu": mu})
    meta = {"dataset": key, "T_usable": T, "T0": T0, "n_test": len(origins),
            "n_refits": len(refit_at), "R": R, "first_test_date": int(out["date"].iloc[0]),
            "last_test_date": int(out["date"].iloc[-1]),
            "n_F0": len(F0_COLS), "n_F1": log_rows[-1]["n_F1"], "n_F2": len(F2_NAMES),
            "n_F3": log_rows[-1]["n_F1"] + len(F2_NAMES),
            "source_cols_kept": "; ".join(keep_src), "source_cols_excluded": "; ".join(drop_src),
            "runtime_s": round(time.time() - t_start, 1)}
    return {"key": key, "pred": out, "imp": imp, "eig": eig, "log": pd.DataFrame(log_rows),
            "meta": meta, "audit": pd.DataFrame(audit_rows), "F1_cols": cols}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def dm_test_hln(e1, e2, h=1):
    """Diebold-Mariano on squared-error loss with HLN small-sample correction.

    d_t = e1_t^2 - e2_t^2; negative statistic favours model 1. Long-run variance
    from autocovariances up to lag h-1 (lag 0 for h=1); p-value from t(n-1).
    """
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(h)]
    lrv = gamma[0] + 2 * sum(gamma[1:])
    dm = dbar / np.sqrt(lrv / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return float(hln), float(p)


def plot_importance_panels(results, save_stem: Path, top_n=20):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, (letter, r) in zip(axes, zip("ABC", results)):
        top = r["imp"].head(top_n).iloc[::-1]
        vals = top["importance"].values
        vmax = vals.max() if vals.max() > 0 else 1.0
        bars = ax.barh(range(len(top)), vals, color=[PAPER_CMAP(v / vmax) for v in vals], height=0.7)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + vmax * 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                    va="center", ha="left", fontsize=8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values, fontsize=9)
        ax.set_xlim(0, vmax * 1.22)
        ax.set_xlabel("Gain importance (normalised)", fontsize=10, fontweight="bold")
        ax.set_title(f"({letter}) {DATASETS[r['key']][1]}", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("eps", "jpg"):
        fig.savefig(f"{save_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max-refits", type=int, default=None, help="smoke test: only the last N refit blocks")
    ap.add_argument("--output-dir", default=str(RESULTS / "csv" / "predictor_sets"))
    ap.add_argument("--fig-stem", default=str(RESULTS / "figures" / "feature_score_koopman_rev"))
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    threads = max(1, (os.cpu_count() or 4) // max(1, args.jobs))
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_dataset, k, args.R, args.epochs, args.max_refits, threads)
                for k in args.datasets]
        results = [f.result() for f in futs]

    preds = pd.concat([r["pred"] for r in results], ignore_index=True)
    preds.to_csv(out / "forecasts.csv", index=False)
    pd.concat([r["log"] for r in results]).to_csv(out / "refit_log.csv", index=False)
    pd.concat([r["eig"] for r in results]).to_csv(out / "koopman_eigenvalues_final_window.csv", index=False)
    pd.concat([r["audit"] for r in results]).to_csv(out / "source_column_audit.csv", index=False)
    pd.DataFrame([r["meta"] for r in results]).to_csv(out / "feature_counts_and_sample.csv", index=False)
    for r in results:
        r["imp"].to_csv(out / f"importance_F3_{r['key']}.csv", index=False)
        pd.Series(r["F1_cols"], name="feature").to_csv(out / f"F1_features_{r['key']}.csv", index=False)

    methods = ["zero", "hist_mean", "F0", "F1", "F2", "F3"]
    rmse_rows, dm_rows = [], []
    for r in results:
        p = r["pred"]
        err = {m: p["y"].values - p[m].values for m in methods}
        rmse_rows.append({"dataset": r["key"], "n_test": len(p),
                          **{m: float(np.sqrt(np.mean(err[m] ** 2))) for m in methods}})
        for a, b in [("F2", "F1"), ("F3", "F1"), ("F1", "F0"), ("F1", "zero"), ("F3", "zero"),
                     ("F1", "hist_mean"), ("F3", "hist_mean")]:
            stat, pv = dm_test_hln(err[a], err[b])
            dm_rows.append({"dataset": r["key"], "comparison": f"{a} vs {b}", "DM_HLN": stat, "p_value": pv})
    rmse = pd.DataFrame(rmse_rows)
    dm = pd.DataFrame(dm_rows)
    rmse.to_csv(out / "rmse_table.csv", index=False)
    dm.to_csv(out / "dm_tests.csv", index=False)

    rmse100 = rmse.set_index("dataset")[methods].T * 100
    rmse100.to_csv(out / "rmse_table_x100_paper_layout.csv")
    print("\nRMSE x 10^-2 (rows: method, cols: dataset)\n", rmse100.round(4).to_string())
    print("\nDM (HLN) tests\n", dm.round(4).to_string(index=False))
    print("\nFeature counts\n", pd.DataFrame([r["meta"] for r in results]).to_string(index=False))

    if args.max_refits is None and len(results) == 3:
        order = {k: i for i, k in enumerate(DATASETS)}
        plot_importance_panels(sorted(results, key=lambda r: order[r["key"]]), Path(args.fig_stem))
        print(f"\nSaved figure: {args.fig_stem}.eps/.jpg")


if __name__ == "__main__":
    main()
