"""Leakage-free crash early-warning system (paper Section 5.5).

Produces the early-warning results in Section 5.5, Figure 5 and the inputs of
Supplementary Tables S5-S7 and Figure S1 (run once with --alpha 0.10 and once
with --alpha 0.20).

Event (ex ante, fixed horizon h):
    C_t = 1{ min_{1<=k<=h} p_{t+k}/p_t - 1 <= -q_tau },
i.e. a loss of at least q within the next h trading days. The threshold q_tau is
the (1 - alpha) quantile of the h-day forward maximum loss over the training
window (labels fully observed at tau), so it uses no test information.

Protocol (same as Algorithm 1 / koopman_revision.py):
  * expanding window, initial training = first floor(0.8 T) rows, refit every R;
  * purging: at origin tau only rows with t + h <= tau enter training
    (their label is observed at tau); features use information up to t only;
  * scaler and Koopman autoencoder are refit on the window (pairs t+1 <= tau);
  * XGBoost classifier, same tree hyperparameters as the regressors.
Signals: P(C_tau = 1 | x_tau) for F1, F2, F3; benchmark = trailing 20-day
realised volatility (classic EWS indicator); alarm when the probability exceeds
the training-window (1 - base rate) quantile of in-window fitted probabilities.

Evaluation on test origins whose label is observed (tau + h <= T - 1): ROC-AUC,
Brier skill score vs. the training base rate, hit rate, false-alarm rate,
precision; AUC differences vs. the volatility benchmark with a moving-block
bootstrap (block = h) for serial dependence of overlapping labels.

Outputs: results/csv/early_warning_alpha0.10/*.csv and results/figures/fig_ews_signals.{eps,jpg}
(use --output-dir and --fig-stem for alpha = 0.20).
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
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from koopman_revision import (
    DATASETS,
    F2_NAMES,
    SEED,
    koopman_features,
    load_dataset,
    train_koopman_window,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "dataset"
RESULTS = HERE / "results"
SETS = ["F1", "F2", "F3"]
SET_LABELS = {"F1": "technical", "F2": "Koopman", "F3": "technical + Koopman"}


def build_xgb_classifier(seed=SEED):
    # Same tree hyperparameters as build_xgb_model in koopman_core.py (Table 5).
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbosity=0, eval_metric="logloss",
    )


def forward_max_loss(close, h):
    """L_t = min_{1<=k<=h} p_{t+k}/p_t - 1 (NaN if t+h beyond sample)."""
    p = np.asarray(close, dtype=float)
    out = np.full(len(p), np.nan)
    for t in range(len(p) - h):
        out[t] = p[t + 1: t + h + 1].min() / p[t] - 1.0
    return out


def run_dataset(key, h=20, alpha=0.10, R=20, epochs=200, torch_threads=4):
    torch.set_num_threads(torch_threads)
    t0 = time.time()
    feats, feat_cols, _, _, _ = load_dataset(key)
    close_col = "cp" if "cp" in feats.columns else "closed"
    close = feats[close_col].values.astype(float)
    T = len(feats)
    T0 = int(np.floor(0.8 * T))
    X_raw = feats[feat_cols].values.astype(float)
    L = forward_max_loss(close, h)
    logret = np.log(close)
    vol20 = pd.Series(np.diff(logret, prepend=np.nan)).rolling(20).std().values

    origins = np.arange(T0, T)
    refit_at = origins[(origins - T0) % R == 0]
    prob = {s: np.full(len(origins), np.nan) for s in SETS}
    alarm_thr = {s: np.full(len(origins), np.nan) for s in SETS}
    q_at = np.full(len(origins), np.nan)
    base_at = np.full(len(origins), np.nan)
    vol_thr = np.full(len(origins), np.nan)
    log_rows = []

    for j, tau in enumerate(refit_at):
        block = (origins >= tau) & (origins < (refit_at[j + 1] if j + 1 < len(refit_at) else T))
        tr = np.arange(0, tau - h + 1)  # purged: label of row t observed iff t + h <= tau
        q = -np.quantile(L[tr], alpha)  # loss threshold (positive number)
        ytr = (L[tr] <= -q).astype(int)
        base = ytr.mean()

        fit_rows = np.arange(0, tau)  # features up to tau-1 for scaler (as in Exercise B)
        miss = np.isnan(X_raw[fit_rows]).mean(axis=0)
        ci = [i for i, m in enumerate(miss) if m <= 0.5]
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_raw[fit_rows][:, ci])
        Xs = scaler.transform(X_raw[:, ci])
        x_ae = np.nan_to_num(Xs, nan=0.0).astype(np.float32)
        ae, ae_loss = train_koopman_window(x_ae[: tau + 1], n_epochs=epochs)
        F2, _ = koopman_features(ae, x_ae)
        mats = {"F1": Xs, "F2": F2, "F3": np.hstack([Xs, F2])}

        for s in SETS:
            m = build_xgb_classifier()
            m.fit(mats[s][tr], ytr)
            p_in = m.predict_proba(mats[s][tr])[:, 1]
            prob[s][block] = m.predict_proba(mats[s][origins[block]])[:, 1]
            alarm_thr[s][block] = np.quantile(p_in, 1 - base)
        vtr = vol20[tr]
        vol_thr[block] = np.nanquantile(vtr, 1 - base)
        q_at[block] = q
        base_at[block] = base
        log_rows.append({"dataset": key, "refit_origin": int(tau), "n_train": len(tr),
                         "q_loss": q, "train_event_rate": base, "ae_train_loss": ae_loss})
        print(f"[{key}] refit {j + 1}/{len(refit_at)} tau={tau} n_train={len(tr)} "
              f"q={q:.4f} base={base:.3f} ({time.time() - t0:.0f}s)", flush=True)

    out = pd.DataFrame({
        "dataset": key, "t": origins, "date": feats["date"].values[origins],
        "close": close[origins], "fwd_max_loss": L[origins], "q_loss": q_at,
        "train_event_rate": base_at, "vol20": vol20[origins], "vol20_thr": vol_thr,
        **{f"p_{s}": prob[s] for s in SETS}, **{f"thr_{s}": alarm_thr[s] for s in SETS},
    })
    out["event"] = np.where(out["fwd_max_loss"].notna(),
                            (out["fwd_max_loss"] <= -out["q_loss"]).astype(float), np.nan)
    return {"key": key, "pred": out, "log": pd.DataFrame(log_rows),
            "runtime_s": round(time.time() - t0, 1)}


def block_bootstrap_auc_diff(y, s1, s2, block, n_boot=2000, seed=SEED):
    """Moving-block bootstrap of AUC(s1) - AUC(s2); returns diff, 95% CI, two-sided p."""
    rng = np.random.default_rng(seed)
    n = len(y)
    base = roc_auc_score(y, s1) - roc_auc_score(y, s2)
    nb = int(np.ceil(n / block))
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        diffs.append(roc_auc_score(yb, s1[idx]) - roc_auc_score(yb, s2[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    # two-sided p-value: bootstrap distribution centred at the estimate (null: diff = 0)
    p = np.mean(np.abs(diffs - base) >= abs(base))
    return base, lo, hi, float(p), len(diffs)


def evaluate(res, h):
    p = res["pred"].dropna(subset=["event"]).reset_index(drop=True)
    y = p["event"].values.astype(int)
    rows, tests = [], []
    signals = {"Volatility (20d)": (p["vol20"].values, p["vol20_thr"].values)}
    for s in SETS:
        signals[f"XGBoost {s}"] = (p[f"p_{s}"].values, p[f"thr_{s}"].values)
    # distinct event episodes = runs of consecutive event days
    episodes = int(np.sum((y[1:] == 1) & (y[:-1] == 0)) + (y[0] == 1)) if len(y) else 0
    for name, (score, thr) in signals.items():
        alarm = (score >= thr).astype(int)
        tp = int(np.sum((alarm == 1) & (y == 1)))
        fp = int(np.sum((alarm == 1) & (y == 0)))
        fn = int(np.sum((alarm == 0) & (y == 1)))
        tn = int(np.sum((alarm == 0) & (y == 0)))
        row = {"dataset": res["key"], "signal": name, "n_eval": len(y), "event_days": int(y.sum()),
               "event_episodes": episodes, "event_rate": float(y.mean()),
               "auc": float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else np.nan,
               "hit_rate": tp / max(tp + fn, 1), "false_alarm_rate": fp / max(fp + tn, 1),
               "precision": tp / max(tp + fp, 1), "alarm_rate": float(alarm.mean()),
               "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        if name.startswith("XGBoost"):
            clim = p["train_event_rate"].values
            bs = brier_score_loss(y, score)
            bs_ref = np.mean((clim - y) ** 2)
            row.update({"brier": bs, "brier_climatology": bs_ref, "brier_skill": 1 - bs / bs_ref})
        rows.append(row)
    if 0 < y.sum() < len(y):
        for s in SETS:
            d, lo, hi, pv, nb = block_bootstrap_auc_diff(y, p[f"p_{s}"].values, p["vol20"].values, block=h)
            tests.append({"dataset": res["key"], "comparison": f"AUC XGBoost {s} - Volatility",
                          "diff": d, "ci_lo": lo, "ci_hi": hi, "p_value": pv, "n_boot_valid": nb})
        for a, b in [("F3", "F1"), ("F2", "F1")]:
            d, lo, hi, pv, nb = block_bootstrap_auc_diff(y, p[f"p_{a}"].values, p[f"p_{b}"].values, block=h)
            tests.append({"dataset": res["key"], "comparison": f"AUC XGBoost {a} - {b}",
                          "diff": d, "ci_lo": lo, "ci_hi": hi, "p_value": pv, "n_boot_valid": nb})
    return rows, tests


def to_gregorian_label(d):
    """yyyymmdd int -> 'yyyy-mm'; DAX file uses Buddhist-era years (BE = CE + 543)."""
    y, m = int(d) // 10000, (int(d) // 100) % 100
    if y > 2400:
        y -= 543
    return f"{y}-{m:02d}"


def plot_signals(results, save_stem, sig="F3"):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    handles = None
    for ax, (letter, r) in zip(axes, zip("ABC", results)):
        p = r["pred"]
        x = np.arange(len(p))
        ax.plot(x, p["close"].values, color="black", lw=1.0, label="Index close")
        ax.set_ylabel("Close", fontsize=10)
        ev = p["event"].values == 1
        alarm = p[f"p_{sig}"].values >= p[f"thr_{sig}"].values
        ymin, ymax = np.nanmin(p["close"]), np.nanmax(p["close"])
        ax.fill_between(x, ymin, ymax, where=ev, color="#d62728", alpha=0.18, step="mid",
                        label=f"Event: loss $\\geq q$ within next h days")
        ax2 = ax.twinx()
        ax2.plot(x, p[f"p_{sig}"].values, color="#3a7ca5", lw=1.0,
                 label=f"P(event), XGBoost {SET_LABELS[sig]}")
        ax2.plot(x, p[f"thr_{sig}"].values, color="#3a7ca5", lw=0.8, ls="--", label="Alarm threshold")
        ax2.scatter(x[alarm], np.full(alarm.sum(), 1.02), marker="|", color="#4a1a6b", s=40,
                    label="Alarm")
        ax2.set_ylim(0, 1.06)
        ax2.set_ylabel("Probability", fontsize=10)
        ticks = np.linspace(0, len(p) - 1, 6).astype(int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([to_gregorian_label(d) for d in p["date"].values[ticks]], fontsize=9)
        ax.set_title(f"({letter}) {DATASETS[r['key']][1]}", fontsize=12, fontweight="bold", loc="left")
        if handles is None:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            handles = (h1 + h2, l1 + l2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(*handles, loc="upper center", ncol=5, fontsize=9, frameon=False)
    for ext in ("eps", "jpg"):
        fig.savefig(f"{save_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--output-dir", default=str(RESULTS / "csv" / "early_warning_alpha0.10"))
    ap.add_argument("--fig-stem", default=str(RESULTS / "figures" / "fig_ews_signals"))
    ap.add_argument("--replot", action="store_true", help="only redraw the figure from ews_signals.csv")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        sig = pd.read_csv(out / "ews_signals.csv")
        plot_signals([{"key": k, "pred": sig[sig["dataset"] == k].reset_index(drop=True)}
                      for k in DATASETS], args.fig_stem)
        print(f"Saved figure: {args.fig_stem}.eps/.jpg")
        return
    threads = max(1, (os.cpu_count() or 4) // max(1, args.jobs))
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_dataset, k, args.h, args.alpha, args.R, args.epochs, threads)
                for k in args.datasets]
        results = [f.result() for f in futs]

    pd.concat([r["pred"] for r in results]).to_csv(out / "ews_signals.csv", index=False)
    pd.concat([r["log"] for r in results]).to_csv(out / "ews_refit_log.csv", index=False)
    rows, tests = [], []
    for r in results:
        a, b = evaluate(r, args.h)
        rows += a
        tests += b
    perf = pd.DataFrame(rows)
    tst = pd.DataFrame(tests)
    perf.to_csv(out / "ews_performance.csv", index=False)
    tst.to_csv(out / "ews_auc_tests.csv", index=False)
    pd.DataFrame([{"h": args.h, "alpha": args.alpha, "R": args.R, "epochs": args.epochs,
                   **{f"runtime_{r['key']}": r["runtime_s"] for r in results}}]).to_csv(
        out / "ews_settings.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nEWS performance\n", perf.round(4).to_string(index=False))
    print("\nAUC tests (moving-block bootstrap)\n", tst.round(4).to_string(index=False))
    if len(results) == 3:
        order = {k: i for i, k in enumerate(DATASETS)}
        plot_signals(sorted(results, key=lambda r: order[r["key"]]), args.fig_stem)
        print(f"\nSaved figure: {args.fig_stem}.eps/.jpg")


if __name__ == "__main__":
    main()
