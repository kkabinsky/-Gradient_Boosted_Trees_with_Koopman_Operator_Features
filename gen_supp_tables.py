"""Generate the LaTeX bodies of Supplementary Tables S1-S8 from the saved result CSVs.

Every number in results/latex_tables/*.tex is read from:
  results/csv/predictor_sets/            (comparison of predictor sets)
  results/csv/early_warning_alpha0.10/,
  results/csv/early_warning_alpha0.20/   (early-warning system)
  results/csv/daily_backtest/            (daily XGBoost--LSTM backtest)
The realised-volatility signal computed by koopman_ews.py is not reported in
the paper, so its rows are omitted from the tables.
"""

import re
from pathlib import Path

import pandas as pd

from replot_importance import readable

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "csv"
OUT = HERE / "results" / "latex_tables"
REV = CSV / "predictor_sets"
EWS10 = CSV / "early_warning_alpha0.10"
EWS20 = CSV / "early_warning_alpha0.20"
NAMES = {"dax": "DAX", "u30": "DJIA", "u500": r"S\&P~500"}
SETNAME = {"zero": "Zero-return forecast", "hist_mean": "Historical mean", "F0": "XGBoost, lagged returns",
           "F1": "XGBoost, technical predictors", "F2": "XGBoost, Koopman features",
           "F3": "XGBoost, technical + Koopman"}
SHORT = {"zero": "zero-return forecast", "hist_mean": "historical mean", "F0": "lagged returns",
         "F1": "technical", "F2": "Koopman", "F3": "technical + Koopman"}
SIGNAL = {"Volatility (20d)": r"Realised volatility (20d)", "XGBoost F1": "XGBoost, technical",
          "XGBoost F2": "XGBoost, Koopman", "XGBoost F3": "XGBoost, technical + Koopman"}


def tex(s):
    return s.replace("&", r"\&").replace("%", r"\%")


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body.strip() + "\n", encoding="utf-8")
    print("wrote", name)


AUDIT_TEXT = {
    "return": r"Equals $p_t/p_{t-1}-1$ computed from the close (largest difference below $10^{-7}$); known at $t$.",
    "Return of u30": r"Equals $p_t/p_{t-1}-1$ exactly; known at $t$.",
    "Return of U500": r"Equals $p_t/p_{t-1}-1$ exactly; known at $t$.",
    "direction of dax": r"Equals $(r_t-r_{t-1})/r_{t-1}$; known at $t$. Infinite values and the code 65535 (division by a zero return) are treated as missing.",
    "direction of u30": r"Equals $\mathbf{1}\{r_t>0\}$ on every row; known at $t$.",
    "direction of U500": r"Equals $\mathbf{1}\{r_t>0\}$ on every row; known at $t$.",
    "Crash Indicator dax (10%)": r"Equals one exactly when the direction column is positive, on every row; known at $t$.",
    "drawdown": r"Takes a new value on every day with $r_t<0$ and $r_{t+1}\ge 0$ (557 of 557 such days) and on only 20 other days, i.e.\ at the end of a run of declines, which is known only at $t+1$. Look-ahead; excluded.",
    "Crash Indicator u30 (1%)": r"Cannot be reproduced from returns up to $t$ (e.g.\ a $-4.6\%$ day is not flagged, and 9\% of flagged days have $r_t>-1\%$); timing cannot be verified. Excluded as a precaution.",
    "Crash Indicator U500 (1%)": r"Cannot be reproduced from returns up to $t$ (e.g.\ a $-4.2\%$ day is not flagged, and 8\% of flagged days have $r_t>-1\%$); timing cannot be verified. Excluded as a precaution.",
}


def source_audit():
    a = pd.read_csv(REV / "source_column_audit.csv")
    rows = []
    for _, r in a.iterrows():
        dec = "kept" if r.decision == "keep" else r"\textbf{excluded}"
        rows.append(f"{NAMES[r.dataset]} & ``{tex(r.column)}'' & {dec} & {AUDIT_TEXT[r.column]} \\\\")
    write("s_source_audit.tex", "\n".join(rows))


GROUPS = [
    (r"ret_lag_(\d+)", "Return over the last $k$ days, $p_t/p_{t-k}-1$", "k"),
    (r"close_lag_(\d+)", "Closing price $k$ days ago", "k"),
    (r"close_sma_(\d+)", "Moving average of the close", "w"),
    (r"close_ema_(\d+)", "Exponential moving average of the close", "w"),
    (r"close_std_(\d+)", "Rolling standard deviation of the close", "w"),
    (r"close_z_(\d+)", "$z$-score of the close", "w"),
    (r"rsi_(\d+)", "Relative strength index (RSI)", "w"),
    (r"roc_(\d+)", "Rate of change", "w"),
]


def predictor_list():
    rows = []
    sets = {k: pd.read_csv(REV / f"F1_features_{k}.csv")["feature"].tolist() for k in NAMES}
    feats = sets["dax"]
    used = set()
    for pat, label, kind in GROUPS:
        ks = sorted(int(re.fullmatch(pat, f).group(1)) for f in feats if re.fullmatch(pat, f))
        used |= {f for f in feats if re.fullmatch(pat, f)}
        par = (f"$k={ks[0]},\\dots,{ks[-1]}$" if kind == "k" else "windows " + ", ".join(map(str, ks)) + " days")
        rows.append(f"{label} & {par} & {len(ks)} & {len(ks)} & {len(ks)} \\\\")
    macd = [f for f in feats if f.startswith("macd")]
    bb = [f for f in feats if f.startswith("bb_")]
    used |= set(macd) | set(bb)
    rows.append(f"MACD (12, 26, 9): line, signal line, histogram & -- & {len(macd)} & {len(macd)} & {len(macd)} \\\\")
    rows.append(f"Bollinger bands (20 days, 2 s.d.): middle, upper, lower, width, \\%B & -- & {len(bb)} & {len(bb)} & {len(bb)} \\\\")
    src = {k: [f for f in v if f not in used] for k, v in sets.items()}
    for lab in ["Daily return (source)", "Direction indicator (source)", "Crash indicator (source)"]:
        cnt = [sum(readable(f) == lab for f in src[k]) for k in NAMES]
        rows.append(f"{lab.replace(' (source)', ' (source file)')} & value at day $t$ & " + " & ".join(map(str, cnt)) + r" \\")
    tot = [len(sets[k]) for k in NAMES]
    rows.append(r"\midrule")
    rows.append("Total technical predictors & & " + " & ".join(map(str, tot)) + r" \\")
    write("s_predictors.tex", "\n".join(rows))


def dm_all():
    dm = pd.read_csv(REV / "dm_tests.csv")
    rm = pd.read_csv(REV / "rmse_table.csv").set_index("dataset")
    rows = []
    for _, r in dm.iterrows():
        a, b = r.comparison.split(" vs ")
        rows.append(f"{NAMES[r.dataset]} & {SHORT[a]} vs {SHORT[b]} & {100 * rm.loc[r.dataset, a]:.4f} & "
                    f"{100 * rm.loc[r.dataset, b]:.4f} & {r.DM_HLN:.3f} & {r.p_value:.4f} \\\\")
    write("s_dm_all.tex", "\n".join(rows))


def refit_summary():
    lg = pd.read_csv(REV / "refit_log.csv")
    meta = pd.read_csv(REV / "feature_counts_and_sample.csv").set_index("dataset")
    rows = []
    for k in NAMES:
        g = lg[lg.dataset == k]
        m = meta.loc[k]
        rows.append(f"{NAMES[k]} & {int(m.T_usable)} & {int(m.T0)} & {int(m.n_test)} & {int(m.n_refits)} & "
                    f"{g.ae_train_loss.min():.3f}--{g.ae_train_loss.max():.3f} & "
                    f"{g.mu_min.min():.3f}--{g.mu_max.max():.3f} \\\\")
    write("s_refit.tex", "\n".join(rows))


def eigenvalues():
    e = pd.read_csv(REV / "koopman_eigenvalues_final_window.csv")
    piv = e.pivot(index="j", columns="dataset", values="mu")
    rows = [f"{j} & " + " & ".join(f"{piv.loc[j, k]:.4f}" for k in NAMES) + r" \\" for j in piv.index]
    write("s_eigen.tex", "\n".join(rows))


def ews_table(d, name):
    perf = pd.read_csv(d / "ews_performance.csv")
    tst = pd.read_csv(d / "ews_auc_tests.csv")
    lines = []
    for k in NAMES:
        p = perf[perf.dataset == k]
        f = p.iloc[0]
        ep = int(f.event_episodes)
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{NAMES[k]} ({int(f.event_days)} event days, {ep} "
                     f"episode{'s' if ep != 1 else ''})}}}} \\\\")
        for _, r in p.iterrows():
            if r.signal.startswith("XGBoost"):
                t = tst[(tst.dataset == k) & (tst.comparison == f"AUC {r.signal} - Volatility")].iloc[0]
                dauc = f"${t['diff']:+.2f}$ [${t.ci_lo:+.2f}$, ${t.ci_hi:+.2f}$]"
                bss = f"{r.brier_skill:+.3f}"
            else:
                continue  # realised-volatility signal: not reported in the paper
            lines.append(f" & {SIGNAL[r.signal]} & {r.auc:.3f} & {dauc} & {bss} & {r.hit_rate:.2f} & "
                         f"{r.false_alarm_rate:.2f} & {r.precision:.2f} \\\\")
        if k != "u500":
            lines.append(r"\midrule")
    write(name, "\n".join(lines))


def ews_tests():
    rows = []
    for alpha, d in [("0.10", EWS10), ("0.20", EWS20)]:
        t = pd.read_csv(d / "ews_auc_tests.csv")
        for _, r in t.iterrows():
            if "Volatility" in r.comparison:
                continue  # realised-volatility signal: not reported in the paper
            comp = r.comparison.replace("AUC ", "")
            for code, lab in [("XGBoost F1", "technical"), ("XGBoost F2", "Koopman"),
                              ("XGBoost F3", "technical + Koopman"), ("F1", "technical"), ("F2", "Koopman"),
                              ("F3", "technical + Koopman"), ("Volatility", "realised volatility")]:
                comp = re.sub(rf"\b{re.escape(code)}\b", lab, comp)
            comp = comp.replace(" - ", " minus ")
            rows.append(f"{alpha} & {NAMES[r.dataset]} & {comp} & ${r['diff']:+.3f}$ & "
                        f"[${r.ci_lo:+.3f}$, ${r.ci_hi:+.3f}$] & {r.p_value:.3f} & {int(r.n_boot_valid)} \\\\")
        if alpha == "0.10":
            rows.append(r"\midrule")
    write("s_ews_tests.tex", "\n".join(rows))


def ews_thresholds():
    rows = []
    for alpha, d in [("0.10", EWS10), ("0.20", EWS20)]:
        lg = pd.read_csv(d / "ews_refit_log.csv")
        for k in NAMES:
            g = lg[lg.dataset == k]
            rows.append(f"{alpha} & {NAMES[k]} & {100 * g.q_loss.min():.2f}--{100 * g.q_loss.max():.2f} & "
                        f"{len(g)} & {g.n_train.min()}--{g.n_train.max()} \\\\")
    write("s_ews_thresholds.tex", "\n".join(rows))


def backtest():
    p = pd.read_csv(CSV / "daily_backtest" / "per_dataset.csv")
    rows = []
    prev = None
    for _, r in p.iterrows():
        key = (r["index"], r.episode)
        if prev is not None and key != prev:
            rows.append(r"\midrule")
        prev = key
        w = "--" if r.model == "Buy-and-hold" else str(int(r.window))
        hit = "--" if pd.isna(r.hit_rate) else f"{r.hit_rate:.2f}"
        rows.append(f"{tex(r['index'])} & {r.episode.replace('-', '--')} & {r.model} & {w} & {int(r.n_test)} & "
                    f"{r.turnover:.2f} & {hit} & ${r.cum_c0:.2f}$ & ${r.cum_c1:.2f}$ & ${r.cum_c5:.2f}$ \\\\")
    write("s_backtest.tex", "\n".join(rows))


if __name__ == "__main__":
    predictor_list()
    dm_all()
    refit_summary()
    eigenvalues()
    ews_table(EWS20, "s_ews_alpha20.tex")
    ews_tests()
    ews_thresholds()
    backtest()
