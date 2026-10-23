"""Collect the saved results into Excel workbooks.

results/excel/paper_tables.xlsx
    One sheet per table (or result paragraph) of the paper and its
    supplementary material, laid out as in the paper.
results/excel/<experiment>_raw.xlsx
    Every CSV written by the experiment scripts, one sheet per file.

Run after koopman_revision.py, koopman_ews.py (alpha 0.10 and 0.20) and
daily_backtest.py. Reads results/csv/ only.
"""

from pathlib import Path

import pandas as pd

from replot_importance import readable

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "csv"
XLSX = HERE / "results" / "excel"
NAMES = {"dax": "DAX", "u30": "DJIA", "u500": "S&P 500"}
SIGNAL = {"XGBoost F1": "XGBoost, technical", "XGBoost F2": "XGBoost, Koopman",
          "XGBoost F3": "XGBoost, technical + Koopman"}
SET_ROWS = {"zero": "Zero-return forecast", "hist_mean": "Historical mean",
            "F0": "XGBoost, lagged returns", "F1": "XGBoost, technical predictors",
            "F2": "XGBoost, Koopman features", "F3": "XGBoost, technical + Koopman"}
SHORT = {"F0": "lagged returns", "F1": "technical", "F2": "Koopman", "F3": "technical + Koopman",
         "zero": "zero-return forecast", "hist_mean": "historical mean"}


def comparison_label(c):
    """'AUC XGBoost F3 - F1' -> 'technical + Koopman minus technical'."""
    c = c.replace("AUC ", "").replace("XGBoost ", "")
    a, b = [s.strip() for s in c.split(" - ")]
    return f"{SHORT.get(a, a)} minus {SHORT.get(b, b)}"


def table4():
    m = pd.read_csv(CSV / "predictor_sets" / "feature_counts_and_sample.csv")
    rows = [("Lagged returns", "n_F0"), ("Technical", "n_F1"), ("Koopman", "n_F2"), ("Technical + Koopman", "n_F3")]
    out = pd.DataFrame({"Predictor set": [r[0] for r in rows]})
    for _, r in m.iterrows():
        out[NAMES[r.dataset]] = [int(r[c]) for _, c in rows]
    return out


def table9():
    t = pd.read_csv(CSV / "daily_backtest" / "table_netcost_daily.csv")
    out = pd.DataFrame({"Model": t.model, "Window": t.window.astype(int).astype(str).replace("0", "--"),
                        "Turnover": t.turnover.round(2)})
    for c in (0, 1, 5):
        out[f"Cum. return (%), c={c} bp"] = t[f"cum_c{c}"].round(2)
        out[f"95% CI low, c={c}"] = t[f"ci_lo_c{c}"].round(1)
        out[f"95% CI high, c={c}"] = t[f"ci_hi_c{c}"].round(1)
    return out


def table10():
    r = pd.read_csv(CSV / "predictor_sets" / "rmse_table.csv").set_index("dataset")
    dm = pd.read_csv(CSV / "predictor_sets" / "dm_tests.csv")
    rows = []
    for k, lab in SET_ROWS.items():
        rows.append({"Row": lab, **{NAMES[d]: round(100 * r.loc[d, k], 4) for d in NAMES}})
    for comp, lab in [("F2 vs F1", "DM: Koopman vs technical (p-value)"),
                      ("F3 vs F1", "DM: technical + Koopman vs technical (p-value)"),
                      ("F1 vs F0", "DM: technical vs lagged returns (p-value)")]:
        g = dm[dm.comparison == comp].set_index("dataset")
        rows.append({"Row": lab, **{NAMES[d]: f"{g.loc[d, 'DM_HLN']:.3f} ({g.loc[d, 'p_value']:.3f})" for d in NAMES}})
    out = pd.DataFrame(rows)
    out.columns = ["Row (RMSE x 10^-2; DM statistic with p-value)"] + list(NAMES.values())
    return out


def ews_performance(alpha_dir, with_dauc):
    perf = pd.read_csv(CSV / alpha_dir / "ews_performance.csv")
    tst = pd.read_csv(CSV / alpha_dir / "ews_auc_tests.csv")
    rows = []
    for _, r in perf.iterrows():
        if not r.signal.startswith("XGBoost"):
            continue  # realised-volatility signal: computed but not reported in the paper
        row = {"Index": NAMES[r.dataset], "Event days": int(r.event_days), "Episodes": int(r.event_episodes),
               "Signal": SIGNAL[r.signal], "AUC": round(r.auc, 3)}
        if with_dauc:
            t = tst[(tst.dataset == r.dataset) & (tst.comparison == f"AUC {r.signal} - Volatility")].iloc[0]
            row.update({"Delta AUC vs realised volatility": round(t["diff"], 2),
                        "Delta AUC 95% CI low": round(t.ci_lo, 2), "Delta AUC 95% CI high": round(t.ci_hi, 2)})
        row.update({"BSS": round(r.brier_skill, 3), "Hit": round(r.hit_rate, 2),
                    "FA": round(r.false_alarm_rate, 2), "Prec.": round(r.precision, 2)})
        rows.append(row)
    return pd.DataFrame(rows)


def ews_tests():
    rows = []
    for alpha, d in [("0.10", "early_warning_alpha0.10"), ("0.20", "early_warning_alpha0.20")]:
        t = pd.read_csv(CSV / d / "ews_auc_tests.csv")
        for _, r in t.iterrows():
            if "Volatility" in r.comparison:
                continue  # realised-volatility signal: not reported in the paper
            rows.append({"alpha": alpha, "Index": NAMES[r.dataset], "Difference in AUC": comparison_label(r.comparison),
                         "Estimate": round(r["diff"], 3), "95% CI low": round(r.ci_lo, 3),
                         "95% CI high": round(r.ci_hi, 3), "p-value": round(r.p_value, 3),
                         "Valid resamples": int(r.n_boot_valid)})
    return pd.DataFrame(rows)


def ews_thresholds():
    rows = []
    for alpha, d in [("0.10", "early_warning_alpha0.10"), ("0.20", "early_warning_alpha0.20")]:
        lg = pd.read_csv(CSV / d / "ews_refit_log.csv")
        for k, lab in NAMES.items():
            g = lg[lg.dataset == k]
            rows.append({"alpha": alpha, "Index": lab, "q_tau min (%)": round(100 * g.q_loss.min(), 2),
                         "q_tau max (%)": round(100 * g.q_loss.max(), 2), "Re-estimations": len(g),
                         "Training size min": int(g.n_train.min()), "Training size max": int(g.n_train.max())})
    return pd.DataFrame(rows)


def predictors():
    sets = {k: pd.read_csv(CSV / "predictor_sets" / f"F1_features_{k}.csv")["feature"].tolist() for k in NAMES}
    allf = list(dict.fromkeys(sum(sets.values(), [])))
    return pd.DataFrame({"Predictor (code name)": allf, "Description": [readable(f) for f in allf],
                         **{NAMES[k]: ["x" if f in sets[k] else "" for f in allf] for k in NAMES}})


def refit_summary():
    lg = pd.read_csv(CSV / "predictor_sets" / "refit_log.csv")
    meta = pd.read_csv(CSV / "predictor_sets" / "feature_counts_and_sample.csv").set_index("dataset")
    rows = []
    for k, lab in NAMES.items():
        g, m = lg[lg.dataset == k], meta.loc[k]
        rows.append({"Index": lab, "Usable observations": int(m.T_usable), "Initial training size": int(m.T0),
                     "Test forecasts": int(m.n_test), "Re-estimations": int(m.n_refits),
                     "AE training loss min": round(g.ae_train_loss.min(), 3),
                     "AE training loss max": round(g.ae_train_loss.max(), 3),
                     "Eigenvalue min": round(g.mu_min.min(), 3), "Eigenvalue max": round(g.mu_max.max(), 3)})
    return pd.DataFrame(rows)


def dm_all():
    dm = pd.read_csv(CSV / "predictor_sets" / "dm_tests.csv")
    rm = pd.read_csv(CSV / "predictor_sets" / "rmse_table.csv").set_index("dataset")
    rows = []
    for _, r in dm.iterrows():
        a, b = r.comparison.split(" vs ")
        rows.append({"Index": NAMES[r.dataset], "Comparison": f"{SHORT[a]} vs {SHORT[b]}",
                     "RMSE first x100": round(100 * rm.loc[r.dataset, a], 4),
                     "RMSE second x100": round(100 * rm.loc[r.dataset, b], 4),
                     "DM (HLN)": round(r.DM_HLN, 3), "p-value": round(r.p_value, 4)})
    return pd.DataFrame(rows)


def eigenvalues():
    e = pd.read_csv(CSV / "predictor_sets" / "koopman_eigenvalues_final_window.csv")
    piv = e.pivot(index="j", columns="dataset", values="mu")[list(NAMES)].round(4)
    piv.columns = [NAMES[c] for c in piv.columns]
    return piv.reset_index().rename(columns={"j": "j (eigenvalue order)"})


def importance():
    frames = []
    for k, lab in NAMES.items():
        imp = pd.read_csv(CSV / "predictor_sets" / f"importance_F3_{k}.csv").head(20)
        frames.append(pd.DataFrame({"Index": lab, "Rank": range(1, len(imp) + 1),
                                    "Predictor": imp.feature.map(readable), "Code name": imp.feature,
                                    "Gain importance": imp.importance.round(4)}))
    return pd.concat(frames, ignore_index=True)


def backtest_per_dataset():
    p = pd.read_csv(CSV / "daily_backtest" / "per_dataset.csv")
    return p.rename(columns={"cum_c0": "Cum. return (%) c=0", "cum_c1": "Cum. return (%) c=1",
                             "cum_c5": "Cum. return (%) c=5"})


def autosize(writer, sheet, df):
    ws = writer.sheets[sheet]
    for i, col in enumerate(df.columns, start=1):
        width = max([len(str(col))] + [len(str(v)) for v in df.iloc[:, i - 1].tolist()[:200]])
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(max(10, width + 2), 60)


def main():
    XLSX.mkdir(parents=True, exist_ok=True)
    sheets = {
        "Table4_predictor_sets": table4(),
        "Table9_netcost_daily": table9(),
        "Table10_RMSE_DM": table10(),
        "Sec5.5_EWS_alpha0.10": ews_performance("early_warning_alpha0.10", with_dauc=False),
        "Fig4_gain_importance": importance(),
        "S1_predictors": predictors(),
        "S2_refit_samples": refit_summary(),
        "S3_DM_all": dm_all(),
        "S4_eigenvalues": eigenvalues(),
        "S5_EWS_thresholds": ews_thresholds(),
        "S6_EWS_alpha0.20": ews_performance("early_warning_alpha0.20", with_dauc=True),
        "S7_EWS_AUC_tests": ews_tests(),
        "S8_backtest_per_dataset": backtest_per_dataset(),
    }
    with pd.ExcelWriter(XLSX / "paper_tables.xlsx", engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
            autosize(w, name, df)
    print("wrote", XLSX / "paper_tables.xlsx")

    for sub in sorted(p for p in CSV.iterdir() if p.is_dir()):
        out = XLSX / f"{sub.name}_raw.xlsx"
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            for f in sorted(sub.glob("*.csv")):
                df = pd.read_csv(f)
                name = f.stem[:31]
                df.to_excel(w, sheet_name=name, index=False)
                autosize(w, name, df)
        print("wrote", out)


if __name__ == "__main__":
    main()
