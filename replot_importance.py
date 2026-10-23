"""Redraw Figure 4 (feature_score_koopman_rev) with readable predictor names.

Reads results/csv/predictor_sets/importance_F3_*.csv written by koopman_revision.py.
"""

import re
from pathlib import Path

import pandas as pd

from koopman_revision import DATASETS, plot_importance_panels

HERE = Path(__file__).resolve().parent
SRC = HERE / "results" / "csv" / "predictor_sets"

FIXED = {
    "koop_resid_lag": "Koopman residual (lagged)",
    "macd": "MACD line", "macd_signal": "MACD signal line", "macd_diff": "MACD histogram",
    "bb_mavg_20": "Bollinger middle band (20d)", "bb_hband_20": "Bollinger upper band (20d)",
    "bb_lband_20": "Bollinger lower band (20d)", "bb_wband_20": "Bollinger band width (20d)",
    "bb_pband_20": "Bollinger %B (20d)",
    "return": "Daily return (source)", "Return of u30": "Daily return (source)",
    "Return of U500": "Daily return (source)",
    "direction of dax": "Direction indicator (source)", "direction of u30": "Direction indicator (source)",
    "direction of U500": "Direction indicator (source)",
    "Crash Indicator dax (10%)": "Crash indicator (source)",
}
PATTERNS = [
    (r"z_(\d+)", "Koopman latent coordinate {}"),
    (r"phi_(\d+)", "Koopman eigen-coordinate {}"),
    (r"ret_lag_(\d+)", "{}-day return"),
    (r"close_lag_(\d+)", "Close, lag {} days"),
    (r"close_sma_(\d+)", "Moving average of close ({}d)"),
    (r"close_ema_(\d+)", "Exponential moving average ({}d)"),
    (r"close_std_(\d+)", "Rolling std. of close ({}d)"),
    (r"close_z_(\d+)", "z-score of close ({}d)"),
    (r"rsi_(\d+)", "RSI ({}d)"),
    (r"roc_(\d+)", "Rate of change ({}d)"),
]


def readable(name):
    if name in FIXED:
        return FIXED[name]
    for pat, fmt in PATTERNS:
        m = re.fullmatch(pat, name)
        if m:
            return fmt.format(m.group(1))
    raise KeyError(f"no readable name for {name}")


def main():
    results = []
    for key in DATASETS:
        imp = pd.read_csv(SRC / f"importance_F3_{key}.csv")
        imp["feature"] = imp["feature"].map(readable)
        results.append({"key": key, "imp": imp})
    stem = HERE / "results" / "figures" / "feature_score_koopman_rev"
    plot_importance_panels(results, stem)
    print(f"Saved {stem}.eps/.jpg")


if __name__ == "__main__":
    main()
