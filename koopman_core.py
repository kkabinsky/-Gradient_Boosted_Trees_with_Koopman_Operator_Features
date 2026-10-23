"""Helper functions shared by the experiment scripts.

The functions below are copied without change from the authors' original
koopman.py, which also contains code for an unrelated project. Only the parts
used by the scripts in this repository are kept here.
"""

import warnings

import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor

    XGB_AVAILABLE = False
    print("XGBoost not found. Falling back to GradientBoostingRegressor.")

PAPER_CMAP = LinearSegmentedColormap.from_list(
    "paper", ["#2d8a6e", "#3a7ca5", "#5c4d99", "#4a1a6b"], N=256
)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# ---------------------------------------------------------------------------
# Technical-indicator feature engineering
# ---------------------------------------------------------------------------

def resolve_close_col(df: pd.DataFrame, preferred: str = "cp") -> str:
    if preferred in df.columns:
        return preferred
    for cand in ["closed", "close", "Close", "CP"]:
        if cand in df.columns:
            print(f"  Warning: close column '{preferred}' not found; using '{cand}' instead.")
            return cand
    raise ValueError("No close price column found (expected 'cp' or 'closed').")


def build_technical_features(df: pd.DataFrame, close_col: str) -> pd.DataFrame:
    """Technical features from available columns (OHLCV if possible, else close-only)."""
    df = df.copy()

    has_ohlcv = all(c in df.columns for c in ["Open", "High", "Low", "Volume"])
    if has_ohlcv:
        df = ta.add_all_ta_features(
            df,
            open="Open",
            high="High",
            low="Low",
            close=close_col,
            volume="Volume",
            fillna=True,
        )

    close = df[close_col].astype(float)
    extra = {}

    # Close-only lags/returns (works for all datasets)
    for lag in range(1, 16):
        extra[f"{close_col}_lag_{lag}"] = close.shift(lag)
        extra[f"ret_lag_{lag}"] = close.pct_change(lag)

    # Close-only rolling stats
    for w in [3, 5, 7, 10, 14, 20, 30]:
        r = close.rolling(w)
        extra[f"close_sma_{w}"] = r.mean()
        extra[f"close_std_{w}"] = r.std()
        extra[f"close_ema_{w}"] = close.ewm(span=w, adjust=False).mean()
        extra[f"close_z_{w}"] = (close - r.mean()) / (r.std() + 1e-12)

    # Close-only TA indicators
    for w in [6, 10, 14, 20]:
        extra[f"rsi_{w}"] = ta.momentum.RSIIndicator(close=close, window=w, fillna=True).rsi()
        extra[f"roc_{w}"] = ta.momentum.ROCIndicator(close=close, window=w, fillna=True).roc()

    macd = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9, fillna=True)
    extra["macd"] = macd.macd()
    extra["macd_signal"] = macd.macd_signal()
    extra["macd_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2, fillna=True)
    extra["bb_mavg_20"] = bb.bollinger_mavg()
    extra["bb_hband_20"] = bb.bollinger_hband()
    extra["bb_lband_20"] = bb.bollinger_lband()
    extra["bb_wband_20"] = bb.bollinger_wband()
    extra["bb_pband_20"] = bb.bollinger_pband()

    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        for lag in range(1, 11):
            extra[f"volume_lag_{lag}"] = vol.shift(lag)
        for w in [5, 10, 20]:
            extra[f"volume_sma_{w}"] = vol.rolling(w).mean()
            extra[f"volume_std_{w}"] = vol.rolling(w).std()

    df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)
    return df


# ---------------------------------------------------------------------------
# Koopman autoencoder
# ---------------------------------------------------------------------------

class KoopmanAE(nn.Module):
    """Koopman autoencoder with a structured latent transition matrix."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.log_magnitude = nn.Parameter(torch.zeros(latent_dim))
        self.rotation_basis = nn.Parameter(
            torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim)
        )

    def get_K(self):
        magnitudes = torch.clamp(torch.exp(self.log_magnitude), 0.5, 1.1)
        basis, _ = torch.linalg.qr(self.rotation_basis)
        koopman_matrix = basis @ torch.diag(magnitudes) @ basis.T
        return koopman_matrix, magnitudes, basis

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_t, x_tp1):
        z_t = self.encode(x_t)
        z_tp1 = self.encode(x_tp1)

        koopman_matrix, magnitudes, basis = self.get_K()
        z_tp1_pred = z_t @ koopman_matrix.T

        x_recon = self.decode(z_t)
        x_tp1_pred = self.decode(z_tp1_pred)
        residual = z_tp1 - z_tp1_pred

        return {
            "z_t": z_t,
            "z_tp1": z_tp1,
            "z_tp1_pred": z_tp1_pred,
            "x_recon": x_recon,
            "x_tp1_pred": x_tp1_pred,
            "residual": residual,
            "K": koopman_matrix,
            "magnitudes": magnitudes,
            "basis": basis,
        }


def koopman_loss(
    outputs, x_t, x_tp1,
    w_inv: float = 10.0,
    w_recon: float = 1.0,
    w_consist: float = 1.0,
    w_spec: float = 0.5,
):
    mse = nn.MSELoss()
    loss_inv = mse(outputs["z_tp1_pred"], outputs["z_tp1"].detach())
    loss_recon = mse(outputs["x_recon"], x_t)
    loss_consist = mse(outputs["x_tp1_pred"], x_tp1)
    loss_spec = torch.mean(torch.relu(outputs["magnitudes"] - 1.05))

    total = (
        w_inv * loss_inv
        + w_recon * loss_recon
        + w_consist * loss_consist
        + w_spec * loss_spec
    )
    metrics = {
        "loss": total.item(),
        "loss_inv": loss_inv.item(),
        "loss_recon": loss_recon.item(),
        "loss_consist": loss_consist.item(),
        "loss_spec": loss_spec.item(),
    }
    return total, metrics


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def build_xgb_model(seed: int = 42):
    if XGB_AVAILABLE:
        return xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=0,
        )
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=3,
        learning_rate=0.05, random_state=seed,
    )
