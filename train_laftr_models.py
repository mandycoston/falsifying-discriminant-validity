#!/usr/bin/env python3
"""
Train LAFTR (Learning Adversarially Fair and Transferable Representations)
models for LSAC and COMPAS, following Madras et al. (2018).

Outputs prediction CSVs in the same format/order as train_fairlearn_models.py
so the R falsification scripts can consume them directly.

Usage:
  pip install -r requirements_laftr.txt
  python train_laftr_models.py

Then run from R:
  source("run_lsac_laftr.R")
  source("run_compas_laftr.R")
"""

import numpy as np
import pandas as pd
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# LAFTR model components
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z)


class Adversary(nn.Module):
    """Predicts sensitive attribute from representation (demographic parity)."""
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z)


class Reconstructor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


def train_laftr(X_train, y_train, sens_train, X_pred,
                latent_dim=32, hidden_dim=64,
                adv_weight=1.0, recon_weight=0.1,
                n_epochs=100, batch_size=256, lr=1e-3,
                adv_steps=1):
    """
    Train LAFTR with demographic-parity adversary and return P(Y=1|X) on X_pred.

    Architecture (Madras et al. 2018):
      encoder:       X -> Z
      classifier:    Z -> Y_hat
      adversary:     Z -> A_hat   (sensitive attribute)
      reconstructor: Z -> X_hat

    Loss = class_loss + recon_weight * recon_loss - adv_weight * adv_loss
    Adversary is trained to *maximise* adv_loss; encoder to *minimise* it.
    """
    device = torch.device("cpu")
    input_dim = X_train.shape[1]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_pr = scaler.transform(X_pred)

    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    s_t = torch.tensor(sens_train, dtype=torch.float32, device=device).unsqueeze(1)

    dataset = TensorDataset(X_t, y_t, s_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))

    enc = Encoder(input_dim, hidden_dim, latent_dim).to(device)
    clf = Classifier(latent_dim, hidden_dim).to(device)
    adv = Adversary(latent_dim, hidden_dim).to(device)
    rec = Reconstructor(latent_dim, hidden_dim, input_dim).to(device)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    opt_main = optim.Adam(
        list(enc.parameters()) + list(clf.parameters()) + list(rec.parameters()),
        lr=lr,
    )
    opt_adv = optim.Adam(adv.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for xb, yb, sb in loader:
            # --- adversary step(s) ---
            for _ in range(adv_steps):
                z = enc(xb).detach()
                a_logit = adv(z)
                loss_adv = bce(a_logit, sb)
                opt_adv.zero_grad()
                loss_adv.backward()
                opt_adv.step()

            # --- main step (encoder + classifier + reconstructor) ---
            z = enc(xb)
            y_logit = clf(z)
            x_hat = rec(z)
            a_logit = adv(z)

            loss_cls = bce(y_logit, yb)
            loss_rec = mse(x_hat, xb)
            loss_adv_enc = bce(a_logit, sb)

            loss_main = loss_cls + recon_weight * loss_rec - adv_weight * loss_adv_enc

            opt_main.zero_grad()
            loss_main.backward()
            opt_main.step()

    # --- predict on calib+eval ---
    enc.eval()
    clf.eval()
    with torch.no_grad():
        X_pt = torch.tensor(X_pr, dtype=torch.float32, device=device)
        z = enc(X_pt)
        logits = clf(z).squeeze(1)
        probs = torch.sigmoid(logits).numpy()

    return np.clip(probs, 1e-15, 1 - 1e-15)


# ---------------------------------------------------------------------------
# LSAC
# ---------------------------------------------------------------------------

def train_lsac_laftr():
    url = "https://raw.githubusercontent.com/damtharvey/law-school-dataset/master/law_dataset.csv"
    print("Loading LSAC from GitHub...")
    df = pd.read_csv(url)

    cols_lower = {c.lower(): c for c in df.columns}
    if "zfygpa" in cols_lower:
        df["fygpa"] = df[cols_lower["zfygpa"]]
    if "zgpa" in cols_lower:
        df["gpa"] = df[cols_lower["zgpa"]]
    if "gpa" not in df.columns and "ugpa" in df.columns:
        df["gpa"] = df["ugpa"]
    if "lsat" not in df.columns and "lsat_score" in df.columns:
        df["lsat"] = df["lsat_score"]
    if "race" not in df.columns and "racetxt" in df.columns:
        df["race"] = df["racetxt"].astype(str)

    df["fygpa_binary"] = (df["fygpa"] > df["fygpa"].median()).astype(np.float64)
    df["gpa_binary"] = (df["gpa"] > df["gpa"].median()).astype(np.float64)

    race_str = df["race"].astype(str).str.lower()
    df["race_white"] = (
        race_str.str.contains("white", na=False) | (race_str == "1")
    ).astype(np.float64)

    feature_cols = [c for c in ["lsat", "ugpa"] if c in df.columns]
    if not feature_cols:
        feature_cols = [
            c for c in df.columns
            if df[c].dtype in [np.int64, np.float64]
            and c not in ["fygpa_binary", "gpa_binary", "race_white", "fygpa"]
        ]
    X = df[feature_cols].fillna(df[feature_cols].median()).values
    y = df["fygpa_binary"].values
    sens = df["race_white"].values

    n = len(df)
    calib_size = int(n * 0.25)
    indices_file = os.path.join(SCRIPT_DIR, "lsac_fairlearn_indices.csv")
    if os.path.isfile(indices_file):
        idx_df = pd.read_csv(indices_file)
        combined_0based = idx_df["index"].values.astype(int) - 1
        calib_idx = combined_0based[:calib_size]
        eval_idx = combined_0based[calib_size:]
        train_idx = np.setdiff1d(np.arange(n), combined_0based)
        print("Using split from lsac_fairlearn_indices.csv (from R)")
    else:
        perm = np.random.RandomState(SEED).permutation(n)
        train_size = int(n * 0.5)
        train_idx = perm[:train_size]
        calib_idx = perm[train_size: train_size + calib_size]
        eval_idx = perm[train_size + calib_size:]
        print("Using Python split")

    X_train, y_train, sens_train = X[train_idx], y[train_idx], sens[train_idx]
    X_calib_eval = np.vstack([X[calib_idx], X[eval_idx]])

    print("Training LAFTR (demographic parity) on LSAC...")
    proba = train_laftr(
        X_train, y_train, sens_train, X_calib_eval,
        latent_dim=32, hidden_dim=64,
        adv_weight=1.0, recon_weight=0.1,
        n_epochs=150, batch_size=256, lr=1e-3,
    )
    out = pd.DataFrame({"prediction": proba})
    out.to_csv("predictions_lsac_laftr.csv", index=False)
    print(f"Wrote predictions_lsac_laftr.csv  ({len(proba)} rows = calib then eval)")
    return True


# ---------------------------------------------------------------------------
# COMPAS
# ---------------------------------------------------------------------------

def train_compas_laftr():
    data_file = os.path.join(SCRIPT_DIR, "compas-analysis", "compas-scores-two-years.csv")
    if not os.path.isfile(data_file):
        alt = os.path.join(SCRIPT_DIR, "compas-analysis", "compas-scores.csv")
        if os.path.isfile(alt):
            data_file = alt
        else:
            print(
                "COMPAS data not found. Clone:\n"
                "  git clone https://github.com/propublica/compas-analysis.git compas-analysis"
            )
            return False

    print("Loading COMPAS from", data_file)
    df = pd.read_csv(data_file)

    recid_col = next(
        (c for c in ["two_year_recid", "is_recid", "recid"] if c in df.columns), None
    )
    if recid_col is None:
        print("Recidivism column not found")
        return False
    df["two_year_recid"] = pd.to_numeric(df[recid_col], errors="coerce")
    if not df["two_year_recid"].dropna().isin([0, 1]).all():
        df["two_year_recid"] = (
            df["two_year_recid"] > df["two_year_recid"].median()
        ).astype(np.float64)
    df["two_year_recid"] = df["two_year_recid"].fillna(0).astype(np.float64)

    if "age" in df.columns:
        df["age_cat_Lessthan25"] = (df["age"] < 25).astype(np.float64)
    else:
        print("Age column not found")
        return False

    if "race" in df.columns:
        df["race_aa"] = (
            df["race"].astype(str).str.contains("African|Black|Afro", case=False, na=False)
        ).astype(np.float64)
    else:
        print("Race column not found")
        return False

    feat_candidates = ["age", "priors_count", "juv_fel_count", "juv_misd_count"]
    feature_cols = [c for c in feat_candidates if c in df.columns]
    if not feature_cols:
        feature_cols = [
            c for c in df.columns
            if df[c].dtype in [np.int64, np.float64]
            and c not in ["two_year_recid", "race_aa", "age_cat_Lessthan25"]
        ]
    X = df[feature_cols].fillna(0).values
    y = df["two_year_recid"].values
    sens = df["race_aa"].values

    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isnan(sens))
    X, y, sens = X[valid], y[valid], sens[valid]

    n = len(y)
    calib_size = int(n * 0.25)
    indices_file = os.path.join(SCRIPT_DIR, "compas_fairlearn_indices.csv")
    if os.path.isfile(indices_file):
        idx_df = pd.read_csv(indices_file)
        combined_0based = idx_df["index"].values.astype(int) - 1
        calib_idx = combined_0based[:calib_size]
        eval_idx = combined_0based[calib_size:]
        train_idx = np.setdiff1d(np.arange(n), combined_0based)
        print("Using split from compas_fairlearn_indices.csv (from R)")
    else:
        perm = np.random.RandomState(SEED).permutation(n)
        train_size = int(n * 0.5)
        train_idx = perm[:train_size]
        calib_idx = perm[train_size: train_size + calib_size]
        eval_idx = perm[train_size + calib_size:]
        print("Using Python split")

    X_train, y_train, sens_train = X[train_idx], y[train_idx], sens[train_idx]
    X_calib_eval = np.vstack([X[calib_idx], X[eval_idx]])

    print("Training LAFTR (demographic parity) on COMPAS...")
    proba = train_laftr(
        X_train, y_train, sens_train, X_calib_eval,
        latent_dim=32, hidden_dim=64,
        adv_weight=1.0, recon_weight=0.1,
        n_epochs=150, batch_size=256, lr=1e-3,
    )
    out = pd.DataFrame({"prediction": proba})
    out.to_csv("predictions_compas_laftr.csv", index=False)
    print(f"Wrote predictions_compas_laftr.csv  ({len(proba)} rows = calib then eval)")
    return True


# ---------------------------------------------------------------------------

def main():
    print("Training LAFTR models for LSAC and COMPAS...\n")
    ok_lsac = train_lsac_laftr()
    print()
    ok_compas = train_compas_laftr()
    print("\nDone.")
    if ok_lsac:
        print("  Run in R: source('run_lsac_laftr.R')")
    if ok_compas:
        print("  Run in R: source('run_compas_laftr.R')")
    return 0 if (ok_lsac and ok_compas) else 1


if __name__ == "__main__":
    sys.exit(main())
