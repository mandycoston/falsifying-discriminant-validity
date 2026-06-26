#!/usr/bin/env python3
"""
Train two LAFTR models on LSAC:
  1) adversary targets race   (already saved, retrain if missing)
  2) adversary targets gender  (new)

Then compute the same metrics as Table 1 in the paper for both models,
evaluated against every proxy: 1styearGPA, GPA, Pass Bar, Race, Gender.

Outputs:
  predictions_lsac_laftr_gender.csv   (new gender-adversary predictions)
  lsac_laftr_table1_metrics.csv       (Table-1-style metrics for both models)
"""

import numpy as np
import pandas as pd
import os, sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# LAFTR components (same as train_laftr_models.py)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, d_in, d_hid, d_lat):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Linear(d_hid, d_lat), nn.ReLU())
    def forward(self, x): return self.net(x)

class Head(nn.Module):
    def __init__(self, d_lat, d_hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_lat, d_hid), nn.ReLU(),
                                 nn.Linear(d_hid, 1))
    def forward(self, z): return self.net(z)

class Reconstructor(nn.Module):
    def __init__(self, d_lat, d_hid, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_lat, d_hid), nn.ReLU(),
                                 nn.Linear(d_hid, d_out))
    def forward(self, z): return self.net(z)


def train_laftr(X_train, y_train, sens_train, X_pred,
                latent_dim=32, hidden_dim=64,
                adv_weight=1.0, recon_weight=0.1,
                n_epochs=150, batch_size=256, lr=1e-3):
    device = torch.device("cpu")
    d_in = X_train.shape[1]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_pr = scaler.transform(X_pred)

    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(sens_train, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))

    enc = Encoder(d_in, hidden_dim, latent_dim)
    clf = Head(latent_dim, hidden_dim)
    adv = Head(latent_dim, hidden_dim)
    rec = Reconstructor(latent_dim, hidden_dim, d_in)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    opt_main = optim.Adam(list(enc.parameters()) + list(clf.parameters()) +
                          list(rec.parameters()), lr=lr)
    opt_adv = optim.Adam(adv.parameters(), lr=lr)

    for _ in range(n_epochs):
        for xb, yb, sb in loader:
            z = enc(xb).detach()
            opt_adv.zero_grad()
            bce(adv(z), sb).backward()
            opt_adv.step()

            z = enc(xb)
            loss = (bce(clf(z), yb)
                    + recon_weight * mse(rec(z), xb)
                    - adv_weight * bce(adv(z), sb))
            opt_main.zero_grad()
            loss.backward()
            opt_main.step()

    enc.eval(); clf.eval()
    with torch.no_grad():
        logits = clf(enc(torch.tensor(X_pr, dtype=torch.float32))).squeeze(1)
        probs = torch.sigmoid(logits).numpy()
    return np.clip(probs, 1e-15, 1 - 1e-15)


# ---------------------------------------------------------------------------
# Metric helpers (matching Table 1 in the paper)
# ---------------------------------------------------------------------------

def compute_auc(pred, labels):
    if len(np.unique(labels)) < 2: return np.nan
    return roc_auc_score(labels, pred)

def compute_aupr(pred, labels):
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels): return np.nan
    idx = np.argsort(-pred)
    lab = labels[idx]
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    prec = tp / (tp + fp)
    rec = tp / n_pos
    ur = np.unique(rec); up = np.array([prec[rec == r].max() for r in ur])
    si = np.argsort(ur); ur, up = ur[si], up[si]
    return float(np.trapz(up, ur))

def compute_mse(pred, labels):
    return float(np.mean((pred - labels) ** 2))

def compute_ppv_topk(pred, labels, k_pct):
    k = max(1, int(len(pred) * k_pct / 100))
    top_idx = np.argsort(-pred)[:k]
    return float(labels[top_idx].mean())

def metrics_row(pred, labels, proxy_name):
    return {
        "Proxy": proxy_name,
        "AUC": compute_auc(pred, labels),
        "AU PR": compute_aupr(pred, labels),
        "MSE": compute_mse(pred, labels),
        "PPV Top 2%": compute_ppv_topk(pred, labels, 2),
        "PPV Top 10%": compute_ppv_topk(pred, labels, 10),
        "PPV Top 50%": compute_ppv_topk(pred, labels, 50),
        "PPV Top 75%": compute_ppv_topk(pred, labels, 75),
    }


# ---------------------------------------------------------------------------
# Load LSAC + split (same as train_laftr_models.py)
# ---------------------------------------------------------------------------

url = "https://raw.githubusercontent.com/damtharvey/law-school-dataset/master/law_dataset.csv"
print("Loading LSAC…")
df = pd.read_csv(url)

cols_lower = {c.lower(): c for c in df.columns}
if "zfygpa" in cols_lower: df["fygpa"] = df[cols_lower["zfygpa"]]
if "zgpa" in cols_lower:   df["gpa"]   = df[cols_lower["zgpa"]]
if "gpa" not in df.columns and "ugpa" in df.columns: df["gpa"] = df["ugpa"]
if "lsat" not in df.columns and "lsat_score" in df.columns: df["lsat"] = df["lsat_score"]
if "race" not in df.columns and "racetxt" in df.columns: df["race"] = df["racetxt"].astype(str)

df["fygpa_binary"] = (df["fygpa"] > df["fygpa"].median()).astype(np.float64)
df["gpa_binary"]   = (df["gpa"]   > df["gpa"].median()).astype(np.float64)

race_str = df["race"].astype(str).str.lower()
df["race_white"] = (race_str.str.contains("white", na=False) | (race_str == "1")).astype(np.float64)
df["male"] = df["male"].astype(np.float64)

feature_cols = [c for c in ["lsat", "ugpa"] if c in df.columns]
X = df[feature_cols].fillna(df[feature_cols].median()).values
y = df["fygpa_binary"].values

n = len(df)
calib_size = int(n * 0.25)

idx_file = os.path.join(SCRIPT_DIR, "lsac_fairlearn_indices.csv")
if os.path.isfile(idx_file):
    combined_0 = pd.read_csv(idx_file)["index"].values.astype(int) - 1
    calib_idx = combined_0[:calib_size]
    eval_idx  = combined_0[calib_size:]
    train_idx = np.setdiff1d(np.arange(n), combined_0)
    print("Using split from lsac_fairlearn_indices.csv")
else:
    perm = np.random.RandomState(SEED).permutation(n)
    train_size = int(n * 0.5)
    train_idx = perm[:train_size]
    calib_idx = perm[train_size:train_size + calib_size]
    eval_idx  = perm[train_size + calib_size:]

X_train = X[train_idx]
y_train = y[train_idx]
X_calib_eval = np.vstack([X[calib_idx], X[eval_idx]])
eval_rel = slice(len(calib_idx), len(calib_idx) + len(eval_idx))

# Eval-set labels for every proxy
eval_data = df.iloc[eval_idx]
labels = {
    "1styearGPA": eval_data["fygpa_binary"].values,
    "GPA":        eval_data["gpa_binary"].values,
    "Pass Bar":   eval_data["pass_bar"].values.astype(float),
    "Race":       eval_data["race_white"].values,
    "Gender":     eval_data["male"].values,
}


# ---------------------------------------------------------------------------
# Model 1: LAFTR adversary = race
# ---------------------------------------------------------------------------

pred_file_race = "predictions_lsac_laftr.csv"
if os.path.isfile(pred_file_race):
    print("Loading existing LAFTR-race predictions…")
    preds_race = pd.read_csv(pred_file_race)["prediction"].values[eval_rel]
else:
    print("Training LAFTR (adversary=race) on LSAC…")
    torch.manual_seed(SEED)
    sens_race = df["race_white"].values[train_idx]
    all_preds = train_laftr(X_train, y_train, sens_race, X_calib_eval)
    pd.DataFrame({"prediction": all_preds}).to_csv(pred_file_race, index=False)
    preds_race = all_preds[eval_rel]

print(f"  LAFTR-race eval predictions: n={len(preds_race)}, "
      f"mean={preds_race.mean():.4f}")


# ---------------------------------------------------------------------------
# Model 2: LAFTR adversary = gender
# ---------------------------------------------------------------------------

pred_file_gender = "predictions_lsac_laftr_gender.csv"
print("Training LAFTR (adversary=gender) on LSAC…")
torch.manual_seed(SEED)
sens_gender = df["male"].values[train_idx]
all_preds_g = train_laftr(X_train, y_train, sens_gender, X_calib_eval)
pd.DataFrame({"prediction": all_preds_g}).to_csv(pred_file_gender, index=False)
preds_gender = all_preds_g[eval_rel]

print(f"  LAFTR-gender eval predictions: n={len(preds_gender)}, "
      f"mean={preds_gender.mean():.4f}")


# ---------------------------------------------------------------------------
# Compute Table 1 metrics for both models
# ---------------------------------------------------------------------------

rows = []
for proxy_name, lab in labels.items():
    rows.append({"Model": "LAFTR (adv=race)", **metrics_row(preds_race, lab, proxy_name)})
    rows.append({"Model": "LAFTR (adv=gender)", **metrics_row(preds_gender, lab, proxy_name)})

results = pd.DataFrame(rows)
results.to_csv("lsac_laftr_table1_metrics.csv", index=False)

pd.set_option("display.float_format", lambda x: f"{x:.4f}")
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 20)

print("\n" + "=" * 140)
print("TABLE 1 METRICS FOR LAFTR MODELS (LSAC, eval set)")
print("=" * 140)

for model_name in ["LAFTR (adv=race)", "LAFTR (adv=gender)"]:
    sub = results[results["Model"] == model_name].drop(columns="Model")
    print(f"\n--- {model_name} ---")
    print(sub.to_string(index=False))

print(f"\nSaved to lsac_laftr_table1_metrics.csv")
