"""
LASSO Logistic Regression — BNPL Adoption & Delinquency
========================================================
Two parallel models:
  Model A: Full data 2022-2025 (25 features)
  Model B: 2025 only           (26 features, includes EF5D)

Imputation: mode per column, applied only to feature columns
            (never touches BNPL1, BNPL3, META columns)

Targets:
  BNPL1 — adoption       (full sample)
  BNPL3 — delinquency    (BNPL users only)

Outputs → lasso_outputs/
    01_roc_curves.png
    02_coef_adoption_full.png
    03_coef_adoption_2025.png
    04_coef_delinquency_full.png
    05_coef_delinquency_2025.png
    06_results_summary.csv
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, classification_report

warnings.filterwarnings("ignore")
os.makedirs("lasso_outputs", exist_ok=True)

BLUE = "#185FA5"; RED = "#A32D2D"; TEAL = "#0F6E56"; GRAY = "#4A4A4A"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

# ══════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════
df = pd.read_csv("data_merged.csv", low_memory=False)

EXCLUDE = ["BNPL1","BNPL3","Unnamed: 0","weight","year",
           "BNPL4_a","BNPL4_b","BNPL4_c","BNPL4_d","BNPL4_e","BNPL4_f",
           "BNPL1A","BNPL5"]

# ══════════════════════════════════════════════════════════════
# 2. ENCODING  (shared ordinal / binary / nominal maps)
# ══════════════════════════════════════════════════════════════
ordinal_maps = {
    "ppfs1482": {"Very poor":1,"Poor":2,"Fair":3,"Good":4,"Excellent":5,
                 "Don\u2019t know":np.nan},
    "A6":       {"Not confident":1,"Somewhat confident":2,"Very confident":3,
                 "Don\u2019t know":np.nan},
    "ppinc7":   {"Less than $10,000":1,"$10,000 to $24,999":2,
                 "$25,000 to $49,999":3,"$50,000 to $74,999":4,
                 "$75,000 to $99,999":5,"$100,000 to $149,999":6,
                 "$150,000 or more":7},
    "ppeducat": {"No high school diploma or GED":1,
                 "High school graduate (high school diploma or the equivalent GED)":2,
                 "Some college or Associate\u2019s degree":3,
                 "Bachelor\u2019s degree or higher":4},
    "I20":      {"Less than your income":1,"The same as your income":2,
                 "More than your income":3},
    "C4A":      {"Never carried an unpaid balance (always pay in full)":1,
                 "Once":2,"Some of the time":3,"Most or all of the time":4},
    "ppfsasset":{"Under $50,000":1,"$50,000 - $99,999":2,
                 "$100,000 - $249,999":3,"$250,000 - $499,999":4,
                 "$500,000 - $999,999":5,"$1,000,000 or more":6,
                 "Not sure":np.nan},
    "INF4":     {"Much worse":1,"Somewhat worse":2,"Little or no effect":3,
                 "Somewhat better":4,"Much better":5},
    "C3P":      {"Did not pay or paid less than the minimum payment on at least one card":1,
                 "Paid at least the minimum payment on all credit cards":2,
                 "Did not use any of my credit cards so had no balances":3},
}

def encode(data):
    """Encode a dataframe slice. Returns encoded df, does not touch EXCLUDE cols."""
    d = data.copy()
    for col, m in ordinal_maps.items():
        if col in d.columns:
            d[col] = d[col].map(m)
    for c in [c for c in d.columns if c not in EXCLUDE
              and d[c].dropna().isin(["Yes","No"]).all()]:
        d[c] = d[c].map({"Yes":1,"No":0})
    nominal = [c for c in ["ppethm","ppgender","ppemploy","ppmarit5"]
               if c in d.columns]
    d = pd.get_dummies(d, columns=nominal, drop_first=True, dtype=float)
    return d

# ══════════════════════════════════════════════════════════════
# 3. PREPARE DATASET  — drop high-miss cols, mode-impute rest
# ══════════════════════════════════════════════════════════════
def prepare(data, label):
    """
    1. Identify feature columns (not EXCLUDE)
    2. Drop cols with >50% missing across this slice
    3. Encode
    4. Mode-impute remaining NA in feature cols only
    """
    print(f"\n── {label} ──")
    feat_cols = [c for c in data.columns if c not in EXCLUDE]

    # Drop >50% missing features
    miss       = data[feat_cols].isna().mean()
    drop_cols  = miss[miss > 0.50].index.tolist()
    feat_cols  = [c for c in feat_cols if c not in drop_cols]
    print(f"  Dropped (>50% missing): {drop_cols}")

    # Encode
    working   = encode(data[feat_cols + ["BNPL1","BNPL3","weight"]].copy())
    feat_cols = [c for c in working.columns
                 if c not in ["BNPL1","BNPL3","weight"]]

    # Mode-impute feature columns only
    n_missing = working[feat_cols].isna().sum().sum()
    if n_missing > 0:
        for c in feat_cols:
            if working[c].isna().any():
                mode_val = working[c].mode()[0]
                working[c] = working[c].fillna(mode_val)
        print(f"  Mode-imputed {n_missing} values across feature cols")

    X = StandardScaler().fit_transform(working[feat_cols].values)

    # BNPL1 — adoption
    y1 = (working["BNPL1"] == "Yes").astype(int).values
    w1 = working["weight"].values

    # BNPL3 — delinquency (BNPL users only; BNPL3 NA = non-user, not imputed)
    mask   = working["BNPL1"] == "Yes"
    y3     = (working.loc[mask, "BNPL3"] == "Yes").astype(int).values
    X3     = X[mask]
    w3     = w1[mask]

    print(f"  Adoption model  — N={len(y1):,}  positive={y1.mean():.1%}")
    print(f"  Delinquency model — N={len(y3):,}  positive={y3.mean():.1%}")
    return X, y1, w1, X3, y3, w3, feat_cols

X_A, y1_A, w1_A, X3_A, y3_A, w3_A, feats_A = prepare(df, "Model A — Full 2022-2025")
X_B, y1_B, w1_B, X3_B, y3_B, w3_B, feats_B = prepare(df[df["year"]==2025].copy(), "Model B — 2025 only")

# ══════════════════════════════════════════════════════════════
# 4. LASSO FIT
# ══════════════════════════════════════════════════════════════
def fit_lasso(X, y, weights, label):
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegressionCV(
        Cs=20, cv=cv, penalty="l1", solver="liblinear",
        scoring="roc_auc", max_iter=1000, random_state=42, n_jobs=-1,
    )
    model.fit(X, y, sample_weight=weights)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    auc    = roc_auc_score(y, y_prob, sample_weight=weights)
    f1     = f1_score(y, y_pred, sample_weight=weights)
    n_sel  = int(np.sum(model.coef_[0] != 0))
    print(f"\n  {label}")
    print(f"    Best C={model.C_[0]:.4f} | AUC={auc:.4f} | F1={f1:.4f} | "
          f"Selected {n_sel}/{X.shape[1]} features")
    print(classification_report(y, y_pred, sample_weight=weights,
                                target_names=["No","Yes"], digits=3))
    return model, auc, f1

print("\n══ Fitting models ══")
m_adopt_A,  auc_a1, f1_a1 = fit_lasso(X_A,  y1_A, w1_A, "Adoption — Full")
m_adopt_B,  auc_b1, f1_b1 = fit_lasso(X_B,  y1_B, w1_B, "Adoption — 2025")
m_delin_A,  auc_a3, f1_a3 = fit_lasso(X3_A, y3_A, w3_A, "Delinquency — Full")
m_delin_B,  auc_b3, f1_b3 = fit_lasso(X3_B, y3_B, w3_B, "Delinquency — 2025")

# ══════════════════════════════════════════════════════════════
# 5. PLOTS
# ══════════════════════════════════════════════════════════════
def coef_plot(model, features, title, filename, top_n=20):
    coefs  = pd.Series(model.coef_[0], index=features)
    coefs  = coefs[coefs != 0]
    if len(coefs) == 0:
        print(f"  No non-zero coefficients for {filename}"); return
    coefs  = coefs.reindex(
        coefs.abs().sort_values(ascending=False).head(top_n).index
    ).sort_values()
    colors = [BLUE if v > 0 else RED for v in coefs.values]

    fig, ax = plt.subplots(figsize=(9, max(4, len(coefs) * 0.38)))
    bars = ax.barh(coefs.index, coefs.values, color=colors, height=0.65)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("LASSO Coefficient (standardised)")
    ax.set_title(title)
    for bar, val in zip(bars, coefs.values):
        ax.text(val + (0.003 if val >= 0 else -0.003),
                bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8,
                ha="left" if val >= 0 else "right", color=GRAY)
    ax.legend(handles=[
        mpatches.Patch(color=BLUE, label="Increases probability"),
        mpatches.Patch(color=RED,  label="Decreases probability"),
    ], fontsize=9)
    plt.tight_layout()
    fig.savefig(f"lasso_outputs/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {filename}")

coef_plot(m_adopt_A, feats_A,
          f"Adoption — Full 2022-2025  (AUC={auc_a1:.3f})",
          "02_coef_adoption_full.png")
coef_plot(m_adopt_B, feats_B,
          f"Adoption — 2025 only  (AUC={auc_b1:.3f})",
          "03_coef_adoption_2025.png")
coef_plot(m_delin_A, feats_A,
          f"Delinquency — Full 2022-2025  (AUC={auc_a3:.3f})",
          "04_coef_delinquency_full.png")
coef_plot(m_delin_B, feats_B,
          f"Delinquency — 2025 only  (AUC={auc_b3:.3f})",
          "05_coef_delinquency_2025.png")

# ROC curves — all 4 on one chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, pairs, title in [
    (axes[0], [
        (m_adopt_A, X_A,  y1_A, w1_A, f"Full 2022-2025 (AUC={auc_a1:.3f})", BLUE),
        (m_adopt_B, X_B,  y1_B, w1_B, f"2025 only (AUC={auc_b1:.3f})",       TEAL),
    ], "ROC — BNPL Adoption (BNPL1)"),
    (axes[1], [
        (m_delin_A, X3_A, y3_A, w3_A, f"Full 2022-2025 (AUC={auc_a3:.3f})", BLUE),
        (m_delin_B, X3_B, y3_B, w3_B, f"2025 only (AUC={auc_b3:.3f})",       TEAL),
    ], "ROC — BNPL Delinquency (BNPL3, users only)"),
]:
    for model, X, y, w, label, col in pairs:
        fpr, tpr, _ = roc_curve(y, model.predict_proba(X)[:,1], sample_weight=w)
        ax.plot(fpr, tpr, color=col, lw=2, label=label)
    ax.plot([0,1],[0,1], color=GRAY, ls="--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig("lasso_outputs/01_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → 01_roc_curves.png")

# ══════════════════════════════════════════════════════════════
# 6. SUMMARY CSV
# ══════════════════════════════════════════════════════════════
# Align features across both models for comparison
all_feats = sorted(set(feats_A) | set(feats_B))
rows = []
coef_a1 = dict(zip(feats_A, m_adopt_A.coef_[0]))
coef_b1 = dict(zip(feats_B, m_adopt_B.coef_[0]))
coef_a3 = dict(zip(feats_A, m_delin_A.coef_[0]))
coef_b3 = dict(zip(feats_B, m_delin_B.coef_[0]))

for f in all_feats:
    rows.append({
        "feature":                    f,
        "adopt_full_coef":            round(coef_a1.get(f, np.nan), 4),
        "adopt_2025_coef":            round(coef_b1.get(f, np.nan), 4),
        "delin_full_coef":            round(coef_a3.get(f, np.nan), 4),
        "delin_2025_coef":            round(coef_b3.get(f, np.nan), 4),
        "selected_adopt_full":        int(coef_a1.get(f, 0) != 0),
        "selected_adopt_2025":        int(coef_b1.get(f, 0) != 0),
        "selected_delin_full":        int(coef_a3.get(f, 0) != 0),
        "selected_delin_2025":        int(coef_b3.get(f, 0) != 0),
    })

summary = pd.DataFrame(rows).sort_values("adopt_full_coef", key=abs, ascending=False)
summary.to_csv("lasso_outputs/06_results_summary.csv", index=False)
print("  Saved → 06_results_summary.csv")

# ── Final summary ─────────────────────────────────────────────
print("\n" + "="*55)
print("FINAL SUMMARY")
print("="*55)
print(f"{'':30} {'Full':>10}  {'2025':>10}")
print(f"{'Adoption AUC':<30} {auc_a1:>10.4f}  {auc_b1:>10.4f}")
print(f"{'Adoption F1':<30} {f1_a1:>10.4f}  {f1_b1:>10.4f}")
print(f"{'Delinquency AUC':<30} {auc_a3:>10.4f}  {auc_b3:>10.4f}")
print(f"{'Delinquency F1':<30} {f1_a3:>10.4f}  {f1_b3:>10.4f}")
print("="*55)
