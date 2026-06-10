"""
Pre-PCA Suitability Evaluation — BNPL SHED Data (2022–2025)
============================================================
Purpose:
    Rigorously assess whether PCA is an appropriate dimensionality
    reduction technique for this dataset BEFORE committing to it.
    This section belongs in the methodology of any serious project
    or portfolio write-up.

Evaluation sections:
    1.  Data preparation (encoding, imputation)
    2.  Correlation structure  — heatmap + mean absolute correlation
    3.  KMO (Kaiser-Meyer-Olkin) — overall & per-variable
    4.  Bartlett's Test of Sphericity
    5.  VIF (Variance Inflation Factor) — collinearity between features
    6.  Eigenvalue analysis  — how many components are even meaningful
    7.  Explained variance curve  — n_components for 80% / 90%
    8.  Communalities  — how much variance PCA can actually recover per feature
    9.  Anti-image correlation matrix  — MSA per variable
    10. Verdict summary  — pass/fail scorecard with recommendation

Outputs saved to ./pca_eval_outputs/:
    01_correlation_heatmap.png
    02_kmo_per_variable.png
    03_vif_chart.png
    04_eigenvalue_scree.png
    05_explained_variance_curve.png
    06_communalities.png
    07_anti_image_msa.png
    08_pca_evaluation_scorecard.csv
    09_full_metrics_report.txt
"""

import os, warnings, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from scipy.stats import chi2 as scipy_chi2

warnings.filterwarnings("ignore")
os.makedirs("pca_eval_outputs", exist_ok=True)

# ── Color palette ────────────────────────────────────────────
BLUE   = "#185FA5"
LBLUE  = "#378ADD"
RED    = "#A32D2D"
LRED   = "#E24B4A"
TEAL   = "#0F6E56"
AMBER  = "#854F0B"
LAMBER = "#E8A838"
GRAY   = "#4A4A4A"
LGRAY  = "#9A9A9A"
GREEN  = "#2E7D32"
BG     = "white"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

SEED = 42
np.random.seed(SEED)

# ── Readable label map ────────────────────────────────────────
LABEL = {
    "ppfs1482": "Financial Health",    "A6":    "Financial Confidence",
    "ppinc7":   "Income Bracket",      "ppeducat": "Education Level",
    "I20":      "Spend vs Income",     "C4A":   "CC Revolving",
    "ppfsasset":"Asset Level",         "INF4":  "Inflation Impact",
    "C3P":      "CC Payment Behavior", "ppage": "Age",
    "EF1":      "Financial Literacy",  "EF5C":  "EF5C",
    "K21_a":    "K21a",
    "EF3_a":"Hardship: Bills",    "EF3_b":"Hardship: Medical",
    "EF3_c":"Hardship: Savings",  "EF3_d":"Hardship: Food Pantry",
    "EF3_e":"Hardship: Rent",     "EF3_f":"Hardship: Payday Loan",
    "EF3_g":"Hardship: Sold",     "EF3_h":"Hardship: Borrowed",
}

report_lines = []   # collects full text report

def rprint(msg=""):
    print(msg)
    report_lines.append(msg)

def section(title):
    bar = "=" * 60
    rprint(f"\n{bar}")
    rprint(f"  {title}")
    rprint(bar)

# ══════════════════════════════════════════════════════════════
# 1. DATA PREPARATION  (same pipeline as main PCA script)
# ══════════════════════════════════════════════════════════════
section("1. DATA PREPARATION")

df = pd.read_csv("data_merged.csv", low_memory=False)
rprint(f"  Raw shape: {df.shape}")

TARGETS = ["BNPL1", "BNPL3"]
META    = ["Unnamed: 0", "weight", "year",
           "BNPL4_a","BNPL4_b","BNPL4_c","BNPL4_d","BNPL4_e","BNPL4_f",
           "BNPL1A","BNPL5"]
EXCLUDE = set(TARGETS + META)

all_cols   = [c for c in df.columns if c not in EXCLUDE]
miss_rates = df[all_cols].isna().mean()
HIGH_MISS  = miss_rates[miss_rates > 0.50].index.tolist()
feat_cols  = [c for c in all_cols if c not in HIGH_MISS]
rprint(f"  Dropped {len(HIGH_MISS)} cols with >50% missing")
rprint(f"  Feature columns retained: {len(feat_cols)}")

df_feat = df[feat_cols].copy()

# Ordinal encoding
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
for col, m in ordinal_maps.items():
    if col in df_feat.columns:
        df_feat[col] = df_feat[col].map(m)

binary_cols = [c for c in df_feat.columns
               if df_feat[c].dropna().isin(["Yes","No"]).all()]
for c in binary_cols:
    df_feat[c] = df_feat[c].map({"Yes":1,"No":0})

nominal_cols = [c for c in ["ppethm","ppgender","ppemploy","ppmarit5"]
                if c in df_feat.columns]
df_feat = pd.get_dummies(df_feat, columns=nominal_cols, drop_first=True, dtype=float)

imputer  = SimpleImputer(strategy="median")
X_imp    = imputer.fit_transform(df_feat)
X_df     = pd.DataFrame(X_imp, columns=df_feat.columns)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
X_std    = pd.DataFrame(X_scaled, columns=df_feat.columns)

n_features = X_scaled.shape[1]
n_obs      = X_scaled.shape[0]
rprint(f"  Final encoded shape: {X_std.shape}")

# Human-readable labels for display
disp_labels = {c: LABEL.get(c, c.replace("ppethm_","eth:").replace("ppmarit5_","mar:")
                                  .replace("ppemploy_","emp:").replace("ppgender_","gen:"))
               for c in df_feat.columns}

# ══════════════════════════════════════════════════════════════
# 2. CORRELATION STRUCTURE
# ══════════════════════════════════════════════════════════════
section("2. CORRELATION STRUCTURE")

corr = X_std.corr()
corr_disp = corr.rename(index=disp_labels, columns=disp_labels)

# Mean absolute off-diagonal correlation
mask_diag  = ~np.eye(n_features, dtype=bool)
mean_abs_r = corr.values[mask_diag].reshape(n_features, n_features-1)
mean_abs_r = np.abs(mean_abs_r).mean()

# Proportion of pairs with |r| > thresholds
pairs      = corr.values[np.tril_indices(n_features, k=-1)]
pct_r20    = (np.abs(pairs) > 0.20).mean() * 100
pct_r40    = (np.abs(pairs) > 0.40).mean() * 100
pct_r60    = (np.abs(pairs) > 0.60).mean() * 100

rprint(f"  Mean absolute off-diagonal correlation:  {mean_abs_r:.4f}")
rprint(f"  Pairs with |r| > 0.20:  {pct_r20:.1f}%")
rprint(f"  Pairs with |r| > 0.40:  {pct_r40:.1f}%")
rprint(f"  Pairs with |r| > 0.60:  {pct_r60:.1f}%")
rprint(f"  Interpretation: PCA benefits most when mean |r| > 0.30 and")
rprint(f"  many pairs exceed 0.40. Low correlation = diffuse structure.")

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr_disp, dtype=bool))
sns.heatmap(corr_disp, mask=mask, annot=False, cmap="RdBu_r",
            vmin=-0.7, vmax=0.7, linewidths=0.3, linecolor="#e0e0e0",
            ax=ax, cbar_kws={"shrink":0.65, "label":"Pearson r"})
ax.set_title(
    f"Feature Correlation Matrix  (lower triangle)\n"
    f"Mean |r| = {mean_abs_r:.3f} | Pairs |r|>0.40: {pct_r40:.1f}%  "
    f"| Pairs |r|>0.60: {pct_r60:.1f}%",
    pad=14
)
plt.tight_layout()
fig.savefig("pca_eval_outputs/01_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
rprint("  Saved → 01_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════
# 3. KMO — KAISER-MEYER-OLKIN MEASURE OF SAMPLING ADEQUACY
# ══════════════════════════════════════════════════════════════
section("3. KMO — KAISER-MEYER-OLKIN (Sampling Adequacy)")

rprint("  What it measures:")
rprint("    KMO compares the magnitude of observed correlations to")
rprint("    partial correlations. High KMO means most correlation")
rprint("    between variables is shared (common) variance — exactly")
rprint("    what PCA exploits. Low KMO means correlations are driven")
rprint("    by unique variable-pair relationships, not a shared factor.")
rprint("")
rprint("  Thresholds:  <0.50 unacceptable | 0.50-0.59 miserable |")
rprint("               0.60-0.69 mediocre  | 0.70-0.79 middling  |")
rprint("               0.80-0.89 meritorious | 0.90+ marvelous")

sample_idx = np.random.choice(n_obs, size=min(10000, n_obs), replace=False)
X_sample   = X_scaled[sample_idx]

kmo_per_var, kmo_overall = calculate_kmo(X_sample)

def kmo_label(v):
    if   v >= 0.90: return "Marvelous"
    elif v >= 0.80: return "Meritorious"
    elif v >= 0.70: return "Middling"
    elif v >= 0.60: return "Mediocre"
    elif v >= 0.50: return "Miserable"
    else:           return "Unacceptable"

rprint(f"\n  ► Overall KMO:  {kmo_overall:.4f}  [{kmo_label(kmo_overall)}]")

kmo_df = pd.DataFrame({
    "feature":    df_feat.columns,
    "KMO":        kmo_per_var,
    "label":      [kmo_label(v) for v in kmo_per_var],
    "display":    [disp_labels[c] for c in df_feat.columns]
}).sort_values("KMO")

rprint(f"\n  Per-variable KMO range: {kmo_per_var.min():.3f} – {kmo_per_var.max():.3f}")
rprint(f"  Variables below 0.60 (mediocre): "
       f"{(kmo_per_var < 0.60).sum()} / {len(kmo_per_var)}")
rprint(f"  Variables below 0.50 (unacceptable): "
       f"{(kmo_per_var < 0.50).sum()} / {len(kmo_per_var)}")

# Plot
def kmo_color(v):
    if v >= 0.80: return GREEN
    if v >= 0.70: return TEAL
    if v >= 0.60: return LBLUE
    if v >= 0.50: return LAMBER
    return LRED

colors_kmo = [kmo_color(v) for v in kmo_df["KMO"]]

fig, ax = plt.subplots(figsize=(10, max(7, len(kmo_df)*0.28)))
bars = ax.barh(kmo_df["display"], kmo_df["KMO"], color=colors_kmo, height=0.65)
ax.axvline(0.60, color=LBLUE,  ls="--", lw=1.3, label="0.60 Mediocre threshold")
ax.axvline(0.70, color=TEAL,   ls="--", lw=1.3, label="0.70 Middling threshold")
ax.axvline(0.80, color=GREEN,  ls="--", lw=1.3, label="0.80 Meritorious threshold")
ax.axvline(kmo_overall, color=RED, ls="-", lw=2.0,
           label=f"Overall KMO = {kmo_overall:.3f}")
ax.set_xlim(0, 1.0)
ax.set_xlabel("KMO Score")
ax.set_title("KMO Measure of Sampling Adequacy — Per Variable\n"
             "How much each feature contributes to shared (factorable) variance")
ax.legend(fontsize=8, loc="lower right")
for bar, val in zip(bars, kmo_df["KMO"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8, color=GRAY)
plt.tight_layout()
fig.savefig("pca_eval_outputs/02_kmo_per_variable.png", dpi=150, bbox_inches="tight")
plt.close()
rprint("  Saved → 02_kmo_per_variable.png")

# ══════════════════════════════════════════════════════════════
# 4. BARTLETT'S TEST OF SPHERICITY
# ══════════════════════════════════════════════════════════════
section("4. BARTLETT'S TEST OF SPHERICITY")

rprint("  What it tests:")
rprint("    H0: The correlation matrix is an identity matrix (all")
rprint("    off-diagonal correlations = 0). If we cannot reject H0,")
rprint("    variables share no structure and PCA is meaningless.")
rprint("    Rejecting H0 is a NECESSARY but not SUFFICIENT condition.")

chi2_stat, p_val = calculate_bartlett_sphericity(X_sample)

# Degrees of freedom = p*(p-1)/2
df_bart = n_features * (n_features - 1) / 2
effect  = chi2_stat / (n_obs * df_bart)   # effect size approximation

rprint(f"\n  ► χ² statistic:   {chi2_stat:,.2f}")
rprint(f"  ► Degrees of freedom: {int(df_bart)}")
rprint(f"  ► p-value:        {p_val:.2e}")
rprint(f"  ► Effect size (χ²/N·df): {effect:.5f}")
rprint(f"  ► Result:  {'REJECT H0 ✓ — correlation matrix is not identity' if p_val < 0.05 else 'FAIL TO REJECT H0 ✗'}")
rprint(f"\n  Note: With N={n_obs:,}, Bartlett's test is extremely sensitive.")
rprint(f"  A significant p-value is almost guaranteed at this sample size.")
rprint(f"  The effect size ({effect:.5f}) is the more informative metric here.")
rprint(f"  Small effect size despite significant p confirms that while")
rprint(f"  correlations exist, they are weak in practical magnitude.")

# ══════════════════════════════════════════════════════════════
# 5. VIF — VARIANCE INFLATION FACTOR
# ══════════════════════════════════════════════════════════════
section("5. VIF — VARIANCE INFLATION FACTOR")

rprint("  What it measures:")
rprint("    VIF quantifies how much the variance of a regression")
rprint("    coefficient is inflated due to collinearity with other")
rprint("    features. VIF = 1/(1 - R²) where R² is from regressing")
rprint("    that feature on all others.")
rprint("")
rprint("  Thresholds:  VIF = 1 (no collinearity) | 1-5 low |")
rprint("               5-10 moderate | >10 severe collinearity")

vif_vals = [variance_inflation_factor(X_scaled, i) for i in range(n_features)]
vif_df   = pd.DataFrame({
    "feature": df_feat.columns,
    "VIF":     vif_vals,
    "display": [disp_labels[c] for c in df_feat.columns]
}).sort_values("VIF", ascending=False)

def vif_color(v):
    if v >= 10:  return LRED
    if v >= 5:   return LAMBER
    if v >= 2:   return LBLUE
    return TEAL

rprint(f"\n  VIF range: {min(vif_vals):.2f} – {max(vif_vals):.2f}")
rprint(f"  Features with VIF > 10 (severe):    {(vif_df['VIF']>10).sum()}")
rprint(f"  Features with VIF 5-10 (moderate):  {((vif_df['VIF']>5)&(vif_df['VIF']<=10)).sum()}")
rprint(f"  Features with VIF 1-5 (low):        {(vif_df['VIF']<=5).sum()}")
rprint(f"\n  Top 10 highest VIF:")
for _, row in vif_df.head(10).iterrows():
    flag = " ← moderate/high" if row["VIF"] > 5 else ""
    rprint(f"    {row['display']:<35} VIF = {row['VIF']:.2f}{flag}")

colors_vif = [vif_color(v) for v in vif_df["VIF"]]
fig, ax = plt.subplots(figsize=(10, max(7, len(vif_df)*0.28)))
bars = ax.barh(vif_df["display"][::-1], vif_df["VIF"][::-1],
               color=colors_vif[::-1], height=0.65)
ax.axvline(5,  color=LAMBER, ls="--", lw=1.3, label="VIF = 5  (moderate)")
ax.axvline(10, color=LRED,   ls="--", lw=1.3, label="VIF = 10 (severe)")
ax.set_xlabel("Variance Inflation Factor (VIF)")
ax.set_title("VIF — Collinearity Between Features\n"
             "Higher VIF = feature is more predictable from others")
ax.legend(fontsize=9)

legend_patches = [
    mpatches.Patch(color=TEAL,   label="VIF < 2 (minimal)"),
    mpatches.Patch(color=LBLUE,  label="VIF 2–5 (low)"),
    mpatches.Patch(color=LAMBER, label="VIF 5–10 (moderate)"),
    mpatches.Patch(color=LRED,   label="VIF > 10 (severe)"),
]
ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
for bar, val in zip(bars, vif_df["VIF"][::-1]):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=8, color=GRAY)
plt.tight_layout()
fig.savefig("pca_eval_outputs/03_vif_chart.png", dpi=150, bbox_inches="tight")
plt.close()
rprint("  Saved → 03_vif_chart.png")

# ══════════════════════════════════════════════════════════════
# 6. EIGENVALUE ANALYSIS & SCREE
# ══════════════════════════════════════════════════════════════
section("6. EIGENVALUE ANALYSIS")

rprint("  What eigenvalues tell you:")
rprint("    Each eigenvalue = variance captured by that component.")
rprint("    Kaiser rule: retain components with eigenvalue > 1.")
rprint("    (A component eigenvalue < 1 explains less variance than")
rprint("    a single original standardised variable — not worth keeping.)")
rprint("    Elbow rule: keep components before the 'elbow' in the scree.")

pca_full     = PCA(random_state=SEED)
pca_full.fit(X_scaled)
eigenvalues  = pca_full.explained_variance_
explained    = pca_full.explained_variance_ratio_
cumulative   = np.cumsum(explained)

kaiser_n     = int(np.sum(eigenvalues > 1))
n_comp_80    = int(np.argmax(cumulative >= 0.80)) + 1
n_comp_90    = int(np.argmax(cumulative >= 0.90)) + 1
n_comp_95    = int(np.argmax(cumulative >= 0.95)) + 1
reduction_80 = (1 - n_comp_80 / n_features) * 100
reduction_90 = (1 - n_comp_90 / n_features) * 100

rprint(f"\n  Total features (dimensions):    {n_features}")
rprint(f"  Kaiser rule (eigenvalue > 1):   {kaiser_n} components  "
       f"({cumulative[kaiser_n-1]*100:.1f}% variance)")
rprint(f"  Components for 80% variance:    {n_comp_80}  "
       f"(reduces dimensions by {reduction_80:.0f}%)")
rprint(f"  Components for 90% variance:    {n_comp_90}  "
       f"(reduces dimensions by {reduction_90:.0f}%)")
rprint(f"  Components for 95% variance:    {n_comp_95}")
rprint(f"\n  ► Ideal PCA: 3-5 components explain 80%+.")
rprint(f"  ► This dataset needs {n_comp_80} components for 80% —")
rprint(f"    a {reduction_80:.0f}% reduction vs {100-100/n_features*n_comp_80:.0f}% ideal.")

n_show = min(n_features, 30)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(1, n_show+1), eigenvalues[:n_show],
        marker="o", color=BLUE, lw=2, ms=5, zorder=3)
ax.fill_between(range(1, n_show+1), eigenvalues[:n_show],
                alpha=0.08, color=BLUE)
ax.axhline(1.0, color=AMBER, ls="--", lw=1.5,
           label=f"Kaiser threshold (eigenvalue=1) → {kaiser_n} components")
# Mark elbow region
ax.axvspan(kaiser_n-0.5, kaiser_n+0.5, alpha=0.12, color=AMBER)

# Annotate first few
for i in range(min(5, n_show)):
    ax.annotate(f"  {eigenvalues[i]:.2f}",
                xy=(i+1, eigenvalues[i]),
                fontsize=8, color=BLUE, va="bottom")

ax.set_xlabel("Principal Component")
ax.set_ylabel("Eigenvalue")
ax.set_title(f"Scree Plot — Eigenvalues per Component\n"
             f"Kaiser rule: {kaiser_n} components (eigenvalue > 1); "
             f"explains {cumulative[kaiser_n-1]*100:.1f}% of variance")
ax.set_xticks(range(1, n_show+1))
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig("pca_eval_outputs/04_eigenvalue_scree.png", dpi=150, bbox_inches="tight")
plt.close()
rprint("  Saved → 04_eigenvalue_scree.png")

# ══════════════════════════════════════════════════════════════
# 7. EXPLAINED VARIANCE CURVE
# ══════════════════════════════════════════════════════════════
section("7. EXPLAINED VARIANCE CURVE")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: cumulative curve
ax = axes[0]
ax.plot(range(1, n_features+1), cumulative*100,
        color=TEAL, lw=2, marker="o", ms=3)
for thresh, col, n_t in [(0.80, AMBER, n_comp_80),
                         (0.90, LBLUE, n_comp_90),
                         (0.95, RED,   n_comp_95)]:
    ax.axhline(thresh*100, color=col, ls="--", lw=1.2,
               label=f"{int(thresh*100)}% → {n_t} components")
    ax.axvline(n_t, color=col, ls=":", lw=1, alpha=0.6)
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Variance Explained (%)")
ax.set_title("Cumulative Explained Variance")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

# Right: individual bars (first 20)
ax2 = axes[1]
n_bar = min(20, n_features)
bar_colors = [BLUE if i < kaiser_n else LGRAY for i in range(n_bar)]
ax2.bar(range(1, n_bar+1), explained[:n_bar]*100, color=bar_colors, width=0.7)
ax2.axvline(kaiser_n+0.5, color=AMBER, ls="--", lw=1.5,
            label=f"Kaiser cutoff at PC{kaiser_n}")
ax2.set_xlabel("Principal Component")
ax2.set_ylabel("Individual Variance Explained (%)")
ax2.set_title(f"Variance per Component (first {n_bar})\nBlue = retained by Kaiser rule")
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax2.legend(fontsize=9)

plt.suptitle(f"Explained Variance Analysis  |  {n_features} features → Kaiser: {kaiser_n} components  "
             f"|  80% needs {n_comp_80} components",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig("pca_eval_outputs/05_explained_variance_curve.png", dpi=150, bbox_inches="tight")
plt.close()
rprint(f"  Saved → 05_explained_variance_curve.png")

# ══════════════════════════════════════════════════════════════
# 8. COMMUNALITIES
# ══════════════════════════════════════════════════════════════
section("8. COMMUNALITIES")

rprint("  What communalities tell you:")
rprint("    Communality = proportion of a variable's variance that is")
rprint("    captured by the retained PCA components. High communality")
rprint("    (>0.60) means PCA represents that variable well. Low")
rprint("    communality (<0.30) means the variable has substantial")
rprint("    unique variance PCA cannot recover — it would be lost.")
rprint("    If many variables have low communality, PCA is a poor fit.")

pca_k   = PCA(n_components=kaiser_n, random_state=SEED)
pca_k.fit(X_scaled)
# Communality = sum of squared loadings across retained components
communalities = np.sum(pca_k.components_.T ** 2, axis=1)

comm_df = pd.DataFrame({
    "feature":      df_feat.columns,
    "communality":  communalities,
    "display":      [disp_labels[c] for c in df_feat.columns]
}).sort_values("communality")

n_low_30 = (communalities < 0.30).sum()
n_low_50 = (communalities < 0.50).sum()
n_low_60 = (communalities < 0.60).sum()
mean_comm = communalities.mean()

rprint(f"\n  Mean communality:           {mean_comm:.3f}")
rprint(f"  Variables with comm < 0.30 (poor):    {n_low_30} / {n_features}")
rprint(f"  Variables with comm < 0.50 (marginal): {n_low_50} / {n_features}")
rprint(f"  Variables with comm < 0.60 (acceptable):{n_low_60} / {n_features}")
rprint(f"\n  ► Ideal: mean communality > 0.60, few below 0.30.")

def comm_color(v):
    if v >= 0.70: return GREEN
    if v >= 0.60: return TEAL
    if v >= 0.40: return LBLUE
    if v >= 0.30: return LAMBER
    return LRED

bar_colors_c = [comm_color(v) for v in comm_df["communality"]]

fig, ax = plt.subplots(figsize=(10, max(7, len(comm_df)*0.28)))
bars = ax.barh(comm_df["display"], comm_df["communality"],
               color=bar_colors_c, height=0.65)
ax.axvline(0.30, color=LRED,   ls="--", lw=1.3, label="0.30 Poor threshold")
ax.axvline(0.50, color=LAMBER, ls="--", lw=1.3, label="0.50 Marginal threshold")
ax.axvline(0.60, color=TEAL,   ls="--", lw=1.3, label="0.60 Acceptable threshold")
ax.axvline(mean_comm, color=RED, ls="-", lw=2.0,
           label=f"Mean = {mean_comm:.3f}")
ax.set_xlim(0, 1.0)
ax.set_xlabel("Communality (proportion of variance captured by PCA)")
ax.set_title(f"Communalities — How Well PCA Recovers Each Variable\n"
             f"({kaiser_n} components retained; {n_low_30} variables with communality < 0.30)")

legend_patches = [
    mpatches.Patch(color=GREEN,  label="≥ 0.70 (good)"),
    mpatches.Patch(color=TEAL,   label="0.60–0.70 (acceptable)"),
    mpatches.Patch(color=LBLUE,  label="0.40–0.60 (marginal)"),
    mpatches.Patch(color=LAMBER, label="0.30–0.40 (poor)"),
    mpatches.Patch(color=LRED,   label="< 0.30 (very poor)"),
    mpatches.Patch(color=RED,    label=f"Mean = {mean_comm:.3f}"),
]
ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
for bar, val in zip(bars, comm_df["communality"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8, color=GRAY)
plt.tight_layout()
fig.savefig("pca_eval_outputs/06_communalities.png", dpi=150, bbox_inches="tight")
plt.close()
rprint("  Saved → 06_communalities.png")

# ══════════════════════════════════════════════════════════════
# 9. ANTI-IMAGE CORRELATION — MSA PER VARIABLE
# ══════════════════════════════════════════════════════════════
section("9. ANTI-IMAGE CORRELATION MATRIX (MSA)")

rprint("  What anti-image tells you:")
rprint("    The anti-image correlation matrix contains the negatives")
rprint("    of partial correlations between each pair of variables,")
rprint("    controlling for all others. Small off-diagonal values")
rprint("    mean correlations ARE explained by common factors (good")
rprint("    for PCA). The diagonal of the anti-image matrix = each")
rprint("    variable's individual MSA (Measure of Sampling Adequacy),")
rprint("    which is the per-variable KMO score.")
rprint("    Variables with MSA < 0.50 should be dropped before PCA.")

msa_df = kmo_df.sort_values("KMO")   # already computed in section 3

# Plot MSA as a heatmap-style bar sorted by value
fig, ax = plt.subplots(figsize=(10, max(7, len(msa_df)*0.28)))
bar_colors_m = [kmo_color(v) for v in msa_df["KMO"]]
bars = ax.barh(msa_df["display"], msa_df["KMO"],
               color=bar_colors_m, height=0.65)
ax.axvline(0.50, color=LRED,  ls="--", lw=1.5, label="0.50 Drop threshold")
ax.axvline(0.60, color=AMBER, ls="--", lw=1.3, label="0.60 Minimum acceptable")
ax.axvline(kmo_overall, color=RED, ls="-", lw=2.0,
           label=f"Overall MSA = {kmo_overall:.3f}")
ax.set_xlim(0, 1.0)
ax.set_xlabel("MSA (Measure of Sampling Adequacy = per-variable KMO)")
ax.set_title("Anti-Image Diagonal — MSA Per Variable\n"
             "Variables below 0.50 should be dropped from PCA")
legend_patches = [
    mpatches.Patch(color=GREEN,  label="≥ 0.80 Meritorious"),
    mpatches.Patch(color=TEAL,   label="0.70–0.80 Middling"),
    mpatches.Patch(color=LBLUE,  label="0.60–0.70 Mediocre"),
    mpatches.Patch(color=LAMBER, label="0.50–0.60 Miserable"),
    mpatches.Patch(color=LRED,   label="< 0.50 Unacceptable"),
]
ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
for bar, val in zip(bars, msa_df["KMO"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8, color=GRAY)
plt.tight_layout()
fig.savefig("pca_eval_outputs/07_anti_image_msa.png", dpi=150, bbox_inches="tight")
plt.close()
rprint("  Saved → 07_anti_image_msa.png")

# ══════════════════════════════════════════════════════════════
# 10. SCORECARD & VERDICT
# ══════════════════════════════════════════════════════════════
section("10. PCA SUITABILITY SCORECARD & VERDICT")

def score_check(label, value, good_thresh, warn_thresh, higher_is_better=True, fmt=".3f"):
    if higher_is_better:
        status = "PASS" if value >= good_thresh else ("WARN" if value >= warn_thresh else "FAIL")
    else:
        status = "PASS" if value <= good_thresh else ("WARN" if value <= warn_thresh else "FAIL")
    symbol = {"PASS": "✓", "WARN": "~", "FAIL": "✗"}[status]
    rprint(f"  [{symbol} {status}]  {label:<45} {value:{fmt}}")
    return status, value

checks = []
checks.append(score_check("KMO overall (≥0.70 good, ≥0.60 acceptable)",
                           kmo_overall, 0.70, 0.60))
checks.append(score_check("Bartlett p < 0.05",
                           p_val, 0.05, 0.05, higher_is_better=False, fmt=".2e"))
checks.append(score_check("Mean absolute correlation (>0.30 good)",
                           mean_abs_r, 0.30, 0.20))
checks.append(score_check(f"% pairs with |r|>0.40 (>30% good)",
                           pct_r40, 30.0, 15.0, fmt=".1f"))
checks.append(score_check(f"Kaiser components / total features ratio (>0.50 = poor reduction)",
                           kaiser_n/n_features, 0.50, 0.65,
                           higher_is_better=False, fmt=".2f"))
checks.append(score_check(f"Components for 80% variance (≤8 good, ≤12 acceptable)",
                           n_comp_80, 8, 12, higher_is_better=False, fmt="d"))
checks.append(score_check(f"Dimension reduction at 80% (>60% good)",
                           reduction_80, 60.0, 40.0, fmt=".1f"))
checks.append(score_check(f"Mean communality Kaiser (>0.60 good)",
                           mean_comm, 0.60, 0.45))
checks.append(score_check(f"Variables with communality < 0.30 (0 good, <20% ok)",
                           n_low_30/n_features*100, 0.0, 20.0,
                           higher_is_better=False, fmt=".1f"))
checks.append(score_check(f"Variables with MSA < 0.50 (0 good)",
                           (kmo_per_var < 0.50).sum(), 0, 2,
                           higher_is_better=False, fmt="d"))
checks.append(score_check(f"Max VIF (≤5 good, ≤10 acceptable)",
                           max(vif_vals), 5.0, 10.0,
                           higher_is_better=False, fmt=".2f"))

passes = sum(1 for s, _ in checks if s == "PASS")
warns  = sum(1 for s, _ in checks if s == "WARN")
fails  = sum(1 for s, _ in checks if s == "FAIL")
total  = len(checks)
score_pct = (passes + 0.5*warns) / total * 100

rprint(f"\n  Score: {passes} PASS / {warns} WARN / {fails} FAIL  "
       f"({score_pct:.0f}% weighted)")

# Verdict
rprint("\n" + "─" * 60)
if score_pct >= 75:
    verdict = "PCA IS APPROPRIATE"
    detail  = "Dataset shows adequate structure for PCA."
    rec     = "Proceed with PCA. Use Kaiser rule for component selection."
elif score_pct >= 50:
    verdict = "PCA IS MARGINAL — USE WITH CAUTION"
    detail  = ("Dataset has some factorizable structure but also many "
               "independent dimensions. PCA will work technically but "
               "dimension reduction will be limited.")
    rec     = ("Consider: (1) Drop variables with MSA < 0.60 and rerun. "
               "(2) Use PCA for collinearity diagnosis only, then select "
               "original variables for modeling. (3) Use LASSO instead.")
else:
    verdict = "PCA IS NOT RECOMMENDED"
    detail  = ("Dataset lacks the shared variance structure PCA requires. "
               "Forcing PCA will produce many components with minimal "
               "reduction and poor interpretability.")
    rec     = ("Recommended alternatives: (1) LASSO logistic regression "
               "for automatic feature selection with interpretable "
               "coefficients. (2) Recursive Feature Elimination (RFE). "
               "(3) Use PCA output only as a collinearity diagnostic to "
               "inform manual variable selection.")

rprint(f"  VERDICT: {verdict}")
rprint(f"  {detail}")
rprint(f"\n  RECOMMENDATION:")
for line in textwrap.wrap(rec, width=56):
    rprint(f"    {line}")
rprint("─" * 60)

# ── Save scorecard CSV ─────────────────────────────────────
scorecard = pd.DataFrame([
    {"metric": "KMO Overall",              "value": round(kmo_overall, 4),     "threshold_pass": 0.70, "threshold_warn": 0.60,  "status": checks[0][0]},
    {"metric": "Bartlett p-value",         "value": float(f"{p_val:.4e}"),     "threshold_pass": 0.05, "threshold_warn": 0.05,  "status": checks[1][0]},
    {"metric": "Bartlett chi2",            "value": round(chi2_stat, 2),        "threshold_pass": None, "threshold_warn": None,  "status": "INFO"},
    {"metric": "Bartlett effect size",     "value": round(effect, 6),           "threshold_pass": None, "threshold_warn": None,  "status": "INFO"},
    {"metric": "Mean abs correlation",     "value": round(mean_abs_r, 4),      "threshold_pass": 0.30, "threshold_warn": 0.20,  "status": checks[2][0]},
    {"metric": "% pairs |r|>0.40",         "value": round(pct_r40, 2),         "threshold_pass": 30.0, "threshold_warn": 15.0,  "status": checks[3][0]},
    {"metric": "% pairs |r|>0.20",         "value": round(pct_r20, 2),         "threshold_pass": None, "threshold_warn": None,  "status": "INFO"},
    {"metric": "% pairs |r|>0.60",         "value": round(pct_r60, 2),         "threshold_pass": None, "threshold_warn": None,  "status": "INFO"},
    {"metric": "Total features",           "value": n_features,                 "threshold_pass": None, "threshold_warn": None,  "status": "INFO"},
    {"metric": "Kaiser components",        "value": kaiser_n,                   "threshold_pass": None, "threshold_warn": None,  "status": "INFO"},
    {"metric": "Kaiser comp / n_features", "value": round(kaiser_n/n_features,3),"threshold_pass":0.50, "threshold_warn": 0.65,  "status": checks[4][0]},
    {"metric": "Components for 80% var",   "value": n_comp_80,                  "threshold_pass": 8,    "threshold_warn": 12,    "status": checks[5][0]},
    {"metric": "Components for 90% var",   "value": n_comp_90,                  "threshold_pass": None, "threshold_warn": None,  "status": "INFO"},
    {"metric": "Components for 95% var",   "value": n_comp_95,                  "threshold_pass": None, "threshold_warn": None,  "status": "INFO"},
    {"metric": "Dimension reduction @80%", "value": round(reduction_80, 1),     "threshold_pass": 60.0, "threshold_warn": 40.0,  "status": checks[6][0]},
    {"metric": "Dimension reduction @90%", "value": round(reduction_90, 1),     "threshold_pass": None, "threshold_warn": None,  "status": "INFO"},
    {"metric": "Mean communality (Kaiser)","value": round(mean_comm, 4),        "threshold_pass": 0.60, "threshold_warn": 0.45,  "status": checks[7][0]},
    {"metric": "Variables comm < 0.30 (%)", "value": round(n_low_30/n_features*100,1),"threshold_pass":0,"threshold_warn":20,"status": checks[8][0]},
    {"metric": "Variables MSA < 0.50",     "value": int((kmo_per_var<0.50).sum()),"threshold_pass":0,  "threshold_warn": 2,     "status": checks[9][0]},
    {"metric": "Max VIF",                  "value": round(max(vif_vals), 2),    "threshold_pass": 5.0,  "threshold_warn": 10.0,  "status": checks[10][0]},
    {"metric": "PASS count",               "value": passes,                     "threshold_pass": None, "threshold_warn": None,  "status": "SUMMARY"},
    {"metric": "WARN count",               "value": warns,                      "threshold_pass": None, "threshold_warn": None,  "status": "SUMMARY"},
    {"metric": "FAIL count",               "value": fails,                      "threshold_pass": None, "threshold_warn": None,  "status": "SUMMARY"},
    {"metric": "Score (%)",                "value": round(score_pct, 1),        "threshold_pass": 75.0, "threshold_warn": 50.0,  "status": "SUMMARY"},
    {"metric": "VERDICT",                  "value": verdict,                    "threshold_pass": None, "threshold_warn": None,  "status": "VERDICT"},
])
scorecard.to_csv("pca_eval_outputs/08_pca_evaluation_scorecard.csv", index=False)
rprint("\n  Saved → 08_pca_evaluation_scorecard.csv")

# ── Save full text report ──────────────────────────────────
with open("pca_eval_outputs/09_full_metrics_report.txt", "w") as f:
    f.write("\n".join(report_lines))
rprint("  Saved → 09_full_metrics_report.txt")

rprint("\n  All outputs in ./pca_eval_outputs/")
rprint("=" * 60)
