# %% [markdown]
# # Life-Insurance Policy Lapse Prediction
# **Trainer:** Ashok Kumar P (TFTPL230) · **Track B – Classification** · TERV AIML Trainer Project 06
#
# **Business question.** About 9% of new life-insurance policies lapse at the 13th month because the
# second-year premium is never paid. Each lapse costs about ₹18,000; a retention call costs about ₹450.
# By month 10 of a policy we want to score the lapse risk and choose **how many customers to call**
# so that expected cost (calls + lost policies) is as low as possible.
#
# **Plan**
# 1. Data-quality audit → 2. Cleaning → 3. Feature engineering → 4. EDA (charts 1–7)
# 5. Time-ordered split → 6. 3 models × 3 imbalance strategies with stratified 5-fold CV (chart 8)
# 7. Test-period evaluation (chart 9) → 8. Cost-based call threshold (charts 10–11)
# 9. Cumulative gains + permutation importance (charts 12–13) → 10. Insights, recommendations, saved model
#
# *Run this notebook top to bottom (or run `python run_project.py`). Everything is saved to `outputs/`, `data/processed/` and `models/`.*

# %%
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

try:                                   # inside Jupyter -> inline charts; plain script -> save only
    get_ipython()                      # noqa: F821
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

try:
    from IPython.display import display
except ImportError:                    # running as a plain script without IPython
    display = print

from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, make_scorer,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

warnings.filterwarnings("ignore")

# ---- locate the project root and make src/ importable ----
_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
ROOT = next(p for p in [_start, *_start.parents] if (p / "data").is_dir() and (p / "src").is_dir())
sys.path.insert(0, str(ROOT / "src"))

from lapse_utils import (CALL_COST, CAT_FEATURES, FEATURES, LOSS_PER_LAPSE, MODEL_NAMES, NUM_FEATURES,
                         RANDOM_STATE, SAVE_RATE, STRATEGIES, TEST_START, add_features, clean_policies,
                         drop_duplicates, expected_cost, fix_invalid_values, impute_missing,
                         make_pipeline, no_call_cost, parse_dates, reason_codes, standardise_categories)

RAW = ROOT / "data" / "raw" / "life_policies_13m.csv"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
REP = ROOT / "outputs" / "reports"
MODELS = ROOT / "models"
for d in (PROC, FIG, REP, MODELS):
    d.mkdir(parents=True, exist_ok=True)

if not RAW.exists():                   # no CSV supplied -> build the synthetic one
    import generate_data
    generate_data.main()

sns.set_theme(style="whitegrid", context="notebook")
BLUE, RED, GREY, GREEN = "#2f6db5", "#d1495b", "#8d99ae", "#3a9a6b"


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.show()
    plt.close(fig)


print("Project root:", ROOT)

# %% [markdown]
# ## 1 · Data understanding & quality audit

# %%
raw = pd.read_csv(RAW)
print(f"Shape: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
display(raw.head())
display(raw.dtypes.astype(str).to_frame("dtype").T)
display(raw.describe().T.round(2))

# %%
is_agent = raw["sales_channel"].eq("Agent")
iso_date = raw["issue_date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")

audit = pd.DataFrame({"issue": [
    "Exact duplicate rows (double uploads)",
    "Missing annual_income_lakh",
    "Missing contact_verified",
    "agent_tenure blank - non-agent channels (by design)",
    "agent_tenure blank - inside Agent channel",
    "issue_date in DD-MM-YYYY (not YYYY-MM-DD)",
    "Invalid ages (outside 18-80)",
    "Income placeholder 999",
    "Negative premiums (reversal entries)",
], "count": [
    int(raw.duplicated().sum()),
    int(raw["annual_income_lakh"].isna().sum()),
    int(raw["contact_verified"].isna().sum()),
    int(raw.loc[~is_agent, "agent_tenure_months"].isna().sum()),
    int(raw.loc[is_agent, "agent_tenure_months"].isna().sum()),
    int((~iso_date).sum()),
    int((~raw["age"].between(18, 80)).sum()),
    int((raw["annual_income_lakh"] == 999).sum()),
    int((raw["annual_premium_inr"] < 0).sum()),
]})
audit["% of rows"] = (audit["count"] / len(raw) * 100).round(2)
display(audit)

print("Invalid age values found:", sorted(raw.loc[~raw["age"].between(18, 80), "age"].unique()))
for col in ["premium_mode", "payment_method", "status_13m"]:
    print(f"\n{col}: {raw[col].nunique()} spellings -> {sorted(raw[col].unique())}")

# %% [markdown]
# **Audit findings.** Every issue listed in the handbook is present: duplicates, mixed date formats, impossible ages,
# a `999` income placeholder, reversal (negative-premium) entries, missing values and many spellings of the same category.
# The target itself is spelled several ways, so it must be standardised *before* any modelling.

# %% [markdown]
# ## 2 · Cleaning (each decision justified)

# %% [markdown]
# **2.1 Drop exact duplicates.** These are double uploads of the same policy; keeping them would count a policy twice and
# let identical rows land in both CV training and validation folds (optimistic scores).

# %%
df = drop_duplicates(raw)
print(f"Rows: {len(raw):,} -> {len(df):,}  (removed {len(raw) - len(df):,} duplicates)")

# %% [markdown]
# **2.2 Parse both date formats.** `YYYY-MM-DD` and `DD-MM-YYYY` cannot be confused (the 4-digit year is in a different
# position), so each is parsed with an explicit format instead of letting pandas guess day-first vs month-first.

# %%
df = parse_dates(df)
print("Issue dates:", df["issue_date"].min().date(), "->", df["issue_date"].max().date())

# %% [markdown]
# **2.3 Standardise spellings** (`premium_mode`, `payment_method`, `status_13m`) with explicit maps. The function raises an error
# if it meets a spelling it does not know, so nothing is silently mislabelled.

# %%
df = standardise_categories(df)
for col in ["premium_mode", "payment_method", "status_13m"]:
    print(col, "->", df[col].value_counts().to_dict())

# %% [markdown]
# **2.4 Invalid values.** Ages outside 18-80 (0, 7, 112, 140) and the `999` income placeholder are impossible values, not real
# data, so they become missing and are imputed. Negative premiums are reversal entries with the wrong sign; the absolute value
# is the real premium.

# %%
df = fix_invalid_values(df)
print("Missing age:", int(df["age"].isna().sum()), "| missing income:", int(df["annual_income_lakh"].isna().sum()),
      "| negative premiums left:", int((df["annual_premium_inr"] < 0).sum()))

# %% [markdown]
# **2.5 Impute missing values.**
# * `age`, `annual_income_lakh` → median of the same **occupation × city tier** (income depends strongly on both, so this is
#   closer than one global median). No target information is used.
# * `agent_tenure_months` → blank for non-agent channels *by design*, so it is a structural 0 (the `sales_channel` column
#   already tells the model it is not an agent); the 4% blanks inside the Agent channel get the agent median.
# * `contact_verified` → new category `Unknown` (missingness itself may be informative).

# %%
df = impute_missing(df)
df["lapsed"] = (df["status_13m"] == "Lapsed").astype(int)
print("Missing values left:", int(df.isna().sum().sum()))
print(f"Clean data: {df.shape[0]:,} rows x {df.shape[1]} columns | lapse rate = {df['lapsed'].mean():.1%}")
df.to_csv(PROC / "life_policies_clean.csv", index=False)

# %% [markdown]
# ## 3 · Feature engineering
# Created first because EDA charts 4 and 7 use them (nothing is fitted here, so there is no leakage).
#
# | Feature | Meaning |
# |---|---|
# | `premium_to_income` | annual premium ÷ annual income – affordability |
# | `high_burden_flag` | premium above 15% of income |
# | `autopay_flag` | NACH or UPI AutoPay mandate |
# | `new_agent_flag` | sold by an agent with < 12 months' experience |
# | `log_premium` | log of annual premium (premiums are right-skewed) |
# | `sum_assured_multiple` | sum assured ÷ premium |
# | `login_flag` | at least one portal login in the first 6 months |
# | `issue_month` | calendar month of issue |

# %%
df = add_features(df)
display(df[["premium_to_income", "high_burden_flag", "autopay_flag", "new_agent_flag", "log_premium",
            "sum_assured_multiple", "login_flag", "issue_month"]].describe().T.round(3))

# %% [markdown]
# ## 4 · Exploratory data analysis (charts 1–7)

# %%
overall = df["lapsed"].mean() * 100


def bar_rate(ax, series, title, color=BLUE):
    bars = ax.bar(series.index.astype(str), series.values, color=color)
    ax.axhline(overall, ls="--", c=GREY, lw=1)
    for b, v in zip(bars, series.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Lapse rate (%)")
    ax.tick_params(axis="x", rotation=20)


def rate_by(col):
    return df.groupby(col, observed=True)["lapsed"].mean() * 100


# ---- Chart 1: class balance ----
counts = df["status_13m"].value_counts().reindex(["In-force", "Lapsed"])
fig, ax = plt.subplots(figsize=(5.5, 4))
bars = ax.bar(counts.index, counts.values, color=[BLUE, RED])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom")
ax.set_ylim(0, counts.max() * 1.18)
ax.set_title("Chart 1 · Class balance (cleaned policies)")
ax.set_ylabel("Policies")
savefig(fig, "01_class_balance")
print(f"Insight: only {df['lapsed'].mean():.1%} of {len(df):,} policies lapse, so predicting 'in-force' for everyone is "
      f"{1 - df['lapsed'].mean():.0%} accurate and useless - PR-AUC and recall are the right yardsticks.")

# %%
# ---- Chart 2: payment method x premium mode ----
pm_order = ["NACH auto-debit", "UPI AutoPay", "Card", "Cash/Cheque"]
mode_order = ["Annual", "Half-yearly", "Quarterly", "Monthly"]
piv = df.pivot_table(index="payment_method", columns="premium_mode", values="lapsed",
                     aggfunc="mean").reindex(index=pm_order, columns=mode_order) * 100
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(piv, annot=True, fmt=".1f", cmap="Reds", cbar_kws={"label": "Lapse rate (%)"}, ax=ax)
ax.set_title("Chart 2 · Lapse rate (%) by payment method and premium mode")
ax.set_xlabel("")
ax.set_ylabel("")
savefig(fig, "02_lapse_by_payment_and_mode")
auto = df.loc[df["autopay_flag"] == 1, "lapsed"].mean()
print(f"Insight: Cash/Cheque payers lapse at {piv.loc['Cash/Cheque'].min():.0f}-{piv.loc['Cash/Cheque'].max():.0f}% "
      f"(worst on monthly mode) vs {auto:.1%} for NACH/UPI AutoPay mandates.")

# %%
# ---- Chart 3: channel and product ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ch = rate_by("sales_channel").sort_values(ascending=False)
pr = rate_by("product_type").sort_values(ascending=False)
bar_rate(axes[0], ch, "By sales channel")
bar_rate(axes[1], pr, "By product type", color=GREEN)
fig.suptitle("Chart 3 · Lapse rate by channel and product (dashed = overall)", y=1.02)
savefig(fig, "03_lapse_by_channel_and_product")
print(f"Insight: {ch.index[0]} ({ch.iloc[0]:.1f}%) and {pr.index[0]} ({pr.iloc[0]:.1f}%) lapse most; "
      f"{ch.index[-1]} ({ch.iloc[-1]:.1f}%) and {pr.index[-1]} ({pr.iloc[-1]:.1f}%) lapse least.")

# %%
# ---- Chart 4: affordability ----
df["pti_band"] = pd.cut(df["premium_to_income"], [0, .03, .08, .15, .25, np.inf],
                        labels=["<3%", "3-8%", "8-15%", "15-25%", ">25%"])
aff = rate_by("pti_band")
fig, ax = plt.subplots(figsize=(7, 4))
bar_rate(ax, aff, "Chart 4 · Lapse rate by premium-to-income band", color=RED)
ax.set_xlabel("Annual premium as % of annual income")
ax.tick_params(axis="x", rotation=0)
savefig(fig, "04_lapse_by_affordability")
print(f"Insight: lapse rises from {aff.iloc[0]:.1f}% (premium < 3% of income) to {aff.loc['15-25%']:.1f}% at 15-25% "
      f"and {aff.loc['>25%']:.0f}% above 25% - affordability is the single strongest signal.")

# %%
# ---- Chart 5: agent tenure ----
band = pd.Series("Non-agent", index=df.index, dtype=object)
ag = df["sales_channel"] == "Agent"
band[ag] = pd.cut(df.loc[ag, "agent_tenure_months"], [-1, 11, 36, 60, np.inf],
                  labels=["Agent <12 m", "Agent 12-36 m", "Agent 36-60 m", "Agent >60 m"]).astype(str)
df["tenure_band"] = band
ten = (df.groupby("tenure_band")["lapsed"].mean() * 100).reindex(
    ["Agent <12 m", "Agent 12-36 m", "Agent 36-60 m", "Agent >60 m", "Non-agent"])
fig, ax = plt.subplots(figsize=(8, 4))
bar_rate(ax, ten, "Chart 5 · Lapse rate by selling-agent experience", color=BLUE)
savefig(fig, "05_lapse_by_agent_tenure")
exp_rate = df.loc[ag & (df["agent_tenure_months"] >= 12), "lapsed"].mean() * 100
print(f"Insight: policies sold by first-year agents lapse at {ten.iloc[0]:.1f}% vs {exp_rate:.1f}% for agents with "
      f"more than a year's experience.")

# %%
# ---- Chart 6: engagement and complaints ----
df["login_band"] = pd.cut(df["portal_logins_6m"], [-1, 0, 2, 5, np.inf], labels=["0", "1-2", "3-5", "6+"])
df["complaint_band"] = df["complaints_6m"].clip(upper=3).astype(int).astype(str).replace({"3": "3+"})
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
lg = rate_by("login_band")
cp = rate_by("complaint_band")
bar_rate(axes[0], lg, "By portal logins (first 6 months)", color=GREEN)
bar_rate(axes[1], cp, "By complaints (first 6 months)", color=RED)
for a in axes:
    a.tick_params(axis="x", rotation=0)
fig.suptitle("Chart 6 · Lapse rate by engagement and complaints", y=1.02)
savefig(fig, "06_lapse_by_engagement_and_complaints")
print(f"Insight: customers with zero logins lapse at {lg.iloc[0]:.1f}% vs {lg.iloc[-1]:.1f}% for 6+ logins; lapse climbs "
      f"from {cp.iloc[0]:.1f}% with no complaints to {cp.iloc[-1]:.1f}% with 3+.")

# %%
# ---- Chart 7: correlation heatmap ----
corr_cols = ["lapsed", "premium_to_income", "complaints_6m", "service_requests_6m", "portal_logins_6m",
             "other_policies_held", "autopay_flag", "new_agent_flag", "age", "annual_income_lakh",
             "log_premium", "policy_term_years"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5, ax=ax,
            cbar_kws={"label": "Pearson correlation"})
ax.set_title("Chart 7 · Correlation heatmap")
savefig(fig, "07_correlation_heatmap")
c = corr["lapsed"].drop("lapsed").sort_values()
print(f"Insight: strongest positive correlations with lapse - {', '.join(c.index[::-1][:2])}; "
      f"strongest negative - {', '.join(c.index[:2])}.")

# %% [markdown]
# ## 5 · Time-ordered split
# Policies issued **before July 2024** are used for training / cross-validation; policies issued **July – December 2024** are the
# held-out test period. This mimics real use (train on the past, score the future) and avoids the optimism of a random split.

# %%
train = df[df["issue_date"] < TEST_START]
test = df[df["issue_date"] >= TEST_START]
X_train, y_train = train[FEATURES], train["lapsed"]
X_test, y_test = test[FEATURES], test["lapsed"]
print(f"Train: {len(train):,} policies ({train['issue_date'].min().date()} to {train['issue_date'].max().date()}), "
      f"lapse rate {y_train.mean():.1%}")
print(f"Test : {len(test):,} policies ({test['issue_date'].min().date()} to {test['issue_date'].max().date()}), "
      f"lapse rate {y_test.mean():.1%}  <- random-list PR-AUC baseline")

# %% [markdown]
# ## 6 · Three models × three imbalance strategies (stratified 5-fold CV on the training period)
# * **Models:** Logistic Regression, Random Forest, LightGBM.
# * **Strategies:** `none`, `class_weight='balanced'`, `SMOTE`.
# * SMOTE sits **inside** an `imblearn` Pipeline, so synthetic samples are created only from each fold's training part.
# * Metrics: **PR-AUC** (headline for imbalanced data), ROC-AUC, and recall / precision / F1 at the default 0.5 threshold.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc",
           "recall": make_scorer(recall_score, zero_division=0),
           "precision": make_scorer(precision_score, zero_division=0),
           "f1": make_scorer(f1_score, zero_division=0)}

rows = []
for m in MODEL_NAMES:
    for s in STRATEGIES:
        t0 = time.time()
        res = cross_validate(make_pipeline(m, s), X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        rows.append({"model": m, "strategy": s,
                     **{k: res[f"test_{k}"].mean() for k in scoring},
                     "pr_auc_std": res["test_pr_auc"].std()})
        print(f"  {m:<20} {s:<13} PR-AUC={rows[-1]['pr_auc']:.3f}  recall@0.5={rows[-1]['recall']:.2f}  "
              f"({time.time() - t0:.0f}s)")
cv_results = pd.DataFrame(rows)
display(cv_results.round(3))

# %%
# ---- Chart 8: imbalance strategies compared ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
sns.barplot(data=cv_results, x="model", y="pr_auc", hue="strategy", ax=axes[0], palette="Set2")
axes[0].axhline(y_train.mean(), ls="--", c="grey")
axes[0].set_title("CV PR-AUC (dashed = random baseline)")
axes[0].set_xlabel("")
sns.barplot(data=cv_results, x="model", y="recall", hue="strategy", ax=axes[1], palette="Set2")
axes[1].set_title("CV recall at threshold 0.5")
axes[1].set_xlabel("")
fig.suptitle("Chart 8 · Imbalance strategies compared (5-fold stratified CV)", y=1.03)
savefig(fig, "08_imbalance_strategies_compared")
lr = cv_results[cv_results["model"] == "Logistic Regression"].set_index("strategy")
print(f"Insight: within each model PR-AUC barely changes across strategies (spread "
      f"{cv_results.groupby('model')['pr_auc'].apply(lambda s: s.max() - s.min()).max():.3f} at most), but class_weight lifts "
      f"Logistic Regression recall at 0.5 from {lr.loc['none', 'recall']:.2f} to {lr.loc['class_weight', 'recall']:.2f}. "
      f"Imbalance handling mainly moves the threshold, not the ranking quality.")

# %% [markdown]
# ## 7 · Best combination and test-period evaluation
# The best (model, strategy) pair is chosen on **CV PR-AUC** (training data only). For the test-period curves each model is
# refitted on the whole training period with its own best strategy.

# %%
best = cv_results.sort_values("pr_auc", ascending=False).iloc[0]
print(f"Best on CV PR-AUC: {best['model']} + {best['strategy']}  (PR-AUC {best['pr_auc']:.3f} ± {best['pr_auc_std']:.3f})")

best_by_model = cv_results.loc[cv_results.groupby("model")["pr_auc"].idxmax()]
fitted, test_proba, test_rows = {}, {}, []
for _, r in best_by_model.iterrows():
    pipe = make_pipeline(r["model"], r["strategy"]).fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    fitted[r["model"]], test_proba[r["model"]] = pipe, p
    test_rows.append({"model": r["model"], "strategy": r["strategy"],
                      "pr_auc": average_precision_score(y_test, p), "roc_auc": roc_auc_score(y_test, p),
                      "recall": recall_score(y_test, pred, zero_division=0),
                      "precision": precision_score(y_test, pred, zero_division=0),
                      "f1": f1_score(y_test, pred, zero_division=0)})
test_results = pd.DataFrame(test_rows)
display(test_results.round(3))
base = y_test.mean()
print(f"Random-list PR-AUC baseline = {base:.3f}")

# %%
# ---- Chart 9: PR and ROC curves on the test period ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for (name, p), colr in zip(test_proba.items(), [BLUE, GREEN, RED]):
    pr_, rc_, _ = precision_recall_curve(y_test, p)
    fpr, tpr, _ = roc_curve(y_test, p)
    row = test_results[test_results["model"] == name].iloc[0]
    axes[0].plot(rc_, pr_, color=colr, label=f"{name} (PR-AUC {row['pr_auc']:.3f})")
    axes[1].plot(fpr, tpr, color=colr, label=f"{name} (ROC-AUC {row['roc_auc']:.3f})")
axes[0].axhline(base, ls="--", c="grey", label=f"Random ({base:.3f})")
axes[1].plot([0, 1], [0, 1], ls="--", c="grey")
axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (Jul-Dec 2024)")
axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC (Jul-Dec 2024)")
axes[0].legend()
axes[1].legend()
fig.suptitle("Chart 9 · Test-period curves", y=1.02)
savefig(fig, "09_pr_and_roc_curves_test")
top = test_results.sort_values("pr_auc", ascending=False).iloc[0]
print(f"Insight: all three models reach ROC-AUC of {test_results['roc_auc'].min():.2f}-{test_results['roc_auc'].max():.2f}; "
      f"{top['model']} has the best PR-AUC ({top['pr_auc']:.3f}, about {top['pr_auc'] / base:.1f}x the {base:.3f} random baseline).")

# %% [markdown]
# ## 8 · Cost-based call threshold
# For a customer with lapse probability *p*, calling costs ₹450 and saves 30% × ₹18,000 = ₹5,400 in expectation, so the
# break-even is roughly *p* ≈ 450 / 5,400 ≈ 0.08. Rather than trusting that formula (model probabilities are not perfectly
# calibrated, especially with `class_weight` or SMOTE), the threshold is **chosen empirically on out-of-fold training
# predictions** – never on the test data – by minimising
#
# `expected cost = calls × ₹450 + missed lapses × ₹18,000 + called lapses × 70% × ₹18,000`
#
# *Note:* the ~0.08 break-even assumes perfectly calibrated probabilities. `class_weight` and SMOTE deliberately inflate the scores
# of the rare class, so with those strategies the cost-optimal cut-off lands higher. That is exactly why the threshold is read off
# the cost curve instead of being fixed in advance.

# %%
final_name, final_strategy = best["model"], best["strategy"]
oof = cross_val_predict(make_pipeline(final_name, final_strategy), X_train, y_train, cv=cv,
                        method="predict_proba")[:, 1]
grid = np.round(np.arange(0.01, 0.99, 0.01), 2)
cost_curve = np.array([expected_cost(y_train, oof, t) for t in grid])
THRESHOLD = float(grid[cost_curve.argmin()])
no_call = no_call_cost(y_train)
print(f"Chosen threshold = {THRESHOLD:.2f}")
print(f"OOF expected cost: no calls = ₹{no_call:,.0f} | threshold {THRESHOLD:.2f} = ₹{cost_curve.min():,.0f} | "
      f"threshold 0.50 = ₹{expected_cost(y_train, oof, 0.5):,.0f}")

# %%
# ---- Chart 10: expected cost vs threshold ----
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(grid, cost_curve / 1e5, color=BLUE, lw=2, label="Expected cost (out-of-fold)")
ax.axhline(no_call / 1e5, ls="--", c=GREY, label="No calls at all")
ax.axvline(THRESHOLD, ls=":", c=RED)
ax.scatter([THRESHOLD], [cost_curve.min() / 1e5], c=RED, zorder=5, label=f"Minimum at threshold {THRESHOLD:.2f}")
ax.set(xlabel="Call threshold (risk score)", ylabel="Expected cost (₹ lakh)",
       title="Chart 10 · Expected cost vs threshold")
ax.legend()
savefig(fig, "10_expected_cost_vs_threshold")
print(f"Insight: cost is lowest at a threshold of {THRESHOLD:.2f}, saving ₹{(no_call - cost_curve.min()) / 1e5:,.1f} lakh "
      f"on the training period versus making no calls, compared with ₹{(no_call - expected_cost(y_train, oof, 0.5)) / 1e5:,.1f} "
      f"lakh at the default 0.5. Where the minimum sits depends on the imbalance strategy: class_weight / SMOTE inflate the "
      f"scores, so the best cut-off moves up from the ~0.08 break-even; with 'none' it sits close to 0.08.")

# %%
# ---- Chart 11: confusion matrix on the test period at the chosen threshold ----
final_pipe = fitted[final_name]
p_test = test_proba[final_name]
flag = (p_test >= THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, flag).ravel()
recall_t, precision_t = tp / (tp + fn), tp / max(tp + fp, 1)
f1_t = f1_score(y_test, flag)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(np.array([[tn, fp], [fn, tp]]), annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Not called", "Called"], yticklabels=["Stays in-force", "Lapses"])
ax.set_title(f"Chart 11 · Confusion matrix (threshold {THRESHOLD:.2f})")
ax.set_xlabel("Model decision")
ax.set_ylabel("Actual outcome")
savefig(fig, "11_confusion_matrix_test")
print(f"Insight: at threshold {THRESHOLD:.2f} the model flags {tp + fp:,} of {len(y_test):,} test policies and catches "
      f"{tp} of the {tp + fn} lapses (recall {recall_t:.0%}, precision {precision_t:.0%}).")

# %%
# ---- savings on the test period ----
test_no_call = no_call_cost(y_test)
test_cost = expected_cost(y_test, p_test, THRESHOLD)
test_cost_05 = expected_cost(y_test, p_test, 0.5)
net_saving = test_no_call - test_cost
calls_cost = (tp + fp) * CALL_COST
saved_premium_value = tp * SAVE_RATE * LOSS_PER_LAPSE
print(f"Test period (Jul-Dec 2024), {len(y_test):,} policies")
print(f"  Calls made                : {tp + fp:,}  (cost ₹{calls_cost:,.0f})")
print(f"  Value of policies retained: ₹{saved_premium_value:,.0f}  ({tp} lapses reached x {SAVE_RATE:.0%} save rate x ₹{LOSS_PER_LAPSE:,})")
print(f"  Net saving vs no calls    : ₹{net_saving:,.0f}")
print(f"  Net saving at threshold 0.5: ₹{test_no_call - test_cost_05:,.0f}")

pr_auc_test = average_precision_score(y_test, p_test)
roc_auc_test = roc_auc_score(y_test, p_test)
criteria = {
    "Test PR-AUC >= 3x lapse rate": bool(pr_auc_test >= 3 * base),
    "Recall >= 60% at chosen threshold (Jul-Dec 2024)": bool(recall_t >= 0.60),
    "Positive net saving vs no calls": bool(net_saving > 0),
}
for k, v in criteria.items():
    print(("  PASS  " if v else "  FAIL  ") + k)

# %% [markdown]
# ## 9 · Who lapses and why (charts 12–13)

# %%
# ---- Chart 12: cumulative gains ----
order = np.argsort(-p_test)
ys = y_test.values[order]
x = np.arange(1, len(ys) + 1) / len(ys)
gains = np.cumsum(ys) / ys.sum()
top20 = gains[int(0.2 * len(ys)) - 1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(np.r_[0, x], np.r_[0, gains], color=BLUE, lw=2, label=f"{final_name} ranking")
ax.plot([0, 1], [0, 1], ls="--", c=GREY, label="Random call list")
ax.scatter([0.2], [top20], c=RED, zorder=5)
ax.annotate(f"Top 20% of policies\ncontain {top20:.0%} of lapses", (0.2, top20), xytext=(0.33, top20 - 0.17),
            arrowprops=dict(arrowstyle="->"))
ax.set(xlabel="Share of policies called (ranked by risk)", ylabel="Share of all lapses captured",
       title="Chart 12 · Cumulative gains (Jul-Dec 2024)")
ax.legend(loc="lower right")
savefig(fig, "12_cumulative_gains")
print(f"Insight: ranking by predicted risk, the top 20% of policies contain {top20:.0%} of all lapses, against 20% for a random call list.")

# %%
# ---- Chart 13: permutation importance (on the test period, scored by PR-AUC) ----
pi = permutation_importance(final_pipe, X_test, y_test, scoring="average_precision",
                            n_repeats=10, random_state=RANDOM_STATE, n_jobs=1)
imp = (pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean, "std": pi.importances_std})
       .sort_values("importance", ascending=False))
top_imp = imp.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_imp["feature"], top_imp["importance"], xerr=top_imp["std"], color=BLUE)
ax.set(xlabel="Drop in PR-AUC when the feature is shuffled", title="Chart 13 · Top lapse drivers (permutation importance)")
savefig(fig, "13_permutation_importance")
display(imp.head(10).round(4))
print(f"Insight: the most important feature is {imp['feature'].iloc[0]}, followed by "
      f"{', '.join(imp['feature'].iloc[1:5])}.")

# %% [markdown]
# ## 10 · Monthly call list with reasons
# A ranked list for the retention desk. Each row carries a short reason so the customer-service team knows how to open the call.

# %%
scored = test[["policy_id", "issue_date", "product_type", "sales_channel", "payment_method", "premium_to_income",
               "autopay_flag", "premium_mode", "complaints_6m", "portal_logins_6m", "new_agent_flag"]].copy()
scored["risk_score"] = p_test.round(4)
scored["call"] = scored["risk_score"] >= THRESHOLD
call_list = scored[scored["call"]].sort_values("risk_score", ascending=False).copy()
call_list["reason"] = reason_codes(call_list)
call_list = call_list[["policy_id", "risk_score", "reason", "product_type", "sales_channel", "payment_method"]]
call_list.to_csv(REP / "call_list_test_period.csv", index=False)
display(call_list.head(10))

# %% [markdown]
# ## 11 · Insights and recommendations

# %%
aff_lo, aff_hi = aff.iloc[0], aff.loc[">25%"]
cash_rate = df.loc[df["payment_method"] == "Cash/Cheque", "lapsed"].mean() * 100
auto_rate = df.loc[df["autopay_flag"] == 1, "lapsed"].mean() * 100

insights = [
    f"Only {df['lapsed'].mean():.1%} of policies lapse, so accuracy is meaningless; PR-AUC and recall are the right metrics "
    f"(test PR-AUC {pr_auc_test:.3f} is {pr_auc_test / base:.1f}x the {base:.3f} random baseline).",
    f"Affordability is a leading driver (top feature: {imp['feature'].iloc[0]}): lapse rises from {aff_lo:.1f}% when the premium is under 3% of income to "
    f"{aff_hi:.0f}% above 25%.",
    f"Payment method matters: Cash/Cheque payers lapse at {cash_rate:.1f}% versus {auto_rate:.1f}% for NACH/UPI AutoPay mandates.",
    f"Sales quality differs by source: {ch.index[0]} ({ch.iloc[0]:.1f}%) and {pr.index[0]} ({pr.iloc[0]:.1f}%) lapse most, "
    f"{ch.index[-1]} ({ch.iloc[-1]:.1f}%) and {pr.index[-1]} ({pr.iloc[-1]:.1f}%) least; first-year agents lapse at {ten.iloc[0]:.1f}%.",
    f"{final_name} ({final_strategy}) with a cost-based threshold of {THRESHOLD:.2f} flags {tp + fp:,} of {len(y_test):,} test "
    f"policies, catches {recall_t:.0%} of lapses and saves about ₹{net_saving / 1e5:,.1f} lakh versus making no calls "
    f"(top 20% of risk scores hold {top20:.0%} of lapses).",
]
recommendations = [
    "Affordability gate at underwriting: for premium above 15% of income, require income proof, suggest a lower premium or "
    "a longer term, and flag the case for an early month-4 check-in call.",
    "Move payers to auto-pay at issue: make NACH / UPI AutoPay the default, and give cash/cheque and monthly-mode customers "
    "a small incentive (or reminder schedule) to switch before month 10.",
    "Fix sales quality at the source: review suitability and incentives for bancassurance and ULIP sales, and add mentoring or "
    "a clawback-linked persistency target for first-year agents.",
    f"Run the model every month: call policies scoring at or above {THRESHOLD:.2f} (about {(tp + fp) / len(y_test):.0%} of the book), "
    f"highest risk first, and open each call with the reason code from the call list.",
]
print("KEY INSIGHTS")
for i, t in enumerate(insights, 1):
    print(f"{i}. {t}")
print("\nRECOMMENDATIONS")
for i, t in enumerate(recommendations, 1):
    print(f"{i}. {t}")

# %% [markdown]
# ## 12 · Save everything

# %%
metrics_cv = cv_results.assign(stage="cv_5fold_train_period")
metrics_test = test_results.assign(stage="test_jul_dec_2024_at_0.5")
metrics_final = pd.DataFrame([{"stage": "test_jul_dec_2024_at_cost_threshold", "model": final_name,
                               "strategy": final_strategy, "pr_auc": pr_auc_test, "roc_auc": roc_auc_test,
                               "recall": recall_t, "precision": precision_t, "f1": f1_t}])
metrics = pd.concat([metrics_cv, metrics_test, metrics_final], ignore_index=True)
metrics = metrics[["stage", "model", "strategy", "pr_auc", "roc_auc", "recall", "precision", "f1", "pr_auc_std"]]
metrics.round(4).to_csv(REP / "metrics.csv", index=False)

summary = {
    "dataset": {"raw_rows": int(len(raw)), "clean_rows": int(len(df)), "lapse_rate": round(float(df["lapsed"].mean()), 4),
                "train_rows": int(len(train)), "test_rows": int(len(test)), "test_lapse_rate": round(float(base), 4)},
    "best_model": {"model": final_name, "imbalance_strategy": final_strategy,
                   "cv_pr_auc": round(float(best["pr_auc"]), 4)},
    "business_assumptions": {"call_cost_inr": CALL_COST, "loss_per_lapse_inr": LOSS_PER_LAPSE, "save_rate": SAVE_RATE},
    "threshold": THRESHOLD,
    "test_metrics_at_threshold": {"pr_auc": round(float(pr_auc_test), 4), "pr_auc_vs_random": round(float(pr_auc_test / base), 2),
                                  "roc_auc": round(float(roc_auc_test), 4), "recall": round(float(recall_t), 4),
                                  "precision": round(float(precision_t), 4), "f1": round(float(f1_t), 4),
                                  "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                                  "policies_flagged": int(tp + fp), "top20pct_captures_lapses": round(float(top20), 4)},
    "savings_test_period_inr": {"no_call_cost": float(test_no_call), "model_cost": float(test_cost),
                                "net_saving": float(net_saving), "net_saving_at_threshold_0.5": float(test_no_call - test_cost_05)},
    "success_criteria": criteria,
    "top_drivers": imp.head(7)["feature"].tolist(),
    "insights": insights,
    "recommendations": recommendations,
}
(REP / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

md = ["# Results (auto-generated)", "", "## Key insights", *[f"{i}. {t}" for i, t in enumerate(insights, 1)], "",
      "## Recommendations", *[f"{i}. {t}" for i, t in enumerate(recommendations, 1)], ""]
(REP / "insights.md").write_text("\n".join(md), encoding="utf-8")

joblib.dump({"pipeline": final_pipe, "threshold": THRESHOLD, "features": FEATURES,
             "model_name": final_name, "imbalance_strategy": final_strategy,
             "business_assumptions": summary["business_assumptions"]}, MODELS / "lapse_model.joblib")

print("Saved:")
for p in [PROC / "life_policies_clean.csv", REP / "metrics.csv", REP / "summary.json", REP / "insights.md",
          REP / "call_list_test_period.csv", MODELS / "lapse_model.joblib"]:
    print("  ", p.relative_to(ROOT))
print(f"   {len(list(FIG.glob('*.png')))} charts in outputs/figures/")
display(metrics.round(3))
