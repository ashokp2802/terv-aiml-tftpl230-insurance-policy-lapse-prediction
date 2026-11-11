"""
Shared helpers for the Life-Insurance Policy Lapse project.

Used by the notebook / analysis script (src/lapse_analysis.py) and by src/predict.py.
Keeping them here means cleaning, feature engineering and modelling are written once.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Business constants (from the handbook)
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
CALL_COST = 450          # INR per retention call / agent visit
LOSS_PER_LAPSE = 18_000  # INR lost margin + commission + issue cost per lapsed policy
SAVE_RATE = 0.30         # share of called customers who are retained
TEST_START = "2024-07-01"  # time-ordered split: train before, test from this date

MODEL_NAMES = ["Logistic Regression", "Random Forest", "LightGBM"]
STRATEGIES = ["none", "class_weight", "SMOTE"]


def find_root(start=None):
    """Return the project root (the folder that contains data/ and src/)."""
    start = Path(start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "data").is_dir() and (p / "src").is_dir():
            return p
    raise RuntimeError("Project root not found - run from inside the project folder.")


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def _norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


MODE_MAP = {"annual": "Annual", "yearly": "Annual", "anl": "Annual",
            "halfyearly": "Half-yearly", "hly": "Half-yearly",
            "quarterly": "Quarterly", "qly": "Quarterly", "qtrly": "Quarterly",
            "monthly": "Monthly", "mly": "Monthly", "mthly": "Monthly"}
PAY_MAP = {"nachautodebit": "NACH auto-debit", "nach": "NACH auto-debit", "ecs": "NACH auto-debit",
           "upiautopay": "UPI AutoPay", "upi": "UPI AutoPay",
           "card": "Card", "creditdebitcard": "Card",
           "cashcheque": "Cash/Cheque", "cash": "Cash/Cheque", "cheque": "Cash/Cheque"}
TARGET_MAP = {"inforce": "In-force", "lapsed": "Lapsed", "lapse": "Lapsed"}


def drop_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


def parse_dates(df):
    """Parse both YYYY-MM-DD and DD-MM-YYYY. The two formats cannot be confused because the
    4-digit year sits in a different position, so each is parsed with an explicit format."""
    out = df.copy()
    iso = pd.to_datetime(out["issue_date"], format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(out["issue_date"], format="%d-%m-%Y", errors="coerce")
    out["issue_date"] = iso.fillna(dmy)
    if out["issue_date"].isna().any():
        raise ValueError(f"{out['issue_date'].isna().sum()} issue_date values could not be parsed")
    return out


def standardise_categories(df):
    """Map every spelling variant of premium_mode, payment_method and the target to one label."""
    out = df.copy()
    for col, mapping in [("premium_mode", MODE_MAP), ("payment_method", PAY_MAP),
                         ("status_13m", TARGET_MAP)]:
        if col not in out.columns:
            continue
        mapped = out[col].map(lambda v: mapping.get(_norm(v)))
        unknown = out.loc[mapped.isna(), col].unique()
        if len(unknown):
            raise ValueError(f"Unmapped spellings in {col}: {list(unknown)} - add them to the map.")
        out[col] = mapped
    return out


def fix_invalid_values(df, age_range=(18, 80)):
    """Invalid ages and the 999 income placeholder become NaN; reversal entries (negative
    premiums) have their sign corrected."""
    out = df.copy()
    out.loc[~out["age"].between(*age_range), "age"] = np.nan
    out.loc[out["annual_income_lakh"] == 999, "annual_income_lakh"] = np.nan
    out["annual_premium_inr"] = out["annual_premium_inr"].abs()
    return out


def impute_missing(df, reference=None):
    """Group-median imputation (occupation x city tier) for age and income, structural fill for
    agent tenure, 'Unknown' for contact_verified. No target information is used.
    `reference` (default: df itself) supplies the medians - predict.py passes the training data."""
    out = df.copy()
    ref = out if reference is None else reference
    for col in ["age", "annual_income_lakh"]:
        med = ref.groupby(["occupation", "city_tier"])[col].median().rename("_grp_median")
        grp = out.join(med, on=["occupation", "city_tier"])["_grp_median"]
        out[col] = out[col].fillna(grp).fillna(ref[col].median())
    out["age"] = out["age"].round().astype(int)

    is_agent = out["sales_channel"] == "Agent"
    agent_median = ref.loc[ref["sales_channel"] == "Agent", "agent_tenure_months"].median()
    out.loc[is_agent, "agent_tenure_months"] = out.loc[is_agent, "agent_tenure_months"].fillna(agent_median)
    out.loc[~is_agent, "agent_tenure_months"] = 0.0      # not applicable -> structural zero

    out["contact_verified"] = out["contact_verified"].fillna("Unknown")
    return out


def clean_policies(raw, reference=None):
    """Full cleaning chain (used by predict.py; the notebook runs each step separately)."""
    df = drop_duplicates(raw)
    df = parse_dates(df)
    df = standardise_categories(df)
    df = fix_invalid_values(df)
    df = impute_missing(df, reference)
    if "status_13m" in df.columns:
        df["lapsed"] = (df["status_13m"] == "Lapsed").astype(int)
    return df


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
NUM_FEATURES = ["age", "annual_income_lakh", "policy_term_years", "agent_tenure_months",
                "other_policies_held", "portal_logins_6m", "service_requests_6m", "complaints_6m",
                "premium_to_income", "high_burden_flag", "autopay_flag", "new_agent_flag",
                "log_premium", "sum_assured_multiple", "login_flag", "issue_month"]
CAT_FEATURES = ["product_type", "premium_mode", "payment_method", "sales_channel", "has_rider",
                "gender", "occupation", "city_tier", "contact_verified", "nominee_registered"]
FEATURES = NUM_FEATURES + CAT_FEATURES


def add_features(df):
    out = df.copy()
    out["premium_to_income"] = out["annual_premium_inr"] / (out["annual_income_lakh"] * 1e5)
    out["high_burden_flag"] = (out["premium_to_income"] > 0.15).astype(int)
    out["autopay_flag"] = out["payment_method"].isin(["NACH auto-debit", "UPI AutoPay"]).astype(int)
    out["new_agent_flag"] = ((out["sales_channel"] == "Agent")
                             & (out["agent_tenure_months"] < 12)).astype(int)
    out["log_premium"] = np.log1p(out["annual_premium_inr"])
    out["sum_assured_multiple"] = out["sum_assured_inr"] / out["annual_premium_inr"]
    out["login_flag"] = (out["portal_logins_6m"] > 0).astype(int)
    out["issue_month"] = out["issue_date"].dt.month
    return out


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def make_pipeline(model_name, strategy):
    """preprocessing -> [SMOTE] -> classifier. SMOTE lives inside the imblearn Pipeline, so it is
    applied only to the training part of each CV fold and never leaks into validation data."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])
    cw = "balanced" if strategy == "class_weight" else None
    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=3000, class_weight=cw)
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight=cw,
                                     n_jobs=-1, random_state=RANDOM_STATE)
    elif model_name == "LightGBM":
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=30, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, class_weight=cw, random_state=RANDOM_STATE,
                             verbose=-1)
    else:
        raise ValueError(model_name)
    steps = [("prep", pre)]
    if strategy == "SMOTE":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


# ----------------------------------------------------------------------------
# Business cost logic
# ----------------------------------------------------------------------------
def expected_cost(y_true, proba, threshold):
    """Expected rupee cost of calling every policy with risk >= threshold.
    - each call costs CALL_COST
    - a lapsing policy that is NOT called is lost in full (LOSS_PER_LAPSE)
    - a lapsing policy that IS called is still lost with probability (1 - SAVE_RATE)"""
    y_true = np.asarray(y_true)
    flag = np.asarray(proba) >= threshold
    tp = (flag & (y_true == 1)).sum()
    fn = (~flag & (y_true == 1)).sum()
    return flag.sum() * CALL_COST + fn * LOSS_PER_LAPSE + tp * (1 - SAVE_RATE) * LOSS_PER_LAPSE


def no_call_cost(y_true):
    return int(np.sum(y_true)) * LOSS_PER_LAPSE


def reason_codes(df):
    """Short, human-readable reasons for each policy (for the customer-service call script)."""
    reasons = []
    for _, r in df.iterrows():
        why = []
        if r["premium_to_income"] > 0.25:
            why.append("premium above 25% of income")
        elif r["premium_to_income"] > 0.15:
            why.append("premium above 15% of income")
        if r["payment_method"] == "Cash/Cheque":
            why.append("cash/cheque payer")
        if r["autopay_flag"] == 0 and r["premium_mode"] == "Monthly":
            why.append("monthly premium without auto-pay")
        if r["complaints_6m"] >= 1:
            why.append(f"{int(r['complaints_6m'])} complaint(s) in first 6 months")
        if r["portal_logins_6m"] == 0:
            why.append("never logged in to portal")
        if r["new_agent_flag"] == 1:
            why.append("sold by first-year agent")
        if r["sales_channel"] == "Bancassurance":
            why.append("bancassurance sale")
        if r["product_type"] == "ULIP":
            why.append("ULIP product")
        reasons.append("; ".join(why[:3]) if why else "model risk score")
    return reasons
