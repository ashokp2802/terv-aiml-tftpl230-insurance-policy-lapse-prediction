"""
Synthetic data generator for the Life-Insurance Policy Lapse project (TFTPL230).

The handbook describes the dataset (12,524 rows x 23 columns, Jan 2023 - Dec 2024,
~8.6% lapse after cleaning, specific data-quality problems) but does not ship the CSV.
This script builds a dataset with the SAME schema, the SAME data-quality problems and
lapse patterns that follow the handbook's EDA findings:

  * affordability (premium / income) is the strongest driver
  * Cash/Cheque payers lapse far more than NACH / UPI AutoPay payers (worst on monthly mode)
  * Bancassurance and ULIPs lapse most; Online and Term lapse least
  * first-year agents lapse more; portal logins and other policies reduce lapse;
    complaints raise it

If you have the real file, just drop it at data/raw/life_policies_13m.csv and the rest of
the project runs on it unchanged.

Usage:  python src/generate_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_UNIQUE = 12_400          # rows after de-duplication
N_DUP = 124                # ~1% double-uploaded rows  -> 12,524 raw rows
TARGET_LAPSE = 0.0905


def _pick(rng, options, probs, size):
    return rng.choice(options, size=size, p=np.array(probs) / np.sum(probs))


def build_clean_frame(n=N_UNIQUE, seed=SEED):
    rng = np.random.default_rng(seed)

    # ---- issue dates: Jan 2023 - Dec 2024 (last 6 months = ~22.7% of policies) ----
    month_w = np.r_[np.ones(18), np.full(6, 0.88)]
    month_idx = rng.choice(24, size=n, p=month_w / month_w.sum())
    day = rng.integers(1, 29, size=n)
    dates = pd.Series(pd.to_datetime(
        {"year": 2023 + month_idx // 12, "month": month_idx % 12 + 1, "day": day}
    ))
    dates = dates.sort_values(kind="stable").reset_index(drop=True)

    # ---- product, channel, payment, mode ----
    product = _pick(rng, ["Term", "Endowment", "ULIP", "Money-back", "Pension"],
                    [30, 22, 20, 16, 12], n)
    channel = np.empty(n, dtype=object)
    ch_probs = {                        # Agent, Bancassurance, Online, Broker
        "Term":       [30, 15, 30, 25],
        "ULIP":       [35, 40, 10, 15],
        "Endowment":  [55, 20, 5, 20],
        "Money-back": [55, 20, 5, 20],
        "Pension":    [55, 20, 5, 20],
    }
    for p, pr in ch_probs.items():
        m = product == p
        channel[m] = _pick(rng, ["Agent", "Bancassurance", "Online", "Broker"], pr, m.sum())

    pay = np.empty(n, dtype=object)
    pay_probs = {                       # NACH, UPI AutoPay, Card, Cash/Cheque
        "Agent":         [33, 17, 15, 35],
        "Bancassurance": [45, 15, 20, 20],
        "Online":        [20, 45, 35, 0],
        "Broker":        [40, 20, 20, 20],
    }
    for c, pr in pay_probs.items():
        m = channel == c
        pay[m] = _pick(rng, ["NACH auto-debit", "UPI AutoPay", "Card", "Cash/Cheque"], pr, m.sum())

    mode = np.empty(n, dtype=object)
    mode_probs = {                      # Annual, Half-yearly, Quarterly, Monthly
        "NACH auto-debit": [28, 10, 20, 42],
        "UPI AutoPay":     [25, 10, 20, 45],
        "Card":            [45, 15, 20, 20],
        "Cash/Cheque":     [25, 15, 25, 35],
    }
    for pm, pr in mode_probs.items():
        m = pay == pm
        mode[m] = _pick(rng, ["Annual", "Half-yearly", "Quarterly", "Monthly"], pr, m.sum())
    flip = (product == "Term") & (rng.random(n) < 0.35)
    mode[flip] = "Annual"

    # ---- customer profile ----
    gender = _pick(rng, ["Male", "Female"], [62, 38], n)
    occupation = _pick(rng, ["Salaried", "Self-employed", "Business", "Farmer", "Gig worker"],
                       [45, 20, 15, 8, 12], n)
    city_tier = _pick(rng, ["Tier 1", "Tier 2", "Tier 3"], [40, 35, 25], n)
    age = np.clip(rng.normal(38, 9.5, n), 21, 65).round().astype(int)

    inc_med = {"Salaried": 8.0, "Self-employed": 9.0, "Business": 12.0, "Farmer": 3.5, "Gig worker": 3.2}
    tier_mult = {"Tier 1": 1.25, "Tier 2": 1.0, "Tier 3": 0.8}
    income = np.array([inc_med[o] * tier_mult[t] for o, t in zip(occupation, city_tier)])
    income = np.clip(income * rng.lognormal(0, 0.5, n), 1.2, 80).round(2)

    ratio_med = {"Term": 0.035, "Endowment": 0.09, "ULIP": 0.10, "Money-back": 0.085, "Pension": 0.08}
    ratio = np.array([ratio_med[p] for p in product]) * rng.lognormal(0, 0.6, n)
    premium = np.maximum((income * 1e5 * ratio / 100).round() * 100, 3000.0)
    ratio = premium / (income * 1e5)

    mult = np.where(product == "Term", rng.uniform(120, 320, n), rng.uniform(10, 25, n))
    sum_assured = (premium * mult / 10_000).round() * 10_000
    term_opts = {"Term": [20, 25, 30, 35, 40], "Endowment": [15, 20, 25], "ULIP": [10, 15, 20],
                 "Money-back": [20, 25], "Pension": [15, 20, 25]}
    term_years = np.array([rng.choice(term_opts[p]) for p in product])
    has_rider = np.where(rng.random(n) < np.where(product == "Term", 0.45, 0.25), "Yes", "No")

    # ---- agent & engagement ----
    tenure = np.full(n, np.nan)
    is_agent = channel == "Agent"
    tenure[is_agent] = np.maximum(1, rng.gamma(1.2, 40, is_agent.sum()).round())
    new_agent = is_agent & (tenure < 12)

    other_pol = np.clip(rng.poisson(0.9, n), 0, 6)
    online = (channel == "Online").astype(float)
    cash = (pay == "Cash/Cheque").astype(float)
    eng = rng.normal(0, 1, n)
    logins = rng.poisson(np.exp(1.0 + 0.45 * eng + 0.6 * online - 0.3 * cash))
    dissat = rng.normal(0, 1, n)
    complaints = np.clip(rng.poisson(0.18 * np.exp(0.5 * dissat + 0.5 * (ratio > 0.15))), 0, 4)
    service_req = rng.poisson(0.8 + 0.5 * complaints + 0.1 * np.exp(0.3 * dissat))
    contact_ok = rng.random(n) < 0.88
    nominee_ok = rng.random(n) < 0.82

    # ---- latent lapse model ----
    prod_eff = pd.Series(product).map({"Term": -0.7, "Endowment": 0.0, "ULIP": 0.55,
                                       "Money-back": -0.15, "Pension": -0.2}).values
    chan_eff = pd.Series(channel).map({"Agent": 0.0, "Bancassurance": 0.45, "Online": -0.9,
                                       "Broker": 0.1}).values
    pay_eff = pd.Series(pay).map({"NACH auto-debit": 0.0, "UPI AutoPay": -0.1, "Card": 0.5,
                                  "Cash/Cheque": 1.1}).values
    mode_eff = pd.Series(mode).map({"Annual": 0.0, "Half-yearly": 0.2, "Quarterly": 0.4,
                                    "Monthly": 0.65}).values
    mode_eff = mode_eff + 0.5 * ((pay == "Cash/Cheque") & (mode == "Monthly"))
    afford = 0.95 * np.log(ratio / 0.08) + 1.0 * (ratio > 0.25) + 0.8 * (ratio > 0.40)
    occ_eff = pd.Series(occupation).map({"Salaried": 0, "Self-employed": 0.1, "Business": 0.0,
                                         "Farmer": 0.2, "Gig worker": 0.3}).values
    z = (prod_eff + chan_eff + pay_eff + mode_eff + afford + occ_eff
         + 0.55 * new_agent - 0.28 * other_pol - 0.45 * np.log1p(logins)
         + 0.55 * complaints + 0.05 * service_req
         - 0.35 * contact_ok - 0.25 * nominee_ok + 0.25 * (age < 30)
         + rng.normal(0, 0.6, n))

    lo, hi = -12.0, 6.0                 # solve the intercept so the overall lapse rate hits the target
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > TARGET_LAPSE:
            hi = mid
        else:
            lo = mid
    p = 1 / (1 + np.exp(-(z + mid)))
    lapsed = rng.random(n) < p

    df = pd.DataFrame({
        "policy_id": [f"LP{i:06d}" for i in range(1, n + 1)],
        "issue_date": dates.dt.strftime("%Y-%m-%d"),
        "product_type": product, "premium_mode": mode, "payment_method": pay,
        "sales_channel": channel, "annual_premium_inr": premium, "sum_assured_inr": sum_assured,
        "policy_term_years": term_years, "has_rider": has_rider, "age": age,
        "gender": gender, "occupation": occupation, "annual_income_lakh": income,
        "city_tier": city_tier, "agent_tenure_months": tenure,
        "other_policies_held": other_pol, "portal_logins_6m": logins,
        "service_requests_6m": service_req, "complaints_6m": complaints,
        "contact_verified": np.where(contact_ok, "Yes", "No"),
        "nominee_registered": np.where(nominee_ok, "Yes", "No"),
        "status_13m": np.where(lapsed, "Lapsed", "In-force"),
    })
    return df


def inject_quality_issues(df, seed=SEED):
    """Add the messy-data problems listed in the handbook's Dataset section."""
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    n = len(df)

    # spelling variants (~30% of rows get an alternative spelling)
    variants = {
        "premium_mode": {"Annual": ["Yearly", "ANNUAL", "ANL"],
                         "Half-yearly": ["Half Yearly", "HLY", "Half-Yearly"],
                         "Quarterly": ["QLY", "quarterly", "Qtrly"],
                         "Monthly": ["Mly", "MONTHLY", "Mthly"]},
        "payment_method": {"NACH auto-debit": ["NACH", "ECS", "nach auto debit"],
                           "UPI AutoPay": ["UPI", "UPI Autopay", "upi autopay"],
                           "Card": ["CARD", "Credit/Debit Card", "card"],
                           "Cash/Cheque": ["Cheque", "Cash", "CHEQUE"]},
        "status_13m": {"In-force": ["Inforce", "IN-FORCE", "In force"],
                       "Lapsed": ["LAPSED", "lapsed", "Lapse"]},
    }
    for col, mapping in variants.items():
        swap = rng.random(n) < 0.30
        orig = df[col].copy()
        for canon, alts in mapping.items():
            idx = np.where(swap & (orig == canon).values)[0]
            df.loc[df.index[idx], col] = rng.choice(alts, size=len(idx))

    # mixed date formats: 10% DD-MM-YYYY
    idx = rng.choice(n, int(0.10 * n), replace=False)
    d = pd.to_datetime(df.loc[df.index[idx], "issue_date"])
    df.loc[df.index[idx], "issue_date"] = d.dt.strftime("%d-%m-%Y")

    # invalid ages (18), income placeholders (25), reversal entries (12)
    df["age"] = df["age"].astype(float)
    pool = rng.permutation(n)
    df.loc[df.index[pool[:18]], "age"] = rng.choice([0, 7, 112, 140], size=18)
    df.loc[df.index[pool[18:43]], "annual_income_lakh"] = 999.0
    df.loc[df.index[pool[43:55]], "annual_premium_inr"] *= -1

    # missing values: income 5%, contact_verified 2%, 4% blanks inside the Agent channel
    df.loc[df.index[(rng.random(n) < 0.05) & ~df.index.isin(df.index[pool[18:43]])],
           "annual_income_lakh"] = np.nan
    df.loc[df.index[rng.random(n) < 0.02], "contact_verified"] = np.nan
    agent_blank = (df["sales_channel"] == "Agent").values & (rng.random(n) < 0.04)
    df.loc[df.index[agent_blank], "agent_tenure_months"] = np.nan
    df["age"] = df["age"].astype(int)

    # exact duplicate rows (double uploads), placed next to the original
    dup = df.iloc[rng.choice(n, N_DUP, replace=False)]
    df = pd.concat([df, dup]).sort_index(kind="stable").reset_index(drop=True)
    return df


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "raw" / "life_policies_13m.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = inject_quality_issues(build_clean_frame())
    raw.to_csv(out, index=False)
    print(f"Wrote {out}  ({raw.shape[0]:,} rows x {raw.shape[1]} columns)")


if __name__ == "__main__":
    main()
