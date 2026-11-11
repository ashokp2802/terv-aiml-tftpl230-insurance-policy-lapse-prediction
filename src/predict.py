"""
Score new policies with the saved model.

    python src/predict.py --input data/raw/life_policies_13m.csv --output outputs/reports/scored_policies.csv

The input CSV must have the same columns as the raw data (status_13m is optional). The file is cleaned
with the same steps used in training, scored, and each policy gets a call decision plus a short reason.
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lapse_utils import add_features, clean_policies, reason_codes  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV of policies in the raw-data format")
    ap.add_argument("--output", default=str(ROOT / "outputs" / "reports" / "scored_policies.csv"))
    args = ap.parse_args()

    bundle = joblib.load(ROOT / "models" / "lapse_model.joblib")
    raw = pd.read_csv(args.input)
    ref_path = ROOT / "data" / "processed" / "life_policies_clean.csv"
    reference = pd.read_csv(ref_path) if ref_path.exists() else None   # training medians for imputation

    df = add_features(clean_policies(raw, reference))
    df["risk_score"] = bundle["pipeline"].predict_proba(df[bundle["features"]])[:, 1].round(4)
    df["call"] = df["risk_score"] >= bundle["threshold"]
    df["reason"] = ""
    df.loc[df["call"], "reason"] = reason_codes(df[df["call"]])

    out = df[["policy_id", "risk_score", "call", "reason"]].sort_values("risk_score", ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Scored {len(out):,} policies | threshold {bundle['threshold']:.2f} | "
          f"{int(out['call'].sum()):,} flagged for a retention call -> {args.output}")


if __name__ == "__main__":
    main()
