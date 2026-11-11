"""
One-command runner for the Life-Insurance Policy Lapse project.

    python run_project.py              # run the whole analysis (charts, metrics, model) as a script
    python run_project.py --notebook   # build + execute the Jupyter notebook instead (needs jupyter)
    python run_project.py --new-data   # re-create the synthetic CSV first (overwrites data/raw/...)
"""
import argparse
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def check_dependencies():
    missing = []
    for mod, pip_name in [("pandas", "pandas"), ("numpy", "numpy"), ("sklearn", "scikit-learn"),
                          ("imblearn", "imbalanced-learn"), ("lightgbm", "lightgbm"),
                          ("matplotlib", "matplotlib"), ("seaborn", "seaborn"), ("joblib", "joblib")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        sys.exit("Missing packages: " + ", ".join(missing) + "\nInstall them with:  pip install -r requirements.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", action="store_true", help="build and execute the notebook instead of the script")
    ap.add_argument("--new-data", action="store_true", help="regenerate the synthetic dataset first")
    args = ap.parse_args()
    check_dependencies()

    if args.new_data:
        import generate_data
        generate_data.main()

    if args.notebook:
        import build_notebook
        nb = build_notebook.main()
        subprocess.check_call([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                               "--inplace", "--ExecutePreprocessor.timeout=1800", str(nb)])
        print(f"Executed notebook saved to {nb}")
    else:
        runpy.run_path(str(ROOT / "src" / "lapse_analysis.py"), run_name="__main__")


if __name__ == "__main__":
    main()
