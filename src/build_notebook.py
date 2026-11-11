"""Convert src/lapse_analysis.py (cell-marked script) into notebooks/Life_Insurance_Policy_Lapse_Prediction.ipynb."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "lapse_analysis.py"
OUT = ROOT / "notebooks" / "Life_Insurance_Policy_Lapse_Prediction.ipynb"


def _strip(lines):
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def main():
    cells, cur = [], None
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            if cur:
                cells.append(cur)
            cur = {"kind": "markdown" if "[markdown]" in line else "code", "lines": []}
        elif cur is not None:
            cur["lines"].append(line)
    if cur:
        cells.append(cur)

    nb_cells = []
    for c in cells:
        lines = _strip(c["lines"])
        if c["kind"] == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])
        cell = {"cell_type": c["kind"], "metadata": {}, "source": src}
        if c["kind"] == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb_cells.append(cell)

    nb = {"cells": nb_cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(nb_cells)} cells)")
    return OUT


if __name__ == "__main__":
    main()
