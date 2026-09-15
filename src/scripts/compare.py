#!/usr/bin/env python3
"""Build THE comparison table from results/*/metrics.json.

    python scripts/compare.py                 # markdown, to stdout
    python scripts/compare.py --latex         # LaTeX, for the report

The rubric requires exactly one comparison table across all models on common
metrics, so this script is its only source.
"""

import argparse
import json
from pathlib import Path

MODEL_NAMES = {
    "m1": "Classical ORB/LK + MLP",
    "m2": "CNN + GRU/LSTM",
    "m3": "Self-supervised PoseNet+DepthNet",
    "m4": "CNN + CBAM/SE attention",
}

COLUMNS = [
    ("ate_sim3", "ATE sim(3) [m]", "{:.3f}", "min"),
    ("ate_se3", "ATE SE(3) [m]", "{:.3f}", "min"),
    ("rpe_trans_rmse", "RPE trans [m]", "{:.4f}", "min"),
    ("rpe_rot_rmse_deg", "RPE rot [deg]", "{:.4f}", "min"),
    ("t_rel_pct", "t_rel [%]", "{:.2f}", "min"),
    ("r_rel_deg_per_100m", "r_rel [deg/100m]", "{:.4f}", "min"),
    ("params_m", "Params [M]", "{:.2f}", None),
    ("ms_per_frame", "ms/frame", "{:.1f}", None),
]


def load(results_dir: Path) -> dict:
    """Load each model's metrics from ``results_dir/*/metrics.json``."""
    rows = {}
    for path in sorted(results_dir.glob("*/metrics.json")):
        rows[path.parent.name] = json.loads(path.read_text())
    return rows


def render(rows: dict, latex: bool = False) -> str:
    """Render loaded model metrics as a Markdown or LaTeX table."""
    models = sorted(rows)
    best = {}
    for key, _, _, direction in COLUMNS:
        values = [
            rows[model].get(key)
            for model in models
            if isinstance(rows[model].get(key), (int, float))
        ]
        if values and direction == "min":
            best[key] = min(values)

    header = ["Model", "Owner"] + [label for _, label, _, _ in COLUMNS]
    lines = []
    if latex:
        lines.append("\\begin{tabular}{ll" + "r" * len(COLUMNS) + "}")
        lines.append("\\hline")
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\hline")
    else:
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for model in models:
        row = rows[model]
        cells = [MODEL_NAMES.get(model, model), row.get("owner", "?")]
        for key, _, fmt, _ in COLUMNS:
            value = row.get(key)
            if not isinstance(value, (int, float)):
                cells.append("—")
                continue
            cell = fmt.format(value)
            if key in best and abs(value - best[key]) < 1e-12:
                cell = f"\\textbf{{{cell}}}" if latex else f"**{cell}**"
            cells.append(cell)
        separator = " & " if latex else " | "
        line = separator.join(cells)
        lines.append(line + " \\\\" if latex else f"| {line} |")

    if latex:
        lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = load(Path(args.results))
    if not rows:
        raise SystemExit(f"no metrics.json under {args.results}/ — run scripts/eval.py first")

    missing = sorted(set(MODEL_NAMES) - set(rows))
    table = render(rows, args.latex)

    note = (
        "\n\n> Scale handling differs by design: M1/M2/M4 use ground-truth scale "
        "alignment, M3 recovers scale from its own depth prediction. Both "
        "ATE sim(3) (scale solved) and ATE SE(3) (scale fixed) are therefore "
        "reported. Bold marks the best value per column."
    )
    if missing:
        note += f"\n>\n> Not yet evaluated: {', '.join(missing)}."

    text = table + ("" if args.latex else note)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
