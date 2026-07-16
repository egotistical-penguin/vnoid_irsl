from __future__ import annotations
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CPP_LOG = Path("out_cpp.txt")
PY_LOG = Path("out_py.txt")
OUT_DIR = Path("vnoid_cpp_python_compare_markers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_KEYS = ["com_ref", "dcm_ref", "dcm_target", "zmp_ref", "zmp_target"]

def parse_state_log(path: Path, language: str | None = None) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")

    if language == "C++":
        start = text.find("===== C++ INITIAL STATE")
        if start >= 0:
            text = text[start:]
        py_start = text.find("===== PYTHON INITIAL STATE")
        if py_start >= 0:
            text = text[:py_start]
    elif language == "PYTHON":
        start = text.find("===== PYTHON INITIAL STATE")
        if start >= 0:
            text = text[start:]

    header_re = re.compile(
        r"===== (?:C\+\+|PYTHON)?\s*AFTER_STEP_SIMULATION STATE "
        r"count=(\d+) time=([0-9eE+\-.]+) ====="
    )
    vec_re = re.compile(
        r"^\s*(com_ref|dcm_ref|dcm_target|zmp_ref|zmp_target)\s*="
        r"\(\s*([0-9eE+\-.]+)\s*,\s*([0-9eE+\-.]+)\s*,\s*([0-9eE+\-.]+)\s*\)",
        re.MULTILINE,
    )

    headers = list(header_re.finditer(text))
    rows = []

    for i, match in enumerate(headers):
        block_start = match.end()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[block_start:block_end]

        row = {"count": int(match.group(1)), "time": float(match.group(2))}
        for vec_match in vec_re.finditer(block):
            key = vec_match.group(1)
            vals = [float(vec_match.group(j)) for j in range(2, 5)]
            for axis, val in zip(("x", "y", "z"), vals):
                row[f"{key}_{axis}"] = val

        required = [f"{key}_{axis}" for key in VECTOR_KEYS for axis in ("x", "y", "z")]
        if all(col in row for col in required):
            rows.append(row)

    return pd.DataFrame(rows).sort_values("count").reset_index(drop=True)

cpp_df = parse_state_log(CPP_LOG)
py_df = parse_state_log(PY_LOG, language="PYTHON")
merged = pd.merge(cpp_df, py_df, on=["count", "time"], suffixes=("_cpp", "_py"))

summary_rows = []
for key in VECTOR_KEYS:
    for axis in ("x", "y", "z"):
        fig, ax = plt.subplots(figsize=(10, 5.5))

        cpp_vals = merged[f"{key}_{axis}_cpp"]
        py_vals = merged[f"{key}_{axis}_py"]
        diff = cpp_vals - py_vals

        ax.plot(merged["time"], cpp_vals, marker="^", markersize=5, label="C++ (line + triangle)")
        ax.plot(merged["time"], py_vals, marker="o", markersize=4, label="Python (line + circle)")

        ax.set_title(f"{key} ({axis.upper()}): C++ triangles / Python circles")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Value [m]")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{key}_{axis}_triangles_vs_circles.png", dpi=180)
        plt.close(fig)

        summary_rows.append(
            {
                "variable": key,
                "axis": axis,
                "max_abs_diff": float(np.max(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff**2))),
            }
        )

pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
print(f"saved to {OUT_DIR.resolve()}")