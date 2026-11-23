"""Assemble notebook/trained_not_prompted.ipynb from the source parts in src/parts and src/cbs_slm.py.

The notebook is the single entry point that runs on Kaggle. Its code lives here as plain Python files with
`# %%` cell markers (jupytext percent format), so diffs are readable. Requires: pip install jupytext
Usage:  python src/build_notebook.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ["part1_setup.py", "part2_data.py", "part3_attacks.py", "part4_slm_jobs.py", "part5_classical.py", "part6_loso_cpu.py",
         "part7_assemble.py", "part8_metrics.py", "part9_figures.py", "part10_tables.py", "part11_review.py", "part13_v52.py",
         "part14_v53.py", "part15_v54.py", "part12_report.py"]


def main():
    src = "\n".join((ROOT / "src" / "parts" / p).read_text().rstrip() + "\n" for p in PARTS)
    module = (ROOT / "src" / "cbs_slm.py").read_text()
    assert src.count("__CBS_SLM_SOURCE__") == 1, "placeholder for the worker module not found"
    tmp = ROOT / "src" / "_notebook_source.py"
    tmp.write_text(src.replace("__CBS_SLM_SOURCE__", module))
    out = ROOT / "notebook" / "trained_not_prompted.ipynb"
    subprocess.run([sys.executable, "-m", "jupytext", "--to", "ipynb", "--opt", "notebook_metadata_filter=-all", str(tmp), "-o", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    tmp.unlink()
    nb = json.loads(out.read_text())
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                      "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True, "dataSources": []}}
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    out.write_text(json.dumps(nb, indent=1))
    print(f"{out.relative_to(ROOT)}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
