"""Recompute headline numbers of the paper from the released detector scores (CPU only, a few seconds).

Usage:  python scripts/reproduce_key_numbers.py [--results results]

Detection rate (DR) is the share of a malicious set scored above the detector's threshold, which is fixed at a
1% false-positive rate on the 3,792 legitimate validation emails. Thresholds of the first seed are read from
results/thresholds.json; thresholds of extra seeds are recomputed from results/scores_val.parquet with the same rule.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ALPHA = 0.01
MAIN = ["LR-word", "SVM-char", "ZS-1.5B", "ZS-L3B", "Probe-1.5B", "Probe-3B", "LoRA-Gen-1.5B", "LoRA-Prompt-1.5B",
        "LoRA-Mean-1.5B", "LoRA-1.5B", "LoRA-3B", "LoRA-L3B", "RoBERTa-FT", "LoRA-AugMix-1.5B"]
SETS = ["goodword", "ham_pad", "paraphrase", "neutral", "inject", "inject_aware", "para_inject"]


def thr_at_fpr(legit_scores, alpha=ALPHA):
    s = np.sort(np.asarray(legit_scores, float))
    return s[min(int(np.floor((1 - alpha) * len(s))), len(s) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    R = Path(ap.parse_args().results)
    thr = json.loads((R / "thresholds.json").read_text())["THR"][str(ALPHA)]
    val = pd.read_parquet(R / "scores_val.parquet")
    val_legit = val[val.label == 0]

    def threshold(det):
        return thr[det] if det in thr else thr_at_fpr(val_legit[f"s_{det}"].dropna())

    def load(name):
        d = pd.read_parquet(R / f"scores_{name}.parquet")
        return d[d["valid"].astype(bool)] if "valid" in d else d

    print("Table 1 (first seed): detection rate (%) at 1% validation FPR")
    frames = {s: load(s) for s in SETS}
    rows = []
    for det in MAIN:
        t = threshold(det)
        rows.append({"detector": det, **{s: round(100 * (f[f"s_{det}"] > t).mean(), 1) if f"s_{det}" in f else np.nan
                                         for s, f in frames.items()}})
    print(pd.DataFrame(rows).set_index("detector").to_string(), "\n")

    print("Natural padding (80 words) over seeds, mean +- SD (%)")
    hp = frames["ham_pad"]
    for base in ["LoRA-1.5B", "LoRA-Gen-1.5B", "LoRA-Mean-1.5B", "LoRA-Prompt-1.5B", "RoBERTa-FT", "LoRA-AugMix-1.5B"]:
        cols = [base] + [c[2:] for c in hp.columns if c.startswith(f"s_{base}@s")]
        v = [100 * (hp[f"s_{c}"] > threshold(c)).mean() for c in cols]
        print(f"  {base:18s} {np.mean(v):5.1f} +- {np.std(v, ddof=1):4.1f}   ({len(v)} seeds)")
    print()

    print("False-positive rate (%) on legitimate mail with ordinary instructions, per seed")
    ri = load("legit_realinstr")
    it = ri.kind.astype(str).str.startswith("an IT").values
    for base in ["LoRA-1.5B", "LoRA-AugMix-1.5B"]:
        cols = [base] + [c[2:] for c in ri.columns if c.startswith(f"s_{base}@s")]
        f = {c: (ri[f"s_{c}"] > threshold(c)).values for c in cols}
        print(f"  {base:18s} all: " + ", ".join(f"{100 * v.mean():.1f}" for v in f.values())
              + " | IT notices: " + ", ".join(f"{100 * v[it].mean():.0f}" for v in f.values()))


if __name__ == "__main__":
    main()
