# Trained, Not Prompted

Code, detector scores and analysis outputs for the paper

> **Trained, Not Prompted: Read-out Choice and Benign Padding in Small-LLM Email Threat Detection**
> *Anonymous submission, under review.*

One open-weight language model can become an email filter in several ways: prompted to answer with a label word, probed with a linear classifier, or fine-tuned to emit the label word or through a classification head. We call this choice the **read-out**. Holding the backbone fixed, the study changes one factor of the read-out at a time on three small models (Qwen2.5-1.5B, Qwen2.5-3B, Llama-3.2-3B), with classical and encoder baselines, on 74,260 public emails at a fixed 1% false-positive rate. Every manipulation has a matched control, and defenses are tested against attackers who see nothing, only the verdict, the score, or the gradients.

## Main findings

| | Finding | Key number |
|---|---|---|
| 1 | Fine-tuning, not the output layer or input format, removes the prompt-injection weakness. | AUC lost to a label-aware instruction: 0.510 prompted vs. 0.0011 after fine-tuning the same label word (3 seeds) |
| 2 | Prompted read-outs stay injectable at 7B against an adaptive attacker, measured against an equally searched benign control. | Qwen2.5-7B: 74/74 caught emails evaded with instructions vs. 54/74 with benign sentences |
| 3 | The word-salad good-word test hides a benign-padding weakness of trained detectors. | Salad caught at least 97%; 80 words of ordinary legitimate text leave 57.8% ± 12.0 for a raw-email LoRA head |
| 4 | Padding-aware training holds against attackers who see verdicts, not against attackers who see scores, and raises false positives on instruction-bearing legitimate mail. | Label-only 17.0%, score-guided 49.0% attack success; FPR 4.1% vs. 0.7% (seed means) |
| 5 | Fusion with a classical filter does not repair a blind spot both components share. | Fused-score attacker: 94.6% (Hybrid-OR) vs. 95.3% for its strongest component |

## Repository layout

```
notebook/   trained_not_prompted.ipynb   single Kaggle notebook that runs the whole study
src/        build_notebook.py            assembles the notebook from the parts below
            cbs_slm.py                   GPU worker: scoring, LoRA training, probes, adaptive attacks
            parts/                       notebook source in jupytext percent format, one file per stage
results/    scores_*.parquet             per-email detector scores for every evaluation set (no email text)
            *.csv, summary.json          every number reported in the paper, with intervals
            thresholds.json              detector thresholds at 1% and 5% validation FPR
figures/    fig*.pdf / .png              all figures, including those in the paper
tables/     tab*.tex / .csv              all tables as LaTeX and CSV
audit/      *_ratings.csv                validity-audit ratings (identifiers and ratings only)
scripts/    reproduce_key_numbers.py     recomputes headline numbers from the released scores
docs/       REPRODUCING.md, RESULTS_INDEX.md, DATA.md, AUDIT.md, RESPONSIBLE_RELEASE.md
```

## Quick start: check the numbers (CPU, seconds)

```bash
pip install pandas pyarrow numpy
python scripts/reproduce_key_numbers.py
```

This recomputes Table 1, the seed means for natural padding and the false-positive rates on instruction-bearing legitimate mail directly from `results/scores_*.parquet`. To compute any other cell yourself:

```python
import json, pandas as pd
thr = json.load(open("results/thresholds.json"))["THR"]["0.01"]
s = pd.read_parquet("results/scores_ham_pad.parquet")          # 80 words of natural padding
s = s[s["valid"]]
print((s["s_LoRA-1.5B"] > thr["LoRA-1.5B"]).mean())           # 0.580, Table 1
```

[docs/RESULTS_INDEX.md](docs/RESULTS_INDEX.md) maps every table and figure of the paper to the file that holds its numbers.

## Full reproduction (Kaggle, GPU)

The study runs as one notebook on Kaggle with two T4 GPUs. From scratch it needs two or three 12-hour sessions; each session's output is attached to the next and finished jobs are reused from a content-addressed cache. Measured GPU time is 57.2 hours. Step-by-step instructions, including the Hugging Face access token for Llama-3.2 and the optional audit sheets, are in [docs/REPRODUCING.md](docs/REPRODUCING.md).

Software used for the reported run: Python 3.13, PyTorch 2.11, Transformers 5.20, PEFT 0.21, scikit-learn 1.6, XGBoost 3.4 (see `requirements.txt`).

## Data

The main corpus is the public [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) (CEAS-08, Enron, Ling, Nazario, Nigerian-Fraud, SpamAssassin). The modern replication uses PhishFuzzer (CC BY-NC 4.0). Neither is redistributed here; the notebook reads them from their original sources. See [docs/DATA.md](docs/DATA.md).

## What is not released, and why

The repository contains code, configuration, scores and aggregate results. It does not contain the generated adversarial emails, paraphrases, padding text or adaptive-attack outputs, because these are ready-to-use evasion material. They are available to researchers on request after the review period. See [docs/RESPONSIBLE_RELEASE.md](docs/RESPONSIBLE_RELEASE.md).

## Citation

Citation details will be added after the review period.

## License

Code: MIT ([LICENSE](LICENSE)). Scores, results, figures, tables and audit ratings: CC BY 4.0 ([LICENSE-DATA.md](LICENSE-DATA.md)). The source datasets keep their own licenses.
