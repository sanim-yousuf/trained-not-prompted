# Reproducing the study

The whole study is one notebook, `notebook/trained_not_prompted.ipynb`, designed for Kaggle with two NVIDIA T4 GPUs. A small "smoke" mode runs the same pipeline on tiny models and data for testing.

## 1. Kaggle setup

1. Create a new Kaggle notebook and import `notebook/trained_not_prompted.ipynb` (File > Import Notebook).
2. **Add Input** > Datasets: `naserabdullahalam/phishing-email-dataset`.
3. **Settings**: Accelerator *GPU T4 x2*, Internet *On*.
4. **Secrets** (Add-ons > Secrets): add `HF_TOKEN`, a Hugging Face read token from an account that has accepted the Llama-3.2 license. The notebook reads the secret by name; never paste the token into a cell.
5. **Save Version > Save & Run All (Commit)**.

## 2. Sessions and the cache

A full run needs about 57 GPU hours, more than one 12-hour Kaggle session. The notebook schedules jobs by priority and stops before the time limit. Finished jobs are written to a cache keyed by a hash of the configuration (`v31_dbe5a9a9` for the reported run).

To continue: open the notebook again, **Add Input > Your Work** and attach the output of the previous version, then Save & Run All. Cached jobs are skipped and the remaining ones run. Repeat until the run report (last section) lists no deferred jobs. Changing a configuration value that enters the cache hash starts a new cache; evaluation-only settings do not.

## 3. Optional inputs

- **Validity audit.** The run writes blank audit sheets to `cbs/private/`. To include agreement statistics, fill the `_yn` columns (1 or 0), rename each file to `*_audit_filled.csv`, upload them as a **private** Kaggle dataset and attach it. The released ratings are in `audit/` without the email text.
- **PhishFuzzer.** Downloaded from the authors' public release when Internet is on, or read from its JSON files attached as a Kaggle input; the modern replication is skipped if neither is available.

## 4. Outputs

- `cbs_paper_assets.zip`: `results/`, `figures/` and `tables/`, the content of this repository's folders of the same names. It contains scores and aggregates only.
- `cbs/private/` (kept out of the zip): generated adversarial emails, paraphrases, padding text and audit sheets with text. Do not publish this folder.
- The last notebook section prints a run report: jobs run, cached, deferred and failed, GPU hours, and software versions.

## 5. Smoke test (CPU, minutes)

Set `CBS_SMOKE=1` to replace every model with a tiny one and every set with a few emails. Useful to check an edited notebook end to end before spending GPU time. Numbers from smoke mode are meaningless.

## 6. Editing the code

The notebook is generated. Edit the files in `src/parts/` and `src/cbs_slm.py`, then rebuild:

```bash
pip install jupytext
python src/build_notebook.py
```

## 7. Determinism

Seeds are fixed (42, with 43 and 44 for the seed study) and resample the 8,000 training emails of each trained detector. GPU kernels are not bitwise deterministic, so retraining can move individual scores slightly; the reported seed standard deviations are far larger than this effect. The released scores are the exact values behind every number in the paper.
