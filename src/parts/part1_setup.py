# %% [markdown]
# # Trained, Not Prompted: Read-out Choice, Shortcut Robustness and Natural Padding in Small-LLM Phishing Detection
#
# **Notebook version 5.2** (Kaggle, GPU T4 x2; runs on one GPU or CPU, more slowly)
#
# ## Summary
# Local small language models (SLMs) are increasingly proposed as phishing filters, prompted zero-shot or fine-tuned. This notebook holds the **backbone fixed and
# varies only the read-out**: the same forward pass is read (i) through the probability of a generated label word (prompted), (ii) through a linear probe on its
# hidden states, or (iii) through a LoRA-fine-tuned classification head. Every attack is paired with a **control** that changes the input the same way without the
# adversarial ingredient, the comparison is **replicated across three backbones from two model families**, against **stronger prompted baselines** (4-shot, 7B),
# and on **modern, out-of-time data**. Five classical detectors and three hybrid defenses complete the comparison.
#
# ## Contributions (each tested in this notebook)
# * **C1. The read-out, not the model, decides injection susceptibility.** An embedded instruction inverts the ranking of the prompted read-out (AUC below
#   chance) while a LoRA head on the same backbone is unaffected; measured net of a length-matched no-instruction control, on a threshold-free scale, at three
#   backbones and two families, with 4-shot and 7B prompted baselines, and on modern data.
# * **C2. Clean-accuracy layer selection can pick the most injection-susceptible probe layer**: layer-wise instruction effects with CIs.
# * **C3. Apparent robustness can be a shortcut**: probes and learned stackers that "resist" injection do so by treating instruction-like text as malicious, which
#   is paid for in false positives on benign mail that carries such text (collateral sets).
# * **C4. Natural padding, not word salad, is the threat**: good-word robustness of SLMs is a word-salad artifact; natural benign text defeats every detector and
#   hybrid, including an adaptive attacker; dose response by padding share.
# * **C5. Padding-aware fine-tuning, and how far it generalizes**: `LoRA-Aug` (suffix padding, 3 seeds) and `LoRA-AugMix` (mixed positions and lengths), tested on
#   padding they never saw (prefix, interleaved, 240 words, LLM-written text, hidden HTML), against an augmentation-aware adaptive attacker, on modern data, and
#   with their cost on benign mail.
#
# | Family | Detectors |
# |---|---|
# | Classical ML | `NB`, `LR-word` (word 1-2 gram TF-IDF + logistic regression), `SVM-char` (char 3-5 gram TF-IDF + calibrated linear SVM), `RF`, `XGBoost` |
# | SLM read-outs, per backbone (Qwen2.5-1.5B, Qwen2.5-3B, and Llama-3.2-3B or SmolLM2-1.7B) | `ZS-<tag>` (prompted), `Probe-<tag>` (logistic probe, same forward pass), `LoRA-<tag>` (LoRA head) |
# | Stronger prompted baselines | `ZS-4shot-1.5B` (4 demonstrations), `ZS-7B` (Qwen2.5-7B-Instruct, 4-bit) |
# | Defenses (primary backbone) | `LoRA-Aug`, `LoRA-AugMix`, `Hybrid-OR` (FPR-matched union), `Hybrid-Stack` (cross-fitted stacker), `Stack-Aug`, `Triage` |
#
# | Input manipulation | Adversarial version | Matched control |
# |---|---|---|
# | Appended text | `goodword`: 80 legitimate-indicative words (word salad) | `ham_pad` (natural pad): 80 words of a real legitimate email, also an attack in its own right |
# | Out-of-distribution padding (v4.0) | prefix, interleaved, 240 words, LLM-written benign text, hidden HTML | |
# | Embedded instruction | `inject` (unseen wording), `inject_aware` (uses the exact label word) | `neutral`: same position and length, no instruction |
# | Rewording | `paraphrase` by a different model family (Phi-3.5-mini) | (clean original) |
# | Combined | `para_inject`; adaptive black-box attacker with word-salad, natural and out-of-distribution arms (100 candidates per email) | |
# | Benign mail (collateral) | legitimate emails with an instruction-like suffix, natural padding or out-of-distribution padding | the same emails clean |
#
# ## Research questions
# * **RQ1 (detection).** Classical versus SLM detectors on clean mail, on held-out sources and on modern out-of-time mail.
# * **RQ2 (read-out).** For a fixed backbone, does susceptibility to embedded instructions depend on the read-out beyond appending neutral text, across backbones, families, prompts and model size, and at which probe layers?
# * **RQ3 (padding).** Is SLM robustness to good-word padding real, and how damaging is natural and out-of-distribution padding?
# * **RQ4 (defense).** Do hybrids and padding-aware training recover robustness at a fixed FPR, against an adaptive attacker, and at what cost on benign mail?
# * **RQ5 (stability).** Seeds (LoRA and LoRA-Aug), leave-one-source-out, a held-out-source attack matrix, and human validity checks of the attacks.
#
# ## What changed in version 5.4 (second expert review, CPU only)
# Section 14c adds seven analyses on cached scores and attack logs: paired McNemar tests of LoRA-AugMix against LR-word under the adaptive padding
# attacks; LR-word term contributions before and after padding (phishing vs spam); uncapped measures for the adaptive injection attack; recent
# human-written PhishFuzzer mail by malicious type; IT-like emails in each seed's training sample; partial AUC (FPR at most 1%); and a more legible
# adaptive-attack figure (fig35). No model is trained or rerun. Attach the **v5.3 output** and run as before.
#
# ## What changed in version 5.3 (validity audit)
# * **Generator-artifact check** (section 14b): fixed patterns flag placeholders (`[Your Name]`, `[URL]`), notes about the rewrite, duplicated second versions
#   and leaked prompt text in every paraphrase, URLs lost in the rewrite, and "(Word count: N)" or meta-commentary in every ordinary-instruction email. The
#   flags are checked against the filled audit sheets, and paraphrase detection and collateral false positives are re-reported on the artifact-free emails.
# * **Audit agreement**: Cohen's kappa is reported as undefined (NaN) when every item gets the same rating, and raw agreement is reported alongside it.
# * No GPU job runs. Attach the **v5.2 output** and a private dataset holding the five `*_audit_filled.csv` sheets, keep the usual settings, then Save & Run All.
#
# ## What changed in version 5.2 (second peer review)
# * **Seeds for every one-factor read-out variant**: `LoRA-Gen`, `LoRA-Prompt`, `LoRA-Mean` and `RoBERTa-FT` get seeds 43 and 44 (seed 42 exists), so the
#   read-out ranking under natural padding is reported as mean and SD over three seeds and tested across seeds (section 14.1).
# * **Adaptive injection attack with a matched control**: the score-guided attacker adds instruction sentences addressed to the classifier (generic and
#   label-aware, 400 sentences) and, as the control, benign sentences of the same length distribution, under the same budget (3 steps, 16 candidates, 75
#   words). The instruction-attributable success is the paired difference. Targets: `LoRA-Gen` and `LoRA-1.5B` (trained) and `ZS-7B` and `ZS-L3B` (prompted).
# * **Phishing-only pool**: 500 phishing and fraud test emails (Nazario, Nigerian-Fraud) outside the main attack pool receive natural padding (80 words,
#   prefix, 240 words), the neutral suffix and the label-aware injection; the score-guided and label-only padding attacks are rerun on 120 of them.
# * **Collateral false positives by kind of ordinary instruction**, and a GPU-time account from the job logs.
# * Attach the **v5.1 output**: everything is reused and only the v5.2 jobs run (about 6 to 7 hours on two T4 GPUs).
#
# ## What changed in version 5.1
# * **Adaptive attack on fusion**: the score-guided padding attacker now targets the fused detectors themselves, `Hybrid-OR` (`LR-word` OR `LoRA-1.5B`) and the
#   new `Hybrid-OR-AugMix` (`LR-word` OR `LoRA-AugMix-1.5B`), both FPR-matched on validation (section 13.6b).
# * **Label-only attacker**: the same padding attack when the attacker only learns whether the email was flagged, against `LR-word`, `LoRA-1.5B`,
#   `LoRA-AugMix-1.5B` and both fused detectors.
# * **Modern data for the v5.0 detectors**: `LoRA-Gen`, `LoRA-Prompt`, `LoRA-Mean`, `RoBERTa` and spotlighting rows in the PhishFuzzer tables (scores already cached).
# * Fig. 28 legend moved below the plot. Attach the **v5.0 output**: everything is reused and only the new attack jobs run (about 2 hours).
#
# ## What changed in version 5.0 (response to a full peer review)
# Every new item is a separately named job or analysis; the v3.1/v4.0 cache is reused, so attaching the finished **v4.0 output** runs only this list.
# * **Read-out ablation that separates "trained vs prompted" from "head vs label word"**: `LoRA-Gen` (LoRA fine-tuned to *generate* MALICIOUS or BENIGN for the
#   prompted read-out's own prompt, read out exactly like `ZS`) and `LoRA-Prompt` (classification head trained on the prompted input instead of the raw email).
# * **Score-guided adaptive attacks**: a black-box greedy padding attack (the attacker sees the detector's score and adds the benign sentence that lowers it most,
#   up to 240 words) against `LR-word`, `LoRA`, `LoRA-Aug`, `LoRA-AugMix`, `LoRA-Gen` and the encoder; and a white-box GCG-style optimized suffix against `LoRA`,
#   `LoRA-AugMix` and the encoder, with success checked after decoding the suffix to text.
# * **Seeds**: `LoRA-AugMix` seeds 43 and 44; one more seed for `LoRA-3B` and `LoRA-L3B` (if time allows). Headlines are reported as mean and SD over seeds.
# * **Operating point**: natural padding reported at 1% and 5% FPR and on the threshold-free scale, plus two score-normalization defenses that need no retraining:
#   length-conditional thresholds and max-over-windows scoring (`LoRA-Win`).
# * **Phishing vs spam**: every key result split into phishing/fraud sources (Nazario, Nigerian-Fraud) and the spam-heavy sources.
# * **Fine-tuned encoder baseline**: `RoBERTa-FT` (and `RoBERTa-AugMix`), to test whether the padding blind spot is specific to LLM heads.
# * **Realistic benign mail with instructions**: legitimate emails carrying ordinary instructions (forwarded assistant answers, IT notices, prompt-writing
#   newsletters, task lists) instead of injection templates, for an unarguable collateral false-positive test.
# * **Mechanism**: last-token vs mean pooling (`LoRA-Mean`), detection by truncation status and email length, and position effects.
# * **Prompt-injection defense for the prompted read-out**: spotlighting by datamarking (`ZS-Spot`) on the 1.5B, Llama and 7B models; the 7B model is now
#   also scored on natural padding, collateral and modern sets.
#
# ## What changed in version 4.0
# * Second model family (Llama-3.2-3B-Instruct with a Hugging Face token that has accepted Meta's license; otherwise SmolLM2-1.7B-Instruct, no token needed).
# * Stronger prompted baselines: 4-shot prompt on the primary backbone, and Qwen2.5-7B-Instruct in 4-bit precision.
# * Out-of-distribution padding sets, an augmentation-aware adaptive arm, `LoRA-Aug` seeds 43 and 44, `LoRA-AugMix`, and padding on legitimate mail.
# * Modern data: PhishFuzzer (human seeds up to 2025 and LLM-rephrased 2026), decontaminated against the training corpus.
# * Human validity sheets for paraphrases and padded emails, with Cohen's kappa once filled. LOSO tables report trained detectors only and separate single-class sources.
# * v4.0 keeps the v3.1/v3.2 cache: attach a finished v3.2 output and only the v4.0 work runs.
#
# ## Positioning (related work to cite)
# * Prompt injection against **prompted** LLM phishing detectors: *Clouding the Mirror* (arXiv 2602.05484), *LLM-PEA* (arXiv 2512.10104), *Inference Attacks and Defenses on LLM-Powered Phishing Email Detection Systems* (HumSec 2026). None holds the backbone fixed and varies the read-out, uses a no-instruction control, or measures collateral cost.
# * Hidden-state probes for **detecting** injections or task drift: arXiv 2406.00799, 2505.06311, 2608.02657, 2610.18408. Here the probe is the task classifier itself.
# * Prompting versus trained heads on the same LLM features (arXiv 2608.02415, intent routing; no injection). Good-word attacks: Lowd and Meek (CEAS 2005). LLM paraphrase attacks: arXiv 2411.13874.
# * Modern evaluation data: PhishFuzzer (Toth, Gruschka and Bisztray, arXiv 2511.21448; CC BY-NC 4.0).
#
# ## Protocol highlights (for reviewers)
# * Exact and near-duplicate removal; source-stratified splits; LOSO including LoRA for every source; a full attack matrix on a held-out source (Enron); modern data decontaminated against the training corpus.
# * Thresholds fixed **on validation** at a target FPR (primary 1%, secondary 5%) from all legitimate validation emails, never re-tuned on test, attacked or modern data.
# * Injection detector and stackers are cross-fitted. Defender wording (`T_train`) and padding text are disjoint from the attacker's (`T_test`, attacker padding pool); `LoRA-AugMix` never sees LLM-written or hidden padding.
# * Statistics: paired bootstrap 95% CIs, exact McNemar with Holm correction, threshold-free AUC effects and difference-in-differences, Yule's Q, exact query-budget ASR curves, seeds.
# * GPU stages run as parallel jobs on both T4s with identical code (`cbs_slm.py`); every stage is cached and resumable; an optional job that fails is reported and excluded instead of stopping the run.
#
# ## How to run
# 1. **Add Input**: the dataset **Phishing Email Dataset** (`naserabdullahalam/phishing-email-dataset`), your **finished v5.3 version output** (Add Input > Your Work) to reuse all earlier GPU work (the cache tag is unchanged, so no GPU job reruns), and the private dataset with the five `*_audit_filled.csv` sheets.
# 2. Settings: **Accelerator = GPU T4 x2**, **Internet = On** (models and PhishFuzzer are downloaded). For Llama-3.2-3B: accept the license at huggingface.co/meta-llama/Llama-3.2-3B-Instruct and add your token as the Kaggle secret `HF_TOKEN` (Add-ons > Secrets); without it the notebook uses SmolLM2-1.7B automatically.
# 3. **Save Version > Save & Run All (Commit)**. With the v5.3 output attached nothing new runs on the GPU (about 1 to 2 hours of reloading and statistics). With only the v4.0 output: about 11 to 13 hours, so expect the lowest-priority jobs to be deferred to a second session (attach this output and run again). From scratch: two or three sessions (attach each session's output to the next).
# 4. If the summary lists deferred jobs, attach this version's output and run again; finished work is skipped.
# 5. Outputs: `/kaggle/working/cbs/` (figures as PDF + PNG, LaTeX tables, raw scores, `summary.json`, `key_findings.md`) and `cbs_paper_assets.zip`.
# 6. Human checks: download `cbs/private/paraphrase_audit.csv` and `padding_audit.csv`, fill the `rater1_`/`rater2_` `_yn` columns (1 or 0), save as `*_audit_filled.csv`, upload them as a private Kaggle dataset, attach it, and rerun (minutes, everything else is cached).
#
# > **Ethics and licensing.** Adversarial variants of *existing* malicious emails are generated for defensive evaluation only; generated texts stay in `cbs/cache` and `cbs/private`, which are excluded from the paper-assets zip. Keep the notebook and its output **private**. PhishFuzzer is CC BY-NC 4.0: non-commercial research use with attribution.

# %% [markdown]
# ## 1. Environment setup

# %%
import importlib, subprocess, sys, os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")      # keeps the saved notebook free of download widgets
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _ensure(pkg, pip_name=None):
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"Installing {pip_name or pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or pkg], check=False)

_ensure("peft", "peft>=0.11")
_ensure("accelerate")
_ensure("pyarrow")
_ensure("xgboost")
if os.environ.get("CBS_SMOKE", "0") != "1":
    _ensure("bitsandbytes")     # 4-bit loading of the 7B prompted baseline (v4.0)

# %%
import re, gc, json, time, math, glob, shutil, random, hashlib, platform, warnings, threading, zipfile
from collections import Counter
from pathlib import Path
from itertools import combinations
T_START = time.time()

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
import joblib
from scipy import stats
from scipy.special import comb, expit, logit
from scipy.stats import norm, gaussian_kde
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             f1_score, brier_score_loss)
import torch
import transformers

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
pd.set_option("display.width", 200, "display.max_columns", 40)

# %% [markdown]
# ## 2. Configuration
#
# Edit `CFG` to change scale. Environment variables prefixed `CBS_` override paths and models (used by the CPU smoke test; not needed on Kaggle).

# %%
SMOKE = os.environ.get("CBS_SMOKE", "0") == "1"   # tiny CPU end-to-end test; keep False on Kaggle

CFG = dict(
    run_name="cbs_v3",
    seed=42,
    data_root=os.environ.get("CBS_DATA_ROOT", "/kaggle/input/datasets/naserabdullahalam/phishing-email-dataset"),
    input_root=os.environ.get("CBS_INPUT_ROOT", "/kaggle/input"),   # searched for a previous run's cache (resume)
    work_dir=os.environ.get("CBS_WORK_DIR", "/kaggle/working"),

    # Backbones for the read-out study. The first is the primary backbone (hybrids, adaptive attacker, seeds, LOSO).
    backbones=[
        dict(tag="1.5B", model=os.environ.get("CBS_BACKBONE_1", "Qwen/Qwen2.5-1.5B-Instruct"), lora_seeds=[42, 43, 44]),
        dict(tag="3B", model=os.environ.get("CBS_BACKBONE_2", "Qwen/Qwen2.5-3B-Instruct"), lora_seeds=[42]),
    ],
    attacker_model=os.environ.get("CBS_ATTACKER_MODEL", "microsoft/Phi-3.5-mini-instruct"),   # different family on purpose
    semsim_model=os.environ.get("CBS_SEMSIM_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),  # "none" to skip
    hf_token_secret="HF_TOKEN",

    # Text handling
    max_chars=3000, min_chars=40,
    slm_max_tokens=512,                # head+tail truncation keeps prefix/suffix content visible
    lora_max_tokens=384,

    # Sample sizes
    n_val_legit=None,                  # None = every legitimate validation email (threshold estimation at 1% FPR)
    n_val_mal=1000,
    n_val_inj=2000,                    # validation emails given T_train injections (injection detector / stacker)
    n_val_pad=1000,                    # validation malicious emails given defender-side natural padding (Stack-Aug)
    n_test_legit=3000, n_test_mal=1000,
    n_attack=800, n_adaptive=300,
    n_probe_train=4000, n_lora_train=8000,

    # LoRA fine-tuning
    lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_lr=2e-4, lora_epochs=1,
    lora_batch=8, lora_grad_accum=2, lora_warmup=0.05,
    lora_loso_sources=["Enron", "Nazario"],   # LoRA retrained without these sources (primary backbone, seed 42; full eval sets)
    lora_loso_more=["CEAS-08", "Ling", "Nigerian-Fraud", "SpamAssassin"],   # added in v3.2: LOSO-only LoRA runs, so every source is covered
    lora_aug=dict(enabled=True, frac=0.5, k=80),   # added in v3.2: LoRA trained with defender-side natural padding on both classes
    xsrc_source="Enron",                       # full attack matrix with every detector trained without this source

    # Inference / generation
    force_fp32=False,                  # set True only if fp16 produces NaN scores on your GPU
    score_batch=16, gen_batch=8, gen_max_new_tokens=512, gen_input_max_tokens=700,
    gen_temperature=0.7, gen_top_p=0.9, paraphrase_shards=2,

    # Attacks and controls
    goodword_k=80, adaptive_pad_levels=[0, 40, 80], goodword_lexicon_size=400,
    para_min_semsim=0.6, para_len_range=[0.4, 2.5],
    n_benign_suffix=800,               # added in v3.2: legitimate test emails carrying a neutral suffix / an injection (collateral false positives)
    adaptive_nat_levels=[40, 80],      # added in v3.2: natural-padding candidates for the adaptive attacker

    # ---- v4.0 additions (all excluded from the cache hash: they only add new, separately named jobs and sets) ----
    # A second model family, read out the same three ways. Candidates are tried in order; a gated model without access is skipped.
    extra_backbones=[dict(candidates=[(os.environ.get("CBS_BACKBONE_3", "meta-llama/Llama-3.2-3B-Instruct"), "L3B"),
                                      ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "S1.7B")], lora_seeds=[42])],
    # Stronger prompted baselines: the primary backbone with a 4-shot prompt, and a 7B model in 4-bit precision (zero-shot).
    prompt_fewshot=dict(enabled=True, k=4, shot_tokens=96),
    prompt_large=dict(enabled=True, model=os.environ.get("CBS_LARGE_MODEL", "Qwen/Qwen2.5-7B-Instruct"), tag="7B", quant4=True),
    # Out-of-distribution padding: positions, lengths and text sources that LoRA-Aug never saw in training.
    ood_pad=dict(k=80, k_long=240, n_llm_paragraphs=400, adaptive=True),
    lora_aug_seeds=[43, 44],                   # extra seeds for LoRA-Aug (seed 42 exists since v3.2)
    lora_augmix=dict(enabled=True, frac=0.5, k_range=[40, 240], mix=True),   # mixed positions and lengths; hidden HTML and LLM text held out
    # Modern data: PhishFuzzer (Toth, Gruschka and Bisztray, 2026; CC BY-NC 4.0): human seed emails up to 2025 and LLM-rephrased variants (2026).
    modern=dict(enabled=True, local_glob="**/PhishFuzzer_emails_*_v1.json",
                base_url="https://raw.githubusercontent.com/DataPhish/PhishFuzzer/main/",
                files=["PhishFuzzer_emails_original_seed_v1.json", "PhishFuzzer_emails_entity_rephrased_v1.json"],
                min_year_human=2019, n_llm=3000, n_attack=600, decontam_cos=0.8),
    n_pad_audit=50,                            # padded emails exported for the human validity check

    # ---- v5.0 additions (all excluded from the cache hash: new, separately named jobs, sets and analyses) ----
    v5=dict(
        lora_gen=True,                         # LoRA trained to generate the label word for the prompted read-out's prompt
        lora_prompt=True,                      # LoRA classification head on the prompted input
        lora_mean=True,                        # LoRA head with mean pooling (pooling ablation)
        window=dict(enabled=True, words=160, stride=160),   # max-over-windows scoring of the trained LoRA head (no retraining)
        encoder=dict(enabled=True, model=os.environ.get("CBS_ENCODER_MODEL", "FacebookAI/roberta-base"), tag="RoBERTa", max_tokens=512,
                     lr=2e-5, epochs=2, batch=16, warmup=0.06, augmix=True),
        augmix_seeds=[43, 44],
        big_seeds=[43],                        # one extra LoRA seed for the 3B and Llama backbones (lowest priority)
        spot=dict(enabled=True, extra_backbone=True, large=True),
        large_more_sets=True,                  # 7B: natural padding, collateral and modern sets too
        realinstr=dict(enabled=True, n_paragraphs=400),
        greedy=dict(enabled=True, n_emails=150, steps=8, cands=24, max_words=240, positions=["suffix", "prefix"], n_sentences=3000),
        gcg=dict(enabled=True, n_emails=30, suffix_tokens=20, steps=25, topk=64, batch=24),
        v51=dict(hybrid_attack=True, label_only=True),   # v5.1: adaptive attack on the fused detectors; label-only attacker
        # v5.2: (1) two more seeds for every one-factor read-out variant; (2) an adaptive INJECTION attack with a matched benign-sentence control
        # on prompted and trained read-outs; (3) a phishing-only manipulation pool and adaptive attacks on phishing emails.
        v52=dict(
            readout_seeds=[43, 44], readout_variants=["gen", "prompt", "mean", "encoder"],
            inj_attack=dict(enabled=True, n_emails=100, steps=3, cands=16, max_words=75, positions=["suffix", "prefix"], n_sentences=400,
                            targets=["gen", "lora", "large", "extra_zs"]),
            phish=dict(enabled=True, n_pool=500, n_greedy=120, attack_targets=["lora", "augmix"], label_targets=["augmix"]),
        ),
    ),


    # Evaluation
    target_fprs=[0.01, 0.05], primary_fpr=0.01, n_boot=1000, ece_bins=10, n_folds=5,
    loso_max_train=30000,

    # Scheduling
    time_budget_h=11.0,                # optional GPU jobs are skipped if they would end after this; rerun with cache to finish
    max_parallel_jobs=None,            # None = one job per GPU (two on T4 x2)
)

if SMOKE:
    CFG.update(n_val_legit=60, n_val_mal=40, n_val_inj=40, n_val_pad=20, n_test_legit=60, n_test_mal=40, n_attack=24, n_adaptive=8,
               n_probe_train=80, n_lora_train=80, lora_batch=4, lora_grad_accum=1, score_batch=8, gen_batch=4,
               gen_max_new_tokens=48, gen_input_max_tokens=128, slm_max_tokens=160, lora_max_tokens=128,
               n_boot=100, primary_fpr=0.05, target_fprs=[0.05, 0.10], n_folds=3, loso_max_train=600,
               para_min_semsim=0.0, para_len_range=[0.0, 100.0], n_benign_suffix=24)
    CFG["backbones"][1]["lora_seeds"] = [42]
    CFG["ood_pad"].update(n_llm_paragraphs=12, k_long=120)
    CFG["modern"].update(n_llm=60, n_attack=16, min_year_human=2019)
    CFG["n_pad_audit"] = 6
    CFG["v5"]["encoder"].update(epochs=1, batch=8, max_tokens=128)
    CFG["v5"]["realinstr"].update(n_paragraphs=12)
    CFG["v5"]["greedy"].update(n_emails=6, steps=2, cands=4, n_sentences=60)
    CFG["v5"]["gcg"].update(n_emails=3, suffix_tokens=4, steps=2, topk=8, batch=4)
    CFG["v5"]["v52"]["inj_attack"].update(n_emails=6, steps=2, cands=4, max_words=40, n_sentences=40)
    CFG["v5"]["v52"]["phish"].update(n_pool=24, n_greedy=6)

SEED = CFG["seed"]
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed()

# Cache signature: settings that change data, attacks or models are hashed into the cache folder name, so a resumed run only reuses
# results computed with identical settings. Evaluation-only settings are excluded, so figures and tables can be regenerated cheaply.
# CODE_VERSION names the cache generation, not the notebook: v3.2 only ADDS jobs and sets (keys below marked "added"), so it keeps the v3.1
# cache and a v3.1 output can be attached to resume. Keys that only add new, separately named cached outputs are excluded from the hash.
CODE_VERSION = "3.1"
NOTEBOOK_VERSION = "5.4"
_EVAL_ONLY = {"target_fprs", "primary_fpr", "n_boot", "ece_bins", "work_dir", "data_root", "input_root", "hf_token_secret",
              "run_name", "time_budget_h", "max_parallel_jobs", "score_batch",
              "lora_loso_more", "lora_aug", "n_benign_suffix", "adaptive_nat_levels",
              "extra_backbones", "prompt_fewshot", "prompt_large", "ood_pad", "lora_aug_seeds", "lora_augmix", "modern", "n_pad_audit", "v5"}
CACHE_TAG = f"v{CODE_VERSION.replace('.', '')}_" + hashlib.md5(
    json.dumps({k: v for k, v in CFG.items() if k not in _EVAL_ONLY}, sort_keys=True).encode()).hexdigest()[:8]

WORK = Path(CFG["work_dir"]) / "cbs"
DIRS = {k: WORK / k for k in ["figures", "tables", "results", "logs", "jobs", "code", "private"]}
DIRS["cache"] = WORK / "cache" / CACHE_TAG
for d in DIRS.values(): d.mkdir(parents=True, exist_ok=True)
(DIRS["cache"] / "sets").mkdir(exist_ok=True)

N_GPU = torch.cuda.device_count()
print(f"SMOKE={SMOKE} | GPUs={N_GPU} | cache tag={CACHE_TAG}")
print(f"python {platform.python_version()} | torch {torch.__version__} | transformers {transformers.__version__}")
for i in range(N_GPU):
    print(f"  GPU{i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB")
with open(DIRS["results"] / "config.json", "w") as f: json.dump(CFG, f, indent=2)

try:   # optional Hugging Face token from Kaggle Secrets
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret(CFG["hf_token_secret"])
    print("HF token loaded from Kaggle Secrets.")
except Exception:
    pass

def model_access(mid):
    """True if the model can be downloaded with the current token (gated models such as Llama need an accepted license and HF_TOKEN)."""
    if os.path.isdir(mid): return True, "local"
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(mid, "config.json", token=os.environ.get("HF_TOKEN")); return True, "ok"
    except Exception as e:
        return False, type(e).__name__

def _family_of(mid):
    m = mid.lower()
    return ("Llama-3.2" if "llama" in m else "SmolLM2" if "smol" in m else "Qwen2.5" if "qwen" in m else Path(mid).name)

EXTRA_BB = []
for eb in CFG["extra_backbones"]:
    for mid, tag in eb["candidates"]:
        ok, why = model_access(mid)
        print(f"Second model family candidate {mid}: {'available' if ok else 'not available (' + why + ')'}")
        if ok:
            EXTRA_BB.append(dict(tag=tag, model=mid, lora_seeds=eb["lora_seeds"], family=_family_of(mid))); break
for b in CFG["backbones"]: b.setdefault("family", _family_of(b["model"]))
LARGE_OK = CFG["prompt_large"]["enabled"] and model_access(CFG["prompt_large"]["model"])[0]
print(f"Backbones: {[(b['tag'], b['model']) for b in CFG['backbones'] + EXTRA_BB]} | 7B prompted baseline available: {LARGE_OK}")

# %% [markdown]
# ### 2.1 Caching and resume
# Expensive results live in `cbs/cache/<tag>/`. If the output of a previous run of this notebook is attached as an input, the cell below copies the cache with
# the same tag, so finished stages and finished GPU jobs are skipped.

# %%
def _restore_cache():
    restored = 0
    for src in Path(CFG["input_root"]).glob(f"**/cbs/cache/{CACHE_TAG}"):
        if src.resolve() == DIRS["cache"].resolve():
            continue
        for p in src.rglob("*"):
            dst = DIRS["cache"] / p.relative_to(src)
            if p.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            elif not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst); restored += 1
    print(f"Restored {restored} cached files (tag {CACHE_TAG}) from a previous run." if restored
          else f"No previous cache with tag {CACHE_TAG}; starting fresh.")
_restore_cache()

def cpath(name): return DIRS["cache"] / name
def have(name): return cpath(name).exists()

TIMINGS = {}
class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self): self.t = time.time(); return self
    def __exit__(self, *a):
        TIMINGS[self.name] = TIMINGS.get(self.name, 0) + time.time() - self.t
        print(f"[time] {self.name}: {time.time()-self.t:.1f}s | {mem_status()}")

def elapsed_h(): return (time.time() - T_START) / 3600

def mem_status():
    """RAM of the machine, of this kernel and of the GPU worker processes (a silent kernel death on Kaggle is almost always RAM)."""
    try:
        import psutil
        vm = psutil.virtual_memory(); me = psutil.Process()
        kids = sum(c.memory_info().rss for c in me.children(recursive=True) if c.is_running())
        return (f"RAM used {(vm.total - vm.available)/2**30:.1f}/{vm.total/2**30:.1f} GB | kernel {me.memory_info().rss/2**30:.1f} GB"
                f" | workers {kids/2**30:.1f} GB")
    except Exception as e:
        return f"RAM n/a ({e})"
print(mem_status())

# %% [markdown]
# ### 2.2 Shared model code and the GPU job runner
# All model computations live in `cbs_slm.py` (written below and imported here). GPU work is submitted as **jobs**: each job runs that module in its own
# process pinned to one GPU, so the two T4s work in parallel while the notebook continues with CPU stages. Logs go to `cbs/logs/`.
# A job whose outputs already exist is skipped; optional jobs that would finish after `time_budget_h` are deferred to a resumed run.

# %%
CBS_SLM_SRC = r'''__CBS_SLM_SOURCE__'''
MOD_PATH = DIRS["code"] / "cbs_slm.py"
MOD_PATH.write_text(CBS_SLM_SRC)
if str(DIRS["code"]) not in sys.path: sys.path.insert(0, str(DIRS["code"]))
if "cbs_slm" in sys.modules: del sys.modules["cbs_slm"]
import cbs_slm as S
if S._TORCHAO_PATCHED:
    print("Note: an old torchao is installed (Kaggle image); peft's torchao path is disabled because it is not used here.")
print(f"cbs_slm.py written ({len(CBS_SLM_SRC.splitlines())} lines); peft {S.peft.__version__}")

JOB_CFG_KEYS = ["force_fp32", "slm_max_tokens", "lora_max_tokens", "score_batch", "gen_batch", "gen_max_new_tokens",
                "gen_input_max_tokens", "gen_temperature", "gen_top_p", "lora_r", "lora_alpha", "lora_dropout", "lora_lr",
                "lora_epochs", "lora_batch", "lora_grad_accum", "lora_warmup"]
JOB_CFG = {k: CFG[k] for k in JOB_CFG_KEYS}

class JobQueue:
    """Runs jobs as subprocesses, one per slot (GPU). Non-blocking: start() returns immediately; wait() blocks."""
    def __init__(self, name):
        self.name, self.jobs, self.state, self.proc, self.t0, self.t1 = name, [], {}, {}, {}, {}
        n_slots = CFG["max_parallel_jobs"] or (max(1, N_GPU) if N_GPU else 2)
        self.slots = [str(i) if N_GPU else "" for i in range(n_slots)]
        self.free = list(self.slots); self.thread = None
    def submit(self, job, outputs, est_min=1.0, optional=False, after=()):
        """after: names of jobs in this queue that must finish first (v5.0: attack jobs need their target's trained model)."""
        job = dict(job, cfg=JOB_CFG, cache=str(DIRS["cache"]))
        jp = DIRS["jobs"] / f"{job['name']}.json"; jp.write_text(json.dumps(job, indent=1))
        self.jobs.append(dict(name=job["name"], path=jp, outputs=[Path(o) for o in outputs], est=est_min, optional=optional, after=list(after)))
        self.state[job["name"]] = "queued"
    def _done(self, j): return all(o.exists() for o in j["outputs"])
    def _ready(self, j):
        st = [self.state.get(d, "done") for d in j.get("after", [])]   # a dependency outside this queue counts as done
        if any(x in ("failed", "deferred") for x in st): return "blocked"
        return "ready" if all(x in ("done", "cached") for x in st) else "wait"
    def _loop(self):
        pending = list(self.jobs)
        while pending or any(s == "running" for s in self.state.values()):
            for n, p in list(self.proc.items()):
                if p.poll() is not None:
                    self.state[n] = "done" if p.returncode == 0 else "failed"; self.t1[n] = time.time()
                    self.free.append(self._slot[n]); del self.proc[n]
            while self.free and pending:
                k_ = None
                for i_, j_ in enumerate(pending):
                    r_ = self._ready(j_)
                    if r_ == "blocked" and not self._done(j_):
                        self.state[j_["name"]] = "deferred"; pending.pop(i_); k_ = -1; break
                    if r_ == "ready" or self._done(j_):
                        k_ = i_; break
                if k_ is None: break          # every pending job waits for a running one
                if k_ == -1: continue
                j = pending.pop(k_)
                if self._done(j):
                    self.state[j["name"]] = "cached"; continue
                if j["optional"] and elapsed_h() + j["est"] / 60 > CFG["time_budget_h"]:
                    self.state[j["name"]] = "deferred"; continue
                slot = self.free.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot, PYTHONUNBUFFERED="1")
                logf = open(DIRS["logs"] / f"{j['name']}.log", "a")
                self.proc[j["name"]] = subprocess.Popen([sys.executable, str(MOD_PATH), str(j["path"])],
                                                        stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(DIRS["code"]))
                self._slot = getattr(self, "_slot", {}); self._slot[j["name"]] = slot
                self.state[j["name"]] = "running"; self.t0[j["name"]] = time.time()
            time.sleep(3)
    def start(self):
        self._slot = {}
        self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start()
        print(f"[{self.name}] started {len(self.jobs)} jobs on slots {self.slots or ['cpu']}")
        return self
    def summary(self):
        return {n: s for n, s in self.state.items()}
    def wait(self, report_every=900):
        last = time.time()
        while self.thread.is_alive():
            time.sleep(5)
            if time.time() - last > report_every:
                run = [f"{n} ({(time.time()-self.t0[n])/60:.0f} min)" for n, s in self.state.items() if s == "running"]
                print(f"[{self.name}] {elapsed_h():.2f} h | running: {', '.join(run) or '-'} | "
                      f"done: {sum(s in ('done','cached') for s in self.state.values())}/{len(self.jobs)} | {mem_status()}", flush=True)
                last = time.time()
        failed = [n for n, s in self.state.items() if s == "failed"]
        opt = {j["name"] for j in self.jobs if j["optional"]}
        for n in self.state:
            dur = f"{(self.t1[n]-self.t0[n])/60:.1f} min" if n in self.t1 else ""
            print(f"  {n:<28} {self.state[n]:<9} {dur}")
        for n in failed:
            print(f"\n----- tail of {n}.log -----\n" + "".join((DIRS['logs'] / f"{n}.log").read_text().splitlines(True)[-40:]))
        self.failed_optional = [n for n in failed if n in opt]
        if self.failed_optional:
            print(f"[{self.name}] optional jobs failed and are left out of the analysis: {self.failed_optional}")
        core_failed = [n for n in failed if n not in opt]
        if core_failed:
            raise RuntimeError(f"[{self.name}] failed core jobs: {core_failed}. See cbs/logs/.")
        return self.summary()

# %% [markdown]
# ### 2.3 Publication figure style
# IEEE-ready defaults: Times-style serif text at 7 to 8.5 pt, single-column width 3.5 in, double-column 7.16 in, TrueType-embedded vector PDF plus 600-dpi PNG.
# Dense scatters are rasterized inside the PDF. Colors come from a palette validated for color-vision deficiency, and every series also has a marker, line style
# or direct label. Read-outs share a hue across backbone sizes (filled marker = 1.5B, hollow = 3B), so the read-out, not the size, carries the color.

# %%
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#9a9893"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CMAP_RED = LinearSegmentedColormap.from_list("seq_red", ["#fbe3e2", "#f2aaa9", "#e5706f", "#c23a3a", "#7f1d1d"])
CMAP_DIV = LinearSegmentedColormap.from_list("div_rb", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a83232"])
COL1, COL2 = 3.5, 7.16

CLS_DETS = ["NB", "LR-word", "SVM-char", "RF", "XGBoost"]
ALL_BACKBONES = [dict(b) for b in CFG["backbones"]] + EXTRA_BB
BB_TAGS = [b["tag"] for b in ALL_BACKBONES]
FAMILY_BB = {b["tag"]: b.get("family", "Qwen2.5") for b in ALL_BACKBONES}
P_TAG = BB_TAGS[0]
READOUTS = ["ZS", "Probe", "LoRA"]
READOUT_LABEL = {"ZS": "Prompted (generative label)", "Probe": "Linear probe (hidden state)", "LoRA": "LoRA classification head"}
SLM_ALL = [f"{r}-{t}" for t in BB_TAGS for r in READOUTS]
ZS_P, PROBE_P, LORA_P = (f"{r}-{P_TAG}" for r in READOUTS)
AUG_DET = f"LoRA-Aug-{P_TAG}"          # LoRA trained with defender-side natural padding (a defense, not a read-out)
AUGMIX_DET = f"LoRA-AugMix-{P_TAG}"    # v4.0: padding at mixed positions and lengths
SLM_EXTRA = [AUG_DET, AUGMIX_DET]
# v4.0 stronger prompted baselines (scored on a subset of sets; analysed separately, never inside hybrids)
FS_DET = f"ZS-4shot-{P_TAG}"
LARGE_DET = f"ZS-{CFG['prompt_large']['tag']}"
PROMPT_X = [FS_DET, LARGE_DET]
# v5.0 detectors (scored on the sets each analysis needs; analysed in section 14, never inside the v4.0 hybrids)
GEN_DET, PROMPTHEAD_DET, MEAN_DET, WIN_DET = f"LoRA-Gen-{P_TAG}", f"LoRA-Prompt-{P_TAG}", f"LoRA-Mean-{P_TAG}", f"LoRA-Win-{P_TAG}"
ENC_DET, ENC_AUGMIX_DET = f"{CFG['v5']['encoder']['tag']}-FT", f"{CFG['v5']['encoder']['tag']}-AugMix"
SPOT_DET = {t: f"ZS-Spot-{t}" for t in BB_TAGS} | {CFG["prompt_large"]["tag"]: f"ZS-Spot-{CFG['prompt_large']['tag']}"}
V5_NAMED = [GEN_DET, PROMPTHEAD_DET, MEAN_DET, WIN_DET, ENC_DET, ENC_AUGMIX_DET] + list(SPOT_DET.values())
HYB_DETS = ["Hybrid-OR", "Hybrid-Stack", "Stack-Aug"]
ABL_DETS = ["Stack-noInj"]
FAMILY = ({d: "Classical" for d in CLS_DETS} | {d: "Small LLM" for d in SLM_ALL + SLM_EXTRA + PROMPT_X + V5_NAMED} | {ENC_DET: "Encoder", ENC_AUGMIX_DET: "Encoder"}
          | {d: "Hybrid" for d in HYB_DETS + ABL_DETS})
FAM_COL = {"Classical": PAL[0], "Small LLM": PAL[1], "Hybrid": PAL[2]}
RO_COL = {"ZS": PAL[5], "Probe": PAL[6], "LoRA": PAL[7]}
RO_MARK = {"ZS": "D", "Probe": "P", "LoRA": "X"}
DET_COL = dict(zip(CLS_DETS, PAL[:5])) | {d: RO_COL[d.split("-")[0]] for d in SLM_ALL} | \
          {"Hybrid-OR": "#52514e", "Hybrid-Stack": "#0b0b0b", "Stack-Aug": "#8a8883", "Stack-noInj": "#b3b0a9", AUG_DET: "#8f1d1d",
           AUGMIX_DET: "#c2410c", FS_DET: "#4d7c0f", LARGE_DET: "#14532d"}
DET_MARK = dict(zip(CLS_DETS, ["o", "s", "^", "v", "<"])) | {d: RO_MARK[d.split("-")[0]] for d in SLM_ALL} | \
           {"Hybrid-OR": "h", "Hybrid-Stack": "*", "Stack-Aug": "p", "Stack-noInj": "H", AUG_DET: "d", AUGMIX_DET: "8", FS_DET: "s", LARGE_DET: "^"}
DET_COL |= {GEN_DET: "#b45309", PROMPTHEAD_DET: "#9d174d", MEAN_DET: "#be123c", WIN_DET: "#7c2d12", ENC_DET: "#1d4ed8", ENC_AUGMIX_DET: "#0369a1"} | \
           {d: "#65a30d" for d in SPOT_DET.values()}
DET_MARK |= {GEN_DET: "v", PROMPTHEAD_DET: "<", MEAN_DET: ">", WIN_DET: "H", ENC_DET: "o", ENC_AUGMIX_DET: "s"} | {d: "D" for d in SPOT_DET.values()}
DET_FILL = {d: ("white" if (d in SLM_ALL and not d.endswith(P_TAG)) else DET_COL[d]) for d in DET_COL}
# Backbone styles: primary solid and filled; second Qwen size dotted and hollow; other model family dash-dot and hollow
_BB_LS = {t: ("-" if i == 0 else (":" if FAMILY_BB[t] == FAMILY_BB[P_TAG] else (0, (5, 1.5, 1, 1.5)))) for i, t in enumerate(BB_TAGS)}
DET_LW = {d: (1.9 if FAMILY[d] == "Hybrid" else 1.3) for d in DET_COL}
DET_LS = ({d: ("--" if FAMILY[d] == "Classical" else "-") for d in DET_COL} | {d: _BB_LS[d.split("-")[-1]] for d in SLM_ALL}
          | {AUG_DET: "-.", AUGMIX_DET: (0, (3, 1)), FS_DET: (0, (1, 1)), LARGE_DET: (0, (4, 1, 1, 1, 1, 1))})

ATTACKS = ["clean", "goodword", "ham_pad", "paraphrase", "neutral", "inject", "inject_aware", "para_inject"]
ATT_LABEL = {"clean": "Clean", "goodword": "Good-word", "ham_pad": "Natural pad", "paraphrase": "Paraphrase",
             "neutral": "Neutral suffix (ctrl)", "inject": "Injection", "inject_aware": "Injection (aware)", "para_inject": "Para.+Inject."}
# The neutral suffix is a pure control. Natural padding is both the control for word salad and, as the v3.1 run showed, the strongest attack,
# so it counts as an attack in worst-case robustness.
CONTROLS = {"neutral"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "legend.fontsize": 6.8, "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.2,
    "axes.titlepad": 4, "axes.labelpad": 2.5, "xtick.direction": "out", "ytick.direction": "out",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "figure.dpi": 140, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

FIG_INDEX = []
def savefig(fig, name, caption=""):
    for ext in ("pdf", "png"):
        fig.savefig(DIRS["figures"] / f"{name}.{ext}")
    FIG_INDEX.append((name, caption))
    plt.show()
    plt.close(fig)

def panel_label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")

def write_latex(df, name, caption, label, col_format=None, note=None):
    """Minimal booktabs writer (no jinja2 dependency). Saves .tex and .csv and returns the LaTeX for copy-paste."""
    cols = list(df.columns)
    col_format = col_format or ("l" + "c" * (len(cols) - 1))
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\footnotesize", rf"\begin{{tabular}}{{{col_format}}}", r"\toprule",
             " & ".join(rf"\textbf{{{esc(c)}}}" for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(esc(v) for v in r.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note: lines.append(rf"\\[2pt]{{\scriptsize {esc(note)}}}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)
    (DIRS["tables"] / f"{name}.tex").write_text(tex)
    df.to_csv(DIRS["tables"] / f"{name}.csv", index=False)
    return tex
