# Where each number comes from

All paths are relative to the repository root. Detection rates use the thresholds in `results/thresholds.json` (key `THR` > `"0.01"`), applied to the per-email scores in `results/scores_<set>.parquet` (column `s_<detector>`, rows with `valid == True` for manipulated sets). Detector names with a suffix such as `@s43` are additional training seeds.

## Main text

| Paper item | Content | Files |
|---|---|---|
| Table 1 | Detection under padding, paraphrase, injection and controls | `results/attack_matrix.csv`, `tables/tab32_v5_detection_matrix.*` |
| Table 2 | One-factor read-out ablation (AUC lost to the instruction) | `results/v5_readout_ablation.csv`, `results/auc_effects.csv`, `results/modern_effects.csv` |
| Figure 1 | Adaptive injection with a matched benign control | `results/v52_injection_adaptive.csv`, `figures/fig33_injection_adaptive.*` |
| §4.1, uncapped injection | Success within one and two search steps | `results/v54_injection_uncapped.csv` |
| Table 3 | Defenses by attacker knowledge | `results/v52_readout_seeds.csv`, `results/ood_padding_matrix.csv`, `results/adaptive_by_arm.csv`, `results/v5_score_guided_attacks.csv`, `results/v51_fused_detectors.csv`, `results/scores_legit_realinstr.parquet` |
| §4.2, dilution and operating point | Truncation, padding share, 5% FPR, partial AUC | `results/v5_mechanism.csv`, `results/v5_operating_point.csv`, `results/v54_partial_auc.csv` |
| §4.2, phishing pool | Padding and injection on 500 phishing emails | `results/v52_phish_pool.csv`, `results/v54_lrword_contributions.csv`, `results/v54_lrword_top_terms_phishing.csv` |
| §4.2, recent human-written mail | AUC by malicious type | `results/v54_modern_human_by_type.csv`, `results/modern_results.csv` |
| §4.3, adaptive attacks | Random-candidate, label-only, score-guided, GCG; phishing-pool attacks and paired tests | `results/adaptive_by_arm.csv`, `results/v5_score_guided_attacks.csv`, `results/v52_phish_attacks.csv`, `results/v54_augmix_vs_lrword_mcnemar.csv` |
| §4.3, cost on legitimate mail | FPR on instruction-bearing mail, IT notices, training samples | `results/v53_realinstr_clean.csv`, `results/v52_realinstr_by_kind.csv`, `results/v54_it_like_by_seed.csv` |
| §4.4, fusion | Co-occurring misses, transfer, fused-score attacks | `results/complementarity.csv`, `results/v51_fused_detectors.csv`, `results/v5_score_guided_attacks.csv` |

## Appendix

| Paper item | Files |
|---|---|
| Table 4 (corpus) | `tables/tab01_dataset.*` |
| Table 5 (full matrix with intervals) | `results/attack_matrix.csv` |
| Table 6 (threshold-free effects) | `results/auc_effects.csv` |
| Table 7 (phishing pool) | `results/v52_phish_pool.csv`, `tables/tab43_phish_pool.*` |
| Table 8 (read-out seeds) | `results/v52_readout_seeds.csv`, `results/v52_readout_seed_tests.csv`, `tables/tab41_readout_seeds.*` |
| Table 9 (leave-one-source-out) | `results/loso.csv`, `results/loso_scores.parquet` |
| Table 10 (PhishFuzzer, LLM-rephrased) | `results/modern_results.csv` |
| Tables 11 and 12 (validity audit, artifact-free subsets) | `audit/*_ratings.csv`, `results/v53_paraphrase_clean.csv`, `results/v53_realinstr_clean.csv` |
| Table 13 (recent human-written mail) | `results/v54_modern_human_by_type.csv` |
| Table 14 (uncapped injection) | `results/v54_injection_uncapped.csv` |
| Table 15 (partial AUC) | `results/v54_partial_auc.csv` |
| Figure 2 (probe layers) | `results/probe_layer_sweep.csv`, `figures/fig17_probe_layers.*` |
| Figure 3 (mechanism) | `results/v5_mechanism.csv`, `figures/fig31_mechanism.*` |
| Figure 4 (read-out ablation) | `figures/fig27_readout_ablation.*` |
| Figure 5 (operating point) | `results/v5_operating_point.csv`, `figures/fig29_operating_point.*` |
| Figure 6 (collateral FPR) | `results/v5_realinstr_collateral.csv`, `figures/fig32_realinstr_collateral.*` |
| Figure 7 (adaptive attacks) | `results/v5_score_guided_attacks.csv`, `figures/fig35_adaptive_attacks.*` |
| Figure 8 (read-out seeds) | `results/v52_readout_seeds_long.csv`, `figures/fig34_readout_seeds.*` |

`results/summary.json` collects the headline values of every section, and `results/key_findings.md` lists them in prose, in the order the notebook computes them. `results/figure_captions.json` holds the notebook's caption for every figure, including those not used in the paper.
