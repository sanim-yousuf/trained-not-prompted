# %% [markdown]
# ## 15. Run report, summary and packaging

# %%
summary = {
    "code_version": CODE_VERSION, "cache_tag": CACHE_TAG, "smoke": SMOKE, "corpus": DATA_STATS, "n_test_eval": len(tc), "n_attack_pool": len(POOL),
    "n_val_legit": int(len(VAL_LEG)), "primary_fpr": A0, "target_fprs": CFG["target_fprs"], "C_star": C_STAR, "S_star": S_STAR,
    "probe_layer": BEST_LAYER, "available_slm": SLM_DETS, "deferred_jobs": DEFERRED, "lora_seeds_done": int(N_SEEDS),
    "clean_auc": dict(zip(CLEAN[A0].detector, CLEAN[A0].auc.round(5))),
    "full_test_auc_classical": {k: round(v, 5) for k, v in FULL_TEST_AUC.items()},
    "dr_matrix": M0.pivot(index="detector", columns="attack", values="dr").round(4).to_dict(),
    "effects": EFF[["effect", "detector", "delta", "delta_lo", "delta_hi", "p_holm"]].round(5).to_dict(orient="records"),
    "readout_interaction": DID.round(5).to_dict(orient="records"),
    "yules_q_Cstar_Sstar": dict(zip(COMP.attack, COMP.yules_q.round(3))),
    "adaptive_asr_at_N": {d: round(float(c["mean"][-1]), 4) for d, c in ADAPT_CURVES.items()},
    "triage": TRI.round(4).to_dict(orient="records"),
    "loso": LOSO.round(4).to_dict(orient="records"), "loso_common_sources": LOSO_COMMON,
    "loso_mean_common": LOSO_MEAN.mean_common.round(4).to_dict(),
    "auc_effects": AUC_EFF.round(5).to_dict(orient="records"), "auc_readout_interaction": AUC_DID.round(5).to_dict(orient="records"),
    "natural_pad_dose": PAD_DOSE.round(4).to_dict(orient="records"),
    "collateral_fpr": COLL.round(4).to_dict(orient="records") if len(COLL) else None,
    "probe_layer_sweep": PL.round(4).to_dict(orient="records"), "notebook_version": NOTEBOOK_VERSION,
    "backbones": {t: FAMILY_BB[t] for t in BB_TAGS}, "prompted_baselines": PROMPT_TAB.round(5).to_dict(orient="records"),
    "ood_padding": OOD_MAT.round(4).to_dict(orient="records"), "aug_seeds": AUG_SEED_TAB.round(4).to_dict(orient="records"),
    "adaptive_by_arm": ADAPT_ARM.round(4).to_dict(orient="records"), "modern": MODERN_TAB.round(4).to_dict(orient="records"),
    "modern_effects": MODERN_EFF.round(4).to_dict(orient="records"), "modern_stats": MODERN_STATS, "audit": AUDIT, "failed_optional": FAILED_OPT,
    "latency_ms": {k: round(v, 4) for k, v in lat_ms.items()},
    "paraphrase_funnel": PARA_FUNNEL, "stacker_coefficients": STACK_COEF, "classical_hp": CLASSICAL_HP,
    "timings_sec": {k: round(v, 1) for k, v in TIMINGS.items()},
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": S.peft.__version__,
                 "sklearn": __import__("sklearn").__version__, "xgboost": __import__("xgboost").__version__, "python": platform.python_version()},
    "models": {"backbones": CFG["backbones"], "attacker": CFG["attacker_model"], "semsim": CFG["semsim_model"]},
    "v5": V5_SUMMARY,
    "v52": V52_SUMMARY,
    "v53": V53_SUMMARY,
    "v54": V54_SUMMARY,
}
(DIRS["results"] / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
EFF.to_csv(DIRS["results"] / "readout_effects.csv", index=False); M0.to_csv(DIRS["results"] / "attack_matrix.csv", index=False)
COMP_ALL.to_csv(DIRS["results"] / "complementarity.csv", index=False); MCN.to_csv(DIRS["results"] / "mcnemar.csv", index=False)
LOSO.to_csv(DIRS["results"] / "loso.csv", index=False)
if len(XSRC_MAT): XSRC_MAT.to_csv(DIRS["results"] / "heldout_source_matrix.csv", index=False)
if len(SEEDS): SEEDS.to_csv(DIRS["results"] / "seeds.csv", index=False)

print("Stage timings (minutes):")
for k, v in TIMINGS.items(): print(f"  {k:<32} {v/60:7.2f}")
print(f"Total wall time: {elapsed_h():.2f} h")
print("\nFigures:")
for n, c in FIG_INDEX: print(f"  {n}: {c}")
(DIRS["results"] / "figure_captions.json").write_text(json.dumps(dict(FIG_INDEX), indent=2))
# Paper assets only: figures, tables, results. The cache (models, generated attack text) and private/ (audit sheet) are NOT included.
with zipfile.ZipFile(WORK.parent / "cbs_paper_assets.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for sub in ["figures", "tables", "results"]:
        for p in (WORK / sub).rglob("*"):
            if p.is_file(): zf.write(p, p.relative_to(WORK))
print(f"\nPaper assets zipped to {WORK.parent / 'cbs_paper_assets.zip'}. To resume or complete deferred jobs: attach this version's output and run again.")
