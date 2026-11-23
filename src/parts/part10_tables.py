# %% [markdown]
# ## 10. Read-out by instruction interaction test
# The central hypothesis is an *interaction*: the instruction harms the prompted read-out more than a trained read-out of the same backbone. For each pair of read-outs
# $(r_1, r_2)$ on the same backbone we bootstrap, over the same paired emails, the difference-in-differences
# $\big[\text{DR}_{r_1}(\text{neutral}) - \text{DR}_{r_1}(\text{inject})\big] - \big[\text{DR}_{r_2}(\text{neutral}) - \text{DR}_{r_2}(\text{inject})\big]$.
# A CI that excludes zero means the read-out changes how much the instruction itself hurts, which no appended-text or clean-accuracy difference can explain.

# %%
def did_rows(alpha, a_ctrl, a_att, label):
    A, B = paired(a_ctrl, a_att); A, B = A.reset_index(), B.reset_index(); rows = []
    for t in BB_TAGS:
        ros = [r for r in READOUTS if f"{r}-{t}" in AVAIL]
        for i, r1 in enumerate(ros):
            for r2 in ros[i + 1:]:
                d1, d2 = f"{r1}-{t}", f"{r2}-{t}"
                e1 = flag(A, d1, alpha).astype(float) - flag(B, d1, alpha).astype(float)
                e2 = flag(A, d2, alpha).astype(float) - flag(B, d2, alpha).astype(float)
                m, lo, hi = boot_diff(e1, e2)
                rows.append(dict(effect=label, backbone=t, readout_1=r1, readout_2=r2, n=len(A), delta_1=e1.mean(), delta_2=e2.mean(),
                                 did=m, did_lo=lo, did_hi=hi, excludes_zero=bool(lo > 0 or hi < 0)))
    return rows

DID = pd.DataFrame(did_rows(A0, "neutral", "inject", "instruction") + did_rows(A0, "neutral", "inject_aware", "instruction (aware)")
                   + did_rows(A0, "ham_pad", "goodword", "word salad vs natural pad"))
DID.to_csv(DIRS["results"] / "readout_interaction.csv", index=False)
display(DID.round(4))

# %% [markdown]
# ## 11. LaTeX tables (booktabs)
# Each table is printed for copy-paste into Overleaf and saved as `.tex` and `.csv` in `cbs/tables/`.

# %%
fmt_ci = lambda m, lo, hi, p=3: "n/a" if pd.isna(m) else (f"{m:.{p}f}" if pd.isna(lo) else f"{m:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]")
pct = lambda v: "n/a" if pd.isna(v) else f"{100*v:.1f}"
pv = lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"

t = CLEAN[A0]
tab2 = pd.DataFrame({"Detector": t.detector, "Family": t.family,
                     "ROC-AUC [95% CI]": [fmt_ci(a, b, c, 4) for a, b, c in zip(t.auc, t.auc_lo, t.auc_hi)],
                     "PR-AUC": t.ap.map(lambda v: f"{v:.4f}"),
                     f"TPR@{A0:g} [95% CI]": [fmt_ci(a, b, c) for a, b, c in zip(t.tpr, t.tpr_lo, t.tpr_hi)],
                     "Test FPR [95% CI]": [fmt_ci(a, b, c) for a, b, c in zip(t.fpr, t.fpr_lo, t.fpr_hi)],
                     "F1": t.f1.map(lambda v: f"{v:.3f}"), "ECE": t.ece.map(lambda v: f"{v:.3f}"), "Brier": t.brier.map(lambda v: f"{v:.4f}")})
print(write_latex(tab2, "tab02_clean", f"Clean test performance (source-stratified test subset, n={len(tc)}). Thresholds fixed on validation at FPR={A0:g}.",
                  "tab:clean", col_format="llccccccc", note="ECE uses equal-count bins. Stack-noInj is an ablation of Hybrid-Stack."))

def matrix_table(M, value="dr"):
    rows = []
    for d in EVAL_DETS:
        r = {"Detector": d}
        for att in (ATTACKS if value == "dr" else ATTACKS[1:]):
            x = M[(M.detector == d) & (M.attack == att)].iloc[0]
            r[ATT_LABEL[att]] = "n/a" if pd.isna(x[value]) else f"{100*x[value]:.1f} ({100*x[value+'_lo']:.1f}-{100*x[value+'_hi']:.1f})"
        rows.append(r)
    return pd.DataFrame(rows)

print(write_latex(matrix_table(M0, "dr"), "tab03_detection_matrix", f"Detection rate (\\%) under each manipulation at validation FPR={A0:g}, with 95\\% bootstrap CI.",
                  "tab:dr", note="Clean uses the same malicious emails that are manipulated. Neutral suffix is a control; Natural pad is the control for Good-word and an attack in its own right."))
print(write_latex(matrix_table(M0, "asr"), "tab04_asr", f"Conditional attack success rate (\\%) at FPR={A0:g} (95\\% CI).", "tab:asr"))
for a2 in CFG["target_fprs"]:
    if a2 == A0: continue
    print(write_latex(matrix_table(MAT[a2], "dr"), f"tab03b_detection_matrix_fpr{a2:g}".replace(".", "p"),
                      f"Detection rate (\\%) at the secondary operating point FPR={a2:g}.", f"tab:dr{a2:g}".replace(".", "p")))

# Central table: effects net of controls
EFF_ORDER = ["instruction", "instruction (aware)", "neutral suffix", "word salad vs natural pad", "natural padding"]
EFF_NAME = {"instruction": "Instruction (generic)", "instruction (aware)": "Instruction (label-aware)", "neutral suffix": "Neutral suffix vs clean",
            "word salad vs natural pad": "Word salad vs natural pad", "natural padding": "Natural pad vs clean"}
_e = EFF[EFF.effect.isin(EFF_ORDER) & EFF.detector.isin(ALL_DETS)]
tab5 = _e.pivot(index="detector", columns="effect", values="delta").loc[ALL_DETS, EFF_ORDER]
tab5_txt = tab5.copy().astype(object)
for d in ALL_DETS:
    for e in EFF_ORDER:
        r = _e[(_e.detector == d) & (_e.effect == e)].iloc[0]
        tab5_txt.loc[d, e] = f"{100*r.delta:+.1f} [{100*r.delta_lo:+.1f}, {100*r.delta_hi:+.1f}]{stars(r.p_holm)}"
tab5_txt = tab5_txt.rename(columns=EFF_NAME).reset_index().rename(columns={"detector": "Detector"})
print(write_latex(tab5_txt, "tab05_readout_effects", "Paired effects in detection-rate points (control minus manipulation; positive = the manipulation lowers detection), 95\\% CI.",
                  "tab:effects", note="Stars: exact McNemar, Holm-corrected within each column (* p<0.05, ** p<0.01, *** p<0.001)."))

tdid = DID.assign(Effect=DID.effect.map(EFF_NAME), Backbone=DID.backbone, Pair=DID.readout_1 + " - " + DID.readout_2,
                  **{"Effect r1": (100 * DID.delta_1).map(lambda v: f"{v:+.1f}"), "Effect r2": (100 * DID.delta_2).map(lambda v: f"{v:+.1f}"),
                     "Difference [95% CI]": [f"{100*a:+.1f} [{100*b:+.1f}, {100*c:+.1f}]" for a, b, c in zip(DID.did, DID.did_lo, DID.did_hi)],
                     "n": DID.n})[["Effect", "Backbone", "Pair", "Effect r1", "Effect r2", "Difference [95% CI]", "n"]]
print(write_latex(tdid, "tab06_readout_interaction", "Read-out by manipulation interaction (difference-in-differences, percentage points) on the same paired emails.",
                  "tab:did", col_format="lllcccc"))

mc = MCN.copy()
tab7 = pd.DataFrame({"Manipulation": mc.attack.map(ATT_LABEL), "Comparison": mc.A + " vs " + mc.B, "DR A": mc.dr_A.map(pct), "DR B": mc.dr_B.map(pct),
                     "A only": mc.A_only, "B only": mc.B_only, "p": mc.p.map(pv), "p (Holm)": mc.p_holm.map(pv)})
print(write_latex(tab7, "tab07_mcnemar", "Paired exact McNemar tests on detect/miss outcomes (Holm-adjusted over all rows).", "tab:mcnemar", col_format="llcccccc"))

cq = COMP_ALL.copy(); cq["pair"] = cq.a + " / " + cq.b
cq["txt"] = [("n/a" if np.isnan(q) else (f"{q:+.2f}" if np.isnan(lo) else f"{q:+.2f} [{lo:+.2f}, {hi:+.2f}]")) for q, lo, hi in zip(cq.yules_q, cq.q_lo, cq.q_hi)]
tab8 = cq.pivot(index="pair", columns="attack", values="txt")[ATTACKS].rename(columns=ATT_LABEL).reset_index().rename(columns={"pair": "Classical / SLM"})
print(write_latex(tab8, "tab08_yules_q", "Yule's Q between per-email correctness of classical and primary-backbone SLM detectors (95\\% CI; lower = more complementary).", "tab:q"))

tab9 = TRI.copy()
for c in ["block", "review", "allow"]: tab9[c] = tab9[c].map(pct)
tab9.columns = ["Emails", "Auto-block (%)", "Review (%)", "Auto-allow (%)"]
print(write_latex(tab9, "tab09_triage", f"Disagreement-gated triage with {C_STAR} and {S_STAR}.", "tab:triage", col_format="lccc"))

ks = sorted(set([1, 5, max(1, N_CAND // 2), N_CAND]))
tab10 = pd.DataFrame([{"Defense": d, "Emails": c["n"], **{f"ASR@{k}": fmt_ci(c['mean'][k-1], c['lo'][k-1], c['hi'][k-1], 3) for k in ks if k <= len(c['mean'])}}
                      for d, c in ADAPT_CURVES.items()])
print(write_latex(tab10, "tab10_adaptive", f"Adaptive attacker: probability of evasion within k of {N_CAND} candidate variants (95\\% CI).", "tab:adaptive"))
combo = (ADAPT_COMP[ADAPT_COMP.defense.isin(["Hybrid-OR", "Hybrid-Stack", "Stack-Aug"] + ADAPT_DEFS_EXTRA)].sort_values("success_rate", ascending=False)
         .groupby("defense").head(4).assign(success_rate=lambda x: x.success_rate.map(pct)))
combo = combo[["defense", "arm", "base", "family", "pad", "success_rate"]]
combo.columns = ["Defense", "Arm", "Base text", "Injection family", "Padding", "Success (%)"]
print(write_latex(combo, "tab10b_adaptive_combos", "Most successful candidate types against each fused defense.", "tab:combos"))

q = PARA.copy(); qv = q[q.valid] if q.valid.any() else q
tab11 = pd.DataFrame([{"Rewrites": len(q), "Valid (%)": pct(q.valid.mean()), "Refusals (%)": pct(q.refusal.mean()),
                       "URLs kept before repair (%)": pct(q.url_kept.sum() / max(1, q.url_total.sum())),
                       "Length ratio": f"{qv.length_ratio.median():.2f}", "Lexical sim.": f"{qv.lex_sim.median():.2f}",
                       "Semantic sim.": "n/a" if qv.sem_sim.isna().all() else f"{qv.sem_sim.median():.2f}"}])
print(write_latex(tab11, "tab11_attack_quality", f"Paraphrase attack quality (attacker: {CFG['attacker_model']}); medians over valid rewrites.", "tab:quality",
                  note="Funnel: " + "; ".join(f"{k}: {v}" for k, v in PARA_FUNNEL.items())))

_lt = LOSO.copy()
_lt["txt"] = [("n/r" if pd.isna(v) else f"{v:.3f}") + (" (TPR)" if s else "") for v, s in zip(_lt.score, _lt.single_class)]
tab12 = _lt.pivot(index="source", columns="detector", values="txt").reindex(columns=LOSO_DETS).fillna("n/r").reset_index()
tab12.columns.name = None; tab12 = tab12.rename(columns={"source": "Held-out source"})
print(write_latex(tab12, "tab12_loso", "Leave-one-source-out balanced accuracy at a validation-fixed FPR (thresholds from the other sources).", "tab:loso",
                  note=f"n/r: LoRA retrained only without {', '.join(CFG['lora_loso_sources'])}. ZS needs no training; Probe is refit per held-out source."))

if len(XSRC_MAT):
    _x = XSRC_MAT.copy(); _x["txt"] = _x.dr.map(pct)
    tab13 = _x.pivot(index="detector", columns="attack", values="txt").reindex(columns=ATTACKS).rename(columns=ATT_LABEL)
    tab13.insert(0, f"FPR on {XSRC} (%)", _x.groupby("detector").fpr_test.first().map(pct))
    tab13 = tab13.loc[[d for d in list(dict.fromkeys(LOSO_DETS + ["Hybrid-OR"])) if d in tab13.index]].reset_index().rename(columns={"detector": "Detector"})
    print(write_latex(tab13, "tab13_heldout_source_attacks", f"Detection rate (\\%) on the held-out source {XSRC} under every manipulation; no detector saw {XSRC} in training.",
                      "tab:xsrc"))

if N_SEEDS:
    _s = SEEDS.groupby("attack").value.agg(["mean", "std", "min", "max", "count"])
    _order = ["AUC (clean test)"] + ATTACKS + [f"ASR {a}" for a in ATTACKS[1:]]
    _s = _s.reindex([o for o in _order if o in _s.index])
    tab14 = pd.DataFrame({"Quantity": [o if o.startswith(("AUC", "ASR")) else f"DR {ATT_LABEL[o]}" for o in _s.index],
                          "Mean": _s["mean"].map(lambda v: f"{v:.4f}"), "SD": _s["std"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.4f}"),
                          "Min": _s["min"].map(lambda v: f"{v:.4f}"), "Max": _s["max"].map(lambda v: f"{v:.4f}"), "Seeds": _s["count"].astype(int)})
    print(write_latex(tab14, "tab14_seeds", f"{LORA_P} across training seeds (sample, initialization and order redrawn).", "tab:seeds", col_format="lccccc"))

tab15 = pd.DataFrame([{"Detector": d, "ms / email": f"{lat_ms[d]:.3f}" if lat_ms[d] < 1 else f"{lat_ms[d]:.1f}",
                       "Hardware": "CPU" if FAMILY[d] == "Classical" else ("GPU" if N_GPU else "CPU"),
                       "Clean DR": f"{piv.loc[d, 'clean']:.3f}", "Worst-case DR": f"{worst[d]:.3f}", "Mean DR (attacks)": f"{mean_[d]:.3f}"} for d in ALL_DETS])
print(write_latex(tab15, "tab15_cost", "Inference cost and robustness (adversarial manipulations only; controls excluded).", "tab:cost"))

print(write_latex(TPL_TABLE, "tab16_templates", "Injection templates: families, positions and lengths of the training (defender) wording, unseen test wording and the matched neutral control.",
                  "tab:templates"))
_hp = pd.DataFrame([{"Model": n, "Hyper-parameter": CLASSICAL_SPECS[n]["hp"], "Grid": ", ".join(map(str, CLASSICAL_SPECS[n]["grid"])),
                     "Selected": CLASSICAL_HP[n], "Full-test ROC-AUC": f"{FULL_TEST_AUC[n]:.4f}"} for n in CLS_DETS])
_hp = pd.concat([_hp, pd.DataFrame([{"Model": f"Probe-{t}", "Hyper-parameter": "layer", "Grid": f"{len(ZS_META[t]['layers'])} layers",
                                     "Selected": f"{BEST_LAYER[t]} of {ZS_META[t]['n_layers']}", "Full-test ROC-AUC": "n/a"} for t in BEST_LAYER])], ignore_index=True)
print(write_latex(_hp, "tab17_hyperparameters", "Model selection on validation ROC-AUC (classical grids and probe layer).", "tab:hp"))
_sc = pd.DataFrame(STACK_COEF).T.reset_index().rename(columns={"index": "Stacker"}).round(3).fillna("")
print(write_latex(_sc, "tab18_stacker_coefficients", "Stacker coefficients on standardized logit features.", "tab:stack"))

# v3.2 tables
_ae = AUC_EFF[AUC_EFF.detector.isin(ALL_DETS)].copy()
_ae["txt"] = [f"{a:+.4f} [{b:+.4f}, {c:+.4f}]" for a, b, c in zip(_ae.d_auc, _ae.d_auc_lo, _ae.d_auc_hi)]
tab19 = _ae.pivot(index="detector", columns="effect", values="txt").reindex([d for d in ALL_DETS if d in set(_ae.detector)])
tab19 = tab19[[c for c in ["instruction", "instruction (aware)", "natural padding", "word salad vs natural pad"] if c in tab19]].rename(columns=EFF_NAME)
tab19.insert(0, "Clean AUC (pool)", _ae[_ae.effect == "natural padding"].set_index("detector").auc_control.reindex(tab19.index).map(lambda v: f"{v:.4f}"))
print(write_latex(tab19.reset_index().rename(columns={"detector": "Detector"}), "tab19_auc_effects",
                  "Threshold-free effects: change in ROC-AUC against the same clean legitimate test emails (control minus manipulation, 95\\% CI).", "tab:auc"))
_ad = AUC_DID.assign(Effect=AUC_DID.effect.map(EFF_NAME), Pair=AUC_DID.readout_1 + " - " + AUC_DID.readout_2,
                     **{"Difference [95% CI]": [f"{a:+.4f} [{b:+.4f}, {c:+.4f}]" for a, b, c in zip(AUC_DID.did, AUC_DID.did_lo, AUC_DID.did_hi)]})
print(write_latex(_ad[["Effect", "backbone", "Pair", "Difference [95% CI]"]].rename(columns={"backbone": "Backbone"}), "tab20_auc_interaction",
                  "Read-out by instruction interaction on the threshold-free (AUC) scale.", "tab:aucdid", col_format="lllc"))
_pd = PAD_DOSE.copy(); _pd["txt"] = [f"{100*a:.1f} ({100*b:.1f}-{100*c:.1f})" for a, b, c in zip(_pd.dr, _pd.dr_lo, _pd.dr_hi)]
_bins = _pd.groupby("bin").agg(lo=("share_lo", "first"), hi=("share_hi", "first"), n=("n", "first"))
tab21 = _pd.pivot(index="detector", columns="bin", values="txt").loc[PAD_DETS]
tab21.columns = [f"{100*_bins.lo[b]:.0f}-{100*_bins.hi[b]:.0f}% (n={_bins.n[b]})" for b in tab21.columns]
print(write_latex(tab21.reset_index().rename(columns={"detector": "Detector"}), "tab21_natural_pad_dose",
                  "Detection rate (\\%) under natural padding by share of appended benign text (quartiles).", "tab:dose"))
if len(COLL):
    _c = COLL.copy(); _c["txt"] = [f"{100*a:.1f} [{100*b:.1f}, {100*c:.1f}]{stars(q)}" for a, b, c, q in zip(_c.fpr, _c.fpr_lo, _c.fpr_hi, _c.p_holm)]
    tab22 = _c.pivot(index="detector", columns="set", values="txt").loc[[d for d in EVAL_DETS if d in set(_c.detector)]]
    tab22.insert(0, "Clean", _c[_c.set == "legit_neutral"].set_index("detector").fpr_clean.reindex(tab22.index).map(lambda v: f"{100*v:.1f}"))
    tab22 = tab22.rename(columns={"legit_neutral": "+ neutral suffix", "legit_inject": "+ instruction-like suffix"})
    print(write_latex(tab22.reset_index().rename(columns={"detector": "Detector"}), "tab22_collateral_fpr",
                      f"False-positive rate (\\%) on the same {int(_c.n.iloc[0])} legitimate test emails with appended text (95\\% CI).", "tab:collateral",
                      note="Stars: Holm-corrected McNemar against the same emails in clean form."))
_lm = LOSO_MEAN.reset_index().rename(columns={"detector": "Detector", "mean_all_available": "Mean (all sources)", "n_sources": "Sources",
                                              "mean_common": "Balanced acc. (mixed sources)", "min_common": "Min (mixed)",
                                              "tpr_single_class": "TPR (all-malicious sources)"}).round(3)
print(write_latex(_lm, "tab23_loso_means", "LOSO generalization of the trained detectors: balanced accuracy over held-out sources with both classes and detection rate over all-malicious sources.", "tab:losomean"))

# v4.0 tables
if len(OOD_MAT):
    _o = OOD_MAT.copy(); _o["txt"] = [f"{100*a:.1f} ({100*b:.1f}-{100*c:.1f})" for a, b, c in zip(_o.dr, _o.dr_lo, _o.dr_hi)]
    tab24 = _o.pivot(index="detector", columns="pad", values="txt").reindex(index=[d for d in OOD_DETS if d in set(_o.detector)], columns=PAD_COLS)
    tab24 = tab24.rename(columns=PAD_LABEL).fillna("n/r").reset_index().rename(columns={"detector": "Detector"})
    print(write_latex(tab24, "tab24_ood_padding", f"Detection rate (\\%) under in-distribution and out-of-distribution padding at FPR={A0:g} (95\\% CI).", "tab:ood",
                      note="LoRA-Aug trained only on natural text appended at the end; LoRA-AugMix also on prefix and interleaved positions and 40 to 240 words. "
                           "LLM-written and hidden-HTML padding are unseen by both."))
if len(AUG_SEED_TAB):
    _a = AUG_SEED_TAB.groupby("pad").dr.agg(["mean", "std", "min", "max", "count"]).reindex(PAD_COLS).dropna(how="all")
    tab25 = pd.DataFrame({"Padding": [PAD_LABEL[i] for i in _a.index], "Mean": _a["mean"].map(lambda v: f"{100*v:.1f}"),
                          "SD": _a["std"].map(lambda v: "n/a" if pd.isna(v) else f"{100*v:.1f}"), "Min": _a["min"].map(lambda v: f"{100*v:.1f}"),
                          "Max": _a["max"].map(lambda v: f"{100*v:.1f}"), "Seeds": _a["count"].astype(int)})
    print(write_latex(tab25, "tab25_aug_seeds", f"{AUG_DET} detection rate (\\%) across training seeds.", "tab:augseeds", col_format="lccccc"))
if len(MCN_OOD):
    tab26 = pd.DataFrame({"Padding": MCN_OOD["pad"].map(PAD_LABEL), "Comparison": MCN_OOD.A + " vs " + MCN_OOD.B, "DR A": MCN_OOD.dr_A.map(pct),
                          "DR B": MCN_OOD.dr_B.map(pct), "A only": MCN_OOD.A_only, "B only": MCN_OOD.B_only, "p (Holm)": MCN_OOD.p_holm.map(pv)})
    print(write_latex(tab26, "tab26_ood_mcnemar", "Paired McNemar tests between padding-aware and plain detectors on each padding variant.", "tab:oodmcn", col_format="llccccc"))
if len(ADAPT_ARM):
    _r = ADAPT_ARM.copy(); _r["txt"] = [f"{100*a:.1f} [{100*b:.1f}, {100*c:.1f}]" for a, b, c in zip(_r.asr, _r.lo, _r.hi)]
    tab27 = _r.pivot(index="defense", columns="arm", values="txt").reindex([d for d in DEFENSES if d in set(_r.defense)]).reset_index()
    tab27.columns.name = None
    print(write_latex(tab27.rename(columns={"defense": "Defense", "words": "Word salad arm", "natural": "Natural-pad arm",
                                            "out-of-distribution": "OOD-pad arm"}), "tab27_adaptive_arms",
                      "Adaptive attacker: evasion probability (\\%) when every candidate of one arm is tried (95\\% CI).", "tab:arms"))
if len(PROMPT_TAB):
    _p = PROMPT_TAB
    tab28 = pd.DataFrame({"Detector": _p.detector, "Clean AUC [95% CI]": [fmt_ci(a, b, c, 3) for a, b, c in zip(_p.auc, _p.auc_lo, _p.auc_hi)],
                          f"TPR@{A0:g}": _p.tpr.map(lambda v: f"{v:.3f}"), "AUC under aware inj.": _p.auc_inject_aware.map(lambda v: f"{v:.3f}"),
                          "AUC lost (generic)": [fmt_ci(a, b, c, 3) for a, b, c in zip(_p.instr_auc, _p.instr_auc_lo, _p.instr_auc_hi)],
                          "AUC lost (aware)": [fmt_ci(a, b, c, 3) for a, b, c in zip(_p.instr_aware_auc, _p.instr_aware_auc_lo, _p.instr_aware_auc_hi)],
                          "DR natural pad": _p.dr_ham_pad.map(pct), "FPR benign+instr.": _p.fpr_legit_inject.map(pct)})
    print(write_latex(tab28, "tab28_prompted_baselines", "Stronger prompted baselines versus zero-shot read-outs and LoRA (AUC scale is threshold-free).", "tab:prompted"))
if len(MODERN_TAB):
    _m = MODERN_TAB.copy()
    for k_ in ["tpr", "fpr", "tpr_phishing", "tpr_spam"]: _m[k_] = _m[k_].map(pct)
    _m["auc"] = [fmt_ci(a, b, c, 3) for a, b, c in zip(_m.auc, _m.auc_lo, _m.auc_hi)]
    tab29 = _m[["set", "detector", "n", "auc", "tpr", "tpr_phishing", "tpr_spam", "fpr"]].rename(columns={
        "set": "Subset", "detector": "Detector", "n": "n", "auc": "ROC-AUC [95% CI]", "tpr": "DR (%)", "tpr_phishing": "DR phishing (%)",
        "tpr_spam": "DR spam (%)", "fpr": "FPR (%)"})
    tab29["Subset"] = tab29.Subset.map({"modern_human": "Human", "modern_llm": "LLM-rephrased"})
    print(write_latex(tab29, "tab29_modern", "Out-of-time test on PhishFuzzer (decontaminated), thresholds from the original validation split.", "tab:modern",
                      note=f"Removed as near-duplicates of the training corpus: {MODERN_STATS.get('removed_as_near_duplicates', 'n/a')}. "
                           "PhishFuzzer: Toth, Gruschka and Bisztray (2026), CC BY-NC 4.0."))
if len(MODERN_EFF):
    _e2 = MODERN_EFF.copy()
    tab30 = pd.DataFrame({"Detector": _e2.detector, "DR clean": _e2.dr_clean.map(pct), "DR natural pad": _e2.get("dr_pad", pd.Series(np.nan, index=_e2.index)).map(pct),
                          "AUC (neutral)": _e2.auc_neutral.map(lambda v: f"{v:.3f}"),
                          "AUC lost to aware inj. [95% CI]": [fmt_ci(a, b, c, 3) for a, b, c in zip(_e2.instr_aware_auc, _e2.lo, _e2.hi)]})
    print(write_latex(tab30, "tab30_modern_effects", "Read-out and padding results replicated on modern malicious mail (AUC against modern valid mail).", "tab:moderneff"))

# %% [markdown]
# ## 12. Key findings (computed, with guarded wording)
# Each sentence is generated from this run. A claim is only stated as an effect when its 95% CI excludes zero and the Holm-corrected test is significant;
# otherwise the text says the difference was not detected. Use these to draft the abstract, then verify against the figures.

# %%
def drv(d, att): return float(M0[(M0.detector == d) & (M0.attack == att)].dr.iloc[0])
def eff(e, d): return EFF[(EFF.effect == e) & (EFF.detector == d)].iloc[0]
def sigtxt(r): return f"{100*r.delta:+.1f} pp [{100*r.delta_lo:+.1f}, {100*r.delta_hi:+.1f}], p_Holm={pv(r.p_holm)}"
def is_sig(r): return (r.p_holm < 0.05) and (r.delta_lo > 0 or r.delta_hi < 0)

F = []
best_cls = CLEAN[A0][CLEAN[A0].family == "Classical"].sort_values("auc").iloc[-1]
best_slm = CLEAN[A0][CLEAN[A0].family == "Small LLM"].sort_values("auc").iloc[-1]
F.append(f"RQ1 (clean). Best classical {best_cls.detector} ROC-AUC {best_cls.auc:.4f} [{best_cls.auc_lo:.4f}, {best_cls.auc_hi:.4f}]; best SLM {best_slm.detector} "
         f"{best_slm.auc:.4f} [{best_slm.auc_lo:.4f}, {best_slm.auc_hi:.4f}]. Zero-shot: " +
         ", ".join(f"{d} {CLEAN[A0].set_index('detector').auc[d]:.3f}" for d in SLM_DETS if d.startswith("ZS")) + ".")

for t in BB_TAGS:
    ros = [f"{r}-{t}" for r in READOUTS if f"{r}-{t}" in AVAIL]
    if not ros: continue
    parts = []
    for d in ros:
        r = eff("instruction", d); ra = eff("instruction (aware)", d)
        parts.append(f"{d}: generic {sigtxt(r)}{'' if is_sig(r) else ' (not significant)'}; label-aware {sigtxt(ra)}{'' if is_sig(ra) else ' (not significant)'}")
    F.append(f"RQ2 (read-out, {t}). Instruction effect net of the neutral suffix: " + "; ".join(parts) + ".")
    for _, r in DID[(DID.backbone == t) & (DID.effect.str.startswith("instruction"))].iterrows():
        verdict = ("the instruction hurts " + (r.readout_1 if r.did > 0 else r.readout_2) + " more") if r.excludes_zero else "no interaction detected"
        F.append(f"  Interaction {r.readout_1} vs {r.readout_2} ({EFF_NAME[r.effect]}): {100*r.did:+.1f} pp [{100*r.did_lo:+.1f}, {100*r.did_hi:+.1f}], {verdict}.")
zs_sig = [t for t in BB_TAGS if f"ZS-{t}" in AVAIL and (is_sig(eff("instruction", f"ZS-{t}")) or is_sig(eff("instruction (aware)", f"ZS-{t}")))]
tr_sig = [d for t in BB_TAGS for d in (f"Probe-{t}", f"LoRA-{t}") if d in AVAIL and (is_sig(eff("instruction", d)) or is_sig(eff("instruction (aware)", d)))]
if zs_sig and len(zs_sig) == sum(f"ZS-{t}" in AVAIL for t in BB_TAGS) and not tr_sig:
    F.append(f"  => Replicated at {len(zs_sig)} scale(s): only the prompted read-out is hurt by the instruction itself; trained read-outs of the same backbone are not.")
elif zs_sig:
    F.append(f"  => Prompted read-out hurt at {zs_sig}; trained read-outs with a significant instruction effect: {tr_sig or 'none'}. Report the interaction table, not a blanket claim.")
else:
    F.append("  => No significant instruction effect on the prompted read-out at this operating point; the read-out hypothesis is not supported by this run.")

art = [d for d in ALL_DETS if eff("word salad vs natural pad", d).delta_hi < 0 and eff("word salad vs natural pad", d).p_holm < 0.05]
hurt = [d for d in ALL_DETS if is_sig(eff("natural padding", d)) and eff("natural padding", d).delta > 0]
F.append(f"RQ3 (padding). Detectors whose detection is significantly HIGHER under word salad than under natural padding (artifact, not robustness): {art or 'none'}. "
         f"Detectors significantly hurt by natural padding: {hurt or 'none'}. Good-word DR: " + ", ".join(f"{d} {drv(d, 'goodword'):.2f}" for d in [C_STAR, S_STAR, ZS_P]) + ".")
F.append(f"RQ2 (complementarity, {C_STAR} vs {S_STAR}). Yule's Q: " +
         ", ".join(f"{ATT_LABEL[a]} {q:+.2f}" + ("" if np.isnan(lo) else f" [{lo:+.2f}, {hi:+.2f}]") for a, q, lo, hi in zip(COMP.attack, COMP.yules_q, COMP.q_lo, COMP.q_hi) if not np.isnan(q)) + ".")

def mcn_row(att, a, b):
    r = MCN[(MCN.attack == att) & (MCN.A == a) & (MCN.B == b)]; return None if r.empty else r.iloc[0]
for att in ["para_inject", "goodword", "inject_aware"]:
    r = mcn_row(att, "Hybrid-OR", "Hybrid-Stack")
    if r is not None:
        better = "Hybrid-OR" if r.dr_A > r.dr_B else "Hybrid-Stack"
        F.append(f"RQ4 ({ATT_LABEL[att]}). Hybrid-OR DR {r.dr_A:.3f} vs Hybrid-Stack {r.dr_B:.3f} (discordant {r.A_only}/{r.B_only}, p_Holm={pv(r.p_holm)}): "
                 + (f"{better} is better." if r.p_holm < 0.05 else "no significant difference."))
r = mcn_row("goodword", "Stack-Aug", "Hybrid-Stack")
if r is not None:
    F.append(f"RQ4 (defender augmentation). Stack-Aug vs Hybrid-Stack on good-word: {r.dr_A:.3f} vs {r.dr_B:.3f}, p_Holm={pv(r.p_holm)}.")
if ADAPT_CURVES:
    F.append(f"RQ4 (adaptive, k={N_CAND}). " + ", ".join(f"{d} {100*c['mean'][-1]:.1f}% [{100*c['lo'][-1]:.1f}, {100*c['hi'][-1]:.1f}]" for d, c in ADAPT_CURVES.items()) + ".")
r_leg = TRI[TRI.group == "Legitimate (clean)"].iloc[0]
F.append(f"RQ4 (triage). Review burden on legitimate mail {r_leg.review:.1%}; auto-allowed paraphrase+injection {TRI[TRI.group == ATT_LABEL['para_inject']].allow.iloc[0]:.1%}.")

F.append("RQ5 (LOSO, trained detectors; balanced accuracy over held-out sources with both classes, and TPR over all-malicious sources): " +
         ", ".join(f"{d} {LOSO_MEAN.mean_common[d]:.3f} (min {LOSO_MEAN.min_common[d]:.3f}; TPR {LOSO_MEAN.tpr_single_class[d]:.3f})" for d in LOSO_TRAINED) + ".")
if len(XSRC_MAT):
    F.append(f"RQ5 (held-out {XSRC}, trained without it). Injection DR: " +
             ", ".join(f"{d} {XSRC_MAT[(XSRC_MAT.detector == d) & (XSRC_MAT.attack == 'inject')].dr.iloc[0]:.2f}" for d in XSRC_MAT.detector.unique()) + ".")
if N_SEEDS > 1:
    _ss = SEEDS[SEEDS.attack.isin(["clean", "inject", "inject_aware", "goodword"])].groupby("attack").value.agg(["mean", "std"])
    F.append(f"RQ5 (seeds, {LORA_P}, {N_SEEDS} seeds). DR mean (SD): " + ", ".join(f"{ATT_LABEL[a]} {m:.3f} ({s:.3f})" for a, (m, s) in _ss.iterrows()) + ".")
# v3.2 findings
for t in BB_TAGS:
    if f"ZS-{t}" not in AVAIL: continue
    parts = []
    for d in [f"{r}-{t}" for r in READOUTS if f"{r}-{t}" in AVAIL]:
        for e in ["instruction", "instruction (aware)"]:
            r = AUC_EFF[(AUC_EFF.effect == e) & (AUC_EFF.detector == d)].iloc[0]
            parts.append(f"{d} {EFF_NAME[e].split('(')[1][:-1]} {r.d_auc:+.4f} [{r.d_auc_lo:+.4f}, {r.d_auc_hi:+.4f}]")
    F.append(f"RQ2 (threshold-free, {t}). AUC lost to the instruction, net of the neutral suffix: " + "; ".join(parts) + ".")
for t in PL.backbone.unique():
    g = PL[PL.backbone == t]; bl = g[g.layer == BEST_LAYER[t]].iloc[0]
    hurt = g[(g.eff_aware_lo > 0) | (g.eff_lo > 0)]
    F.append(f"RQ2 (probe depth, {t}). At the selected layer {BEST_LAYER[t]} the instruction effect is {100*bl.eff:+.1f} pp (generic) and {100*bl.eff_aware:+.1f} pp (label-aware); "
             f"layers with a significant effect: {hurt.layer.tolist() or 'none'} of {g.layer.tolist()}.")
if AUG_DET in AVAIL:
    r_aug, r_s = drv(AUG_DET, "ham_pad"), drv(S_STAR, "ham_pad")
    m_ = mcnemar(flag(attack_frame("ham_pad"), AUG_DET), flag(attack_frame("ham_pad"), S_STAR))
    F.append(f"RQ4 (padding-aware training). Natural pad DR: {AUG_DET} {r_aug:.3f} vs {S_STAR} {r_s:.3f} (discordant {m_[0]}/{m_[1]}, p={pv(m_[2])}); "
             f"clean DR {drv(AUG_DET, 'clean'):.3f} vs {drv(S_STAR, 'clean'):.3f}; good-word {drv(AUG_DET, 'goodword'):.3f}; injection {drv(AUG_DET, 'inject'):.3f}.")
_q = PAD_DOSE[PAD_DOSE.detector.isin([C_STAR, S_STAR])].pivot(index="detector", columns="bin", values="dr")
F.append("RQ3 (natural padding dose). DR from the lowest to the highest quartile of appended share: " +
         ", ".join(f"{d} {_q.loc[d].iloc[0]:.2f} -> {_q.loc[d].iloc[-1]:.2f}" for d in _q.index) + ".")
if len(COLL):
    _w = COLL[(COLL.set == "legit_inject")].sort_values("fpr", ascending=False).head(4)
    F.append("Collateral FPR on legitimate mail with an instruction-like suffix (highest): " +
             ", ".join(f"{r.detector} {100*r.fpr:.1f}% (clean {100*r.fpr_clean:.1f}%)" for r in _w.itertuples()) + ".")
# v4.0 findings
_zs_all = [f"ZS-{t}" for t in BB_TAGS if f"ZS-{t}" in AVAIL]
_zs_hit = [d for d in _zs_all if AUC_EFF[(AUC_EFF.effect == "instruction (aware)") & (AUC_EFF.detector == d)].d_auc_lo.iloc[0] > 0]
_lo_hit = [d for d in [f"LoRA-{t}" for t in BB_TAGS] if d in AVAIL and AUC_EFF[(AUC_EFF.effect == "instruction (aware)") & (AUC_EFF.detector == d)].d_auc_lo.iloc[0] > 0.01]
F.append(f"C1 (replication across {len(BB_TAGS)} backbones, {len(set(FAMILY_BB.values()))} families {sorted(set(FAMILY_BB.values()))}). Prompted read-outs losing AUC "
         f"to the label-aware instruction (CI above 0): {_zs_hit} of {_zs_all}; LoRA read-outs losing more than 0.01 AUC: {_lo_hit or 'none'}.")
if len(PROMPT_TAB):
    F.append("C1 (stronger prompts and models). AUC under label-aware injection (clean AUC): " +
             ", ".join(f"{r.detector} {r.auc_inject_aware:.3f} ({r.auc:.3f})" for r in PROMPT_TAB.itertuples()) + ".")
if len(OOD_MAT):
    _ov = OOD_MAT.pivot(index="detector", columns="pad", values="dr")
    for d in [LORA_P, AUG_DET, AUGMIX_DET, "Hybrid-OR"]:
        if d in _ov.index:
            F.append(f"C4/C5 (padding, {d}). DR: " + ", ".join(f"{PAD_LABEL[c]} {_ov.loc[d, c]:.3f}" for c in PAD_COLS if c in _ov.columns and not np.isnan(_ov.loc[d, c])) + ".")
if len(AUG_SEED_TAB) and AUG_SEED_TAB.seed.nunique() > 1:
    _g = AUG_SEED_TAB.groupby("pad").dr.agg(["mean", "std"])
    F.append(f"C5 (seeds, {AUG_DET}, {AUG_SEED_TAB.seed.nunique()} seeds). Mean (SD): " + ", ".join(f"{PAD_LABEL[i]} {r['mean']:.3f} ({r['std']:.3f})" for i, r in _g.iterrows()) + ".")
if len(ADAPT_ARM):
    _aa = ADAPT_ARM.pivot(index="defense", columns="arm", values="asr")
    F.append("C4/C5 (adaptive, per arm, ASR when the whole arm is tried): " + "; ".join(f"{d}: " + ", ".join(f"{a} {100*_aa.loc[d, a]:.1f}%" for a in _aa.columns if not np.isnan(_aa.loc[d, a])) for d in _aa.index) + ".")
if len(COLL):
    _pl = COLL[COLL.set.isin(PAD_LEGIT)].pivot(index="detector", columns="set", values="fpr")
    _w = [d for d in [LORA_P, AUG_DET, AUGMIX_DET, "Hybrid-Stack", "Stack-Aug"] if d in _pl.index]
    F.append("C5 (cost on benign mail with padding, FPR): " + ", ".join(f"{d} " + "/".join(f"{100*_pl.loc[d, c]:.1f}%" for c in _pl.columns) for d in _w) + f" ({'/'.join(_pl.columns)}).")
if len(MODERN_TAB):
    for m in ["modern_human", "modern_llm"]:
        T = MODERN_TAB[MODERN_TAB.set == m].set_index("detector")
        if len(T):
            F.append(f"Modern ({m}). AUC / DR / FPR: " + ", ".join(f"{d} {T.loc[d, 'auc']:.3f}/{T.loc[d, 'tpr']:.2f}/{T.loc[d, 'fpr']:.3f}" for d in T.index
                                                                if d in [C_STAR, ZS_P, PROBE_P, LORA_P, AUG_DET, AUGMIX_DET, "Hybrid-OR"]) + ".")
if len(MODERN_EFF):
    F.append("Modern (read-out replication, AUC lost to the label-aware instruction): " +
             ", ".join(f"{r.detector} {r.instr_aware_auc:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" for r in MODERN_EFF.itertuples() if r.detector in SLM_DETS + PROMPT_AVAIL) + ".")
if AUDIT:
    F.append("Human validity checks: " + json.dumps(AUDIT))
else:
    F.append("Human validity checks: pending (fill cbs/private/*_audit.csv, save as *_audit_filled.csv and attach).")
if FAILED_OPT:
    F.append(f"NOTE: optional jobs failed and are excluded: {FAILED_OPT} (see cbs/logs/).")
if DEFERRED:
    F.append(f"NOTE: deferred jobs {DEFERRED}; rerun with this output attached to complete the optional analyses.")
print("KEY FINDINGS (auto-generated; verify against figures and CIs)\n" + "\n".join(F))
(DIRS["results"] / "key_findings.md").write_text("\n".join(F))

