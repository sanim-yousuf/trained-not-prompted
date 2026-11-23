# %% [markdown]
# ## 14c. Analyses requested in the second expert review (v5.4, CPU only)
# Every analysis below reuses cached scores and attack logs; no model is trained or rerun.
# 1. Paired McNemar tests for LoRA-AugMix against LR-word under the adaptive padding attacks (phishing pool and main pool).
# 2. Why LR-word resists padding on phishing: logit contributions of its terms before and after natural padding, phishing vs spam-heavy mail.
# 3. Uncapped measures for the adaptive injection attack: success within small query budgets and paired queries to first success.
# 4. Recent human-written PhishFuzzer mail: AUC by malicious type (phishing vs legitimate, spam vs legitimate) with intervals.
# 5. IT-like legitimate emails in each seed's training sample, against each seed's false-positive rate on IT notices.
# 6. Partial AUC (FPR at most 1%, McClish-standardized) for every detector.
# 7. A more legible version of the adaptive-attack figure (fig35).

# %%
from sklearn.metrics import roc_auc_score
V54_SUMMARY = {}
def _wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

# ---- 1. paired McNemar: LoRA-AugMix vs LR-word on emails both detectors catch ----
_rows = []
for pool_, key_score, key_label in (("phishing pool", "phgreedy", "phlabel"), ("main pool", "greedy", "label")):
    for mode_, key_ in (("score-guided", key_score), ("label-only", key_label)):
        fa, fb = ATTACK_FILES.get(key_, {}).get(AUGMIX_DET), ATTACK_FILES.get(key_, {}).get(C_STAR)
        if not (fa and fb and Path(fa).exists() and Path(fb).exists()): continue
        A_, B_ = pd.read_parquet(fa).set_index("id"), pd.read_parquet(fb).set_index("id")
        ids = A_.index[A_.caught.astype(bool)].intersection(B_.index[B_.caught.astype(bool)])
        if not len(ids): continue
        sa, sb = A_.loc[ids].success.astype(bool).values, B_.loc[ids].success.astype(bool).values
        n10, n01, p_ = mcnemar(sa, sb)
        _rows.append(dict(pool=pool_, attack=mode_, n_both_caught=len(ids), asr_augmix=sa.mean(), asr_augmix_ci=_wilson(sa.sum(), len(ids)),
                          asr_lrword=sb.mean(), asr_lrword_ci=_wilson(sb.sum(), len(ids)), only_augmix=n10, only_lrword=n01, p_mcnemar=p_))
MCN54 = pd.DataFrame(_rows)
if len(MCN54): MCN54["p_holm"] = holm(MCN54.p_mcnemar.values)
MCN54.to_csv(DIRS["results"] / "v54_augmix_vs_lrword_mcnemar.csv", index=False)
display(MCN54.round(4)) if len(MCN54) else print("Attack logs for the paired test not found.")

# %% [markdown]
# ### 14c.2 LR-word on phishing vs spam: term contributions before and after natural padding
# For a linear model on L2-normalized TF-IDF, appending text lowers every original term's weight (the vector is renormalized) and adds the padding
# terms' own (mostly negative) contributions. If phishing emails carry many strongly malicious terms, their logit stays above the threshold after this
# dilution. We report, per subset: the clean and padded logit, the drop, the share of the clean positive logit mass carried by the top-10 terms, and the
# number of terms with a positive contribution.

# %%
_LRD = "LR-word" if "LR-word" in CLASSICAL else C_STAR   # the analysis is about the TF-IDF logistic regression
_pipe = CLASSICAL.get(_LRD)
LR54 = pd.DataFrame(); LR54_TERMS = pd.DataFrame()
if _pipe is not None and hasattr(_pipe.named_steps["clf"], "coef_"):
    _vec, _clf = _pipe.named_steps["vec"], _pipe.named_steps["clf"]; _w = _clf.coef_.ravel(); _b = float(_clf.intercept_[0])
    _thr_logit = float(np.log(THR[A0][_LRD] / (1 - THR[A0][_LRD]))) if 0 < THR[A0][_LRD] < 1 else np.nan
    def _contrib(texts):
        X = _vec.transform(list(texts)).tocsr(); out = []
        for i in range(X.shape[0]):
            row = X.getrow(i); c = row.data * _w[row.indices]; pos = np.sort(c[c > 0])[::-1]
            out.append(dict(logit=float(c.sum() + _b), n_pos=int((c > 0).sum()), top10_share=float(pos[:10].sum() / pos.sum()) if pos.sum() > 0 else np.nan,
                            pos_mass=float(pos.sum())))
        return pd.DataFrame(out)
    _base_text = SC["test_clean"].set_index("id").text
    _subsets = []
    if {"phish_clean", "phish_ham_pad"} <= set(SC):
        A_, B_ = paired("phish_clean", "phish_ham_pad"); _subsets.append(("phishing pool", A_.text.values, B_.text.values))
    if "ham_pad" in SC:
        hp = attack_frame("ham_pad"); hp = hp[hp.base_id.isin(_base_text.index)]
        ph_ = hp.source.isin(PHISH_SRC).values
        for nm_, m_ in (("main pool, phishing/fraud", ph_), ("main pool, spam-heavy", ~ph_)):
            _subsets.append((nm_, _base_text.loc[hp.base_id[m_]].values, hp.text[m_].values))
    _rows = []
    for nm_, tc, tp in _subsets:
        c_, p_ = _contrib(tc), _contrib(tp)
        _rows.append(dict(subset=nm_, n=len(c_), logit_clean=c_.logit.median(), logit_padded=p_.logit.median(), logit_drop=(c_.logit - p_.logit).median(),
                          top10_share_clean=c_.top10_share.median(), n_pos_terms_clean=c_.n_pos.median(), pos_mass_clean=c_.pos_mass.median(),
                          dr_clean=float((c_.logit > _thr_logit).mean()), dr_padded=float((p_.logit > _thr_logit).mean())))
    LR54 = pd.DataFrame(_rows); LR54.to_csv(DIRS["results"] / "v54_lrword_contributions.csv", index=False)
    display(LR54.round(3))
    # most malicious-indicative vocabulary terms that occur in phishing-pool emails (vocabulary only, no email text)
    if "phish_clean" in SC:
        Xp = _vec.transform(SC["phish_clean"].text); df_ = np.asarray((Xp > 0).sum(0)).ravel(); voc = np.array(_vec.get_feature_names_out())
        top = np.argsort(-_w)[:3000]; top = top[df_[top] >= 15][:25]
        LR54_TERMS = pd.DataFrame({"term": voc[top], "weight": _w[top], "phishing_pool_doc_freq": df_[top]})
        LR54_TERMS.to_csv(DIRS["results"] / "v54_lrword_top_terms_phishing.csv", index=False); display(LR54_TERMS.head(15))
else:
    print("LR-word pipeline without linear coefficients; analysis skipped.")

# %% [markdown]
# ### 14c.3 Adaptive injection: uncapped measures
# Against ZS-7B the instruction search succeeds on every caught email, which caps the instruction-attributable difference. Success within small query
# budgets and the paired number of queries to first success are not capped this way. Failures count as one query past the budget.

# %%
_rows = []
_qcap = V52["inj_attack"]["steps"] * V52["inj_attack"]["cands"]
for d, P_ in INJ.items():
    a, b = P_["inj"].set_index("id"), P_["ben"].set_index("id")
    ids = a.index[a.caught.astype(bool)].intersection(b.index[b.caught.astype(bool)])
    if not len(ids): continue
    qa = np.where(a.loc[ids].success.astype(bool), a.loc[ids].queries, _qcap + 1).astype(float)
    qb = np.where(b.loc[ids].success.astype(bool), b.loc[ids].queries, _qcap + 1).astype(float)
    r_ = dict(target=d, n_caught=len(ids))
    for q in (4, 8, 16, 32):
        r_[f"asr_inst_q{q}"], r_[f"asr_ben_q{q}"] = float((qa <= q).mean()), float((qb <= q).mean())
    dq = qb - qa; m_, lo_, hi_ = boot_mean(dq)
    try: p_w = float(stats.wilcoxon(qa, qb).pvalue) if np.any(qa != qb) else 1.0
    except Exception: p_w = np.nan
    r_.update(median_q_inst=float(np.median(qa)), median_q_ben=float(np.median(qb)), mean_q_saved_by_instruction=m_, lo=lo_, hi=hi_, p_wilcoxon=p_w)
    _rows.append(r_)
INJ54 = pd.DataFrame(_rows); INJ54.to_csv(DIRS["results"] / "v54_injection_uncapped.csv", index=False)
display(INJ54.round(3)) if len(INJ54) else print("Adaptive injection logs not found.")
_ie = cpath("sets/inj_attack_emails.parquet")
INJ54_PHISH = None
if _ie.exists():
    _iem = pd.read_parquet(_ie); INJ54_PHISH = int(_iem.source.isin(PHISH_SRC).sum())
    print(f"Adaptive injection emails: {len(_iem)}, of which phishing or fraud: {INJ54_PHISH}")

# %% [markdown]
# ### 14c.4 Recent human-written PhishFuzzer mail by malicious type

# %%
HUM54 = pd.DataFrame()
if "modern_human" in SC and "ptype" in SC["modern_human"]:
    H_ = SC["modern_human"]; rng_ = np.random.default_rng(SEED + 541)
    _dets = [c[2:] for c in H_.columns if c.startswith("s_") and H_[c].notna().all() and c[2:] != "InjDet"]
    def _auc_ci(y, s, B=500):
        y, s = np.asarray(y), np.asarray(s); i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; v = []
        for _ in range(B):
            j = np.r_[rng_.choice(i1, len(i1)), rng_.choice(i0, len(i0))]; v.append(roc_auc_score(y[j], s[j]))
        return roc_auc_score(y, s), np.quantile(v, 0.025), np.quantile(v, 0.975)
    _rows = []
    for d in _dets:
        r_ = dict(detector=d)
        for pos in ("Phishing", "Spam"):
            sub = H_[H_.ptype.isin([pos, "Valid"])]
            r_[f"auc_{pos.lower()}"], r_[f"auc_{pos.lower()}_lo"], r_[f"auc_{pos.lower()}_hi"] = _auc_ci((sub.ptype == pos).astype(int), sub[f"s_{d}"])
        r_["auc_all"] = roc_auc_score(H_.label, H_[f"s_{d}"])
        if d in THR[A0]: r_["fpr_valid"] = float((H_[H_.label == 0][f"s_{d}"] > THR[A0][d]).mean())
        _rows.append(r_)
    HUM54 = pd.DataFrame(_rows); HUM54.to_csv(DIRS["results"] / "v54_modern_human_by_type.csv", index=False)
    display(HUM54.round(3))
    print("Ranges over detectors:", {c: (round(HUM54[c].min(), 3), round(HUM54[c].max(), 3)) for c in ["auc_phishing", "auc_spam", "auc_all"]})

# %% [markdown]
# ### 14c.5 IT-like legitimate emails per training sample and IT-notice false positives per seed
# Seeds resample the 8,000 training emails. We count legitimate (and malicious) training emails that look like IT notices, using a fixed keyword rule,
# for each seed's sample, next to that seed's false-positive rate on the 100 IT notices of the ordinary-instruction set. With three seeds this is
# descriptive only.

# %%
_IT_RE = re.compile(r"\b(password|passcode|log[ -]?in|sign[ -]?in|credential|help ?desk|it (?:department|support|team|services)|vpn|two[- ]factor|"
                    r"2fa|mfa|authenticat\w*|account (?:security|verification)|reset)\b", re.I)
IT54 = pd.DataFrame()
if "legit_realinstr" in SC:
    _ri = SC["legit_realinstr"]; _itm = _ri.kind.astype(str).str.startswith("an IT").values
    _rows = []
    for k_, x_ in enumerate(LORA_SEEDS[P_TAG]):
        try:
            tr_ = S.train_sample(dict(cache=str(DIRS["cache"]), n_train=CFG["n_lora_train"]), x_)
        except Exception as ex:
            print("train_sample failed:", repr(ex)[:160]); break
        hits = tr_.text.map(lambda t: len(set(m.lower() for m in _IT_RE.findall(str(t)))))
        r_ = dict(seed=x_, it_like_legit=int(((hits >= 2) & (tr_.label == 0)).sum()), it_like_malicious=int(((hits >= 2) & (tr_.label == 1)).sum()))
        for base_ in (LORA_P, AUGMIX_DET):
            col_ = base_ if k_ == 0 else f"{base_}@s{x_}"
            if f"s_{col_}" in _ri and f"s_{col_}" in SC["val"] and _ri[f"s_{col_}"].notna().all():
                _vl = SC["val"][SC["val"].label == 0][f"s_{col_}"].dropna().values
                _t = THR[A0][col_] if col_ in THR[A0] else float(np.sort(_vl)[min(int(np.floor((1 - A0) * len(_vl))), len(_vl) - 1)])
                f_ = (_ri[f"s_{col_}"].values > _t).astype(float); r_[f"fpr_it_{base_}"] = float(f_[_itm].mean()); r_[f"fpr_instr_{base_}"] = float(f_.mean())
        _rows.append(r_)
    IT54 = pd.DataFrame(_rows); IT54.to_csv(DIRS["results"] / "v54_it_like_by_seed.csv", index=False); display(IT54.round(3))

# %% [markdown]
# ### 14c.6 Partial AUC (FPR at most 1%)
# McClish-standardized partial AUC against the 3,000 legitimate test emails: 0.5 is chance and 1 is perfect within the region FPR at most 1%.

# %%
_PA_SETS = [s_ for s_ in ["clean", "ham_pad", "pad_prefix", "pad_long", "inject_aware", "phish_clean", "phish_ham_pad"] if s_ == "clean" or s_ in SC]
_leg = SC["test_clean"][SC["test_clean"].label == 0]; _mal_clean = SC["test_clean"][SC["test_clean"].label == 1]
_rows = []
for d in dict.fromkeys(EVAL_DETS + PROMPT_AVAIL + V5_ALL):
    if not d or not _has(d, "test_clean"): continue
    r_ = dict(detector=d)
    for s_ in _PA_SETS:
        M_ = _mal_clean if s_ == "clean" else attack_frame(s_)
        if f"s_{d}" not in M_ or M_[f"s_{d}"].isna().any(): continue
        y = np.r_[np.zeros(len(_leg)), np.ones(len(M_))]; sc = np.r_[_leg[f"s_{d}"].values, M_[f"s_{d}"].values]
        r_[f"pauc_{s_}"] = roc_auc_score(y, sc, max_fpr=0.01)
    _rows.append(r_)
PAUC54 = pd.DataFrame(_rows); PAUC54.to_csv(DIRS["results"] / "v54_partial_auc.csv", index=False); display(PAUC54.round(3))

# %% [markdown]
# ### Fig. 35: adaptive attacks, legible version
# (a) score-guided and (b) label-only padding attacks, each with at most six detectors labeled at the line end; (c) the white-box GCG sanity check.

# %%
if len(GREEDY_CURVES):
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 3.0), gridspec_kw={"width_ratios": [1.2, 1.2, 0.75]})
    _keep = {"score": [C_STAR, LORA_P, AUG_DET, AUGMIX_DET, "Hybrid-OR", HYB_AUGMIX],
             "label": [C_STAR, LORA_P, AUGMIX_DET, "Hybrid-OR", HYB_AUGMIX]}
    for ax, mode_, ttl in ((axes[0], "score", "(a) Score-guided"), (axes[1], "label", "(b) Label-only")):
        for d in _keep[mode_]:
            if (mode_, d) not in GREEDY_CURVES: continue
            ax.plot(Q_GRID, GREEDY_CURVES[(mode_, d)], color=DET_COL.get(d, INK2), lw=1.5, marker=DET_MARK.get(d, "o"), ms=3,
                    markevery=max(1, len(Q_GRID) // 6), label=d.replace("-1.5B", ""))
        ax.set_xlim(0, Q_GRID[-1]); ax.set_ylim(0, 1.03); ax.set_xlabel("Queries to the detector"); ax.set_title(ttl, fontsize=7.5, loc="left")
        ax.legend(fontsize=5.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, handlelength=1.8, columnspacing=0.8)
    axes[0].set_ylabel("Attack success (caught emails)")
    ax = axes[2]; gc_ = ATK5[ATK5.attack == "GCG suffix"] if len(ATK5) else pd.DataFrame()
    if len(gc_):
        x = np.arange(len(gc_))
        ax.bar(x, gc_.asr, 0.55, color=[DET_COL.get(d, INK2) for d in gc_.target])
        ax.errorbar(x, gc_.asr, yerr=[gc_.asr - gc_.asr_lo, gc_.asr_hi - gc_.asr], fmt="none", ecolor=INK, lw=0.8, capsize=2)
        ax.set_xticks(x, [d.replace("-1.5B", "") for d in gc_.target], rotation=30, ha="right", fontsize=6); ax.set_ylim(0, 1.03)
    ax.set_title("(c) GCG, 30 emails", fontsize=7.5, loc="left")
    fig.tight_layout(w_pad=1.5)
    savefig(fig, "fig35_adaptive_attacks", "Adaptive attacks on emails each detector catches in clean form. (a) Score-guided and (b) label-only padding "
            "attack success by query budget (LoRA-Gen and RoBERTa-FT reach 100% like LoRA and are omitted from (a)); (c) white-box GCG suffix "
            "success after decoding to text (95% CI).")

# %% [markdown]
# ### 14c.7 Key findings (v5.4)

# %%
F54 = []
if len(MCN54):
    F54 += [f"{r.pool}, {r.attack}: AugMix {100*r.asr_augmix:.1f}% vs LR-word {100*r.asr_lrword:.1f}% on {r.n_both_caught} emails both catch "
            f"(only AugMix evaded {r.only_augmix}, only LR-word {r.only_lrword}; McNemar p={r.p_mcnemar:.3g}, Holm {r.p_holm:.3g})" for r in MCN54.itertuples()]
if len(LR54):
    F54 += [f"LR-word {r.subset}: median logit {r.logit_clean:.2f} -> {r.logit_padded:.2f} (drop {r.logit_drop:.2f}); top-10 share {r.top10_share_clean:.2f}; "
            f"positive terms {r.n_pos_terms_clean:.0f}; DR {100*r.dr_clean:.1f} -> {100*r.dr_padded:.1f}" for r in LR54.itertuples()]
if len(INJ54):
    F54 += [f"Injection {r.target}: success within 8/16 queries, instruction {100*r.asr_inst_q8:.0f}/{100*r.asr_inst_q16:.0f}% vs benign "
            f"{100*r.asr_ben_q8:.0f}/{100*r.asr_ben_q16:.0f}%; median queries {r.median_q_inst:.0f} vs {r.median_q_ben:.0f} (Wilcoxon p={r.p_wilcoxon:.3g})"
            for r in INJ54.itertuples()]
if INJ54_PHISH is not None: F54.append(f"Adaptive injection emails that are phishing or fraud: {INJ54_PHISH}")
if len(HUM54):
    F54.append("Human-written PhishFuzzer AUC ranges: " + ", ".join(f"{c} {HUM54[c].min():.2f} to {HUM54[c].max():.2f}" for c in ["auc_phishing", "auc_spam", "auc_all"]))
if len(IT54): F54.append("IT-like training emails and IT-notice FPR by seed: " + json.dumps(IT54.round(3).to_dict(orient="records")))
if len(PAUC54):
    F54.append("Partial AUC (FPR<=1%), natural pad: " + "; ".join(f"{r.detector} {getattr(r, 'pauc_ham_pad', np.nan):.3f}" for r in PAUC54.itertuples()
                                                                  if np.isfinite(getattr(r, "pauc_ham_pad", np.nan))))
print("KEY FINDINGS v5.4\n" + "\n".join(F54))
with open(DIRS["results"] / "key_findings.md", "a") as f_:
    f_.write("\n\n## v5.4 analyses for the second expert review\n" + "\n".join(F54))
V54_SUMMARY = {"augmix_vs_lrword": MCN54.round(5).to_dict(orient="records"), "lrword_contributions": LR54.round(5).to_dict(orient="records"),
               "injection_uncapped": INJ54.round(5).to_dict(orient="records"), "injection_phishing_emails": INJ54_PHISH,
               "modern_human_by_type": HUM54.round(5).to_dict(orient="records"), "it_like_by_seed": IT54.round(5).to_dict(orient="records"),
               "partial_auc": PAUC54.round(5).to_dict(orient="records")}
