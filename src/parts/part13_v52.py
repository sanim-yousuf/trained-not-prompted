# %% [markdown]
# ## 14. Second review: read-out seeds, adaptive injection, phishing pool (v5.2)
# Each subsection answers one point of the second review. Thresholds are the validation-fixed ones (1% FPR); intervals are 95% bootstrap CIs. Detectors or
# sets whose jobs were deferred or failed are skipped and listed.

# %%
V52 = CFG["v5"]["v52"]
def _ok(d, s):
    return (d == "Hybrid-OR" and s in SC) or _has(d, s)
def _seedcols(d):
    return [c for c in [d] + V52_SEED_GROUPS.get(d, []) if c in THR[A0]]
print("v5.2 seed columns available:", {d: _seedcols(d) for d in V52_SEED_GROUPS})
print("v5.2 phishing sets scored:", [s_ for s_ in PHISH_SETS if s_ in SC])

# %% [markdown]
# ### 14.1 Seeds for every one-factor read-out variant
# Natural padding varied by 12 points over LoRA seeds, so a one-seed ranking of read-outs is not enough. Every variant is reported over its seeds, and each is
# compared with `LoRA-1.5B` in two ways: a Welch t-test over seeds (seed variation) and a bootstrap over emails of the seed-averaged difference (email
# sampling). Holm correction over all comparisons.

# %%
RO_GROUPS = {LORA_P: [c for c in SEED_GROUPS.get(LORA_P, [LORA_P]) if c in THR[A0]]}
for d in [GEN_DET, PROMPTHEAD_DET, MEAN_DET, ENC_DET]:
    if d in THR[A0]: RO_GROUPS[d] = _seedcols(d)
if AUGMIX_DET in THR[A0]: RO_GROUPS[AUGMIX_DET] = [c for c in SEED_GROUPS.get(AUGMIX_DET, [AUGMIX_DET]) if c in THR[A0]]
PAD_ALL = [s_ for s_ in ["ham_pad"] + OOD_SETS if s_ in SC]
_rows = []
for g_, cols in RO_GROUPS.items():
    for c in cols:
        r = dict(detector=g_, column=c, auc=roc_auc_score(SC["test_clean"].label, SC["test_clean"][f"s_{c}"]), dr_clean=float(flag(CLEAN_POOL, c).mean()))
        for s_ in PAD_ALL + ["inject_aware"]:
            r[f"dr_{s_}"] = float(flag(_frame(s_), c).mean()) if _has(c, s_) else np.nan
        _pv = [r[f"dr_{s_}"] for s_ in PAD_ALL if np.isfinite(r[f"dr_{s_}"])]
        r["dr_worst_pad"] = min(_pv) if _pv else np.nan
        r["auc_lost_aware"] = auc_effect(c, "neutral", "inject_aware")[0] if _has(c, "neutral") and _has(c, "inject_aware") else np.nan
        r["fpr_realinstr"] = float(flag(SC["legit_realinstr"], c).mean()) if _has(c, "legit_realinstr") else np.nan
        _rows.append(r)
RO_SEED_LONG = pd.DataFrame(_rows)
_m = [c for c in RO_SEED_LONG.columns if c not in ("detector", "column")]
RO_SEED_TAB = RO_SEED_LONG.groupby("detector", sort=False)[_m].agg(["mean", "std", "min", "max", "count"])
RO_SEED_TAB.columns = [f"{a}_{b}" for a, b in RO_SEED_TAB.columns]; RO_SEED_TAB = RO_SEED_TAB.reset_index()
RO_SEED_LONG.to_csv(DIRS["results"] / "v52_readout_seeds_long.csv", index=False)
RO_SEED_TAB.to_csv(DIRS["results"] / "v52_readout_seeds.csv", index=False)
display(RO_SEED_TAB[["detector", "dr_ham_pad_count", "dr_ham_pad_mean", "dr_ham_pad_std", "dr_ham_pad_min", "dr_ham_pad_max", "dr_worst_pad_mean",
                     "auc_lost_aware_mean", "fpr_realinstr_mean"]].round(4))

def _seed_avg_flags(cols, s_):
    df_ = _frame(s_).set_index("base_id")
    return df_, np.mean([flag(df_.reset_index(), c).astype(float) for c in cols], axis=0)
_rows = []
_pairs = [(d, LORA_P) for d in RO_GROUPS if d != LORA_P] + ([(PROMPTHEAD_DET, GEN_DET)] if {PROMPTHEAD_DET, GEN_DET} <= set(RO_GROUPS) else [])
for d1, d2 in _pairs:
    for s_ in ["ham_pad", "pad_prefix", "pad_long"]:
        c1 = [c for c in RO_GROUPS[d1] if _has(c, s_)]; c2 = [c for c in RO_GROUPS[d2] if _has(c, s_)]
        if not c1 or not c2: continue
        v1 = RO_SEED_LONG.set_index("column").loc[c1, f"dr_{s_}"].values; v2 = RO_SEED_LONG.set_index("column").loc[c2, f"dr_{s_}"].values
        p_w = float(stats.ttest_ind(v1, v2, equal_var=False).pvalue) if len(v1) > 1 and len(v2) > 1 else np.nan
        _, f1 = _seed_avg_flags(c1, s_); _, f2 = _seed_avg_flags(c2, s_)
        m_, lo_, hi_ = boot_diff(f1, f2)
        _rows.append(dict(variant=d1, reference=d2, set=s_, seeds_variant=len(v1), seeds_reference=len(v2), mean_variant=v1.mean(), mean_reference=v2.mean(),
                          dr_diff=m_, dr_diff_lo=lo_, dr_diff_hi=hi_, p_welch=p_w, separated=bool(v1.min() > v2.max() or v1.max() < v2.min())))
RO_SEED_CMP = pd.DataFrame(_rows)
if len(RO_SEED_CMP):
    _pw = RO_SEED_CMP.p_welch.fillna(1.0).values
    RO_SEED_CMP["p_welch_holm"] = holm(_pw)
RO_SEED_CMP.to_csv(DIRS["results"] / "v52_readout_seed_tests.csv", index=False)
if len(RO_SEED_CMP): display(RO_SEED_CMP.round(4))

# %% [markdown]
# ### 14.2 Adaptive injection attack with a matched benign control
# The template injections are fixed. Here the score-guided attacker searches over 400 instruction sentences addressed to the classifier (half name the label
# word) and, separately, over length-matched benign sentences, with the same budget, positions, seed and emails. Adding any text can lower a score, so the
# **instruction-attributable success** is the paired difference between the two pools on the same caught emails (McNemar test). A read-out that follows
# instructions should show a large positive difference; one that does not should show none.

# %%
INJ = {d: {k: pd.read_parquet(p) for k, p in pools.items()} for d, pools in ATTACK_FILES.get("injatk", {}).items()
       if len(pools) == 2 and all(Path(p).exists() for p in pools.values())}
_rows, INJ_CURVES = [], {}
_qmax = V52["inj_attack"]["steps"] * V52["inj_attack"]["cands"]
Q_INJ = np.arange(0, _qmax + 1, max(1, V52["inj_attack"]["cands"]))
for d, P_ in INJ.items():
    a, b = P_["inj"].set_index("id"), P_["ben"].set_index("id")
    ids = a.index[a.caught.astype(bool)].intersection(b.index[b.caught.astype(bool)])
    if not len(ids): continue
    si, sb = a.loc[ids].success.astype(bool).values, b.loc[ids].success.astype(bool).values
    mi, mb, dd = boot_mean(si), boot_mean(sb), boot_diff(si.astype(float), sb.astype(float))
    n10, n01, p_ = mcnemar(si, sb)
    for k_, g_ in (("inj", a.loc[ids]), ("ben", b.loc[ids])):
        sq = np.where(g_.success.astype(bool).values, g_.queries.values, np.inf); INJ_CURVES[(d, k_)] = [float((sq <= q).mean()) for q in Q_INJ]
    adv = a.loc[ids][si]
    _rows.append(dict(target=d, kind="prompted" if d.startswith("ZS-") else "trained", n_caught=len(ids), asr_instruction=mi[0], asr_instruction_lo=mi[1],
                      asr_instruction_hi=mi[2], asr_benign=mb[0], asr_benign_lo=mb[1], asr_benign_hi=mb[2], asr_diff=dd[0], asr_diff_lo=dd[1], asr_diff_hi=dd[2],
                      only_instruction=n10, only_benign=n01, p_mcnemar=p_,
                      median_words_instruction=float(adv.words_added.median()) if len(adv) else np.nan,
                      share_label_word=float(adv.adv_text.str.contains(BEN_WORD).mean()) if len(adv) else np.nan,
                      mean_score_drop_instruction=float((a.loc[ids].s_clean - a.loc[ids].s_final).mean()),
                      mean_score_drop_benign=float((b.loc[ids].s_clean - b.loc[ids].s_final).mean())))
INJ_TAB = pd.DataFrame(_rows)
if len(INJ_TAB): INJ_TAB["p_holm"] = holm(INJ_TAB.p_mcnemar.values)
INJ_TAB.to_csv(DIRS["results"] / "v52_injection_adaptive.csv", index=False)
if len(INJ_TAB): display(INJ_TAB.round(4))
else: print("Adaptive injection jobs not finished:", list(ATTACK_FILES.get("injatk", {})))
_ia_aud = [P_["inj"][P_["inj"].success.astype(bool)].assign(target=d) for d, P_ in INJ.items() if P_["inj"].success.any()]
if _ia_aud:   # private sheet: are successful instruction-attack emails still the same threat?
    _a = pd.concat(_ia_aud).sample(min(30, sum(map(len, _ia_aud))), random_state=SEED)
    _a.assign(rater1_still_malicious_yn="", rater2_still_malicious_yn="")[["id", "target", "words_added", "adv_text", "rater1_still_malicious_yn",
                                                                            "rater2_still_malicious_yn"]].to_csv(DIRS["private"] / "injection_attack_audit.csv", index=False)

# %% [markdown]
# ### 14.3 Phishing-only pool
# 500 phishing and fraud test emails outside the main attack pool, under natural padding (80 words suffix and prefix, 240 words), the neutral suffix and the
# label-aware injection. Instruction effects are net of the neutral suffix; padding effects are against the same emails clean.

# %%
PH_DETS = [d for d in dict.fromkeys([C_STAR, "SVM-char", ZS_P, PROBE_P, LARGE_DET, LORA_P, MEAN_DET, GEN_DET, PROMPTHEAD_DET, ENC_DET, AUG_DET, AUGMIX_DET,
                                     ENC_AUGMIX_DET, "Hybrid-OR", "Hybrid-Stack"]) if d and _ok(d, "phish_clean")]
_rows = []
for d in PH_DETS:
    r = dict(detector=d, n=len(SC["phish_clean"]) if "phish_clean" in SC else 0)
    for s_ in PHISH_SETS:
        r[f"dr_{s_}"], r[f"dr_{s_}_lo"], r[f"dr_{s_}_hi"] = boot_mean(flag(attack_frame(s_), d)) if _ok(d, s_) else (np.nan,) * 3
    if _ok(d, "phish_neutral") and _ok(d, "phish_inject_aware"):
        A, B = paired("phish_neutral", "phish_inject_aware")
        r["instr_dr"], r["instr_dr_lo"], r["instr_dr_hi"] = boot_diff(flag(A.reset_index(), d).astype(float), flag(B.reset_index(), d).astype(float))
        if d != "Hybrid-OR":
            r["instr_auc"], r["instr_auc_lo"], r["instr_auc_hi"] = boot_diff(rank_scores(d, A[f"s_{d}"].values), rank_scores(d, B[f"s_{d}"].values))
    if _ok(d, "phish_ham_pad"):
        A, B = paired("phish_clean", "phish_ham_pad")
        r["pad_loss_dr"], r["pad_loss_dr_lo"], r["pad_loss_dr_hi"] = boot_diff(flag(A.reset_index(), d).astype(float), flag(B.reset_index(), d).astype(float))
    _rows.append(r)
PH_TAB = pd.DataFrame(_rows); PH_TAB.to_csv(DIRS["results"] / "v52_phish_pool.csv", index=False)
if len(PH_TAB): display(PH_TAB[["detector", "n"] + [f"dr_{s_}" for s_ in PHISH_SETS] + [c for c in ["instr_dr", "instr_auc", "pad_loss_dr"] if c in PH_TAB]].round(4))

# Adaptive padding attacks on phishing emails: GPU targets from the job queue, the classical detector here on CPU
_gpe = cpath("sets/greedy_phish_emails.parquet")
if V52["phish"]["enabled"] and _gpe.exists() and CFG["v5"]["greedy"]["enabled"]:
    _gpe_ = pd.read_parquet(_gpe); _gs_ = pd.read_parquet(cpath("sets/greedy_sentences.parquet")).text.tolist()
    for mode_, key_ in (("score", "phgreedy"), ("label", "phlabel")):
        _out = cpath(f"{key_}_{_slug(C_STAR)}.parquet")
        if not _out.exists():
            _res = S.greedy_pad_attack(lambda t: CLASSICAL[C_STAR].predict_proba(list(t))[:, 1], _gpe_.id.tolist(), _gpe_.text.tolist(), _gs_,
                                       float(THR[A0][C_STAR]), CFG["v5"]["greedy"], SEED + 701, log_every=0, mode=mode_)
            _res["threshold"] = float(THR[A0][C_STAR]); _res["target"] = C_STAR; _res.to_parquet(_out, index=False)
        ATTACK_FILES.setdefault(key_, {})[C_STAR] = _out
_rows = []
for mode_, key_ in (("score-guided", "phgreedy"), ("label-only", "phlabel")):
    for d, p in ATTACK_FILES.get(key_, {}).items():
        if not Path(p).exists(): continue
        g = pd.read_parquet(p); c = g[g.caught.astype(bool)]
        if not len(c): continue
        m_, lo_, hi_ = boot_mean(c.success.astype(float))
        _rows.append(dict(attack=f"{mode_} padding (phishing)", target=d, n=len(c), asr=m_, asr_lo=lo_, asr_hi=hi_,
                          median_queries=float(c.queries[c.success.astype(bool)].median()) if c.success.any() else np.nan,
                          median_words=float(c.words_added[c.success.astype(bool)].median()) if c.success.any() else np.nan))
PH_ATK = pd.DataFrame(_rows); PH_ATK.to_csv(DIRS["results"] / "v52_phish_attacks.csv", index=False)
if len(PH_ATK): display(PH_ATK.round(4))

# %% [markdown]
# ### 14.4 Collateral false positives by kind of ordinary instruction
# The 9% FPR of `LoRA-AugMix` on legitimate mail with ordinary instructions is not uniform; per-kind rates show which everyday mail a defense would block.

# %%
RI_KIND = pd.DataFrame()
if "legit_realinstr" in SC and "kind" in SC["legit_realinstr"]:
    _ri = SC["legit_realinstr"]
    _kd = [d for d in dict.fromkeys([C_STAR, LORA_P, AUG_DET, AUGMIX_DET, GEN_DET, PROMPTHEAD_DET, MEAN_DET, ENC_DET, ENC_AUGMIX_DET, WIN_DET, LARGE_DET,
                                     PROBE_P, "Hybrid-OR"]) if d and _ok(d, "legit_realinstr")]
    _rows = []
    for d in _kd:
        f_ = flag(_ri, d)
        for k_, idx in _ri.groupby("kind").indices.items():
            m_, lo_, hi_ = boot_mean(f_[idx]); _rows.append(dict(detector=d, kind=k_, n=len(idx), fpr=m_, fpr_lo=lo_, fpr_hi=hi_))
    RI_KIND = pd.DataFrame(_rows); RI_KIND.to_csv(DIRS["results"] / "v52_realinstr_by_kind.csv", index=False)
    display(RI_KIND.pivot(index="kind", columns="detector", values="fpr").round(3))

# %% [markdown]
# ### 14.5 Compute account
# Measured GPU time from the cached metadata of every job in this cache (training time, per-email scoring time times set size, attack time), so it covers all
# sessions that built the cache. Model loading, paraphrasing and text generation are not timed there and are added from this session's logs.

# %%
def _gpu_account():
    rows = []
    n_set = {k_: len(v_) for k_, v_ in SETS.items()}
    for p in DIRS["cache"].glob("*_trainmeta.json"):
        try: rows.append(dict(item=p.name[:-15], part="train", sec=float(json.loads(p.read_text()).get("train_sec", 0))))
        except Exception: pass
    for p in list(DIRS["cache"].glob("*_scoremeta.json")) + list(DIRS["cache"].glob("zs_*_meta.json")):
        try:
            m = json.loads(p.read_text()); m = m.get("sec_per_email", m) if p.name.startswith("zs_") else m
            rows.append(dict(item=p.name.rsplit("_", 1)[0], part="score", sec=float(sum(float(v) * n_set.get(k, 0) for k, v in m.items() if isinstance(v, (int, float))))))
        except Exception: pass
    for p in DIRS["cache"].glob("*.parquet"):
        if p.name.startswith(("greedy_", "label_", "gcg_", "injatk_", "phgreedy_", "phlabel_")):
            try:
                g = pd.read_parquet(p, columns=None)
                if "sec_total" in g: rows.append(dict(item=p.stem, part="attack", sec=float(g.sec_total.iloc[0])))
            except Exception: pass
    sess = 0.0
    for p in DIRS["logs"].glob("*.log"):
        for m in re.finditer(r"job (\S+) finished in ([\d.]+) min", p.read_text(errors="ignore")):
            if m.group(1).startswith(("para", "gen_", "benign", "realinstr")): sess += float(m.group(2)) * 60
    rows.append(dict(item="generation jobs (this session's logs)", part="generate", sec=sess))
    return pd.DataFrame(rows)
GPU_ACC = _gpu_account(); GPU_ACC.to_csv(DIRS["results"] / "v52_gpu_account.csv", index=False)
GPU_HOURS = {k: round(v / 3600, 2) for k, v in GPU_ACC.groupby("part").sec.sum().items()} | {"total": round(GPU_ACC.sec.sum() / 3600, 2)}
print("Measured GPU hours by part:", GPU_HOURS)

# %% [markdown]
# ### Fig. 33: Adaptive injection attack vs matched benign control
# (a) Success by query budget, instruction sentences (solid) vs length-matched benign sentences (dashed). (b) Instruction-attributable success (paired
# difference, 95% CI).

# %%
if len(INJ_TAB):
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.9), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for d in INJ_TAB.target:
        c_ = DET_COL.get(d, INK2); mk = DET_MARK.get(d, "o")
        ax.plot(Q_INJ, INJ_CURVES[(d, "inj")], color=c_, marker=mk, ms=3.2, lw=1.4, ls="-", label=f"{d}, instructions")
        ax.plot(Q_INJ, INJ_CURVES[(d, "ben")], color=c_, marker=mk, ms=3.2, lw=1.1, ls="--", mfc="white", label=f"{d}, benign control")
    ax.set_xlabel("Queries to the detector"); ax.set_ylabel("Attack success (share of caught emails)"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=5.5, ncol=2, loc="upper center", bbox_to_anchor=(0.55, -0.22), frameon=False); panel_label(ax, "(a)", x=-0.12)
    ax = axes[1]; yy = np.arange(len(INJ_TAB))[::-1]
    for yi, (_, r) in zip(yy, INJ_TAB.iterrows()):
        c_ = DET_COL.get(r.target, INK2)
        ax.plot([r.asr_diff_lo, r.asr_diff_hi], [yi, yi], color=c_, lw=1.4, alpha=0.7); ax.plot(r.asr_diff, yi, marker=DET_MARK.get(r.target, "o"), color=c_, ms=5.5, ls="none")
    ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(yy, INJ_TAB.target); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Instruction-attributable success\n(instructions minus benign control)"); panel_label(ax, "(b)", x=-0.35)
    fig.tight_layout()
    savefig(fig, "fig33_injection_adaptive", "Adaptive injection attack: score-guided search over instruction sentences addressed to the classifier vs length-matched "
            "benign sentences under the same budget. (a) success by queries; (b) paired difference on the same caught emails (95% CI).")

# %% [markdown]
# ### Fig. 34: Read-out variants over seeds (natural padding and worst padding variant)

# %%
if len(RO_SEED_LONG):
    rows34 = list(RO_GROUPS)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 0.32 * len(rows34) + 1.0), sharey=True)
    yy = np.arange(len(rows34))[::-1]
    for ax, col_, lab in ((axes[0], "dr_ham_pad", "Detection rate, natural pad (80 words)"), (axes[1], "dr_worst_pad", "Detection rate, worst padding variant")):
        for yi, d in zip(yy, rows34):
            v = RO_SEED_LONG[RO_SEED_LONG.detector == d][col_].dropna().values; c_ = DET_COL.get(d, INK2)
            if not len(v): continue
            ax.plot(v, np.full(len(v), yi), marker=DET_MARK.get(d, "o"), color=c_, mfc="white", ms=4.5, ls="none", mew=0.9)
            ax.plot(v.mean(), yi, marker="|", color=c_, ms=11, mew=2)
        ax.set_xlim(0, 1.03); ax.grid(axis="y", visible=False); ax.set_xlabel(lab)
    axes[0].set_yticks(yy, rows34); panel_label(axes[0], "(a)", x=-0.42); panel_label(axes[1], "(b)", x=-0.05)
    fig.tight_layout()
    savefig(fig, "fig34_readout_seeds", "Read-out variants over seeds: each open marker is one seed, the bar is the mean. (a) natural padding; (b) worst of six "
            "padding variants.")

# %% [markdown]
# ### 14.6 Tables and key findings (v5.2)

# %%
if len(RO_SEED_TAB):
    _f = lambda m, s: "n/a" if not np.isfinite(m) else (f"{100*m:.1f} $\\pm$ {100*s:.1f}" if np.isfinite(s) else f"{100*m:.1f}")
    t41 = pd.DataFrame({"Detector": RO_SEED_TAB.detector, "Seeds": RO_SEED_TAB.dr_ham_pad_count.astype(int),
                        "Clean": [_f(m, s) for m, s in zip(RO_SEED_TAB.dr_clean_mean, RO_SEED_TAB.dr_clean_std)],
                        "Natural pad": [_f(m, s) for m, s in zip(RO_SEED_TAB.dr_ham_pad_mean, RO_SEED_TAB.dr_ham_pad_std)],
                        "Worst pad": [_f(m, s) for m, s in zip(RO_SEED_TAB.dr_worst_pad_mean, RO_SEED_TAB.dr_worst_pad_std)],
                        "AUC lost (aware)": [f"{m:.4f}" for m in RO_SEED_TAB.auc_lost_aware_mean],
                        "FPR +instructions": [_f(m, s) for m, s in zip(RO_SEED_TAB.fpr_realinstr_mean, RO_SEED_TAB.fpr_realinstr_std)]})
    write_latex(t41, "tab41_readout_seeds", "Read-out variants over seeds (detection and FPR in \\%, mean $\\pm$ SD).", "tab:roseeds")
if len(INJ_TAB):
    t42 = pd.DataFrame({"Target": INJ_TAB.target, "Caught": INJ_TAB.n_caught,
                        "Instructions": [f"{100*a:.1f} [{100*b:.1f}, {100*c:.1f}]" for a, b, c in zip(INJ_TAB.asr_instruction, INJ_TAB.asr_instruction_lo, INJ_TAB.asr_instruction_hi)],
                        "Benign control": [f"{100*a:.1f} [{100*b:.1f}, {100*c:.1f}]" for a, b, c in zip(INJ_TAB.asr_benign, INJ_TAB.asr_benign_lo, INJ_TAB.asr_benign_hi)],
                        "Difference": [f"{100*a:+.1f} [{100*b:+.1f}, {100*c:+.1f}]" for a, b, c in zip(INJ_TAB.asr_diff, INJ_TAB.asr_diff_lo, INJ_TAB.asr_diff_hi)],
                        "Holm p": INJ_TAB.p_holm.map(lambda v: f"{v:.3g}")})
    write_latex(t42, "tab42_injection_adaptive", "Adaptive injection attack vs matched benign control (success in \\% of caught emails).", "tab:injadaptive")
if len(PH_TAB):
    _lab43 = {**ATT_LABEL, **OOD_LABEL}
    t43 = pd.DataFrame({"Detector": PH_TAB.detector} | {_lab43.get(s_[6:], s_[6:]) if s_ != "phish_clean" else "Clean": PH_TAB[f"dr_{s_}"].map(pct)
                                                        for s_ in PHISH_SETS if f"dr_{s_}" in PH_TAB}
                       | {"Instruction effect (pts)": _col(PH_TAB, "instr_dr").map(lambda v: "n/a" if not np.isfinite(v) else f"{100*v:+.1f}")})
    write_latex(t43, "tab43_phish_pool", "Phishing-only pool: detection rate (\\%) at 1\\% validation FPR.", "tab:phishpool")
if len(PH_ATK):
    t44 = pd.DataFrame({"Attack": PH_ATK.attack, "Target": PH_ATK.target, "Emails": PH_ATK.n,
                        "Success [95% CI]": [f"{100*a:.1f} [{100*b:.1f}, {100*c:.1f}]" for a, b, c in zip(PH_ATK.asr, PH_ATK.asr_lo, PH_ATK.asr_hi)]})
    write_latex(t44, "tab44_phish_attacks", "Adaptive padding attacks on phishing emails (\\%).", "tab:phishattacks")
if len(RI_KIND):
    t45 = RI_KIND.assign(txt=[f"{100*a:.1f}" for a in RI_KIND.fpr]).pivot(index="kind", columns="detector", values="txt").reset_index()
    write_latex(t45, "tab45_realinstr_by_kind", "False-positive rate (\\%) on legitimate mail with ordinary instructions, by kind.", "tab:rikind")

F52 = []
if len(RO_SEED_TAB):
    F52.append("Read-out variants over seeds (natural pad mean +/- SD, n seeds; worst padding): " + "; ".join(
        f"{r.detector} {100*r.dr_ham_pad_mean:.1f} +/- {100*(r.dr_ham_pad_std if np.isfinite(r.dr_ham_pad_std) else 0):.1f} (n={int(r.dr_ham_pad_count)}); "
        f"{100*r.dr_worst_pad_mean:.1f}" for _, r in RO_SEED_TAB.iterrows()))
for _, r in RO_SEED_CMP.iterrows():
    F52.append(f"{r.variant} vs {r.reference} [{r.set}]: {100*r.dr_diff:+.1f} pts [{100*r.dr_diff_lo:+.1f}, {100*r.dr_diff_hi:+.1f}] (emails), Welch p (Holm) "
               f"{r.p_welch_holm:.3g}, seeds separated: {r.separated}")
for _, r in INJ_TAB.iterrows():
    F52.append(f"Adaptive injection vs {r.target} ({r.kind}): instructions {100*r.asr_instruction:.1f}%, benign control {100*r.asr_benign:.1f}%, "
               f"difference {100*r.asr_diff:+.1f} [{100*r.asr_diff_lo:+.1f}, {100*r.asr_diff_hi:+.1f}], Holm p {r.p_holm:.3g} (n={r.n_caught})")
if len(PH_TAB):
    F52.append("Phishing pool (clean / natural pad / aware injection DR): " + "; ".join(
        f"{r.detector} {100*r.get('dr_phish_clean', np.nan):.1f} / {100*r.get('dr_phish_ham_pad', np.nan):.1f} / {100*r.get('dr_phish_inject_aware', np.nan):.1f}"
        for _, r in PH_TAB.iterrows()))
for _, r in PH_ATK.iterrows():
    F52.append(f"{r.attack} vs {r.target}: {100*r.asr:.1f}% [{100*r.asr_lo:.1f}, {100*r.asr_hi:.1f}] (n={r.n})")
if len(RI_KIND):
    _t = RI_KIND.sort_values("fpr", ascending=False).head(6)
    F52.append("Highest per-kind FPR on ordinary-instruction mail: " + "; ".join(f"{r.detector} [{r.kind[:40]}] {100*r.fpr:.1f}% (n={r.n})" for _, r in _t.iterrows()))
F52.append(f"Measured GPU hours: {GPU_HOURS}")
def _is_v52_job(n):
    return n.startswith(("injatk", "phgreedy", "phlabel")) or bool(re.search(r"_s\d+_(gen|prompt|mean)$|^enc_[a-z]+_s\d+$", n))
_v52_missing = [n for n, s_ in JOB_STATE.items() if s_ in ("deferred", "failed") and _is_v52_job(n)]
if _v52_missing: F52.append(f"NOTE: v5.2 jobs not finished in this session: {_v52_missing}. Attach this output and run again to complete them.")
print("KEY FINDINGS v5.2\n" + "\n".join(F52))
with open(DIRS["results"] / "key_findings.md", "a") as f_:
    f_.write("\n\n## v5.2 second-review analyses\n" + "\n".join(F52))
V52_SUMMARY = {"readout_seeds": RO_SEED_TAB.round(5).to_dict(orient="records"), "readout_seed_tests": RO_SEED_CMP.round(5).to_dict(orient="records"),
               "injection_adaptive": INJ_TAB.round(5).to_dict(orient="records"), "phish_pool": PH_TAB.round(5).to_dict(orient="records"),
               "phish_attacks": PH_ATK.round(5).to_dict(orient="records"), "realinstr_by_kind": RI_KIND.round(5).to_dict(orient="records"),
               "gpu_hours": GPU_HOURS}
