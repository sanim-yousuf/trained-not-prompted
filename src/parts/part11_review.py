# %% [markdown]
# ## 13. Review-response analyses (v5.0)
# Each subsection answers one reviewer concern. Detectors or sets whose jobs were deferred or failed are skipped and listed; nothing here changes the v4.0
# results above. Thresholds are the validation-fixed ones (1% FPR unless stated); intervals are paired bootstrap 95% CIs.

# %%
V5_ALL = [d for d in V5_FILES if d in V5_AVAIL]
for a_ in CFG["target_fprs"]:   # thresholds of the extra LoRA seeds (seed analysis below)
    for t_ in BB_TAGS:
        for x_ in LORA_SEEDS[t_][1:]:
            c_ = f"LoRA-{t_}@s{x_}"
            if f"s_{c_}" in VAL_LEG and VAL_LEG[f"s_{c_}"].notna().all(): THR[a_][c_] = thr_at_fpr(VAL_LEG[f"s_{c_}"].values, a_)
def _has(d, s):
    return s in SC and f"s_{d}" in SC[s] and SC[s][f"s_{d}"].notna().all() and (d in THR[A0])
def _frame(s):
    return attack_frame(s) if (s in ATTACKS or s in OOD_SETS) else SC[s]
def dr_ci(d, s, alpha=None, mask=None):
    if not _has(d, "test_clean" if s == "clean" else s): return (np.nan,) * 3
    df = _frame(s)
    if mask is not None: df = df[mask(df)]
    return boot_mean(flag(df, d, alpha or A0)) if len(df) else (np.nan,) * 3
def avail(dets): return [d for d in dict.fromkeys(dets) if d and d in THR[A0]]
print("v5.0 detectors with scores:", V5_ALL)
print("v5.0 jobs not finished (deferred or failed):", [n for n in JOB_STATE if n not in ("cached",) and JOB_STATE[n] in ("deferred", "failed")])

# %% [markdown]
# ### 13.1 Read-out ablation: trained vs prompted, head vs label word, prompt vs raw email
# The reviewer's point: `LoRA` changes the weights *and* reads the raw email *and* uses a classification head, so "read-out" is three things at once.
# The chain `ZS` -> `LoRA-Gen` (train the label word, same prompt, same read-out) -> `LoRA-Prompt` (same prompt, classification head) -> `LoRA` (raw email)
# changes one factor per step. Each step's effect on the instruction's AUC loss is a difference in differences with a bootstrap CI.

# %%
_xt = BB_TAGS[-1] if len(BB_TAGS) > 2 else None
RO5 = avail([ZS_P, SPOT_DET.get(P_TAG), FS_DET, LARGE_DET, SPOT_DET.get(CFG["prompt_large"]["tag"]), GEN_DET, PROBE_P, PROMPTHEAD_DET, LORA_P,
             MEAN_DET, WIN_DET, ENC_DET] + ([f"ZS-{_xt}", SPOT_DET.get(_xt), f"LoRA-{_xt}"] if _xt else []))
_rows = []
for d in RO5:
    tc_ = SC["test_clean"]; a_ = boot_auc(tc_.label, tc_[f"s_{d}"])
    r = dict(detector=d, auc=a_[0], auc_lo=a_[1], auc_hi=a_[2], tpr=flag(tc_[tc_.label == 1], d).mean())
    for e_, att_ in (("generic", "inject"), ("aware", "inject_aware")):
        r[f"auc_lost_{e_}"], r[f"auc_lost_{e_}_lo"], r[f"auc_lost_{e_}_hi"] = auc_effect(d, "neutral", att_) if _has(d, att_) and _has(d, "neutral") else (np.nan,) * 3
    for s_ in ["inject_aware", "ham_pad"]:
        r[f"dr_{s_}"], r[f"dr_{s_}_lo"], r[f"dr_{s_}_hi"] = dr_ci(d, s_)
    for s_ in ["legit_inject", "legit_realinstr"]:
        r[f"fpr_{s_}"] = flag(SC[s_], d).mean() if _has(d, s_) else np.nan
    _rows.append(r)
RO5_TAB = pd.DataFrame(_rows); RO5_TAB.to_csv(DIRS["results"] / "v5_readout_ablation.csv", index=False)
display(RO5_TAB[["detector", "auc", "tpr", "auc_lost_generic", "auc_lost_aware", "dr_inject_aware", "dr_ham_pad", "fpr_legit_inject", "fpr_legit_realinstr"]].round(4))

def auc_did(d1, d2, att="inject_aware"):
    """(AUC lost by d1) - (AUC lost by d2) to the instruction, paired over the same emails."""
    if not all(_has(d, s) for d in (d1, d2) for s in ("neutral", att)): return (np.nan,) * 3
    A, B = paired("neutral", att)
    e1 = rank_scores(d1, A[f"s_{d1}"].values) - rank_scores(d1, B[f"s_{d1}"].values)
    e2 = rank_scores(d2, A[f"s_{d2}"].values) - rank_scores(d2, B[f"s_{d2}"].values)
    return boot_diff(e1, e2)
_steps = [("train the label word (same prompt, same read-out)", ZS_P, GEN_DET), ("label word -> classification head (same prompt)", GEN_DET, PROMPTHEAD_DET),
          ("prompted input -> raw email (head)", PROMPTHEAD_DET, LORA_P), ("prompted -> trained, all at once", ZS_P, LORA_P),
          ("prompt defense: spotlighting", ZS_P, SPOT_DET.get(P_TAG)), ("scale: 1.5B -> 7B (prompted)", ZS_P, LARGE_DET)]
_rows = []
for lab, d1, d2 in _steps:
    for att in ("inject", "inject_aware"):
        m_, lo_, hi_ = auc_did(d1, d2, att) if d1 in THR[A0] and d2 in THR[A0] else (np.nan,) * 3
        _rows.append(dict(step=lab, from_=d1, to=d2, wording="generic" if att == "inject" else "label-aware", did=m_, lo=lo_, hi=hi_,
                          excludes_zero=bool(np.isfinite(lo_) and (lo_ > 0 or hi_ < 0))))
RO5_DID = pd.DataFrame(_rows); RO5_DID.to_csv(DIRS["results"] / "v5_readout_steps.csv", index=False)
display(RO5_DID.round(4))

# %% [markdown]
# ### 13.2 Detection matrix of the v5.0 detectors (every manipulation, 1% and 5% FPR)

# %%
def matrix_for(dets, alpha):
    rows = []
    for d in dets:
        for att in ATTACKS:
            m_, lo_, hi_ = dr_ci(d, att, alpha)
            if np.isfinite(m_): rows.append(dict(detector=d, attack=att, alpha=alpha, dr=m_, dr_lo=lo_, dr_hi=hi_))
    return pd.DataFrame(rows)
V5_MAT = pd.concat([matrix_for(avail(V5_ALL + [LORA_P, AUGMIX_DET, C_STAR]), a) for a in CFG["target_fprs"]], ignore_index=True)
V5_MAT.to_csv(DIRS["results"] / "v5_detection_matrix.csv", index=False)
if len(V5_MAT):
    display(V5_MAT[V5_MAT.alpha == A0].pivot(index="detector", columns="attack", values="dr").reindex(columns=ATTACKS).round(3))

# %% [markdown]
# ### 13.3 Operating point and score normalization (natural and out-of-distribution padding)
# Natural padding is reported at 1% and 5% FPR and as an AUC change against clean legitimate test mail. A large detection drop with a small AUC change means
# padding shifts scores past a fixed threshold rather than destroying the ranking. Two remedies that need no retraining are tested:
# **length-conditional thresholds** (a separate 1%-FPR threshold per quartile of email length on legitimate validation mail; padded email is longer) and
# **max-over-windows scoring** (`LoRA-Win`: the trained LoRA head scores the whole email and windows of 160 words, and the maximum is used).

# %%
PAD5 = [s_ for s_ in ["ham_pad"] + OOD_SETS if s_ in SC]
OP_DETS = avail([C_STAR, "SVM-char", LORA_P] + [f"LoRA-{t}" for t in BB_TAGS[1:]] + [PROBE_P, AUG_DET, AUGMIX_DET, GEN_DET, PROMPTHEAD_DET, MEAN_DET,
                WIN_DET, ENC_DET, ENC_AUGMIX_DET])
_rows = []
for d in OP_DETS:
    for s_ in PAD5:
        if not _has(d, s_): continue
        r = dict(detector=d, variant=s_)
        for a in CFG["target_fprs"]:
            r[f"dr_clean_{a:g}"] = dr_ci(d, "clean", a)[0]; m_, lo_, hi_ = dr_ci(d, s_, a); r[f"dr_{a:g}"], r[f"dr_{a:g}_lo"], r[f"dr_{a:g}_hi"] = m_, lo_, hi_
        m_, lo_, hi_ = auc_effect(d, "clean", s_)
        r.update(d_auc=m_, d_auc_lo=lo_, d_auc_hi=hi_, auc_clean=float(np.mean(rank_scores(d, CLEAN_POOL[f"s_{d}"].values))))
        r["auc_pad"] = r["auc_clean"] - m_
        _rows.append(r)
OP_TAB = pd.DataFrame(_rows); OP_TAB.to_csv(DIRS["results"] / "v5_operating_point.csv", index=False)
if len(OP_TAB):
    display(OP_TAB[OP_TAB.variant == "ham_pad"][["detector", f"dr_clean_{A0:g}", f"dr_{A0:g}", f"dr_{CFG['target_fprs'][-1]:g}", "auc_clean", "auc_pad", "d_auc"]].round(4))

# Length-conditional thresholds (validation only), compared with the single threshold
def _nw(texts): return np.array([len(str(t).split()) for t in texts])
_vl_w = _nw(VAL_LEG.text.values)
LEN_EDGES = np.quantile(_vl_w, [0.25, 0.5, 0.75])
def _bucket(w): return np.searchsorted(LEN_EDGES, w, side="right")
THR_LEN = {d: [thr_at_fpr(VAL_LEG[f"s_{d}"].values[_bucket(_vl_w) == b], A0) if (_bucket(_vl_w) == b).sum() > 20 else THR[A0][d] for b in range(4)]
           for d in OP_DETS}
def flag_len(df, d):
    b = _bucket(_nw(df.text.values)); return df[f"s_{d}"].values > np.asarray(THR_LEN[d])[b]
_rows = []
_ltest = SC["test_clean"][SC["test_clean"].label == 0]
for d in OP_DETS:
    r = dict(detector=d, fpr_single=flag(_ltest, d).mean(), fpr_len=flag_len(_ltest, d).mean(),
             dr_clean_single=flag(CLEAN_POOL, d).mean(), dr_clean_len=flag_len(CLEAN_POOL, d).mean())
    for s_ in ["ham_pad", "pad_long"]:
        if _has(d, s_): r[f"dr_{s_}_single"], r[f"dr_{s_}_len"] = flag(_frame(s_), d).mean(), flag_len(_frame(s_), d).mean()
    for s_ in ["legit_pad", "legit_pad_ood"]:
        if _has(d, s_): r[f"fpr_{s_}_single"], r[f"fpr_{s_}_len"] = flag(SC[s_], d).mean(), flag_len(SC[s_], d).mean()
    _rows.append(r)
LEN_TAB = pd.DataFrame(_rows); LEN_TAB.to_csv(DIRS["results"] / "v5_length_thresholds.csv", index=False)
display(LEN_TAB.round(4))

# %% [markdown]
# ### 13.4 Seeds: headline numbers as mean and SD
# `LoRA-1.5B`, `LoRA-Aug` and `LoRA-AugMix` have three seeds each (if the v5.0 seed jobs finished); `LoRA-3B` and `LoRA-L3B` two (if time allowed).

# %%
SEED_GROUPS = {LORA_P: [LORA_P] + [f"{LORA_P}@s{x}" for x in LORA_SEEDS[P_TAG][1:]], AUG_DET: [AUG_DET] + [f"{AUG_DET}@s{x}" for x in AUG_SEEDS],
               AUGMIX_DET: [AUGMIX_DET] + [c for c in V5_SEED_AVAIL if c.startswith(AUGMIX_DET + "@")]}
for t in BB_TAGS[1:]:
    SEED_GROUPS[f"LoRA-{t}"] = [f"LoRA-{t}"] + [c for c in V5_SEED_AVAIL if c.startswith(f"LoRA-{t}@")]
_rows = []
for g_, cols in SEED_GROUPS.items():
    cols = [c for c in cols if c in THR[A0]]
    for s_ in ["clean", "ham_pad", "inject_aware", "paraphrase"] + OOD_SETS:
        if s_ not in SC and s_ != "clean": continue
        df_ = _frame(s_) if s_ != "clean" else CLEAN_POOL
        v = [float((df_[f"s_{c}"] > THR[A0][c]).mean()) for c in cols if f"s_{c}" in df_ and df_[f"s_{c}"].notna().all()]
        if v: _rows.append(dict(detector=g_, set=s_, n_seeds=len(v), mean=np.mean(v), sd=np.std(v, ddof=1) if len(v) > 1 else np.nan, min=min(v), max=max(v)))
SEED5 = pd.DataFrame(_rows); SEED5.to_csv(DIRS["results"] / "v5_seeds.csv", index=False)
if len(SEED5): display(SEED5[SEED5.set.isin(["clean", "ham_pad", "pad_prefix", "pad_long"])].round(4))

# %% [markdown]
# ### 13.5 Phishing and fraud vs spam
# The corpus's positive class mixes phishing, fraud and spam. Every key result is split by source: Nazario and Nigerian-Fraud are phishing and advance-fee
# fraud; the other sources are spam-heavy. The instruction effect and the padding loss are recomputed within each subset.

# %%
PHISH_SRC = {"Nazario", "Nigerian-Fraud"}
SUBSETS = {"phishing/fraud": lambda df: df.source.isin(PHISH_SRC).values, "spam-heavy": lambda df: ~df.source.isin(PHISH_SRC).values}
PS_DETS = avail([C_STAR, ZS_P, f"ZS-{_xt}" if _xt else None, LARGE_DET, PROBE_P, LORA_P, f"LoRA-{_xt}" if _xt else None, AUG_DET, AUGMIX_DET,
                 GEN_DET, PROMPTHEAD_DET, ENC_DET])
_rows = []
for d in PS_DETS:
    for sub, mk in SUBSETS.items():
        r = dict(detector=d, subset=sub, n=int(mk(CLEAN_POOL).sum()))
        for s_ in ["clean", "ham_pad", "pad_long", "neutral", "inject_aware", "paraphrase"]:
            r[f"dr_{s_}"], r[f"dr_{s_}_lo"], r[f"dr_{s_}_hi"] = dr_ci(d, s_, mask=mk) if (s_ == "clean" or s_ in SC) else (np.nan,) * 3
        if _has(d, "neutral") and _has(d, "inject_aware"):
            A, B = paired("neutral", "inject_aware"); m = mk(A)
            r["instr_aware_dr"], r["instr_aware_dr_lo"], r["instr_aware_dr_hi"] = boot_diff(flag(A.reset_index()[m], d).astype(float), flag(B.reset_index()[m], d).astype(float))
            r["instr_aware_auc"], r["instr_aware_auc_lo"], r["instr_aware_auc_hi"] = boot_diff(rank_scores(d, A[f"s_{d}"].values[m]), rank_scores(d, B[f"s_{d}"].values[m]))
        tc_ = SC["test_clean"]; keep = (tc_.label == 0).values | ((tc_.label == 1).values & mk(tc_))
        r["auc_clean"] = roc_auc_score(tc_.label[keep], tc_[f"s_{d}"][keep])
        _rows.append(r)
PS_TAB = pd.DataFrame(_rows); PS_TAB.to_csv(DIRS["results"] / "v5_phish_vs_spam.csv", index=False)
display(PS_TAB[["detector", "subset", "n", "auc_clean", "dr_clean", "dr_ham_pad", "dr_inject_aware", "instr_aware_auc"]].round(4))

# %% [markdown]
# ### 13.6 Mechanism: truncation, email length, pooling and position
# The LLM detectors read at most `slm_max_tokens` tokens (head and tail kept). A padded email that exceeds the budget loses part of its middle, so detection is
# split by whether the padded email fits. Pooling: the default head reads the last token, which sits inside suffix padding; `LoRA-Mean` averages all tokens.

# %%
TOK_P = S.load_tok(ALL_BACKBONES[0]["model"])
def _ntok(texts): return np.array([len(TOK_P(t, add_special_tokens=False)["input_ids"]) for t in texts])
_rows = []
for s_ in [x for x in ["ham_pad", "pad_long", "pad_prefix"] if x in SC]:
    df_ = _frame(s_).copy(); df_["ntok"] = _ntok(df_.text.values)
    df_["truncated"] = df_.ntok > CFG["slm_max_tokens"]
    df_["orig_words"] = _nw(CLEAN_POOL.set_index("id").loc[df_.base_id].text.values)
    df_["len_tertile"] = pd.qcut(df_.orig_words, 3, labels=["short", "medium", "long"], duplicates="drop") if df_.orig_words.nunique() > 3 else "all"
    for d in avail([LORA_P, MEAN_DET, WIN_DET, PROMPTHEAD_DET, AUGMIX_DET, C_STAR]):
        if not _has(d, s_): continue
        for key, grp in list(df_.groupby("truncated")) + list(df_.groupby("len_tertile", observed=True)):
            m_, lo_, hi_ = boot_mean(flag(grp, d))
            _rows.append(dict(set=s_, detector=d, split="truncated" if isinstance(key, (bool, np.bool_)) else "original length",
                              group=str(key), n=len(grp), dr=m_, dr_lo=lo_, dr_hi=hi_))
MECH_TAB = pd.DataFrame(_rows); MECH_TAB.to_csv(DIRS["results"] / "v5_mechanism.csv", index=False)
if len(MECH_TAB): display(MECH_TAB.pivot_table(index=["detector", "split", "group"], columns="set", values="dr").round(3))
_trunc_share = {s_: float((_ntok(_frame(s_).text.values) > CFG["slm_max_tokens"]).mean()) for s_ in [x for x in ["ham_pad", "pad_long"] if x in SC]}
print("Share of padded emails longer than the token budget:", {k: round(v, 3) for k, v in _trunc_share.items()})

# %% [markdown]
# ### 13.6b Adaptive attacks on fusion and with label-only feedback (v5.1)
# Two questions the v5.0 attacks left open. **(1) Does fusion help against an adaptive attacker?** Score-guided adversarial emails rarely transferred from
# one detector to another, so the OR of `LR-word` and an LLM detector might hold. The greedy attacker is now run against the fused detector itself
# (it sees the fused score: the larger of the two components' logit margins over their own thresholds), for `Hybrid-OR` (`LR-word` OR `LoRA-1.5B`) and for
# `Hybrid-OR-AugMix` (`LR-word` OR `LoRA-AugMix-1.5B`), both with thresholds FPR-matched on validation. **(2) What if the attacker only learns whether
# the email was flagged?** The label-only attacker adds benign sentences blindly and stops at the first candidate that gets through (same sentences,
# positions and 192-query budget). These GPU jobs run here because they need the fused thresholds, which exist only after section 7.

# %%
HYB_AUGMIX = "Hybrid-OR-AugMix"
DET_COL.setdefault(HYB_AUGMIX, "#0f766e"); DET_MARK.setdefault(HYB_AUGMIX, "P"); DET_FILL.setdefault(HYB_AUGMIX, "#0f766e")
OR_X = {"Hybrid-OR": (C_STAR, S_STAR, OR_THR)}
if AUGMIX_DET in THR[A0]:
    OR_X[HYB_AUGMIX] = (C_STAR, AUGMIX_DET, {a: or_thresholds(VAL_LEG[f"s_{C_STAR}"].values, VAL_LEG[f"s_{AUGMIX_DET}"].values, a)
                                             for a in CFG["target_fprs"]})
def flag_or(df, h, alpha=None):
    c_, s_, T_ = OR_X[h]; _, tc, ts = T_[alpha or A0]
    return ((df[f"s_{c_}"] > tc) | (df[f"s_{s_}"] > ts)).values
_rows = []
for h in OR_X:
    r = dict(detector=h, components=f"{OR_X[h][0]} OR {OR_X[h][1]}", fpr_val=float(flag_or(VAL_LEG, h).mean()),
             fpr_test=float(flag_or(SC["test_clean"][SC["test_clean"].label == 0], h).mean()), dr_clean=float(flag_or(CLEAN_POOL, h).mean()))
    for s_ in ["ham_pad", "inject_aware", "paraphrase"] + OOD_SETS:
        if s_ in SC: r[f"dr_{s_}"] = float(flag_or(_frame(s_), h).mean())
    for s_ in ["legit_inject", "legit_realinstr", "legit_pad", "legit_pad_ood"]:
        if s_ in SC: r[f"fpr_{s_}"] = float(flag_or(SC[s_], h).mean())
    _rows.append(r)
HYB51_TAB = pd.DataFrame(_rows); HYB51_TAB.to_csv(DIRS["results"] / "v51_fused_detectors.csv", index=False)
display(HYB51_TAB.round(4))

ATTACK_FILES.setdefault("label", {})
_V51 = CFG["v5"]["v51"]
_ge_p, _gs_p = str(cpath("sets/greedy_emails.parquet")), str(cpath("sets/greedy_sentences.parquet"))
_pipe_p = cpath(f"classical_{C_STAR}_pipeline.joblib")
if not _pipe_p.exists(): joblib.dump(CLASSICAL[C_STAR], _pipe_p)
def _hspec(h):
    c_, s_, T_ = OR_X[h]; sp, _ = _tspec(s_)
    if sp is None or sp["kind"] != "lora": return None
    _, tc, ts = T_[A0]
    return dict(kind="hybrid_or", lora=sp, classical=str(_pipe_p), thr_c=float(tc), thr_s=float(ts), fixed_threshold=0.0)
QV = JobQueue("v51_attacks"); _nq = 0
def _sub51(mode, det, spec):
    global _nq
    if spec is None: print(f"skip {mode} attack on {det}: no GPU spec"); return
    key = "greedy" if mode == "score" else "label"
    out = cpath(f"{key}_{_slug(det)}.parquet")
    ATTACK_FILES[key][det] = out
    if out.exists(): return
    QV.submit(dict(kind=key, name=f"{key}_{_slug(det)}", target=spec, target_name=det, alpha=A0, seed=SEED + 501, out=str(out), emails=_ge_p,
                   sentences=_gs_p, greedy=CFG["v5"]["greedy"], mode=mode), outputs=[out], est_min=30, optional=True); _nq += 1
if CFG["v5"]["greedy"]["enabled"]:
    if _V51["hybrid_attack"]:
        for h in OR_X: _sub51("score", h, _hspec(h))
    if _V51["label_only"]:
        for d in [LORA_P, AUGMIX_DET]: _sub51("label", d, _tspec(d)[0])
        for h in OR_X: _sub51("label", h, _hspec(h))
    if _nq:
        with Timer("v51_attacks"):
            QV.start().wait(report_every=900)
    _lc = cpath(f"label_{_slug(C_STAR)}.parquet")   # label-only attack on the classical detector (CPU)
    if _V51["label_only"] and not _lc.exists():
        _ge_ = pd.read_parquet(_ge_p); _gs_ = pd.read_parquet(_gs_p).text.tolist()
        _res = S.greedy_pad_attack(lambda t: CLASSICAL[C_STAR].predict_proba(list(t))[:, 1], _ge_.id.tolist(), _ge_.text.tolist(), _gs_,
                                   float(THR[A0][C_STAR]), CFG["v5"]["greedy"], SEED + 501, log_every=0, mode="label")
        _res["threshold"] = float(THR[A0][C_STAR]); _res["target"] = C_STAR; _res.to_parquet(_lc, index=False)
    if _V51["label_only"]: ATTACK_FILES["label"][C_STAR] = _lc
print("v5.1 attack outputs:", {k: sorted(v) for k, v in ATTACK_FILES.items()})

# %% [markdown]
# ### 13.7 Score-guided adaptive attacks
# **Greedy padding (black-box, score access).** For each email the defense catches, each step proposes 24 benign sentences (attacker half of legitimate
# training mail and LLM-written text) at the suffix or after the subject line, keeps the one that lowers the score most, and stops at the threshold or at
# 240 added words (at most 8 steps, 192 queries). The same attack is run on `LR-word` here on CPU. **GCG-style suffix (white-box).** A 20-token suffix of
# plain words optimized with gradients against the classification head; success is counted on token ids and, strictly, after decoding the suffix to text.

# %%
_ge = pd.read_parquet(cpath("sets/greedy_emails.parquet")); _gs = pd.read_parquet(cpath("sets/greedy_sentences.parquet")).text.tolist()
_gc = cpath(f"greedy_{re.sub(r'[^A-Za-z0-9]+', '_', C_STAR)}.parquet")
if CFG["v5"]["greedy"]["enabled"]:
    if not _gc.exists():
        with Timer("greedy_classical"):
            _res = S.greedy_pad_attack(lambda t: CLASSICAL[C_STAR].predict_proba(list(t))[:, 1], _ge.id.tolist(), _ge.text.tolist(), _gs,
                                       float(THR[A0][C_STAR]), CFG["v5"]["greedy"], SEED + 501, log_every=0)
            _res["threshold"] = float(THR[A0][C_STAR]); _res["target"] = C_STAR; _res.to_parquet(_gc, index=False)
    ATTACK_FILES["greedy"][C_STAR] = _gc
GREEDY = {(m_k, d): pd.read_parquet(p) for m_k, key in (("score", "greedy"), ("label", "label")) for d, p in ATTACK_FILES.get(key, {}).items()
          if Path(p).exists()}
GCG = {d: pd.read_parquet(p) for d, p in ATTACK_FILES["gcg"].items() if Path(p).exists()}
_src = _ge.set_index("id").source
Q_GRID = np.arange(0, CFG["v5"]["greedy"]["steps"] * CFG["v5"]["greedy"]["cands"] + 1, max(1, CFG["v5"]["greedy"]["cands"]))
_rows, GREEDY_CURVES = [], {}
for (mode_, d), g in GREEDY.items():
    c = g[g.caught.astype(bool)]
    if not len(c): continue
    m_, lo_, hi_ = boot_mean(c.success.astype(float))
    ph = c.id.map(_src).isin(PHISH_SRC).values
    succ_q = np.where(c.success.values, c.queries.values, np.inf)
    GREEDY_CURVES[(mode_, d)] = [float((succ_q <= q).mean()) for q in Q_GRID]
    tr_ = np.nan
    if mode_ == "score" and d != C_STAR and d not in OR_X and c.success.any():   # transfer: do texts that evade this detector also evade the classical one?
        sv = c[c.success.astype(bool)]
        tr_ = float((CLASSICAL[C_STAR].predict_proba(sv.adv_text.tolist())[:, 1] <= THR[A0][C_STAR]).mean())
    _rows.append(dict(attack="score-guided padding" if mode_ == "score" else "label-only padding", target=d, n=len(c), asr=m_, asr_lo=lo_, asr_hi=hi_,
                      asr_phish=float(c.success[ph].mean()) if ph.any() else np.nan, asr_spam=float(c.success[~ph].mean()) if (~ph).any() else np.nan,
                      median_queries=float(c.queries[c.success.astype(bool)].median()) if c.success.any() else np.nan,
                      median_words=float(c.words_added[c.success.astype(bool)].median()) if c.success.any() else np.nan, transfer_to_classical=tr_))
for d, g in GCG.items():
    c = g[g.caught.astype(bool)] if "caught" in g else g
    if not len(c): continue
    mi = boot_mean(c.success_ids.astype(float)); mt = boot_mean(c.success_text.astype(float))
    _rows.append(dict(attack="GCG suffix", target=d, n=len(c), asr=mt[0], asr_lo=mt[1], asr_hi=mt[2], asr_ids=mi[0], asr_ids_lo=mi[1], asr_ids_hi=mi[2],
                      median_steps=float(c.steps.median())))
ATK5 = pd.DataFrame(_rows); ATK5.to_csv(DIRS["results"] / "v5_score_guided_attacks.csv", index=False)
if len(ATK5): display(ATK5.round(4))
# Human validity sheet: successful greedy adversarial emails (private, never zipped)
_aud = [g[g.success.astype(bool)].assign(target=f"{d} ({m_k})") for (m_k, d), g in GREEDY.items() if "success" in g and g.success.any()]
if _aud:
    _a = pd.concat(_aud).sample(min(30, sum(map(len, _aud))), random_state=SEED)
    _a.assign(rater1_still_malicious_yn="", rater2_still_malicious_yn="")[["id", "target", "words_added", "adv_text", "rater1_still_malicious_yn",
                                                                            "rater2_still_malicious_yn"]].to_csv(DIRS["private"] / "greedy_audit.csv", index=False)
    print(f"Wrote {len(_a)} successful greedy adversarial emails for the human validity check to {DIRS['private'] / 'greedy_audit.csv'}")

# %% [markdown]
# ### 13.8 Collateral false positives on realistic benign mail with ordinary instructions
# Legitimate test emails with an appended paragraph of everyday instructions (no injection template). Paired against the same emails in clean form.

# %%
RI_TAB = pd.DataFrame()
if "legit_realinstr" in SC:
    _clean_leg = SC["test_clean"].set_index("id"); df_ = SC["legit_realinstr"]; c_ = _clean_leg.loc[df_.base_id].reset_index()
    _rows = []
    for d in [x for x in dict.fromkeys(EVAL_DETS + PROMPT_AVAIL + V5_ALL) if _has(x, "legit_realinstr") or x in ("Hybrid-OR",)]:
        f1_, f0_ = flag(df_, d), flag(c_, d); n10, n01, p_ = mcnemar(f1_, f0_); m_, lo_, hi_ = boot_mean(f1_)
        r = dict(detector=d, n=len(df_), fpr_clean=f0_.mean(), fpr=m_, fpr_lo=lo_, fpr_hi=hi_, new_fp=n10, removed_fp=n01, p=p_,
                 fpr_template=flag(SC["legit_inject"], d).mean() if _has(d, "legit_inject") or d == "Hybrid-OR" else np.nan)
        for k_, g_ in df_.assign(_f=f1_).groupby("kind"): r[f"fpr[{k_[:40]}]"] = g_._f.mean()
        _rows.append(r)
    RI_TAB = pd.DataFrame(_rows); RI_TAB["p_holm"] = holm(RI_TAB.p.values)
    RI_TAB.to_csv(DIRS["results"] / "v5_realinstr_collateral.csv", index=False)
    display(RI_TAB[["detector", "fpr_clean", "fpr", "fpr_lo", "fpr_hi", "fpr_template", "p_holm"]].round(4))
    _ra = df_.sample(min(30, len(df_)), random_state=SEED)
    pd.DataFrame({"id": _ra.id, "kind": _ra.kind, "email": _ra.text, "rater1_plausible_legitimate_yn": "", "rater2_plausible_legitimate_yn": ""}).to_csv(
        DIRS["private"] / "realinstr_audit.csv", index=False)
else:
    print("Realistic-instruction set not scored (generation disabled or deferred).")

# %% [markdown]
# ### Fig. 27: Read-out ablation
# (a) AUC lost to the label-aware instruction, net of the neutral suffix. (b) Detection under 80 words of natural padding (o: clean).

# %%
if len(RO5_TAB):
    rows27 = list(RO5_TAB.detector)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 0.24 * len(rows27) + 1.0), sharey=True)
    yy = np.arange(len(rows27))[::-1]
    R = RO5_TAB.set_index("detector")
    for yi, d in zip(yy, rows27):
        r = R.loc[d]; col = DET_COL.get(d, INK2); mk = DET_MARK.get(d, "o")
        if np.isfinite(r.auc_lost_aware):
            axes[0].plot([r.auc_lost_aware_lo, r.auc_lost_aware_hi], [yi, yi], color=col, lw=1.3, alpha=0.7)
            axes[0].plot(r.auc_lost_aware, yi, marker=mk, color=col, ms=5, ls="none")
        if np.isfinite(r.auc_lost_generic):
            axes[0].plot(r.auc_lost_generic, yi, marker=mk, mfc="white", color=col, ms=4.5, ls="none", mew=0.9)
        if np.isfinite(r.dr_ham_pad):
            axes[1].plot([r.dr_ham_pad_lo, r.dr_ham_pad_hi], [yi, yi], color=col, lw=1.3, alpha=0.7)
            axes[1].plot(r.dr_ham_pad, yi, marker=mk, color=col, ms=5, ls="none")
            axes[1].plot(dr_ci(d, "clean")[0], yi, "o", mfc="white", mec=INK2, ms=4.5, mew=0.8)
    axes[0].axvline(0, color=INK2, lw=0.8); axes[0].set_yticks(yy, rows27); axes[0].grid(axis="y", visible=False)
    axes[0].set_xlabel("AUC lost to the instruction (net of neutral suffix)"); panel_label(axes[0], "(a)", x=-0.42)
    axes[0].legend(handles=[Line2D([], [], marker="o", color=INK2, ls="none", label="label-aware"),
                            Line2D([], [], marker="o", mfc="white", color=INK2, ls="none", label="generic")], loc="lower right")
    axes[1].set_xlim(0, 1.03); axes[1].grid(axis="y", visible=False); axes[1].set_xlabel(f"Detection rate, natural pad (FPR = {A0:g})")
    panel_label(axes[1], "(b)", x=-0.05)
    fig.tight_layout()
    savefig(fig, "fig27_readout_ablation", "Read-out ablation on the primary backbone: prompted (ZS, spotlighting, 4-shot, 7B), LoRA trained to generate the label word "
            "(LoRA-Gen), probe, LoRA head on the prompted input (LoRA-Prompt) and on the raw email (LoRA), mean pooling, windowed scoring and a fine-tuned "
            "encoder. (a) AUC lost to the instruction; (b) detection under natural padding (95% CI).")

# %% [markdown]
# ### Fig. 28: Score-guided attacks

# %%
if len(ATK5):
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.1), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for (mode_, d), curve in GREEDY_CURVES.items():
        ax.plot(Q_GRID, curve, color=DET_COL.get(d, INK2), marker=DET_MARK.get(d, "o"), ms=3.2, lw=1.4 if mode_ == "score" else 1.1,
                ls="-" if mode_ == "score" else "--", mfc=None if mode_ == "score" else "white", label=f"{d}{'' if mode_ == 'score' else ' (label-only)'}")
    ax.set_xlabel("Queries to the detector (padding attack)"); ax.set_ylabel("Attack success (share of caught emails)")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=5.5, ncol=3, loc="upper center", bbox_to_anchor=(0.62, -0.2), frameon=False)
    panel_label(ax, "(a)", x=-0.12)
    ax = axes[1]
    gc_ = ATK5[ATK5.attack == "GCG suffix"]
    if len(gc_):
        x = np.arange(len(gc_))
        ax.bar(x - 0.18, gc_.asr_ids, 0.34, color=[DET_COL.get(d, INK2) for d in gc_.target], alpha=0.45, label="token ids")
        ax.bar(x + 0.18, gc_.asr, 0.34, color=[DET_COL.get(d, INK2) for d in gc_.target], label="decoded text")
        ax.errorbar(x + 0.18, gc_.asr, yerr=[gc_.asr - gc_.asr_lo, gc_.asr_hi - gc_.asr], fmt="none", ecolor=INK, lw=0.8, capsize=2)
        ax.set_xticks(x, gc_.target, rotation=20, ha="right"); ax.set_ylim(0, 1.02); ax.set_ylabel("GCG suffix success")
        ax.legend(fontsize=6, loc="upper right")
    else:
        ax.text(0.5, 0.5, "GCG jobs not run", ha="center", va="center", transform=ax.transAxes)
    panel_label(ax, "(b)", x=-0.2); fig.tight_layout()
    savefig(fig, "fig28_score_guided_attacks", "Adaptive attacks on emails each detector catches in clean form: (a) black-box padding attack success by query budget, "
            "solid: score-guided, dashed: label-only feedback, including the fused detectors; (b) white-box GCG-style suffix success on token ids and after decoding "
            "to text (95% CI).")

# %% [markdown]
# ### Fig. 29: Operating point and threshold-free view of natural padding

# %%
_op = OP_TAB[OP_TAB.variant == "ham_pad"].reset_index(drop=True) if len(OP_TAB) else pd.DataFrame()
if len(_op):
    a2 = CFG["target_fprs"][-1]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 0.24 * len(_op) + 1.0), sharey=True)
    yy = np.arange(len(_op))[::-1]
    for yi, (_, r) in zip(yy, _op.iterrows()):
        col = DET_COL.get(r.detector, INK2); mk = DET_MARK.get(r.detector, "o")
        axes[0].plot([r[f"dr_{A0:g}"], r[f"dr_{a2:g}"]], [yi, yi], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
        axes[0].plot(r[f"dr_{A0:g}"], yi, marker=mk, color=col, ms=5, ls="none", zorder=3)
        axes[0].plot(r[f"dr_{a2:g}"], yi, marker=mk, mfc="white", color=col, ms=5, mew=0.9, ls="none", zorder=3)
        axes[1].plot([r.d_auc_lo, r.d_auc_hi], [yi, yi], color=col, lw=1.3, alpha=0.7); axes[1].plot(r.d_auc, yi, marker=mk, color=col, ms=5, ls="none")
    axes[0].set_yticks(yy, _op.detector); axes[0].set_xlim(0, 1.03); axes[0].grid(axis="y", visible=False)
    axes[0].set_xlabel(f"Detection rate under natural pad: FPR {A0:g} (filled), {a2:g} (open)"); panel_label(axes[0], "(a)", x=-0.42)
    axes[1].axvline(0, color=INK2, lw=0.8); axes[1].grid(axis="y", visible=False); axes[1].set_xlabel("AUC lost to natural padding (95% CI)")
    panel_label(axes[1], "(b)", x=-0.05); fig.tight_layout()
    savefig(fig, "fig29_operating_point", "Natural padding at two operating points (a) and on the threshold-free scale (b): a large drop at 1% FPR with a small AUC "
            "change means padding shifts scores past a fixed threshold rather than destroying the ranking.")

# %% [markdown]
# ### Fig. 30: Phishing and fraud vs spam

# %%
if len(PS_TAB):
    rows30 = list(dict.fromkeys(PS_TAB.detector))
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 0.26 * len(rows30) + 1.0), sharey=True)
    yy = np.arange(len(rows30))[::-1]
    for ax, col_, lab in ((axes[0], "dr_ham_pad", "Detection rate, natural pad"), (axes[1], "dr_inject_aware", "Detection rate, label-aware injection")):
        for yi, d in zip(yy, rows30):
            for sub, off, mfc in (("phishing/fraud", 0.12, None), ("spam-heavy", -0.12, "white")):
                r = PS_TAB[(PS_TAB.detector == d) & (PS_TAB.subset == sub)]
                if not len(r) or not np.isfinite(r[col_].iloc[0]): continue
                r = r.iloc[0]; c_ = DET_COL.get(d, INK2)
                ax.plot([r[f"{col_}_lo"], r[f"{col_}_hi"]], [yi + off] * 2, color=c_, lw=1.1, alpha=0.6)
                ax.plot(r[col_], yi + off, marker=DET_MARK.get(d, "o"), color=c_, mfc=mfc or c_, ms=4.6, mew=0.9, ls="none")
        ax.set_xlim(0, 1.03); ax.grid(axis="y", visible=False); ax.set_xlabel(lab)
    axes[0].set_yticks(yy, rows30); panel_label(axes[0], "(a)", x=-0.42); panel_label(axes[1], "(b)", x=-0.05)
    axes[1].legend(handles=[Line2D([], [], marker="o", color=INK2, ls="none", label=f"phishing/fraud (n={int(PS_TAB[PS_TAB.subset == 'phishing/fraud'].n.iloc[0])})"),
                            Line2D([], [], marker="o", mfc="white", color=INK2, ls="none", label="spam-heavy")], loc="lower left", fontsize=6)
    fig.tight_layout()
    savefig(fig, "fig30_phish_vs_spam", "Key results split by source type: phishing and fraud (Nazario, Nigerian-Fraud) vs spam-heavy sources (95% CI).")

# %% [markdown]
# ### Fig. 31: Mechanism: truncation and architecture under padding

# %%
_m31 = OP_TAB[OP_TAB.detector.isin(avail([LORA_P, MEAN_DET, WIN_DET, PROMPTHEAD_DET, GEN_DET, ENC_DET, ENC_AUGMIX_DET, AUGMIX_DET]))] if len(OP_TAB) else pd.DataFrame()
if len(_m31):
    piv = _m31.pivot(index="detector", columns="variant", values=f"dr_{A0:g}").reindex(columns=PAD5)
    piv = piv.loc[[d for d in avail([LORA_P, MEAN_DET, WIN_DET, PROMPTHEAD_DET, GEN_DET, ENC_DET, ENC_AUGMIX_DET, AUGMIX_DET]) if d in piv.index]]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 0.3 * len(piv) + 1.3), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    mt = MECH_TAB[(MECH_TAB.split == "truncated") & (MECH_TAB.set == "ham_pad")] if len(MECH_TAB) else pd.DataFrame()
    for i, d in enumerate([d for d in piv.index if len(mt) and d in set(mt.detector)]):
        for k_, (grp, mfc) in enumerate((("False", None), ("True", "white"))):
            r = mt[(mt.detector == d) & (mt.group == grp)]
            if not len(r): continue
            r = r.iloc[0]; c_ = DET_COL.get(d, INK2)
            ax.errorbar(i + (k_ - 0.5) * 0.3, r.dr, yerr=[[r.dr - r.dr_lo], [r.dr_hi - r.dr]], fmt=DET_MARK.get(d, "o"), color=c_, mfc=mfc or c_, ms=4.5, capsize=2)
    ax.set_xticks(range(len([d for d in piv.index if len(mt) and d in set(mt.detector)])),
                  [d for d in piv.index if len(mt) and d in set(mt.detector)], rotation=30, ha="right")
    ax.set_ylim(0, 1.03); ax.set_ylabel("Detection rate, natural pad"); panel_label(ax, "(a)", x=-0.25)
    ax.legend(handles=[Line2D([], [], marker="o", color=INK2, ls="none", label="fits the token budget"),
                       Line2D([], [], marker="o", mfc="white", color=INK2, ls="none", label="truncated")], fontsize=6, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax = axes[1]
    im = ax.imshow(piv.values, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(PAD5)), [OOD_LABEL.get(c, c) for c in PAD5], rotation=30, ha="right"); ax.set_yticks(range(len(piv)), piv.index); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=f"Detection rate (FPR = {A0:g})"); panel_label(ax, "(b)", x=-0.35)
    fig.tight_layout()
    savefig(fig, "fig31_mechanism", "Mechanism of the padding failure: (a) detection of naturally padded emails that fit the token budget vs those truncated; "
            "(b) pooling, input format, windowed scoring, generative training and a fine-tuned encoder under every padding variant.")

# %% [markdown]
# ### Fig. 32: Collateral false positives on realistic benign mail with instructions

# %%
if len(RI_TAB):
    _key32 = avail([C_STAR, "SVM-char"] + SLM_ALL + [AUG_DET, AUGMIX_DET, GEN_DET, PROMPTHEAD_DET, MEAN_DET, ENC_DET, ENC_AUGMIX_DET,
                    SPOT_DET.get(P_TAG), LARGE_DET, "Hybrid-Stack"]) + ["Hybrid-OR"]
    rows32 = [d for d in RI_TAB.detector if d in DET_COL and d in _key32]
    fig, ax = plt.subplots(figsize=(COL1 * 1.35, 0.22 * len(rows32) + 1.0))
    yy = np.arange(len(rows32))[::-1]; R = RI_TAB.set_index("detector")
    for yi, d in zip(yy, rows32):
        r = R.loc[d]; c_ = DET_COL[d]
        ax.plot(floor_(r.fpr_clean), yi, "o", mfc="white", mec=INK2, ms=4.5, mew=0.8, zorder=3)
        if np.isfinite(r.fpr_template): ax.plot(floor_(r.fpr_template), yi - 0.14, marker="x", color=c_, ms=4, ls="none", zorder=3)
        ax.plot([floor_(r.fpr_lo), floor_(r.fpr_hi)], [yi + 0.12] * 2, color=c_, lw=1.0, alpha=0.6)
        ax.plot(floor_(r.fpr), yi + 0.12, marker=DET_MARK.get(d, "D"), color=c_, ms=4.2, ls="none", zorder=4)
        if r.p_holm < 0.05: ax.text(min(0.95, floor_(r.fpr_hi) * 1.3), yi + 0.12, stars(r.p_holm), fontsize=6, va="center")
    ax.set_xscale("log"); ax.set_xlim(FLOOR * 0.8, 1); ax.axvline(A0, color=INK2, lw=0.8, ls=":")
    ax.set_yticks(yy, rows32); ax.grid(axis="y", visible=False); ax.set_xlabel("False-positive rate on legitimate mail (log scale)")
    ax.legend(handles=[Line2D([], [], marker="o", mfc="white", mec=INK2, ls="none", label="clean"),
                       Line2D([], [], marker="D", color=INK2, ls="none", label="+ ordinary instructions"),
                       Line2D([], [], marker="x", color=INK2, ls="none", label="+ injection template")], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6)
    fig.tight_layout()
    savefig(fig, "fig32_realinstr_collateral", "False positives on legitimate mail that carries ordinary instructions (forwarded assistant answers, IT notices, task lists) "
            "compared with the same mail clean and with an injection template (95% CI; stars: Holm-corrected McNemar vs clean).")

# %% [markdown]
# ### 13.9 Tables for the paper (v5.0)

# %%
pct = lambda v: "n/a" if not np.isfinite(v) else f"{100 * v:.1f}"
def _col(df, c): return df[c] if c in df else pd.Series(np.nan, index=df.index)
ci3 = lambda m, lo, hi, k=3: "n/a" if not np.isfinite(m) else f"{m:.{k}f} [{lo:.{k}f}, {hi:.{k}f}]"
if len(RO5_TAB):
    t31 = pd.DataFrame({"Detector": RO5_TAB.detector, "Clean AUC": RO5_TAB.auc.map(lambda v: f"{v:.4f}"), "TPR@1%": RO5_TAB.tpr.map(pct),
                        "AUC lost (generic)": [ci3(a, b, c) for a, b, c in zip(RO5_TAB.auc_lost_generic, RO5_TAB.auc_lost_generic_lo, RO5_TAB.auc_lost_generic_hi)],
                        "AUC lost (aware)": [ci3(a, b, c) for a, b, c in zip(RO5_TAB.auc_lost_aware, RO5_TAB.auc_lost_aware_lo, RO5_TAB.auc_lost_aware_hi)],
                        "DR natural pad": RO5_TAB.dr_ham_pad.map(pct), "FPR +template": RO5_TAB.fpr_legit_inject.map(pct),
                        "FPR +instructions": RO5_TAB.fpr_legit_realinstr.map(pct)})
    print(write_latex(t31, "tab31_readout_ablation", "Read-out ablation on the primary backbone (detection and FPR in \\%, at 1\\% validation FPR).", "tab:ablation"))
if len(RO5_DID):
    t31b = RO5_DID.assign(**{"Step": RO5_DID.step, "From": RO5_DID.from_, "To": RO5_DID.to, "Wording": RO5_DID.wording,
                             "Change in AUC lost [95% CI]": [ci3(a, b, c) for a, b, c in zip(RO5_DID.did, RO5_DID.lo, RO5_DID.hi)]})[
        ["Step", "From", "To", "Wording", "Change in AUC lost [95% CI]"]]
    write_latex(t31b, "tab31b_readout_steps", "One factor at a time: change in the AUC lost to the instruction (positive: the first detector loses more).", "tab:steps")
if len(V5_MAT):
    t32 = V5_MAT[V5_MAT.alpha == A0].assign(txt=lambda d: [f"{100*a:.1f} ({100*b:.1f}-{100*c:.1f})" for a, b, c in zip(d.dr, d.dr_lo, d.dr_hi)]).pivot(
        index="detector", columns="attack", values="txt").reindex(columns=ATTACKS).rename(columns=ATT_LABEL).reset_index().rename(columns={"detector": "Detector"})
    write_latex(t32, "tab32_v5_detection_matrix", "Detection rate (\\%, 95\\% CI) of the v5.0 detectors at 1\\% FPR.", "tab:v5matrix")
if len(OP_TAB):
    a2 = CFG["target_fprs"][-1]; _o = OP_TAB[OP_TAB.variant == "ham_pad"]
    t33 = pd.DataFrame({"Detector": _o.detector, f"DR clean @{A0:g}": _o[f"dr_clean_{A0:g}"].map(pct), f"DR pad @{A0:g}": _o[f"dr_{A0:g}"].map(pct),
                        f"DR pad @{a2:g}": _o[f"dr_{a2:g}"].map(pct), "AUC clean": _o.auc_clean.map(lambda v: f"{v:.3f}"),
                        "AUC pad": _o.auc_pad.map(lambda v: f"{v:.3f}"), "AUC lost [95% CI]": [ci3(a, b, c) for a, b, c in zip(_o.d_auc, _o.d_auc_lo, _o.d_auc_hi)]})
    write_latex(t33, "tab33_operating_point", "Natural padding at two operating points and on the threshold-free scale.", "tab:oppoint")
if len(LEN_TAB):
    _c = [c for c in LEN_TAB.columns if c != "detector"]
    write_latex(LEN_TAB.assign(**{c: LEN_TAB[c].map(pct) for c in _c}), "tab34_length_thresholds",
                "Single validation threshold vs length-conditional thresholds (\\%): FPR on clean legitimate test mail, detection of clean and padded "
                "malicious mail, and FPR on padded legitimate mail.", "tab:lenthr")
if len(SEED5):
    t35 = SEED5.assign(**{"Mean (SD)": [f"{100*m:.1f} ({100*s:.1f})" if np.isfinite(s) else f"{100*m:.1f}" for m, s in zip(SEED5["mean"], SEED5.sd)],
                          "Range": [f"{100*a:.1f} to {100*b:.1f}" for a, b in zip(SEED5["min"], SEED5["max"])]})[["detector", "set", "n_seeds", "Mean (SD)", "Range"]]
    write_latex(t35.rename(columns={"detector": "Detector", "set": "Set", "n_seeds": "Seeds"}), "tab35_seeds_v5", "Detection rate (\\%) over seeds.", "tab:seeds5")
if len(PS_TAB):
    t36 = pd.DataFrame({"Detector": PS_TAB.detector, "Subset": PS_TAB.subset, "n": PS_TAB.n, "AUC": PS_TAB.auc_clean.map(lambda v: f"{v:.4f}"),
                        "DR clean": PS_TAB.dr_clean.map(pct), "DR natural pad": PS_TAB.dr_ham_pad.map(pct), "DR 240 words": PS_TAB.dr_pad_long.map(pct),
                        "DR aware inj.": PS_TAB.dr_inject_aware.map(pct),
                        "AUC lost (aware)": [ci3(a, b, c) for a, b, c in zip(_col(PS_TAB, "instr_aware_auc"), _col(PS_TAB, "instr_aware_auc_lo"),
                                                                              _col(PS_TAB, "instr_aware_auc_hi"))]})
    write_latex(t36, "tab36_phish_vs_spam", "Key results by source type (detection in \\%).", "tab:phishspam")
if len(ATK5):
    t37 = pd.DataFrame({"Attack": ATK5.attack, "Target": ATK5.target, "Emails": ATK5.n,
                        "Success [95% CI]": [f"{100*a:.1f} [{100*b:.1f}, {100*c:.1f}]" for a, b, c in zip(ATK5.asr, ATK5.asr_lo, ATK5.asr_hi)],
                        "Token-level": _col(ATK5, "asr_ids").map(pct),
                        "Median queries": _col(ATK5, "median_queries").map(lambda v: "n/a" if not np.isfinite(v) else f"{v:.0f}"),
                        "Median words": _col(ATK5, "median_words").map(lambda v: "n/a" if not np.isfinite(v) else f"{v:.0f}"),
                        "Transfer to classical": _col(ATK5, "transfer_to_classical").map(pct)})
    write_latex(t37, "tab37_score_guided", "Score-guided attacks on emails each detector catches in clean form (\\%).", "tab:scoreguided")
if len(HYB51_TAB):
    _c = [c for c in HYB51_TAB.columns if c.startswith(("dr_", "fpr_"))]
    write_latex(HYB51_TAB.assign(**{c: HYB51_TAB[c].map(pct) for c in _c}), "tab40_fused_detectors",
                "FPR-matched OR of the classical detector with LoRA or LoRA-AugMix: detection and false-positive rates (\\%).", "tab:fused")
if len(RI_TAB):
    t38 = pd.DataFrame({"Detector": RI_TAB.detector, "Clean": RI_TAB.fpr_clean.map(pct),
                        "+ ordinary instructions": [f"{100*a:.1f} [{100*b:.1f}, {100*c:.1f}]{stars(q)}" for a, b, c, q in zip(RI_TAB.fpr, RI_TAB.fpr_lo, RI_TAB.fpr_hi, RI_TAB.p_holm)],
                        "+ injection template": RI_TAB.fpr_template.map(pct)})
    write_latex(t38, "tab38_realinstr", "False-positive rate (\\%) on legitimate mail with ordinary instructions vs an injection template.", "tab:realinstr")
if len(MECH_TAB):
    t39 = MECH_TAB.assign(DR=[f"{100*a:.1f} [{100*b:.1f}, {100*c:.1f}]" for a, b, c in zip(MECH_TAB.dr, MECH_TAB.dr_lo, MECH_TAB.dr_hi)])[
        ["set", "detector", "split", "group", "n", "DR"]]
    write_latex(t39, "tab39_mechanism", "Detection (\\%) of padded emails by truncation status and original length.", "tab:mechanism")

# %% [markdown]
# ### 13.10 Key findings (v5.0, auto-generated; verify against the tables and CIs)

# %%
F5 = []
_R = RO5_TAB.set_index("detector") if len(RO5_TAB) else pd.DataFrame()
def _rv(d, c): return float(_R.loc[d, c]) if len(_R) and d in _R.index and c in _R.columns else np.nan
if len(_R):
    F5.append("Read-out ablation (AUC lost to the label-aware instruction; DR natural pad): " + "; ".join(
        f"{d} {_rv(d, 'auc_lost_aware'):+.3f}, {100 * _rv(d, 'dr_ham_pad'):.1f}%" for d in _R.index))
for _, r in RO5_DID.iterrows():
    if np.isfinite(r.did) and r.wording == "label-aware":
        F5.append(f"Step '{r.step}' ({r.from_} -> {r.to}): change in AUC lost {r.did:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]{' (CI excludes 0)' if r.excludes_zero else ''}.")
if len(OP_TAB):
    _o = OP_TAB[OP_TAB.variant == "ham_pad"].set_index("detector")
    F5.append("Natural padding by operating point (DR at 1% / 5% FPR; AUC lost): " + "; ".join(
        f"{d} {100*_o.loc[d, f'dr_{A0:g}']:.1f}% / {100*_o.loc[d, 'dr_' + format(CFG['target_fprs'][-1], 'g')]:.1f}%; {_o.loc[d, 'd_auc']:+.3f}" for d in _o.index))
if len(LEN_TAB):
    F5.append("Length-conditional thresholds (natural pad DR single -> length-conditional): " + "; ".join(
        f"{r.detector} {100*r.get('dr_ham_pad_single', np.nan):.1f} -> {100*r.get('dr_ham_pad_len', np.nan):.1f} (clean legit FPR {100*r.fpr_len:.1f})"
        for _, r in LEN_TAB.iterrows()))
if len(SEED5):
    _s = SEED5[SEED5.set == "ham_pad"]
    F5.append("Natural pad over seeds (mean, SD): " + "; ".join(f"{r.detector} {100*r['mean']:.1f}% ({100*r.sd:.1f}, n={r.n_seeds})" for _, r in _s.iterrows()))
if len(PS_TAB):
    F5.append("Phishing/fraud vs spam-heavy (DR natural pad; DR aware injection): " + "; ".join(
        f"{r.detector} [{r.subset}] {100*r.dr_ham_pad:.1f}%, {100*r.dr_inject_aware:.1f}%" for _, r in PS_TAB.iterrows()))
if len(ATK5):
    F5.append("Adaptive padding and suffix attacks (success on caught emails): " + "; ".join(
        f"{r.attack} vs {r.target} {100*r.asr:.1f}% [{100*r.asr_lo:.1f}, {100*r.asr_hi:.1f}] (n={r.n})" for _, r in ATK5.iterrows()))
if len(HYB51_TAB):
    F5.append("Fused detectors (non-adaptive): " + "; ".join(f"{r.detector} natural pad {100*r.get('dr_ham_pad', np.nan):.1f}%, test FPR {100*r.fpr_test:.2f}%, "
                                                           f"FPR with ordinary instructions {100*r.get('fpr_legit_realinstr', np.nan):.1f}%" for _, r in HYB51_TAB.iterrows()))
if len(RI_TAB):
    _t = RI_TAB.sort_values("fpr", ascending=False).head(6)
    F5.append("FPR on legitimate mail with ordinary instructions (highest): " + "; ".join(
        f"{r.detector} {100*r.fpr:.1f}% (clean {100*r.fpr_clean:.1f}%, template {100*r.fpr_template:.1f}%)" for _, r in _t.iterrows()))
if len(MECH_TAB):
    _m = MECH_TAB[(MECH_TAB.split == "truncated") & (MECH_TAB.set == "ham_pad")]
    F5.append(f"Truncation (share of padded emails over the {CFG['slm_max_tokens']}-token budget {_trunc_share}); natural-pad DR fits vs truncated: " + "; ".join(
        f"{d} {100*g[g.group == 'False'].dr.mean():.1f}% vs {100*g[g.group == 'True'].dr.mean():.1f}%" for d, g in _m.groupby("detector")))
_v5_missing = [n for n, s_ in JOB_STATE.items() if s_ in ("deferred", "failed")]
if _v5_missing:
    F5.append(f"NOTE: jobs not finished in this session: {_v5_missing}. Attach this output and run again to complete them.")
print("KEY FINDINGS v5.0\n" + "\n".join(F5))
with open(DIRS["results"] / "key_findings.md", "a") as f_:
    f_.write("\n\n## v5.0 review-response analyses\n" + "\n".join(F5))
V5_SUMMARY = {"readout_ablation": RO5_TAB.round(5).to_dict(orient="records"), "readout_steps": RO5_DID.round(5).to_dict(orient="records"),
              "operating_point": OP_TAB.round(5).to_dict(orient="records"), "length_thresholds": LEN_TAB.round(5).to_dict(orient="records"),
              "seeds": SEED5.round(5).to_dict(orient="records"), "phish_vs_spam": PS_TAB.round(5).to_dict(orient="records"),
              "score_guided": ATK5.round(5).to_dict(orient="records"), "realinstr": RI_TAB.round(5).to_dict(orient="records"),
              "mechanism": MECH_TAB.round(5).to_dict(orient="records"), "truncated_share": _trunc_share, "latency_ms": {k: round(v * 1e3, 3) for k, v in V5_LAT.items()},
              "v5_available": V5_ALL, "v5_seed_columns": V5_SEED_AVAIL, "fused_detectors": HYB51_TAB.round(5).to_dict(orient="records")}
