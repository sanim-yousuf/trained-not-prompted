# %% [markdown]
# ## 8. Metrics and statistics
#
# * **Detection rate (DR)** on malicious emails at the validation-fixed threshold. **Attack success rate (ASR)** is conditional: among emails a detector flags in clean
#   form, the share it misses after the manipulation (paired by `base_id`).
# * **Instruction effect** $\Delta_{\text{instr}} = \text{DR}(\text{neutral}) - \text{DR}(\text{inject})$ on the *same* emails with the same family and position: the drop caused by the
#   instruction itself, net of appending text. **Padding artifact check**: $\text{DR}(\text{natural pad}) - \text{DR}(\text{good-word})$.
# * 95% CIs from `n_boot` paired bootstrap resamples of emails; exact McNemar tests on paired outcomes with Holm correction within each family of tests.
# * **Yule's Q** on per-email correctness of two detectors, with bootstrap CI (only when every cell of the 2x2 table holds at least 5 emails); lower means more
#   complementary errors. Raw discordant counts are always reported.
# * **Threshold-free effects** (v3.2): each malicious email gets a rank score $r(x)=\Pr_{\ell}[s(\ell)<s(x)]$ against the clean legitimate test emails, so its mean is the
#   ROC-AUC. Differences of mean rank scores are differences in AUC, free of the operating point and of how strong the detector is on clean mail.

# %%
RNG = np.random.default_rng(SEED)
NB_ = CFG["n_boot"]

def boot_mean(x):
    x = np.asarray(x, float)
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    m = x[RNG.integers(0, len(x), (NB_, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def boot_auc(y, s):
    y, s = np.asarray(y), np.asarray(s); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                          np.r_[s[RNG.choice(pos, len(pos))], s[RNG.choice(neg, len(neg))]]) for _ in range(NB_)]
    return float(roc_auc_score(y, s)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def ece(y, p, bins=CFG["ece_bins"]):
    """Equal-mass binning (each bin holds the same number of emails), robust to scores piled up near 0 and 1."""
    p = np.clip(np.asarray(p, float), 0, 1); y = np.asarray(y)
    order = np.argsort(p, kind="stable"); e = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk): e += len(chunk) / len(p) * abs(y[chunk].mean() - p[chunk].mean())
    return float(e)

def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return n10, n01, float(p)

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj

def yules_q(ca, cb):
    ca, cb = np.asarray(ca, bool), np.asarray(cb, bool)
    n11, n00, n10, n01 = (ca & cb).sum(), (~ca & ~cb).sum(), (ca & ~cb).sum(), (~ca & cb).sum()
    den = n11 * n00 + n10 * n01
    return float((n11 * n00 - n10 * n01) / den) if den else np.nan

def boot_q(ca, cb, min_cell=5):
    """Yule's Q with a bootstrap CI. With fewer than min_cell emails in any cell of the 2x2 table, Q is pinned near +/-1 by a handful of emails and
    its bootstrap CI is meaningless, so only the point value is returned (reported as 'few errors')."""
    ca, cb = np.asarray(ca, bool), np.asarray(cb, bool); n = len(ca)
    cells = [(ca & cb).sum(), (~ca & ~cb).sum(), (ca & ~cb).sum(), (~ca & cb).sum()]
    if min(cells) < min_cell: return (yules_q(ca, cb), np.nan, np.nan)
    qs = [yules_q(ca[i], cb[i]) for i in RNG.integers(0, n, (NB_, n))]
    qs = np.array([q for q in qs if np.isfinite(q)])
    q0 = yules_q(ca, cb)
    return (q0, float(np.percentile(qs, 2.5)), float(np.percentile(qs, 97.5))) if len(qs) > NB_ // 2 else (q0, np.nan, np.nan)

def boot_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); n = len(a)
    idx = RNG.integers(0, n, (NB_, n)); d = (a[idx] - b[idx]).mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

POOL_IDS = set(POOL.id)
CLEAN_POOL = SC["test_clean"][SC["test_clean"].id.isin(POOL_IDS)].set_index("id").loc[POOL.id].reset_index()
CLEAN_POOL["attack"] = "clean"

def attack_frame(att):
    if att == "clean": return CLEAN_POOL
    df = SC[att]; return df[df["valid"].astype(bool)]

def paired(att_a, att_b):
    """Rows of two attack sets aligned on base_id (both valid)."""
    A, B = attack_frame(att_a).set_index("base_id"), attack_frame(att_b).set_index("base_id")
    ids = A.index.intersection(B.index)
    return A.loc[ids], B.loc[ids]

# %% [markdown]
# ### 8.1 Clean performance (RQ1)

# %%
def clean_table(alpha):
    tc = SC["test_clean"]; y = tc.label.values; rows = []
    for d in EVAL_DETS:
        auc = boot_auc(y, tc[f"s_{d}"]); f = flag(tc, d, alpha)
        tpr = boot_mean(f[y == 1]); fpr = boot_mean(f[y == 0])
        rows.append(dict(detector=d, family=FAMILY[d], auc=auc[0], auc_lo=auc[1], auc_hi=auc[2],
                         ap=average_precision_score(y, tc[f"s_{d}"]), tpr=tpr[0], tpr_lo=tpr[1], tpr_hi=tpr[2],
                         fpr=fpr[0], fpr_lo=fpr[1], fpr_hi=fpr[2], f1=f1_score(y, f), ece=ece(y, tc[f"s_{d}"]),
                         brier=brier_score_loss(y, np.clip(tc[f"s_{d}"], 0, 1))))
    return pd.DataFrame(rows)

CLEAN = {a: clean_table(a) for a in CFG["target_fprs"]}
display(CLEAN[A0].round(4))

# %% [markdown]
# ### 8.2 Detection and attack success under every manipulation (RQ2, RQ3)

# %%
def attack_matrix(alpha):
    rows = []
    for d in EVAL_DETS:
        clean_flag = dict(zip(CLEAN_POOL.id, flag(CLEAN_POOL, d, alpha)))
        for att in ATTACKS:
            df = attack_frame(att); f = flag(df, d, alpha); dr_ = boot_mean(f)
            base_det = np.array([clean_flag.get(b, False) for b in df.base_id])
            asr = boot_mean(~f[base_det]) if att != "clean" else (0.0, 0.0, 0.0)
            rows.append(dict(detector=d, attack=att, n=len(df), dr=dr_[0], dr_lo=dr_[1], dr_hi=dr_[2],
                             asr=asr[0], asr_lo=asr[1], asr_hi=asr[2], n_asr=int(base_det.sum()), n_asr_succ=int((~f[base_det]).sum())))
    return pd.DataFrame(rows)

MAT = {a: attack_matrix(a) for a in CFG["target_fprs"]}
M0 = MAT[A0]
display(M0.pivot(index="detector", columns="attack", values="dr").loc[EVAL_DETS, ATTACKS].round(3))

# Paired attack-vs-clean significance for every cell (Holm over all cells)
_pv, _cells = [], []
for d in EVAL_DETS:
    for att in ATTACKS[1:]:
        A, B = paired("clean", att) if att != "clean" else (None, None)
        _pv.append(mcnemar(flag(A.reset_index(), d), flag(B.reset_index(), d))[2]); _cells.append((d, att))
SIG = pd.DataFrame([(d, a, p, q) for (d, a), p, q in zip(_cells, _pv, holm(_pv))], columns=["detector", "attack", "p", "p_holm"])
SIG.to_csv(DIRS["results"] / "attack_vs_clean_mcnemar.csv", index=False)

# %% [markdown]
# ### 8.3 Read-out analysis: instruction effect and padding artifact (RQ2, RQ3)

# %%
def effect_rows(alpha, a_ctrl, a_att, label):
    A, B = paired(a_ctrl, a_att); rows = []
    for d in EVAL_DETS:
        fa, fb = flag(A.reset_index(), d, alpha), flag(B.reset_index(), d, alpha)
        n10, n01, p = mcnemar(fa, fb); dm, lo, hi = boot_diff(fa, fb)
        rows.append(dict(effect=label, detector=d, control=a_ctrl, attack=a_att, n=len(A), dr_control=fa.mean(), dr_attack=fb.mean(),
                         delta=dm, delta_lo=lo, delta_hi=hi, ctrl_only=n10, att_only=n01, p=p))
    return rows

def readout_effects(alpha):
    rows = (effect_rows(alpha, "neutral", "inject", "instruction") + effect_rows(alpha, "neutral", "inject_aware", "instruction (aware)")
            + effect_rows(alpha, "ham_pad", "goodword", "word salad vs natural pad") + effect_rows(alpha, "clean", "neutral", "neutral suffix")
            + effect_rows(alpha, "clean", "ham_pad", "natural padding"))
    t = pd.DataFrame(rows); t["p_holm"] = np.nan
    for e in t.effect.unique():
        m = t.effect == e; t.loc[m, "p_holm"] = holm(t.loc[m, "p"].values)
    return t

EFF = readout_effects(A0)
display(EFF[EFF.effect.isin(["instruction", "word salad vs natural pad"])][["effect", "detector", "dr_control", "dr_attack", "delta", "delta_lo", "delta_hi", "p_holm"]].round(4))

# %% [markdown]
# ### 8.4 Complementarity of C* and S*, and key paired comparisons (RQ2, RQ4)

# %%
def complementarity(alpha, a_det, b_det):
    rows = []
    for att in ATTACKS:
        df = attack_frame(att); ca, cb = flag(df, a_det, alpha), flag(df, b_det, alpha); q = boot_q(ca, cb)
        rows.append(dict(attack=att, a=a_det, b=b_det, both_catch=np.mean(ca & cb), only_a_miss=np.mean(~ca & cb), only_b_miss=np.mean(ca & ~cb),
                         both_miss=np.mean(~ca & ~cb), n_only_a_miss=int((~ca & cb).sum()), n_only_b_miss=int((ca & ~cb).sum()),
                         n_both_miss=int((~ca & ~cb).sum()), yules_q=q[0], q_lo=q[1], q_hi=q[2], n=len(df)))
    return pd.DataFrame(rows)

COMP = complementarity(A0, C_STAR, S_STAR)
COMP_ALL = pd.concat([complementarity(A0, c, s) for c in CLS_DETS for s in [ZS_P, PROBE_P, LORA_P]], ignore_index=True)
display(COMP.round(3))

def mcnemar_table(alpha):
    pairs = [(C_STAR, S_STAR), ("Hybrid-OR", S_STAR), ("Hybrid-OR", C_STAR), ("Hybrid-Stack", S_STAR), ("Hybrid-OR", "Hybrid-Stack"),
             ("Stack-Aug", "Hybrid-Stack"), ("Hybrid-Stack", "Stack-noInj")]
    rows = []
    for att in ATTACKS:
        df = attack_frame(att)
        for a, b in pairs:
            fa, fb = flag(df, a, alpha), flag(df, b, alpha); n10, n01, p = mcnemar(fa, fb)
            rows.append(dict(attack=att, A=a, B=b, dr_A=fa.mean(), dr_B=fb.mean(), A_only=n10, B_only=n01, p=p))
    t = pd.DataFrame(rows); t["p_holm"] = holm(t.p.values); return t

MCN = mcnemar_table(A0)
display(MCN[MCN.p_holm < 0.05].round(4))

# %% [markdown]
# ### 8.5 Triage (disagreement-gated review)

# %%
def triage_table(alpha):
    rows = []
    for att in ATTACKS:
        t = triage(attack_frame(att), alpha)
        rows.append(dict(group=ATT_LABEL[att], block=np.mean(t == "block"), review=np.mean(t == "review"), allow=np.mean(t == "allow")))
    t = triage(SC["test_clean"].query("label==0"), alpha)
    rows.append(dict(group="Legitimate (clean)", block=np.mean(t == "block"), review=np.mean(t == "review"), allow=np.mean(t == "allow")))
    return pd.DataFrame(rows)
TRI = triage_table(A0)
display(TRI.round(3))

# %% [markdown]
# ### 8.6 Adaptive attacker under a query budget (RQ4)
# For an email with $m$ successful candidates out of $N$, a random-order attacker succeeds within $k$ queries with probability $1-\binom{N-m}{k}/\binom{N}{k}$;
# ASR@k averages this over emails the defense catches in clean form (exact, no sampling of query orders).

# %%
ADV = SC["adaptive"]
DEFENSES = list(dict.fromkeys([C_STAR, S_STAR] + ADAPT_DEFS_EXTRA + ["Hybrid-OR", "Hybrid-Stack", "Stack-Aug", "Triage"]))

def adaptive_curves(alpha):
    out, comp_ = {}, []
    clean_ad = SC["test_clean"].set_index("id").loc[ADAPT.id].reset_index()
    for dname in DEFENSES:
        if dname == "Triage":
            succ = triage(ADV, alpha) == "allow"; caught0 = triage(clean_ad, alpha) != "allow"
        else:
            succ = ~flag(ADV, dname, alpha); caught0 = flag(clean_ad, dname, alpha)
        g = pd.DataFrame({"base_id": ADV.base_id, "succ": succ}).groupby("base_id", sort=False)["succ"]
        m = g.sum().reindex(ADAPT.id).values[caught0]; N = g.size().reindex(ADAPT.id).values[caught0]
        ks = np.arange(1, N_CAND + 1)
        P = np.array([[1 - comb(n - mi, k) / comb(n, k) if k <= n else float(mi > 0) for k in ks] for mi, n in zip(m, N)]) if len(m) else np.zeros((0, len(ks)))
        if len(P):
            boots = P[RNG.integers(0, len(P), (NB_, len(P)))].mean(1)
            out[dname] = dict(k=ks, mean=P.mean(0), lo=np.percentile(boots, 2.5, 0), hi=np.percentile(boots, 97.5, 0), n=len(P))
        for (arm, b, fam_, pad), grp in ADV.assign(succ=succ).groupby(["arm", "base", "family", "pad"]):
            comp_.append(dict(defense=dname, arm=arm, base=b, family=fam_, pad=pad, success_rate=grp.succ.mean()))
        # success within each arm alone (all of that arm's candidates tried), over emails caught in clean form
        for arm in ADV.arm.unique():
            ga = pd.DataFrame({"base_id": ADV.base_id, "succ": succ})[(ADV.arm == arm).values].groupby("base_id")["succ"].any()
            v = ga.reindex(ADAPT.id).fillna(False).values[caught0].astype(float)
            m_, lo_, hi_ = boot_mean(v)
            ARM_ROWS.append(dict(defense=dname, arm=arm, n_cand=int((ADV.arm == arm).sum() // max(1, ADV.base_id.nunique())), asr=m_, lo=lo_, hi=hi_))
    return out, pd.DataFrame(comp_)

ARM_ROWS = []

ADAPT_CURVES, ADAPT_COMP = adaptive_curves(A0)
ADAPT_ARM = pd.DataFrame(ARM_ROWS); ADAPT_ARM.to_csv(DIRS["results"] / "adaptive_by_arm.csv", index=False)
display(ADAPT_ARM.pivot(index="defense", columns="arm", values="asr").round(3))
for d, c in ADAPT_CURVES.items():
    print(f"  {d:<13} n={c['n']:>4}  ASR@1={c['mean'][0]:.3f}  ASR@5={c['mean'][min(4, len(c['mean'])-1)]:.3f}  ASR@{N_CAND}={c['mean'][-1]:.3f}")

# %% [markdown]
# ### 8.7 Leave-one-source-out, including LoRA, and the held-out-source attack matrix (RQ1, RQ5)
# Classical models come from Section 6.1. `ZS` needs no training. `Probe` is refit without the held-out source. `LoRA` is retrained without the source by a GPU job
# (sources in `lora_loso_sources`). Thresholds always come from legitimate validation emails of the *other* sources.

# %%
def loso_slm_rows():
    rows = []; t = P_TAG; L = BEST_LAYER[t]
    ptr = zs_npz(t, "probe_train")[f"H{L}"].astype(np.float32); ytr = SETS["probe_train"].label.values; src_tr = SETS["probe_train"].source.values
    H = {s: zs_npz(t, s)[f"H{L}"].astype(np.float32) for s in SCORE_SETS_ALL}
    for src in sorted(emails.source.unique()):
        parts = {"val": (SETS["val"].source != src).values & (SETS["val"].label == 0).values,
                 "test_clean": (SETS["test_clean"].source == src).values}
        if src == XSRC:
            for a in ATTACKS[1:]: parts[a] = (SETS[a].source == src).values
        pm_ = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=PROBES[t].named_steps["lr"].C, max_iter=3000))]).fit(ptr[src_tr != src], ytr[src_tr != src])
        lname = f"{LORA_MAIN[t]}_x{src}"
        for k, m in parts.items():
            if not m.any(): continue
            base = dict(source=src, set=k, id=SETS[k].id.values[m], label=SETS[k].label.values[m])
            rows.append(pd.DataFrame(dict(base, detector=ZS_P, score=SC[k][f"s_{ZS_P}"].values[m])))
            rows.append(pd.DataFrame(dict(base, detector=PROBE_P, score=pm_.predict_proba(H[k][m])[:, 1])))
            lv = lora_npy(lname, k)
            if lv is not None: rows.append(pd.DataFrame(dict(base, detector=LORA_P, score=lv[m])))
    return pd.concat(rows, ignore_index=True)

LOSO_RAW = pd.concat([LOSO_CLS, loso_slm_rows()], ignore_index=True)
LOSO_RAW.to_parquet(DIRS["results"] / "loso_scores.parquet", index=False)

def loso_metrics(raw, alpha):
    rows = []
    for (src, d), g in raw[raw.set.isin(["val", "test_clean"])].groupby(["source", "detector"], sort=True):
        ev, vl = g[g.set == "test_clean"], g[g.set == "val"]
        f = ev.score.values > thr_at_fpr(vl.score.values, alpha); y = ev.label.values
        tpr = f[y == 1].mean() if (y == 1).any() else np.nan; tnr = (~f[y == 0]).mean() if (y == 0).any() else np.nan
        both = (y == 1).any() and (y == 0).any()
        rows.append(dict(source=src, detector=d, n=len(ev), tpr=tpr, tnr=tnr, single_class=not both,
                         score=np.nanmean([tpr, tnr]), auc=roc_auc_score(y, ev.score.values) if both else np.nan))
    return pd.DataFrame(rows)

LOSO = loso_metrics(LOSO_RAW, A0)
LOSO_DETS = [d for d in CLS_DETS + [ZS_P, PROBE_P, LORA_P] if d in set(LOSO.detector)]
display(LOSO.pivot(index="source", columns="detector", values="score").reindex(columns=LOSO_DETS).round(3))

def xsrc_matrix(raw, alpha):
    g = raw[raw.source == XSRC]
    if g.set.nunique() <= 2: return pd.DataFrame()
    dets = [d for d in LOSO_DETS if (g.detector == d).any() and g[(g.detector == d) & (g.set == "goodword")].shape[0]]
    piv = {d: g[g.detector == d].set_index(["set", "id"]).score for d in dets}
    thr = {d: thr_at_fpr(piv[d].loc["val"].values, alpha) for d in dets}
    clean_ids = [i for i in POOL.id if i in set(piv[dets[0]].loc["test_clean"].index)]
    s_or = LORA_P if LORA_P in dets else PROBE_P
    _, tc_, ts_ = or_thresholds(piv[C_STAR].loc["val"].values, piv[s_or].loc["val"].values, alpha)
    def fl(d, k, ids):
        if d == "Hybrid-OR":
            return (piv[C_STAR].loc[k].loc[ids].values > tc_) | (piv[s_or].loc[k].loc[ids].values > ts_)
        return piv[d].loc[k].loc[ids].values > thr[d]
    rows = []; leg_ids = SETS["test_clean"][(SETS["test_clean"].source == XSRC) & (SETS["test_clean"].label == 0)].id.tolist()
    for d in dets + ["Hybrid-OR"]:
        f0 = dict(zip(clean_ids, fl(d, "test_clean", clean_ids)))
        fpr = fl(d, "test_clean", leg_ids).mean() if leg_ids else np.nan
        for att in ATTACKS:
            if att == "clean":
                ids = clean_ids; k = "test_clean"; base = ids
            else:
                fr = SC[att][(SC[att].source == XSRC) & SC[att].valid.astype(bool)]
                ids, k, base = fr.id.tolist(), att, fr.base_id.tolist()
            f = fl(d, k, ids); caught = np.array([f0.get(b, False) for b in base])
            rows.append(dict(detector=d, attack=att, n=len(ids), dr=f.mean(), asr=(~f[caught]).mean() if caught.any() else np.nan, fpr_test=fpr))
    return pd.DataFrame(rows)

XSRC_MAT = xsrc_matrix(LOSO_RAW, A0)
if len(XSRC_MAT):
    display(XSRC_MAT.pivot(index="detector", columns="attack", values="dr").reindex(columns=ATTACKS).round(3))

# %% [markdown]
# ### 8.8 Seed stability of the LoRA read-out (RQ5)

# %%
def seed_table(alpha):
    t = P_TAG; rows = []
    for k, seed in enumerate(LORA_SEEDS[t]):
        col = f"s_LoRA-{t}" if k == 0 else f"s_LoRA-{t}@s{seed}"
        if SC["test_clean"][col].isna().any(): continue
        thr = thr_at_fpr(VAL_LEG[col].values, alpha)
        cf = dict(zip(CLEAN_POOL.id, CLEAN_POOL[col].values > thr))
        rows.append(dict(seed=seed, attack="AUC (clean test)", value=roc_auc_score(SC["test_clean"].label, SC["test_clean"][col])))
        for att in ATTACKS:
            df = attack_frame(att); f = df[col].values > thr
            rows.append(dict(seed=seed, attack=att, value=f.mean()))
            if att != "clean":
                caught = np.array([cf.get(b, False) for b in df.base_id])
                rows.append(dict(seed=seed, attack=f"ASR {att}", value=(~f[caught]).mean() if caught.any() else np.nan))
    return pd.DataFrame(rows)

SEEDS = seed_table(A0)
N_SEEDS = SEEDS.seed.nunique() if len(SEEDS) else 0
if N_SEEDS:
    display(SEEDS.groupby("attack").value.agg(["mean", "std", "min", "max"]).round(4))

# %% [markdown]
# ### 8.9 Threshold-free effects (v3.2): does the read-out result depend on the operating point or on clean strength?
# The prompted read-out detects few emails at 1% FPR, so a large drop in its detection rate could reflect a weak detector sitting near its threshold.
# Here every effect is measured as a change in AUC against the same clean legitimate test emails, paired over the same malicious emails.

# %%
_LEG_TEST = SC["test_clean"][SC["test_clean"].label == 0]
def rank_scores(det, vals):
    L = np.sort(_LEG_TEST[f"s_{det}"].values.astype(float)); v = np.asarray(vals, float)
    return (np.searchsorted(L, v, "left") + np.searchsorted(L, v, "right")) / (2 * len(L))

def auc_effect_rows(a_ctrl, a_att, label):
    A, B = paired(a_ctrl, a_att); rows = []
    for d in BASE_DETS + ["Hybrid-Stack", "Stack-Aug"]:
        ra, rb = rank_scores(d, A[f"s_{d}"].values), rank_scores(d, B[f"s_{d}"].values)
        m, lo, hi = boot_diff(ra, rb)
        rows.append(dict(effect=label, detector=d, n=len(A), auc_control=ra.mean(), auc_attack=rb.mean(), d_auc=m, d_auc_lo=lo, d_auc_hi=hi))
    return rows

AUC_EFF = pd.DataFrame(auc_effect_rows("neutral", "inject", "instruction") + auc_effect_rows("neutral", "inject_aware", "instruction (aware)")
                       + auc_effect_rows("clean", "ham_pad", "natural padding") + auc_effect_rows("ham_pad", "goodword", "word salad vs natural pad"))
AUC_EFF.to_csv(DIRS["results"] / "auc_effects.csv", index=False)

def auc_did_rows(a_ctrl, a_att, label):
    A, B = paired(a_ctrl, a_att); rows = []
    for t in BB_TAGS:
        ros = [r for r in READOUTS if f"{r}-{t}" in AVAIL]
        for i, r1 in enumerate(ros):
            for r2 in ros[i + 1:]:
                d1, d2 = f"{r1}-{t}", f"{r2}-{t}"
                e1 = rank_scores(d1, A[f"s_{d1}"]) - rank_scores(d1, B[f"s_{d1}"]); e2 = rank_scores(d2, A[f"s_{d2}"]) - rank_scores(d2, B[f"s_{d2}"])
                m, lo, hi = boot_diff(e1, e2)
                rows.append(dict(effect=label, backbone=t, readout_1=r1, readout_2=r2, d_auc_1=e1.mean(), d_auc_2=e2.mean(), did=m, did_lo=lo, did_hi=hi,
                                 excludes_zero=bool(lo > 0 or hi < 0)))
    return rows
AUC_DID = pd.DataFrame(auc_did_rows("neutral", "inject", "instruction") + auc_did_rows("neutral", "inject_aware", "instruction (aware)"))
AUC_DID.to_csv(DIRS["results"] / "auc_readout_interaction.csv", index=False)
display(AUC_EFF[AUC_EFF.effect.str.startswith("instruction") & AUC_EFF.detector.isin(SLM_DETS + [C_STAR])].round(4))
display(AUC_DID.round(4))

# %% [markdown]
# ### 8.10 Natural padding: dose response (v3.2)
# The appended text has a fixed length (about 80 words), so its share of the email depends on the original length. Detection is reported by quartile of the
# padding share; a detector that only fails on short emails is diluted, one that fails at every share is steered by the benign content itself.

# %%
_hp = attack_frame("ham_pad").copy()
_orig_words = CLEAN_POOL.set_index("id").loc[_hp.base_id].text.map(lambda t: len(str(t).split())).values
_pad_words = np.array([len(t.split()) for t in SETS["ham_pad"].set_index("id").loc[_hp.id].text.values]) - _orig_words
_hp["pad_share"] = _pad_words / np.maximum(1, _pad_words + _orig_words)
_hp["share_bin"] = pd.qcut(_hp.pad_share, 4, labels=False, duplicates="drop")
PAD_DETS = list(dict.fromkeys([C_STAR, "SVM-char", S_STAR] + [d for d in (AUG_DET, AUGMIX_DET) if d in AVAIL] + [PROBE_P, "Hybrid-OR", "Stack-Aug"]))
_rows = []
for b_, g in _hp.groupby("share_bin"):
    for d in PAD_DETS:
        m_, lo_, hi_ = boot_mean(flag(g, d))
        _rows.append(dict(bin=int(b_), share_lo=g.pad_share.min(), share_hi=g.pad_share.max(), share_med=g.pad_share.median(), n=len(g), detector=d,
                          dr=m_, dr_lo=lo_, dr_hi=hi_))
PAD_DOSE = pd.DataFrame(_rows); PAD_DOSE.to_csv(DIRS["results"] / "natural_pad_dose.csv", index=False)
display(PAD_DOSE.pivot(index="detector", columns="bin", values="dr").loc[PAD_DETS].round(3))

# %% [markdown]
# ### 8.11 Collateral false positives (v3.2)
# Legitimate test emails that carry a neutral suffix or an instruction-like suffix (as benign mail does when it forwards or quotes such text). The paired change in
# FPR against the same emails in clean form shows what each defense costs on benign mail; an injection-aware stacker is expected to pay most here.

# %%
COLL = pd.DataFrame()
COLL_SETS = [s_ for s_ in LEGIT_SETS + PAD_LEGIT if s_ in SC]
if COLL_SETS:
    _clean_leg = SC["test_clean"].set_index("id")
    _rows = []
    for s_ in COLL_SETS:
        df_ = SC[s_]; c_ = _clean_leg.loc[df_.base_id].reset_index()
        for d in dets_on(s_, EVAL_DETS):
            f1_, f0_ = flag(df_, d), flag(c_, d); n10, n01, p_ = mcnemar(f1_, f0_)
            m_, lo_, hi_ = boot_mean(f1_)
            _rows.append(dict(set=s_, detector=d, n=len(df_), fpr_clean=f0_.mean(), fpr=m_, fpr_lo=lo_, fpr_hi=hi_, new_fp=n10, removed_fp=n01, p=p_))
    COLL = pd.DataFrame(_rows); COLL["p_holm"] = holm(COLL.p.values)
    COLL.to_csv(DIRS["results"] / "collateral_fpr.csv", index=False)
    display(COLL.pivot(index="detector", columns="set", values="fpr").round(4))
else:
    print("Collateral sets not scored by every SLM yet (deferred); rerun with this output attached.")

# %% [markdown]
# ### 8.12 LOSO on common sources (v3.2)
# Means over held-out sources are only comparable over the same sources; this table restricts every detector to the sources where all of them were evaluated.

# %%
# v4.0: the prompted read-out has no training, so its "LOSO" row only moves a threshold; it is reported apart. Sources with a single class
# (all-malicious) give a detection rate, not a balanced accuracy, so the two kinds of source are averaged separately.
LOSO_TRAINED = [d for d in LOSO_DETS if d != ZS_P]
_lp = LOSO.pivot(index="source", columns="detector", values="score").reindex(columns=LOSO_TRAINED)
_single = LOSO.groupby("source").single_class.first()
LOSO_COMMON = list(_lp.dropna().index)
_mixed = [x for x in LOSO_COMMON if not _single[x]]; _onecls = [x for x in LOSO_COMMON if _single[x]]
LOSO_MEAN = pd.DataFrame({"mean_all_available": _lp.mean(), "n_sources": _lp.notna().sum(),
                          "mean_common": _lp.loc[_mixed].mean(), "min_common": _lp.loc[_mixed].min(),
                          "tpr_single_class": _lp.loc[_onecls].mean() if _onecls else np.nan})
print(f"Common sources: balanced accuracy over {_mixed}; detection rate over single-class sources {_onecls}")
display(LOSO_MEAN.round(3))


# %% [markdown]
# ### 8.13 Out-of-distribution padding (v4.0): does padding-aware training generalize?
# Detection on the same malicious emails padded in ways `LoRA-Aug` never saw in training (position, length, text source, hidden HTML). `LoRA-AugMix` was trained
# with mixed positions and lengths but never with LLM-written or hidden text, so `pad_llm` and `pad_hidden` are held out for both padding-aware models.
# `LoRA-Aug` seeds: mean and range over the available seeds.

# %%
PAD_COLS = ["clean", "ham_pad"] + [s_ for s_ in OOD_SETS if s_ in SC]
PAD_LABEL = {"clean": "Clean"} | OOD_LABEL
OOD_DETS = list(dict.fromkeys([C_STAR, "SVM-char"] + [d for d in SLM_DETS] + ["Hybrid-OR", "Hybrid-Stack", "Stack-Aug"]))
_rows = []
for d in OOD_DETS:
    for a_ in PAD_COLS:
        if a_ != "clean" and d not in dets_on(a_, [d]): continue
        m_, lo_, hi_ = boot_mean(flag(attack_frame(a_), d))
        _rows.append(dict(detector=d, pad=a_, dr=m_, dr_lo=lo_, dr_hi=hi_, n=len(attack_frame(a_))))
OOD_MAT = pd.DataFrame(_rows); OOD_MAT.to_csv(DIRS["results"] / "ood_padding_matrix.csv", index=False)
display(OOD_MAT.pivot(index="detector", columns="pad", values="dr").reindex(columns=PAD_COLS).round(3))

# LoRA-Aug across seeds on every padding variant
_rows = []
for s_, col in [(LORA_SEEDS[P_TAG][0], AUG_DET)] + [(x, f"{AUG_DET}@s{x}") for x in AUG_SEEDS]:
    if AUG_DET not in AVAIL: break
    for a_ in PAD_COLS:
        df_ = attack_frame(a_)
        if df_[f"s_{col}"].isna().any(): continue
        _rows.append(dict(seed=s_, pad=a_, dr=float((df_[f"s_{col}"] > THR[A0][col]).mean())))
AUG_SEED_TAB = pd.DataFrame(_rows)
if len(AUG_SEED_TAB):
    display(AUG_SEED_TAB.groupby("pad").dr.agg(["mean", "std", "min", "max", "count"]).reindex(PAD_COLS).round(4))

# Paired comparisons on each padding variant: is padding-aware training better than plain LoRA, and does mixing generalize further?
_rows = []
for a_ in PAD_COLS[1:]:
    df_ = attack_frame(a_)
    for A_, B_ in [(AUG_DET, LORA_P), (AUGMIX_DET, LORA_P), (AUGMIX_DET, AUG_DET), (AUG_DET, "Hybrid-OR")]:
        if A_ in AVAIL and B_ in (AVAIL | {"Hybrid-OR"}) and set(dets_on(a_, [A_, B_])) == {A_, B_}:
            fa, fb = flag(df_, A_), flag(df_, B_); n10, n01, p_ = mcnemar(fa, fb)
            _rows.append(dict(pad=a_, A=A_, B=B_, dr_A=fa.mean(), dr_B=fb.mean(), A_only=n10, B_only=n01, p=p_))
MCN_OOD = pd.DataFrame(_rows)
if len(MCN_OOD):
    MCN_OOD["p_holm"] = holm(MCN_OOD.p.values); MCN_OOD.to_csv(DIRS["results"] / "mcnemar_ood.csv", index=False)
    display(MCN_OOD.round(4))

# %% [markdown]
# ### 8.14 Stronger prompted baselines (v4.0)
# Is the prompted read-out's injection failure a property of a weak prompt or a small model? The same zero-shot read-out with a 4-shot prompt on the primary
# backbone and with a 7B model (4-bit) are scored on clean data, the neutral control, both injections, natural padding and benign mail with an instruction-like suffix.

# %%
def auc_effect(d, a_ctrl, a_att):
    A, B = paired(a_ctrl, a_att)
    if A[f"s_{d}"].isna().any() or B[f"s_{d}"].isna().any(): return (np.nan,) * 3
    return boot_diff(rank_scores(d, A[f"s_{d}"].values), rank_scores(d, B[f"s_{d}"].values))

def dr_effect(d, a_ctrl, a_att):
    A, B = paired(a_ctrl, a_att)
    if A[f"s_{d}"].isna().any() or B[f"s_{d}"].isna().any(): return (np.nan,) * 3
    return boot_diff(flag(A.reset_index(), d).astype(float), flag(B.reset_index(), d).astype(float))

PROMPT_ROWS = list(dict.fromkeys([ZS_P] + [f"ZS-{t}" for t in BB_TAGS if f"ZS-{t}" in AVAIL] + PROMPT_AVAIL + [LORA_P]))
_rows = []
for d in PROMPT_ROWS:
    tc_ = SC["test_clean"]; a_, lo_, hi_ = boot_auc(tc_.label, tc_[f"s_{d}"])
    r = dict(detector=d, auc=a_, auc_lo=lo_, auc_hi=hi_, tpr=flag(tc_[tc_.label == 1], d).mean())
    for e_, (c_, t_) in {"instr": ("neutral", "inject"), "instr_aware": ("neutral", "inject_aware")}.items():
        r[f"{e_}_dr"], r[f"{e_}_dr_lo"], r[f"{e_}_dr_hi"] = dr_effect(d, c_, t_)
        r[f"{e_}_auc"], r[f"{e_}_auc_lo"], r[f"{e_}_auc_hi"] = auc_effect(d, c_, t_)
    r["auc_inject_aware"] = float(np.mean(rank_scores(d, attack_frame("inject_aware")[f"s_{d}"].values)))   # below 0.5 = ranking inverted
    r["dr_ham_pad"] = flag(attack_frame("ham_pad"), d).mean() if attack_frame("ham_pad")[f"s_{d}"].notna().all() else np.nan
    r["fpr_legit_inject"] = flag(SC["legit_inject"], d).mean() if "legit_inject" in SC and SC["legit_inject"][f"s_{d}"].notna().all() else np.nan
    _rows.append(r)
PROMPT_TAB = pd.DataFrame(_rows); PROMPT_TAB.to_csv(DIRS["results"] / "prompted_baselines.csv", index=False)
display(PROMPT_TAB[["detector", "auc", "tpr", "instr_auc", "instr_aware_auc", "auc_inject_aware", "dr_ham_pad", "fpr_legit_inject"]].round(4))

# %% [markdown]
# ### 8.15 Modern, out-of-time data (v4.0)
# Thresholds are the ones fixed on the original validation split. Reported per subset (human seeds up to 2025; LLM-rephrased 2026): ROC-AUC, detection rate on
# malicious mail (phishing and spam separately), false-positive rate on valid mail. Then the read-out and padding results are replicated on modern malicious mail:
# instruction effect (label-aware injection vs neutral suffix) on the AUC scale against modern valid mail, and detection under natural padding.

# %%
MODERN_TAB, MODERN_EFF = pd.DataFrame(), pd.DataFrame()
_msets = [m for m in ["modern_human", "modern_llm"] if m in SC]
if _msets:
    _rows = []
    MOD_DETS = list(dict.fromkeys(BASE_DETS + PROMPT_AVAIL + V5_AVAIL + ["Hybrid-OR", "Hybrid-Stack", "Stack-Aug"]))   # v5.1: + v5.0 detectors
    for m in _msets:
        df_ = SC[m] if "ptype" in SC[m] else SC[m].merge(SETS[m][["id", "ptype"]], on="id", how="left")
        for d in dets_on(m, MOD_DETS):
            y_ = df_.label.values; f_ = flag(df_, d)
            sc_ = np.maximum(df_[f"s_{C_STAR}"], df_[f"s_{S_STAR}"]) if d == "Hybrid-OR" else df_[f"s_{d}"]
            a_, lo_, hi_ = boot_auc(y_, sc_) if 0 < y_.mean() < 1 else (np.nan,) * 3
            r = dict(set=m, detector=d, n=len(df_), auc=a_, auc_lo=lo_, auc_hi=hi_, tpr=f_[y_ == 1].mean(), fpr=f_[y_ == 0].mean() if (y_ == 0).any() else np.nan)
            for pt in ["Phishing", "Spam"]:
                mk = (df_.ptype == pt).values
                r[f"tpr_{pt.lower()}"] = f_[mk].mean() if mk.any() else np.nan
            _rows.append(r)
    MODERN_TAB = pd.DataFrame(_rows); MODERN_TAB.to_csv(DIRS["results"] / "modern_results.csv", index=False)
    display(MODERN_TAB.pivot(index="detector", columns="set", values="auc").round(4))
    # read-out and padding replication on modern malicious mail
    _mall = pd.concat([SC[m] for m in _msets], ignore_index=True)
    _mleg = _mall[_mall.label == 0]
    _rows = []
    for d in dets_on("modern_neutral", BASE_DETS + PROMPT_AVAIL + V5_AVAIL):
        if not {"modern_inject_aware", "modern_pad"} <= set(SC) or d not in dets_on("modern_inject_aware", [d]): continue
        L_ = np.sort(_mleg[f"s_{d}"].values.astype(float))
        rk = lambda v: (np.searchsorted(L_, v, "left") + np.searchsorted(L_, v, "right")) / (2 * len(L_))
        A_ = SC["modern_neutral"].set_index("base_id"); B_ = SC["modern_inject_aware"].set_index("base_id").loc[A_.index]
        e_, lo_, hi_ = boot_diff(rk(A_[f"s_{d}"].values), rk(B_[f"s_{d}"].values))
        C_ = _mall.set_index("id").loc[A_.index]
        r = dict(detector=d, n=len(A_), auc_neutral=rk(A_[f"s_{d}"].values).mean(), instr_aware_auc=e_, lo=lo_, hi=hi_,
                 dr_clean=float((C_[f"s_{d}"] > THR[A0][d]).mean()))
        if d in dets_on("modern_pad", [d]):
            P_ = SC["modern_pad"].set_index("base_id").loc[A_.index]; r["dr_pad"] = float((P_[f"s_{d}"] > THR[A0][d]).mean())
        _rows.append(r)
    MODERN_EFF = pd.DataFrame(_rows); MODERN_EFF.to_csv(DIRS["results"] / "modern_effects.csv", index=False)
    display(MODERN_EFF.round(4))
else:
    print("Modern data not available (download failed or jobs deferred).")

# %% [markdown]
# ### 8.16 Human validity checks (v4.0)
# Looks for filled audit sheets (`*_audit_filled.csv`, attached as an input). Reports the share judged valid by each rater and Cohen's kappa between raters.

# %%
def _kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int); po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 - 1e-12 else float("nan")   # undefined when neither rater varies (all items rated the same)
AUDIT = {}
for f in glob.glob(str(Path(CFG["input_root"]) / "**/*audit_filled.csv"), recursive=True):
    d = pd.read_csv(f); cols = [c for c in d.columns if c.endswith("_yn")]
    rs = sorted({c.split("_")[0] for c in cols})
    if len(rs) < 2: continue
    out = {}
    for crit in sorted({c[len(rs[0]) + 1:] for c in cols if c.startswith(rs[0])}):
        a_, b_ = d[f"{rs[0]}_{crit}"], d[f"{rs[1]}_{crit}"]; ok = a_.notna() & b_.notna()
        out[crit] = dict(n=int(ok.sum()), rater1=float(a_[ok].mean()), rater2=float(b_[ok].mean()), kappa=float(_kappa(a_[ok], b_[ok])),
                         raw_agreement=float((a_[ok] == b_[ok]).mean()),
                         both_valid=float(((a_[ok] == 1) & (b_[ok] == 1)).mean()))
    AUDIT[Path(f).name] = out
print(json.dumps(AUDIT, indent=1) if AUDIT else "No filled audit sheets found. Fill cbs/private/*_audit.csv, save as *_audit_filled.csv and attach.")
