# %% [markdown]
# ## 7. Collect the GPU results and assemble score tables

# %%
with Timer("slm_jobs_wait"):
    JOB_STATE = JOBS.wait()
DEFERRED = [n for n, s in JOB_STATE.items() if s == "deferred"]
if DEFERRED:
    print("Deferred jobs (attach this version's output and run again to complete them):", DEFERRED)

def zs_npz(tag, s):
    p = cpath(f"zs_{tag}_{s}.npz"); return np.load(p) if p.exists() else None
def lora_npy(name, s):
    p = cpath(f"{name}_{s}.npy"); return np.load(p) if p.exists() else None
ZS_META = {t: json.loads(cpath(f"zs_{t}_meta.json").read_text()) for t in BB_TAGS if have(f"zs_{t}_meta.json")}
LORA_SEEDS = {b["tag"]: b["lora_seeds"] for b in ALL_BACKBONES}
LORA_MAIN = {t: f"lora_{t}_s{LORA_SEEDS[t][0]}" for t in BB_TAGS}

_need = ["val"] + EVAL_SETS
AVAIL = set()
for t in BB_TAGS:
    if all(have(f"zs_{t}_{s}.npz") for s in _need) and "best_layer" in ZS_META.get(t, {}):
        AVAIL |= {f"ZS-{t}", f"Probe-{t}"}
    if all(have(f"{LORA_MAIN[t]}_{s}.npy") for s in _need):
        AVAIL.add(f"LoRA-{t}")
AUG_NAME = f"{LORA_MAIN[P_TAG]}_aug"
AUGMIX_NAME = f"{LORA_MAIN[P_TAG]}_augmix"
AUG_SEEDS = [s_ for s_ in CFG["lora_aug_seeds"] if all(have(f"lora_{P_TAG}_s{s_}_aug_{s}.npy") for s in _need)]
for d_, n_ in ((AUG_DET, AUG_NAME), (AUGMIX_DET, AUGMIX_NAME)):
    if all(have(f"{n_}_{s}.npy") for s in _need):
        AVAIL.add(d_)
# v4.0 prompted baselines: file tags and availability (they need validation and clean-test scores)
PROMPT_FILE = {FS_DET: f"fs_{P_TAG}", LARGE_DET: CFG["prompt_large"]["tag"]}
PROMPT_AVAIL = [d for d in PROMPT_X if all(have(f"zs_{PROMPT_FILE[d]}_{s}.npz") for s in ["val", "test_clean"])]
# v5.0 detectors and extra seeds: available once their validation and clean-test scores exist
def _v5_have(kind, name, s): return have(f"zs_{name}_{s}.npz") if kind == "zs" else have(f"{name}_{s}.npy")
V5_AVAIL = [d for d, (k_, n_) in V5_FILES.items() if all(_v5_have(k_, n_, s) for s in ["val", "test_clean"])]
def _sf(n_): return n_ if isinstance(n_, (tuple, list)) else ("npy", n_)   # v5.2: extra seeds of LoRA-Gen are ZS-style .npz files
V5_SEED_AVAIL = [c for c, n_ in V5_SEED_FILES.items() if all(_v5_have(*_sf(n_), s) for s in ["val", "test_clean"])]
print("v5.0 detectors available:", V5_AVAIL, "| extra seeds:", V5_SEED_AVAIL)
FAILED_OPT = getattr(JOBS, "failed_optional", [])
SLM_DETS = [d for d in SLM_ALL + SLM_EXTRA if d in AVAIL]
assert {ZS_P, PROBE_P, LORA_P} <= AVAIL, f"Core SLM outputs missing: {set([ZS_P, PROBE_P, LORA_P]) - AVAIL}"
BASE_DETS = CLS_DETS + SLM_DETS
print("Available SLM detectors:", SLM_DETS, "| prompted baselines:", PROMPT_AVAIL, "| LoRA-Aug extra seeds:", AUG_SEEDS)
print("Backbones:", {t: FAMILY_BB[t] for t in BB_TAGS})

# %% [markdown]
# ### 7.1 Probes (refit here from the cached hidden states at each backbone's selected layer)

# %%
PROBES, BEST_LAYER = {}, {}
for t in BB_TAGS:
    if f"Probe-{t}" not in AVAIL: continue
    L = ZS_META[t]["best_layer"]; BEST_LAYER[t] = L
    auc, c, m = S.fit_probe(zs_npz(t, "probe_train")[f"H{L}"].astype(np.float32), SETS["probe_train"].label.values,
                            zs_npz(t, "val")[f"H{L}"].astype(np.float32), SETS["val"].label.values)
    PROBES[t] = m
    print(f"Probe-{t}: layer {L} of {ZS_META[t]['n_layers']}, C={c}, validation AUC {auc:.4f}")

# %% [markdown]
# ### 7.2 One score table per evaluation set
# Columns `s_<detector>` hold P(malicious). Classical latency is measured on CPU (one process), SLM latency on one GPU at batch size `score_batch`.

# %%
def _have_set(s, dets):
    """True when every listed detector has GPU scores for set s (ZS/Probe share one file per backbone)."""
    ok = True
    for d in dets:
        t = d.split("-")[-1]
        if d.startswith(("ZS-", "Probe-")): ok &= have(f"zs_{t}_{s}.npz")
        elif d == AUG_DET: ok &= have(f"{AUG_NAME}_{s}.npy")
        elif d == AUGMIX_DET: ok &= have(f"{AUGMIX_NAME}_{s}.npy")
        elif d.startswith("LoRA-"): ok &= have(f"{LORA_MAIN[t]}_{s}.npy")
    return bool(ok)
# A set enters the score tables when the primary backbone's three read-outs (the stackers' inputs) have scored it; other detectors that have not are
# NaN there and are left out of that set's analysis (see dets_on()).
_CORE = [ZS_P, PROBE_P, LORA_P]
LEGIT_AVAIL = all(_have_set(s, _CORE) for s in LEGIT_SETS)
ADAPT_NAT_AVAIL = _have_set("adaptive_nat", _CORE)
ADAPT_OOD_AVAIL = "adaptive_ood" in SETS and _have_set("adaptive_ood", _CORE)
NEW_OK = [s for s in OOD_SETS + PAD_LEGIT + MODERN_SETS + V5_EVAL + PHISH_SETS if _have_set(s, _CORE)]   # v5.2: + phishing pool
SCORE_SETS = (["val", "val_inj", "val_pad", "test_clean"] + ATTACKS[1:] + ["adaptive"] + (LEGIT_SETS if LEGIT_AVAIL else [])
              + (["adaptive_nat"] if ADAPT_NAT_AVAIL else []) + (["adaptive_ood"] if ADAPT_OOD_AVAIL else []) + NEW_OK)
print(f"Collateral sets: {LEGIT_AVAIL} | adaptive natural arm: {ADAPT_NAT_AVAIL} | adaptive OOD arm: {ADAPT_OOD_AVAIL} | v4.0 sets: {NEW_OK}")
LAT = {}
with Timer("assemble_scores"):
    SC = {}
    for s in SCORE_SETS:
        df = SETS[s].copy()
        cs = score_classical(df.text)
        if s == "test_clean":
            feats = {}
            for n, m in CLASSICAL.items():
                t1 = time.time(); m.predict_proba(df.text.tolist()); LAT[n] = (time.time() - t1) / len(df)
        for n, v in cs.items(): df[f"s_{n}"] = v
        for t in BB_TAGS:
            z = zs_npz(t, s)
            if f"ZS-{t}" in AVAIL and z is not None:
                df[f"s_ZS-{t}"] = z["p"]
                df[f"s_Probe-{t}"] = PROBES[t].predict_proba(z[f"H{BEST_LAYER[t]}"].astype(np.float32))[:, 1]
            else:
                df[f"s_ZS-{t}"] = np.nan; df[f"s_Probe-{t}"] = np.nan
            for k, seed in enumerate(LORA_SEEDS[t]):
                v = lora_npy(f"lora_{t}_s{seed}", s)
                col = f"s_LoRA-{t}" if k == 0 else f"s_LoRA-{t}@s{seed}"
                df[col] = v if v is not None else np.nan
        for d_, n_ in ((AUG_DET, AUG_NAME), (AUGMIX_DET, AUGMIX_NAME)):
            v = lora_npy(n_, s); df[f"s_{d_}"] = v if v is not None else np.nan
        for s_ in CFG["lora_aug_seeds"]:
            v = lora_npy(f"lora_{P_TAG}_s{s_}_aug", s); df[f"s_{AUG_DET}@s{s_}"] = v if v is not None else np.nan
        for d_ in PROMPT_X:
            z = zs_npz(PROMPT_FILE[d_], s); df[f"s_{d_}"] = z["p"] if z is not None else np.nan
        for d_, (k_, n_) in V5_FILES.items():   # v5.0
            if k_ == "zs":
                z = zs_npz(n_, s); df[f"s_{d_}"] = z["p"] if z is not None else np.nan
            else:
                v = lora_npy(n_, s); df[f"s_{d_}"] = v if v is not None else np.nan
        for c_, n_ in V5_SEED_FILES.items():
            k_, n_ = _sf(n_)
            if k_ == "zs":
                z = zs_npz(n_, s); df[f"s_{c_}"] = z["p"] if z is not None else np.nan
            else:
                v = lora_npy(n_, s); df[f"s_{c_}"] = v if v is not None else np.nan
        if "valid" not in df: df["valid"] = True
        SC[s] = df
for t in BB_TAGS:
    if f"ZS-{t}" in AVAIL:
        LAT[f"ZS-{t}"] = LAT[f"Probe-{t}"] = ZS_META[t]["sec_per_email"]["test_clean"]
    if f"LoRA-{t}" in AVAIL:
        LAT[f"LoRA-{t}"] = json.loads(cpath(f"{LORA_MAIN[t]}_scoremeta.json").read_text())["test_clean"]
for d_, n_ in ((AUG_DET, AUG_NAME), (AUGMIX_DET, AUGMIX_NAME)):
    if d_ in AVAIL: LAT[d_] = json.loads(cpath(f"{n_}_scoremeta.json").read_text())["test_clean"]
for d_ in PROMPT_AVAIL:
    LAT[d_] = json.loads(cpath(f"zs_{PROMPT_FILE[d_]}_meta.json").read_text())["sec_per_email"]["test_clean"]
V5_LAT = {}   # v5.0 latencies kept apart from the v4.0 cost figure
for d_ in V5_AVAIL:
    k_, n_ = V5_FILES[d_]
    try:
        V5_LAT[d_] = (json.loads(cpath(f"zs_{n_}_meta.json").read_text())["sec_per_email"]["test_clean"] if k_ == "zs"
                      else json.loads(cpath(f"{n_}_scoremeta.json").read_text())["test_clean"])
    except Exception: pass
print("Latency (ms/email):", {k: round(v * 1000, 2) for k, v in LAT.items()})

# %% [markdown]
# ### 7.3 Operating thresholds (validation only)
# For detector $d$ and target false-positive rate $\alpha$, $\tau_d(\alpha)$ is the $(1-\alpha)$ empirical quantile of $d$'s scores on **all** legitimate validation emails.
# A test email is flagged when $s_d(x) > \tau_d$. The same thresholds are used on clean, control and attacked data.

# %%
def thr_at_fpr(scores_legit, alpha):
    s = np.sort(np.asarray(scores_legit)); k = int(np.floor((1 - alpha) * len(s)))
    return s[min(k, len(s) - 1)]

VAL_LEG = SC["val"][SC["val"].label == 0]
THR = {a: {d: thr_at_fpr(VAL_LEG[f"s_{d}"], a) for d in BASE_DETS + PROMPT_AVAIL} for a in CFG["target_fprs"]}
for a in CFG["target_fprs"]:      # extra LoRA-Aug seeds (seed-stability analysis)
    for s_ in AUG_SEEDS: THR[a][f"{AUG_DET}@s{s_}"] = thr_at_fpr(VAL_LEG[f"s_{AUG_DET}@s{s_}"], a)
    for d_ in V5_AVAIL + V5_SEED_AVAIL: THR[a][d_] = thr_at_fpr(VAL_LEG[f"s_{d_}"], a)   # v5.0

def dets_on(s, dets):
    """Detectors that have scores for every email of set s."""
    return [d for d in dets if d == "Hybrid-OR" or (f"s_{d}" in SC[s] and SC[s][f"s_{d}"].notna().all())]
A0 = CFG["primary_fpr"]
print(f"Thresholds from {len(VAL_LEG)} legitimate validation emails (the 1% tail holds {int(round(0.01 * len(VAL_LEG)))} emails).")

# %% [markdown]
# ### 7.4 Injection detector (defender-side, `T_train` wording only), cross-fitted
# Out-of-fold scores (grouped 5-fold by original email) are used wherever the detector's own training rows appear (validation), so the stackers below never see
# in-sample injection scores. The final detector, fit on all validation rows, scores test, attack and adaptive sets.

# %%
_lg = lambda x: logit(np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6))
def _injdet():
    return Pipeline([("vec", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200_000)),
                     ("lr", LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))])

_inj_rows = pd.concat([SC["val"].assign(_y=0), SC["val_inj"].assign(_y=1)], ignore_index=True)
_oof = np.zeros(len(_inj_rows))
for tr_i, te_i in GroupKFold(n_splits=CFG["n_folds"]).split(_inj_rows, _inj_rows._y, _inj_rows.base_id):
    m = _injdet().fit(_inj_rows.text.iloc[tr_i].tolist(), _inj_rows._y.iloc[tr_i])
    _oof[te_i] = m.predict_proba(_inj_rows.text.iloc[te_i].tolist())[:, 1]
INJDET = _injdet().fit(_inj_rows.text.tolist(), _inj_rows._y)
SC["val"]["s_InjDet"] = _oof[:len(SC["val"])]; SC["val_inj"]["s_InjDet"] = _oof[len(SC["val"]):]
for s in SCORE_SETS:
    if s not in ("val", "val_inj"): SC[s]["s_InjDet"] = INJDET.predict_proba(SC[s].text.tolist())[:, 1]
_t = time.time(); INJDET.predict_proba(SC["test_clean"].text.tolist()); LAT["InjDet"] = (time.time() - _t) / len(SC["test_clean"])
INJ_THR = thr_at_fpr(SC["val"]["s_InjDet"], A0)
print(f"InjDet (OOF validation AUC {roc_auc_score(_inj_rows._y, _oof):.4f}): flags {np.mean(SC['inject'].s_InjDet > INJ_THR):.1%} of T_test injections, "
      f"{np.mean(SC['neutral'].s_InjDet > INJ_THR):.1%} of neutral suffixes, {np.mean(SC['test_clean'].query('label==0').s_InjDet > INJ_THR):.1%} of clean legitimate test")

# %% [markdown]
# ### 7.5 Hybrid defenses (primary backbone)
# $C^\*$ and $S^\*$ are the strongest classical and primary-backbone SLM detectors by **validation** ROC-AUC.
#
# * **Hybrid-OR (FPR-matched).** Flag if $s_{C^\*} > \tau_{C^\*}(\beta)$ or $s_{S^\*} > \tau_{S^\*}(\beta)$, with $\beta$ chosen so the union has FPR $\alpha$ on validation.
# * **Hybrid-Stack.** Logistic regression on $[\operatorname{logit} s_{C^\*}, \operatorname{logit} s_{S^\*}, \operatorname{logit} s_{\text{ZS}}, \operatorname{logit} s_{\text{InjDet}}]$, trained on validation emails plus
#   validation malicious emails carrying `T_train` injections. **Cross-fitted**: its threshold comes from out-of-fold scores on legitimate validation emails.
# * **Stack-noInj.** The same without the injection-detector feature (ablation).
# * **Stack-Aug.** Hybrid-Stack whose training rows also include validation malicious emails padded with *defender-side* legitimate text (disjoint from the attacker's pool;
#   the attacker's word-salad lexicon is never used). Tests whether learned fusion can be made padding-robust without knowing the attack.
# * **Triage.** Auto-block if both $C^\*$ and $S^\*$ flag, auto-allow if neither does, human review if they disagree.

# %%
def _auc(df, d): return roc_auc_score(df.label, df[f"s_{d}"])
C_STAR = max(CLS_DETS, key=lambda d: _auc(SC["val"], d))
S_STAR = max([ZS_P, PROBE_P, LORA_P], key=lambda d: _auc(SC["val"], d))
print(f"C* = {C_STAR}, S* = {S_STAR}")

def or_thresholds(lc, ls, alpha):
    best = None
    for beta in np.linspace(alpha / 4, alpha, 200):
        tc, ts = thr_at_fpr(lc, beta), thr_at_fpr(ls, beta)
        if np.mean((lc > tc) | (ls > ts)) <= alpha and (best is None or beta > best[0]): best = (beta, tc, ts)
    return best or (alpha / 4, thr_at_fpr(lc, alpha / 4), thr_at_fpr(ls, alpha / 4))

OR_THR = {a: or_thresholds(VAL_LEG[f"s_{C_STAR}"].values, VAL_LEG[f"s_{S_STAR}"].values, a) for a in CFG["target_fprs"]}

FEATS = {"Hybrid-Stack": list(dict.fromkeys([f"s_{C_STAR}", f"s_{S_STAR}", f"s_{ZS_P}", "s_InjDet"]))}
FEATS["Stack-noInj"] = [f for f in FEATS["Hybrid-Stack"] if f != "s_InjDet"]
FEATS["Stack-Aug"] = FEATS["Hybrid-Stack"]
_rows_base = pd.concat([SC["val"], SC["val_inj"][SC["val_inj"].label == 1]], ignore_index=True)
STACK_ROWS = {"Hybrid-Stack": _rows_base, "Stack-noInj": _rows_base, "Stack-Aug": pd.concat([_rows_base, SC["val_pad"]], ignore_index=True)}
def _stacker(): return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=2000))])

STACKERS, STACK_COEF = {}, {}
for name in ["Hybrid-Stack", "Stack-noInj", "Stack-Aug"]:
    R, F = STACK_ROWS[name], FEATS[name]
    oof = np.zeros(len(R))
    for tr_i, te_i in GroupKFold(n_splits=CFG["n_folds"]).split(R, R.label, R.base_id):
        oof[te_i] = _stacker().fit(_lg(R[F].iloc[tr_i]), R.label.iloc[tr_i]).predict_proba(_lg(R[F].iloc[te_i]))[:, 1]
    STACKERS[name] = _stacker().fit(_lg(R[F]), R.label)
    STACK_COEF[name] = {f[2:]: float(c) for f, c in zip(F, STACKERS[name].named_steps["lr"].coef_.ravel())}
    leg_oof = oof[((R.attack == "clean") & (R.label == 0)).values]
    for a in CFG["target_fprs"]:
        THR[a][name] = thr_at_fpr(leg_oof, a)
    for s in SCORE_SETS:
        SC[s][f"s_{name}"] = STACKERS[name].predict_proba(_lg(SC[s][F]))[:, 1]
    _vmask = (R.attack == "clean").values
    SC["val"][f"s_{name}"] = oof[_vmask][:len(SC["val"])]   # validation rows carry out-of-fold scores
    print(f"{name:<13} coefficients (standardized logits): { {k: round(v, 3) for k, v in STACK_COEF[name].items()} }")
for s in SCORE_SETS:
    SC[s]["s_Hybrid-OR"] = np.maximum(SC[s][f"s_{C_STAR}"], SC[s][f"s_{S_STAR}"])   # display only; decisions use flag()

def flag(df, det, alpha=None):
    alpha = alpha or A0
    if det == "Hybrid-OR":
        _, tc, ts = OR_THR[alpha]
        return ((df[f"s_{C_STAR}"] > tc) | (df[f"s_{S_STAR}"] > ts)).values
    return (df[f"s_{det}"] > THR[alpha][det]).values

def triage(df, alpha=None):
    alpha = alpha or A0
    c = df[f"s_{C_STAR}"].values > THR[alpha][C_STAR]; s = df[f"s_{S_STAR}"].values > THR[alpha][S_STAR]
    return np.where(c & s, "block", np.where(~c & ~s, "allow", "review"))

ALL_DETS = BASE_DETS + HYB_DETS
EVAL_DETS = ALL_DETS + ABL_DETS
joblib.dump({"THR": THR, "OR_THR": OR_THR, "C_STAR": C_STAR, "S_STAR": S_STAR, "BEST_LAYER": BEST_LAYER, "STACK_COEF": STACK_COEF},
            DIRS["results"] / "thresholds.joblib")
# The adaptive attacker's natural-padding arm (v3.2) and out-of-distribution arm (v4.0) join its candidate pool (same emails, more candidates)
for arm, ok in (("adaptive_nat", ADAPT_NAT_AVAIL), ("adaptive_ood", ADAPT_OOD_AVAIL)):
    if ok:
        SC[arm]["arm"] = "natural" if arm == "adaptive_nat" else "out-of-distribution"
        SC["adaptive"] = pd.concat([SC["adaptive"].assign(arm=SC["adaptive"].get("arm", "words")), SC.pop(arm)], ignore_index=True)
        SCORE_SETS.remove(arm)
SC["adaptive"]["arm"] = SC["adaptive"]["arm"].fillna("words") if "arm" in SC["adaptive"] else "words"
N_CAND = len(SC["adaptive"]) // max(1, SC["adaptive"].base_id.nunique())
ADAPT_DEFS_EXTRA = [d_ for d_ in (AUG_DET, AUGMIX_DET) if d_ in AVAIL and SC["adaptive"][f"s_{d_}"].notna().all()]
print(f"Adaptive attacker: {N_CAND} candidates per email ({SC['adaptive']['pad'].value_counts().to_dict()})")
for s in SCORE_SETS:
    SC[s].drop(columns=["text"]).to_parquet(DIRS["results"] / f"scores_{s}.parquet", index=False)
print("Raw scores saved (texts excluded) to", DIRS["results"])
