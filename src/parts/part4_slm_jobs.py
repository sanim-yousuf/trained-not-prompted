# %% [markdown]
# ## 5. Small-LLM detectors: three read-outs of the same backbone (GPU jobs)
#
# For each backbone the email is tokenized and, if longer than the budget, its **head (67%) and tail (33%)** are kept, so prefix and suffix content is never truncated away.
#
# * **ZS (prompted, generative label).** Score $\sigma\big(\log\sum_{w\in\mathcal{M}} P(w\mid x) - \log\sum_{w\in\mathcal{B}} P(w\mid x)\big)$, where $P(w\mid x)$ is the probability of the complete label word
#   (all of its tokens, by teacher forcing on the cached prompt), $\mathcal{M}=\{$MALICIOUS, Malicious$\}$, $\mathcal{B}=\{$BENIGN, Benign$\}$. Deterministic, no sampling.
# * **Probe.** L2 logistic regression on the hidden state of the last prompt token of the **same forward pass**, at the layer chosen by validation AUC.
#   ZS and Probe therefore see identical activations; only the read-out differs.
# * **LoRA.** The backbone with a classification head, LoRA-fine-tuned on raw email text (no instructions in the input).
#
# * **v4.0 additions.** A second model family (Llama-3.2-3B-Instruct if the Hugging Face token has access, otherwise SmolLM2-1.7B-Instruct) read out the same
#   three ways; a 4-shot prompt on the primary backbone and a 7B model (Qwen2.5-7B-Instruct, 4-bit) as stronger prompted baselines; `LoRA-Aug` with three
#   seeds; `LoRA-AugMix` (padding at mixed positions and lengths); and every detector scored on out-of-distribution padding, padded legitimate mail and the
#   modern PhishFuzzer data.
#
# Jobs run in parallel, one per GPU, in priority order. Core jobs always run; optional jobs are deferred to a resumed session if they would end after
# `time_budget_h`, and an optional job that fails is reported and left out of the analysis instead of stopping the notebook. A job whose outputs are partly
# cached only computes the missing sets (a saved adapter is reloaded), so attaching a v3.2 output runs only the v4.0 work.

# %%
EVAL_SETS = ["test_clean"] + ATTACKS[1:]

def _scale(tag):
    try: return float(re.findall(r"[\d.]+", tag)[0]) / 1.5
    except Exception: return 1.0

def _n(sets): return sum(len(SETS[s]) for s in sets)

LEGIT_SETS = ["legit_neutral", "legit_inject"]   # v3.2 collateral false-positive sets
PAD_LEGIT = ["legit_pad", "legit_pad_ood"]        # v4.0 collateral sets with padding on legitimate mail
NEW_EVAL = OOD_SETS + PAD_LEGIT + MODERN_SETS     # v4.0 evaluation sets
ADV_PRIMARY = ["adaptive", "adaptive_nat"] + (["adaptive_ood"] if "adaptive_ood" in SETS else [])

V5 = CFG["v5"]
V5_EVAL = ["legit_realinstr"] if "legit_realinstr" in SETS else []   # v5.0: realistic benign mail with ordinary instructions

def zs_sets(primary):
    return (["probe_train", "val"] + (["val_inj", "val_pad"] if primary else []) + EVAL_SETS + LEGIT_SETS + V5_EVAL + NEW_EVAL
            + (ADV_PRIMARY if primary else []) + (PHISH_SETS if primary else []))   # v5.2: phishing pool on the primary backbone

def lora_sets(primary_main, first=True):
    return (["val"] + (["val_inj", "val_pad"] if primary_main else []) + EVAL_SETS + (LEGIT_SETS + V5_EVAL + NEW_EVAL if first else OOD_SETS)
            + (ADV_PRIMARY if primary_main else []) + (PHISH_SETS if primary_main else []))

# Few-shot demonstrations for the 4-shot prompted baseline: short training emails, half malicious, from different sources (fixed seed)
def fewshot_examples(k):
    tr = emails[(emails.split == "train") & emails.n_chars.between(250, 900)]
    ex = []
    for l in (1, 0):
        d = tr[tr.label == l].sample(frac=1, random_state=SEED + 83 + l).drop_duplicates("source").head(k // 2)
        ex += [[t, int(l)] for t in d.text]
    return [ex[i] for i in np.random.default_rng(SEED + 84).permutation(len(ex))]
PROMPT_SETS = ["val", "test_clean", "neutral", "inject", "inject_aware", "ham_pad", "legit_inject"] + [m for m in ["modern_human", "modern_llm"] if m in MODERN_SETS]

def _rate(job):   # seconds per email, for time estimates
    sc = _scale(job.get("tag", P_TAG))
    if job["kind"] in ("zs", "lora_gen"):
        return 0.09 * sc * (2.5 if job.get("fewshot") else 1.0) * (1.4 if job.get("quant4") else 1.0) * (1.3 if job.get("style") == "spot" else 1.0)
    if job["kind"] == "encoder":
        return 0.012
    return 0.075 * sc * (2.0 if job.get("window") else 1.0) * (1.15 if job.get("prompt_input") else 1.0)

def _est(job, outs):
    k = job["kind"]
    if k in ("greedy", "gcg"):
        if Path(job["out"]).exists(): return 0.0
        tr_ = job["target"].get("rate", {"encoder": 0.012, "gen": 0.09}.get(job["target"]["kind"], 0.075))
        if k == "greedy":
            g = job["greedy"]; return g["n_emails"] * g["steps"] * g["cands"] * tr_ * 0.7 / 60 + 3
        g = job["gcg"]; return g["n_emails"] * g["steps"] * (g["batch"] * tr_ * 0.6 + 0.4) * 0.8 / 60 + 3
    miss = sum(len(SETS[s_]) for s_, o in zip(job["sets"], outs) if not Path(o).exists())
    train = 0.0
    if k == "lora" and not job.get("window") and not (Path(job["adapter_dir"]) / "adapter_config.json").exists():
        train = 0.27 * job["n_train"] * _scale(job["tag"]) * (1.15 if job.get("prompt_input") else 1.0)
    if k == "lora_gen" and not (Path(job["adapter_dir"]) / "adapter_config.json").exists():
        train = 0.33 * job["n_train"] * _scale(job["tag"])
    if k == "encoder" and not (Path(job["model_dir"]) / "config.json").exists():
        train = 0.035 * job["n_train"] * job["enc"]["epochs"]
    return (train + _rate(job) * miss) / 60 + (2 if miss else 0)

def _lora(name, mid, tag, seed, sets, **kw):
    return dict(kind="lora", name=name, model=mid, tag=tag, seed=seed, sets=sets, n_train=CFG["n_lora_train"],
                adapter_dir=str(cpath(f"{name}_adapter")), **kw)

JOBS = JobQueue("slm")
_plan = []   # (priority, job, outputs, optional)
for bi, bb in enumerate(ALL_BACKBONES):
    tag, mid = bb["tag"], bb["model"]
    prim, extra = bi == 0, bb in EXTRA_BB
    ss = zs_sets(prim)
    _plan.append((0 if prim else (2.5 if extra else 2), dict(kind="zs", name=f"zs_{tag}", model=mid, tag=tag, sets=ss,
                  best_layer_only=["adaptive", "adaptive_nat", "adaptive_ood"] + PHISH_SETS), [cpath(f"zs_{tag}_{s}.npz") for s in ss], not prim))
    for k, seed in enumerate(bb["lora_seeds"]):
        main = prim and k == 0; name = f"lora_{tag}_s{seed}"; ls = lora_sets(main, k == 0)
        _plan.append((1 if main else (2.6 if extra else (3 if k == 0 else 4)), _lora(name, mid, tag, seed, ls),
                      [cpath(f"{name}_{s}.npy") for s in ls], not main))
    if prim:
        seed = bb["lora_seeds"][0]
        for src in CFG["lora_loso_sources"]:
            name = f"lora_{tag}_s{seed}_x{src}"; ls = ["val"] + EVAL_SETS
            _plan.append((5, _lora(name, mid, tag, seed, ls, exclude_source=src), [cpath(f"{name}_{s}.npy") for s in ls], True))
        for src in CFG["lora_loso_more"]:
            name = f"lora_{tag}_s{seed}_x{src}"; ls = ["val", "test_clean"]
            _plan.append((6, _lora(name, mid, tag, seed, ls, exclude_source=src), [cpath(f"{name}_{s}.npy") for s in ls], True))
        # Padding-aware training: v3.2 suffix augmentation (3 seeds in v4.0) and v4.0 mixed-position/length augmentation
        if CFG["lora_aug"]["enabled"]:
            aug = dict(frac=CFG["lora_aug"]["frac"], k=CFG["lora_aug"]["k"])
            name = f"lora_{tag}_s{seed}_aug"; ls = lora_sets(True)
            _plan.append((1.5, _lora(name, mid, tag, seed, ls, pad_aug=aug), [cpath(f"{name}_{s}.npy") for s in ls], False))
            for s2 in CFG["lora_aug_seeds"]:
                name = f"lora_{tag}_s{s2}_aug"; ls = ["val"] + EVAL_SETS + OOD_SETS + LEGIT_SETS + V5_EVAL + PAD_LEGIT
                _plan.append((4, _lora(name, mid, tag, s2, ls, pad_aug=aug), [cpath(f"{name}_{s}.npy") for s in ls], True))
        if CFG["lora_augmix"]["enabled"]:
            mx = {k_: v for k_, v in CFG["lora_augmix"].items() if k_ != "enabled"}
            name = f"lora_{tag}_s{seed}_augmix"; ls = lora_sets(True)
            _plan.append((1.6, _lora(name, mid, tag, seed, ls, pad_aug=mx), [cpath(f"{name}_{s}.npy") for s in ls], True))
        # Stronger prompted baselines (no probe; prompt-level comparison only)
        if CFG["prompt_fewshot"]["enabled"]:
            ft = f"fs_{tag}"
            _plan.append((3.5, dict(kind="zs", name=f"zs_{ft}", model=mid, tag=tag, file_tag=ft, sets=PROMPT_SETS, no_probe=True,
                                    fewshot=fewshot_examples(CFG["prompt_fewshot"]["k"])), [cpath(f"zs_{ft}_{s}.npz") for s in PROMPT_SETS], True))
        if LARGE_OK:
            lt = CFG["prompt_large"]["tag"]
            _ls = ["val", "test_clean", "neutral", "inject", "inject_aware"]   # the 7B model is slow in 4-bit: the sets the read-out question needs
            if V5["large_more_sets"]:   # v5.0: natural padding, collateral and modern sets as well (the earlier five stay cached)
                _ls = list(dict.fromkeys(_ls + PROMPT_SETS + ["legit_neutral"] + V5_EVAL + [m for m in ["modern_neutral", "modern_inject_aware"] if m in MODERN_SETS]))
            _ls += [x for x in ["phish_clean", "phish_ham_pad", "phish_neutral", "phish_inject_aware"] if x in PHISH_SETS]   # v5.2
            _plan.append((3.6, dict(kind="zs", name=f"zs_{lt}", model=CFG["prompt_large"]["model"], tag=lt, sets=_ls, no_probe=True,
                                    quant4=CFG["prompt_large"]["quant4"]), [cpath(f"zs_{lt}_{s}.npz") for s in _ls], True))
# ---- v5.0 jobs (every one optional and separately named; with the v4.0 output attached, only these and the new sets run) ----
_AFTER = {}                       # job name -> names of jobs it needs first
V5_FILES, V5_SEED_FILES = {}, {}  # detector -> (file kind, file prefix); extra seed column -> file prefix
_s0 = ALL_BACKBONES[0]["lora_seeds"][0]; _pm = ALL_BACKBONES[0]["model"]
_main = f"lora_{P_TAG}_s{_s0}"
V5_SETS = ["val"] + EVAL_SETS + LEGIT_SETS + V5_EVAL + OOD_SETS + PAD_LEGIT + MODERN_SETS + PHISH_SETS   # v5.2: + phishing pool
_mx = {k_: v for k_, v in CFG["lora_augmix"].items() if k_ != "enabled"}
_add = lambda pr, job, outs, after=(): (_plan.append((pr, job, outs, True)), _AFTER.__setitem__(job["name"], list(after)))
if V5["lora_gen"]:
    _n = f"{_main}_gen"; _ft = f"gen_{P_TAG}"
    _add(1.7, dict(kind="lora_gen", name=_n, model=_pm, tag=P_TAG, seed=_s0, sets=V5_SETS, n_train=CFG["n_lora_train"],
                   adapter_dir=str(cpath(f"{_n}_adapter")), file_tag=_ft), [cpath(f"zs_{_ft}_{s}.npz") for s in V5_SETS])
    V5_FILES[GEN_DET] = ("zs", _ft)
if V5["lora_prompt"]:
    _n = f"{_main}_prompt"
    _add(1.8, _lora(_n, _pm, P_TAG, _s0, V5_SETS, prompt_input=True), [cpath(f"{_n}_{s}.npy") for s in V5_SETS])
    V5_FILES[PROMPTHEAD_DET] = ("npy", _n)
_enc = V5["encoder"]
if _enc["enabled"]:
    for _aug in ([False, True] if _enc["augmix"] else [False]):
        _n = f"enc_{_enc['tag'].lower()}" + ("_augmix" if _aug else "")
        _j = dict(kind="encoder", name=_n, model=_enc["model"], tag="enc", seed=_s0, sets=V5_SETS, n_train=CFG["n_lora_train"],
                  model_dir=str(cpath(f"{_n}_model")), enc={k_: v for k_, v in _enc.items() if k_ not in ("enabled", "augmix", "model", "tag")})
        if _aug: _j["pad_aug"] = _mx
        _add(4.6 if _aug else 1.75, _j, [cpath(f"{_n}_{s}.npy") for s in V5_SETS])
        V5_FILES[ENC_AUGMIX_DET if _aug else ENC_DET] = ("npy", _n)
for _x in V5["augmix_seeds"]:
    _n = f"lora_{P_TAG}_s{_x}_augmix"; _ls = ["val"] + EVAL_SETS + OOD_SETS + LEGIT_SETS + V5_EVAL + PAD_LEGIT
    _add(2.0, _lora(_n, _pm, P_TAG, _x, _ls, pad_aug=_mx), [cpath(f"{_n}_{s}.npy") for s in _ls])
    V5_SEED_FILES[f"{AUGMIX_DET}@s{_x}"] = _n
if V5["lora_mean"]:
    _n = f"{_main}_mean"
    _ls = [x for x in V5_SETS if x not in MODERN_SETS]
    _add(2.5, _lora(_n, _pm, P_TAG, _s0, _ls, pooling="mean"), [cpath(f"{_n}_{s}.npy") for s in _ls])
    V5_FILES[MEAN_DET] = ("npy", _n)
if V5["window"]["enabled"]:
    _n = f"{_main}_win"
    _ls = [x for x in V5_SETS if x not in MODERN_SETS + PHISH_SETS]
    _j = _lora(_n, _pm, P_TAG, _s0, _ls, window={k_: v for k_, v in V5["window"].items() if k_ != "enabled"})
    _j["adapter_dir"] = str(cpath(f"{_main}_adapter"))
    _add(2.6, _j, [cpath(f"{_n}_{s}.npy") for s in _ls], after=[_main])
    V5_FILES[WIN_DET] = ("npy", _n)
_SPOT_SETS = list(dict.fromkeys(PROMPT_SETS + ["legit_neutral"] + V5_EVAL + [m for m in ["modern_neutral", "modern_inject_aware"] if m in MODERN_SETS]))
if V5["spot"]["enabled"]:
    _spot = [(P_TAG, _pm, 2.3)] + ([(EXTRA_BB[0]["tag"], EXTRA_BB[0]["model"], 2.7)] if EXTRA_BB and V5["spot"]["extra_backbone"] else [])
    if LARGE_OK and V5["spot"]["large"]:
        _spot.append((CFG["prompt_large"]["tag"], CFG["prompt_large"]["model"], 8.0))
    for _t, _m, _pr in _spot:
        _ft = f"spot_{_t}"; _big = _t == CFG["prompt_large"]["tag"]
        _ss = [x for x in _SPOT_SETS if x in ("val", "test_clean", "neutral", "inject", "inject_aware", "legit_inject", "legit_realinstr")] if _big else _SPOT_SETS
        _add(_pr, dict(kind="zs", name=f"zs_{_ft}", model=_m, tag=_t, file_tag=_ft, sets=_ss, no_probe=True, style="spot",
                       quant4=bool(_big and CFG["prompt_large"]["quant4"])), [cpath(f"zs_{_ft}_{s}.npz") for s in _ss])
        V5_FILES[SPOT_DET[_t]] = ("zs", _ft)
for _bb in ALL_BACKBONES[1:]:
    for _x in V5["big_seeds"]:
        if _x in _bb["lora_seeds"]: continue
        _n = f"lora_{_bb['tag']}_s{_x}"; _ls = ["val"] + EVAL_SETS
        _add(9, _lora(_n, _bb["model"], _bb["tag"], _x, _ls), [cpath(f"{_n}_{s}.npy") for s in _ls])
        V5_SEED_FILES[f"LoRA-{_bb['tag']}@s{_x}"] = _n

# Score-guided attacks: target specs point at the trained models and their cached validation scores (for the 1%-FPR threshold)
def _tspec(det):
    if det == LORA_P: n_ = _main
    elif det == AUG_DET: n_ = f"{_main}_aug"
    elif det == AUGMIX_DET: n_ = f"{_main}_augmix"
    elif det == GEN_DET:
        return dict(kind="gen", model=_pm, adapter_dir=str(cpath(f"{_main}_gen_adapter")), val_scores=str(cpath(f"zs_gen_{P_TAG}_val.npz")),
                    seed=_s0, n_train=CFG["n_lora_train"], cache=str(DIRS["cache"]), name=f"{_main}_gen"), f"{_main}_gen"
    elif det == ENC_DET:
        n_ = f"enc_{_enc['tag'].lower()}"
        return dict(kind="encoder", model=_enc["model"], model_dir=str(cpath(f"{n_}_model")), val_scores=str(cpath(f"{n_}_val.npy")),
                    enc={k_: v for k_, v in _enc.items() if k_ not in ("enabled", "augmix", "model", "tag")}, seed=_s0, n_train=CFG["n_lora_train"],
                    cache=str(DIRS["cache"]), name=n_), n_
    else: return None, None
    return dict(kind="lora", model=_pm, adapter_dir=str(cpath(f"{n_}_adapter")), val_scores=str(cpath(f"{n_}_val.npy"))), n_
ATTACK_FILES = {"greedy": {}, "gcg": {}}
_slug = lambda d: re.sub(r"[^A-Za-z0-9]+", "_", d)
_gtargets = [LORA_P, AUG_DET, AUGMIX_DET] + ([GEN_DET] if V5["lora_gen"] else []) + ([ENC_DET] if _enc["enabled"] else [])
_ctargets = [LORA_P, AUGMIX_DET] + ([ENC_DET] if _enc["enabled"] else [])
for kind_, on_, targets_, pr_ in (("greedy", V5["greedy"]["enabled"], _gtargets, 2.2), ("gcg", V5["gcg"]["enabled"], _ctargets, 2.4)):
    if not on_: continue
    for d in targets_:
        spec, dep = _tspec(d)
        if spec is None: continue
        out = cpath(f"{kind_}_{_slug(d)}.parquet")
        j = dict(kind=kind_, name=f"{kind_}_{_slug(d)}", target=spec, target_name=d, alpha=CFG["primary_fpr"], seed=SEED + 501, out=str(out),
                 emails=str(cpath(f"sets/{kind_}_emails.parquet")), **({"greedy": V5["greedy"], "sentences": str(cpath("sets/greedy_sentences.parquet"))}
                                                                     if kind_ == "greedy" else {"gcg": V5["gcg"]}))
        _add(pr_ + (0.05 if d != LORA_P else 0), j, [out], after=[dep])
        ATTACK_FILES[kind_][d] = out

# ---- v5.2 jobs (all optional, separately named; attaching the v5.1 output runs only these and the phishing-pool scoring) ----
V52 = CFG["v5"]["v52"]
_seed_sets = ["val"] + EVAL_SETS + OOD_SETS + LEGIT_SETS + V5_EVAL
V52_SEED_GROUPS = {}   # base detector -> [extra seed columns]
for _x in V52["readout_seeds"]:
    if _x == _s0: continue
    for _v in V52["readout_variants"]:
        if _v == "gen" and V5["lora_gen"]:
            _n = f"lora_{P_TAG}_s{_x}_gen"; _ft = f"gen_{P_TAG}_s{_x}"; _c = f"{GEN_DET}@s{_x}"
            _add(1.9, dict(kind="lora_gen", name=_n, model=_pm, tag=P_TAG, seed=_x, sets=_seed_sets, n_train=CFG["n_lora_train"],
                           adapter_dir=str(cpath(f"{_n}_adapter")), file_tag=_ft), [cpath(f"zs_{_ft}_{s}.npz") for s in _seed_sets])
            V5_SEED_FILES[_c] = ("zs", _ft); V52_SEED_GROUPS.setdefault(GEN_DET, []).append(_c)
        elif _v == "prompt" and V5["lora_prompt"]:
            _n = f"lora_{P_TAG}_s{_x}_prompt"; _c = f"{PROMPTHEAD_DET}@s{_x}"
            _add(1.95, _lora(_n, _pm, P_TAG, _x, _seed_sets, prompt_input=True), [cpath(f"{_n}_{s}.npy") for s in _seed_sets])
            V5_SEED_FILES[_c] = _n; V52_SEED_GROUPS.setdefault(PROMPTHEAD_DET, []).append(_c)
        elif _v == "mean" and V5["lora_mean"]:
            _n = f"lora_{P_TAG}_s{_x}_mean"; _c = f"{MEAN_DET}@s{_x}"
            _add(2.05, _lora(_n, _pm, P_TAG, _x, _seed_sets, pooling="mean"), [cpath(f"{_n}_{s}.npy") for s in _seed_sets])
            V5_SEED_FILES[_c] = _n; V52_SEED_GROUPS.setdefault(MEAN_DET, []).append(_c)
        elif _v == "encoder" and _enc["enabled"]:
            _n = f"enc_{_enc['tag'].lower()}_s{_x}"; _c = f"{ENC_DET}@s{_x}"
            _add(1.92, dict(kind="encoder", name=_n, model=_enc["model"], tag="enc", seed=_x, sets=_seed_sets, n_train=CFG["n_lora_train"],
                            model_dir=str(cpath(f"{_n}_model")), enc={k_: v for k_, v in _enc.items() if k_ not in ("enabled", "augmix", "model", "tag")}),
                 [cpath(f"{_n}_{s}.npy") for s in _seed_sets])
            V5_SEED_FILES[_c] = _n; V52_SEED_GROUPS.setdefault(ENC_DET, []).append(_c)

def _zspec(det):
    """Prompted targets for the adaptive injection attack: read out exactly like the cached ZS scores (validation scores give the threshold)."""
    if det == LARGE_DET and LARGE_OK:
        lt = CFG["prompt_large"]["tag"]
        return dict(kind="zs", model=CFG["prompt_large"]["model"], quant4=bool(CFG["prompt_large"]["quant4"]), val_scores=str(cpath(f"zs_{lt}_val.npz")),
                    rate=0.09 * _scale(lt) * 1.4), f"zs_{lt}"
    for b_ in EXTRA_BB:
        if det == f"ZS-{b_['tag']}":
            return dict(kind="zs", model=b_["model"], val_scores=str(cpath(f"zs_{b_['tag']}_val.npz")), rate=0.09 * _scale(b_["tag"])), f"zs_{b_['tag']}"
    return None, None
_ia = V52["inj_attack"]
_IA_DETS = {"gen": GEN_DET if V5["lora_gen"] else None, "lora": LORA_P, "large": LARGE_DET if LARGE_OK else None,
            "extra_zs": f"ZS-{EXTRA_BB[0]['tag']}" if EXTRA_BB else None}
ATTACK_FILES["injatk"] = {}
if _ia["enabled"]:
    _gi = {k_: v for k_, v in _ia.items() if k_ not in ("enabled", "targets", "n_sentences")}
    for _k in _ia["targets"]:
        d = _IA_DETS.get(_k)
        if d is None: continue
        spec, dep = _tspec(d) if _k in ("gen", "lora") else _zspec(d)
        if spec is None: continue
        for _pool, _sf in (("inj", "inj_sentences"), ("ben", "inj_benign_sentences")):
            out = cpath(f"injatk_{_pool}_{_slug(d)}.parquet")
            j = dict(kind="greedy", name=f"injatk_{_pool}_{_slug(d)}", target=spec, target_name=d, alpha=CFG["primary_fpr"], seed=SEED + 601, out=str(out),
                     emails=str(cpath("sets/inj_attack_emails.parquet")), sentences=str(cpath(f"sets/{_sf}.parquet")), greedy=_gi, mode="score")
            _add(2.15 if _k in ("gen", "lora") else 2.35, j, [out], after=[dep])
            ATTACK_FILES["injatk"].setdefault(d, {})[_pool] = out
ATTACK_FILES["phgreedy"], ATTACK_FILES["phlabel"] = {}, {}
if V52["phish"]["enabled"] and V5["greedy"]["enabled"] and PHISH_SETS:
    _gp = dict(V5["greedy"], n_emails=V52["phish"]["n_greedy"])
    _pt = {"lora": LORA_P, "augmix": AUGMIX_DET}
    for mode_, key_, targets_ in (("score", "phgreedy", V52["phish"]["attack_targets"]), ("label", "phlabel", V52["phish"]["label_targets"])):
        for _k in targets_:
            spec, dep = _tspec(_pt[_k])
            if spec is None: continue
            out = cpath(f"{key_}_{_slug(_pt[_k])}.parquet")
            j = dict(kind="greedy", name=f"{key_}_{_slug(_pt[_k])}", target=spec, target_name=_pt[_k], alpha=CFG["primary_fpr"], seed=SEED + 701,
                     out=str(out), emails=str(cpath("sets/greedy_phish_emails.parquet")), sentences=str(cpath("sets/greedy_sentences.parquet")),
                     greedy=_gp, mode=mode_)
            _add(2.45, j, [out], after=[dep])
            ATTACK_FILES[key_][_pt[_k]] = out

for pr, job, outs, opt in sorted(_plan, key=lambda x: x[0]):
    est = _est(job, outs)
    JOBS.submit(job, outs, est_min=est, optional=opt, after=_AFTER.get(job["name"], ()))
    print(f"  queued {job['name']:<30} est {est:6.1f} min  {'optional' if opt else 'core'}")
with Timer("slm_jobs_launch"):
    JOBS.start()
print("GPU jobs run in the background; the notebook continues with the CPU stages (classical models and classical LOSO).")
