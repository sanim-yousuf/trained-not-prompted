# %% [markdown]
# ## 14b. Generator-artifact check (v5.3)
# The validity audit found generator artifacts in LLM-written text: placeholders (`[Your Name]`, `[URL]`), notes about the rewrite, duplicated second
# versions and leaked prompt instructions in paraphrases, and "(Word count: 90)" or meta-commentary in the ordinary-instruction paragraphs. This section
# flags such artifacts in **every** paraphrase and every ordinary-instruction email with fixed patterns, checks the flags against the audit sheet when it
# is attached, and re-reports the affected results on the artifact-free emails. No model is rerun: all scores are cached.

# %%
_P_ART = [
    (r"\[(?:url|link|website|your[^\]]{0,40}|name|recipient[^\]]{0,30}|company[^\]]{0,30}|insert[^\]]{0,40})\]", "placeholder"),
    (r"\b(?:rephrased|rewritten|paraphrased|revised)\s+(?:email|version|message|text|content)\b", "rewrite note"),
    (r"\bhas\s+been\s+(?:rephrased|rewritten|paraphrased|revised)\b|\bas per the instructions?\b", "rewrite note"),
    (r"please note that (?:all )?(?:the )?(?:urls|rewritten|revised|content above|email above)", "rewrite note"),
    (r"\bensure to retain\b|^\s*output\s*:|\boutput\s*:\s*subject\s*:", "prompt leak"),
    (r"rewritten email\s*:|here is the (?:rewritten|rephrased|paraphrased)", "rewrite note"),
]
_R_ART = [
    (r"\(\s*word count\s*:?\s*\d+\s*\)", "word count"),
    (r"\bthis paragraph (?:outlines|provides|describes|explains|includes)\b", "meta-commentary"),
    (r"\bthe (?:above|following) paragraph\b", "meta-commentary"),
]
def _arts(text, pats):
    t = str(text); found = [lab for p, lab in pats if re.search(p, t, flags=re.I | re.M)]
    return sorted(set(found))
def _dup_subject(para, orig):
    return len(re.findall(r"subject\s*:", str(para), flags=re.I)) > len(re.findall(r"subject\s*:", str(orig), flags=re.I))
_URL = re.compile(r"(?:https?[ \t]*:[ \t]*/[ \t]*/[ \t]*|www[ \t]*\.[ \t]*)(?:[\w%~#+]+|[ \t]*[./?=&:_\-][ \t]*)+", re.I)   # tolerates tokenized "http : / / a . com"
def _urls(t):
    return {re.sub(r"[ \t]+", "", u).rstrip(".,;:!?-").lower() for u in _URL.findall(str(t))} - {""}
def _payload_lost(orig, para):   # a URL of the original that does not appear verbatim in the rewrite (missing or altered)
    w = re.sub(r"[ \t]+", "", str(para)).lower(); return any(len(u) > 10 and u not in w for u in _urls(orig))

# ---- flag every paraphrase (the POOL-level rewrite reused by the paraphrase and para_inject sets) ----
_PA = PARA[["base_id", "para_text", "valid"]].merge(POOL[["id", "text"]].rename(columns={"id": "base_id", "text": "orig"}), on="base_id", how="left")
_PA["kinds"] = [_arts(p, _P_ART) + (["duplicate version"] if _dup_subject(p, o) else []) for p, o in zip(_PA.para_text, _PA.orig)]
_PA["artifact"] = _PA.kinds.map(bool)
_PA["payload_lost"] = [_payload_lost(o, p) for o, p in zip(_PA.orig, _PA.para_text)]
PARA_ART = _PA.set_index("base_id")
_pv = _PA[_PA.valid.astype(bool)]
ART_COUNTS = {"paraphrase_valid": int(len(_pv)), "paraphrase_artifact": int(_pv.artifact.sum()), "paraphrase_payload_lost": int(_pv.payload_lost.sum()),
              "paraphrase_artifact_kinds": pd.Series([k for ks in _pv.kinds for k in ks]).value_counts().to_dict()}
# ---- flag every ordinary-instruction email ----
RI_ART = pd.Series(dtype=object)
if "legit_realinstr" in SC:
    _ri = SC["legit_realinstr"]
    RI_ART = pd.Series([_arts(t, _R_ART) for t in _ri.text], index=_ri.index)
    ART_COUNTS.update({"realinstr_n": int(len(_ri)), "realinstr_artifact": int(RI_ART.map(bool).sum()),
                       "realinstr_artifact_kinds": pd.Series([k for ks in RI_ART for k in ks]).value_counts().to_dict()})
print(json.dumps(ART_COUNTS, indent=1))
pd.DataFrame({"base_id": _PA.base_id, "valid": _PA.valid, "artifact": _PA.artifact, "kinds": _PA.kinds.map("; ".join),
              "payload_lost": _PA.payload_lost}).to_csv(DIRS["private"] / "paraphrase_artifact_flags.csv", index=False)

# ---- agreement of the pattern flags with the audit (if the filled sheets are attached) ----
ART_AUDIT = {}
for f in glob.glob(str(Path(CFG["input_root"]) / "**/paraphrase_audit_filled.csv"), recursive=True):
    a = pd.read_csv(f)
    fcols = [c for c in a.columns if c.endswith("fluent_yn")]
    if not fcols: continue
    a["human_artifact_or_disfluent"] = (a[fcols].min(axis=1) == 0)
    a["flag"] = a.base_id.map(PARA_ART.artifact).fillna(False).astype(bool)
    tp = int((a.flag & a.human_artifact_or_disfluent).sum()); fp = int((a.flag & ~a.human_artifact_or_disfluent).sum())
    fn = int((~a.flag & a.human_artifact_or_disfluent).sum())
    ART_AUDIT["paraphrase"] = dict(n=len(a), human_disfluent=int(a.human_artifact_or_disfluent.sum()), flagged=int(a.flag.sum()), tp=tp, fp=fp, fn=fn,
                                   precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
                                   missed_ids=a.loc[~a.flag & a.human_artifact_or_disfluent, "base_id"].tolist())
    pcols = [c for c in a.columns if c.endswith("payload_kept_yn")]
    if pcols:
        a["human_payload_lost"] = a[pcols].min(axis=1) == 0; a["pl"] = a.base_id.map(PARA_ART.payload_lost).fillna(False).astype(bool)
        ART_AUDIT["payload"] = dict(human_lost=int(a.human_payload_lost.sum()), flagged=int(a.pl.sum()), both=int((a.pl & a.human_payload_lost).sum()))
    break
for f in glob.glob(str(Path(CFG["input_root"]) / "**/realinstr_audit_filled.csv"), recursive=True):
    a = pd.read_csv(f); pcols = [c for c in a.columns if c.endswith("plausible_legitimate_yn")]
    if not pcols or not len(RI_ART): continue
    _ids = dict(zip(SC["legit_realinstr"].id, RI_ART.map(bool)))
    a["human_bad"] = a[pcols].min(axis=1) == 0; a["flag"] = a.id.map(_ids).fillna(False).astype(bool)
    ART_AUDIT["realinstr"] = dict(n=len(a), human_implausible=int(a.human_bad.sum()), flagged=int(a.flag.sum()),
                                  tp=int((a.flag & a.human_bad).sum()), fp=int((a.flag & ~a.human_bad).sum()), fn=int((~a.flag & a.human_bad).sum()))
    break
print("Pattern flags vs audit:", json.dumps(ART_AUDIT, indent=1) if ART_AUDIT else "audit sheets not attached")

# %% [markdown]
# ### 14b.1 Paraphrase results with and without artifacts
# Detection on all valid paraphrases (as reported), on the artifact-free ones, and on the flagged ones. A detector whose rate does not move when the
# artifacts are removed is not being helped (or hurt) by them. The artifact-free subset also drops paraphrases in which a URL of the original is missing or altered.

# %%
_dets53 = [d for d in dict.fromkeys(EVAL_DETS + PROMPT_AVAIL + V5_ALL + ["Hybrid-OR"]) if d and (d == "Hybrid-OR" or d in THR[A0])]
_rows = []
for s_ in ["paraphrase", "para_inject"]:
    if s_ not in SC: continue
    df_ = attack_frame(s_); art = df_.base_id.map(PARA_ART.artifact).fillna(False).astype(bool).values
    lost = df_.base_id.map(PARA_ART.payload_lost).fillna(False).astype(bool).values
    keep = ~art & ~lost
    for d in _dets53:
        if not (d == "Hybrid-OR" or _has(d, s_)): continue
        f_ = flag(df_, d).astype(float)
        m_all = f_.mean(); mc = boot_mean(f_[keep]); ma = f_[art].mean() if art.any() else np.nan
        _rows.append(dict(set=s_, detector=d, n_all=len(f_), n_clean=int(keep.sum()), n_artifact=int(art.sum()), dr_all=m_all, dr_clean=mc[0],
                          dr_clean_lo=mc[1], dr_clean_hi=mc[2], dr_artifact=ma, clean_minus_all=mc[0] - m_all))
PARA_CLEAN = pd.DataFrame(_rows); PARA_CLEAN.to_csv(DIRS["results"] / "v53_paraphrase_clean.csv", index=False)
if len(PARA_CLEAN):
    display(PARA_CLEAN[PARA_CLEAN.set == "paraphrase"][["detector", "n_clean", "dr_all", "dr_clean", "dr_artifact", "clean_minus_all"]].round(4))
    print("Largest absolute change, clean vs all:", PARA_CLEAN.assign(a=PARA_CLEAN.clean_minus_all.abs()).sort_values("a", ascending=False)
          .head(3)[["set", "detector", "dr_all", "dr_clean"]].round(4).to_dict(orient="records"))

# %% [markdown]
# ### 14b.2 False positives on ordinary-instruction mail with and without artifacts
# The same rates on legitimate emails whose added paragraph carries no artifact, overall and for IT notices, for every seed where scores exist.

# %%
_rows = []
if len(RI_ART):
    _ri = SC["legit_realinstr"]; clean_ri = ~RI_ART.map(bool).values
    it = _ri.kind.astype(str).str.startswith("an IT").values
    _ricols = list(dict.fromkeys(_dets53 + [c for g_ in SEED_GROUPS.values() for c in g_] + [c for v_ in V52_SEED_GROUPS.values() for c in v_]))
    for d in _ricols:
        if not (d == "Hybrid-OR" or _has(d, "legit_realinstr")): continue
        f_ = flag(_ri, d).astype(float)
        mc = boot_mean(f_[clean_ri])
        _rows.append(dict(detector=d, n_all=len(f_), n_clean=int(clean_ri.sum()), fpr_all=f_.mean(), fpr_clean=mc[0], fpr_clean_lo=mc[1], fpr_clean_hi=mc[2],
                          fpr_artifact=f_[~clean_ri].mean() if (~clean_ri).any() else np.nan,
                          n_it_clean=int((it & clean_ri).sum()), fpr_it_all=f_[it].mean() if it.any() else np.nan,
                          fpr_it_clean=f_[it & clean_ri].mean() if (it & clean_ri).any() else np.nan))
RI_CLEAN = pd.DataFrame(_rows); RI_CLEAN.to_csv(DIRS["results"] / "v53_realinstr_clean.csv", index=False)
if len(RI_CLEAN): display(RI_CLEAN[["detector", "n_clean", "fpr_all", "fpr_clean", "fpr_artifact", "fpr_it_all", "fpr_it_clean"]].round(4))

# %% [markdown]
# ### 14b.3 Tables and key findings (v5.3)

# %%
if len(PARA_CLEAN):
    _t = PARA_CLEAN[PARA_CLEAN.set == "paraphrase"]
    t46 = pd.DataFrame({"Detector": _t.detector, "All valid": _t.dr_all.map(pct), "Artifact-free": _t.dr_clean.map(pct),
                        "Flagged": _t.dr_artifact.map(pct), "Change (pts)": _t.clean_minus_all.map(lambda v: f"{100*v:+.1f}")})
    write_latex(t46, "tab46_paraphrase_clean", "Paraphrase detection (\\%) on all valid paraphrases and on those without generator artifacts.", "tab:paraclean")
if len(RI_CLEAN):
    t47 = pd.DataFrame({"Detector": RI_CLEAN.detector, "FPR all": RI_CLEAN.fpr_all.map(pct), "FPR artifact-free": RI_CLEAN.fpr_clean.map(pct),
                        "IT notices, all": RI_CLEAN.fpr_it_all.map(pct), "IT notices, artifact-free": RI_CLEAN.fpr_it_clean.map(pct)})
    write_latex(t47, "tab47_realinstr_clean", "False-positive rate (\\%) on ordinary-instruction mail, with and without artifact-bearing emails.", "tab:riclean")
F53 = [f"Artifacts: {ART_COUNTS.get('paraphrase_artifact')}/{ART_COUNTS.get('paraphrase_valid')} valid paraphrases "
       f"({ART_COUNTS.get('paraphrase_artifact_kinds')}), payload lost in {ART_COUNTS.get('paraphrase_payload_lost')}; "
       f"{ART_COUNTS.get('realinstr_artifact')}/{ART_COUNTS.get('realinstr_n')} ordinary-instruction emails ({ART_COUNTS.get('realinstr_artifact_kinds')})."]
if ART_AUDIT: F53.append(f"Pattern flags vs audit: {ART_AUDIT}")
if len(PARA_CLEAN):
    _t = PARA_CLEAN[PARA_CLEAN.set == "paraphrase"]
    F53.append("Paraphrase DR all -> artifact-free: " + "; ".join(f"{r.detector} {100*r.dr_all:.1f} -> {100*r.dr_clean:.1f}" for _, r in _t.iterrows()))
    F53.append(f"Max absolute change over detectors and paraphrase sets: {100*PARA_CLEAN.clean_minus_all.abs().max():.1f} points")
if len(RI_CLEAN):
    F53.append("Ordinary-instruction FPR all -> artifact-free (IT notices all -> artifact-free): " + "; ".join(
        f"{r.detector} {100*r.fpr_all:.1f} -> {100*r.fpr_clean:.1f} ({100*r.fpr_it_all:.0f} -> {100*r.fpr_it_clean:.0f})" for _, r in RI_CLEAN.iterrows()))
print("KEY FINDINGS v5.3\n" + "\n".join(F53))
with open(DIRS["results"] / "key_findings.md", "a") as f_:
    f_.write("\n\n## v5.3 generator-artifact check\n" + "\n".join(F53))
V53_SUMMARY = {"artifact_counts": ART_COUNTS, "artifact_vs_audit": ART_AUDIT, "paraphrase_clean": PARA_CLEAN.round(5).to_dict(orient="records"),
               "realinstr_clean": RI_CLEAN.round(5).to_dict(orient="records")}
