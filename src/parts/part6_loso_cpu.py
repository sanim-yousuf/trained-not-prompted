# %% [markdown]
# ### 6.1 Leave-one-source-out: classical models (CPU, while the GPU jobs run)
# For each source $s$, every classical model is retrained on the training split **without** $s$ (hyper-parameters fixed to the main run; training capped at
# `loso_max_train` emails for time) and scored on the test emails of $s$ and on legitimate validation emails of the *other* sources (used for the threshold).
# For the held-out attack source (`xsrc_source`, Enron by default) the models also score every attacked and control variant of that source's emails.
# Raw scores are cached; thresholds are applied later, so a different FPR target needs no retraining.

# %%
XSRC = CFG["xsrc_source"]
SCORE_SETS_ALL = ["val", "test_clean"] + ATTACKS[1:]

def loso_classical():
    if have("loso_classical.parquet"): return pd.read_parquet(cpath("loso_classical.parquet"))
    rows = []
    for src in sorted(emails.source.unique()):
        part_fn = f"loso_classical_{src}.parquet"
        if have(part_fn):                       # per-source checkpoint: a resumed session continues with the next source
            rows.append(pd.read_parquet(cpath(part_fn))); print(f"  LOSO {src}: cached"); continue
        tr = emails[(emails.split == "train") & (emails.source != src)]
        if len(tr) > CFG["loso_max_train"]: tr = strat_sample(tr, CFG["loso_max_train"], "strat")
        parts = {"val": SETS["val"][(SETS["val"].source != src) & (SETS["val"].label == 0)],
                 "test_clean": SETS["test_clean"][SETS["test_clean"].source == src]}
        if src == XSRC:
            for a in ATTACKS[1:]:
                parts[a] = SETS[a][SETS[a].source == src]
        if len(parts["test_clean"]) == 0 or len(parts["val"]) < 10: continue
        fitted, src_rows = {}, []
        for n, spec in CLASSICAL_SPECS.items():
            if spec["vec"] not in fitted:
                v = VECTORIZERS[spec["vec"]](); Xt = v.fit_transform(tr.text)
                fitted[spec["vec"]] = (v, Xt, {k: v.transform(d.text.tolist()) for k, d in parts.items()})
            v, Xt, Xp = fitted[spec["vec"]]
            clf = spec["clf"](CLASSICAL_HP[n]).fit(Xt, tr.label)
            for k, d in parts.items():
                if len(d) == 0: continue
                src_rows.append(pd.DataFrame(dict(source=src, detector=n, set=k, id=d.id.values, label=d.label.values,
                                                  score=clf.predict_proba(Xp[k])[:, 1])))
            del clf; gc.collect()
        del fitted; gc.collect()
        src_df = pd.concat(src_rows, ignore_index=True); src_df.to_parquet(cpath(part_fn), index=False); rows.append(src_df)
        print(f"  LOSO {src}: classical models done ({len(tr)} training emails) | {mem_status()}", flush=True)
    out = pd.concat(rows, ignore_index=True); out.to_parquet(cpath("loso_classical.parquet"), index=False)
    return out

with Timer("loso_classical"):
    LOSO_CLS = loso_classical()
print(LOSO_CLS.groupby(["source", "set"]).id.nunique().unstack(fill_value=0))
