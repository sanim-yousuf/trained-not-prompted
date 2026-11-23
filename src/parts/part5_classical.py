# %% [markdown]
# ## 6. Classical detectors (CPU, while the GPU jobs run)
#
# Five standard lexical pipelines covering the main model families used in phishing-detection papers: generative (NB), linear (LR, SVM), bagged trees (RF) and boosted trees (XGBoost).
# Each vectorizer is fit once on the training split and shared by the models that use the same features; the main hyper-parameter is selected on the
# **full validation split** by ROC-AUC, then the model is frozen. Grids extend past the values chosen in earlier runs; a remaining edge choice is reported
# with its AUC gain over the neighbouring value (a gain below 0.0002 is a plateau: the data are close to separable and weaker regularization stops mattering).
# `min_samples_leaf=1` is a natural lower bound, not an edge. XGBoost grows trees leaf-wise (grid over the number of leaves) so its memory stays bounded.
# `SVM-char` uses sigmoid calibration (3-fold) so that every detector outputs a probability.

# %%
CPU_JOBS = max(1, (os.cpu_count() or 2) - len(JOBS.slots))   # leave one core per GPU worker
VECTORIZERS = {
    "word12": lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True, strip_accents="unicode", dtype=np.float32),
    "char35": lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300_000, sublinear_tf=True, dtype=np.float32),
    # Tree ensembles use a compact vocabulary (trees gain little from 200k sparse columns and train much faster)
    "word12_20k": lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20_000, sublinear_tf=True, strip_accents="unicode", dtype=np.float32),
}
# XGBoost grows trees leaf-wise with a bounded number of leaves and a bounded histogram cache. Depth-wise growth (max_depth 12 to 16) on 20k sparse
# TF-IDF columns needs a histogram per node at the deepest level, so its RAM grows exponentially with depth; that is what killed the v3.0 kernel.
# Leaf-wise growth with max_cached_hist_node keeps peak RAM near 2 GB for any number of leaves (measured on data of the same shape).
XGB_FIXED = dict(n_estimators=300, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5, tree_method="hist", grow_policy="lossguide",
                 max_depth=0, max_bin=64, max_cached_hist_node=64, random_state=SEED, eval_metric="logloss", verbosity=0)
CLASSICAL_SPECS = {
    "NB": dict(vec="word12", clf=lambda c: MultinomialNB(alpha=c), grid=[0.0001, 0.0003, 0.001, 0.003, 0.01], hp="alpha"),
    "LR-word": dict(vec="word12", clf=lambda c: LogisticRegression(C=c, max_iter=5000, solver="liblinear"), grid=[16.0, 64.0, 256.0, 1024.0], hp="C"),
    "SVM-char": dict(vec="char35", clf=lambda c: CalibratedClassifierCV(LinearSVC(C=c, max_iter=10000), cv=3, method="sigmoid"),
                     grid=[2.0, 8.0, 32.0], hp="C"),
    "RF": dict(vec="word12_20k", clf=lambda c: RandomForestClassifier(n_estimators=400, min_samples_leaf=c, max_features="sqrt", n_jobs=CPU_JOBS,
                                                                      class_weight="balanced_subsample", random_state=SEED),
               grid=[1, 2, 4], hp="min_samples_leaf", natural_low=True),
    "XGBoost": dict(vec="word12_20k", clf=lambda c: XGBClassifier(max_leaves=c, n_jobs=CPU_JOBS, **XGB_FIXED), grid=[31, 127, 511], hp="max_leaves"),
}
PLATEAU = 2e-4   # an edge choice whose AUC gain over its neighbour is below this is reported as a plateau, not a missed optimum

CLASSICAL_HP = {}
def fit_classical(train_df, val_df):
    """Fits each vectorizer once and each model's grid; every finished model is cached on its own, so a crash or a resumed session loses at most one model."""
    report, vecs = [], {}
    for name, spec in CLASSICAL_SPECS.items():
        fn = f"classical_{name}.joblib"
        if have(fn):
            d = joblib.load(cpath(fn)); report += d["report"]; CLASSICAL_HP[name] = d["hp"]
            print(f"  {name:<9} loaded from cache ({spec['hp']}={d['hp']})"); continue
        if spec["vec"] not in vecs:
            vfn = f"vec_{spec['vec']}.joblib"
            vec = joblib.load(cpath(vfn)) if have(vfn) else VECTORIZERS[spec["vec"]]().fit(train_df.text)
            if not have(vfn): joblib.dump(vec, cpath(vfn))
            vecs[spec["vec"]] = (vec, vec.transform(train_df.text), vec.transform(val_df.text))
        vec, Xtr, Xva = vecs[spec["vec"]]
        best, rows, t0 = None, [], time.time()
        for c in spec["grid"]:
            clf = spec["clf"](c).fit(Xtr, train_df.label)
            auc = roc_auc_score(val_df.label, clf.predict_proba(Xva)[:, 1])
            rows.append(dict(model=name, hp_name=spec["hp"], hp=c, val_auc=auc))
            if best is None or auc > best[0]: best = (auc, c, clf)
            del clf; gc.collect()
        aucs = [r["val_auc"] for r in rows]; k = spec["grid"].index(best[1])
        edges = [len(spec["grid"]) - 1] + ([] if spec.get("natural_low") else [0])
        note = ""
        if k in edges:
            nb_ = aucs[k - 1] if k == len(aucs) - 1 else aucs[k + 1]
            note = f" (grid edge, plateau: +{best[0]-nb_:.5f} over neighbour)" if best[0] - nb_ < PLATEAU else f" (grid edge, gain +{best[0]-nb_:.5f})"
        joblib.dump(dict(clf=best[2], hp=best[1], report=rows), cpath(fn))
        CLASSICAL_HP[name] = best[1]; report += rows
        print(f"  {name:<9} best {spec['hp']}={best[1]!s:<7} val AUC={float(best[0]):.5f}{note} | {(time.time()-t0)/60:.1f} min | {mem_status()}", flush=True)
        del best; gc.collect()
    del vecs; gc.collect()
    models, vec_obj = {}, {}
    for name, spec in CLASSICAL_SPECS.items():
        if spec["vec"] not in vec_obj: vec_obj[spec["vec"]] = joblib.load(cpath(f"vec_{spec['vec']}.joblib"))   # shared object, transformed once
        models[name] = Pipeline([("vec", vec_obj[spec["vec"]]), ("clf", joblib.load(cpath(f"classical_{name}.joblib"))["clf"])])
    return models, pd.DataFrame(report)

with Timer("classical_train"):
    tr = emails[emails.split == "train"]; va = emails[emails.split == "val"]
    CLASSICAL, grid_report = fit_classical(tr, va)
    grid_report.to_csv(DIRS["results"] / "classical_grid.csv", index=False)
    del tr, va; gc.collect()

def score_classical(texts):
    """Transforms each distinct vectorizer once, then scores every model that uses it."""
    texts = list(texts); out, feats = {}, {}
    for n, m in CLASSICAL.items():
        key = id(m.named_steps["vec"])
        if key not in feats: feats[key] = m.named_steps["vec"].transform(texts)
        out[n] = m.named_steps["clf"].predict_proba(feats[key])[:, 1]
    return out

# Classical models on the FULL test split (supplementary: shows the balanced subset is representative)
full_test = emails[emails.split == "test"]
_sc = score_classical(full_test.text)
FULL_TEST_AUC = {n: float(roc_auc_score(full_test.label, s)) for n, s in _sc.items()}
print("Full-test ROC-AUC:", {k: round(v, 4) for k, v in FULL_TEST_AUC.items()})

