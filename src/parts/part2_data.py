# %% [markdown]
# ## 3. Data: loading, cleaning, de-duplication and splits
#
# The Kaggle *Phishing Email Dataset* ships one CSV per source corpus (CEAS-08, Enron, Ling, Nazario, Nigerian Fraud, SpamAssassin) plus a merged file.
# We load the per-source files (the merged file duplicates them and is skipped), build `text = "Subject: ..." + body`, map labels to
# **1 = malicious (phishing/spam/fraud)** and **0 = legitimate**, and keep the source of every email for source-aware analysis.
#
# *Threats to validity handled here:* (i) exact and near-duplicate emails (normalized hashing of lower-cased text with URLs and digits masked) are removed, so the same message cannot appear in train and test;
# (ii) duplicates that carry conflicting labels are dropped entirely; (iii) Nazario and Nigerian Fraud contain only malicious emails, so a model could learn *source style* rather than maliciousness. The leave-one-source-out analysis in Section 10.9 quantifies this.

# %%
SOURCE_ALIASES = {"ceas_08": "CEAS-08", "ceas08": "CEAS-08", "ceas": "CEAS-08", "enron": "Enron", "ling": "Ling",
                  "nazario": "Nazario", "nigerian_fraud": "Nigerian-Fraud", "nigerian": "Nigerian-Fraud",
                  "spamassasin": "SpamAssassin", "spamassassin": "SpamAssassin", "spam_assassin": "SpamAssassin"}
COMBINED_STEMS = {"phishing_email", "phishing_emails", "phishing_email_dataset", "combined"}
URL_RE = re.compile(r"(https?://[^\s<>\"']+|www\.[^\s<>\"']+)", re.I)

def _label_map(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)): return int(v > 0)
    s = str(v).strip().lower()
    if s in {"1", "phishing", "phishing email", "spam", "malicious", "fraud", "scam"}: return 1
    if s in {"0", "safe", "safe email", "ham", "legitimate", "benign", "legit"}: return 0
    return np.nan

def _read_one(path, source):
    df = pd.read_csv(path, encoding_errors="replace", on_bad_lines="skip", low_memory=False)
    cols = {c.lower().strip(): c for c in df.columns}
    body_col = next((cols[c] for c in ["body", "text_combined", "text", "email text", "message", "content"] if c in cols), None)
    lab_col = next((cols[c] for c in ["label", "class", "email type", "is_phishing", "target"] if c in cols), None)
    if body_col is None or lab_col is None:
        print(f"  skip {path.name}: no text/label column ({list(df.columns)[:8]})"); return None
    body = df[body_col].fillna("").astype(str)
    if "subject" in cols:
        subj = df[cols["subject"]].fillna("").astype(str).str.strip()
        text = np.where(subj.str.len() > 0, "Subject: " + subj + "\n\n" + body, body)
    else:
        text = body.values
    out = pd.DataFrame({"text": text, "label": df[lab_col].map(_label_map), "source": source})
    return out.dropna(subset=["label"]).astype({"label": int})

def load_corpus(root):
    csvs = sorted(p for p in Path(root).rglob("*.csv") if "/cbs/" not in str(p).replace("\\", "/"))
    per, comb_ = [], []
    for p in csvs:
        stem = re.sub(r"[\s\-]+", "_", p.stem.lower())
        if stem in SOURCE_ALIASES: per.append((SOURCE_ALIASES[stem], p))
        elif stem in COMBINED_STEMS: comb_.append(p)
    if not per:   # generic fallback: any CSV with text + label, one source per file
        per = [(p.stem, p) for p in csvs if p not in comb_] or [("combined", p) for p in comb_]
    elif comb_:
        print(f"Skipping merged file(s) {[p.name for p in comb_]} (they duplicate the per-source files).")
    frames = []
    for src, p in per:
        d = _read_one(p, src)
        if d is not None:
            print(f"  {p.name:<28} -> {src:<15} n={len(d):>6}  malicious={d.label.mean():.2f}")
            frames.append(d)
    if not frames: raise FileNotFoundError(f"No usable CSV under {root}. Attach the Kaggle 'Phishing Email Dataset'.")
    return pd.concat(frames, ignore_index=True)

def _clean_text(t):
    t = str(t).replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t[:CFG["max_chars"]]

def _norm_key(t):
    t = URL_RE.sub(" URL ", t.lower())
    t = re.sub(r"\d+", "0", t)
    t = re.sub(r"[^a-z0 ]+", " ", t)
    return hashlib.sha1(re.sub(r"\s+", " ", t).strip().encode()).hexdigest()

def strat_sample(df, n, key, seed=SEED):
    return S.strat_sample(df, n, key, seed)

with Timer("data"):
    if have("emails.parquet"):
        emails = pd.read_parquet(cpath("emails.parquet"))
        DATA_STATS = json.loads(cpath("data_stats.json").read_text())
        print(f"Loaded cached corpus: {len(emails)} emails")
    else:
        root = Path(CFG["data_root"])
        if not root.exists() or not any(root.rglob("*.csv")):
            print(f"Data path {root} not found or empty; searching {CFG['input_root']} instead.")
            root = Path(CFG["input_root"])
        print(f"Loading CSVs from {root}")
        raw = load_corpus(root)
        st = {"raw": len(raw)}
        raw["text"] = raw["text"].map(_clean_text)
        raw = raw[raw["text"].str.len() >= CFG["min_chars"]]
        st["after_length_filter"] = len(raw)
        raw["key"] = raw["text"].map(_norm_key)
        conflict = raw.groupby("key")["label"].nunique()
        raw = raw[~raw["key"].isin(conflict[conflict > 1].index)]
        st["label_conflicts_removed"] = int((conflict > 1).sum())
        emails = raw.drop_duplicates("key").reset_index(drop=True)
        st["after_dedup"] = len(emails)
        emails["id"] = [f"e{i:06d}" for i in range(len(emails))]
        emails["strat"] = emails["source"] + "|" + emails["label"].astype(str)
        vc = emails["strat"].value_counts()
        emails.loc[emails["strat"].isin(vc[vc < 10].index), "strat"] = "rare"
        tr, te = train_test_split(emails, test_size=0.2, stratify=emails["strat"], random_state=SEED)
        tr, va = train_test_split(tr, test_size=0.125, stratify=tr["strat"], random_state=SEED)
        emails["split"] = "train"
        emails.loc[emails["id"].isin(va["id"]), "split"] = "val"
        emails.loc[emails["id"].isin(te["id"]), "split"] = "test"
        emails["n_chars"] = emails["text"].str.len()
        emails = emails.drop(columns=["key"])
        emails.to_parquet(cpath("emails.parquet"), index=False)
        DATA_STATS = st
        cpath("data_stats.json").write_text(json.dumps(st))

print(json.dumps(DATA_STATS, indent=1))
print(emails.groupby(["split", "label"]).size().unstack().rename(columns={0: "legit", 1: "malicious"}))

# Fixed evaluation subsets shared by ALL detectors (paired comparisons). Validation keeps every legitimate email (or n_val_legit) because the
# 1% FPR threshold is a tail quantile of legitimate scores: with 1,000 emails it would rest on about 10 of them.
def _subset(split, n_legit, n_mal, tag):
    out = []
    for lab, n in ((0, n_legit), (1, n_mal)):
        pool = emails[(emails.split == split) & (emails.label == lab)]
        out.append(pool.copy() if n is None else strat_sample(pool, n, "strat", seed=SEED + lab))
    s = pd.concat(out).sample(frac=1, random_state=SEED).reset_index(drop=True)
    s["set"] = tag
    return s

VAL = _subset("val", CFG["n_val_legit"], CFG["n_val_mal"], "val")
TEST = _subset("test", CFG["n_test_legit"], CFG["n_test_mal"], "test_clean")
print(f"Validation: {len(VAL)} ({(VAL.label == 0).sum()} legitimate, {(VAL.label == 1).sum()} malicious) | "
      f"Test: {len(TEST)} ({(TEST.label == 0).sum()} legitimate, {(TEST.label == 1).sum()} malicious), source-stratified")
assert not set(VAL.id) & set(TEST.id)

# %% [markdown]
# ### 3.1 Dataset figure and table

# %%
by_src = emails.groupby(["source", "label"]).size().unstack(fill_value=0).rename(columns={0: "Legitimate", 1: "Malicious"})
for c in ["Legitimate", "Malicious"]:
    if c not in by_src: by_src[c] = 0
by_src = by_src.loc[by_src.sum(axis=1).sort_values().index]

fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
y = np.arange(len(by_src))
ax.barh(y, by_src["Legitimate"], color=PAL[0], height=0.62, label="Legitimate", edgecolor="white", linewidth=1)
ax.barh(y, by_src["Malicious"], left=by_src["Legitimate"], color=PAL[1], height=0.62, label="Malicious", edgecolor="white", linewidth=1)
for yi, tot in zip(y, by_src.sum(axis=1)):
    ax.text(tot * 1.01, yi, f"{tot:,}", va="center", fontsize=6.5, color=INK2)
ax.set_yticks(y, by_src.index); ax.set_xlabel("Emails after de-duplication")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.grid(axis="y", visible=False); ax.legend(loc="lower right"); ax.set_xlim(0, by_src.sum(axis=1).max() * 1.15)
panel_label(ax, "(a)", x=-0.3)
ax = axes[1]
bins = np.logspace(np.log10(CFG["min_chars"]), np.log10(CFG["max_chars"] + 1), 40)
for lab, name, col in [(0, "Legitimate", PAL[0]), (1, "Malicious", PAL[1])]:
    ax.hist(emails.loc[emails.label == lab, "n_chars"], bins=bins, histtype="step", lw=1.4, color=col, label=name, density=True)
ax.set_xscale("log"); ax.set_xlabel("Email length (characters, truncated at %d)" % CFG["max_chars"]); ax.set_ylabel("Density")
ax.legend(loc="upper left"); panel_label(ax, "(b)")
fig.tight_layout()
savefig(fig, "fig02_dataset", "Corpus composition by source and label (a) and email length distribution (b).")

tab = by_src.iloc[::-1].reset_index()
tab["Total"] = tab["Legitimate"] + tab["Malicious"]
for sp in ["train", "val", "test"]:
    tab[sp.capitalize()] = tab["source"].map(emails[emails.split == sp].groupby("source").size()).fillna(0).astype(int)
tab = tab.rename(columns={"source": "Source"})
tot = {"Source": "Total", **{c: tab[c].sum() for c in tab.columns if c != "Source"}}
tab = pd.concat([tab, pd.DataFrame([tot])], ignore_index=True)
tab_fmt = tab.copy()
for c in tab_fmt.columns[1:]: tab_fmt[c] = tab_fmt[c].map(lambda v: f"{int(v):,}")
print(write_latex(tab_fmt, "tab01_dataset", "Corpus statistics after cleaning and de-duplication.", "tab:dataset",
                  note=f"Raw emails: {DATA_STATS['raw']:,}; removed by de-duplication: {DATA_STATS['after_length_filter']-DATA_STATS['after_dedup']:,}; label-conflicting groups dropped: {DATA_STATS['label_conflicts_removed']:,}."))
display(tab)

# %% [markdown]
# ### 3.2 Experimental pipeline figure

# %%
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(COL2, 2.45))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    def box(x, y, w, h, title, sub, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=col + "1f", ec=col, lw=0.9))
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=7, fontweight="bold")
        ax.text(x + w / 2, y + h - 6.8, sub, ha="center", va="top", fontsize=5.9, color=INK2, linespacing=1.25)
    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK2, mutation_scale=8))
    box(0.5, 15, 14, 29, "Corpus", "6 public sources\ndedup + conflict\nremoval\nsource-stratified\nsplits + LOSO", NEUTRAL)
    box(18.5, 24, 23, 20, "Attacks | matched controls", "good-word | natural padding\ninjection | neutral suffix\nlabel-aware injection\nLLM paraphrase (Phi-3.5)\npara.+inject., adaptive", PAL[7])
    box(18.5, 1, 23, 19, "Defender knowledge", "validation at fixed FPR\ninjection wording T_train\n(disjoint from T_test)\ndisjoint padding pool", NEUTRAL)
    box(45.5, 32, 22, 12, "Classical ML", "NB, LR, SVM, RF, XGBoost", PAL[0])
    box(45.5, 13.5, 22, 15.5, "Same SLM, three read-outs", "prompted label | probe | LoRA\nQwen2.5 1.5B and 3B\n3 LoRA seeds", PAL[1])
    box(45.5, 0.5, 22, 10, "Injection detector", "char TF-IDF + LR (cross-fit)", PAL[6])
    box(71, 13, 13.5, 31, "Hybrids", "FPR-matched OR\nstacking\n(cross-fitted)\nstacking + pad.\naugmentation\ntriage", PAL[2])
    box(88, 4, 11.5, 40, "Evaluation", "DR, ASR\nbootstrap CI\nMcNemar+Holm\nYule's Q\nadaptive ASR@k\nLOSO incl. LoRA\nheld-out source\nattack matrix\nseeds, latency", INK2)
    arrow(14.7, 34, 18.3, 34); arrow(14.7, 20, 18.3, 12); arrow(41.7, 36, 45.3, 38); arrow(41.7, 30, 45.3, 22)
    arrow(41.7, 8, 45.3, 6); arrow(67.7, 38, 70.8, 34); arrow(67.7, 21, 70.8, 26); arrow(67.7, 6, 70.8, 17)
    arrow(84.7, 27, 87.8, 27)
    fig.tight_layout()
    return fig
savefig(draw_pipeline(), "fig01_pipeline", "Experimental design: every attack has a matched control, and one backbone is read out three ways at two scales.")
