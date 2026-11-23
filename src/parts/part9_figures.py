# %% [markdown]
# ## 9. Figures
# Every figure is saved as vector PDF and 600-dpi PNG in `cbs/figures/` and shown inline. Rates that sit near 0 or 1 are plotted as miss rates or ASR on a
# **log scale** (zeros drawn hollow at the floor and labeled), so differences such as 0.1% versus 1% versus 4% stay visible. Rows are grouped by family with separators.

# %%
MSZ = {d: (7 if DET_MARK.get(d) in ("*", "P", "p") else 4.8) for d in DET_COL}
tc = SC["test_clean"]
def stars(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
def group_breaks(rows):
    fam = [FAMILY[d] for d in rows]; return [i - 0.5 for i in range(1, len(rows)) if fam[i] != fam[i - 1]]
def att_tick(a): return ATT_LABEL[a]
FLOOR = 1e-3
def floor_(v): return np.maximum(np.asarray(v, float), FLOOR)
def logpct_axis(axis):
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{100*v:g}%"))
ATT_BREAKS = [0.5, 2.5, 3.5, 6.5]   # clean | padding | paraphrase | suffix/injection | combined
def mark_controls(ax, cols, axis="x"):
    labs = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for lab, c in zip(labs, cols):
        if c in CONTROLS: lab.set_fontstyle("italic"); lab.set_color(INK2)

# %% [markdown]
# ### Fig. 3: ROC and DET curves on clean test data

# %%
ROC_DETS = BASE_DETS + ["Hybrid-OR"]
fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.8))
y = tc.label.values
for d in ROC_DETS:
    s = tc[f"s_{d}"].values; fpr, tpr, _ = roc_curve(y, s)
    kw = dict(color=DET_COL[d], ls=DET_LS[d], lw=DET_LW[d])
    axes[0].plot(np.maximum(fpr, 1e-4), tpr, label=f"{d} ({roc_auc_score(y, s):.3f})", **kw)
    f = flag(tc, d)
    axes[0].plot(max(f[y == 0].mean(), 1e-4), f[y == 1].mean(), marker=DET_MARK[d], color=DET_COL[d], mfc=DET_FILL[d], ms=MSZ[d], mew=0.8, ls="none")
    fnr = 1 - tpr; ok = (fpr > 0) & (fnr > 0)
    axes[1].plot(norm.ppf(np.clip(fpr[ok], 1e-4, 0.5)), norm.ppf(np.clip(fnr[ok], 1e-4, 0.5)), **kw)
axes[0].set_xscale("log"); axes[0].set_xlim(1e-3 if not SMOKE else 1e-2, 1); axes[0].set_ylim(0, 1.01)
axes[0].axvline(A0, color=INK2, lw=0.7, ls=":"); axes[0].text(A0 * 1.1, 0.03, f"operating FPR = {A0:g}", fontsize=6.2, color=INK2)
axes[0].set_xlabel("False-positive rate (log scale)"); axes[0].set_ylabel("True-positive rate"); logpct_axis(axes[0].xaxis); panel_label(axes[0], "(a)")
tk = [0.001, 0.005, 0.02, 0.1, 0.3, 0.5]
for setter, lim in ((axes[1].set_xticks, None), (axes[1].set_yticks, None)):
    setter(norm.ppf(tk), [f"{100*t:g}%" for t in tk])
axes[1].set_xlim(norm.ppf(1e-3 if not SMOKE else 1e-2), norm.ppf(0.5)); axes[1].set_ylim(norm.ppf(1e-3 if not SMOKE else 1e-2), norm.ppf(0.5))
axes[1].axvline(norm.ppf(A0), color=INK2, lw=0.7, ls=":")
axes[1].set_xlabel("False-positive rate (normal deviate scale)"); axes[1].set_ylabel("Miss rate (normal deviate scale)"); panel_label(axes[1], "(b)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.2),
           title="Detector (clean ROC-AUC); dashed = classical, solid = SLM 1.5B, dotted = SLM 3B, heavy = hybrid", title_fontsize=6.6)
fig.tight_layout()
savefig(fig, "fig03_roc_det", f"Clean-test ROC (a; markers = validation-fixed operating point at FPR {A0:g}) and DET curves (b), where lower-left is better and differences among strong detectors are visible.")

# %% [markdown]
# ### Fig. 4: Calibration with equal-count bins

# %%
nb = len(BASE_DETS); ncol = 4; nrow = int(np.ceil(nb / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(COL2, 1.55 * nrow + 0.3), sharex=True, sharey=True)
for ax, d in zip(axes.ravel(), BASE_DETS):
    p = np.clip(tc[f"s_{d}"].values, 0, 1); yy_ = tc.label.values; order = np.argsort(p, kind="stable")
    chunks = [c for c in np.array_split(order, CFG["ece_bins"]) if len(c)]
    mp = [p[c].mean() for c in chunks]; mo = [yy_[c].mean() for c in chunks]
    ax.hist(p, bins=20, range=(0, 1), weights=np.full(len(p), 0.35 / len(p) * 20 / 20) * 20 / 7, color=GRID, edgecolor="white", lw=0.5)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.7, ls=":")
    ax.plot(mp, mo, marker=DET_MARK[d], color=DET_COL[d], mfc=DET_FILL[d], lw=1.2, ms=4, mew=0.8)
    ax.set_title(d, pad=2); ax.text(0.04, 0.88, f"ECE {ece(yy_, p):.3f}", transform=ax.transAxes, fontsize=6.3, color=INK2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
for ax in axes.ravel()[nb:]: ax.axis("off")
for ax in axes[-1]: ax.set_xlabel("Predicted P(malicious)")
for ax in axes[:, 0]: ax.set_ylabel("Observed fraction")
fig.legend(handles=[Patch(color=GRID, label="score histogram (scaled)"), Line2D([], [], color=INK2, lw=0.7, ls=":", label="perfect calibration"),
                    Line2D([], [], color=INK2, marker="o", lw=1.2, label=f"{CFG['ece_bins']} equal-count bins")],
           loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout()
savefig(fig, "fig04_calibration", "Reliability on clean test data with equal-count bins (each point averages the same number of emails); ECE uses the same bins.")

# %% [markdown]
# ### Fig. 5: Vulnerability matrix
# Columns in italics are controls. Stars in (b): Holm-corrected paired McNemar test against the same emails in clean form.

# %%
ROWS5 = ALL_DETS
piv = M0.pivot(index="detector", columns="attack", values="dr").loc[ROWS5, ATTACKS]
delta = piv[ATTACKS[1:]].sub(piv["clean"], axis=0)
sig_txt = [[stars(SIG[(SIG.detector == d) & (SIG.attack == a)].p_holm.iloc[0]) for a in ATTACKS[1:]] for d in ROWS5]
fig, axes = plt.subplots(1, 2, figsize=(COL2, 0.26 * len(ROWS5) + 1.3), gridspec_kw={"width_ratios": [8, 7.4]})
ax = axes[0]
im = ax.imshow(piv.values, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6, color="white" if v > 0.55 else INK)
ax.set_xticks(range(len(ATTACKS)), [att_tick(a) for a in ATTACKS], rotation=40, ha="right"); mark_controls(ax, ATTACKS)
ax.set_yticks(range(len(ROWS5)), ROWS5); ax.grid(False); ax.tick_params(length=0)
for y_ in group_breaks(ROWS5): ax.axhline(y_, color="white", lw=2.4)
for x_ in ATT_BREAKS: ax.axvline(x_, color="white", lw=1.6)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label(f"Detection rate (FPR = {A0:g})"); cb.outline.set_visible(False)
panel_label(ax, "(a)", x=-0.3)
ax = axes[1]
lim = max(0.05, float(np.nanmax(np.abs(delta.values))))
im2 = ax.imshow(delta.values, cmap=CMAP_DIV.reversed(), norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
for i in range(delta.shape[0]):
    for j in range(delta.shape[1]):
        v = delta.values[i, j]
        ax.text(j, i, f"{v:+.2f}{sig_txt[i][j]}", ha="center", va="center", fontsize=5.3, color="white" if abs(v) > 0.6 * lim else INK)
ax.set_xticks(range(len(ATTACKS) - 1), [att_tick(a) for a in ATTACKS[1:]], rotation=40, ha="right"); mark_controls(ax, ATTACKS[1:])
ax.set_yticks(range(len(ROWS5)), [""] * len(ROWS5)); ax.grid(False); ax.tick_params(length=0)
for y_ in group_breaks(ROWS5): ax.axhline(y_, color="white", lw=2.4)
for x_ in [b - 1 for b in ATT_BREAKS[1:]]: ax.axvline(x_, color="white", lw=1.6)
cb2 = fig.colorbar(im2, ax=ax, fraction=0.045, pad=0.02); cb2.set_label("Change vs. clean (same emails)"); cb2.outline.set_visible(False)
panel_label(ax, "(b)", x=-0.04)
fig.tight_layout()
savefig(fig, "fig05_vulnerability_matrix", "Detection rate per detector and input manipulation at a validation-fixed FPR (a) and paired change versus the same clean emails (b). Italic columns are controls; stars mark Holm-corrected McNemar significance.")

# %% [markdown]
# ### Fig. 6: The read-out decides injection susceptibility (central result)
# (a) Detection on the same malicious emails: clean, with a neutral suffix (control), and with an injected instruction. (b) Instruction effect
# $\Delta_{\text{instr}} = \text{DR}(\text{neutral}) - \text{DR}(\text{inject})$ with 95% CI, for generic and label-aware wording; stars: Holm-corrected McNemar.

# %%
RO_ROWS = [f"{r}-{t}" for r in READOUTS for t in BB_TAGS if f"{r}-{t}" in AVAIL] + [C_STAR, "Hybrid-OR", "Hybrid-Stack"]
E = EFF.set_index(["effect", "detector"])
fig, axes = plt.subplots(1, 2, figsize=(COL2, 0.3 * len(RO_ROWS) + 1.25), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
yy = np.arange(len(RO_ROWS))[::-1]
ax = axes[0]
for yi, d in zip(yy, RO_ROWS):
    dc = M0[(M0.detector == d) & (M0.attack == "clean")].dr.iloc[0]
    dn = M0[(M0.detector == d) & (M0.attack == "neutral")].dr.iloc[0]
    di = M0[(M0.detector == d) & (M0.attack == "inject")].dr.iloc[0]
    da = M0[(M0.detector == d) & (M0.attack == "inject_aware")].dr.iloc[0]
    ax.plot([min(dc, dn, di, da), max(dc, dn, di, da)], [yi, yi], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
    ax.plot(dc, yi, "o", mfc="white", mec=INK2, ms=5, mew=0.8, zorder=3)
    ax.plot(dn, yi, "s", color=NEUTRAL, ms=4.6, zorder=3)
    ax.plot(di, yi, marker=DET_MARK[d], color=DET_COL[d], mfc=DET_FILL[d], ms=MSZ[d] + 0.4, mew=0.9, ls="none", zorder=4)
    ax.plot(da, yi, marker="|", color=DET_COL[d], ms=8, mew=1.4, ls="none", zorder=4)
ax.set_yticks(yy, RO_ROWS); ax.set_xlim(-0.03, 1.03); ax.set_xlabel(f"Detection rate (FPR = {A0:g})")
for b in group_breaks(RO_ROWS): ax.axhline(len(RO_ROWS) - 1 - b, color=INK2, lw=0.5, alpha=0.6)
ax.axhline(len(RO_ROWS) - 1 - (len(RO_ROWS) - 3.5), color=INK2, lw=0.8)
ax.grid(axis="y", visible=False)
ax.legend(handles=[Line2D([], [], marker="o", mfc="white", mec=INK2, ls="none", label="clean"),
                   Line2D([], [], marker="s", color=NEUTRAL, ls="none", label="neutral suffix (control)"),
                   Line2D([], [], marker="D", color=INK2, ls="none", label="injection"),
                   Line2D([], [], marker="|", color=INK2, ms=8, mew=1.4, ls="none", label="injection (label-aware)")],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
panel_label(ax, "(a)", x=-0.36)
ax = axes[1]
for yi, d in zip(yy, RO_ROWS):
    for eff, off, fill in (("instruction", 0.13, True), ("instruction (aware)", -0.13, False)):
        r = E.loc[(eff, d)]
        ax.plot([r.delta_lo, r.delta_hi], [yi + off] * 2, color=DET_COL[d], lw=1.4, alpha=0.7)
        ax.plot(r.delta, yi + off, marker=DET_MARK[d], color=DET_COL[d], mfc=DET_COL[d] if fill else "white", ms=MSZ[d], mew=0.9, ls="none")
        st = stars(r.p_holm)
        if st: ax.text(max(r.delta_hi, r.delta) + 0.02, yi + off, st, va="center", fontsize=6.4)
ax.axvline(0, color=INK2, lw=0.8)
for b in group_breaks(RO_ROWS): ax.axhline(len(RO_ROWS) - 1 - b, color=INK2, lw=0.5, alpha=0.6)
ax.set_xlabel("Instruction effect: DR(neutral) - DR(injection)"); ax.grid(axis="y", visible=False)
xmax = max(0.1, float(EFF[EFF.effect.str.startswith("instruction")].delta_hi.max()) + 0.12)
ax.set_xlim(min(-0.05, float(EFF[EFF.effect.str.startswith("instruction")].delta_lo.min()) - 0.02), xmax)
ax.legend(handles=[Line2D([], [], marker="o", color=INK2, ls="none", label="generic wording"),
                   Line2D([], [], marker="o", mfc="white", color=INK2, ls="none", label="label-aware wording")],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
panel_label(ax, "(b)", x=-0.04)
fig.tight_layout()
savefig(fig, "fig06_readout", "Same backbone, different read-outs. (a) Detection of the same malicious emails clean, with a neutral suffix (control) and with an injected instruction. (b) Effect of the instruction itself, net of the appended text (95% CI; Holm-corrected McNemar). Filled markers: 1.5B; hollow: 3B.")

# %% [markdown]
# ### Fig. 7: Padding robustness, word salad versus natural text (artifact check)

# %%
PAD_ROWS = ALL_DETS
fig, axes = plt.subplots(1, 2, figsize=(COL2, 0.24 * len(PAD_ROWS) + 1.25), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
yy = np.arange(len(PAD_ROWS))[::-1]
ax = axes[0]
for yi, d in zip(yy, PAD_ROWS):
    v = {a: M0[(M0.detector == d) & (M0.attack == a)].dr.iloc[0] for a in ["clean", "ham_pad", "goodword"]}
    ax.plot([min(v.values()), max(v.values())], [yi, yi], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
    ax.plot(v["clean"], yi, "o", mfc="white", mec=INK2, ms=5, mew=0.8, zorder=3)
    ax.plot(v["ham_pad"], yi, "s", color=NEUTRAL, ms=4.6, zorder=3)
    ax.plot(v["goodword"], yi, marker=DET_MARK[d], color=DET_COL[d], mfc=DET_FILL[d], ms=MSZ[d] + 0.4, mew=0.9, ls="none", zorder=4)
ax.set_yticks(yy, PAD_ROWS); ax.set_xlim(-0.03, 1.03); ax.set_xlabel(f"Detection rate (FPR = {A0:g})")
for b in group_breaks(PAD_ROWS): ax.axhline(len(PAD_ROWS) - 1 - b, color=INK2, lw=0.5, alpha=0.6)
ax.grid(axis="y", visible=False)
ax.legend(handles=[Line2D([], [], marker="o", mfc="white", mec=INK2, ls="none", label="clean"),
                   Line2D([], [], marker="s", color=NEUTRAL, ls="none", label="natural padding"),
                   Line2D([], [], marker="D", color=INK2, ls="none", label="good-word (word salad)")],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
panel_label(ax, "(a)", x=-0.32)
ax = axes[1]
for yi, d in zip(yy, PAD_ROWS):
    r = E.loc[("word salad vs natural pad", d)]
    ax.plot([r.delta_lo, r.delta_hi], [yi, yi], color=DET_COL[d], lw=1.4, alpha=0.7)
    ax.plot(r.delta, yi, marker=DET_MARK[d], color=DET_COL[d], mfc=DET_FILL[d], ms=MSZ[d], mew=0.9, ls="none")
    st = stars(r.p_holm)
    if st: ax.text(r.delta_hi + 0.02 if r.delta >= 0 else r.delta_lo - 0.02, yi, st, va="center", ha="left" if r.delta >= 0 else "right", fontsize=6.4)
ax.axvline(0, color=INK2, lw=0.8)
for b in group_breaks(PAD_ROWS): ax.axhline(len(PAD_ROWS) - 1 - b, color=INK2, lw=0.5, alpha=0.6)
ax.set_xlabel("DR(natural pad) - DR(good-word)"); ax.grid(axis="y", visible=False)
_e = EFF[EFF.effect == "word salad vs natural pad"]
ax.set_xlim(min(-0.08, float(_e.delta_lo.min()) - 0.06), max(0.08, float(_e.delta_hi.max()) + 0.1))
panel_label(ax, "(b)", x=-0.04)
fig.tight_layout()
savefig(fig, "fig07_padding_control", "Padding with word salad (good-word attack) versus natural legitimate text of similar length. A detector whose detection rises under word salad but not under natural text is reacting to the padding artifact, not resisting padding.")

# %% [markdown]
# ### Fig. 8: Conditional attack success rate (log color scale)

# %%
ROWS8 = ALL_DETS
A8 = ATTACKS[1:]
asr = M0.pivot(index="detector", columns="attack", values="asr").loc[ROWS8, A8]
nsu = M0.pivot(index="detector", columns="attack", values="n_asr_succ").loc[ROWS8, A8]
nas = M0.pivot(index="detector", columns="attack", values="n_asr").loc[ROWS8, A8]
fig, ax = plt.subplots(figsize=(COL2 * 0.78, 0.26 * len(ROWS8) + 1.1))
im = ax.imshow(np.clip(asr.values.astype(float), FLOOR, 1), cmap=CMAP_RED, norm=LogNorm(FLOOR, 1), aspect="auto")
for i in range(asr.shape[0]):
    for j in range(asr.shape[1]):
        v = asr.values[i, j]
        if np.isnan(v): ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.4, color=INK2); continue
        s = "0" if nsu.values[i, j] == 0 else (f"{100*v:.1f}" if v < 0.995 else "100")
        ax.text(j, i, s, ha="center", va="center", fontsize=5.8, color="white" if v > 0.15 else INK)
ax.set_xticks(range(len(A8)), [att_tick(a) for a in A8], rotation=40, ha="right"); mark_controls(ax, A8)
ax.set_yticks(range(len(ROWS8)), ROWS8); ax.grid(False); ax.tick_params(length=0)
for y_ in group_breaks(ROWS8): ax.axhline(y_, color="white", lw=2.4)
for x_ in [b - 1 for b in ATT_BREAKS[1:]]: ax.axvline(x_, color="white", lw=1.6)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("ASR (log scale)"); cb.outline.set_visible(False); logpct_axis(cb.ax.yaxis)
fig.tight_layout()
savefig(fig, "fig08_asr_heatmap", "Conditional attack success rate (%): share of emails detected in clean form that are missed after the manipulation. Log color scale; 0 = no successes; n/a = never detected clean.")

# %% [markdown]
# ### Fig. 9: Score plane of C* and S* on a common false-positive scale
# Each axis is $-\log_{10}$ of the validation FPR a threshold at that score would give, so both detectors share units and the dashed lines (thresholds at FPR $\alpha$)
# sit at the same value. Emails scored above every legitimate validation email fall in the shaded strip (jittered for visibility).

# %%
def fpr_scale(det):
    leg = np.sort(VAL_LEG[f"s_{det}"].values); n = len(leg)
    def u(s):
        k = n - np.searchsorted(leg, np.asarray(s), side="left")
        return -np.log10((k + 0.5) / (n + 1))
    return u, -np.log10(0.5 / (n + 1))
uC, mxC = fpr_scale(C_STAR); uS, mxS = fpr_scale(S_STAR)
TOPC, TOPS = mxC - 0.08, mxS - 0.08
def jit(v, top, key):
    r = _rng_for(key).uniform(-0.07, 0.07, len(v)); return np.where(v >= top, v + r - 0.04, v)
panels = [("clean", CLEAN_POOL), ("goodword", attack_frame("goodword")), ("ham_pad", attack_frame("ham_pad")), ("inject", attack_frame("inject"))]
leg_te = tc[tc.label == 0]
lx, ly = uC(leg_te[f"s_{C_STAR}"].values), uS(leg_te[f"s_{S_STAR}"].values)
fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.35), sharex=True, sharey=True)
gx, gy = np.meshgrid(np.linspace(0, mxC + 0.1, 100), np.linspace(0, mxS + 0.1, 100))
try:
    kde = gaussian_kde(np.vstack([lx, ly]) + RNG.normal(0, 0.02, (2, len(lx)))); zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
except Exception:
    zz = None
tx, ty = -np.log10(A0), -np.log10(A0)
for ax, (att, df) in zip(axes, panels):
    if zz is not None:
        lv = np.quantile(zz[zz > zz.max() * 1e-3], [0.5, 0.75, 0.9, 0.97])
        if np.all(np.diff(lv) > 0):
            ax.contourf(gx, gy, zz, levels=list(lv) + [zz.max() * 1.01], colors=["#ecebe8", "#dcdad5", "#c9c7c1", "#b3b0a9"], zorder=0)
    ax.axvspan(TOPC, mxC + 0.15, color=PAL[1], alpha=0.06, lw=0); ax.axhspan(TOPS, mxS + 0.15, color=PAL[1], alpha=0.06, lw=0)
    x = jit(uC(df[f"s_{C_STAR}"].values), TOPC, att + "x"); yv = jit(uS(df[f"s_{S_STAR}"].values), TOPS, att + "y")
    both = (x <= tx) & (yv <= ty)
    ax.scatter(x[~both], yv[~both], s=4, color=PAL[1], alpha=0.55, lw=0, rasterized=True, zorder=2)
    ax.scatter(x[both], yv[both], s=10, facecolor="none", edgecolor=PAL[7], lw=0.7, rasterized=True, zorder=3)
    ax.axvline(tx, color=INK2, lw=0.7, ls="--"); ax.axhline(ty, color=INK2, lw=0.7, ls="--")
    qc, qs = np.mean((x <= tx) & (yv > ty)), np.mean((x > tx) & (yv <= ty))
    kw = dict(transform=ax.transAxes, fontsize=5.8, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.text(0.03, 0.04, f"both miss {both.mean():.0%}", color=PAL[7], **kw)
    ax.text(0.03, 0.96, f"only {C_STAR}\nmisses {qc:.0%}", va="top", **kw)
    ax.text(0.97, 0.04, f"only {S_STAR}\nmisses {qs:.0%}", ha="right", **kw)
    ax.set_title(ATT_LABEL[att]); ax.set_xlabel(f"{C_STAR}: -log10 FPR"); ax.set_xlim(0, mxC + 0.12); ax.set_ylim(0, mxS + 0.12)
axes[0].set_ylabel(f"{S_STAR}: -log10 FPR")
fig.legend(handles=[Patch(color="#c9c7c1", label="legitimate test (density)"), Line2D([], [], color=PAL[1], marker="o", ls="none", ms=3.5, label="malicious"),
                    Line2D([], [], color=PAL[7], marker="o", mfc="none", ls="none", ms=4, label="missed by both"),
                    Patch(color=PAL[1], alpha=0.15, label="above every legitimate validation email"),
                    Line2D([], [], color=INK2, ls="--", lw=0.7, label=f"threshold (FPR {A0:g})")],
           loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), fontsize=6.2)
fig.tight_layout()
savefig(fig, "fig09_score_plane", f"Per-email scores of {C_STAR} and {S_STAR} on a common false-positive scale. Good-word padding moves emails left only (the lexical model misses them); natural padding moves them left and down, so both detectors miss many of the same emails.")

# %% [markdown]
# ### Fig. 10: Error overlap between C* and S*

# %%
comp = COMP.set_index("attack").loc[ATTACKS]
cats = [("both_catch", "Both detect", "#cfcdc6"), ("only_a_miss", f"Only {C_STAR} misses", PAL[0]),
        ("only_b_miss", f"Only {S_STAR} misses", PAL[1]), ("both_miss", "Both miss", PAL[7])]
fig, ax = plt.subplots(figsize=(COL1 * 1.55, 2.6))
yy = np.arange(len(ATTACKS))[::-1]; left = np.zeros(len(ATTACKS))
for key, lab, col in cats:
    v = comp[key].values
    ax.barh(yy, v, left=left, color=col, height=0.66, edgecolor="white", lw=1.2, label=lab)
    for yi, l, w in zip(yy, left, v):
        if w >= 0.07: ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=6, color="white" if col in (PAL[0], PAL[7]) else INK)
    left += v
for yi, (q, lo, hi, a, b_) in zip(yy, comp[["yules_q", "q_lo", "q_hi", "n_only_a_miss", "n_only_b_miss"]].values):
    qs = "n/a" if np.isnan(q) else (f"{q:+.2f}" + (" (few errors)" if np.isnan(lo) else f" [{lo:+.2f}, {hi:+.2f}]"))
    ax.text(1.03, yi, f"{qs}   {int(a)}/{int(b_)}", va="center", fontsize=6, color=INK2)
ax.text(1.03, len(ATTACKS) - 0.35, "Yule's Q [95% CI]   discordant", fontsize=6, color=INK2, fontweight="bold")
ax.set_yticks(yy, [ATT_LABEL[a] for a in ATTACKS]); mark_controls(ax, ATTACKS[::-1][::-1], axis="y")
ax.set_xlim(0, 1); ax.set_xlabel("Share of malicious emails")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0)); ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.25), ncol=2)
fig.tight_layout()
savefig(fig, "fig10_complementarity", f"Error overlap between {C_STAR} and {S_STAR}. Yule's Q near zero or negative means complementary errors; discordant counts are emails only one detector misses.")

# %% [markdown]
# ### Fig. 11: Hybrid defenses (miss rate, log scale) and triage

# %%
HYB_SHOW = list(dict.fromkeys([C_STAR, S_STAR, "Hybrid-OR", "Hybrid-Stack", "Stack-Aug", "Stack-noInj"]))
fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]; yy = np.arange(len(ATTACKS))[::-1]; off = np.linspace(-0.3, 0.3, len(HYB_SHOW))
for o, d in zip(off, HYB_SHOW):
    sub = M0[M0.detector == d].set_index("attack").loc[ATTACKS]
    miss, lo, hi = 1 - sub.dr.values, 1 - sub.dr_hi.values, 1 - sub.dr_lo.values
    zero = miss <= 0
    ax.plot([floor_(lo), floor_(hi)], [yy + o, yy + o], color=DET_COL[d], lw=1.0, alpha=0.6)
    ax.plot(floor_(miss)[~zero], (yy + o)[~zero], marker=DET_MARK[d], color=DET_COL[d], mfc=DET_FILL[d], ms=MSZ[d] - 0.5, mew=0.7, ls="none")
    ax.plot(floor_(miss)[zero], (yy + o)[zero], marker=DET_MARK[d], color=DET_COL[d], mfc="white", ms=MSZ[d] - 0.5, mew=0.9, ls="none")
for yi, att in zip(yy, ATTACKS):
    r = MCN[(MCN.attack == att) & (MCN.A == "Hybrid-OR") & (MCN.B == "Hybrid-Stack")]
    if len(r) and stars(r.p_holm.iloc[0]): ax.text(1.15, yi, stars(r.p_holm.iloc[0]), va="center", fontsize=7)
ax.set_xscale("log"); ax.set_xlim(FLOOR * 0.8, 1.4); logpct_axis(ax.xaxis)
ax.set_yticks(yy, [ATT_LABEL[a] for a in ATTACKS]); ax.set_xlabel(f"Miss rate (FPR = {A0:g}), log scale, 95% CI")
for y_ in yy[:-1] - 0.5: ax.axhline(y_, color=GRID, lw=0.6)
ax.grid(axis="y", visible=False)
ax.legend(handles=[Line2D([], [], color=DET_COL[d], marker=DET_MARK[d], mfc=DET_FILL[d], ms=5, lw=1.0, label=d) for d in HYB_SHOW]
          + [Line2D([], [], color=INK2, marker="o", mfc="white", ls="none", ms=5, label="0 misses (at 0.1%)"),
             Line2D([], [], color=INK2, marker="$*$", ls="none", ms=6, label="OR vs Stack, p<0.05")],
          loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=4, columnspacing=0.9); panel_label(ax, "(a)", x=-0.33)
ax = axes[1]; tri = TRI.set_index("group"); yy2 = np.arange(len(tri))[::-1]; left = np.zeros(len(tri))
for key, lab, col in [("block", "Auto-block", PAL[6]), ("review", "Human review", PAL[3]), ("allow", "Auto-allow", "#cfcdc6")]:
    v = tri[key].values; ax.barh(yy2, v, left=left, color=col, height=0.66, edgecolor="white", lw=1.2, label=lab)
    for yi, l, w_ in zip(yy2, left, v):
        if w_ >= 0.08: ax.text(l + w_ / 2, yi, f"{w_:.0%}", ha="center", va="center", fontsize=6, color="white" if col == PAL[6] else INK)
    left += v
ax.set_yticks(yy2, tri.index); ax.set_xlim(0, 1); ax.set_xlabel("Share of emails"); ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0)); ax.axhline(0.5, color=INK2, lw=0.6)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.3), ncol=3); panel_label(ax, "(b)", x=-0.48)
fig.tight_layout()
savefig(fig, "fig11_hybrids", "Hybrid defenses versus the best single detectors (a; miss rate on a log scale; stars: Hybrid-OR vs Hybrid-Stack, Holm-corrected McNemar) and disagreement-gated triage (b; last row = review burden on legitimate mail).")

# %% [markdown]
# ### Fig. 12: Adaptive attacker (log scale)

# %%
ACOL = {d: DET_COL.get(d, "#8a8883") for d in DEFENSES} | {"Triage": "#8a8883"}
AMARK = {d: DET_MARK.get(d, "o") for d in DEFENSES} | {"Triage": "o"}
ALS = {d: "-" for d in DEFENSES} | {"Triage": (0, (4, 2)), "Stack-Aug": (0, (1.5, 1.2))}
fig, ax = plt.subplots(figsize=(COL1, 2.9))
pick = np.unique(np.clip([0, 4, N_CAND // 2 - 1, N_CAND - 1], 0, N_CAND - 1))
ends = []
for j, (d, c) in enumerate(ADAPT_CURVES.items()):
    ax.plot(c["k"], floor_(c["mean"]), color=ACOL[d], ls=ALS[d], lw=DET_LW.get(d, 1.6), zorder=3)
    jit_ = (j - (len(ADAPT_CURVES) - 1) / 2) * 0.3
    m_, lo_, hi_ = floor_(c["mean"][pick]), floor_(c["lo"][pick]), floor_(c["hi"][pick])
    ax.errorbar(c["k"][pick] + jit_, m_, yerr=[m_ - lo_, hi_ - m_], fmt=AMARK[d], color=ACOL[d], ms=6 if AMARK[d] == "*" else 4,
                mfc="white" if d == "Triage" else ACOL[d], mec=ACOL[d] if d == "Triage" else "white", mew=0.8, elinewidth=0.8, capsize=1.5, ls="none", zorder=4)
    ends.append([float(floor_(c["mean"][-1])), d, c["mean"][-1]])
ends.sort(); last = 0
for v, d, raw in ends:                      # end labels spaced at least 1.28x apart on the log axis, so none overlap
    yv = max(v, last * 1.28) if last else v; last = yv
    ax.text(N_CAND + 1.4, yv, f"{100*raw:.1f}%", fontsize=5.8, va="center", color=ACOL[d])
ax.set_yscale("log"); ax.set_ylim(FLOOR * 0.8, max(1.3, last * 1.3)); logpct_axis(ax.yaxis)
ax.set_xlabel("Query budget k (candidate variants tried)"); ax.set_ylabel("ASR@k (log scale)"); ax.set_xlim(0.5, N_CAND * 1.14)
ax.legend(handles=[Line2D([], [], color=ACOL[d], ls=ALS[d], lw=DET_LW.get(d, 1.6), marker=AMARK[d], ms=5 if AMARK[d] == "*" else 4,
                          mfc="white" if d == "Triage" else ACOL[d], label=f"{d} (n={c['n']})") for d, c in ADAPT_CURVES.items()],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.1)
fig.tight_layout()
savefig(fig, "fig12_adaptive", "Adaptive black-box attacker: probability of evading each defense within k queries (exact over random query orders) on a log scale, with 95% bootstrap CIs at k = 1, 5, N/2 and N.")

# %% [markdown]
# ### Fig. 13: Leave-one-source-out generalization (including LoRA)

# %%
LOSO_FIG = LOSO_TRAINED   # the prompted read-out has no training to leave a source out of; it is reported in the text, not here
lp = LOSO.pivot(index="source", columns="detector", values="score").reindex(columns=LOSO_FIG)
single = LOSO.groupby("source").single_class.first().reindex(lp.index)
fig, ax = plt.subplots(figsize=(COL1 * 1.5, 0.36 * len(lp) + 1.05))
vals = lp.values.astype(float)
im = ax.imshow(np.ma.masked_invalid(vals), cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
ax.imshow(np.where(np.isnan(vals), 1.0, np.nan), cmap=LinearSegmentedColormap.from_list("g", ["#f0efec", "#f0efec"]), aspect="auto")
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        v = vals[i, j]
        ax.text(j, i, "n/r" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6.2, color=INK2 if np.isnan(v) else ("white" if v > 0.55 else INK))
ax.set_xticks(range(len(LOSO_FIG)), LOSO_FIG, rotation=30, ha="right")
ax.set_yticks(range(len(lp)), [f"{s}{' (TPR)' if single[s] else ''}" for s in lp.index]); ax.grid(False); ax.tick_params(length=0)
ax.axvline(len(CLS_DETS) - 0.5, color="white", lw=2.6)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label("Balanced accuracy, held-out source"); cb.outline.set_visible(False)
fig.tight_layout()
savefig(fig, "fig13_loso", "Leave-one-source-out generalization of the trained detectors at a validation-fixed FPR (thresholds from the other sources): balanced accuracy, or detection rate (TPR) for sources that contain only malicious mail. n/r: not run.")

# %% [markdown]
# ### Fig. 14: Attack matrix on a held-out source (every detector trained without it)

# %%
if len(XSRC_MAT):
    xr = list(dict.fromkeys([d for d in LOSO_DETS if d in set(XSRC_MAT.detector)] + ["Hybrid-OR"]))
    xp = XSRC_MAT.pivot(index="detector", columns="attack", values="dr").loc[xr, ATTACKS]
    fp = XSRC_MAT.groupby("detector").fpr_test.first()
    fig, ax = plt.subplots(figsize=(COL2 * 0.72, 0.28 * len(xr) + 1.2))
    im = ax.imshow(xp.values, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(xp.shape[0]):
        for j in range(xp.shape[1]):
            v = xp.values[i, j]; ax.text(j, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(ATTACKS)), [att_tick(a) for a in ATTACKS], rotation=40, ha="right"); mark_controls(ax, ATTACKS)
    ax.set_yticks(range(len(xr)), [f"{d}  (FPR {100*fp[d]:.1f}%)" for d in xr]); ax.grid(False); ax.tick_params(length=0)
    for y_ in group_breaks(xr): ax.axhline(y_, color="white", lw=2.4)
    for x_ in ATT_BREAKS: ax.axvline(x_, color="white", lw=1.6)
    n_x = int(XSRC_MAT[XSRC_MAT.attack == "clean"].n.iloc[0])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.set_label(f"Detection rate (threshold FPR = {A0:g})"); cb.outline.set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig14_heldout_source_attacks", f"Held-out source ({XSRC}, {n_x} malicious emails): every detector was trained without {XSRC}; thresholds from the other sources. Row labels give the realized FPR on {XSRC} legitimate test mail.")
else:
    print("Held-out-source matrix unavailable (LOSO outputs missing).")

# %% [markdown]
# ### Fig. 15: Seed stability of LoRA

# %%
if N_SEEDS > 1:
    S1 = SEEDS[SEEDS.attack.isin(ATTACKS)]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]; xs = np.arange(len(ATTACKS))
    mk = ["o", "s", "^", "D", "v"]
    for k, (seed, g) in enumerate(S1.groupby("seed")):
        g = g.set_index("attack").loc[ATTACKS]
        ax.plot(xs + (k - 1) * 0.12, floor_(1 - g.value.values), marker=mk[k % 5], color=RO_COL["LoRA"], mfc="white" if k else RO_COL["LoRA"],
                ls="none", ms=4.5, mew=0.9, label=f"seed {seed}")
    mean_ = S1.groupby("attack").value.mean().loc[ATTACKS]
    ax.plot(xs, floor_(1 - mean_.values), "_", color=INK, ms=12, mew=1.4, label="mean")
    ax.set_yscale("log"); ax.set_ylim(FLOOR * 0.8, 1.2); logpct_axis(ax.yaxis)
    ax.set_xticks(xs, [att_tick(a) for a in ATTACKS], rotation=35, ha="right"); mark_controls(ax, ATTACKS)
    ax.set_ylabel(f"{LORA_P} miss rate (log)"); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6); panel_label(ax, "(a)", x=-0.12)
    ax = axes[1]
    S2 = SEEDS[SEEDS.attack.str.startswith("ASR ")]
    summ = S2.groupby("attack").value.agg(["mean", "min", "max"]).reindex([f"ASR {a}" for a in ATTACKS[1:]])
    yy = np.arange(len(summ))[::-1]
    ax.hlines(yy, floor_(summ["min"]), floor_(summ["max"]), color=RO_COL["LoRA"], lw=2, alpha=0.5)
    ax.plot(floor_(summ["mean"]), yy, "o", color=RO_COL["LoRA"], ms=4.5)
    ax.set_xscale("log"); ax.set_xlim(FLOOR * 0.8, 1.2); logpct_axis(ax.xaxis)
    ax.set_yticks(yy, [att_tick(a[4:]) for a in summ.index]); ax.set_xlabel("ASR across seeds (min, mean, max)"); ax.grid(axis="y", visible=False)
    panel_label(ax, "(b)", x=-0.5)
    fig.tight_layout()
    savefig(fig, "fig15_seed_stability", f"{LORA_P} across {N_SEEDS} seeds (each seed redraws the training sample, initialization and order): miss rate per manipulation (a) and ASR range (b).")
else:
    print("Only one LoRA seed available; seed-stability figure skipped (deferred seeds finish in a resumed run).")

# %% [markdown]
# ### Fig. 16: Cost versus robustness

# %%
lat_ms = {d: LAT[d] * 1e3 for d in BASE_DETS}
lat_ms["Hybrid-OR"] = lat_ms[C_STAR] + lat_ms[S_STAR]
_zs_extra = lat_ms[ZS_P] if S_STAR != PROBE_P else 0   # Probe shares the zero-shot forward pass
lat_ms["Hybrid-Stack"] = lat_ms["Stack-Aug"] = lat_ms[C_STAR] + lat_ms[S_STAR] + _zs_extra + LAT["InjDet"] * 1e3
ADV_ATT = [a for a in ATTACKS[1:] if a not in CONTROLS]
att_only = M0[M0.attack.isin(ADV_ATT)]
worst = att_only.groupby("detector").dr.min(); mean_ = att_only.groupby("detector").dr.mean()
ROWS16 = ALL_DETS
fig, axes = plt.subplots(1, 2, figsize=(COL2, 0.24 * len(ROWS16) + 1.1), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
yy = np.arange(len(ROWS16))[::-1]
ax = axes[0]
for yi, d in zip(yy, ROWS16):
    c = FAM_COL[FAMILY[d]]
    ax.plot([worst[d], mean_[d]], [yi, yi], color=c, lw=1.8, alpha=0.55, solid_capstyle="round")
    ax.plot(worst[d], yi, marker=DET_MARK[d], color=c, mfc="white" if DET_FILL[d] == "white" else c, ms=MSZ[d] + 0.4, mew=0.8, ls="none")
    ax.plot(mean_[d], yi, marker="|", color=c, ms=8, mew=1.6, ls="none")
    ax.plot(piv.loc[d, "clean"], yi, marker="o", mfc="none", mec=INK2, ms=4.5, mew=0.7, ls="none")
for b in group_breaks(ROWS16): ax.axhline(len(ROWS16) - 1 - b, color=INK2, lw=0.5, alpha=0.5)
ax.set_yticks(yy, ROWS16); ax.set_xlim(-0.02, 1.02); ax.set_xlabel(f"Detection rate (FPR = {A0:g})"); ax.grid(axis="y", visible=False)
ax.legend(handles=[Line2D([], [], color=INK2, marker="o", ls="none", label="worst attack"),
                   Line2D([], [], color=INK2, marker="|", ms=8, mew=1.6, ls="none", label="mean over attacks"),
                   Line2D([], [], color=INK2, marker="o", mfc="none", ls="none", label="clean")],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3); panel_label(ax, "(a)", x=-0.25)
ax = axes[1]
ax.barh(yy, [lat_ms[d] for d in ROWS16], color=[FAM_COL[FAMILY[d]] for d in ROWS16], height=0.6, edgecolor="white", lw=1)
for yi, d in zip(yy, ROWS16): ax.text(lat_ms[d] * 1.15, yi, f"{lat_ms[d]:.2f}" if lat_ms[d] < 1 else (f"{lat_ms[d]:.1f}" if lat_ms[d] < 10 else f"{lat_ms[d]:.0f}"), va="center", fontsize=6.2, color=INK2)
ax.set_xscale("log"); ax.set_xlim(min(lat_ms.values()) / 2, max(lat_ms.values()) * 8)
ax.set_xlabel("Latency (ms / email, log scale)"); ax.grid(axis="y", visible=False); panel_label(ax, "(b)", x=-0.05)
ax.legend(handles=[Patch(color=c, label=f) for f, c in FAM_COL.items()], loc="lower center", bbox_to_anchor=(0.58, 1.0), ncol=3, columnspacing=1.0)
fig.tight_layout()
savefig(fig, "fig16_cost_robustness", "Robustness (a: clean, mean and worst-case detection over adversarial manipulations; controls excluded) versus inference cost (b). Classical latency on CPU, SLM latency on one GPU; hybrids sum their components.")

# %% [markdown]
# ### Fig. 17: Where the probe's injection susceptibility comes from (layer sweep, both backbones)
# For every probed layer: validation AUC, detection under each manipulation (top), and the instruction effect net of the neutral suffix with a 95% paired
# bootstrap CI (bottom). If late layers are hurt and middle layers are not, the probe inherits the generative read-out's susceptibility only where the
# hidden state already encodes the model's planned answer.

# %%
PL_ALL = []
_PL_SETS = ["paraphrase", "inject", "inject_aware", "neutral"]
for t in BB_TAGS:
    if f"Probe-{t}" not in AVAIL: continue
    meta = ZS_META[t]; nL = meta["n_layers"]
    z = {s: zs_npz(t, s) for s in ["probe_train", "val", "test_clean"] + _PL_SETS}
    A_n, A_i = paired("neutral", "inject"); _, A_a = paired("neutral", "inject_aware")
    pos = {s: pd.Series(np.arange(len(SC[s])), index=SC[s].base_id.values) for s in _PL_SETS}
    for L in meta["layers"]:
        auc, c, m = S.fit_probe(z["probe_train"][f"H{L}"].astype(np.float32), SETS["probe_train"].label.values,
                                z["val"][f"H{L}"].astype(np.float32), SETS["val"].label.values)
        th = thr_at_fpr(m.predict_proba(z["val"][f"H{L}"].astype(np.float32)[(SETS["val"].label == 0).values])[:, 1], A0)
        fl_ = {s: m.predict_proba(z[s][f"H{L}"].astype(np.float32))[:, 1] > th for s in _PL_SETS}
        r = dict(backbone=t, layer=int(L), depth=L / nL, val_auc=auc,
                 clean=float((m.predict_proba(z["test_clean"][f"H{L}"].astype(np.float32))[:, 1] > th)[SC["test_clean"].label.values == 1].mean()))
        for s in _PL_SETS:
            v = SC[s]["valid"].values.astype(bool); r[s] = float(fl_[s][v].mean()) if v.any() else np.nan
        for key, B_ in (("eff", "inject"), ("eff_aware", "inject_aware")):
            ids = A_n.index
            e, lo, hi = boot_diff(fl_["neutral"][pos["neutral"].loc[ids].values].astype(float), fl_[B_][pos[B_].loc[ids].values].astype(float))
            r[key], r[key + "_lo"], r[key + "_hi"] = e, lo, hi
        PL_ALL.append(r)
PL = pd.DataFrame(PL_ALL); PL.to_csv(DIRS["results"] / "probe_layer_sweep.csv", index=False)
tags_ = list(PL.backbone.unique())
fig, axes = plt.subplots(2, len(tags_), figsize=(min(COL2, COL1 * (1 if len(tags_) == 1 else 1.02 * len(tags_))), 4.1), sharex=True, squeeze=False,
                         gridspec_kw={"height_ratios": [1.25, 1]})
for j, t in enumerate(tags_):
    g = PL[PL.backbone == t]; ax = axes[0][j]
    for key, lab, col, mk in [("val_auc", "Validation ROC-AUC", PAL[0], "o"), ("clean", "DR, clean", INK2, "s"), ("neutral", "DR, neutral suffix (ctrl)", NEUTRAL, "v"),
                              ("inject", "DR, injection", PAL[3], "D"), ("inject_aware", "DR, injection (label-aware)", PAL[7], "X"),
                              ("paraphrase", "DR, paraphrase", PAL[2], "^")]:
        ax.plot(g.depth, g[key], marker=mk, color=col, lw=1.3, ms=4, mec="white", mew=0.5, label=lab)
    bl = BEST_LAYER[t] / ZS_META[t]["n_layers"]
    for a_ in axes[:, j]: a_.axvline(bl, color=INK2, lw=0.7, ls=":")
    ax.text(bl, 0.02, " selected", fontsize=6.2, color=INK2)
    ax.set_title(f"Probe-{t}, {FAMILY_BB[t]} ({ZS_META[t]['n_layers']} layers)"); ax.set_ylim(0, 1.03); ax.set_xlim(0, 1.02)
    ax = axes[1][j]
    for key, lab, col, mk in [("eff", "effect: generic instruction", "#9a6700", "D"), ("eff_aware", "effect: label-aware instruction", "#8f1d1d", "X")]:
        ax.fill_between(g.depth, g[key + "_lo"], g[key + "_hi"], color=col, alpha=0.18, lw=0)
        ax.plot(g.depth, g[key], marker=mk, color=col, lw=1.3, ms=4, mec="white", mew=0.5, ls=(0, (4, 1.5)), label=lab)
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xlabel("Relative depth of probed layer")
axes[0][0].set_ylabel(f"Score (DR at FPR {A0:g})"); axes[1][0].set_ylabel("Instruction effect (DR points)")
for ax in axes[1]: ax.set_ylim(min(-0.1, PL[["eff_lo", "eff_aware_lo"]].min().min() - 0.02), max(0.1, PL[["eff_hi", "eff_aware_hi"]].max().max() + 0.03))
h_, l_ = axes[0][0].get_legend_handles_labels(); h2, l2 = axes[1][0].get_legend_handles_labels()
fig.legend(h_ + h2, l_ + l2, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4, fontsize=6.1)
panel_label(axes[0][0], "(a)"); panel_label(axes[1][0], "(b)")
fig.tight_layout()
savefig(fig, "fig17_probe_layers", "Linear probe by layer depth: validation AUC and detection under each manipulation (a) and the instruction effect net of the neutral suffix with 95% CI (b). Layers are selected on validation AUC only.")

# %% [markdown]
# ### Fig. 18: Injection families, with the neutral control

# %%
FAM_DETS = [d for d in [ZS_P, f"ZS-{BB_TAGS[-1]}", PROBE_P, LORA_P, C_STAR] if d in ALL_DETS]
FAM_DETS = list(dict.fromkeys(FAM_DETS))
fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.35), sharey=True)
for ax, att in zip(axes, ["neutral", "inject", "inject_aware"]):
    df = attack_frame(att); w = 0.82 / len(FAM_DETS); x = np.arange(len(INJ_FAMILIES))
    for k, d in enumerate(FAM_DETS):
        vals = [flag(df[df.family == f_], d).mean() if (df.family == f_).any() else np.nan for f_ in INJ_FAMILIES]
        xs = x - 0.41 + w * (k + 0.5)
        ax.bar(xs, vals, width=w, color=DET_COL[d], edgecolor=DET_COL[d] if DET_FILL[d] != "white" else DET_COL[d],
               hatch="////" if DET_FILL[d] == "white" else None, fill=DET_FILL[d] != "white", lw=0.6, label=d)
        for xi, v in zip(xs, vals):
            if v == 0: ax.text(xi, 0.012, "0", ha="center", va="bottom", fontsize=5.3, color=INK2)
    ax.set_xticks(x, [f.capitalize() for f in INJ_FAMILIES]); ax.set_ylim(0, 1.05); ax.set_title(ATT_LABEL[att]); ax.grid(axis="x", visible=False)
axes[0].set_ylabel(f"Detection rate (FPR = {A0:g})")
axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
savefig(fig, "fig18_injection_families", "Detection by template family for the neutral control and for generic and label-aware injections (unseen T_test wording). Hatched bars: 3B backbone.")

# %% [markdown]
# ### Fig. 19: Paraphrase attack quality

# %%
pv = PARA[PARA.valid] if PARA.valid.any() else PARA
metrics_ = [("length_ratio", "Length ratio (rewrite / original)"), ("lex_sim", "Lexical similarity (char TF-IDF cosine)")]
if pv.sem_sim.notna().any(): metrics_.append(("sem_sim", "Semantic similarity (MiniLM cosine)"))
fig, axes = plt.subplots(1, len(metrics_), figsize=(COL2 if len(metrics_) == 3 else COL2 * 0.7, 2.05), sharey=True)
for ax, (k, lab) in zip(np.atleast_1d(axes), metrics_):
    v = pv[k].dropna().values
    ax.hist(v, bins=30, color=PAL[0], edgecolor="white", lw=0.6)
    med = np.median(v); ax.axvline(med, color=INK, lw=0.9)
    ax.text(med, 0.95, f" median {med:.2f}", transform=ax.get_xaxis_transform(), fontsize=6.3, va="top")
    ax.set_xlabel(lab); ax.grid(axis="x", visible=False)
np.atleast_1d(axes)[0].set_ylabel("Valid rewrites")
fig.suptitle("Funnel: " + "  >  ".join(f"{k} {v}" for k, v in PARA_FUNNEL.items()), fontsize=6.8, y=1.03)
fig.tight_layout()
savefig(fig, "fig19_attack_quality", "Paraphrase attack quality after filtering: rewrites change wording (lexical similarity) while keeping meaning (semantic similarity).")

# %% [markdown]
# ### Fig. 20: LoRA training curves (all runs)

# %%
logs = sorted(DIRS["cache"].glob("lora_*_trainlog.csv"))
if logs:
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    for p in logs:
        g = pd.read_csv(p); nm = p.name.replace("_trainlog.csv", "")
        main = nm == LORA_MAIN[P_TAG]; loso = "_x" in nm; aug = nm.endswith("_aug"); augmix = nm.endswith("_augmix")
        other = next((t for t in BB_TAGS[1:] if nm.startswith(f"lora_{t}_")), None)
        col = (RO_COL["LoRA"] if main else DET_COL[AUG_DET] if aug else DET_COL[AUGMIX_DET] if augmix
               else (INK if other and FAMILY_BB[other] == FAMILY_BB[P_TAG] else "#2a78d6") if other else NEUTRAL)
        lab = (f"{P_TAG} main" if main else f"{P_TAG} + padding aug. (seeds)" if aug else f"{P_TAG} + mixed padding aug." if augmix
               else f"{other} ({FAMILY_BB[other]})" if other else ("LOSO runs" if loso else "extra seeds"))
        ax.plot(g.opt_step, g.loss.rolling(max(1, len(g) // 25), min_periods=1).mean(), color=col, lw=1.5 if main or other or aug or augmix else 0.8,
                ls=":" if loso else "-", alpha=0.6 if loso else 1, label=lab)
    h_, l_ = ax.get_legend_handles_labels(); keep = dict(zip(l_, h_))
    ax.set_xlabel("Optimizer step"); ax.set_ylabel("Cross-entropy loss (moving avg.)"); ax.legend(keep.values(), keep.keys(), loc="upper right", fontsize=6)
    fig.tight_layout()
    savefig(fig, "fig20_lora_training", "Training loss of every LoRA run (dotted: retrained without a held-out source).")


# %% [markdown]
# ### Fig. 21: Threshold-free check of the read-out result (v3.2)
# Change in ROC-AUC (against the same clean legitimate test emails) caused by the instruction, net of the neutral suffix. Unlike Fig. 6 this does not depend on
# the 1% operating point or on how many emails a detector catches on clean mail, so it rules out "the prompted read-out is just weak" as an explanation.

# %%
RO21 = ([f"{r}-{t}" for r in READOUTS for t in BB_TAGS if f"{r}-{t}" in AVAIL] + [d for d in (AUG_DET, AUGMIX_DET) if d in AVAIL]
        + [C_STAR, "Hybrid-Stack"])
AE = AUC_EFF.set_index(["effect", "detector"])
fig, ax = plt.subplots(figsize=(COL1 * 1.25, 0.27 * len(RO21) + 0.95))
yy = np.arange(len(RO21))[::-1]
for yi, d in zip(yy, RO21):
    for eff, off, filled in (("instruction", 0.13, True), ("instruction (aware)", -0.13, False)):
        r = AE.loc[(eff, d)]
        ax.plot([r.d_auc_lo, r.d_auc_hi], [yi + off] * 2, color=DET_COL[d], lw=1.4, alpha=0.7)
        ax.plot(r.d_auc, yi + off, marker=DET_MARK[d], color=DET_COL[d], mfc=DET_COL[d] if filled else "white", ms=MSZ[d], mew=0.9, ls="none")
ax.axvline(0, color=INK2, lw=0.8)
for b in group_breaks(RO21): ax.axhline(len(RO21) - 1 - b, color=INK2, lw=0.5, alpha=0.6)
ax.set_yticks(yy, RO21); ax.grid(axis="y", visible=False)
ax.set_xlabel("AUC(neutral suffix) - AUC(injection)")
ax.legend(handles=[Line2D([], [], marker="o", color=INK2, ls="none", label="generic wording"),
                   Line2D([], [], marker="o", mfc="white", color=INK2, ls="none", label="label-aware wording")],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout()
savefig(fig, "fig21_threshold_free", "Threshold-free instruction effect: loss of ROC-AUC caused by the instruction itself, paired over the same emails (95% CI). Filled: generic wording; hollow: label-aware wording.")

# %% [markdown]
# ### Fig. 22: Natural padding, the strongest manipulation: dose response and the effect of padding-aware training (v3.2)

# %%
fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.45), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
for d in PAD_DETS:
    g = PAD_DOSE[PAD_DOSE.detector == d].sort_values("bin")
    ax.errorbar(g.share_med, g.dr, yerr=[g.dr - g.dr_lo, g.dr_hi - g.dr], color=DET_COL[d], marker=DET_MARK[d], mfc=DET_FILL[d], ms=MSZ[d] - 0.3,
                ls=DET_LS[d], lw=DET_LW[d], capsize=1.5, elinewidth=0.8, label=d)
ax.set_xlabel("Share of the email that is appended benign text (quartile medians)"); ax.set_ylabel(f"Detection rate (FPR = {A0:g})")
ax.set_ylim(0, 1.03); ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6, ncol=4, columnspacing=0.8); panel_label(ax, "(a)", x=-0.13)
ax = axes[1]
rows22 = [d for d in [C_STAR, S_STAR, AUG_DET, AUGMIX_DET, "Hybrid-OR", "Hybrid-Stack", "Stack-Aug"] if d in ALL_DETS]
yy = np.arange(len(rows22))[::-1]
for yi, d in zip(yy, rows22):
    v = {a: M0[(M0.detector == d) & (M0.attack == a)].iloc[0] for a in ["clean", "ham_pad", "goodword"]}
    ax.plot([v["ham_pad"].dr, v["clean"].dr], [yi, yi], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
    ax.plot([v["ham_pad"].dr_lo, v["ham_pad"].dr_hi], [yi, yi], color=DET_COL[d], lw=1.2, zorder=2)
    ax.plot(v["clean"].dr, yi, "o", mfc="white", mec=INK2, ms=5, mew=0.8, zorder=3)
    ax.plot(v["ham_pad"].dr, yi, marker=DET_MARK[d], color=DET_COL[d], mfc=DET_FILL[d], ms=MSZ[d] + 0.4, mew=0.9, ls="none", zorder=4)
    ax.text(1.04, yi, f"{100*v['ham_pad'].dr:.1f}%", va="center", fontsize=6.2, color=INK2)
ax.set_yticks(yy, rows22); ax.set_xlim(0, 1.15); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.grid(axis="y", visible=False)
ax.set_xlabel("Detection rate: clean (o) and natural pad (95% CI)"); panel_label(ax, "(b)", x=-0.42)
fig.tight_layout()
savefig(fig, "fig22_natural_padding", "Natural padding: detection by the share of appended benign text (a) and recovery by defenses, including LoRA trained with defender-side padding on both classes (b).")

# %% [markdown]
# ### Fig. 23: Collateral false positives on benign mail carrying appended text (v3.2)

# %%
if len(COLL):
    rows23 = [d for d in ALL_DETS if d in set(COLL.detector)]
    fig, ax = plt.subplots(figsize=(COL1 * 1.35, 0.24 * len(rows23) + 1.0))
    yy = np.arange(len(rows23))[::-1]
    for yi, d in zip(yy, rows23):
        g = COLL[COLL.detector == d].set_index("set")
        ax.plot(floor_(g.fpr_clean.iloc[0]), yi, "o", mfc="white", mec=INK2, ms=4.8, mew=0.8, zorder=3)
        for s_, mk, off in (("legit_neutral", "s", 0.12), ("legit_inject", "D", -0.12)):
            r = g.loc[s_]
            ax.plot([floor_(r.fpr_lo), floor_(r.fpr_hi)], [yi + off] * 2, color=DET_COL[d], lw=1.0, alpha=0.6)
            ax.plot(floor_(r.fpr), yi + off, marker=mk, color=DET_COL[d], mfc=DET_COL[d] if s_ == "legit_inject" else "white", ms=4.2, mew=0.8, ls="none", zorder=4)
            if r.p_holm < 0.05: ax.text(floor_(r.fpr_hi) * 1.2, yi + off, stars(r.p_holm), va="center", fontsize=6)
    ax.axvline(A0, color=INK2, lw=0.7, ls=":")
    for b in group_breaks(rows23): ax.axhline(len(rows23) - 1 - b, color=INK2, lw=0.5, alpha=0.6)
    ax.set_xscale("log"); ax.set_xlim(FLOOR * 0.8, 1.2); logpct_axis(ax.xaxis)
    ax.set_yticks(yy, rows23); ax.grid(axis="y", visible=False); ax.set_xlabel("False-positive rate on legitimate mail (log scale)")
    ax.legend(handles=[Line2D([], [], marker="o", mfc="white", mec=INK2, ls="none", label="clean"),
                       Line2D([], [], marker="s", mfc="white", color=INK2, ls="none", label="+ neutral suffix"),
                       Line2D([], [], marker="D", color=INK2, ls="none", label="+ instruction-like suffix")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    savefig(fig, "fig23_collateral_fpr", f"False-positive rate on the same legitimate test emails, clean and with appended text (95% CI; stars: Holm-corrected McNemar vs clean; zero plotted at 0.1%). Dotted line: target FPR {A0:g}.")


# %% [markdown]
# ### Fig. 24: Out-of-distribution padding (v4.0)
# Detection rate on the same malicious emails under every padding variant. Columns left of the white line are what `LoRA-Aug` trained on (none, or
# natural text appended at the end); the rest are unseen positions, lengths and text sources. Grey cells: detector not scored on that variant.

# %%
if len(OOD_MAT):
    rows24 = [d for d in OOD_DETS if (OOD_MAT.detector == d).any()]
    piv24 = OOD_MAT.pivot(index="detector", columns="pad", values="dr").reindex(index=rows24, columns=PAD_COLS)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 0.27 * len(rows24) + 1.4), gridspec_kw={"width_ratios": [len(PAD_COLS), 2.3]})
    ax = axes[0]; v24 = piv24.values.astype(float)
    im = ax.imshow(np.ma.masked_invalid(v24), cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(v24.shape[0]):
        for j in range(v24.shape[1]):
            v = v24[i, j]; ax.text(j, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=5.8, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(PAD_COLS)), [PAD_LABEL[c] for c in PAD_COLS], rotation=35, ha="right")
    ax.set_yticks(range(len(rows24)), rows24); ax.grid(False); ax.tick_params(length=0); ax.set_facecolor("#f0efec")
    for y_ in group_breaks(rows24): ax.axhline(y_, color="white", lw=2.4)
    ax.axvline(1.5, color="white", lw=2.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.set_label(f"Detection rate (FPR = {A0:g})"); cb.outline.set_visible(False)
    panel_label(ax, "(a)", x=-0.28)
    ax = axes[1]
    if len(AUG_SEED_TAB) and AUG_SEED_TAB.seed.nunique() > 1:
        g24 = AUG_SEED_TAB.groupby("pad").dr.agg(["mean", "min", "max"]).reindex(PAD_COLS[::-1]).dropna()
        yy = np.arange(len(g24))
        ax.hlines(yy, g24["min"], g24["max"], color=DET_COL[AUG_DET], lw=2.2, alpha=0.5)
        ax.plot(g24["mean"], yy, marker=DET_MARK[AUG_DET], color=DET_COL[AUG_DET], ls="none", ms=5)
        if AUGMIX_DET in rows24:
            ax.plot(piv24.loc[AUGMIX_DET, g24.index].values, yy, marker=DET_MARK[AUGMIX_DET], color=DET_COL[AUGMIX_DET], mfc="white", ls="none", ms=5,
                    label=AUGMIX_DET)
        ax.plot(piv24.loc[LORA_P, g24.index].values, yy, marker=DET_MARK[LORA_P], color=DET_COL[LORA_P], mfc="white", ls="none", ms=5, label=LORA_P)
        ax.set_yticks(yy, [PAD_LABEL[c] for c in g24.index]); ax.set_xlim(-0.02, 1.02); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Detection rate")
        ax.legend(handles=[Line2D([], [], color=DET_COL[AUG_DET], marker=DET_MARK[AUG_DET], lw=2, alpha=0.7,
                                  label=f"{AUG_DET} ({AUG_SEED_TAB.seed.nunique()} seeds: mean, range)")] + ax.get_legend_handles_labels()[0],
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.8)
        panel_label(ax, "(b)", x=-0.75)
    else:
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, "fig24_ood_padding", "Padding the attacker controls beyond the training distribution: detection rate by padding variant (a) and padding-aware LoRA across seeds versus plain LoRA and mixed-padding LoRA (b).")

# %% [markdown]
# ### Fig. 25: Is the prompted failure a weak-prompt or small-model artifact? (v4.0)

# %%
if len(PROMPT_TAB):
    P25 = PROMPT_TAB.set_index("detector")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 0.32 * len(P25) + 1.2), sharey=True)
    yy = np.arange(len(P25))[::-1]
    ax = axes[0]
    for yi, d in zip(yy, P25.index):
        r = P25.loc[d]; c = DET_COL.get(d, INK2)
        ax.plot([r.auc_inject_aware, r.auc], [yi, yi], color=GRID, lw=2.4, solid_capstyle="round", zorder=1)
        ax.plot(r.auc, yi, "o", mfc="white", mec=INK2, ms=5, zorder=3)
        ax.plot(r.auc_inject_aware, yi, marker=DET_MARK.get(d, "D"), color=c, mfc=DET_FILL.get(d, c), ms=6, ls="none", zorder=4)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":"); ax.text(0.505, yy[-1] - 0.45, "chance", fontsize=6, color=INK2)
    ax.set_yticks(yy, P25.index); ax.set_xlim(0, 1.02); ax.set_xlabel("ROC-AUC: clean (o) and under label-aware injection"); ax.grid(axis="y", visible=False)
    panel_label(ax, "(a)", x=-0.42)
    ax = axes[1]
    for yi, d in zip(yy, P25.index):
        r = P25.loc[d]; c = DET_COL.get(d, INK2)
        for key, off, filled in (("instr", 0.12, True), ("instr_aware", -0.12, False)):
            if np.isnan(r[f"{key}_auc"]): continue
            ax.plot([r[f"{key}_auc_lo"], r[f"{key}_auc_hi"]], [yi + off] * 2, color=c, lw=1.3, alpha=0.7)
            ax.plot(r[f"{key}_auc"], yi + off, marker=DET_MARK.get(d, "D"), color=c, mfc=c if filled else "white", ms=5, ls="none")
    ax.axvline(0, color=INK2, lw=0.8); ax.set_xlabel("AUC lost to the instruction (net of neutral suffix)"); ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", color=INK2, ls="none", label="generic"), Line2D([], [], marker="o", mfc="white", color=INK2, ls="none", label="label-aware")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    panel_label(ax, "(b)", x=-0.04)
    fig.tight_layout()
    savefig(fig, "fig25_prompted_baselines", "Stronger prompted baselines (4-shot prompt, 7B model) against the zero-shot read-outs and LoRA: clean AUC and AUC under label-aware injection (a), and the instruction effect on the AUC scale with 95% CI (b).")

# %% [markdown]
# ### Fig. 26: Modern, out-of-time data (v4.0)

# %%
if len(MODERN_TAB):
    sets26 = [m for m in ["modern_human", "modern_llm"] if m in set(MODERN_TAB.set)]
    rows26 = [d for d in list(dict.fromkeys(CLS_DETS + SLM_DETS + PROMPT_AVAIL + ["Hybrid-OR", "Hybrid-Stack", "Stack-Aug"])) if d in set(MODERN_TAB.detector)]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 0.25 * len(rows26) + 1.3), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1]})
    yy = np.arange(len(rows26))[::-1]; offs = {"modern_human": 0.13, "modern_llm": -0.13}; lab26 = {"modern_human": "human (to 2025)", "modern_llm": "LLM-rephrased (2026)"}
    for ax, (key, xl) in zip(axes, [("auc", "ROC-AUC"), ("tpr", f"Detection rate (FPR target {A0:g})"), ("fpr", "False-positive rate on valid mail")]):
        for m in sets26:
            T = MODERN_TAB[MODERN_TAB.set == m].set_index("detector")
            for yi, d in zip(yy, rows26):
                if d not in T.index or np.isnan(T.loc[d, key]): continue
                c = DET_COL.get(d, INK2)
                if key == "auc" and not np.isnan(T.loc[d, "auc_lo"]):
                    ax.plot([T.loc[d, "auc_lo"], T.loc[d, "auc_hi"]], [yi + offs[m]] * 2, color=c, lw=1.1, alpha=0.6)
                ax.plot(T.loc[d, key], yi + offs[m], marker="o" if m == "modern_human" else "s", color=c, mfc=c if m == "modern_human" else "white", ms=4.2, ls="none")
        for b in group_breaks(rows26): ax.axhline(len(rows26) - 1 - b, color=INK2, lw=0.5, alpha=0.6)
        ax.set_xlabel(xl); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(yy, rows26); axes[2].axvline(A0, color=INK2, lw=0.7, ls=":")
    axes[0].set_xlim(max(0, MODERN_TAB.auc.min() - 0.05), 1.005); axes[1].set_xlim(-0.02, 1.02)
    fig.legend(handles=[Line2D([], [], marker="o", color=INK2, ls="none", label=lab26["modern_human"]),
                        Line2D([], [], marker="s", mfc="white", color=INK2, ls="none", label=lab26["modern_llm"])],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    for ax, l in zip(axes, "abc"): panel_label(ax, f"({l})", x=-0.08 if l != "a" else -0.4)
    fig.tight_layout()
    savefig(fig, "fig26_modern", f"Out-of-time test on PhishFuzzer (decontaminated against the training corpus) with thresholds fixed on the original validation split: ROC-AUC with 95% CI (a), detection rate (b) and false-positive rate (c).")
