"""Step 3 - Correlation & (cautious) causal analysis of the cleaned cohort.

Questions:
  Q1  Who is in this clinic? (site, sex, region, spread, treatment)
  Q2  Which factors move together? (correlation matrix)
  Q3  After curative treatment, what is associated with the cancer coming back? (relapse)
  Q4  How fast does it come back? (Kaplan-Meier, time from diagnosis)
  Q5  Correlation vs causation: a worked confounding example.

Outputs: reports/figures/*.png, reports/tables/*.csv, reports/results.json
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "patients_features.csv"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"

# palette (validated reference instance; light surface)
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)


def hbar(series, title, name, xlabel="Patients", color=BLUE, pct_of=None):
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(s) + 1.2))
    ax.barh(s.index, s.values, color=color, height=0.68)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(s.values):
        lab = f"{v:.0f}" if pct_of is None else f"{v:.0f}  ({100 * v / pct_of:.0f}%)"
        ax.text(v + s.max() * 0.01, i, lab, va="center", color=INK2, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, s.max() * 1.18)
    save(fig, name)


# ------------------------------------------------------------------ Q1 profile
def profile(df, res):
    n = len(df)
    hbar(df.site_group.value_counts(), f"Cancer site group (n={n} patients)", "01_site_groups.png", pct_of=n)
    hbar(df.region.value_counts(), "Where patients come from (region, generalised)", "02_regions.png", pct_of=n)
    b = df[df.site_group == "Breast"]
    hbar(b.breast_subtype.replace("", "Unknown").value_counts(),
         f"Breast cancer receptor subtype (n={len(b)}; most notes don't state it)", "03_breast_subtypes.png",
         color=ORANGE, pct_of=len(b))
    mets = df[[c for c in df.columns if c.startswith("mets_")]].sum().rename(lambda c: c[5:].replace("_", " "))
    hbar(mets, "Documented metastasis sites (patients)", "04_metastasis_sites.png", pct_of=n)
    tx = df[["chemotherapy", "radiotherapy", "hormonal_therapy", "surgery", "palliative_referral", "opioid",
             "bisphosphonate", "neoadjuvant_chemo", "treated_abroad"]].sum()
    hbar(tx.rename(lambda c: c.replace("_", " ")), "Treatments mentioned in notes", "05_treatments.png", pct_of=n)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = df.years_since_dx.dropna()
    ax.hist(y, bins=np.arange(-0.5, y.max() + 1.5, 1), color=BLUE, rwidth=0.85)
    ax.set_xlabel("Years from diagnosis to clinic registration")
    ax.set_ylabel("Patients")
    ax.set_title("Most patients are seen within 2 years of diagnosis; a long tail of survivors")
    save(fig, "06_years_since_dx.png")

    res["profile"] = {
        "patients": n,
        "breast_share": round((df.site_group == "Breast").mean(), 3),
        "sex": df.sex.value_counts().to_dict(),
        "metastatic_any": int(df.metastatic.sum()),
        "metastatic_at_dx": int(df.metastatic_at_dx.sum()),
        "dx_year_known": int(df.dx_year.notna().sum()),
        "median_years_since_dx": float(df.years_since_dx.median()),
        "breast_subtypes": b.breast_subtype.replace("", "Unknown").value_counts().to_dict(),
        "receptor_known_breast": int((b.breast_subtype.fillna("Unknown") != "Unknown").sum()),
    }


# ------------------------------------------------------------------ Q2 correlation
CORR_VARS = {
    "breast": lambda d: (d.site_group == "Breast").astype(int),
    "female": lambda d: (d.sex == "F").astype(int),
    "metastatic": "metastatic", "mets at dx": "metastatic_at_dx", "bone mets": "mets_bone",
    "liver mets": "mets_liver", "lung mets": "mets_lung",
    "surgery": "surgery", "chemo": "chemotherapy", "radiotherapy": "radiotherapy",
    "hormonal tx": "hormonal_therapy", "zoledronic acid": "bisphosphonate", "opioid": "opioid",
    "palliative referral": "palliative_referral", "recurrence/progr.": "recurrence_or_progression",
    "disease-free note": "disease_free_statement", "missed f/u / delay": "missed_followup_or_delay",
    "Khartoum": lambda d: (d.region == "Khartoum").astype(int),
    "≥5 yrs since dx": lambda d: (d.years_since_dx >= 5).astype(int),
    "many visits (≥5)": lambda d: (d.n_visits >= 5).astype(int),
}


def correlation(df, res):
    X = pd.DataFrame({k: (df[v] if isinstance(v, str) else v(df)) for k, v in CORR_VARS.items()}).astype(float)
    C = X.corr()  # Pearson on binaries == phi coefficient
    C.round(3).to_csv(TAB / "phi_correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    mask = np.triu(np.ones_like(C, dtype=bool), k=1)
    im = ax.imshow(np.ma.array(C, mask=mask), cmap=cmap, vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(C)), C.columns, rotation=55, ha="right")
    ax.set_yticks(range(len(C)), C.columns)
    ax.grid(False)
    for i in range(len(C)):
        for j in range(i + 1):
            v = C.iat[i, j]
            if i != j and abs(v) >= 0.2:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.7, label="phi correlation (−: blue, +: red)")
    ax.set_title("How clinical factors move together (phi; |r| ≥ 0.20 labelled)")
    save(fig, "07_correlation_heatmap.png")

    pairs = C.where(~np.tril(np.ones_like(C, dtype=bool))).stack().sort_values(key=abs, ascending=False)
    res["top_correlations"] = [{"a": a, "b": b, "phi": round(v, 2)} for (a, b), v in pairs.head(15).items()]


# ------------------------------------------------------------------ Q3 relapse factors
def odds_ratio(a, b, c, d):
    """2x2: exposed relapse a, exposed no-relapse b, unexposed relapse c, unexposed no-relapse d."""
    _, p = fisher_exact([[a, b], [c, d]])
    a, b, c, d = (x + 0.5 for x in (a, b, c, d))  # Haldane-Anscombe correction
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orr, np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se), p


RELAPSE_FACTORS = {
    "Breast cancer (vs other sites)": lambda d: d.site_group == "Breast",
    "Triple-negative breast": lambda d: d.breast_subtype == "Triple negative",
    "HER2-positive breast": lambda d: d.breast_subtype == "HER2+",
    "Hormone-receptor positive": lambda d: (d.er == 1) | (d.pr == 1),
    "Grade 3 tumour": lambda d: d.grade == 3,
    "Neoadjuvant chemo (big tumour at start)": lambda d: d.neoadjuvant_chemo == 1,
    "Had surgery": lambda d: d.surgery == 1,
    "Had radiotherapy": lambda d: d.radiotherapy == 1,
    "On hormonal therapy": lambda d: d.hormonal_therapy == 1,
    "Treated abroad (Egypt/India/Jordan)": lambda d: d.treated_abroad == 1,
    "Missed follow-up / delayed care": lambda d: d.missed_followup_or_delay == 1,
    "Lives in Khartoum": lambda d: d.region == "Khartoum",
    "≥5 years since diagnosis": lambda d: d.years_since_dx >= 5,
}


def relapse_factors(df, res):
    c = df[df.eligible_for_relapse == 1].copy()
    y = c.relapse.astype(int)
    rows = []
    for name, f in RELAPSE_FACTORS.items():
        x = f(c).fillna(False).astype(bool)
        a, b = int((x & (y == 1)).sum()), int((x & (y == 0)).sum())
        cc, dd = int((~x & (y == 1)).sum()), int((~x & (y == 0)).sum())
        if a + b < 3:
            continue
        orr, lo, hi, p = odds_ratio(a, b, cc, dd)
        rows.append({"factor": name, "n_with": a + b, "relapse_with": f"{a}/{a + b} ({a / (a + b):.0%})",
                     "relapse_without": f"{cc}/{cc + dd} ({cc / max(cc + dd, 1):.0%})",
                     "odds_ratio": round(orr, 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2),
                     "p_fisher": round(p, 4)})
    t = pd.DataFrame(rows)
    t["p_fdr"] = multipletests(t.p_fisher, method="fdr_bh")[1].round(4)
    t = t.sort_values("odds_ratio", ascending=False)
    t.to_csv(TAB / "relapse_univariate_odds_ratios.csv", index=False)

    # forest plot
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(t) + 1.4))
    yy = np.arange(len(t))[::-1]
    for yi, r in zip(yy, t.itertuples()):
        col = ORANGE if r.ci_low > 1 else (BLUE if r.ci_high < 1 else GRAY)
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(r.odds_ratio, yi, "o", color=col, ms=8, mec=SURF, mew=2)
        ax.text(60, yi, r.relapse_with, va="center", fontsize=8, color=INK2)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.03, 200)
    ax.set_yticks(yy, t.factor)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Odds ratio for relapse (log scale, 95% CI)  ← less relapse | more relapse →")
    ax.set_ylim(-0.7, len(t) - 0.1)
    ax.text(60, len(t) - 0.5, "relapse rate", fontsize=8, color=INK2, weight="bold")
    ax.set_title(f"What travels with relapse?\n{len(c)} curatively-treated patients, {int(y.sum())} relapsed")
    save(fig, "08_relapse_forest_plot.png")

    # multivariable logistic (small n -> few covariates, ridge-penalised for stability)
    X = pd.DataFrame({
        "years_since_dx": c.years_since_dx.fillna(c.years_since_dx.median()),
        "breast": (c.site_group == "Breast").astype(int),
        "triple_negative": (c.breast_subtype == "Triple negative").astype(int),
        "hormonal_therapy": c.hormonal_therapy,
        "radiotherapy": c.radiotherapy,
    }).astype(float)
    m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    ci = np.exp(m.conf_int())
    mv = pd.DataFrame({"odds_ratio": np.exp(m.params).round(2), "ci_low": ci[0].round(2),
                       "ci_high": ci[1].round(2), "p": m.pvalues.round(4)}).drop("const")
    mv.to_csv(TAB / "relapse_multivariable_logit.csv")

    res["relapse"] = {
        "eligible": len(c), "relapsed": int(y.sum()), "rate": round(y.mean(), 3),
        "univariate": t.to_dict(orient="records"),
        "multivariable": mv.to_dict(orient="index"),
    }
    return c


# ------------------------------------------------------------------ Q4 time to relapse
def time_to_relapse(c, res):
    d = c[c.dx_year.notna()].copy()
    # event time: documented relapse year if known, else registration year (upper bound)
    d["T"] = np.where(d.relapse == 1, d.recurrence_year.fillna(d.registered_year), d.registered_year) - d.dx_year
    d["T"] = d["T"].clip(lower=0.5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    kmf = KaplanMeierFitter()
    kmf.fit(d["T"], d.relapse, label="All curatively-treated")
    kmf.plot_survival_function(ax=axes[0], color=BLUE, ci_show=True, lw=2)
    axes[0].set_title("Relapse-free survival (all sites)")
    axes[0].set_xlabel("Years since diagnosis")
    axes[0].set_ylabel("Probability still relapse-free")
    surv = {int(t): round(float(kmf.predict(t)), 2) for t in (2, 5, 8)}

    b = d[d.site_group == "Breast"]
    tn = b.breast_subtype == "Triple negative"
    lr = None
    for mask, lab, col in ((tn, "Triple-negative", ORANGE), (~tn, "Other / unknown subtype", BLUE)):
        if mask.sum() >= 3:
            KaplanMeierFitter().fit(b["T"][mask], b.relapse[mask], label=f"{lab} (n={mask.sum()})") \
                .plot_survival_function(ax=axes[1], color=col, ci_show=False, lw=2)
    if tn.sum() >= 3:
        lr = logrank_test(b["T"][tn], b["T"][~tn], b.relapse[tn], b.relapse[~tn]).p_value
    axes[1].set_title(f"Breast: triple-negative vs rest (log-rank p={lr:.2f})" if lr else "Breast")
    axes[1].set_xlabel("Years since diagnosis")
    for ax in axes:
        ax.set_ylim(0, 1.05)
    save(fig, "09_relapse_free_survival.png")
    res["km"] = {"n": len(d), "relapse_free_at_years": surv, "breast_tn_logrank_p": lr}


# ------------------------------------------------------------------ Q5 confounding demo
def confounding_demo(c, res):
    """Does hormonal therapy 'prevent' relapse? Crude vs stratified by cancer site / receptor status."""
    from statsmodels.stats.contingency_tables import StratifiedTable
    c = c.copy()
    c["stratum"] = np.where(c.site_group != "Breast", "Other sites",
                            np.where((c.er == 1) | (c.pr == 1), "Breast, HR+", "Breast, HR- or unknown"))
    out, tab, tables = {}, [], []
    for stratum, g in [("Crude (all)", c)] + list(c.groupby("stratum")):
        x, y = g.hormonal_therapy == 1, g.relapse == 1
        a, b, cc, dd = (x & y).sum(), (x & ~y).sum(), (~x & y).sum(), (~x & ~y).sum()
        orr, lo, hi, p = odds_ratio(a, b, cc, dd)
        out[stratum] = {"OR": round(orr, 2), "ci": [round(lo, 2), round(hi, 2)], "n": len(g),
                        "on_HT": int(x.sum())}
        tab.append([stratum, len(g), f"{a}/{a + b}", f"{cc}/{cc + dd}", f"{orr:.2f} ({lo:.2f}-{hi:.2f})"])
        if stratum != "Crude (all)":
            tables.append(np.array([[a, b], [cc, dd]]) + 0.5)
    st = StratifiedTable(tables)
    lo, hi = st.oddsratio_pooled_confint()
    out["Mantel-Haenszel adjusted"] = {"OR": round(st.oddsratio_pooled, 2), "ci": [round(lo, 2), round(hi, 2)]}
    tab.append(["Mantel-Haenszel (site/HR adjusted)", len(c), "", "",
                f"{st.oddsratio_pooled:.2f} ({lo:.2f}-{hi:.2f})"])
    pd.DataFrame(tab, columns=["stratum", "n", "relapse on HT", "relapse no HT", "OR (95% CI)"]) \
        .to_csv(TAB / "confounding_hormonal_therapy.csv", index=False)
    res["confounding_hormonal_therapy"] = out

    # appointment-based progression proxy agrees with text-mined relapse?
    res["booking_signal"] = {
        "escalated_from_HT_patients": int(c.escalated_from_HT.sum()),
        "relapse_rate_if_escalated": round(c[c.escalated_from_HT == 1].relapse.mean(), 2)
        if c.escalated_from_HT.sum() else None,
    }


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df["breast_subtype"] = df.breast_subtype.fillna("")
    res = {}
    profile(df, res)
    correlation(df, res)
    c = relapse_factors(df, res)
    time_to_relapse(c, res)
    confounding_demo(c, res)
    (ROOT / "reports" / "results.json").write_text(json.dumps(res, indent=2, default=float, ensure_ascii=False))
    print(json.dumps(res, indent=1, default=float, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
