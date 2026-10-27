"""Step 3 - Correlation & (cautious) causal analysis of the cleaned cohort.

THE PREMISE: if someone has had cancer, will it come back - and for each factor linked to
it coming back, is that link correlation only, or plausibly causation?

  Q1  Who is in this clinic? (context)
  Q2  Which factors move together? (correlation matrix, context)
  Q3  How often, and how fast, does cancer come back? (Kaplan-Meier, at-risk cohort)
  Q4  Which factors are linked to it coming back? (crude Cox hazard ratios)
  Q5  Correlation or causation? (adjusted Cox HRs, E-values, trial evidence -> verdict per factor)

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
from lifelines import CoxPHFitter, KaplanMeierFitter
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


# ------------------------------------------------------------------ Q3 how often / how fast
def cohort(df):
    c = df[df.eligible_for_relapse == 1].copy()
    c["breast"] = (c.site_group == "Breast").astype(int)
    c["hr_pos"] = ((c.er == 1) | (c.pr == 1)).astype(int)
    c["triple_negative"] = (c.breast_subtype == "Triple negative").astype(int)
    c["grade3"] = (c.grade == 3).astype(int)
    c["khartoum"] = (c.region == "Khartoum").astype(int)
    c["relapse"] = c.relapse.astype(int)
    return c


def events_needed(hr, share_exposed=0.5, alpha_z=1.96, power_z=0.84):
    """Schoenfeld: relapses needed to detect a hazard ratio with 80% power (two-sided 5%)."""
    return int(np.ceil((alpha_z + power_z) ** 2 / (share_exposed * (1 - share_exposed) * np.log(hr) ** 2)))


def how_often(c, res, df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    kmf = KaplanMeierFitter().fit(c.rfs_years, c.relapse, label=f"All at-risk patients (n={len(c)})")
    kmf.plot_survival_function(ax=axes[0], color=BLUE, lw=2)
    axes[0].set_title("How often does cancer come back?")
    axes[0].set_ylabel("Probability still relapse-free")
    for mask, lab, col in ((c.breast == 1, "Breast", ORANGE), (c.breast == 0, "Other cancers", BLUE)):
        KaplanMeierFitter().fit(c.rfs_years[mask], c.relapse[mask], label=f"{lab} (n={mask.sum()})") \
            .plot_survival_function(ax=axes[1], color=col, lw=2, ci_show=False)
    lr = logrank_test(c.rfs_years[c.breast == 1], c.rfs_years[c.breast == 0],
                      c.relapse[c.breast == 1], c.relapse[c.breast == 0]).p_value
    axes[1].set_title(f"Breast vs other cancers (log-rank p={lr:.3f})")
    for ax in axes:
        ax.set_xlabel("Years since diagnosis")
        ax.set_ylim(0, 1.05)
    save(fig, "09_relapse_free_survival.png")
    res["premise"] = {
        "at_risk_cohort": len(c), "relapsed": int(c.relapse.sum()), "crude_rate": round(c.relapse.mean(), 3),
        "relapse_free": {f"{t}y": round(float(kmf.predict(t)), 2) for t in (2, 5, 10)},
        "median_followup_years": float(c.rfs_years.median()),
        "breast_vs_other_logrank_p": round(lr, 3),
        "second_primary_cancers_mentioned_all_patients": int(df.second_primary.sum()),
        "relapses_needed_to_detect_HR_0.5": events_needed(0.5),
        "relapses_needed_to_detect_HR_0.7": events_needed(0.7),
        "cohort_rule": "malignant, not metastatic at diagnosis, received treatment, diagnosed >= 1 year before note",
    }


# ------------------------------------------------------------------ Q4 + Q5 correlation vs causation
# factor -> (column, adjustment set, population, kind, trial/biology evidence)
FACTORS = {
    "Breast cancer (vs other sites)": ("breast", [], "all", "marker",
                                       "Many breast cancers have good long-term control; varies by subtype"),
    "Hormone-receptor positive (breast)": ("hr_pos", ["grade3"], "breast", "marker",
                                           "HR+ relapses less early on, but can relapse late"),
    "Triple-negative (breast)": ("triple_negative", ["grade3"], "breast", "marker",
                                 "Well established: relapses more often and earlier"),
    "Grade 3 tumour": ("grade3", ["breast"], "all", "marker", "Well established: higher grade, higher risk"),
    "Surgery": ("surgery", ["breast"], "all", "treatment", "Yes, for operable disease"),
    "Radiotherapy": ("radiotherapy", ["breast", "surgery"], "all", "treatment",
                     "Yes: trials show adjuvant RT cuts local recurrence"),
    "Hormonal therapy": ("hormonal_therapy", ["hr_pos", "surgery", "radiotherapy"], "breast", "treatment",
                         "Yes: trials show ~5 yrs of tamoxifen cuts recurrence in HR+ breast cancer"),
    "Neoadjuvant chemo (large tumour at start)": ("neoadjuvant_chemo", ["breast"], "all", "marker",
                                                  "Marks larger / locally advanced tumours"),
    "Treated abroad": ("treated_abroad", ["breast"], "all", "access", "No direct evidence"),
    "Missed follow-up / delayed care": ("missed_followup_or_delay", ["breast"], "all", "access",
                                        "Delay worsens outcomes in general"),
    "Lives in Khartoum (capital)": ("khartoum", ["breast"], "all", "access", "Access to care, no direct evidence"),
}


def cox_hr(d, col, adj):
    cols = [col] + [a for a in adj if a != col and d[a].nunique() > 1]
    cph = CoxPHFitter(penalizer=0.01).fit(d[cols + ["rfs_years", "relapse"]], "rfs_years", "relapse")
    s = cph.summary.loc[col]
    return float(s["exp(coef)"]), float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"]), \
        float(s["p"]), len(cols) - 1


def e_value(hr, lo, hi):
    """VanderWeele & Ding E-value for a hazard ratio with a common outcome (>15%)."""
    def rr(h):
        return (1 - 0.5 ** np.sqrt(h)) / (1 - 0.5 ** np.sqrt(1 / h))

    def ev(r):
        r = 1 / r if r < 1 else r
        return r + np.sqrt(r * (r - 1))
    point = ev(rr(hr))
    bound = 1.0 if lo <= 1 <= hi else ev(rr(lo if hr > 1 else hi))
    return round(point, 2), round(bound, 2)


def verdict(kind, crude_sig, adj_sig, crude_hr, adj_hr, n_adj):
    if not crude_sig and not adj_sig:
        return "No clear link (too few patients to tell)"
    if kind == "marker":
        return "Risk marker: real association, but not something you can change"
    if crude_sig and not adj_sig and n_adj:
        return "Correlation: explained by measured confounders"
    return "Association survives adjustment; causal only if no stronger unmeasured confounder (e.g. stage)"


def correlation_vs_causation(c, res):
    rows = []
    for name, (col, adj, pop, kind, evidence) in FACTORS.items():
        d = c if pop == "all" else c[c.breast == 1]
        x = d[col] == 1
        if x.sum() < 5 or (~x).sum() < 5:
            continue
        hr, lo, hi, p, _ = cox_hr(d, col, [])
        ahr, alo, ahi, ap, n_adj = cox_hr(d, col, adj) if adj else (hr, lo, hi, p, 0)
        ev, ev_ci = e_value(ahr, alo, ahi)
        rows.append({
            "factor": name, "population": "breast only" if pop == "breast" else "all sites", "n": len(d),
            "relapse_with": f"{int(d.relapse[x].sum())}/{int(x.sum())} ({d.relapse[x].mean():.0%})",
            "relapse_without": f"{int(d.relapse[~x].sum())}/{int((~x).sum())} ({d.relapse[~x].mean():.0%})",
            "crude_HR": round(hr, 2), "crude_CI": f"{lo:.2f}-{hi:.2f}", "crude_p": round(p, 3),
            "adjusted_for": ", ".join(adj) or "-", "adjusted_HR": round(ahr, 2), "adjusted_CI": f"{alo:.2f}-{ahi:.2f}",
            "adjusted_p": round(ap, 3), "E_value": ev, "E_value_CI": ev_ci,
            "outside_evidence": evidence,
            "verdict": verdict(kind, p < 0.05, ap < 0.05, hr, ahr, n_adj),
            "_lo": lo, "_hi": hi, "_alo": alo, "_ahi": ahi,
        })
    t = pd.DataFrame(rows)
    t["crude_p_fdr"] = multipletests(t.crude_p, method="fdr_bh")[1].round(3)
    t.drop(columns=[c for c in t.columns if c.startswith("_")]) \
        .to_csv(TAB / "correlation_vs_causation.csv", index=False)

    # forest plot: crude (hollow) vs adjusted (filled)
    t = t.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(t) + 1.6))
    for i, r in t.iterrows():
        col = ORANGE if r._alo > 1 else (BLUE if r._ahi < 1 else GRAY)
        ax.plot([r._lo, r._hi], [i + 0.15, i + 0.15], color=GRAY, lw=1.2)
        ax.plot(r.crude_HR, i + 0.15, "o", mfc=SURF, mec=GRAY, mew=1.5, ms=7)
        ax.plot([r._alo, r._ahi], [i - 0.15, i - 0.15], color=col, lw=2.2, solid_capstyle="round")
        ax.plot(r.adjusted_HR, i - 0.15, "o", color=col, ms=8, mec=SURF, mew=1.5)
        ax.text(25, i, r.relapse_with, va="center", fontsize=8, color=INK2)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 60)
    ax.set_yticks(range(len(t)), [f"{f}  [{p}]" for f, p in zip(t.factor, t.population)])
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(t) - 0.1)
    ax.text(25, len(t) - 0.45, "relapse rate", fontsize=8, color=INK2, weight="bold")
    ax.set_xlabel("Hazard ratio for cancer coming back (log scale, 95% CI)  ← less | more →")
    ax.set_title(f"Correlation (hollow, crude) vs adjusted (filled)\n"
                 f"{len(c)} at-risk patients, {int(c.relapse.sum())} relapses", loc="left")
    save(fig, "08_relapse_forest_plot.png")

    out = t.iloc[::-1].drop(columns=[c for c in t.columns if c.startswith("_")])
    res["correlation_vs_causation"] = out.to_dict(orient="records")

    # multivariable Cox model for the whole picture
    cols = ["breast", "grade3", "surgery", "radiotherapy", "hormonal_therapy", "neoadjuvant_chemo"]
    cph = CoxPHFitter(penalizer=0.01).fit(c[cols + ["rfs_years", "relapse"]], "rfs_years", "relapse")
    mv = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3)
    mv.columns = ["HR", "ci_low", "ci_high", "p"]
    mv.to_csv(TAB / "relapse_multivariable_cox.csv")
    res["multivariable_cox"] = {"concordance": round(cph.concordance_index_, 3), **mv.to_dict(orient="index")}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    for old in ("relapse_univariate_odds_ratios.csv", "relapse_multivariable_logit.csv",
                "confounding_hormonal_therapy.csv"):
        (TAB / old).unlink(missing_ok=True)
    df = pd.read_csv(DATA)
    df["breast_subtype"] = df.breast_subtype.fillna("")
    res = {}
    profile(df, res)
    correlation(df, res)
    c = cohort(df)
    how_often(c, res, df)
    correlation_vs_causation(c, res)
    (ROOT / "reports" / "results.json").write_text(json.dumps(res, indent=2, default=float, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("premise", "multivariable_cox")}, indent=1, default=float))
    print(pd.read_csv(TAB / "correlation_vs_causation.csv")[["factor", "population", "relapse_with", "relapse_without",
          "crude_HR", "crude_CI", "adjusted_HR", "adjusted_CI", "E_value", "verdict"]].to_string())


if __name__ == "__main__":
    main()
