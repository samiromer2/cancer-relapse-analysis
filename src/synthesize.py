"""Step 4 - Generate synthetic patients that look statistically like the real cohort.

Method: Gaussian copula over the structured factors (no free text is synthesised).
  1. every column -> uniform via its empirical CDF (categoricals: randomised PIT)
  2. uniform -> standard normal; estimate the correlation matrix
  3. sample multivariate normal -> back through each column's empirical inverse CDF
  4. enforce clinical logic (site<->sex, breast-only receptor fields, outcome eligibility)
  5. outcome layer: relapse is re-drawn from a logistic model fitted on the real curative
     cohort, so factor->relapse associations survive (a copula alone washes them out)

Validation (reports/synthetic_quality.json + figure):
  * fidelity  - marginal distance (TVD) per column, correlation-matrix difference
  * utility   - relapse rate & key odds ratios recomputed on synthetic data
  * privacy   - no synthetic row is an exact copy of a real row; distance-to-closest-record

Usage:  python src/synthesize.py --n 5000 --seed 7
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "processed" / "patients_features.csv"
OUT = ROOT / "data" / "synthetic"

CATEGORICAL = ["region", "site_group", "sex", "breast_subtype", "histology", "ses_code",
               "er", "pr", "her2", "grade"]
NUMERIC = ["dx_year", "years_since_dx", "n_visits", "n_units_seen"]
BINARY = ["metastatic_at_dx", "metastatic", "mets_bone", "mets_liver", "mets_lung", "mets_brain",
          "surgery", "chemotherapy", "neoadjuvant_chemo", "radiotherapy", "hormonal_therapy",
          "bisphosphonate", "opioid", "palliative_referral", "treated_abroad",
          "missed_followup_or_delay", "recurrence_or_progression", "eligible_for_relapse"]
COLUMNS = CATEGORICAL + NUMERIC + BINARY


def prepare(real: pd.DataFrame) -> pd.DataFrame:
    d = real[COLUMNS + ["site"]].copy()
    for c in CATEGORICAL:
        d[c] = d[c].astype(object).where(d[c].notna(), "missing").astype(str).replace("", "n/a")
    for c in NUMERIC:
        d[c] = d[c].astype(float)
    return d


class GaussianCopula:
    def fit(self, d: pd.DataFrame, rng):
        self.cols = COLUMNS
        self.meta = {}
        Z = np.zeros((len(d), len(self.cols)))
        for j, c in enumerate(self.cols):
            if c in NUMERIC:
                x = d[c]
                self.meta[c] = {"kind": "num", "values": np.sort(x.dropna().values),
                                "p_missing": x.isna().mean()}
                filled = x.fillna(x.median())
                u = (filled.rank(method="average") - 0.5) / len(filled)
            else:
                freq = d[c].astype(str).value_counts(normalize=True)
                cats, cum = freq.index.tolist(), np.concatenate([[0], freq.cumsum().values])
                self.meta[c] = {"kind": "cat", "cats": cats, "cum": cum}
                idx = d[c].astype(str).map({k: i for i, k in enumerate(cats)}).values
                u = rng.uniform(cum[idx], cum[idx + 1])  # randomised probability integral transform
            Z[:, j] = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        self.corr = np.corrcoef(Z, rowvar=False)
        # nearest positive-definite + slight shrinkage for stability with n=335
        self.corr = 0.95 * self.corr + 0.05 * np.eye(len(self.cols))
        self.site_given_group = d.groupby("site_group").site.value_counts(normalize=True)
        return self

    def sample(self, n, rng) -> pd.DataFrame:
        Z = rng.multivariate_normal(np.zeros(len(self.cols)), self.corr, size=n)
        U = norm.cdf(Z)
        out = {}
        for j, c in enumerate(self.cols):
            m, u = self.meta[c], U[:, j]
            if m["kind"] == "num":
                v = np.quantile(m["values"], u, method="inverted_cdf")
                miss = rng.uniform(size=n) < m["p_missing"]
                out[c] = np.where(miss, np.nan, v)
            else:
                idx = np.searchsorted(m["cum"], u, side="right") - 1
                out[c] = np.array(m["cats"], dtype=object)[np.clip(idx, 0, len(m["cats"]) - 1)]
        s = pd.DataFrame(out)
        s["site"] = [rng.choice(self.site_given_group[g].index, p=self.site_given_group[g].values)
                     for g in s.site_group]
        return s


OUTCOME_X = ["hormonal_therapy", "radiotherapy", "surgery", "breast", "years_since_dx"]


def _outcome_design(d):
    X = pd.DataFrame({
        "hormonal_therapy": d.hormonal_therapy, "radiotherapy": d.radiotherapy, "surgery": d.surgery,
        "breast": (d.site_group == "Breast").astype(int),
        "years_since_dx": d.years_since_dx.fillna(d.years_since_dx.median()).clip(0, 15),
    }).astype(float)
    return sm.add_constant(X, has_constant="add")


def fit_outcome_models(real):
    c = real[real.eligible_for_relapse == 1]
    relapse = sm.Logit(c.relapse.astype(int), _outcome_design(c)).fit(disp=0)
    prog_rate = float(real.progression_if_metastatic.mean())
    return relapse, prog_rate, float(c.relapse.mean())


def draw_outcomes(s, relapse_model, prog_rate, target_rate, rng):
    s = s.copy()
    elig = s.eligible_for_relapse == 1
    lin = _outcome_design(s[elig]).values @ relapse_model.params.values
    # calibration-in-the-large: shift the intercept so the overall relapse rate matches the real one
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(lin + mid)))).mean() < target_rate else (lo, mid)
    p = 1 / (1 + np.exp(-(lin + mid)))
    s.loc[elig, "recurrence_or_progression"] = (rng.uniform(size=elig.sum()) < p).astype(int)
    met = s.metastatic_at_dx == 1
    s.loc[met, "recurrence_or_progression"] = (rng.uniform(size=met.sum()) < prog_rate).astype(int)
    s["relapse"] = np.where(elig, s.recurrence_or_progression, np.nan)
    return s


def enforce_logic(s: pd.DataFrame, rng) -> pd.DataFrame:
    s = s.copy()
    for c in BINARY:
        s[c] = s[c].astype(int)
    for c in ("er", "pr", "her2", "grade"):
        s[c] = pd.to_numeric(s[c].replace("missing", np.nan), errors="coerce")
    s["breast_subtype"] = s.breast_subtype.replace("n/a", "")
    s["ses_code"] = s.ses_code.replace("n/a", "")
    breast = s.site_group == "Breast"
    s.loc[~breast, ["er", "pr", "her2"]] = np.nan
    s.loc[~breast, "breast_subtype"] = ""
    s.loc[breast & (s.breast_subtype == ""), "breast_subtype"] = "Unknown"
    s.loc[breast & (s.breast_subtype == "Triple negative"), ["er", "pr", "her2"]] = 0
    s.loc[breast & (s.breast_subtype == "HER2+"), "her2"] = 1
    s.loc[s.site_group == "Gynaecologic", "sex"] = "F"
    s.loc[s.site.isin(["Prostate", "Testis"]), "sex"] = "M"
    s.loc[breast & (s.sex == "M") & (rng.uniform(size=len(s)) > 0.01), "sex"] = "F"  # ~1% male breast ca
    s.loc[s.metastatic_at_dx == 1, "metastatic"] = 1
    met_sites = ["mets_bone", "mets_liver", "mets_lung", "mets_brain"]
    s.loc[s.metastatic == 0, met_sites] = 0  # site flags follow the overall flag, not vice versa
    s["n_met_sites"] = s[met_sites].sum(axis=1)
    s["years_since_dx"] = s.years_since_dx.clip(lower=0)
    s["registered_year"] = np.where(s.dx_year.notna(), (s.dx_year + s.years_since_dx).clip(upper=2021), 2021)
    s.loc[s.metastatic_at_dx == 1, "eligible_for_relapse"] = 0
    s["relapse"] = np.where(s.eligible_for_relapse == 1, s.recurrence_or_progression, np.nan)
    s.insert(0, "patient_id", [f"S{i:06d}" for i in range(1, len(s) + 1)])
    s["is_synthetic"] = 1
    return s


# ------------------------------------------------------------------ validation
def tvd(a: pd.Series, b: pd.Series) -> float:
    if a.dtype.kind in "fi" and a.nunique() > 12:
        bins = np.histogram_bin_edges(pd.concat([a, b]).dropna(), bins=12)
        pa, _ = np.histogram(a.dropna(), bins=bins)
        pb, _ = np.histogram(b.dropna(), bins=bins)
        pa, pb = pa / max(pa.sum(), 1), pb / max(pb.sum(), 1)
        return float(0.5 * np.abs(pa - pb).sum())
    pa = a.astype(str).value_counts(normalize=True)
    pb = b.astype(str).value_counts(normalize=True)
    return float(0.5 * pa.subtract(pb, fill_value=0).abs().sum())


def encode(d: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(d[["region", "site_group", "sex", "breast_subtype"]].astype(str), dtype=float)
    for c in BINARY:
        X[c] = d[c].astype(float)
    for c in ("dx_year", "years_since_dx", "n_visits"):
        v = d[c].astype(float)
        X[c] = (v - v.mean()) / (v.std() or 1)
    return X.fillna(0)


def odds(df, mask):
    c = df[df.eligible_for_relapse == 1]
    x, y = mask(c), c.relapse == 1
    a, b, cc, dd = ((x & y).sum() + .5, (x & ~y).sum() + .5, (~x & y).sum() + .5, (~x & ~y).sum() + .5)
    return round(float(a * dd / (b * cc)), 2)


def validate(real: pd.DataFrame, syn: pd.DataFrame, n_real: int) -> dict:
    r = real.copy()
    r["breast_subtype"] = r.breast_subtype.fillna("")
    r["ses_code"] = r.ses_code.fillna("")
    q = {"marginal_TVD": {c: round(tvd(r[c], syn[c]), 3) for c in COLUMNS}}
    q["marginal_TVD_mean"] = round(float(np.mean(list(q["marginal_TVD"].values()))), 3)
    cr = r[BINARY].astype(float).corr().values
    cs = syn[BINARY].astype(float).corr().values
    iu = np.triu_indices_from(cr, 1)
    q["corr_mean_abs_diff"] = round(float(np.nanmean(np.abs(cr[iu] - cs[iu]))), 3)

    q["utility"] = {
        "relapse_rate": {"real": round(float(r.relapse.mean()), 3), "synthetic": round(float(syn.relapse.mean()), 3)},
        "OR_hormonal_therapy": {"real": odds(r, lambda d: d.hormonal_therapy == 1),
                                "synthetic": odds(syn, lambda d: d.hormonal_therapy == 1)},
        "OR_radiotherapy": {"real": odds(r, lambda d: d.radiotherapy == 1),
                            "synthetic": odds(syn, lambda d: d.radiotherapy == 1)},
        "OR_breast": {"real": odds(r, lambda d: d.site_group == "Breast"),
                      "synthetic": odds(syn, lambda d: d.site_group == "Breast")},
    }

    # privacy: distance to closest real record vs real-to-real nearest neighbour
    both = encode(pd.concat([r.assign(_s=0), syn.assign(_s=1)], ignore_index=True))
    R, S = both.iloc[:n_real].values, both.iloc[n_real:].values
    S = S[np.random.default_rng(0).choice(len(S), size=min(2000, len(S)), replace=False)]
    d_syn = np.sqrt(((S[:, None, :] - R[None, :, :]) ** 2).sum(-1)).min(1)
    dr = np.sqrt(((R[:, None, :] - R[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dr, np.inf)
    d_real = dr.min(1)
    q["privacy"] = {
        "exact_copies_of_real_rows": int((d_syn < 1e-9).sum()),
        "median_DCR_synthetic_to_real": round(float(np.median(d_syn)), 3),
        "median_NN_real_to_real": round(float(np.median(d_real)), 3),
        "share_synthetic_closer_than_5th_pct_real_NN": round(float((d_syn < np.percentile(d_real, 5)).mean()), 3),
    }
    return q


def plot(real, syn, path):
    cols = ["site_group", "region", "sex", "metastatic", "hormonal_therapy", "relapse"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.flat, cols):
        a = real[c].fillna("n/a").astype(str).value_counts(normalize=True)
        b = syn[c].fillna("n/a").astype(str).value_counts(normalize=True).reindex(a.index).fillna(0)
        x = np.arange(len(a))
        ax.bar(x - 0.2, a.values, 0.38, color="#2a78d6", label="real")
        ax.bar(x + 0.2, b.values, 0.38, color="#eb6834", label="synthetic")
        ax.set_xticks(x, [s[:14] for s in a.index], rotation=40, ha="right", fontsize=8)
        ax.set_title(c, fontsize=10, loc="left", weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0")
        ax.set_axisbelow(True)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Real vs synthetic - marginal distributions", x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    real = pd.read_csv(REAL)
    model = GaussianCopula().fit(prepare(real), rng)
    copula_only = enforce_logic(model.sample(args.n, rng), rng)  # ablation: no outcome layer
    syn = draw_outcomes(copula_only, *fit_outcome_models(real), rng)

    OUT.mkdir(parents=True, exist_ok=True)
    syn.to_csv(OUT / "synthetic_patients.csv", index=False)
    q = validate(real, syn, len(real))
    q["ablation_copula_only"] = {
        "relapse_rate": round(float(copula_only.relapse.mean()), 3),
        "OR_hormonal_therapy": odds(copula_only, lambda d: d.hormonal_therapy == 1),
        "OR_radiotherapy": odds(copula_only, lambda d: d.radiotherapy == 1),
        "note": "copula alone washes out factor->relapse links; the outcome layer restores them",
    }
    (ROOT / "reports" / "synthetic_quality.json").write_text(json.dumps(q, indent=2))
    plot(real, syn, ROOT / "reports" / "figures" / "10_real_vs_synthetic.png")
    print(f"wrote {len(syn)} synthetic patients -> {OUT/'synthetic_patients.csv'}")
    print(json.dumps({k: v for k, v in q.items() if k != "marginal_TVD"}, indent=1))


if __name__ == "__main__":
    main()
