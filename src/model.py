"""Stages 6-7 - MODEL & EVALUATE: can we predict which patients' cancer will come back?

Business question: the clinic has limited follow-up slots and scans. Can we flag the curatively
treated patients most likely to relapse so they are seen first?

  target y   : relapse (1 = cancer came back after curative-intent treatment)
  cohort     : eligible_for_relapse == 1  (n = 138, 30 events; same rule for relapsers and non-relapsers)
  features X : only information known BEFORE relapse (see LEAKAGE below)
  validation : repeated stratified 5-fold CV (20 repeats) - n is too small for one train/test split
  threshold  : flag a patient when predicted risk > cohort relapse rate (fixed in advance,
               favours recall: a missed relapse costs more than an extra scan)
  time-aware : cross-validated Cox model (C-index), because follow-up ranges 1-17 years

Runs on the PUBLIC k-anonymous file, so anyone can reproduce it from the repository.
If the private feature file is present, it also runs a fold-honest
"train on synthetic, test on real" (TSTR) experiment.
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data" / "public" / "patients_features_public.csv"
PRIVATE = ROOT / "data" / "processed" / "patients_features.csv"
FIG, TAB = ROOT / "reports" / "figures", ROOT / "reports" / "tables"
sys.path.insert(0, str(Path(__file__).parent))

BLUE, ORANGE, AQUA, GRAY, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e", "#e6e5e0", "#fcfcfb"

CATEGORICAL = ["site_group", "breast_subtype", "region"]
NUMERIC = ["years_since_dx", "grade_known", "grade3"]
BINARY = ["surgery", "radiotherapy", "hormonal_therapy", "neoadjuvant_chemo", "adjuvant_chemo",
          "anti_her2", "treated_abroad", "missed_followup_or_delay", "locally_advanced"]
# LEAKAGE - excluded on purpose: these are consequences of relapse, not causes
LEAKY = ["recurrence_or_progression", "recurrence_year", "metastatic", "n_met_sites", "mets_bone", "mets_liver",
         "mets_lung", "mets_brain", "mets_lymph_node", "lines_of_chemo_2plus", "palliative_referral", "opioid",
         "bisphosphonate", "chemotherapy", "escalated_from_HT", "visits_CT", "visits_PCF", "n_visits",
         "disease_free_statement"]
N_REPEATS, SEED = 20, 42


def load(path):
    d = pd.read_csv(path)
    d = d[d.eligible_for_relapse == 1].copy()
    d["breast_subtype"] = d.breast_subtype.fillna("not breast")
    d["region"] = d.region.fillna("Unknown")
    d["grade_known"] = d.grade.notna().astype(int)
    d["grade3"] = (d.grade == 3).astype(int)
    d["years_since_dx"] = d.years_since_dx.fillna(d.years_since_dx.median()).clip(upper=10)
    return d[CATEGORICAL + NUMERIC + BINARY], d.relapse.astype(int)


class RandomScore(DummyClassifier):
    """Baseline: a random risk score -> flagging the top X% is flagging X% of patients at random."""

    def predict_proba(self, X):
        p = np.random.default_rng(len(X) * 7919 + SEED + RandomScore._calls).uniform(size=len(X))
        RandomScore._calls += 1
        return np.column_stack([1 - p, p])


RandomScore._calls = 0


def models():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=3), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
        ("bin", "passthrough", BINARY),
    ])
    return {
        "Baseline (random flagging)": Pipeline([("pre", pre), ("m", RandomScore())]),
        "Logistic regression": Pipeline([("pre", pre), ("m", LogisticRegression(C=0.5, class_weight="balanced",
                                                                                 max_iter=2000))]),
        "Random forest": Pipeline([("pre", pre), ("m", RandomForestClassifier(
            n_estimators=400, max_depth=3, min_samples_leaf=4, class_weight="balanced", random_state=SEED))]),
    }


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat), "f1": f1_score(y, yhat, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan, "brier": brier_score_loss(y, p),
        "flagged_share": yhat.mean(),
    }


def cross_validate(X, y):
    thr = y.mean()
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(rskf.split(X, y))
    rows, oof_all, coefs = [], {}, []
    for name, pipe in models().items():
        oof = np.zeros((N_REPEATS, len(y)))
        for k, (tr, te) in enumerate(splits):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            oof[k // 5, te] = pipe.predict_proba(X.iloc[te])[:, 1]
            if name == "Logistic regression":
                feat = pipe.named_steps["pre"].get_feature_names_out()
                coefs.append(pd.Series(pipe.named_steps["m"].coef_[0], index=feat))
        # balanced class weights shift probabilities up -> recalibrate threshold to the same flag rate
        for r in range(N_REPEATS):
            t = np.quantile(oof[r], 1 - thr)
            rows.append({"model": name, "repeat": r, **metrics(y.values, oof[r], t)})
        oof_all[name] = oof
    res = pd.DataFrame(rows)
    summary = res.drop(columns="repeat").groupby("model").agg(["mean", "std"]).round(3)
    return res, summary, oof_all, pd.concat(coefs, axis=1)


def operating_points(y, oof):
    """Clinic trade-off: if we flag the top X% riskiest patients, how many relapses do we catch?"""
    rows = []
    for name in ("Logistic regression", "Random forest", "Baseline (random flagging)"):
        for share in (0.2, 0.36, 0.5, 0.7):
            r = [metrics(y, oof[name][k], np.quantile(oof[name][k], 1 - share)) for k in range(N_REPEATS)]
            rows.append({"model": name, "flag_top_share": share,
                         "recall": np.mean([m["recall"] for m in r]), "precision": np.mean([m["precision"] for m in r]),
                         "f1": np.mean([m["f1"] for m in r]),
                         "patients_flagged_per_100": round(share * 100),
                         "relapses_caught_per_100": np.mean([m["recall"] for m in r]) * y.mean() * 100})
    return pd.DataFrame(rows).round(3)


def cox_cv(path):
    """Time-aware check: cross-validated concordance (C-index) of a Cox model.

    Patients are followed 1-17 years, so a yes/no classifier mixes 'did not relapse' with
    'not followed long enough'. The C-index asks: of two patients, does the model give the
    higher risk to the one who relapses first?  0.5 = coin flip.
    """
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    d = pd.read_csv(path)
    d = d[d.eligible_for_relapse == 1].copy()
    d["breast"] = (d.site_group == "Breast").astype(int)
    d["grade3"] = (d.grade == 3).astype(int)
    d["relapse"] = d.relapse.astype(int)
    cols = ["breast", "grade3", "surgery", "radiotherapy", "hormonal_therapy", "neoadjuvant_chemo"]
    out = {}
    for label, use in (("Cox: site + grade + treatments", cols), ("Cox: breast vs other only", ["breast"])):
        scores = []
        for r in range(N_REPEATS):
            skf = StratifiedKFold(5, shuffle=True, random_state=r)
            risk = np.zeros(len(d))
            for tr, te in skf.split(d, d.relapse):
                cph = CoxPHFitter(penalizer=0.1).fit(d.iloc[tr][use + ["rfs_years", "relapse"]], "rfs_years",
                                                     "relapse")
                risk[te] = cph.predict_partial_hazard(d.iloc[te][use]).values
            scores.append(concordance_index(d.rfs_years, -risk, d.relapse))
        out[label] = {"c_index_mean": round(float(np.mean(scores)), 3), "c_index_sd": round(float(np.std(scores)), 3)}
    return out


def tstr(rng_seed=SEED):
    """Train on Synthetic, Test on Real - synthesiser refit inside each fold (no leakage)."""
    if not PRIVATE.exists():
        return None
    from synthesize import GaussianCopula, draw_outcomes, enforce_logic, fit_outcome_models, prepare
    full = pd.read_csv(PRIVATE)
    X_real, y_real = load(PRIVATE)
    idx = X_real.index
    rng = np.random.default_rng(rng_seed)
    out = []
    for rep in range(3):
        skf = StratifiedKFold(5, shuffle=True, random_state=rep)
        p_syn, p_real = np.zeros(len(idx)), np.zeros(len(idx))
        for tr, te in skf.split(X_real, y_real):
            train_full = full.drop(index=idx[te])  # all patients except the held-out test patients
            syn = enforce_logic(GaussianCopula().fit(prepare(train_full), rng).sample(3000, rng), rng)
            syn = draw_outcomes(syn, *fit_outcome_models(train_full), rng)
            syn = syn.assign(adjuvant_chemo=0, anti_her2=0, locally_advanced=0)
            tmp = ROOT / "data" / "synthetic" / "_tstr_tmp.csv"
            syn.to_csv(tmp, index=False)
            Xs, ys = load(tmp)
            tmp.unlink()
            m = models()["Logistic regression"]
            p_syn[te] = m.fit(Xs, ys).predict_proba(X_real.iloc[te])[:, 1]
            p_real[te] = models()["Logistic regression"].fit(X_real.iloc[tr], y_real.iloc[tr]) \
                .predict_proba(X_real.iloc[te])[:, 1]
        out.append({"rep": rep, "auc_train_synthetic": roc_auc_score(y_real, p_syn),
                    "auc_train_real": roc_auc_score(y_real, p_real)})
    o = pd.DataFrame(out)
    return {"auc_train_on_synthetic": round(o.auc_train_synthetic.mean(), 3),
            "auc_train_on_real": round(o.auc_train_real.mean(), 3), "repeats": 3}


# ------------------------------------------------------------------ figures
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)


def plot_metrics(summary):
    keys = ["recall", "precision", "f1", "accuracy", "roc_auc"]
    names = ["Baseline (random flagging)", "Logistic regression", "Random forest"]
    colors = {n: (GRAY if "Baseline" in n else BLUE if "Logistic" in n else ORANGE) for n in names}
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    h = 0.26
    for i, n in enumerate(names):
        m = [summary.loc[n, (k, "mean")] for k in keys]
        s = [summary.loc[n, (k, "std")] for k in keys]
        y = np.arange(len(keys)) + (i - 1) * h
        ax.barh(y, m, h * 0.92, xerr=s, color=colors[n], label=n, error_kw={"ecolor": INK2, "lw": 1})
        for yi, v, sd in zip(y, m, s):
            ax.text(v + sd + 0.015, yi, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(np.arange(len(keys)), ["Recall", "Precision", "F1", "Accuracy", "ROC-AUC"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    style(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Relapse model performance (5-fold CV × 20 repeats, mean ± SD)", loc="left", weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "11_model_metrics.png", dpi=160)
    plt.close(fig)


def plot_confusion(y, oof, thr):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), facecolor=SURF)
    for ax, name in zip(axes, ["Logistic regression", "Random forest"]):
        cms = [confusion_matrix(y, (oof[name][r] >= np.quantile(oof[name][r], 1 - thr)).astype(int))
               for r in range(N_REPEATS)]
        cm = np.mean(cms, axis=0)
        ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", BLUE]))
        labels = [["True negative\n(no relapse, not flagged)", "False positive\n(extra follow-up)"],
                  ["False negative\n(MISSED relapse)", "True positive\n(relapse caught)"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.1f}\n{labels[i][j]}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "#0b0b0b")
        ax.set_xticks([0, 1], ["Predicted: no", "Predicted: relapse"])
        ax.set_yticks([0, 1], ["Actual: no", "Actual: relapse"])
        ax.set_title(name, loc="left", weight="bold", fontsize=10)
    fig.suptitle(f"Average confusion matrix per CV repeat (n = {len(y)} patients)", x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "12_confusion_matrices.png", dpi=160)
    plt.close(fig)


def plot_coefs(coefs):
    c = coefs.mean(axis=1)
    c = c[c.abs() > 0.05].sort_values()
    c.index = [i.split("__", 1)[1].replace("infrequent_sklearn", "other (rare)").replace("_", " ")
               for i in c.index]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(c) + 1.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.barh(c.index, c.values, color=[ORANGE if v > 0 else BLUE for v in c.values], height=0.65)
    ax.axvline(0, color=INK2, lw=1)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Logistic coefficient (log-odds; + = more relapse risk)")
    ax.set_title("What drives the model's risk score\n(associations, not causes)", loc="left", weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "13_model_coefficients.png", dpi=160)
    plt.close(fig)


def main():
    X, y = load(PUBLIC)
    print(f"cohort n={len(y)}, relapses={y.sum()} ({y.mean():.0%})")
    res, summary, oof, coefs = cross_validate(X, y)
    summary.to_csv(TAB / "model_metrics.csv")
    plot_metrics(summary)
    plot_confusion(y.values, oof, y.mean())
    plot_coefs(coefs)
    ops = operating_points(y.values, oof)
    ops.to_csv(TAB / "model_operating_points.csv", index=False)

    # sensitivity: drop hormonal therapy (may be reverse-caused: HT stopped after relapse)
    Xs = X.assign(hormonal_therapy=0)
    _, sens, _, _ = cross_validate(Xs, y)
    out = {
        "cohort": {"n": int(len(y)), "relapses": int(y.sum())},
        "threshold_rule": f"flag the top {y.mean():.0%} highest-risk patients (= cohort relapse rate)",
        "metrics": {m: {k: summary.loc[m, (k, "mean")] for k in ("recall", "precision", "f1", "accuracy",
                                                                   "roc_auc", "brier")} for m in summary.index},
        "operating_points": ops.to_dict(orient="records"),
        "sensitivity_without_hormonal_therapy_auc": {m: sens.loc[m, ("roc_auc", "mean")] for m in sens.index},
        "top_logistic_coefficients": coefs.mean(axis=1).sort_values(key=abs, ascending=False).head(8).round(2)
        .to_dict(),
        "cox_cross_validated": cox_cv(PUBLIC),
        "tstr": tstr(),
    }
    (ROOT / "reports" / "model_results.json").write_text(json.dumps(out, indent=2, default=float))
    print(summary.to_string())
    print(ops.to_string())
    print(json.dumps({k: out[k] for k in ("cox_cross_validated", "sensitivity_without_hormonal_therapy_auc", "top_logistic_coefficients",
                                          "tstr")}, indent=1, default=float))


if __name__ == "__main__":
    main()
