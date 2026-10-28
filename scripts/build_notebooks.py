"""Build the 8 stage notebooks (ACCESS -> WRANGLE -> EXPLORE -> VISUALIZE -> ENGINEER -> MODEL
-> EVALUATE -> RECOMMEND) and execute them.

The notebooks only read PUBLIC files (k-anonymous features, synthetic data, aggregate reports),
so they can be re-run by anyone from the repository and their outputs are safe to publish.
Logic lives in src/; the notebooks import it rather than copy it.

Usage:  .venv/bin/python scripts/build_notebooks.py          (build + execute)
        .venv/bin/python scripts/build_notebooks.py --no-run (build only)
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """\
%matplotlib inline
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
PUBLIC = ROOT / "data" / "public" / "patients_features_public.csv"
SYNTH = ROOT / "data" / "synthetic" / "synthetic_patients.csv"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "axes.axisbelow": True, "figure.dpi": 110})
pd.set_option("display.max_columns", 30); pd.set_option("display.width", 160)
df = pd.read_csv(PUBLIC)"""

PREMISE = ("> **Project premise:** if someone has had cancer and was treated, **will it come back?** "
           "And for every factor linked to it coming back: **correlation or causation?**\n>\n"
           "> All data here is public: a k-anonymous feature table (every combination of site group, region, sex "
           "and diagnosis period is shared by ≥ 3 patients), synthetic patients, and aggregate reports. "
           "The raw database and clinical notes stay private.")


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def notebook(title, stage, goal, business_q, cells, so_what):
    head = md(f"# {title}\n\n**Stage {stage} of 8** · ACCESS → WRANGLE → EXPLORE → VISUALIZE → ENGINEER → "
              f"MODEL → EVALUATE → RECOMMEND\n\n{PREMISE}\n\n**Goal of this stage:** {goal}\n\n"
              f"**Business question:** {business_q}")
    return [head, code(SETUP)] + cells + [md(f"## So what?\n\n{so_what}")]


# =========================================================================================== 01
nb1 = notebook(
    "01 · Access & Filter", 1,
    "load the data, understand its structure, and select the records that can answer the question.",
    "Which patients and which columns do we need to measure how often cancer comes back?",
    [
        md("""## Where the data comes from
The source is a MySQL dump of an oncology clinic's booking system (Sudan, 2020–2021). `src/parse_sql.py`
reads the dump straight into pandas, with no database server needed. The raw file is private,
so this notebook works on the **public, de-identified feature table** that the pipeline produces from it."""),
        code("""from parse_sql import load_dump   # how the raw dump is read (not run here: raw data is private)
quality = json.loads((ROOT / "reports" / "data_quality.json").read_text())
pd.DataFrame(quality["tables"], index=["rows", "columns"]).T.rename_axis("raw table")"""),
        md("### Q1 · How many rows and columns are in the dataset?"),
        code("df.shape"),
        md("### Q2 · Which columns are available, and what data type does each contain?"),
        code("""df.info()"""),
        md("""Every row is **one patient** (335). Most columns are 0/1 flags extracted from the clinical notes
(treatments, metastasis sites), plus a few categories (`site_group`, `region`, `sex`, `breast_subtype`)
and the two outcome columns: `relapse` (did it come back?) and `rfs_years` (years until it did, or until the last note)."""),
        md("### Q3 · Display only the columns needed for the premise"),
        code("""cols = ["patient_id", "site_group", "sex", "region", "surgery", "radiotherapy", "hormonal_therapy",
        "metastatic_at_dx", "eligible_for_relapse", "relapse", "rfs_years"]
df[cols].head()"""),
        md("### Q4 · Which patients have breast cancer? (Boolean filtering)"),
        code("""breast = df[df.site_group == "Breast"]
print(len(breast), "breast-cancer patients =", f"{len(breast)/len(df):.0%} of the clinic")"""),
        md("### Q5 · Which patients were already metastatic at diagnosis? (numeric condition)"),
        code("""denovo = df[df.metastatic_at_dx == 1]
print(len(denovo), "patients had spread at diagnosis. Their cancer never went away, so 'coming back' does not apply.")"""),
        md("### Q6 · Who can answer 'will it come back?' (AND logic)"),
        code("""at_risk = df[(df.non_malignant_tumour == 0) & (df.metastatic_at_dx == 0) &
             (df.eligible_for_relapse == 1)]
print(len(at_risk), "at-risk patients:", int(at_risk.relapse.sum()), "relapsed")"""),
        md("""The at-risk group uses **one rule for everyone**: malignant, not metastatic at diagnosis, treated,
and diagnosed at least a year before the note. (Version 1 of this project used a different rule for
patients with and without relapse. That bug is explained in notebook 05.)"""),
        md("### Q7 · Patients treated with radiotherapy OR hormonal therapy (OR logic) · `isin()` · `between()`"),
        code("""either = at_risk[(at_risk.radiotherapy == 1) | (at_risk.hormonal_therapy == 1)]
gyn_or_breast = at_risk[at_risk.site_group.isin(["Breast", "Gynaecologic"])]
followed_2_5 = at_risk[at_risk.rfs_years.between(2, 5)]
print(f"RT or HT: {len(either)} · breast/gynae: {len(gyn_or_breast)} · followed 2–5 yrs: {len(followed_2_5)}")"""),
        md("### Selecting by position (`iloc`) and by label (`loc`)"),
        code("""display(at_risk.iloc[:3, :6])
at_risk.loc[at_risk.relapse == 1, ["site_group", "radiotherapy", "hormonal_therapy", "rfs_years"]].head()"""),
    ],
    """To measure how often cancer comes back we need the **at-risk patients** (138 of 335), the outcome
(`relapse`, `rfs_years`) and the candidate factors (cancer type, treatments, biology, access). Patients who
were already metastatic at diagnosis (64) answer a different question (progression), not recurrence.""")

# =========================================================================================== 02
nb2 = notebook(
    "02 · Data Wrangling", 2,
    "find and fix the data-quality problems that would make the answer unreliable, and remove anything that identifies a person.",
    "What data-quality issues could make the relapse rate, or the correlation-vs-causation verdict, wrong?",
    [
        md("""The cleaning runs in `src/clean_anonymize.py` on the private raw dump. Below are its **aggregate**
results (from `reports/data_quality.json`), plus the cleaning functions demonstrated on **made-up** examples."""),
        code("""q = json.loads((ROOT / "reports" / "data_quality.json").read_text())
{k: v for k, v in q.items() if k not in ("tables", "region_counts")}"""),
        md("### Q1 · Which columns contain missing or unusable values?"),
        code("""pd.Series({
    "age = 0 (unusable)": f"{q['age_is_zero_share']:.1%} of rows",
    "blank diagnosis": q["rows_with_blank_diagnosis"],
    "blank clinical summary": q["rows_with_blank_summary"],
    "blank residence": q["rows_with_blank_residence"],
    "sex column exists?": q["has_sex_column"],
    "stage column exists?": q["has_stage_column"],
}, name="raw patients table").to_frame()"""),
        md("""### Q2 · How do we handle missing values? (don't automatically delete)
- **Age** is 0 for every real patient → treated as *missing*, not as age 0. It can't be imputed honestly, so it is
  a stated limitation.
- **Sex** has no column → inferred from cancer site and pronouns in the notes; left `Unknown` when unclear.
- **Receptor status / grade** → `NaN` when the note doesn't say; we never fill it with a guess.
- In the public file, missingness itself is visible:"""),
        code("""df[["sex", "grade", "er", "pr", "her2", "rfs_years"]].isna().sum().to_frame("missing")
"""),
        md("### Q3 · Inconsistent categories: one residence, many spellings"),
        code("""from clean_anonymize import residence_to_region
examples = ["Khartoum - Omdurman", "khartum", "الخرطوم بحري", "Gezira - Madani", "نيالا", "Darfur - Fasher"]
pd.Series({e: residence_to_region(e) for e in examples}, name="region")"""),
        md("""Free-typed residences in Arabic and English (with typos) are mapped to **7 regions**. This fixes the
inconsistency and also generalises a quasi-identifier (town → region)."""),
        md("### Q4 · Duplicate records: true duplicates or different people?"),
        code("""print(f"raw rows: {q['patients_raw_rows']}, test records removed: {q['test_records_removed']}")
print(f"merged duplicate entries: {q['rows_merged_or_dropped_as_duplicates']} → unique patients: {q['unique_patients_after_merge']}")
print(f"booking numbers reused for DIFFERENT people: {q['booking_numbers_reused_for_different_people']}")"""),
        md("""Duplicates were resolved by **hospital file number** or **booking number + same name**, never booking number
alone, because one booking number had been reused for two different people. Merging them blindly would have
mixed two patients' histories."""),
        md("### Q5–Q6 · Data types and dates: free-text dates → a consistent year"),
        code("""from clean_anonymize import scrub_text
made_up = ["diagnosed 14/3/2016, operated Dec 2016, seen by Dr Smith",
           "on 2232018 started tamoxifen; ER 8/8 PR 0/8",
           "mastectomy in 5/2015 at Kassala"]
pd.DataFrame({"made-up note": made_up, "after cleaning": [scrub_text(t) for t in made_up]})"""),
        md("""Dates come in many shapes (`dd/mm/yyyy`, glued `dmyyyy`, `Mon yyyy`). They are reduced to the **year**: enough
for time-to-relapse, and safer for privacy. Note that `ER 8/8` (a receptor score) is *not* mistaken for a date."""),
        md("### Q7 · Suspicious values: the 'ses' column and the test record"),
        code("""pd.Series(q["ses_value_counts"], name="ses values").to_frame()"""),
        md("""`ses` looks like a cancer stage (I, II, IV, V) but **isn't**: it doesn't line up with metastasis, palliative care or
site. It is kept as an unexplained code, and the clinic should be asked what it means. One record had a
joke diagnosis ("he's always happy") and was removed as test data. **Investigate before deleting**: here the
evidence was clear."""),
        md("### Privacy: is the public file really anonymous? (k-anonymity validation)"),
        code("""qi = ["site_group", "region", "sex", "dx_period"]
sizes = df.groupby(qi, dropna=False).size()
print("smallest group sharing the same quasi-identifiers: k =", sizes.min())
print("region suppressed for", (df.region == "Suppressed").sum(), "patients to reach k ≥ 3")"""),
    ],
    """The biggest quality issues aren't typos. They are **missing variables**: no age, no stage, no date of death, and receptor
status for only a third of breast patients. Those gaps, not the cleaning, are what limit the causal answer later.""")

# =========================================================================================== 03
nb3 = notebook(
    "03 · Exploratory Data Analysis", 3,
    "understand what the cleaned data says before building any model.",
    "How often does cancer come back, and which factors differ between patients whose cancer came back and those whose didn't?",
    [
        code("""at_risk = df[df.eligible_for_relapse == 1].copy()
at_risk["relapse"] = at_risk.relapse.astype(int)"""),
        md("### Q1 · Descriptive statistics for the key numeric variables"),
        code("""df[["years_since_dx", "rfs_years", "n_met_sites", "n_visits"]].describe().round(2)"""),
        md("### Q2 · How many relapsed vs did not? (the headline KPI)"),
        code("""at_risk.relapse.value_counts().rename({0: "did not come back", 1: "came back"}).to_frame("patients")"""),
        md("""## KPIs
| Clinic goal | KPI | Formula |
|---|---|---|
| Know how often cancer comes back | **Relapse rate** | relapses ÷ at-risk patients |
| Earlier diagnosis | **% metastatic at diagnosis** | metastatic at dx ÷ all patients |
| Better records | **% breast patients with receptor status** | breast with known subtype ÷ breast |
| Palliative capacity | **% referred to palliative care** | referred ÷ all patients |"""),
        code("""breast = df[df.site_group == "Breast"]
kpi = pd.Series({
    "Relapse rate (at-risk)": at_risk.relapse.mean(),
    "% metastatic at diagnosis": df.metastatic_at_dx.mean(),
    "% breast with receptor status": (breast.breast_subtype.fillna("Unknown") != "Unknown").mean(),
    "% referred to palliative care": df.palliative_referral.mean(),
}).map("{:.0%}".format)
kpi.to_frame("value")"""),
        md("### Q3 · Does the relapse rate differ by cancer type?"),
        code("""by_site = at_risk.groupby("site_group").relapse.agg(patients="size", relapses="sum", rate="mean")
by_site.sort_values("patients", ascending=False).style.format({"rate": "{:.0%}"})"""),
        md("### Q4 · Does it differ by treatment received?"),
        code("""rows = {}
for t in ["surgery", "radiotherapy", "hormonal_therapy", "neoadjuvant_chemo", "treated_abroad"]:
    g = at_risk.groupby(t).relapse.mean()
    rows[t] = {"rate if yes": g.get(1), "rate if no": g.get(0), "n yes": int((at_risk[t] == 1).sum())}
pd.DataFrame(rows).T.style.format({"rate if yes": "{:.0%}", "rate if no": "{:.0%}"})"""),
        md("### Q5 · Does follow-up time differ? (why time matters)"),
        code("""at_risk.groupby("relapse").rfs_years.describe().round(1)"""),
        md("""Patients who did **not** relapse have often been followed only a short time. A yes/no rate treats
"not back after 1 year" the same as "not back after 10". That's why the later notebooks use
**time-to-event** methods (Kaplan–Meier, Cox)."""),
        md("### Q6 · Relationship between two numeric variables: follow-up length vs number of visits"),
        code("""print("correlation (Spearman):", round(at_risk[["years_since_dx", "n_visits"]].corr(method="spearman").iloc[0, 1], 2))"""),
        md("### Q7 · Which factors move together? (correlation is not causation)"),
        code("""cols = ["surgery", "radiotherapy", "hormonal_therapy", "neoadjuvant_chemo", "relapse"]
X = at_risk[cols].astype(float).assign(breast=(at_risk.site_group == "Breast").astype(float))
X.corr().round(2)"""),
        md("""**Two patterns worth investigating:**
1. **Breast cancers came back less often** than other cancers.
2. **Hormonal therapy and radiotherapy go with less relapse**, but hormonal therapy is strongly tied to having
   breast cancer, so part of that link may simply be "breast cancer". EDA can't separate the two;
   notebook 08 tries to."""),
    ],
    """About **1 in 5 treated patients (22%)** saw their cancer come back. Breast cancer and some treatments go with
less relapse, but these are **correlations**, and the factors are tangled together (hormonal therapy ≈ breast).
Follow-up length varies a lot, so the next notebooks use time-aware methods.""")

# =========================================================================================== 04
nb4 = notebook(
    "04 · Visualization", 4,
    "turn the EDA findings into charts that answer a question, choosing each chart by the question and the variable types.",
    "How do we show clearly how often, how fast, and for whom cancer comes back?",
    [
        code("""from lifelines import KaplanMeierFitter
at_risk = df[df.eligible_for_relapse == 1].copy(); at_risk["relapse"] = at_risk.relapse.astype(int)"""),
        md("""| Question | Variable types | Chart |
|---|---|---|
| Compare relapse across cancer types | category → rate | **Bar chart** |
| How long have patients been followed? | one numeric | **Histogram** |
| Does follow-up differ by outcome? | numeric × category | **Box plot** |
| How fast does it come back? | time → probability | **Line (step) chart**, Kaplan–Meier |
| Relationship of two numerics | numeric × numeric | **Scatter plot** |"""),
        md("### Q1 · Bar chart: relapse rate by cancer type (only groups with ≥ 5 patients)"),
        code("""g = at_risk.groupby("site_group").relapse.agg(["size", "mean"])
g = g[g["size"] >= 5].sort_values("mean")
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(g.index, g["mean"], color=BLUE)
for i, (n, r) in enumerate(zip(g["size"], g["mean"])):
    ax.text(r + 0.01, i, f"{r:.0%}  (n={n})", va="center", fontsize=9)
ax.set_xlim(0, max(g["mean"]) * 1.4); ax.set_xlabel("Share whose cancer came back"); ax.grid(axis="y", visible=False)
ax.set_title("Relapse rate by cancer type", loc="left", weight="bold"); plt.show()"""),
        md("### Q2 · Histogram: how long have at-risk patients been followed?"),
        code("""fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(at_risk.rfs_years, bins=range(0, 12), color=BLUE, rwidth=0.85)
ax.set_xlabel("Years from diagnosis to relapse or last note (capped at 10 in the public file)"); ax.set_ylabel("Patients")
ax.set_title("Follow-up is short for many patients", loc="left", weight="bold"); plt.show()"""),
        md("### Q3 · Box plot: follow-up time for patients with and without relapse"),
        code("""fig, ax = plt.subplots(figsize=(7, 3))
data = [at_risk.loc[at_risk.relapse == k, "rfs_years"] for k in (0, 1)]
ax.boxplot(data, vert=False, tick_labels=["did not come back", "came back"], widths=0.5)
ax.set_xlabel("Years"); ax.set_title("Time to relapse vs time followed without relapse", loc="left", weight="bold"); plt.show()"""),
        md("### Q4 · Line chart over time: how fast does cancer come back? (Kaplan–Meier)"),
        code("""fig, ax = plt.subplots(figsize=(7, 3.6))
is_breast = at_risk.site_group == "Breast"
for mask, lab, col in ((is_breast, "Breast", ORANGE), (~is_breast, "Other cancers", BLUE)):
    KaplanMeierFitter().fit(at_risk.rfs_years[mask], at_risk.relapse[mask], label=f"{lab} (n={mask.sum()})") \\
        .plot_survival_function(ax=ax, color=col, lw=2)
ax.set_ylim(0, 1.05); ax.set_xlabel("Years since diagnosis"); ax.set_ylabel("Still relapse-free")
ax.set_title("Breast cancers come back later and less often", loc="left", weight="bold"); plt.show()"""),
        md("### Q5 · Scatter plot: years since diagnosis vs number of clinic visits"),
        code("""fig, ax = plt.subplots(figsize=(7, 3))
for k, col, lab in ((0, BLUE, "no relapse"), (1, ORANGE, "relapse")):
    s = at_risk[at_risk.relapse == k]
    ax.scatter(s.years_since_dx + np.random.uniform(-.15, .15, len(s)), s.n_visits, s=18, color=col, alpha=.7, label=lab)
ax.set_xlabel("Years since diagnosis"); ax.set_ylabel("Clinic visits (2020–21)"); ax.legend(frameon=False)
ax.set_title("No strong pattern between follow-up length and visits", loc="left", weight="bold"); plt.show()"""),
        md("### Q6 · The chart that carries the main finding (from `reports/figures`)"),
        code("""from IPython.display import Image
Image(str(ROOT / "reports" / "figures" / "08_relapse_forest_plot.png"), width=760)"""),
        md("""### Q7 · Chart → observation → meaning
- **Observation:** hollow (crude) and filled (adjusted) markers for hormonal therapy and radiotherapy sit left of 1,
  but their lines cross 1. Only "breast vs other" stays clearly left of 1.
- **Meaning:** the only factor this data clearly links to cancer coming back is **cancer type**. Treatment
  effects point the right way, but the data is too small to prove them."""),
    ],
    """A Kaplan–Meier curve answers the premise more honestly than any bar chart, because it respects that patients were
followed for different lengths of time. The forest plot is the bridge from correlation (crude) to the
causation question (adjusted).""")

# =========================================================================================== 05
nb5 = notebook(
    "05 · Feature Engineering", 5,
    "turn free-text clinical notes into model-ready variables, define the target correctly, and check for leakage.",
    "Which variables describe a patient *before* their cancer could come back, and how do we define 'came back' fairly?",
    [
        md("""### Q1 · Where do the features come from?
Most of the useful facts were **not** in database columns. They were written in free-text notes. `src/extract_features.py`
uses plain regular expressions (auditable by a clinician) to create 60 features. Demonstrated on **made-up** notes:"""),
        code("""from extract_features import classify_site, receptor, grade, recurrence_event, metastatic_at_diagnosis, has_not_negated
notes = [
    "Ca breast diagnosed 2015, IDC grade 3, ER+ PR+ HER2 negative, mastectomy, FEC 6, RT, tamoxifen. No evidence of mets.",
    "Ca rectum 2016, operated, 12 cycles FOLFOX, developed liver mets in 2019, on xeloda.",
    "Ca ovary with liver mets, newly diagnosed 2021, plan for chemotherapy.",
]
pd.DataFrame([{
    "site": classify_site("", n)[1], "grade": grade(n), "ER": receptor(n, "er"), "HER2": receptor(n, "her2"),
    "liver mets (negation-aware)": has_not_negated(r"liver|mets", n),
    "metastatic at diagnosis": metastatic_at_diagnosis("", n), "came back": recurrence_event(n),
} for n in notes], index=["note 1", "note 2", "note 3"])"""),
        md("""Note the three different situations: note 1 is **disease-free** ("No evidence of mets" must not count as metastasis),
note 2 **came back** after treatment, and note 3 was **already metastatic at diagnosis** (a different outcome)."""),
        md("### Q2 · Engineered features used in this project"),
        code("""groups = {
    "Cancer & biology": ["site_group", "histology", "grade", "er", "pr", "her2", "breast_subtype"],
    "Spread": ["metastatic_at_dx", "mets_bone", "mets_liver", "mets_lung", "mets_brain", "n_met_sites"],
    "Treatment": ["surgery", "chemotherapy", "neoadjuvant_chemo", "radiotherapy", "hormonal_therapy", "anti_her2"],
    "Access & course": ["region", "treated_abroad", "missed_followup_or_delay", "years_since_dx", "n_visits"],
    "Outcome": ["eligible_for_relapse", "relapse", "rfs_years"],
}
pd.DataFrame([(g, c, df[c].notna().mean()) for g, cs in groups.items() for c in cs],
             columns=["group", "feature", "non-missing"]).style.format({"non-missing": "{:.0%}"})"""),
        md("""### Q3 · The target: defining "came back" fairly (the most important decision in the project)
`relapse` = 1 if the note documents recurrence or new spread **after** treatment. It is defined only for the
**at-risk group**: malignant, not metastatic at diagnosis, treated, diagnosed ≥ 1 year before the note.

⚠️ **Lesson from version 1.** The first version required *curative* treatment only from patients **without**
relapse. Every untreated patient left in the group was therefore a relapse, which made every treatment look
protective (hormonal therapy OR 0.18) and the model look good (AUC 0.71). With **one rule for everyone**, the group is
138 patients and the effects shrink. Selection bias doesn't raise an error; it produces a nice chart."""),
        code("""at_risk = df[df.eligible_for_relapse == 1]
print(f"at-risk: {len(at_risk)}, relapses: {int(at_risk.relapse.sum())} ({at_risk.relapse.mean():.0%})")"""),
        md("### Q4 · Encoding categorical variables"),
        code("""from model import load
X, y = load(PUBLIC)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown="ignore", min_frequency=3).fit(X[["site_group"]])
print(list(enc.get_feature_names_out()))"""),
        md("""Rare categories (< 3 patients) are pooled into "infrequent", so a model can't memorise one patient."""),
        md("### Q5 · Grouping a numeric variable into meaningful bands"),
        code("""bands = pd.cut(at_risk.rfs_years, [0, 2, 5, 10.1], labels=["< 2 yrs", "2–5 yrs", "5+ yrs"], include_lowest=True)
at_risk.groupby(bands, observed=True).relapse.agg(["size", "mean"]).style.format({"mean": "{:.0%}"})"""),
        md("""### Q6–Q7 · Which variables are allowed as predictors? The leakage check
A feature is allowed only if it is known **before** the cancer could come back. These are excluded because they
are **consequences** of relapse, and including them would let the model "predict" the past:"""),
        code("""from model import LEAKY, CATEGORICAL, NUMERIC, BINARY
pd.DataFrame({"allowed predictors": pd.Series(CATEGORICAL + NUMERIC + BINARY),
              "excluded as leakage": pd.Series(LEAKY)}).fillna("")"""),
    ],
    """Feature engineering decided the answer more than any model did. The target definition (who is "at risk")
changed the headline result, and removing leaky features is what keeps the model honest.""")

# =========================================================================================== 06
nb6 = notebook(
    "06 · Modeling", 6,
    "build models that estimate each patient's risk of their cancer coming back, using only pre-relapse information.",
    "Can we rank treated patients by risk so that follow-up goes to the highest-risk patients first?",
    [
        md("""### Q1 · Target
`relapse` (1 = cancer came back), with `rfs_years` (time) for the survival model. 138 at-risk patients, 30 relapses."""),
        md("### Q2–Q3 · Features (X) and target (y)"),
        code("""import model as M
X, y = M.load(PUBLIC)
print("X:", X.shape, "| y:", y.shape, "| relapse rate:", f"{y.mean():.0%}")
X.head()"""),
        md("""### Q4 · Why not a single train/test split?
With 30 relapses, a 20% test set would contain only ~6 relapses, and the score would depend on luck. Instead:
**stratified 5-fold cross-validation, repeated 20 times**. Every patient is tested on a model that never saw them,
and we report the average and spread."""),
        md("""### Q5 · What kind of problem is this?
Two framings:
- **Classification** (came back yes/no): logistic regression, random forest.
- **Survival** (how long until it comes back): Cox proportional-hazards model. This is the better fit, because
  patients were followed 1 to 10+ years and "no relapse yet" depends on how long we've looked."""),
        md("### Q6–Q7 · Train the models and predict on unseen folds"),
        code("""res, summary, oof, coefs = M.cross_validate(X, y)
summary.xs("mean", axis=1, level=1)[["roc_auc", "accuracy", "precision", "recall", "f1"]].round(3)"""),
        code("""cox = M.cox_cv(PUBLIC)
pd.DataFrame(cox).T"""),
        md("""**Reading the results**
- The yes/no models are **close to a coin flip** (AUC ≈ 0.5–0.6). They can't distinguish "didn't relapse" from
  "not followed long enough".
- The time-aware **Cox model reaches a C-index of about 0.63**, and cancer type alone gets about 0.59. So most of the
  (weak) signal is cancer type."""),
        md("### What the logistic model leans on (associations, not causes)"),
        code("""c = coefs.mean(axis=1).sort_values()
c.index = [i.split("__", 1)[1] for i in c.index]
pd.concat([c.head(5), c.tail(5)]).round(2).to_frame("mean coefficient (+ = more risk)")"""),
    ],
    """With this data, relapse is **only weakly predictable**. That's a result, not a failure: it tells the clinic that
risk can't be estimated without the missing variables (stage, age, receptor status).""")

# =========================================================================================== 07
nb7 = notebook(
    "07 · Evaluation", 7,
    "decide whether the model is reliable enough to support a follow-up decision, using more than accuracy.",
    "If the clinic flags patients for early follow-up using this model, how many relapses would it catch, and at what cost?",
    [
        code("""import model as M
X, y = M.load(PUBLIC)
res, summary, oof, _ = M.cross_validate(X, y)
thr = y.mean()"""),
        md("### Q1 · Accuracy"),
        code("""summary.xs("mean", axis=1, level=1)[["accuracy"]].round(3)"""),
        md("""Accuracy looks fine (~0.65–0.73), but **always predicting "no relapse" would already score 78%**, because only 22% relapse.
That's why accuracy alone is misleading here."""),
        md("### Q2 · Confusion matrix (random forest, averaged over the 20 repeats)"),
        code("""from sklearn.metrics import confusion_matrix
rf = oof["Random forest"]
cm = np.mean([confusion_matrix(y, (rf[r] >= np.quantile(rf[r], 1 - thr)).astype(int)) for r in range(M.N_REPEATS)], axis=0)
pd.DataFrame(cm.round(1), index=["actual: no relapse", "actual: relapse"],
             columns=["predicted: no", "predicted: relapse"])"""),
        md("### Q3–Q5 · Precision, recall, F1"),
        code("""summary.xs("mean", axis=1, level=1)[["precision", "recall", "f1", "roc_auc"]].round(3)"""),
        md("""| Metric | Technical meaning | Clinic meaning |
|---|---|---|
| Accuracy | share of correct predictions | how often the flag is right overall |
| **Precision** | correct among flagged | of patients we bring back early, how many really relapse? |
| **Recall** | found among actual relapses | of patients whose cancer will come back, how many do we catch? |
| F1 | balance of the two | coverage vs wasted visits |
| False negative | missed relapse | cancer found later, when it may be harder to treat |
| False positive | flagged, no relapse | an extra visit or scan |"""),
        md("### Q6 · Operating points: how many patients should we flag?"),
        code("""ops = M.operating_points(y.values, oof)
ops.pivot(index="flag_top_share", columns="model", values="recall").style.format("{:.0%}") \\
   .set_caption("Recall (share of relapses caught) when flagging the top X% riskiest")"""),
        md("""### Q7 · Which error matters more?
A **false negative** (a relapse we miss) costs more than a **false positive** (an extra visit), so for follow-up
planning, **recall matters more than precision**. Even so, at every flagging level the models catch only a little
more than random flagging. This model is **not reliable enough** to drive follow-up decisions on its own."""),
        md("### Time-aware evaluation (the primary metric)"),
        code("""pd.DataFrame(M.cox_cv(PUBLIC)).T"""),
        md("""The **C-index** asks: of two patients, does the model give higher risk to the one who relapses first? 0.5 = chance,
1.0 = perfect. A cross-validated ~0.63 means **weak** discrimination."""),
    ],
    """Evaluated honestly (cross-validated, no leakage, compared with a random baseline), the model is **weak**. The
right action is not to deploy it, but to **collect the data that would make it work**: stage, age, receptor status, and
dates of relapse and death.""")

# =========================================================================================== 08
nb8 = notebook(
    "08 · Recommendation", 8,
    "answer the premise: correlation or causation. Then turn the findings into actions and state the limits.",
    "Will the cancer come back, and which links can we treat as cause and effect?",
    [
        md("## The finding (from the full analysis, `reports/tables/correlation_vs_causation.csv`)"),
        code("""t = pd.read_csv(ROOT / "reports" / "tables" / "correlation_vs_causation.csv")
t[["factor", "population", "relapse_with", "relapse_without", "crude_HR", "adjusted_HR", "adjusted_CI",
   "E_value", "verdict"]]"""),
        md("""### How each verdict was reached
For a link to count as **causal**, we asked four questions:
1. **Correlation:** is there a link at all? (crude hazard ratio)
2. **Adjustment:** does it survive adjusting for measured confounders? (Cox model)
3. **Robustness:** how strong would a hidden confounder, such as *stage*, have to be to explain it away? (E-value)
4. **Outside evidence:** do clinical trials agree?

Demonstrated for hormonal therapy (breast cancer only) on the public data:"""),
        code("""import analysis as A
%matplotlib inline
c = A.cohort(df)
b = c[c.breast == 1]
crude = A.cox_hr(b, "hormonal_therapy", [])
adj = A.cox_hr(b, "hormonal_therapy", ["hr_pos", "surgery", "radiotherapy"])
print(f"crude HR {crude[0]:.2f} ({crude[1]:.2f}–{crude[2]:.2f})")
print(f"adjusted HR {adj[0]:.2f} ({adj[1]:.2f}–{adj[2]:.2f})")
print("E-value (point, CI):", A.e_value(*adj[:3]))
print("relapses needed to detect HR 0.5 with 80% power:", A.events_needed(0.5), "— we have", int(c.relapse.sum()))"""),
        md("""## The answer to the premise
| | Answer |
|---|---|
| **Will it come back?** | For **about 1 in 5** treated patients (22%). Relapse-free ≈ 87% at 2 yrs, 78% at 5 yrs. |
| **Clear link** | **Cancer type**: breast cancer came back less than other cancers. A real association, but a *risk marker*, not a modifiable cause. |
| **Correlation, consistent with causation, not proven** | **Hormonal therapy** and **radiotherapy**: both point to fewer relapses, the same direction as clinical trials, but their confidence intervals include "no effect", and unrecorded stage could explain them. |
| **No clear link** | Surgery, grade, triple-negative, region, treated abroad, missed follow-up |

## Recommendations
1. **Record the variables that make causation answerable**: stage, age, ER/PR/HER2, date of relapse, date of death,
   as structured fields. Receptor status is missing for 68% of breast patients, and stage and age for everyone.
2. **Plan follow-up by cancer type.** Non-breast cancers came back sooner and more often. That's the link this data supports.
3. **Don't use the risk model for care decisions.** Its cross-validated C-index of about 0.63 is weak.
4. **Re-run once ~66+ relapses are recorded** with stage and age, and validate on new patients.

## Limitations
- One clinic, 335 patients, 2020–21; 30 relapses.
- Only patients alive and attending are included (survivor bias), so 22% likely **understates** relapse.
- Facts come from free-text notes via rules, so some will be wrong.
- Reverse causation: treatment lists can change *because* of a relapse.

*These are associations from one clinic, not medical advice.*"""),
    ],
    """**Correlation: yes** (cancer type clearly, treatments weakly). **Causation: not provable with this data.** The most
valuable output of the project is knowing exactly which missing variables would turn this into a causal answer.""")

NOTEBOOKS = {
    "01_access_filter.ipynb": nb1, "02_wrangling.ipynb": nb2, "03_eda.ipynb": nb3, "04_visualization.ipynb": nb4,
    "05_feature_engineering.ipynb": nb5, "06_modeling.ipynb": nb6, "07_evaluation.ipynb": nb7,
    "08_recommendation.ipynb": nb8,
}


def main():
    NB.mkdir(exist_ok=True)
    run = "--no-run" not in sys.argv
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                        "language": "python"}})
        path = NB / name
        if run:
            from nbconvert.preprocessors import ExecutePreprocessor
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB)}})
        nbf.write(nb, path)
        print(("executed " if run else "built ") + str(path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
