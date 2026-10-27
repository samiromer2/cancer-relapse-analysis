# Project Plan — Cancer Relapse: from clinic data to a decision

Framework (from `related files/End_to_End_Data_Analysis_Practice_Activity.pdf`):
**ACCESS → WRANGLE → EXPLORE → VISUALIZE → ENGINEER → MODEL → EVALUATE → RECOMMEND**

## Business framing (goal → KPI → metric)

Clinic goal: **find patients whose cancer is likely to come back, so limited follow-up
slots and scans go to them first.**

| Clinic goal | KPI | Supporting metrics | Data we have |
|---|---|---|---|
| Reduce / catch relapse early | **Relapse rate** (curative cohort) — now 36% | relapse-free survival at 2/5 yrs (81% / 70%) | `relapse`, `dx_year`, `recurrence_year` |
| Target follow-up to high-risk patients | **Recall of the relapse model** (share of relapses flagged) | precision, F1, confusion matrix | features + `relapse` |
| Earlier diagnosis | **% metastatic at diagnosis** — now 19% | by region, by site | `metastatic_at_dx`, `region` |
| Better records | **% breast patients with receptor status documented** — now 32% | % with grade, % with dx year | `er/pr/her2`, `grade` |
| Palliative capacity | **% referred to palliative care** — now 17% | opioid use, visits to PCF unit | `palliative_referral`, appointments |

KPI = outcome the clinic wants to move. Metric = what helps explain or move it.

## Where we are vs the 8 stages

| Stage | Status | Where |
|---|---|---|
| 1. Access & Filter | ✅ done (needs a teaching notebook) | `src/parse_sql.py` |
| 2. Wrangling | ✅ done — dedupe, de-identification, audit | `src/clean_anonymize.py` |
| 3. EDA | ✅ done | `src/analysis.py` |
| 4. Visualization | ✅ 10 charts | `reports/figures/` |
| 5. Feature Engineering | ✅ 60 features from notes | `src/extract_features.py` |
| 6. Modeling | ✅ logistic + random forest, repeated CV | `src/model.py` |
| 7. Evaluation | ✅ accuracy/precision/recall/F1/AUC, confusion matrix, operating points | `reports/` |
| 8. Recommendation | ⚠️ draft in README §8 | — |

## Phases

### Phase 0 — Put it on GitHub ✅ done — https://github.com/samiromer2/cancer-relapse-analysis
- `git init`, confirm `.gitignore` blocks `bookingtreat.sql`, `.venv`, `.DS_Store`.
- **Never commit:** the raw `.sql`, and `data/processed/patients_clean.csv` (free-text notes are
  de-identified but still rich). Commit `patients_features.csv` + synthetic data instead.
- Add a pre-commit check that rejects 9-digit numbers / Arabic names in staged CSVs.
- Create the repo **private first**; make public only after a final privacy review.
- Add `LICENSE`, `DATA_STATEMENT.md` (source, consent/ethics note, what was removed).

### Phase 1 — Notebooks following the 8 stages (1–2 days)
One notebook per stage in `notebooks/`, each answering the PDF's questions *for this data*,
importing logic from `src/` (no copy-pasted code):
`01_access_filter` · `02_wrangling` · `03_eda` · `04_visualization` · `05_features` ·
`06_modeling` · `07_evaluation` · `08_recommendation`.
Each opens with **the business question** and ends with **"so what?"** for the clinic.

### Phase 2 — Modeling: predict relapse ✅ done
- Target `y` = `relapse` (curative cohort, n = 87, 31 events). Classification problem.
- Features `X`: site, receptor subtype, grade, surgery, RT, hormonal therapy, neoadjuvant,
  years since dx, region, treated abroad, missed follow-up.
- **Leakage check:** drop anything that happens *because of* relapse — `recurrence_year`,
  `lines_of_chemo_2plus`, `palliative_referral`, `opioid`, `escalated_from_HT`, later mets.
- Models: logistic regression (baseline, interpretable), penalised logistic, random forest.
- n is small → **stratified 5-fold cross-validation repeated 20×**, not a single train/test split.
- Second experiment: **train on synthetic, test on real** (does the synthetic data carry the signal?).

### Phase 3 — Evaluation ✅ done
- Accuracy, precision, recall, F1, confusion matrix, ROC-AUC, calibration plot.
- Clinic translation table (same idea as the PDF's marketing table):

| Metric | Clinic meaning |
|---|---|
| Recall | Of patients who will relapse, how many do we flag for early follow-up? |
| Precision | Of patients we flag, how many really relapse (wasted scans otherwise)? |
| False negative | A relapse we miss → later, harder-to-treat disease |
| False positive | An extra scan / visit for a patient who was fine |

- Threshold chosen for **high recall** (missing a relapse costs more than an extra visit);
  report the precision we pay for it.

### Phase 4 — AI-generated synthetic data (2–3 days, needs an Anthropic API key)
1. **LLM extraction check:** Claude reads each de-identified note into a strict JSON schema;
   compare with the regex features (agreement %), list disagreements for review.
2. **Synthetic notes:** for each synthetic patient row, Claude writes a realistic clinical
   summary consistent with the row → fully synthetic tabular + text dataset.
3. Privacy check on generated text (no near-copies of real notes; n-gram overlap test).
4. Cost estimate before running (5,000 notes ≈ a few dollars with a small model).

### Phase 5 — Recommendation & presentation (1 day)
- One-page recommendation: finding → evidence → action → limitations.
  Draft actions: (1) record stage, age, receptor status as structured fields;
  (2) risk-flag non-breast and no-RT/no-HT curative patients for closer follow-up;
  (3) re-run the analysis when stage/age data exist.
- Interactive report/dashboard (shareable link) + polished README for the GitHub repo.

## Open questions (need your answer)
1. ~~Public or private~~ → public. ~~Course or portfolio~~ → both.
2. Do you have an **Anthropic API key** for Phase 4?
3. Can anyone at the clinic tell us what `ses` means, or provide stage / age / date of death?
