"""Step 2 - Turn the de-identified clinical notes into structured factors.

Rule-based clinical NLP over `clinical_note` + `diagnosis_text`. Every rule is a
plain regex so a clinician can audit it. Output: data/processed/patients_features.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# ------------------------------------------------------------------ cancer site
SITE_RULES = [  # (group, detail, pattern) - first match on diagnosis text, then note
    ("Haematologic", "Myeloma", r"myeloma(?!\s*excluded)"),
    ("Haematologic", "CLL", r"\bcll\b"),
    ("Gynaecologic", "Ovary", r"ovar"),
    ("Gynaecologic", "Cervix", r"cervix|\bca cx\b|\bcx\b"),
    ("Gynaecologic", "Uterus", r"uter|mullerian"),
    ("Gynaecologic", "Vulva", r"vulva"),
    ("Gynaecologic", "Choriocarcinoma", r"chorio"),
    ("Breast", "Breast", r"breast|phyll"),
    ("Colorectal / anal", "Colorectal", r"rect|colon|sigmoid|anal\b|ano\s*rectal"),
    ("Upper GI / hepatobiliary", "Pancreas", r"pancr"),
    ("Upper GI / hepatobiliary", "Liver (HCC)", r"\bhcc\b"),
    ("Upper GI / hepatobiliary", "Biliary", r"cholang|cholag|gall"),
    ("Upper GI / hepatobiliary", "Stomach", r"stomach|gastric"),
    ("Upper GI / hepatobiliary", "Oesophagus", r"oesoph|esoph"),
    ("Upper GI / hepatobiliary", "Small bowel", r"duoden|neuroendocrine"),
    ("Lung", "Lung", r"\blung\b"),
    ("Genitourinary", "Prostate", r"prostat"),
    ("Genitourinary", "Kidney", r"\brcc\b|renal"),
    ("Genitourinary", "Testis", r"seminoma"),
    ("CNS", "Meningioma", r"men[i]?n?gioma"),
    ("CNS", "Glioma", r"\bgbm\b|glio"),
    ("CNS", "Other CNS", r"brain|medullo|pituitar|acoustic|xanthoastro"),
    ("Head & neck", "Nasopharynx", r"naso|\bnpc\b|\bnph\b"),
    ("Head & neck", "Larynx", r"laryn"),
    ("Head & neck", "Thyroid", r"thyroid"),
    ("Head & neck", "Salivary", r"parotid"),
    ("Head & neck", "Oral / jaw", r"mandib|maxill|buccal|oral|nasal"),
    ("Skin", "Melanoma", r"melanoma"),
    ("Skin", "Non-melanoma skin", r"basal cell|scc scalp|skin squamous"),
    ("Sarcoma", "Sarcoma", r"sarcom|ewing|schwan|shwan"),
    ("Unknown primary", "Unknown primary", r"\bmuo\b|unknown|un known"),
]
NON_MALIGNANT = r"men[i]?n?gioma|pituitar|acoustic|schwan|shwan"


def classify_site(diag: str, note: str):
    for text in (diag.lower(), note.lower()):
        for group, detail, pat in SITE_RULES:
            if re.search(pat, text):
                return group, detail
    return "Unknown primary", "Unknown primary"


# ------------------------------------------------------------------ helpers
YEAR_RE = re.compile(r"(?<!\d)(19[89]\d|20[0-2]\d)(?!\d)")
NEG = r"(?:no|not|without|free of|clear of|negative for)\s+(?:\w+\s+){0,3}"


def has(pat, text):
    return int(bool(re.search(pat, text, flags=re.I)))


def has_not_negated(pat, text):
    for m in re.finditer(pat, text, flags=re.I):
        window = text[max(0, m.start() - 35):m.start()].lower()
        if not re.search(r"\b(no|not|without|free of|clear of|negative)\b[^.;|]*$", window):
            return 1
    return 0


def diagnosis_year(note):
    m = re.search(r"diagnos\w*[^|.]{0,40}?(19[89]\d|20[0-2]\d)", note, flags=re.I) or \
        re.search(r"(?:operated|mastectomy|presented)[^|.]{0,25}?(19[89]\d|20[0-2]\d)", note, flags=re.I)
    if m:
        return int(m.group(1))
    years = [int(y) for y in YEAR_RE.findall(note)]
    return min(years) if years else np.nan


def years_ago(note):
    m = re.search(r"(\d{1,2})\s*(?:years?|yrs?)\s*ago|since\s*(\d{1,2})\s*(?:years?|yrs?)", note, flags=re.I)
    return int(m.group(1) or m.group(2)) if m else np.nan


DISTANT = r"(bone|bony|liver|live|lung|brain|peri\s*-?umbilical|adrenal)"


def metastatic_at_diagnosis(diag, note):
    """Distant spread already present when first diagnosed (de-novo stage IV)."""
    if re.search(r"mets|metasta", diag, flags=re.I):
        return 1
    t = note.lower()
    pats = [
        r"diagnos\w*[^|.]{0,90}?(with|\+|and)\s*[^|.]{0,40}?" + DISTANT + r"[^|.]{0,15}?\s*(mets|meta|metastas)",
        r"diagnos\w*[^|.]{0,40}?metastatic (?!carcinoma invasive)",
        r"presented with[^|.]{0,60}?(mets|metasta)", r"newly diagnosed with[^|.]{0,20}?met",
        r"^[^|]{0,60}?" + DISTANT + r"[^|.]{0,10}?mets", r"diagnos\w*[^|.]{0,80}?\bstage\s*(4|iv)\b",
        r"diagnosed as metastatic", r"ct chest showed lung mets", r"unknown primary", r"\bmuo\b",
    ]
    if re.search(r"develop|devolop", t):  # spread described as a later event, not at diagnosis
        pats = [p for p in pats if not p.startswith("^")]
    return int(any(re.search(p, t) for p in pats))


def age_years(note):
    m = re.search(r"(\d{2})\s*(?:year|yr)s?\s*old", note, flags=re.I)
    return int(m.group(1)) if m else np.nan


def infer_sex(group, detail, diag, note):
    text = f"{diag} {note}".lower()
    if group == "Gynaecologic":
        return "F"
    if detail in ("Prostate", "Testis"):
        return "M"
    f = len(re.findall(r"\b(she|her|lady|woman|female|lactating|vaginal|gave birth)\b", text))
    m = len(re.findall(r"\b(he|his|him|man|male)\b", text))
    if group == "Breast":
        return "F" if m < 2 or f >= m else "M"
    if f > m:
        return "F"
    if m > f:
        return "M"
    return "Unknown"


# ------------------------------------------------------------------ receptors (breast)
POS = r"(?:\+\s*ve|\+|positive|pos\b|\+ve)"
NEGV = r"(?:-\s*ve|-|_|negative|nagative|neg\b|\(0\)|0/8)"


def receptor(note, name):
    """Return 1 (positive), 0 (negative) or NaN for ER / PR / HER2."""
    t = note.lower()
    if re.search(r"triple\s*neg", t):
        return 0
    if name == "her2":
        if re.search(r"herceptin", t):
            return 1
        m = re.search(r"her\s*2?\s*[:]?\s*(\+\s*[123]|\(0\)|" + POS + "|" + NEGV + ")", t)
        if m:
            v = m.group(1).replace(" ", "")
            if v in ("+3", "+2") or re.fullmatch(POS, v) and v not in ("+1",):
                return 1 if v != "+1" else 0
            return 0
        if re.search(r"er[-\s]*pr[-\s]*her\s*2?\s*positive", t):
            return 1
        return np.nan
    key = "er" if name == "er" else "pr"
    m = re.search(rf"\b{key}\s*[:/]?\s*(\d)/8", t)
    if m:
        return int(int(m.group(1)) >= 3)
    m = re.search(rf"\b{key}\s*[:/]?\s*({POS}|{NEGV})", t)
    if m:
        return 1 if re.fullmatch(POS, m.group(1).replace(" ", "")) else 0
    # 'ER PR +ve', 'ER-PR-HER2 positive', 'ER/PR positive'
    if re.search(r"\ber\s*[/\-]?\s*pr\s*[-]?\s*(her\s*2?\s*)?(\+\s*ve|positive)", t):
        return 1
    if re.search(r"hormone\s*(\+\s*ve|positive)", t):
        return 1
    return np.nan


def breast_subtype(er, pr, her2, note):
    if re.search(r"triple\s*neg", note, flags=re.I):
        return "Triple negative"
    if her2 == 1:
        return "HER2+"
    hr = 1 if (er == 1 or pr == 1) else (0 if (er == 0 and pr == 0) else np.nan)
    if hr == 1:
        return "HR+/HER2-" if her2 == 0 else "HR+ (HER2 unknown)"
    if hr == 0 and her2 == 0:
        return "Triple negative"
    return "Unknown"


def grade(note):
    m = re.search(r"\bgr(?:ade|ad)?\.?\s*(1|2|3|i{1,3}|11|one|two|three)\b|\bG(I{1,3})\b", note, flags=re.I)
    if not m:
        return np.nan
    v = (m.group(1) or m.group(2)).lower()
    return {"1": 1, "i": 1, "one": 1, "2": 2, "ii": 2, "11": 2, "two": 2, "3": 3, "iii": 3, "three": 3}.get(v, np.nan)


# ------------------------------------------------------------------ recurrence / progression
RECUR_PATTERNS = [
    r"\brecurr?ence\b", r"\brelapse", r"\bregrew\b", r"disease\s*(is\s*)?progress", r"\bprogression\b",
    r"\b(develop|devolop)\w*\s+(?:\w+\s+){0,4}(mets|metas|recurrence|jaundice|hard mass)",
    r"\b(has|had|then)\s+(?:\w+\s+){0,2}(bone|lung|liver|brain|live)\s*(\+\s*\w+\s*)?mets\s+(in\s+)?(19|20)\d\d",
    r"came with (brain|bone|lung|liver) mets", r"\bmets\b[^|]{0,15}\b(?:in|on)\s*(19|20)\d\d",
    r"\b2nd line|second line|third line|3rd line|fourth line|2line", r"long history of recurrence",
]


def recurrence_event(note):
    t = note.lower()
    if re.search(r"no evidence of (mets|recurrence|mets/recurrence)", t) and not re.search(r"developed", t):
        return 0
    return int(any(re.search(p, t) for p in RECUR_PATTERNS))


def recurrence_year(note, dx_year):
    t = note.lower()
    years = []
    for p in RECUR_PATTERNS + [r"\bmets\b"]:
        for m in re.finditer(p, t):
            seg = t[m.start(): m.end() + 30]
            years += [int(y) for y in YEAR_RE.findall(seg)]
    years = [y for y in years if not np.isnan(dx_year) and y > dx_year]
    return min(years) if years else np.nan


# ------------------------------------------------------------------ main
def build(patients: pd.DataFrame, appts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in patients.itertuples():
        note = r.clinical_note if isinstance(r.clinical_note, str) else ""
        diag = r.diagnosis_text if isinstance(r.diagnosis_text, str) else ""
        text = f"{diag} | {note}"
        group, detail = classify_site(diag, note)
        dx = diagnosis_year(note)
        if np.isnan(dx) and not np.isnan(years_ago(note)):
            dx = r.registered_year - years_ago(note)
        if not np.isnan(dx) and dx > r.registered_year:
            dx = np.nan  # typo'd year (e.g. '2028')
        er, pr, her2 = (receptor(note, k) for k in ("er", "pr", "her2")) if group == "Breast" else (np.nan,) * 3

        mets = {s: has_not_negated(p, text) for s, p in {
            "bone": r"\bbon[ey]?\w*\s*(\+\s*\w+\s*)?(mets|metas|mates|lesion)|osseous|bony pain|bone[-\s]+(liver|lung)\s*mets"
                    r"|lung[-\s/]+bone|bone\s+and\s+lung|liver\+bone|bone lung liver|osseous lesion|wbbs: bon",
            "liver": r"liver|live mets|hepatic|jaundice",
            "lung": r"\blung\b|pulmonary",
            "brain": r"brain mets|brain met\b|whole brain|ct brain multiple",
            "lymph_node": r"lymph node|\bln recurrence|\blns? (active|involv)|cervical (mass|nodules)|axillary lymph|inguinal",
        }.items()}
        if group in ("CNS",):
            mets["brain"] = 0
        if detail == "Lung" and not re.search(r"lung\s*(\+\s*\w+\s*)?mets", text, re.I):
            mets["lung"] = 0
        if detail == "Liver (HCC)":
            mets["liver"] = 0
        n_mets = sum(mets.values())
        metastatic = int(n_mets > 0 or has(r"metasta|\bmets\b|stage\s*(4|iv)\b|unknown prim|\bmuo\b", text)
                         and not has(r"no evidence of mets", text))

        f = {
            "patient_id": r.patient_id,
            "region": r.region,
            "registered_year": r.registered_year,
            "ses_code": r.ses_code if isinstance(r.ses_code, str) else "",
            "site_group": group,
            "site": detail,
            "non_malignant_tumour": has(NON_MALIGNANT, diag + " " + note[:80]),
            "sex": infer_sex(group, detail, diag, note),
            "age_mentioned": age_years(note),
            "elderly": has(r"elderly|\b(7\d|8\d)\s*year", note),
            "dx_year": dx,
            "years_since_dx": r.registered_year - dx if not np.isnan(dx) else np.nan,
            "note_length": len(note),
            "note_missing": int(len(note) < 5),
            # tumour biology
            "histology": ("Ductal" if has(r"duct|\bidc\b|\bi dc\b", note) else
                          "Lobular" if has(r"lobular", note) else
                          "Squamous" if has(r"squamous|\bscc\b", text) else
                          "Adenocarcinoma" if has(r"adeno", text) else
                          "Sarcoma" if group == "Sarcoma" else "Not stated"),
            "grade": grade(note),
            "er": er, "pr": pr, "her2": her2,
            # spread
            **{f"mets_{k}": v for k, v in mets.items()},
            "n_met_sites": n_mets,
            "metastatic": metastatic,
            "metastatic_at_dx": metastatic_at_diagnosis(diag, note),
            "locally_advanced": has(r"locally advanced|locally extensive|\b20\s*(x\s*20\s*)?cm|burst|advanced local", text),
            "second_primary": has(r"breast and ovary|\+ ca breast|ca uterus in|history of ca breast", text),
            # treatment
            "surgery": has(r"operat|mastectom|\bmrm\b|\bwle\b|lumpectomy|resect|hemicolectomy|nephrectomy|tah|orchidectomy"
                           r"|excision|surgery\b(?! unit)|removal", note),
            "chemotherapy": has(r"chemo|\bfec\b|\bfac\b|\bac\s*\d|\bac\b|\bec\b|\btec\b|\btc\b|taxo|taxol|folfox|fol fox|"
                                r"fullfox|xeloda|gemzar|gemcit|cisplat|carbo|navelbine|vinorelbine|bep|temozol|ifos|"
                                r"doxo|irren|thalidomide|tpf|cef\b|cmf|fmc", note),
            "neoadjuvant_chemo": has(r"neo\s*-?\s*ad|\bnact\b|new adjuvant", note),
            "adjuvant_chemo": has(r"(?<!neo)(?<!neo-)(?<!neo )adj(u|e|o)v", note),
            "radiotherapy": has(r"radioth|radiation|\brt\b|\bxrt\b|\d+\s*gy|\d+\s*f\b|fractions|phase\s*(1|i|2|ii)", note),
            "hormonal_therapy": has(r"hormon|tamox|tomox|letroz|latroz|letraz|femara|arimidex|armidex|zoladex|zoldex|"
                                    r"casodex|\bht\b", note),
            "anti_her2": has(r"herceptin|targeted", note),
            "bisphosphonate": has(r"zomet|zomenta", note),
            "opioid": has(r"morphine|tramad|opioid", note),
            "palliative_referral": has(r"\bpcu\b|palliat|\bpc\b|supportive|end of life|symptom(atic)? manag", note),
            "lines_of_chemo_2plus": has(r"2nd line|second line|third line|3rd line|fourth line|2line|2 line", note),
            # course / access
            "recurrence_or_progression": recurrence_event(note),
            "disease_free_statement": has(r"no evidence of (mets|recurrence)|clear of disease|free of disease|"
                                          r"doing well|completed (treatment|ttt)|on follow\s*-?up|annual", note),
            "treated_abroad": has(r"cairo|egypt|india|jordan", note),
            "missed_followup_or_delay": has(r"default|lost (her |the )?(follow|file)|file was lost|misdirected|"
                                            r"not referred|refused|for \d+ years|last \d+ month|not operated", note),
            "comorbidity": has(r"diabet|heart|hiv|hbsag|kidney|albumin low", note),
        }
        f["recurrence_year"] = recurrence_year(note, dx) if f["recurrence_or_progression"] else np.nan
        rows.append(f)

    df = pd.DataFrame(rows)
    for c in ("er", "pr", "her2"):
        df[c] = df[c].astype("float")
    df["breast_subtype"] = [
        breast_subtype(r.er, r.pr, r.her2, n) if r.site_group == "Breast" else ""
        for r, n in zip(df.itertuples(), patients.clinical_note.fillna(""))
    ]

    # Outcome A - relapse: "after someone has had cancer, does it come back?"
    # The at-risk cohort is defined by the SAME rule for patients with and without relapse
    # (an earlier version required 'curative' treatment only from non-relapsers, which made
    # every treatment look protective):
    #   malignant, not metastatic at diagnosis, received any cancer treatment,
    #   diagnosis year known and >= 1 year before the note.
    notes = patients.clinical_note.fillna("")
    treated = (df.surgery == 1) | (df.chemotherapy == 1) | (df.radiotherapy == 1) | \
              (df.hormonal_therapy == 1) | pd.Series([has(r"complete[d]? (treatment|ttt)", n) for n in notes],
                                                     index=df.index).astype(bool)
    base = (df.non_malignant_tumour == 0) & (df.metastatic_at_dx == 0)
    event = df.recurrence_or_progression == 1
    eligible = base & treated & (df.years_since_dx >= 1)
    df["eligible_for_relapse"] = eligible.astype(int)
    df["relapse"] = np.where(eligible, event.astype(int), np.nan)
    # time-to-event: years from diagnosis to relapse (documented year, else the note year as an
    # upper bound) or to the note year (censored) for patients without relapse
    t_event = df.recurrence_year.fillna(df.registered_year) - df.dx_year
    t_cens = df.registered_year - df.dx_year
    df["rfs_years"] = np.where(eligible, np.where(event, t_event, t_cens), np.nan)
    df["rfs_years"] = df.rfs_years.clip(lower=0.5)

    # Outcome B - progression of disease that was already metastatic at diagnosis
    df["progression_if_metastatic"] = np.where(df.metastatic_at_dx == 1, event.astype(int), np.nan)

    # ---- appointment-derived features
    a = appts.copy()
    g = a.groupby("patient_id")
    feat = pd.DataFrame({
        "n_visits": g.size(),
        "followup_span_days": g.days_from_first_visit.max(),
        "n_units_seen": g.unit_code.nunique(),
        "first_unit": g.unit_code.first(),
        "last_unit": g.unit_code.last(),
    })
    for u in ("CT", "HT", "OC", "PCF"):
        feat[f"visits_{u}"] = g.unit_code.apply(lambda s, u=u: int((s == u).sum()))
    # moved from survivorship/hormonal clinic into chemo or palliative later on
    feat["escalated_from_HT"] = g.unit_code.apply(
        lambda s: int("HT" in list(s) and any(x in ("CT", "PCF") for x in list(s)[list(s).index("HT") + 1:])))
    df = df.merge(feat, left_on="patient_id", right_index=True, how="left")
    df[["n_visits", "n_units_seen"]] = df[["n_visits", "n_units_seen"]].fillna(0).astype(int)
    return df


def main():
    patients = pd.read_csv(PROC / "patients_clean.csv")
    appts = pd.read_csv(PROC / "appointments_clean.csv")
    df = build(patients, appts)
    df.to_csv(PROC / "patients_features.csv", index=False)
    print(f"wrote {PROC/'patients_features.csv'}  shape={df.shape}")


if __name__ == "__main__":
    main()
