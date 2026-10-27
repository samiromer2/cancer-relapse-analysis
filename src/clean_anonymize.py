"""Step 1 - Clean and de-identify the raw `bookingtreat` dump.

Removes / generalises every HIPAA Safe-Harbor style identifier:
  * names, phone numbers, hospital file numbers, internal row ids   -> dropped
  * residence (town / neighbourhood)                                -> broad region
  * full dates (dd/mm/yyyy, "Mon yyyy", glued dmyyyy ...)                -> year only
  * clinician names in free text ("Dr ...", "Prof ...")             -> [CLINICIAN]
  * sub-state towns mentioned in free text                          -> [TOWN]
  * `users` table (admin password hash), `account`, `extrasummary`  -> not exported

Outputs (data/processed/):
  patients_clean.csv      one row per real patient, pseudonymous id
  appointments_clean.csv  visits keyed by the same pseudonymous id, relative days only
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from parse_sql import load_dump

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "bookingtreat.sql"
OUT = ROOT / "data" / "processed"
SEED = 20260103

# ---------------------------------------------------------------- residence -> region
# Ordered: first match wins. Arabic + English spellings seen in the data.
REGION_RULES = [
    ("South Sudan / abroad", ["جوبا", "جنون السودان", "اويل", "اعالي النيل", "اريتريا"]),
    ("Darfur", ["darfur", "دارفور", "دار فور", "نيالا", "الفاشر", "الجنينة", "الضعين", "مليط", "كلبس", "كاس", "fasher", "nyala"]),
    ("Kordofan", ["kordofan", "كردفان", "كردوفان", "الابيض", "obied", "النهود", "الدلنج", "كادوقلي", "ابوجبيهة",
                  "ام روابة", "المجلد", "الرهيد", "rahad", "bara", "القوز"]),
    ("Eastern", ["kassala", "كسلا", "القضارف", "القضاريف", "بور", "حلفا الجديدة", "halfa"]),
    ("Northern / River Nile", ["north state", "nile state", "nile river", "الشمالية", "الشماليه", "نهر النيل", "النيلشندي",
                               "شندي", "عطبرة", "عطبره", "الدامر", "الزاكياب", "ابو حمد", "وادي حلفا", "كبوشية",
                               "مروي", "دنقلا", "abuhamad", "atbara", "عطب"]),
    ("Central (Gezira / Sennar / White & Blue Nile)", [
        "gezira", "الجزيرة", "مدني", "sennar", "sannar", "سنار", "سنجة", "white nile", "النيل الابيض", "النيل الاابيض",
        "ولاية النيل الابيض", "كوستي", "كوستى", "الدويم", "ربك", "كنانه", "تندلتى", "القطينة", "العلقه", "blue nile",
        "الدمازين", "المسيد", "المناقل", "الكاملين", "اللعوتة", "تنبول", "الحلة الجديدة", "العشره", "dawim", "duwaym"]),
    ("Khartoum", ["khartoum", "khartum", "omdurman", "الخرطوم", "ام درمان", "امدرمان", "بحري", "شرق النيل", "جبل",
                  "الثورة", "امبدة", "ام بده", "الباقير", "المربعات", "السامراب", "صالحة", "الشجره", "كافوري"]),
]


def residence_to_region(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return "Unknown"
    for region, keys in REGION_RULES:
        if any(k in t for k in keys):
            return region
    return "Unknown"


# ---------------------------------------------------------------- free-text scrubbing
MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
YEAR = r"(?:19[89]\d|20[0-2]\d)"
TOWNS = ["Nuhud", "Dongola", "gadarif", "Gadarif", "Kassala", "madani", "Madani"]


def _year_from_compact(m: re.Match) -> str:
    """'1052016' / '2232018' / '15072019' (made-up examples) -> trailing 4-digit year."""
    return m.group(0)[-4:]


def scrub_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r", " ").replace("\n", " ")
    # clinicians
    t = re.sub(r"\b(?:Dr|Prof|Professor)\.?\s+[A-Za-z]+(?:\s+[A-Z][a-z]+){0,3}", "[CLINICIAN]", t)
    # protect Allred receptor scores ('ER 8/8', 'PR 0/8') from the date rules
    t = re.sub(r"\b(ER|PR)\s*(\d)\s*/\s*8\b", r"\1 \2of8", t, flags=re.I)
    # dd/mm/yyyy, dd-mm-yyyy, dd/ mm/yyyy, d/m//yyyy, dd/mm/0yyyy ... (may be glued to words)
    t = re.sub(r"(?<!\d)\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]+\s*0?(\d{4})(?!\d)", r" \1 ", t)
    t = re.sub(r"(?<!\d)\d{1,2}\s*[/\-]\s*" + MONTHS + r"\.?\s*(" + YEAR + r")", r" \1 ", t, flags=re.I)
    # 'Jan 2018', 'march 2017', '5 may 2016'
    t = re.sub(r"\b(?:\d{1,2}\s+)?" + MONTHS + r"\.?\s*,?\s*(" + YEAR + r")(?!\d)", r"\1", t, flags=re.I)
    # month/year: 'mm/yyyy'; month name + typo'd 5-digit year
    t = re.sub(r"(?<!\d)\d{1,2}\s*/\s*(" + YEAR + r")(?!\d)", r" \1", t)
    t = re.sub(r"\b" + MONTHS + r"\s+(\d{5})\b", r"\1", t, flags=re.I)
    # compact ddmmyyyy / dmyyyy glued numbers (e.g. 'on 1052016', 'on22032018')
    t = re.sub(r"(?<!\d)\d{1,4}(?:19[89]\d|20[0-2]\d)(?!\d)", _year_from_compact, t)
    t = re.sub(r"(?<!\d)\d{1,2}\s*/\s*(" + YEAR + r")(?!\d)", r" \1", t)
    # leftover partial dates / typo'd years: 'dd/mm/', 'dd/m yyy', 'dd/m', 'dd/m/yyy'
    t = re.sub(r"(?<!\d)\d{1,2}\s*/\s*\d{1,2}(?:\s*/+\s*\d{0,3})?(?:\s+\d{3})?(?!\d)", " [DATE] ", t)
    t = re.sub(r"\[DATE\]\s*/\s*\d{4,5}", "[DATE]", t)
    t = re.sub(r"\b(ER|PR) (\d)of8\b", r"\1 \2/8", t)
    for town in TOWNS:
        t = re.sub(rf"\b{town}\b", "[TOWN]", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------- identity resolution
def resolve_patients(p: pd.DataFrame) -> pd.DataFrame:
    """Group raw rows that belong to the same human.

    Same person if they share a hospital FileNo, or share a booking Ptno *and* the
    same name. Ptno alone is not trusted (Ptno 91 was reused for two people).
    """
    p = p.copy()
    p["_name"] = p.PtName.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    parent = list(range(len(p)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    idx = list(p.index)
    pos = {ix: k for k, ix in enumerate(idx)}
    for _, g in p[p.FileNo > 0].groupby("FileNo"):
        for a, b in zip(g.index[:-1], g.index[1:]):
            union(pos[a], pos[b])
    for _, g in p.groupby(["Ptno", "_name"]):
        for a, b in zip(g.index[:-1], g.index[1:]):
            union(pos[a], pos[b])
    # blank-name rows (e.g. PID 28) inherit their Ptno group if that Ptno is unambiguous
    for ptno, g in p.groupby("Ptno"):
        named = g[g["_name"] != ""]
        if named["_name"].nunique() == 1:
            for a in g.index:
                union(pos[a], pos[named.index[0]])
    p["_group"] = [find(pos[i]) for i in idx]
    return p


def merge_group(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("whenwasit")
    diag = g.Diagnosis.fillna("").str.strip()
    diag = diag[diag != ""]
    notes = [s for s in g.summary.fillna("").str.strip() if s]
    notes = list(dict.fromkeys(notes))  # de-dup, keep order
    ses = g.ses.fillna("").str.strip()
    ses = ses[ses != ""]
    res = g.Residence.fillna("").str.strip()
    res = res[res != ""]
    return pd.Series({
        "diagnosis_raw": max(diag, key=len) if len(diag) else "",
        "note_raw": " | ".join(notes),
        "ses_code": ses.iloc[-1] if len(ses) else "",
        "residence_raw": max(res, key=len) if len(res) else "",
        "registered_year": pd.to_datetime(g.whenwasit).dt.year.min(),
        "ptnos": sorted(set(g.Ptno)),
        "n_raw_rows": len(g),
    })


def main():
    tables = load_dump(RAW)
    p, a = tables["patients"], tables["appointments"]

    # obvious test record (non-clinical diagnosis "he is always happy", age 21, name 'sam')
    test_rows = p.Diagnosis.str.contains("مبسوط", na=False) | (p.PtName.str.strip().str.lower() == "sam")
    p = p[~test_rows]

    p = resolve_patients(p)
    pts = p.groupby("_group").apply(merge_group, include_groups=False).reset_index(drop=True)
    pts = pts[(pts.diagnosis_raw != "") | (pts.note_raw != "")].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    pts["patient_id"] = [f"P{n:04d}" for n in rng.permutation(len(pts)) + 1]

    pts["region"] = pts.residence_raw.map(residence_to_region)
    pts["diagnosis_text"] = pts.diagnosis_raw.str.replace("’", "", regex=False).str.strip()
    pts["clinical_note"] = pts.note_raw.map(scrub_text)

    # ---- appointments: map Ptno -> patient only when unambiguous
    ptno_map = {}
    for pid, ptnos in zip(pts.patient_id, pts.ptnos):
        for n in ptnos:
            ptno_map.setdefault(n, set()).add(pid)
    ptno_map = {k: next(iter(v)) for k, v in ptno_map.items() if len(v) == 1}

    a = a.copy()
    a["patient_id"] = a.PID.map(ptno_map)
    dropped = a.patient_id.isna().sum()
    a = a.dropna(subset=["patient_id"])
    a["ADate"] = pd.to_datetime(a.ADate)
    a = a.sort_values(["patient_id", "ADate", "SlotID"])
    a["visit_year"] = a.ADate.dt.year
    a["days_from_first_visit"] = (a.ADate - a.groupby("patient_id").ADate.transform("min")).dt.days
    a["visit_no"] = a.groupby("patient_id").cumcount() + 1
    unit_names = tables["unitdrs"].set_index("UnitDrID").Name.to_dict()
    a["unit_code"] = a.UnitDrID
    a["unit"] = a.UnitDrID.map(unit_names)
    appts = a[["patient_id", "visit_no", "visit_year", "days_from_first_visit", "unit_code", "unit"]]

    out_p = pts[["patient_id", "region", "registered_year", "ses_code", "diagnosis_text", "clinical_note"]]
    out_p = out_p.sort_values("patient_id").reset_index(drop=True)

    audit(out_p)
    OUT.mkdir(parents=True, exist_ok=True)
    out_p.to_csv(OUT / "patients_clean.csv", index=False)
    appts.to_csv(OUT / "appointments_clean.csv", index=False)
    print(f"raw patient rows: {len(tables['patients'])}  -> unique patients: {len(out_p)}")
    print(f"appointments kept: {len(appts)}  (dropped unlinkable/ambiguous: {dropped})")
    print(f"wrote {OUT/'patients_clean.csv'} and {OUT/'appointments_clean.csv'}")


def audit(df: pd.DataFrame):
    """Fail loudly if anything identifier-shaped survived."""
    text = " ".join(df.clinical_note) + " " + " ".join(df.diagnosis_text)
    problems = {
        "phone-like numbers": re.findall(r"\b\d{9,}\b", text),
        "full dates": re.findall(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b", text),
        "clinician names": re.findall(r"\bDr\.?\s+(?!\[)[A-Z]\w+", text),
        "arabic text": re.findall(r"[؀-ۿ]{3,}", text),
    }
    bad = {k: v[:5] for k, v in problems.items() if v}
    if bad:
        raise SystemExit(f"De-identification audit FAILED: {bad}")
    print("de-identification audit: passed")


if __name__ == "__main__":
    main()
