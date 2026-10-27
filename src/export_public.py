"""Step 5 - Build the PUBLIC release of the feature table (k-anonymous).

`patients_features.csv` stays private: exact cancer site + region + diagnosis year make 137 of
335 patients unique. The public file:
  * drops: exact site, diagnosis / registration / recurrence year, age, free-text-derived ids
  * bands: diagnosis year -> period, years since diagnosis -> capped at 10
  * enforces k-anonymity (k >= 3) on (site_group, region, sex, dx_period) by suppressing
    region, then sex, in cells that are too small

Output: data/public/patients_features_public.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "patients_features.csv"
OUT = ROOT / "data" / "public" / "patients_features_public.csv"
K = 3
QI = ["site_group", "region", "sex", "dx_period"]
DROP = ["site", "dx_year", "registered_year", "recurrence_year", "age_mentioned", "first_unit", "last_unit",
        "followup_span_days"]


def dx_period(y):
    if np.isnan(y):
        return "unknown"
    return "<=2014" if y <= 2014 else "2015-2017" if y <= 2017 else "2018-2019" if y <= 2019 else "2020-2021"


def small_cells(d):
    size = d.groupby(QI, dropna=False)[QI[0]].transform("size")
    return size < K


def main():
    d = pd.read_csv(SRC)
    d["dx_period"] = d.dx_year.map(dx_period)
    d["years_since_dx"] = d.years_since_dx.clip(upper=10)
    d = d.drop(columns=DROP)

    for col, filler in (("region", "Suppressed"), ("sex", "Unknown"), ("dx_period", "unknown"),
                        ("site_group", "Other / rare")):
        d.loc[small_cells(d), col] = filler
    # last resort: drop any record still unique
    remaining = small_cells(d)
    d = d[~remaining]

    # shuffle row order so it carries no information about the private file
    d = d.sample(frac=1, random_state=11).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    min_k = d.groupby(QI, dropna=False).size().min()
    print(f"public file: {len(d)} rows (dropped {int(remaining.sum())}), min group size k={min_k}, "
          f"region suppressed for {(d.region == 'Suppressed').sum()} rows -> {OUT}")


if __name__ == "__main__":
    main()
