"""Parse the phpMyAdmin MySQL dump into pandas DataFrames (no MySQL server needed)."""
import re
from pathlib import Path

import pandas as pd

INSERT_RE = re.compile(r"INSERT INTO `(\w+)` \(([^)]*)\) VALUES\s*(.*?);\s*$", re.S | re.M)


def _parse_values(blob: str):
    """Tokenise a MySQL VALUES list into rows, honouring quotes and backslash escapes."""
    rows, row, i, n = [], None, 0, len(blob)
    escapes = {"n": "\n", "r": "\r", "t": "\t", "0": "\0", "'": "'", '"': '"', "\\": "\\"}
    while i < n:
        c = blob[i]
        if c == "(" and row is None:
            row, i = [], i + 1
        elif c == ")" and row is not None:
            rows.append(row)
            row, i = None, i + 1
        elif row is None or c in ", \r\n\t":
            i += 1
        elif c == "'":
            i += 1
            buf = []
            while i < n:
                ch = blob[i]
                if ch == "\\" and i + 1 < n:
                    buf.append(escapes.get(blob[i + 1], blob[i + 1]))
                    i += 2
                elif ch == "'" and i + 1 < n and blob[i + 1] == "'":
                    buf.append("'")
                    i += 2
                elif ch == "'":
                    i += 1
                    break
                else:
                    buf.append(ch)
                    i += 1
            row.append("".join(buf))
        else:
            j = i
            while j < n and blob[j] not in ",)":
                j += 1
            tok = blob[i:j].strip()
            if tok.upper() == "NULL":
                row.append(None)
            else:
                try:
                    row.append(int(tok))
                except ValueError:
                    row.append(float(tok))
            i = j
    return rows


def load_dump(path) -> dict[str, pd.DataFrame]:
    text = Path(path).read_text(encoding="utf-8")
    tables: dict[str, list] = {}
    cols: dict[str, list] = {}
    for m in INSERT_RE.finditer(text):
        name = m.group(1)
        cols[name] = [c.strip(" `") for c in m.group(2).split(",")]
        tables.setdefault(name, []).extend(_parse_values(m.group(3)))
    return {k: pd.DataFrame(v, columns=cols[k]) for k, v in tables.items()}


if __name__ == "__main__":
    import sys
    for k, df in load_dump(sys.argv[1]).items():
        print(k, df.shape)
