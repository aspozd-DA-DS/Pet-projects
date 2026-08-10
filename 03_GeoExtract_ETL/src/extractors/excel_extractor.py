import pandas as pd
import xlrd
from pathlib import Path
from .base import wrap_extractor, base_result

@wrap_extractor
def extract_excel(path: Path):
    res = base_result()   # ← ИСПРАВЛЕНО
    res["source"] = "native_excel"
    res["sheets"] = []

    try:
        ext = path.suffix.lower()

        if ext == ".xls":
            book = xlrd.open_workbook(str(path))

            for sheet_idx in range(book.nsheets):
                sheet = book.sheet_by_index(sheet_idx)
                rows = []

                for r in range(sheet.nrows):
                    row_vals = []
                    for c in range(sheet.ncols):
                        val = sheet.cell_value(r, c)
                        row_vals.append("" if val in ("", None) else val)
                    rows.append(row_vals)

                df = pd.DataFrame(rows)

                res["sheets"].append({
                    "sheet_name": sheet.name,
                    "df": df.to_dict(orient="split"),
                    "n_rows": df.shape[0],
                    "n_cols": df.shape[1],
                })

            res["text"] = ""
            res["tables"] = []
            return res

        else:
            sheets = pd.read_excel(path, sheet_name=None)

            for sheet_name, df in sheets.items():
                res["sheets"].append({
                    "sheet_name": sheet_name,
                    "df": df.to_dict(orient="split"),
                    "n_rows": df.shape[0],
                    "n_cols": df.shape[1],
                })

            res["text"] = ""
            res["tables"] = []
            return res

    except Exception as e:
        print(f"[file: {path.name}] Excel error — {e}")   # ← ИСПРАВЛЕНО
        res["source"] = "error"
        res["warnings"].append(str(e))
        return res
