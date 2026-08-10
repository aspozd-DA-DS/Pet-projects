import pandas as pd

from src.segmentation.utils import find_raw_file
# -----------------------------
# XLSX
# -----------------------------
def segment_xlsx_universal(doc_id: str):
    """
    Универсальный сегментатор Excel:
    - использует sheets из шага 2
    - восстанавливает DataFrame из df.to_dict(orient="split")
    """
    from src.segmentation.rule_based import intermediate_cache
    rec = intermediate_cache.get(doc_id, {})
    sheets = rec.get("sheets", [])
    blocks = []

    for sh in sheets:
        df_dict = sh.get("df")
        if not df_dict:
            continue

        # восстановление DataFrame
        df = pd.DataFrame(df_dict["data"], columns=df_dict["columns"])

        # первая строка — заголовки
        if df.shape[0] == 0:
            continue

        columns = list(df.columns)
        data = df.values.tolist()

        blocks.append({
            "type": "table",
            "sheet": sh.get("sheet_name"),
            "columns": columns,
            "data": data,
            "start_page": 1,
        })

    return blocks