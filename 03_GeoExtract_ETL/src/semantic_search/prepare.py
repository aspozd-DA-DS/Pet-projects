def prepare_chunk_text(block):
    t = block.get("type")

    if t == "table" and "table" in block:
        tbl = block["table"]
        parts = []

        caption = tbl.get("caption")
        if caption:
            parts.append(str(caption))

        columns = tbl.get("columns")
        if isinstance(columns, list):
            parts.append(" | ".join(map(str, columns)))

        rows = tbl.get("rows")
        if isinstance(rows, list):
            for row in rows:
                row_str = ["" if v is None else str(v) for v in row]
                parts.append(" | ".join(row_str))

        data = tbl.get("data")
        if isinstance(data, list):
            for row in data:
                row_str = ["" if v is None else str(v) for v in row]
                parts.append(" | ".join(row_str))

        hd = tbl.get("header_descriptions")
        if isinstance(hd, dict):
            desc_lines = []
            for col in columns or []:
                if col in hd:
                    desc_lines.append(f"{col}: {hd[col]}")
            if desc_lines:
                parts.append("\n".join(desc_lines))

        return "\n".join(parts)

    if t == "table":
        parts = []

        caption = block.get("caption")
        if caption:
            parts.append(str(caption))

        columns = block.get("columns", [])
        if columns:
            parts.append(" | ".join(map(str, columns)))

        header_desc = block.get("header_descriptions", {})
        if isinstance(header_desc, dict):
            desc_lines = []
            for col in columns:
                if col in header_desc:
                    desc_lines.append(f"{col}: {header_desc[col]}")
            if desc_lines:
                parts.append("\n".join(desc_lines))

        data = block.get("data", [])
        for row in data:
            row_str = ["" if v is None else str(v) for v in row]
            parts.append(" | ".join(row_str))

        return "\n".join(parts)

    if t in ("paragraph", "header"):
        return block.get("text", "")

    if t == "figure":
        return block.get("caption", "")

    if t == "list":
        return "\n".join(block.get("items", []))

    return block.get("text", "")


def get_features(doc_id, structured):
    feats = []
    gt = structured.get(doc_id, {}).get("extracted_features", {})

    for k, v in gt.items():
        feats.append(k)
        if isinstance(v, list):
            feats.extend(map(str, v))
        else:
            feats.append(str(v))

    return list(dict.fromkeys(feats))
