from pathlib import Path
import subprocess
import mammoth
import re
import docx as docx_lib

from .base import wrap_extractor, base_result

SOFFICE_PATH = r"C:\Program Files\LibreOffice\program\soffice.exe"

TEMP_DOC_DIR = Path(r"E:\projects\My_project\06_Data_extraction_ETL_service\data\doc_to_docx")
TEMP_DOC_DIR.mkdir(parents=True, exist_ok=True)

TEMP_PDF_DIR = Path(r"E:\projects\My_project\06_Data_extraction_ETL_service\data\doc_to_pdf")
TEMP_PDF_DIR.mkdir(parents=True, exist_ok=True)

def convert_docx_to_pdf(path):
    try:
        subprocess.run([
            SOFFICE_PATH, "--headless", "--convert-to", "pdf",
            str(path), "--outdir", str(TEMP_PDF_DIR)
        ], check=True)

        pdf_candidates = list(TEMP_PDF_DIR.glob(path.stem + "*.pdf"))
        if not pdf_candidates:
            return None

        return pdf_candidates[0]

    except Exception:
        return None
    
def convert_doc_to_docx(doc_path: Path):
    doc_path = Path(doc_path)

    temp_dir = TEMP_DOC_DIR / f"tmp_{doc_path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    out_docx = temp_dir / (doc_path.stem + ".docx")

    subprocess.run([
        SOFFICE_PATH, "--headless", "--convert-to", "docx",
        str(doc_path), "--outdir", str(temp_dir)
    ], check=True)

    if not out_docx.exists():
        raise FileNotFoundError(f"LibreOffice не создало DOCX: {out_docx}")

    return out_docx

def extract_docx_text_mammoth(path: Path):
    with open(path, "rb") as f:
        result = mammoth.extract_raw_text(f)
    text = result.value
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text

@wrap_extractor
def extract_docx(path: Path):
    res = base_result()
    ext = path.suffix.lower()
    temp_docx = None

    if ext == ".doc":
        try:
            temp_docx = convert_doc_to_docx(path)
            path = temp_docx
        except Exception as e:
            res["source"] = "error"
            res["warnings"].append(f"DOC→DOCX conversion error: {e}")
            return res

    try:
        text = extract_docx_text_mammoth(path)
        res["source"] = "native_mammoth"
    except Exception as e:
        res["source"] = "error"
        res["warnings"].append(f"Mammoth extract error: {e}")
        text = ""

    table_parts = []
    try:
        docx_obj = docx_lib.Document(str(path))

        for table in docx_obj.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    t = cell.text.strip()
                    if t:
                        row_text.append(t)
                if row_text:
                    table_parts.append(" | ".join(row_text))

    except Exception as te:
        res["warnings"].append(f"table parse warning: {te}")

    if table_parts:
        text = text + "\n" + "\n".join(table_parts)

    res["text"] = text

    if temp_docx and temp_docx.exists():
        try:
            temp_docx.unlink()
        except Exception:
            pass

    return res
