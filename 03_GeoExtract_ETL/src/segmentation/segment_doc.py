import subprocess
import tempfile
from pathlib import Path
import docx2txt
from src.segmentation.utils import find_raw_file

# -----------------------------
# DOC — улучшенный сегментатор
# -----------------------------
def segment_doc(doc_id: str):
    path = Path(find_raw_file(doc_id))

    def safe_segment(text):
        from src.segmentation.rule_based import rule_based_segment
        blocks = rule_based_segment(text, file_type="doc_word", doc_id=doc_id)
        for b in blocks:
            b["start_page"] = 1
        print(f"[DOC DEBUG] safe_segment: text length={len(text)}, blocks={len(blocks)}")
        return blocks

    # Основной и единственный метод: LibreOffice → DOCX → docx2txt
    try:
        import subprocess, tempfile
        tmp_dir = tempfile.mkdtemp()
        out_docx = Path(tmp_dir) / (path.stem + ".docx")

        soffice_path = r"C:\Program Files\LibreOffice\program\soffice.exe"

        subprocess.run([
            soffice_path,
            "--headless",
            "--convert-to", "docx",
            "--outdir", tmp_dir,
            str(path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if out_docx.exists():
            import docx2txt
            text = docx2txt.process(str(out_docx))
            print(f"[DOC DEBUG] LibreOffice→DOCX OK for {doc_id}, text length={len(text)}")
            return safe_segment(text)
        else:
            print(f"[DOC DEBUG] LibreOffice→DOCX produced no file for {doc_id}")

    except Exception as e:
        print(f"[DOC DEBUG] LibreOffice→DOCX failed for {doc_id}: {e}")

    print(f"[WARN] DOC conversion failed completely for {doc_id}")
    return []