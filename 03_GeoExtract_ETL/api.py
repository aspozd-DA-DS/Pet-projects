import json
import shutil
import traceback
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz 
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

# ============================================================
# ПУТИ
# ============================================================

API_DIR = Path("data/api")
RAW_DIR = Path("data/raw")
API_RESULTS_DIR = Path("api_results")
LOG_DIR = Path("logs")

PDF_TEXT_DIR = RAW_DIR / "pdf_text"
PDF_SCANS_DIR = RAW_DIR / "pdf_scans"
DOC_WORD_DIR = RAW_DIR / "doc_Word"
TABLES_DIR = RAW_DIR / "tables"
IMAGES_DIR = RAW_DIR / "images"
TXT_DIR = RAW_DIR / "txt"

for d in [PDF_TEXT_DIR, PDF_SCANS_DIR, DOC_WORD_DIR, TABLES_DIR, IMAGES_DIR, TXT_DIR, API_RESULTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

API_DIR.mkdir(parents=True, exist_ok=True)
API_RESULTS_DIR.mkdir(parents=True, exist_ok=True) 

def is_scanned_pdf(file_path: str, min_text_len: int = 20) -> bool:
    """
    Определяет, является ли PDF сканом.
    Логика:
    - если текстового слоя нет или он слишком маленький → это скан.
    """

    try:
        doc = fitz.open(file_path)
    except Exception:
        return True  # если PDF битый — считаем сканом

    total_text = ""

    for page in doc:
        text = page.get_text("text")
        if text:
            total_text += text.strip()

        # если уже нашли достаточно текста — PDF не скан
        if len(total_text) > min_text_len:
            return False

    # если после всех страниц текста почти нет → скан
    return len(total_text) < min_text_len


# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================

logging.basicConfig(
    filename=LOG_DIR / "api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("API v3.0 started")

# ============================================================
# ИМПОРТЫ ИЗ ШАГА 8
# ============================================================

from src.extract_text import extract_text
from src.segmentation.rule_based import (
    rule_based_segment,
    split_long_paragraphs,
    get_pred_text
)
from src.segmentation.segment_xlsx import segment_xlsx_universal
from src.normalization import load_ocr_fixes, normalize_blocks
from src.features import load_patterns_regex, extract_features_regex
from src.classifier import load_tfidf, load_classifier, classify_document

# ============================================================
# Шаг 7 — локальные чанки
# ============================================================

def build_chunks_for_single_doc(doc_id: str, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for i, b in enumerate(blocks):
        chunk_id = f"{doc_id}_chunk_{i}"

        if b["type"] in ("paragraph", "header", "header_paragraph"):
            text = b.get("text", "")
        elif b["type"] == "list":
            text = "\n".join(b.get("items", []))
        elif b["type"] == "table":
            text = get_pred_text(b)
        else:
            text = get_pred_text(b)

        chunk = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "block_type": b["type"],
            "text": text,
            "features": {}
        }
        chunks.append(chunk)

    return chunks

# ============================================================
# ЗАГРУЗКА МОДЕЛЕЙ ПРИ СТАРТЕ API
# ============================================================

logging.info("Loading TF-IDF and classifier...")
TFIDF = load_tfidf("results/step8/tfidf.pkl")
CLASSIFIER = load_classifier("results/step8/classifier.pkl")

logging.info("Loading OCR fixes and regex patterns...")
OCR_FIXES = load_ocr_fixes("results/step8/ocr_fixes.json")
PATTERNS_REGEX = load_patterns_regex("results/step8/patterns_regex.json")

logging.info("Loading SentenceTransformer for semantic search...")
SEM_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", local_files_only=True)

# ============================================================
# SEMANTIC SEARCH: ЗАГРУЗКА БАЗОВОГО FAISS И ЧАНКОВ
# ============================================================

STEP8_DIR = Path("results/step8")
FAISS_BASE_PATH = STEP8_DIR / "st7_faiss_index.bin"
FAISS_UPD_PATH = STEP8_DIR / "st7_faiss_index_upd.bin"
CHUNKS_BASE_PATH = STEP8_DIR / "st7_chunks.jsonl"
CHUNKS_UPD_PATH = STEP8_DIR / "st7_chunks_upd.jsonl"
EMB_UPD_PATH = STEP8_DIR / "st7_embeddings_upd.npy"  # опционально, для контроля

faiss_index: Optional[faiss.Index] = None
chunks_all: List[Dict[str, Any]] = []
base_chunk_count: int = 0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def init_faiss_and_chunks():
    global faiss_index, chunks_all, base_chunk_count

    # Загружаем базовые чанки
    base_chunks = load_jsonl(CHUNKS_BASE_PATH)
    base_chunk_count = len(base_chunks)

    # Загружаем обновлённые чанки (если есть)
    upd_chunks = load_jsonl(CHUNKS_UPD_PATH)

    chunks_all = base_chunks + upd_chunks

    # Загружаем FAISS: если есть обновлённый индекс — используем его, иначе базовый
    if FAISS_UPD_PATH.exists():
        logging.info("Loading updated FAISS index...")
        faiss_index = faiss.read_index(str(FAISS_UPD_PATH))
    elif FAISS_BASE_PATH.exists():
        logging.info("Loading base FAISS index...")
        faiss_index = faiss.read_index(str(FAISS_BASE_PATH))
    else:
        logging.warning("No FAISS index found; semantic_search will be disabled.")
        faiss_index = None

    logging.info(f"Semantic search: loaded {len(chunks_all)} chunks (base={base_chunk_count}, upd={len(upd_chunks)})")


init_faiss_and_chunks()

# ============================================================
# API
# ============================================================

app = FastAPI(title="Document ETL API", version="3.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ============================================================
# HTML-страница загрузки документа
# ============================================================

@app.get("/upload_page", response_class=HTMLResponse)
async def upload_page():
    return """
    <html>
        <head>
            <title>Document Upload</title>
        </head>
        <body>
            <h2>Upload Document</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <button type="submit">Upload</button>
            </form>
        </body>
    </html>
    """

# ============================================================
# РОУТЕР ДЛЯ РАСКЛАДКИ ФАЙЛОВ
# ============================================================

def route_file_to_raw(file_path: Path):
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        # Проверяем, есть ли текстовый слой
        scanned = is_scanned_pdf(str(file_path))

        if scanned:
            shutil.copy(str(file_path), str(PDF_SCANS_DIR / file_path.name))
        else:
            shutil.copy(str(file_path), str(PDF_TEXT_DIR / file_path.name))
    elif ext in [".doc", ".docx"]:
        shutil.copy(str(file_path), str(DOC_WORD_DIR / file_path.name))
    elif ext in [".xls", ".xlsx"]:
        shutil.copy(str(file_path), str(TABLES_DIR / file_path.name))
    elif ext in [".jpg", ".jpeg", ".png"]:
        shutil.copy(str(file_path), str(IMAGES_DIR / file_path.name))
    elif ext in [".txt", ".rtf"]:
        shutil.copy(str(file_path), str(TXT_DIR / file_path.name))
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}")

    return True

# ============================================================
# МИНИ-ETL ДЛЯ ОДНОГО ДОКУМЕНТА
# ============================================================

def run_etl_single(raw_path: Path) -> Dict[str, Any]:
    doc_id = raw_path.stem
    ext = raw_path.suffix.lower()

    logging.info(f"ETL started for {doc_id}")

    # Шаг 2 — extract_text
    r = extract_text(raw_path)
    raw_text = r.get("text", "")
    meta = r.get("meta", {})
    meta["doc_id"] = doc_id
    meta["extension"] = ext

    # Шаг 4 — сегментация
    if ext in [".xls", ".xlsx"]:
        blocks = segment_xlsx_universal(doc_id)
    else:
        blocks = rule_based_segment(raw_text, file_type=meta.get("source", ""), doc_id=doc_id)

    blocks = split_long_paragraphs(blocks, file_type=meta.get("source", ""))

    # Шаг 5 — нормализация
    blocks_norm = normalize_blocks(blocks, OCR_FIXES)

    # Шаг 5 — regex features (только непустые)
    feats_raw = extract_features_regex(blocks_norm, PATTERNS_REGEX)
    feats_regex = {k: v for k, v in feats_raw.items() if v}

    # Шаг 6 — классификация
    cls_res = classify_document(
        {"raw_text": raw_text, "blocks": blocks_norm},
        TFIDF,
        CLASSIFIER
    )

    # Шаг 7 — чанки
    chunks = build_chunks_for_single_doc(doc_id, blocks_norm)

    # Финальный JSON
    final_doc = {
        "doc_id": doc_id,
        "meta": meta,
        "raw_text": raw_text,
        "blocks": blocks_norm,
        "extracted_features_regex": feats_regex,
        "doc_type": cls_res["doc_type"],
        "classification_confidence": cls_res["confidence"],
        "model_predicted_label": cls_res["confidence"],
        "model_predicted_label": cls_res["model_predicted_label"],
        "model_confidence": cls_res["model_confidence"],
        "chunks": chunks
    }

    # Сохранение JSON
    out_path = API_RESULTS_DIR / f"{doc_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_doc, f, ensure_ascii=False, indent=2)

    logging.info(f"ETL completed for {doc_id}")

    return final_doc

# ============================================================
# SEMANTIC: ДОБАВЛЕНИЕ НОВЫХ ЧАНКОВ В ОБНОВЛЁННЫЙ ИНДЕКС
# ============================================================

def update_semantic_index_with_doc(doc_id: str, chunks: List[Dict[str, Any]]):
    global faiss_index, chunks_all

    if faiss_index is None:
        # Если базового индекса нет — создаём новый
        logging.info("Creating new FAISS index for semantic search...")
        faiss_index = faiss.IndexFlatL2(384)  # размер эмбеддинга MiniLM-L12-v2

    # эмбеддинги для новых чанков
    texts = [c["text"] for c in chunks]
    if not texts:
        return

    embeddings = SEM_MODEL.encode(texts, convert_to_numpy=True)
    faiss_index.add(embeddings.astype(np.float32))

    # сохраняем обновлённый индекс в _upd.bin
    faiss.write_index(faiss_index, str(FAISS_UPD_PATH))

    # добавляем новые чанки в общий список и в _upd.jsonl
    chunks_all.extend(chunks)
    append_jsonl(CHUNKS_UPD_PATH, chunks)

    # опционально сохраняем эмбеддинги
    if EMB_UPD_PATH.exists():
        old_emb = np.load(EMB_UPD_PATH)
        new_emb = np.concatenate([old_emb, embeddings], axis=0)
        np.save(EMB_UPD_PATH, new_emb)
    else:
        np.save(EMB_UPD_PATH, embeddings)

    logging.info(f"Semantic index updated with {len(chunks)} chunks from doc_id={doc_id}")

# ============================================================
# ENDPOINT: /upload (async)
# ============================================================

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file:
        return {"error": "Файл не загружен"}

    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".jpg", ".jpeg", ".png", ".txt", ".rtf"]:
        return {"error": "Неподдерживаемый формат файла"}

    logging.info(f"Uploaded file: {file.filename}")

    # Сохраняем в data/api
    api_path = API_DIR / file.filename
    try:
        contents = await file.read()
        with open(api_path, "wb") as f:
            f.write(contents)
    except Exception:
        logging.error(f"Error saving file {file.filename}")
        return {"error": "Не удалось сохранить файл"}

    # Перемещаем в data/raw (но без определения папки)
    try:
        route_file_to_raw(api_path)
    except Exception as e:
        logging.error(f"Error routing file {file.filename}: {e}")
        return {"error": f"Ошибка перемещения файла: {e}"}

    # 🔥 Правильное определение пути в RAW_DIR
    if ext == ".pdf":
        scanned = is_scanned_pdf(str(api_path))
        subfolder = "pdf_scans" if scanned else "pdf_text"
    else:
        subfolder = {
            ".doc": "doc_Word",
            ".docx": "doc_Word",
            ".xls": "tables",
            ".xlsx": "tables",
            ".jpg": "images",
            ".jpeg": "images",
            ".png": "images",
            ".txt": "txt",
            ".rtf": "txt",
        }[ext]

    raw_target_path = RAW_DIR / subfolder / file.filename

    # Запуск мини-ETL
    try:
        result = run_etl_single(raw_target_path)
    except Exception as e:
        logging.error(f"Error processing {file.filename}: {e}")
        return {"error": f"Ошибка обработки документа: {e}", "trace": traceback.format_exc()}

    # Обновляем semantic index новыми чанками
    try:
        update_semantic_index_with_doc(result["doc_id"], result["chunks"])
    except Exception as e:
        logging.error(f"Error updating semantic index for {file.filename}: {e}")

    return result

# ============================================================
# ОТДЕЛЬНЫЕ КОМАНДЫ ДЛЯ ПРОСМОТРА РЕЗУЛЬТАТОВ
# ============================================================

def load_saved(doc_id: str) -> Optional[Dict[str, Any]]:
    path = API_RESULTS_DIR / f"{doc_id}.json"
    if not path.exists():
        return None
    return json.load(open(path, "r", encoding="utf-8"))

@app.get("/meta/{doc_id}")
async def get_meta(doc_id: str):
    data = load_saved(doc_id)
    if not data:
        return {"error": "document not found"}
    return data["meta"]

@app.get("/blocks/{doc_id}")
async def get_blocks(doc_id: str):
    data = load_saved(doc_id)
    if not data:
        return {"error": "document not found"}
    return data["blocks"]

@app.get("/regex/{doc_id}")
async def get_regex(doc_id: str):
    data = load_saved(doc_id)
    if not data:
        return {"error": "document not found"}
    return data["extracted_features_regex"]

@app.get("/chunks/{doc_id}")
async def get_chunks(doc_id: str):
    data = load_saved(doc_id)
    if not data:
        return {"error": "document not found"}
    return data["chunks"]

# ============================================================
# МАССОВАЯ ОБРАБОТКА ПАПКИ (async)
# ============================================================

@app.post("/process_folder")
async def process_folder(folder_path: str):
    folder = Path(folder_path)
    if not folder.exists():
        logging.error(f"Folder not found: {folder_path}")
        return {"error": "folder not found"}

    logging.info(f"Processing folder: {folder_path}")

    results = []
    for file_path in folder.iterdir():
        if file_path.suffix.lower() not in [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".jpg", ".jpeg", ".png", ".txt", ".rtf"]:
            continue

        try:
            api_copy = API_DIR / file_path.name
            shutil.copy(file_path, api_copy)

            route_file_to_raw(api_copy)

            raw_target_path = RAW_DIR / {
                ".pdf": "pdf_text",
                ".doc": "doc_Word",
                ".docx": "doc_Word",
                ".xls": "tables",
                ".xlsx": "tables",
                ".jpg": "images",
                ".jpeg": "images",
                ".png": "images",
                ".txt": "txt",
                ".rtf": "txt",
            }[file_path.suffix.lower()] / file_path.name

            result = run_etl_single(raw_target_path)
            update_semantic_index_with_doc(result["doc_id"], result["chunks"])

            results.append({"file": file_path.name, "status": "ok"})
        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            results.append({"file": file_path.name, "status": "error", "error": str(e)})

    return {"processed": results}

@app.get("/list_results")
async def list_results():
    files = sorted(API_RESULTS_DIR.glob("*.json"))
    return [f.name for f in files]

@app.get("/get_result/{doc_id}")
async def get_result(doc_id: str):
    path = API_RESULTS_DIR / f"{doc_id}.json"
    if not path.exists():
        return {"error": "document not found"}
    return json.load(open(path, "r", encoding="utf-8"))


# ============================================================
# SEMANTIC SEARCH ENDPOINT
# ============================================================

@app.get("/semantic_search")
async def semantic_search(query: str, top_k: int = 5):
    if faiss_index is None or not chunks_all:
        return {"error": "semantic search not initialized"}

    # эмбеддинг запроса
    q_emb = SEM_MODEL.encode([query], convert_to_numpy=True).astype(np.float32)
    D, I = faiss_index.search(q_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks_all):
            continue
        ch = chunks_all[idx]
        results.append({
            "score": float(score),
            "chunk_id": ch.get("chunk_id"),
            "doc_id": ch.get("doc_id"),
            "block_type": ch.get("block_type"),
            "text": ch.get("text")
        })

    return {
        "query": query,
        "top_k": top_k,
        "results": results
    }
