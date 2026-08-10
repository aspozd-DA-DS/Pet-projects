import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

from .base import wrap_extractor, base_result   # ← ДОБАВЛЕНО

# -----------------------------
# PaddleOCR
# -----------------------------
ocr_paddle = PaddleOCR(
    show_log=False,
    use_angle_cls=True,
    lang='en',
    use_gpu=False,
    enable_mkldnn=True,
    det_limit_side_len=1280,
    rec_image_shape="3, 64, 480",
    det_model_dir=r"E:\PaddleOCR\en_PP-OCRv3_det_infer",
    rec_model_dir=r"E:\PaddleOCR\en_PP-OCRv4_rec_infer",
    cls_model_dir=r"E:\PaddleOCR\ch_ppocr_mobile_v2.0_cls_infer"
)

# -----------------------------
# Safe downscale
# -----------------------------
def safe_downscale(img, max_size=2000):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# OCR-извлечение текста из PDF-сканов 
@wrap_extractor
def extract_pdf_scanned(path: Path):
    res = base_result()   # ← ИСПРАВЛЕНО
    res["source"] = "ocr_pdf"

    try:
        pages = convert_from_path(path, dpi=220)
    except Exception as e:
        res["source"] = "ocr_error"
        res["warnings"].append(str(e))
        return res

    text_parts = []

    for img in pages:
        img = np.array(img)
        img = safe_downscale(img)

        result = ocr_paddle.ocr(img, cls=True)
        page_text = [text for line in result for box, (text, score) in line]

        if len(" ".join(page_text).strip()) < 20:
            result2 = ocr_paddle.ocr(img, cls=True)
            for line in result2:
                for box, (text, score) in line:
                    page_text.append(text)

        if page_text:
            text_parts.append("\n".join(page_text))

    res["text"] = "\n".join(text_parts)
    res["requires_ocr"] = True
    return res
