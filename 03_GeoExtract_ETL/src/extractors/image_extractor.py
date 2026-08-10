import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

from .base import wrap_extractor, base_result

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

# OCR-извлечение текста из изображения 
@wrap_extractor
def extract_image(path: Path):
    res = base_result()   # ← ИСПРАВЛЕНО
    res["source"] = "ocr_image"

    try:
        img = cv2.imread(str(path))
        if img is None:
            res["warnings"].append("Image load failed")
            return res

        img = safe_downscale(img)

        result = ocr_paddle.ocr(img, cls=True)
        lines = [text for line in result for box, (text, score) in line]

        if len(" ".join(lines).strip()) < 20:
            result2 = ocr_paddle.ocr(img, cls=True)
            for line in result2:
                for box, (text, score) in line:
                    lines.append(text)

        res["text"] = "\n".join(lines)
        res["requires_ocr"] = True
        return res

    except Exception as e:
        res["source"] = "ocr_error"
        res["warnings"].append(str(e))
        return res
