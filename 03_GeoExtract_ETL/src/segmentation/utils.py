import os
import re
from pathlib import Path
from src.segmentation.headers import is_header

RAW_DIR = Path("data/raw")

def find_raw_file(doc_id: str):
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if Path(f).stem == doc_id:
                return Path(root) / f
    return None