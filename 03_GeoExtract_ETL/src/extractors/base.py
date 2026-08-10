import time

# Базовая структура результата экстрактора
def base_result():
    return {
        "text": "",
        "source": "unknown",
        "time_sec": 0.0,
        "warnings": [],
        "blocks": [],
        "tables": [],
        "images": [],
        "requires_ocr": False,
    }

# Декоратор для экстракторов: время, ошибки, warnings
def wrap_extractor(func):
    """Декоратор: измеряет время, ловит ошибки, добавляет warnings."""    
    def wrapper(path):
        res = base_result()
        start = time.time()
        try:
            out = func(path)
            res.update(out)
            res["time_sec"] = time.time() - start   # ← ДОБАВИТЬ
        except Exception as e:
            res["source"] = "error"
            res["warnings"].append(str(e))
            res["time_sec"] = time.time() - start
        return res
    return wrapper
