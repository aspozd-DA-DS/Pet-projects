# **GeoExtract ETL: Система интеллектуального анализа и структурирования геологической документации (Oil & Gas)**

## 📌 Описание

GeoExtract ETL — масштабный pet‑проект по построению **полного промышленного ETL‑конвейера** для геологических документов: отчётов по сейсморазведке, данных скважин, геологических карт, интерпретаций и фондовых материалов.

Система автоматически:

- извлекает текст из PDF, DOCX, XLSX, TXT и изображений (native + OCR),
- классифицирует тип документа (scientific_paper / seismic_geology_study / technical_report),
- сегментирует документ на логические блоки (header, paragraph, list, table, figure),
- нормализует текст и исправляет OCR‑ошибки,
- извлекает ключевые геологические параметры (depth, porosity, formation, lithology, horizon, wavelet),
- строит эмбеддинги (MiniLM),
- формирует FAISS‑индекс,
- выполняет семантический поиск (Hit@3 = 0.84–1.0),
- генерирует поисковые запросы (Query Suggestion),
- предоставляет REST API для интеграции в корпоративные системы.

Практическая ценность:

- автоматизация анализа геологических отчётов,
- ускорение интерпретации данных (экономия времени ≈ 99%),
- подготовка данных для RAG‑систем и LLM‑моделей,
- создание корпоративной базы знаний по геологии.

---

## 🔧 Стек технологий

### Извлечение текста
- PyMuPDF  
- pdfplumber  
- pdf2image  
- PaddleOCR, EasyOCR, Tesseract  
- python-docx, mammoth  
- openpyxl, xlrd  
- Pillow, OpenCV  

### Нормализация и обработка текста
- chardet  
- langdetect  
- regex‑модуль (118 геологических паттернов)  
- Levenshtein  
- spaCy  

### ML / NLP
- scikit‑learn (TF‑IDF, LogisticRegression, LinearSVC)  
- SentenceTransformers (MiniLM‑L12‑v2)  
- FAISS (IndexFlatIP)  
- KMeans, HDBSCAN  

### Метрики
- CER, WER  
- Precision, Recall, F1  
- Hit@3  

### API
- FastAPI  
- Uvicorn  

### Визуализация
- Matplotlib  
- Seaborn  

---

## 📊 Данные

Проект использует:

- PDF (text-based + scanned)  
- DOCX/DOC  
- XLSX/XLS  
- TXT/RTF  
- Изображения (карты, разрезы)  
- Golden Set (22 документа с ручной разметкой)

Пайплайн формирует:

- очищенные текстовые блоки,  
- нормализованные таблицы,  
- извлечённые геологические признаки,  
- эмбеддинги,  
- FAISS‑индекс,  
- структурированный JSON для API.

---

## 🧩 Ключевые этапы проекта

### Этап 1 — EDA геологических документов
- анализ форматов,  
- оценка качества PDF и изображений,  
- формирование Golden Set.

### Этап 2 — Извлечение текста
- PyMuPDF для text‑PDF,  
- PaddleOCR для scanned PDF и изображений,  
- DOCX → Mammoth,  
- XLSX → pandas/openpyxl,  
- единый формат результата.

### Этап 3 — OCR‑эксперименты
- сравнение Tesseract, EasyOCR, PaddleOCR,  
- CER/WER по Golden Set,  
- выбор PaddleOCR как основного OCR.

### Этап 4 — Сегментация документа
- rule‑based сегментация,  
- fuzzy‑matching с Golden Set,  
- F1 по типам блоков,  
- формирование чанков.

### Этап 5 — Нормализация и извлечение признаков
- исправление OCR‑ошибок,  
- очистка текста,  
- regex‑matching (118 геологических параметров),  
- GT‑оценка (precision = 1.0, recall ≈ 0.77).

### Этап 6 — Классификация документа
- TF‑IDF + LogisticRegression,  
- accuracy ≈ 0.83,  
- fallback‑логика.

### Этап 7 — Semantic Search
- эмбеддинги MiniLM‑L12‑v2,  
- FAISS‑индекс,  
- Hit@3 = 0.84–1.0,  
- Query Suggestion,  
- t‑SNE визуализация.

### Этап 8 — Полный ETL‑конвейер
- объединение всех шагов,  
- формирование финального JSON.

### Этап 9 — REST API
- загрузка документа,  
- просмотр meta / blocks / regex / chunks,  
- semantic search,  
- Demo‑ноутбук.

---

## 📈 Результаты

- OCR PaddleOCR:  
  - CER ≈ 0.14 (PDF scanned)  
  - CER ≈ 0.07 (images)

- Сегментация:  
  - Weighted F1 = 0.60  
  - F1 paragraph = 0.64  
  - F1 header = 0.70  

- Извлечение признаков:  
  - precision = 1.0  
  - recall ≈ 0.77  

- Классификация:  
  - accuracy ≈ 0.83  

- Semantic Search:  
  - Hit@3 = 0.84–1.0  
  - MiniLM‑L12‑v2 — лучшая модель  

- API:  
  - полностью рабочий REST‑сервис  
  - поддержка всех форматов  
  - интеграция FAISS  

---

## 📁 Структура репозитория

project/
├── api_results/  
│
├── api_test_docs/ 
│
├── data/                                                                   (исходные данные)
│   ├── golden_set/
│   │   ├── annotations/
│   │   ├── gs_data/
│   │   ├── for_metadata_gs_features.txt
│   │   ├── metadata_gs_features.csv
│   │   ├── doc_labels.txt
│   │   └── doc_labels_all.txt
│   └── raw/  
│       ├── pdf_text/
│       ├── pdf_scans/
│       ├── doc_Word/
│       ├── tables/
│       ├── images/
│       └── txt/
│   
├── log/ 
│ 
├── results/ 
│   ├── classification/
│   ├── metadata/
│   ├── ocr/
│   ├── segmentation/
│   ├── step1/
│   ├── step2/
│   ├── step3/
│   ├── step4/
│   ├── step5/
│   ├── step6/
│   ├── step7/
│   └── step8/
│ 
├── src/                                                                    (Python модули
│   ├── extractors/
│   │   ├── base.py
│   │   ├── docx_extractor.py
│   │   ├── excel_extractor.p                                                                             
│   │   ├── image_extractor.py
│   │   ├── pdf_scan_extractor.py                                                                             
│   │   ├── pdf_text_extractor.py                                                                             
│   │   ├── rtf_extractor.py 
│   │   ├── table_image_analyzer.py
│   │   └── txt_extractor.py  
│   ├── segmentation/     
│   │   ├── headers.py
│   │   ├── normalization.py
│   │   ├── postprocess.py                                                                            
│   │   ├── rule_based.py
│   │   ├── segment_doc.py                                                                             
│   │   ├── segment_docx.py                                                                             
│   │   ├── segment_images.py 
│   │   ├── segment_pdf_scans.py
│   │   ├── segment_pdf_text.py                                                                             
│   │   ├── segment_txt.py                                                                             
│   │   ├── segment_xlsx.py                                                                             
│   │   ├── utils.py                                                                                  
│   │   └── well_log.py   
│   ├── semantic_search/                                                                    
│   │   ├── chunks.py
│   │   ├── embeddings.py
│   │   ├── faiss.py                                                                         
│   │   ├── loaders.py
│   │   ├── prepare.py                                                                          
│   │   ├── qs_table.py                                                                          
│   │   ├── query_suggestions.py
│   │   ├── search.py
│   │   └──  suggestions.py                                                                           
│   │   classifier.py
│   │   extract_text.py                                                                             
│   │   features.py                                                                             
│   │   final_json.py    
│   │   normalization.py                                                                            
│   └── structuring.py
│  
├── plan_prj_ETL-serv.pdf                                           (план проекта)                                                                            
├── 01_research_and_pipeline.ipynb                                  (Шаги 1–7 + 10: EDA + model selection + experiments)
├── 02_ETL.ipynb                                                    (Шаги 8: ETL)  
├── 03_DEMO_API.ipynb                                               (Шаги 9: API) 
├── 01_research_and_pipeline.pdf                                    (Шаги 1–7 + 10: EDA + model selection + experiments)
├── 02_ETL.pdf                                                      (Шаги 8: ETL)  
├── 03_DEMO_API.pdf                                                 (Шаги 9: API)                                                                             
├── api.py
├── requirements.txt
└── README.md

---


## 🚀 Как запустить
1. Склонировать репозиторий:  
   ```bash
   git clone https://github.com/aspozd-DA-DS/Pet-projects.git

2. Перейти в папку проекта:

   ```bash
   cd Pet-projects/03_GeoExtract_ETL

3. Установить зависимости
   ```bash
   pip install -r requirements.txt

3. Запустить ноутбуки:

   ```bash
   jupyter notebook 01_research_and_pipeline.ipynb

   ```bash
   jupyter notebook 02_ETL.ipynb

4. Запустить API

   ```bash
   uvicorn api:app --reload

5. Открыть Demo‑ноутбук
   ```bash
   jupyter notebook 03_DEMO_API.ipynb


## 🏷 Topics
 
`ETL` `OCR` `PyMuPDF` `PaddleOCR` `Semantic Search`  `Segmentation`
`FAISS` `MiniLM` `Regex Extraction` `Geology`  `Python`
`Document Processing` `NLP` `Machine Learning`  
`FastAPI` `Data Science` `Oil & Gas`