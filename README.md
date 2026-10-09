<a id="top"></a>

<p align="right">
  <a href="#ru">🇷🇺 RU</a> | <a href="#en">🇬🇧 EN</a>
</p>

---

<a id="ru"></a>

## 🇷🇺 Русский

Здесь собраны проекты, реализованные по направлению Data Science в рамках собственных pet-проектов.

### 📂 Портфолио проектов

| Название проекта | Краткое описание | Стек |
|-|-|-|
| [Семантические пространства литературных стилей (NLP)](01_Word2Vec) | Сравнение стилей русских писателей с помощью методов NLP, реализация Word2Vec, визуализация эмбеддингов | Python, NumPy, Pandas, Keras, NLP, t-SNE, PCA, Google Colab (GPU) |
| [Детекция и классификация объектов на изображениях (Yolo+EfficientNet)](02_Detect_Cls_object) | Полный конвейер компьютерного зрения: детекция объектов (YOLOv9m), классификация типов (EfficientNet-B4), детекция и классификация дефектов (SAM+YOLO detect/cls/seg), определение цвета объектов по HSV/LAB с обучением градиентных бустингов | Python, NumPy, Pandas, PyTorch, Streamlit, OpenCV, YOLOv9m, SAM, EfficientNet-B4, XGBoost, LightGBM, scikit-learn, Matplotlib, Seaborn, Albumentations, GPU/AMP |
| [GeoExtract ETL: Система интеллектуального анализа и структурирования геологической документации (Oil & Gas)](03_GeoExtract_ETL) | Промышленный конвейер обработки геологических отчётов: извлечение текста (PyMuPDF + OCR), сегментация, нормализация, извлечение геологических признаков (regex + GT-matching), классификация документов (TF-IDF + ML), генерация эмбеддингов (MiniLM), построение FAISS-индекса и семантический поиск. Включает API, Demo-ноутбук и полный набор метрик (CER/WER, F1, Hit@3). | Python, NumPy, Pandas, PyMuPDF, pdfplumber, PaddleOCR, EasyOCR, OpenCV, spaCy, regex, scikit-learn, SentenceTransformers, FAISS, Matplotlib, Seaborn, FastAPI, Uvicorn |
| [Structural QC — cross-plot & cross-validation (Oil & Gas)](04_Structural_QC) | Интерактивный Streamlit-дашборд для контроля качества структурных построений по данным сейсморазведки МОГТ 2D/3D. Сравнивает эталонный параметр (например, глубину горизонта по скважине Z_well) со сравниваемым (глубину того же горизонта по структурной карте Z_map) в одних и тех же точках. Оценивает связь через 8 схем кросс-валидации (LOWO, Block, Buffered, Spatial Leave-P-Out, Distance-based exclusion, Stratified by Zone и др.), автоматически определяет лучшую схему по критерию RMSE + RMSE_fold_std. Включает 17 типов аппроксимаций, анализ невязок, карту невязок, детекцию выбросов 4 методами, диагностику по категориям (горизонты, тектонические зоны) и стресс-тест устойчивости через фильтрацию скважин. Двуязычный интерфейс (RU/EN). | Python, NumPy, Pandas, Streamlit, Plotly, scikit-learn, statsmodels, SciPy, Matplotlib, Seaborn, OpenPyXL |

<p align="right"><a href="#top">⬆ наверх</a></p>

---

<a id="en"></a>

## 🇬🇧 English

A collection of projects implemented in the Data Science field as personal pet projects.

### 📂 Project Portfolio

| Project | Short description | Stack |
|-|-|-|
| [Semantic Spaces of Literary Styles (NLP)](01_Word2Vec) | Comparison of the styles of Russian writers using NLP methods, Word2Vec implementation, and embedding visualization | Python, NumPy, Pandas, Keras, NLP, t-SNE, PCA, Google Colab (GPU) |
| [Object Detection and Classification in Images (Yolo+EfficientNet)](02_Detect_Cls_object) | End-to-end computer vision pipeline: object detection (YOLOv9m), type classification (EfficientNet-B4), defect detection and classification (SAM+YOLO detect/cls/seg), and object color determination via HSV/LAB with gradient boosting models | Python, NumPy, Pandas, PyTorch, Streamlit, OpenCV, YOLOv9m, SAM, EfficientNet-B4, XGBoost, LightGBM, scikit-learn, Matplotlib, Seaborn, Albumentations, GPU/AMP |
| [GeoExtract ETL: Intelligent Analysis and Structuring of Geological Documentation (Oil & Gas)](03_GeoExtract_ETL) | Industrial pipeline for processing geological reports: text extraction (PyMuPDF + OCR), segmentation, normalization, extraction of geological features (regex + GT matching), document classification (TF-IDF + ML), embedding generation (MiniLM), FAISS index construction, and semantic search. Includes an API, a demo notebook, and a full set of metrics (CER/WER, F1, Hit@3). | Python, NumPy, Pandas, PyMuPDF, pdfplumber, PaddleOCR, EasyOCR, OpenCV, spaCy, regex, scikit-learn, SentenceTransformers, FAISS, Matplotlib, Seaborn, FastAPI, Uvicorn |
| [Structural QC: Cross-plot & Cross-validation (Oil & Gas)](04_Structural_QC) | Interactive Streamlit dashboard for quality control of structural mapping based on 2D/3D CDP (MOGT) seismic data. Compares a reference parameter (e.g., horizon depth from well data, Z_well) with a compared parameter (depth of the same horizon from a structural map, Z_map) at the same points. Evaluates the relationship using 8 cross-validation schemes (LOWO, Block, Buffered, Spatial Leave-P-Out, Distance-based exclusion, Stratified by Zone, and others), and automatically selects the best scheme by the RMSE + RMSE_fold_std criterion. Includes 17 approximation types, residual analysis, a residual map, outlier detection with 4 methods, diagnostics by category (horizons, tectonic zones), and a robustness stress test via well filtering. Bilingual interface (RU/EN). | Python, NumPy, Pandas, Streamlit, Plotly, scikit-learn, statsmodels, SciPy, Matplotlib, Seaborn, OpenPyXL |

<p align="right"><a href="#top">⬆ back to top</a></p>
