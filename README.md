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
| [Attribute QC_CV — кросс-плот и кросс-валидация для атрибутного анализа (Oil & Gas)](05_Attribute_QC) | Интерактивный Streamlit-дашборд для оценки качества связи между сейсмическими атрибутами и свойствами пласта по данным скважин. Сравнивает эталонный параметр (эффективная толщина H_eff или пористость Phi) с сейсмическим атрибутом в тех же скважинах. Разделяет in-sample и out-of-sample оценку: до 9 схем кросс-валидации (LOWO, Block, Buffered, Spatial Leave-P-Out, Distance-based exclusion, Stratified by Zone и др.), автоматический выбор лучшей схемы по RMSE + RMSE_fold_std. Включает 17 типов аппроксимаций, квантильную регрессию, выбросы CV-невязок четырьмя методами, диагностику по категориям и стресс-тест устойчивости через фильтрацию скважин. Шаг 5: многоатрибутная регрессия отдельно для H_eff и Phi с отбором по RMSE_cv. Двуязычный интерфейс (RU/EN). | Python, NumPy, Pandas, Streamlit, Plotly, Seaborn, scikit-learn, statsmodels, SciPy, OpenPyXL, xlrd, Matplotlib, Jupyter |

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
| [Attribute QC_CV — cross-plot & cross-validation for seismic attribute analysis (Oil & Gas)](05_Attribute_QC) | An interactive Streamlit dashboard for assessing how well seismic attributes predict reservoir properties at wells. It compares a reference parameter (effective thickness H_eff or porosity Phi) with a seismic attribute at the same wells. It separates in-sample and out-of-sample estimates: up to 9 cross-validation schemes (LOWO, Block, Buffered, Spatial Leave-P-Out, Distance-based exclusion, Stratified by Zone, etc.) with automatic selection of the best scheme by RMSE + RMSE_fold_std. Includes 17 approximation types, quantile regression, CV residual outliers detected by four methods, category diagnostics and a stability stress test through well filtering. Step 5: multi-attribute regression for H_eff and Phi separately, selected by RMSE_cv. Bilingual interface (RU/EN). | Python, NumPy, Pandas, Streamlit, Plotly, Seaborn, scikit-learn, statsmodels, SciPy, OpenPyXL, xlrd, Matplotlib, Jupyter |

<p align="right"><a href="#top">⬆ back to top</a></p>
