<a id="top"></a>

<p align="right">
  <a href="#ru">🇷🇺 RU</a> | <a href="#en">🇬🇧 EN</a>
</p>

---

<a id="ru"></a>

# Детекция, классификация и определение цвета объектов в прозрачном пакетике (YOLOv9, EfficientNet, ML)

## 📌 Описание
Масштабный pet‑проект по построению полного компьютерного зрения пайплайна, включающего:
- детекцию объектов внутри прозрачного пакетика
- классификацию типа объекта (capsule / softgel / meltlet / caplet)
- детекцию и классификацию дефектов (chip, crack, leak, wrong_size, no_def)
- сегментацию дефектов (YOLO‑Seg)
- определение цвета объектов с помощью ML‑моделей (LightGBM, XGBoost, RF, kNN)

Проект объединяет детекцию, сегментацию, классификацию и feature engineering, формируя единый промышленный пайплайн для анализа объектов на изображениях.

Практическая ценность:
- автоматизация контроля качества фармацевтических объектов
- определение дефектов и цвета без участия человека
- построение универсального CV‑конвейера, который можно интегрировать в производство
- демонстрация владения YOLO, EfficientNet, SAM, ML‑классификацией, feature engineering и визуализацией

---

## 🔧 Стек технологий

Computer Vision
- YOLOv9m, YOLOv8‑Seg, YOLO‑CLS, YOLO‑Detect
- Segment Anything Model (SAM ViT‑H)
- OpenCV, Albumentations

Deep Learning
- PyTorch, EfficientNet‑B3/B4
- AMP, cosine LR scheduler, AdamW

Machine Learning
- LightGBM, XGBoost, RandomForest, kNN
- KMeans, entropy‑based features, LAB/HSV color analysis

Data & Visualization
- NumPy, Pandas
- Matplotlib, Seaborn

Инфраструктура
- Streamlit (кастомный инструмент разметки)
- GPU‑ускорение

Полностью воспроизводимые Jupyter‑ноутбуки

---

## 📊 Данные

Проект использует несколько уровней данных:
- Исходные изображения пакетиков
- YOLO‑разметка
- Кропы объектов
- Дефекты
- Цветовые данные
  - Пайплайн формирует:
    - очищенные кропы
    - HSV/LAB‑представления
    - валидные пиксели
    - таблицу признаков 
 
---
## Ключевые этапы проекта

Этап 1 — Разметка изображений (Streamlit)
- собственное приложение для разметки
- автоматическое определение цвета и формы
- сохранение YOLO + JSON разметки
- визуализация аннотаций
- формирование единого annotations_all.json

Этап 2 — YOLO‑детекция объектов
- подготовка датасета
- анализ баланса классов
- обучение YOLOv9m
- оценка качества (mAP, Precision, Recall, F1)
- анализ confusion matrix
- подбор оптимального confidence threshold

Этап 3 — Классификация объектов (EfficientNet‑B4)
- подготовка кропов
- анализ размеров и распределения
- аугментации (RandomResizedCrop, flips, jitter, blur)
- обучение EfficientNet‑B4
- оценка качества (Accuracy, F1, ROC‑AUC)

Этап 4 — Детекция атрибутов и дефектов
- Включает 5 подэтапов:
  - SAM‑детектор пакетика
  - YOLO‑детектор пакетика
  - YOLO‑detect дефектов
  - YOLO‑CLS дефектов
  - YOLO‑Seg сегментация дефектов
- Результаты:
  - YOLO‑detect: высокая точность классификации дефектов
  - YOLO‑Seg: mAP50 ≈ 0.99, стабильная сегментация дефектов
- Полный пайплайн дефектов работает end‑to‑end

Этап 5 — Определение цвета объектов
- Smart Crop (SHRINK)
- Удаление бликов
- Извлечение цветовых признаков
- ML‑классификация цвета
  - kNN
  - RandomForest
  - XGBoost
  - LightGBM

---
## 📈 Результаты
- YOLOv9m детектирует объекты с точностью mAP50 = 0.995
- EfficientNet‑B4 классифицирует тип объекта с точностью 98–99%
- YOLO‑detect и YOLO‑Seg уверенно определяют дефекты
- ML‑модель цвета (LightGBM) достигает F1_macro = 0.956
- Построен полный промышленный CV‑пайплайн:
  - детекция → классификация → дефекты → сегментация → цвет

---
## 📁 Структура репозитория

`02_Detect_Cls_object/`

```text
│
├── `01_streamlit_ydb_EfficientNet.ipynb`    # Разметка + YOLO + EfficientNet
├── `02_sam_yolo_paketik_cls.ipynb`          # SAM, YOLO-detect, YOLO-CLS, YOLO-Seg
├── `03_detect_color.ipynb`                  # Полный ML-пайплайн цвета
│
├── `01_streamlit_ydb_EfficientNet.pdf`      # экспорт в PDF
├── `02_sam_yolo_paketik_cls.pdf`            # экспорт в PDF
├── `03_detect_color.pdf`                    # экспорт в PDF
├── `requirements.txt`                       # зависимости проекта
└── `README.md`                              # эта документация
```

---

## 🚀 Как запустить
1. Склонировать репозиторий:  
   ```bash
   git clone https://github.com/aspozd-DA-DS/Pet-projects.git
   ```
2. Перейти в папку проекта:
   ```bash
   cd Pet-projects/02_Detect_Cls_object
   ```
3. Запустить ноутбук:
   ```bash
   jupyter notebook 01_streamlit_ydb_EfficientNet.ipynb
   ```

   ```bash
   jupyter notebook 02_sam_yolo_paketik_cls.ipynb
   ```

   ```bash
   jupyter notebook 03_detect_color.ipynb
   ```

## 🏷 Topics
`Computer Vision` `YOLO` `EfficientNet` `Segmentation` `Machine Learning`  
`LightGBM` `XGBoost` `Color Detection` `Feature Engineering`  
`SAM` `OpenCV` `Deep Learning` `Image Classification`  
`Object Detection` `Python` `Data Science`

<p align="right"><a href="#top">⬆ наверх</a></p>

---

<a id="en"></a>

# Detection, classification and colour determination of objects in a transparent bag (YOLOv9, EfficientNet, ML)

## 📌 Description
A large pet project building a full computer vision pipeline that includes:
- detection of objects inside a transparent bag
- classification of the object type (capsule / softgel / meltlet / caplet)
- detection and classification of defects (chip, crack, leak, wrong_size, no_def)
- defect segmentation (YOLO‑Seg)
- colour determination of objects using ML models (LightGBM, XGBoost, RF, kNN)

The project combines detection, segmentation, classification, and feature engineering into a single industrial pipeline for analyzing objects in images.

Practical value:
- automating quality control of pharmaceutical products
- determining defects and colour without human involvement
- building a universal CV pipeline that can be integrated into production
- demonstrating skills in YOLO, EfficientNet, SAM, ML classification, feature engineering, and visualization

---

## 🔧 Tech stack

Computer Vision
- YOLOv9m, YOLOv8‑Seg, YOLO‑CLS, YOLO‑Detect
- Segment Anything Model (SAM ViT‑H)
- OpenCV, Albumentations

Deep Learning
- PyTorch, EfficientNet‑B3/B4
- AMP, cosine LR scheduler, AdamW

Machine Learning
- LightGBM, XGBoost, RandomForest, kNN
- KMeans, entropy‑based features, LAB/HSV colour analysis

Data & Visualization
- NumPy, Pandas
- Matplotlib, Seaborn

Infrastructure
- Streamlit (custom labelling tool)
- GPU acceleration

Fully reproducible Jupyter notebooks

---

## 📊 Data

The project uses several levels of data:
- Source images of bags
- YOLO annotations
- Object crops
- Defects
- Colour data
  - The pipeline produces:
    - cleaned crops
    - HSV/LAB representations
    - valid pixels
    - a feature table

---
## Key project stages

Stage 1 — Image labelling (Streamlit)
- a custom labelling application
- automatic detection of colour and shape
- saving YOLO + JSON annotations
- visualization of annotations
- building a single annotations_all.json

Stage 2 — YOLO object detection
- dataset preparation
- class balance analysis
- training YOLOv9m
- quality evaluation (mAP, Precision, Recall, F1)
- confusion matrix analysis
- selecting the optimal confidence threshold

Stage 3 — Object classification (EfficientNet‑B4)
- preparing crops
- analysis of sizes and distribution
- augmentations (RandomResizedCrop, flips, jitter, blur)
- training EfficientNet‑B4
- quality evaluation (Accuracy, F1, ROC‑AUC)

Stage 4 — Attribute and defect detection
- Includes 5 sub-stages:
  - SAM bag detector
  - YOLO bag detector
  - YOLO‑detect for defects
  - YOLO‑CLS for defects
  - YOLO‑Seg defect segmentation
- Results:
  - YOLO‑detect: high accuracy in defect classification
  - YOLO‑Seg: mAP50 ≈ 0.99, stable defect segmentation
- The full defect pipeline works end‑to‑end

Stage 5 — Object colour determination
- Smart Crop (SHRINK)
- Glare removal
- Extraction of colour features
- ML colour classification
  - kNN
  - RandomForest
  - XGBoost
  - LightGBM

---
## 📈 Results
- YOLOv9m detects objects with mAP50 = 0.995
- EfficientNet‑B4 classifies object type with 98–99% accuracy
- YOLO‑detect and YOLO‑Seg reliably identify defects
- The ML colour model (LightGBM) reaches F1_macro = 0.956
- A complete industrial CV pipeline was built:
  - detection → classification → defects → segmentation → colour

---
## 📁 Repository structure

`02_Detect_Cls_object/`

```text
│
├── `01_streamlit_ydb_EfficientNet.ipynb`    # Labelling + YOLO + EfficientNet
├── `02_sam_yolo_paketik_cls.ipynb`          # SAM, YOLO-detect, YOLO-CLS, YOLO-Seg
├── `03_detect_color.ipynb`                  # Full ML colour pipeline
│
├── `01_streamlit_ydb_EfficientNet.pdf`      # PDF export
├── `02_sam_yolo_paketik_cls.pdf`            # PDF export
├── `03_detect_color.pdf`                    # PDF export
├── `requirements.txt`                       # project dependencies
└── `README.md`                              # this documentation
```

---

## 🚀 How to run
1. Clone the repository:  
   ```bash
   git clone https://github.com/aspozd-DA-DS/Pet-projects.git
   ```
2. Go to the project folder:
   ```bash
   cd Pet-projects/02_Detect_Cls_object
   ```
3. Run the notebook:
   ```bash
   jupyter notebook 01_streamlit_ydb_EfficientNet.ipynb
   ```

   ```bash
   jupyter notebook 02_sam_yolo_paketik_cls.ipynb
   ```

   ```bash
   jupyter notebook 03_detect_color.ipynb
   ```

## 🏷 Topics
`Computer Vision` `YOLO` `EfficientNet` `Segmentation` `Machine Learning`  
`LightGBM` `XGBoost` `Color Detection` `Feature Engineering`  
`SAM` `OpenCV` `Deep Learning` `Image Classification`  
`Object Detection` `Python` `Data Science`

<p align="right"><a href="#top">⬆ back to top</a></p>
