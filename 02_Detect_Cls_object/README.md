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
│
├── `01_streamlit_ydb_EfficientNet.ipynb`      # Разметка + YOLO + EfficientNet
├── `02_sam_yolo_paketik_cls.ipynb`            # SAM, YOLO-detect, YOLO-CLS, YOLO-Seg
├── `03_detect_color.ipynb`                    # Полный ML-пайплайн цвета
│
├── `01_streamlit_ydb_EfficientNet.pdf`      # экспорт в PDF
├── `02_sam_yolo_paketik_cls.pdf`            # экспорт в PDF
├── `03_detect_color.pdf`                    # экспорт в PDF
│
└── `README.md                                

---

## 🚀 Как запустить
1. Склонировать репозиторий:  
   ```bash
   git clone https://github.com/aspozd-DA-DS/Pet-projects.git

2. Перейти в папку проекта:

   ```bash
   cd Pet-projects/02_Detect_Cls_object

3. Запустить ноутбук:

   ```bash
   jupyter notebook 03_detect_color.ipynb


## 🏷 Topics
`Computer Vision` `YOLO` `EfficientNet` `Segmentation` `Machine Learning`  
`LightGBM` `XGBoost` `Color Detection` `Feature Engineering`  
`SAM` `OpenCV` `Deep Learning` `Image Classification`  
`Object Detection` `Python` `Data Science`
