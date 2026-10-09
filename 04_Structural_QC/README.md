<a id="top"></a>

<p align="right">
  <a href="#ru">🇷🇺 RU</a> | <a href="#en">🇬🇧 EN</a>
</p>

---

<a id="ru"></a>

# **Structural QC — cross-plot & cross-validation (Oil & Gas)**

## 📌 Описание

Structural QC — интерактивный **Streamlit-дашборд** для контроля качества структурных построений по данным сейсморазведки МОГТ 2D/3D. Решает классическую задачу валидации структурной карты против фактических данных в скважинах: сравнивает **эталонный параметр** (например, глубину горизонта по скважине `Z_well`) с **сравниваемым** (глубину того же горизонта по структурной карте `Z_map`) в одних и тех же точках.

Главная методологическая проблема, которую решает инструмент — **data leakage в геологической форме**: если скважины использовались при построении карты (калибровка скоростной модели, привязка ВСП, интерполяция/крайгинг с учётом скважинных данных), то проверка качества карты в этих же точках даёт **завышенную, оптимистичную оценку**. Дашборд применяет полный набор схем кросс-валидации, чтобы честно оценить реальную предсказательную способность.

Инструмент реализован **обобщённо** и применим не только к паре «скважина ↔ карта», а к любой паре числовых параметров:

| Сценарий | PARAM_1 (Reference) / PARAM_2 (Compared) |
|---|---|
| скважина ↔ карта | Z_well / Z_map, T_well / T_map |
| скважина ↔ атрибут | Z_well / Amplitude |
| карта ↔ карта | Z_map_v1 / Z_map_v2 |
| атрибут ↔ атрибут | Amplitude / Sweetness |

Система автоматически:

- загружает CSV/XLSX и проводит полный QC входных данных (дубликаты, пропуски, типы, мусорные значения),
- строит базовый QC-кросс-плот с линией Y=X, линейной регрессией и полным набором метрик,
- накладывает **17 типов аппроксимаций** (полиномы 1–10 ст., степенная, экспонента, логарифм, Robust Huber, Theil-Sen, RANSAC, LOWESS) и **квантильную регрессию** (τ = 0.1 / 0.5 / 0.9),
- честно оценивает предсказательную способность через **8 схем кросс-валидации** (Random K-Fold, LOWO, Block, Buffered, Leave-P-Out, Spatial Leave-P-Out, Stratified by Zone, Distance-based exclusion CV),
- автоматически определяет лучшую схему CV по критерию `score = RMSE + RMSE_fold_std` среди всех восьми схем, **включая Random K-Fold**: он участвует в сравнении как контрольная, заведомо оптимистичная схема, поэтому при интерпретации лучшей схемы учитывайте, что его результат может быть завышен (см. «Ограничения»),
- анализирует невязки, строит **карту невязок** в плане и детектирует выбросы **4 независимыми методами**,
- проводит диагностику по категориям (горизонты, тектонические зоны, тип коллектора и т.д.),
- позволяет в интерактиве фильтровать скважины для стресс-теста устойчивости,
- поддерживает **двуязычный интерфейс RU/EN**.

Практическая ценность:

- объективная оценка точности структурной карты без утечки данных,
- выявление зон систематического смещения (ошибки скоростной модели, тектонические осложнения),
- поиск аномальных скважин для перепроверки,
- обоснование решения о доработке карты перед бурением,
- сравнение нескольких версий интерпретации,
- оценка переносимости модели на новые (необуренные) площади.

---

## 🔧 Стек технологий

### Интерфейс и визуализация
- Streamlit  
- Plotly (graph_objects, express)  
- Seaborn  
- Matplotlib (ноутбук)  

### Обработка данных
- NumPy  
- Pandas  
- OpenPyXL, xlrd  

### ML / статистика
- scikit-learn (LinearRegression, HuberRegressor, TheilSenRegressor, RANSACRegressor, PolynomialFeatures, KFold, LeaveOneGroupOut, r2_score)  
- statsmodels (QuantReg, lowess)  
- SciPy (stats.linregress, spatial.distance.pdist/cdist, gaussian_kde)  

### Архитектура
- Модульная структура: ядро расчётов, утилиты, сценарий UI  
- Кэширование тяжёлых вычислений через `@st.cache_data` с `hash_funcs`  
- Двуязычный интерфейс через словари `RU` / `EN`  

---

## 📊 Данные

На вход принимается CSV/XLSX с обязательными и опциональными колонками:

| Колонка | Роль | Описание |
|---|---|---|
| `well_id` | обязательная | Идентификатор скважины |
| `X_coord`, `Y_coord` | обязательные | Координаты скважины |
| `PARAM_1` (Zprm_well) | обязательная · Reference | Эталонный параметр (например, Z_well — глубина горизонта по скважине) |
| `PARAM_2` (Zprm_map) | обязательная · Compared | Сравниваемый параметр (например, Z_map — глубина того же горизонта по структурной карте) |
| `horizon_id` | опциональная | Идентификатор горизонта — фильтрация анализа |
| `tectonic_zone` | опциональная | Тектоническая зона — используется в Stratified by Zone CV и Шаге 3 |
| любая другая | опциональная | Числовые — в корреляционный анализ; категориальные — в анализ по категориям |

> **Демонстрационные файлы в `data/` синтетические.** Они созданы для показа возможностей инструмента и не отражают реальный участок или реальные скважины. Для собственных данных используйте файлы в формате, описанном выше.

**Автоматическая очистка при загрузке:**

- приведение мусорных значений (`NA`, `NULL`, `---`, `?`, `*` и т.п.) к NaN,
- замена запятой на точку в числовых колонках,
- приведение типов,
- удаление строк с пропусками в обязательных колонках с отчётом о количестве удалённых.

**Результаты пайплайна:**

- очищенный датасет с системными колонками,
- метрики QC-кросс-плота,
- таблица метрик всех аппроксимаций,
- сводная таблица метрик всех схем CV,
- распределение и статистика невязок,
- список скважин-выбросов по 4 методам,
- разбивка точности по категориям,
- сравнение «до / после фильтрации скважин».

---

## 🧩 Ключевые этапы проекта

### Шаг 0 — Загрузка и QC данных
- загрузка CSV/XLSX или выбор demo-файла,
- сопоставление колонок,
- автоматическая очистка,
- QC входных данных (полные дубликаты, дубликаты `well_id`, совпадающие координаты, пропуски, удалённые строки),
- типы данных, категориальные признаки,
- описательная статистика `PARAM_1` и `PARAM_2`,
- корреляционный анализ (Pearson + Spearman, Pairplot Seaborn / Plotly).

### Шаг 1 — Базовый QC-кросс-плот
- кросс-плот с линией Y=X и линейной регрессией,
- метрики QC: R²_lin, RMSE, MSE, MAE, Bias, Std, Variance, MAPE %, P-value, уравнение регрессии,
- 17 типов аппроксимаций и квантильная регрессия (τ = 0.1 / 0.5 / 0.9) со сравнительной таблицей метрик (R²_fit; для каждой τ метрики считаются отдельно),
- распределения `PARAM_1` и `PARAM_2` (гистограммы с KDE, boxplot),
- анализ невязок Δ = PARAM_1 − PARAM_2: график от `PARAM_1`, таблица статистик, гистограмма, boxplot,
- **карта невязок** в плане (X_coord, Y_coord, цвет — величина невязки),
- **детекция выбросов 4 методами**: Sigma rule, IQR rule, Modified Z-score, Percentile rule.

### Шаг 2 — Кросс-валидация
- 8 схем CV с автоматическим подбором рекомендуемых параметров в сайдбаре,
- для каждой схемы: кросс-плот Predicted vs Actual, метрики, статистика по фолдам,
- сводная таблица всех схем + bar chart RMSE,
- автоматический выбор лучшей схемы по `score = RMSE + RMSE_fold_std` (Random K-Fold участвует в сравнении как контрольная схема),
- детальный разбор лучшей схемы: кросс-плот, аппроксимации, распределение CV-невязок, выбросы.

### Шаг 3 — Диагностика по категориям
- работает для любой категориальной колонки (`horizon_id`, `tectonic_zone`, `reservoir_flag`, `fluid_type` и др.),
- круговая и столбчатая диаграммы количества скважин по категориям,
- описательная статистика по каждой категории,
- boxplot распределения параметров по категориям,
- общий кросс-плот по всем категориям + отдельно по подмножеству,
- кросс-плот и аппроксимации отдельно для каждой категории,
- сравнительная таблица метрик аппроксимаций по категориям.

### Шаг 4 — Фильтрация скважин (стресс-тест)
- 4 режима: без фильтра, ручное исключение, случайное исключение (%), выделение кликом по кросс-плоту,
- пересчёт QC-кросс-плота, аппроксимаций и полной таблицы метрик после фильтрации,
- сравнение «до / после»: n скважин, R²_lin, R²_fit, RMSE, MSE, MAE, Bias, Std, Variance, MAPE %.

---

## 📈 Результаты

### Что получает пользователь

- **Честная оценка точности** структурной карты без утечки данных,
- **Диагноз отклонений**: где модель завышает, где занижает, есть ли тренд невязки от глубины,
- **Список аномальных скважин** (кандидаты на перепроверку координат, альтитуды, привязки),
- **Разбивка точности по категориям** — в каких горизонтах / зонах карта «едет»,
- **Оценка устойчивости** к сокращению выборки и к отдельным наблюдениям,
- **Автоматически выбранная лучшая схема CV** как ориентир для дальнейших работ (с учётом того, что Random K-Fold участвует в сравнении и может давать оптимистичную оценку).

### Формулы метрик

| Метрика | Формула |
|---|---|
| R²_lin | `r²` (квадрат Пирсона, `linregress`) |
| R²_fit | `1 − Σ(Δ)² / Σ(y − ȳ)²` (для любой формы кривой) |
| RMSE | `√[ (1/N)·Σ(Δ)² ]` |
| MSE | `(1/N)·Σ(Δ)²` |
| MAE | `(1/N)·Σ\|Δ\|` |
| Bias | `(1/N)·Σ(Δ)` |
| Std | `√[ (1/(N−1))·Σ(Δᵢ − Δ̄)² ]` |
| Variance | `(1/(N−1))·Σ(Δᵢ − Δ̄)²` |
| MAPE, % | `(100/N)·Σ[ \|Δ\| / max(\|y\|, 10⁻⁹) ]` |
| P-value | `t = r·√[(N−2)/(1−r²)]`, `p = 2·(1 − F_t,N−2(\|t\|))` |

---

## ⚠️ Ограничения и допущения

- **Distance-based exclusion CV** — это эвристика по расстояниям: радиус корреляции оценивается как медианное попарное расстояние между скважинами, умноженное на `variogram_factor`. Это не полноценная геостатистическая вариограмма (nugget / sill / range) и не подгонка модели вариограммы.
- **Spatial CV (Block, Buffered, Spatial Leave-P-Out)** требует хорошего пространственного покрытия площади скважинами. На редкой или сильно кластеризованной сетке отдельные блоки или окна могут оказаться пустыми или содержать мало обучающих скважин.
- **При малом числе скважин (меньше 15–20)** практически единственный осмысленный вариант — Leave-One-Well-Out; остальные схемы дают нестабильные оценки.
- **Random K-Fold** участвует в сравнении схем как контрольная, заведомо оптимистичная схема, поскольку игнорирует пространственную корреляцию. Его результат не следует использовать как оценку реальной точности.
- **Выделение точек кликом** на cross-plot доступно только в интерактивном приложении (`app_full.py`); в Jupyter-ноутбуке такой режим не воспроизводится, там фильтрация задаётся списком или процентом.
- **Выводы зависят от данных.** Инструмент даёт диагностику и ориентиры, но не заменяет геологическую интерпретацию и проверку на реальном наборе скважин.

## 📁 Структура репозитория

```text
04_Structural_QC/
│
├── data/                             — демонстрационные файлы (demo datasets)
│   └── *.csv, *.xlsx                 — примеры для быстрого старта
├── test_data_import/                 — файлы  для проверки загрузки данных локально
│   └── *.csv, *.xlsx                 — примеры для быстрого старта
│
├── app_full.py                       — главный Streamlit-сценарий (UI, sidebar, шаги 0–4)
├── app_utils.py                      — утилиты расчётов и графиков
├── structural_qc_full.py             — ядро расчётов и кросс-валидации
├── translations_full.py              — словари RU/EN
├── ui_examples_full_ru.py            — справка по проекту (RU)
├── ui_examples_full_en.py            — справка по проекту (EN)
│
├── structural_qc_demo.ipynb          — демонстрационный ноутбук
│
├── structural_qc_demo.pdf            — демонстрационный ноутбук  в pdf
│
├── structural_qc_plan.pdf            — план проекта
│
├── structural_qc_streamlit_demo.pdf  — визуализация интерфейса
│
├── requirements.txt                  — зависимости проекта
└── README.md                         — документация проекта

```

---

## 🚀 Как запустить

### Локальный запуск

1. Склонировать репозиторий:
   ```bash
   git clone https://github.com/aspozd-DA-DS/Pet-projects.git
   ```

2. Перейти в папку проекта:
   ```bash
   cd Pet-projects/04_Structural_QC
   ```

3. (Рекомендуется) создать виртуальное окружение и активировать:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   .venv\Scripts\activate         # Windows
   ```

4. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```

5. Запустить приложение:
   ```bash
   streamlit run app_full.py
   ```

6. Открыть в браузере: `http://localhost:8501`

### Браузер

https://structural-qc-cv.streamlit.app/

---

## 📋 Требования к входному файлу

**Обязательные колонки:**
- `well_id` — идентификатор скважины,
- `X_coord`, `Y_coord` — координаты,
- `PARAM_1` — эталонный параметр,
- `PARAM_2` — сравниваемый параметр.

**Опциональные колонки:**
- `horizon_id` — горизонт,
- `tectonic_zone` — тектоническая зона,
- `reservoir_flag` — тип коллектора,
- `fluid_type` — тип флюида,
- любые дополнительные числовые или категориальные колонки.

**Пример содержимого:**

| well_id | X_coord | Y_coord | Z_well | Z_map | horizon_id | tectonic_zone |
|---|---|---|---|---|---|---|
| Well_001 | 125000.0 | 745000.0 | 3120.5 | 3108.7 | H1 | Zone_A |
| Well_002 | 126500.0 | 746200.0 | 3188.2 | 3201.5 | H1 | Zone_A |
| Well_003 | 128200.0 | 748100.0 | 3256.4 | 3242.1 | H1 | Zone_B |

---

## 🏷 Topics

`Structural QC` `Cross-Plot` `Cross-Validation` `Data Leakage` `Spatial CV`
`LOWO` `Block CV` `Buffered CV` `Spatial Leave-P-Out` `Distance-based CV`
`Geology` `Seismic` `Oil & Gas` `Well Data` `Structural Map`
`Streamlit` `Plotly` `Python` `Pandas` `NumPy` `scikit-learn` `statsmodels`
`Machine Learning` `Data Science` `QC` `Regression` `Residual Analysis`
`Outlier Detection` `Bilingual UI` `RU/EN`

<p align="right"><a href="#top">⬆ наверх</a></p>

---

<a id="en"></a>

# **Structural QC — cross-plot & cross-validation (Oil & Gas)**

## 📌 Description

Structural QC is an interactive **Streamlit dashboard** for quality control of structural mapping based on 2D/3D CDP (MOGT) seismic data. It addresses the classic task of validating a structural map against actual well data: it compares a **reference parameter** (for example, the depth of a horizon from a well, `Z_well`) with a **compared parameter** (the depth of the same horizon from a structural map, `Z_map`) at the same points.

The main methodological problem the tool addresses is **data leakage in the geological form**: if wells were used to build the map (calibration of the velocity model, VSP tie-in, interpolation/kriging that takes well data into account), then checking the map's quality at those same points gives an **inflated, optimistic estimate**. The dashboard applies a full set of cross-validation schemes to honestly estimate the real predictive ability.

The tool is implemented **generically** and applies not only to the "well ↔ map" pair, but to any pair of numeric parameters:

| Scenario | PARAM_1 (Reference) / PARAM_2 (Compared) |
|---|---|
| well ↔ map | Z_well / Z_map, T_well / T_map |
| well ↔ attribute | Z_well / Amplitude |
| map ↔ map | Z_map_v1 / Z_map_v2 |
| attribute ↔ attribute | Amplitude / Sweetness |

The system automatically:

- loads CSV/XLSX and performs a full QC of the input data (duplicates, missing values, types, garbage values),
- builds a base QC cross-plot with the Y=X line, linear regression, and a full set of metrics,
- overlays **17 fitting types** (polynomials of degree 1–10, power, exponential, logarithmic, Robust Huber, Theil-Sen, RANSAC, LOWESS) and **quantile regression** (τ = 0.1 / 0.5 / 0.9),
- honestly estimates predictive ability through **8 cross-validation schemes** (Random K-Fold, LOWO, Block, Buffered, Leave-P-Out, Spatial Leave-P-Out, Stratified by Zone, Distance-based exclusion CV),
- automatically selects the best CV scheme by the criterion `score = RMSE + RMSE_fold_std` among all eight schemes, **including Random K-Fold**: it takes part in the comparison as a control scheme that is known to be optimistic, so when interpreting the best scheme, keep in mind that its result may be inflated (see "Limitations"),
- analyzes residuals, builds a **residual map** in plan view, and detects outliers with **4 independent methods**,
- performs diagnostics by category (horizons, tectonic zones, reservoir type, etc.),
- lets the user interactively filter wells for a stress test of stability,
- supports a **bilingual RU/EN interface**.

Practical value:

- an objective estimate of structural map accuracy without data leakage,
- identifying zones of systematic bias (velocity model errors, tectonic complications),
- finding anomalous wells for re-checking,
- justifying a decision to revise the map before drilling,
- comparing several versions of an interpretation,
- assessing the transferability of the model to new (undrilled) areas.

---

## 🔧 Tech stack

### Interface and visualization
- Streamlit  
- Plotly (graph_objects, express)  
- Seaborn  
- Matplotlib (notebook)  

### Data processing
- NumPy  
- Pandas  
- OpenPyXL, xlrd  

### ML / statistics
- scikit-learn (LinearRegression, HuberRegressor, TheilSenRegressor, RANSACRegressor, PolynomialFeatures, KFold, LeaveOneGroupOut, r2_score)  
- statsmodels (QuantReg, lowess)  
- SciPy (stats.linregress, spatial.distance.pdist/cdist, gaussian_kde)  

### Architecture
- Modular structure: calculation core, utilities, UI script  
- Caching of heavy computations via `@st.cache_data` with `hash_funcs`  
- Bilingual interface via `RU` / `EN` dictionaries  

---

## 📊 Data

The input is a CSV/XLSX file with required and optional columns:

| Column | Role | Description |
|---|---|---|
| `well_id` | required | Well identifier |
| `X_coord`, `Y_coord` | required | Well coordinates |
| `PARAM_1` (Zprm_well) | required · Reference | Reference parameter (for example, Z_well — horizon depth from the well) |
| `PARAM_2` (Zprm_map) | required · Compared | Compared parameter (for example, Z_map — depth of the same horizon from the structural map) |
| `horizon_id` | optional | Horizon identifier — analysis filtering |
| `tectonic_zone` | optional | Tectonic zone — used in Stratified by Zone CV and Step 3 |
| any other | optional | Numeric — into correlation analysis; categorical — into category analysis |

> **The demo files in `data/` are synthetic.** They were created to demonstrate the tool's capabilities and do not reflect a real field or real wells. For your own data, use files in the format described above.

**Automatic cleaning on load:**

- converting garbage values (`NA`, `NULL`, `---`, `?`, `*`, etc.) to NaN,
- replacing commas with dots in numeric columns,
- type conversion,
- removing rows with missing values in required columns, with a report on how many rows were removed.

**Pipeline results:**

- cleaned dataset with system columns,
- QC cross-plot metrics,
- table of metrics for all fits,
- summary table of metrics for all CV schemes,
- distribution and statistics of residuals,
- list of outlier wells from 4 methods,
- accuracy breakdown by category,
- "before / after well filtering" comparison.

---

## 🧩 Key project stages

### Step 0 — Data loading and QC
- loading CSV/XLSX or choosing a demo file,
- column mapping,
- automatic cleaning,
- QC of the input data (full duplicates, duplicate `well_id`, matching coordinates, missing values, removed rows),
- data types, categorical features,
- descriptive statistics of `PARAM_1` and `PARAM_2`,
- correlation analysis (Pearson + Spearman, Seaborn / Plotly pairplot).

### Step 1 — Base QC cross-plot
- cross-plot with the Y=X line and linear regression,
- QC metrics: R²_lin, RMSE, MSE, MAE, Bias, Std, Variance, MAPE %, P-value, regression equation,
- 17 fitting types and quantile regression (τ = 0.1 / 0.5 / 0.9) with a comparison table of metrics (R²_fit; metrics are computed separately for each τ),
- distributions of `PARAM_1` and `PARAM_2` (histograms with KDE, boxplot),
- residual analysis Δ = PARAM_1 − PARAM_2: plot against `PARAM_1`, statistics table, histogram, boxplot,
- **residual map** in plan view (X_coord, Y_coord, colour — residual magnitude),
- **outlier detection with 4 methods**: Sigma rule, IQR rule, Modified Z-score, Percentile rule.

### Step 2 — Cross-validation
- 8 CV schemes with automatic selection of recommended parameters in the sidebar,
- for each scheme: Predicted vs Actual cross-plot, metrics, per-fold statistics,
- summary table of all schemes + RMSE bar chart,
- automatic selection of the best scheme by `score = RMSE + RMSE_fold_std` (Random K-Fold takes part in the comparison as a control scheme),
- detailed review of the best scheme: cross-plot, fits, distribution of CV residuals, outliers.

### Step 3 — Diagnostics by category
- works for any categorical column (`horizon_id`, `tectonic_zone`, `reservoir_flag`, `fluid_type`, etc.),
- pie and bar charts of the number of wells per category,
- descriptive statistics for each category,
- boxplots of parameter distributions by category,
- overall cross-plot for all categories + separately for a subset,
- cross-plot and fits separately for each category,
- comparison table of fit metrics by category.

### Step 4 — Well filtering (stress test)
- 4 modes: no filter, manual exclusion, random exclusion (%), selection by clicking on the cross-plot,
- recalculation of the QC cross-plot, fits, and the full metrics table after filtering,
- "before / after" comparison: number of wells, R²_lin, R²_fit, RMSE, MSE, MAE, Bias, Std, Variance, MAPE %.

---

## 📈 Results

### What the user gets

- **An honest estimate of accuracy** of the structural map without data leakage,
- **A diagnosis of deviations**: where the model overestimates or underestimates, and whether there is a trend of residuals with depth,
- **A list of anomalous wells** (candidates for re-checking of coordinates, altitudes, and tie-ins),
- **Accuracy breakdown by category** — in which horizons / zones the map "drifts",
- **An assessment of stability** to reduction of the sample and to individual observations,
- **The automatically selected best CV scheme** as a reference for further work (taking into account that Random K-Fold takes part in the comparison and may give an optimistic estimate).

### Metric formulas

| Metric | Formula |
|---|---|
| R²_lin | `r²` (squared Pearson correlation, `linregress`) |
| R²_fit | `1 − Σ(Δ)² / Σ(y − ȳ)²` (for any curve shape) |
| RMSE | `√[ (1/N)·Σ(Δ)² ]` |
| MSE | `(1/N)·Σ(Δ)²` |
| MAE | `(1/N)·Σ\|Δ\|` |
| Bias | `(1/N)·Σ(Δ)` |
| Std | `√[ (1/(N−1))·Σ(Δᵢ − Δ̄)² ]` |
| Variance | `(1/(N−1))·Σ(Δᵢ − Δ̄)²` |
| MAPE, % | `(100/N)·Σ[ \|Δ\| / max(\|y\|, 10⁻⁹) ]` |
| P-value | `t = r·√[(N−2)/(1−r²)]`, `p = 2·(1 − F_t,N−2(\|t\|))` |

---

## ⚠️ Limitations and assumptions

- **Distance-based exclusion CV** is a distance-based heuristic: the correlation radius is estimated as the median pairwise distance between wells, multiplied by `variogram_factor`. This is not a full geostatistical variogram (nugget / sill / range) and does not fit a variogram model.
- **Spatial CV (Block, Buffered, Spatial Leave-P-Out)** requires good spatial coverage of the area by wells. On a sparse or strongly clustered grid, some blocks or windows may be empty or contain few training wells.
- **With a small number of wells (fewer than 15–20)** the only practically meaningful option is Leave-One-Well-Out; the other schemes give unstable estimates.
- **Random K-Fold** takes part in the comparison of schemes as a control, knowingly optimistic scheme, because it ignores spatial correlation. Its result should not be used as an estimate of real accuracy.
- **Selecting points by clicking** on the cross-plot is available only in the interactive application (`app_full.py`); this mode is not reproduced in the Jupyter notebook, where filtering is set by a list or a percentage.
- **Conclusions depend on the data.** The tool provides diagnostics and reference points, but does not replace geological interpretation and validation on a real set of wells.

## 📁 Repository structure

```text
04_Structural_QC/
│
├── data/                             — demo files (demo datasets)
│   └── *.csv, *.xlsx                 — examples for a quick start
├── test_data_import/                 — files for local data-import testing
│   └── *.csv, *.xlsx                 — examples for a quick start
│
├── app_full.py                       — main Streamlit script (UI, sidebar, steps 0–4)
├── app_utils.py                      — calculation and plotting utilities
├── structural_qc_full.py             — calculation and cross-validation core
├── translations_full.py              — RU/EN dictionaries
├── ui_examples_full_ru.py            — project reference (RU)
├── ui_examples_full_en.py            — project reference (EN)
│
├── structural_qc_demo.ipynb          — demo notebook
│
├── structural_qc_demo.pdf            — demo notebook as PDF
│
├── structural_qc_plan.pdf            — project plan
│
├── structural_qc_streamlit_demo.pdf  — interface visualization
│
├── requirements.txt                  — project dependencies
└── README.md                         — project documentation

```

---

## 🚀 How to run

### Local run

1. Clone the repository:
   ```bash
   git clone https://github.com/aspozd-DA-DS/Pet-projects.git
   ```

2. Go to the project folder:
   ```bash
   cd Pet-projects/04_Structural_QC
   ```

3. (Recommended) create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   .venv\Scripts\activate         # Windows
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   streamlit run app_full.py
   ```

6. Open in the browser: `http://localhost:8501`

### Browser

https://structural-qc-cv.streamlit.app/

---

## 📋 Input file requirements

**Required columns:**
- `well_id` — well identifier,
- `X_coord`, `Y_coord` — coordinates,
- `PARAM_1` — reference parameter,
- `PARAM_2` — compared parameter.

**Optional columns:**
- `horizon_id` — horizon,
- `tectonic_zone` — tectonic zone,
- `reservoir_flag` — reservoir type,
- `fluid_type` — fluid type,
- any additional numeric or categorical columns.

**Example content:**

| well_id | X_coord | Y_coord | Z_well | Z_map | horizon_id | tectonic_zone |
|---|---|---|---|---|---|---|
| Well_001 | 125000.0 | 745000.0 | 3120.5 | 3108.7 | H1 | Zone_A |
| Well_002 | 126500.0 | 746200.0 | 3188.2 | 3201.5 | H1 | Zone_A |
| Well_003 | 128200.0 | 748100.0 | 3256.4 | 3242.1 | H1 | Zone_B |

---

## 🏷 Topics

`Structural QC` `Cross-Plot` `Cross-Validation` `Data Leakage` `Spatial CV`
`LOWO` `Block CV` `Buffered CV` `Spatial Leave-P-Out` `Distance-based CV`
`Geology` `Seismic` `Oil & Gas` `Well Data` `Structural Map`
`Streamlit` `Plotly` `Python` `Pandas` `NumPy` `scikit-learn` `statsmodels`
`Machine Learning` `Data Science` `QC` `Regression` `Residual Analysis`
`Outlier Detection` `Bilingual UI` `RU/EN`

<p align="right"><a href="#top">⬆ back to top</a></p>
