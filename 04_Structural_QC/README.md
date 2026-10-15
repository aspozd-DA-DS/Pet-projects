<a id="top"></a>

<p align="right">
  <a href="#ru">🇷🇺 RU</a> | <a href="#en">🇬🇧 EN</a>
</p>

---

<a id="ru"></a>


# Structural QC — кросс-плот и кросс-валидация (Oil & Gas)

**Ссылки:** [Онлайн-приложение](https://structural-qc-cv.streamlit.app/) 

## 📌 Описание

Structural QC — интерактивный **Streamlit-дашборд** для контроля качества структурных построений по данным сейсморазведки МОГТ 2D/3D. Он решает классическую задачу валидации структурной карты по фактическим данным в скважинах: сравнивает **эталонный параметр** (например, глубину горизонта по скважине `Z_well`) со **сравниваемым** (глубиной того же горизонта по структурной карте `Z_map`) в одних и тех же точках.

Главная методологическая проблема, которую решает инструмент, — **data leakage в геологической форме**. Если скважины использовались при построении карты (калибровка скоростной модели, привязка ВСП, интерполяция/крайгинг с учётом скважинных данных), то проверка карты в этих же точках даёт **завышенную, оптимистичную оценку**. Дашборд применяет полный набор схем кросс-валидации, чтобы честно оценить реальную предсказательную способность.

Инструмент реализован **обобщённо**: он применим не только к паре «скважина ↔ карта», а к любой паре числовых параметров:

| Сценарий | PARAM_1 (Reference) / PARAM_2 (Compared) |
|---|---|
| скважина ↔ карта | Z_well / Z_map, T_well / T_map |
| скважина ↔ атрибут | Z_well / Amplitude |
| карта ↔ карта | Z_map_v1 / Z_map_v2 |
| атрибут ↔ атрибут | Amplitude / Sweetness |

### Что делает инструмент

- загружает CSV/XLSX и проводит полный QC входных данных (дубликаты, пропуски, типы, мусорные значения);
- строит базовый QC-кросс-плот с линией Y = X, линейной регрессией и полным набором метрик;
- накладывает **17 типов аппроксимаций** (полиномы 1–10 степени, степенная, экспонента, логарифм, Huber, Theil-Sen, RANSAC, LOWESS) и **квантильную регрессию** (τ = 0.1 / 0.5 / 0.9);
- оценивает предсказательную способность по **8 схемам кросс-валидации**: Random K-Fold, Leave-One-Well-Out, Block, Buffered, Leave-P-Out, Spatial Leave-P-Out, Stratified by Zone, Distance-based exclusion CV;
- автоматически выбирает лучшую схему CV по критерию `score = RMSE + RMSE_fold_std` среди всех восьми схем, **включая Random K-Fold**: он участвует в сравнении как контрольная схема, заведомо оптимистичная (см. «Ограничения»);
- анализирует невязки, строит **карту невязок** в плане и выявляет выбросы **четырьмя независимыми методами**;
- проводит диагностику по категориям (горизонты, тектонические зоны, тип коллектора, тип флюида и т.д.);
- позволяет интерактивно фильтровать скважины для стресс-теста устойчивости;
- поддерживает **двуязычный интерфейс RU/EN**;
- позволяет настраивать в сайдбаре цвета, стили линий, число интервалов и нормировку гистограмм.

### Практическая ценность

- объективная оценка точности структурной карты без утечки данных;
- выявление зон систематического смещения (ошибки скоростной модели, тектонические осложнения);
- поиск аномальных скважин для перепроверки координат, альтитуд и привязок;
- обоснование решения о доработке карты до бурения;
- сравнение нескольких версий интерпретации;
- оценка переносимости модели на новые, необуренные площади.

---

## 🧭 Ключевые решения метода

- **Модель:** во всех шагах, включая кросс-валидацию, используется обычная линейная регрессия (полином степени 1). Аппроксимации более высокой степени доступны только как кривые для визуального сравнения, а не как модель, которая участвует в CV.
- **Знак невязки:** в базовом QC-кросс-плоте Δ = PARAM_1 − PARAM_2. Метрики аппроксимаций и CV считаются по Predicted − Actual. Графики CV-невязок в Шаге 2 строятся как PARAM_1 − Predicted. Это разные соглашения намеренно; знак указан на каждом графике.
- **Выбор лучшей схемы:** `score = RMSE + RMSE_fold_std` минимизируется по всем восьми схемам. Random K-Fold включён намеренно, чтобы контраст с пространственными схемами оставался видимым. Его оптимистичный результат может выиграть выбор, поэтому «лучшую» схему интерпретируйте вместе с пространственными.
- **Distance-based exclusion CV:** радиус исключения `excl_radius = медианное попарное расстояние между скважинами × variogram_factor`. Это эвристика по расстояниям, а не подогнанная вариограмма.

---

## 🔧 Стек технологий

### Интерфейс и визуализация
- Streamlit
- Plotly (graph_objects, express)
- Seaborn (pairplot)
- Matplotlib (ноутбук)

### Обработка данных
- NumPy
- Pandas
- OpenPyXL, xlrd

### ML и статистика
- scikit-learn (LinearRegression, HuberRegressor, TheilSenRegressor, RANSACRegressor, PolynomialFeatures, KFold, LeaveOneGroupOut, r2_score)
- statsmodels (QuantReg, lowess)
- SciPy (stats.linregress, spatial.distance.pdist / cdist, gaussian_kde)

### Архитектура
- модульная структура: ядро расчётов (`structural_qc_full.py`), утилиты и графики (`app_utils.py`), сценарий UI (`app_full.py`), словари переводов (`translations_full.py`);
- тяжёлые вычисления кэшируются через `@st.cache_data` (кросс-валидация, аппроксимации, квантильные подгонки, статистика расстояний);
- двуязычный интерфейс через словари `RU` / `EN`.

---

## 📊 Данные

На вход принимается CSV или XLSX со следующими колонками.

| Колонка | Роль | Описание |
|---|---|---|
| `well_id` | обязательная | Идентификатор скважины |
| `X_coord`, `Y_coord` | обязательные | Координаты скважины |
| `PARAM_1` (Zprm_well) | обязательная · Reference | Эталонный параметр, например Z_well — глубина горизонта по скважине |
| `PARAM_2` (Zprm_map) | обязательная · Compared | Сравниваемый параметр, например Z_map — глубина того же горизонта по структурной карте |
| `horizon_id` | опциональная | Идентификатор горизонта — фильтрация анализа |
| `tectonic_zone` | опциональная | Тектоническая зона — используется в Stratified by Zone CV и Шаге 3 |
| `reservoir_flag`, `fluid_type` | опциональные | Категориальные признаки для Шага 3 |
| любая другая | опциональная | Числовые — в корреляционный анализ; категориальные — в анализ по категориям |

> **Демонстрационные файлы в `data/` синтетические.** Они созданы для показа возможностей инструмента и не описывают реальный участок или реальные скважины. Для собственных данных используйте файлы в формате, описанном выше.

**Автоматическая очистка при загрузке:**
- мусорные значения (`NA`, `NULL`, `---`, `?`, `*` и т.п.) превращаются в NaN;
- запятая в числовых колонках заменяется на точку;
- типы колонок приводятся к нужным;
- строки с пропусками в обязательных колонках удаляются, а количество удалённых строк выводится в отчёт.

**Результаты пайплайна:**
- очищенный датасет с системными колонками;
- метрики QC-кросс-плота;
- таблица метрик всех аппроксимаций;
- сводная таблица метрик всех схем CV;
- распределение и статистика невязок;
- список скважин-выбросов по 4 методам;
- разбивка точности по категориям;
- сравнение «до / после» фильтрации скважин.

---

## 🧩 Ход работы

### Шаг 0 — Загрузка и QC данных
- загрузка CSV/XLSX или выбор демо-файла;
- сопоставление колонок;
- автоматическая очистка;
- QC входных данных: полные дубликаты, дубликаты `well_id` (реальный дубликат — это совпадение `well_id` и `horizon_id`), совпадающие координаты, пропуски, удалённые строки;
- типы данных и категориальные признаки;
- описательная статистика `PARAM_1` и `PARAM_2`;
- статистика по каждой числовой колонке: таблица, гистограмма с KDE, среднее, медиана и ±3σ, boxplot; число интервалов и нормировка задаются для каждой колонки отдельно;
- корреляционный анализ: Пирсон (только линейная связь) и Спирмен (любая монотонная связь), а также pairplot (Seaborn или Plotly; для pairplot рекомендуется не более 6 параметров).

### Шаг 1 — Базовый QC-кросс-плот
- кросс-плот с линией Y = X и линейной регрессией (степень 1);
- метрики QC: R²_lin, RMSE, MSE, MAE, Bias, Std, Variance, MAPE %, P-value, уравнение регрессии;
- 17 аппроксимаций и квантильная регрессия (τ = 0.1 / 0.5 / 0.9) со сравнительной таблицей R²_fit; метрики для каждой τ считаются отдельно;
- распределения `PARAM_1` и `PARAM_2` (гистограммы с KDE, boxplot);
- анализ невязок Δ = PARAM_1 − PARAM_2: график от `PARAM_1`, таблица статистик, гистограмма, boxplot;
- **карта невязок** в плане (X_coord, Y_coord, цвет — величина невязки);
- **детекция выбросов 4 методами** (см. раздел «Детекция выбросов» ниже).

### Шаг 2 — Кросс-валидация
- 8 схем CV с рекомендуемыми параметрами, подбираемыми автоматически в сайдбаре;
- для каждой схемы: кросс-плот Predicted vs Actual, метрики, статистика по фолдам;
- сводная таблица всех схем, столбчатые диаграммы RMSE и среднего RMSE по фолдам;
- boxplot разброса RMSE по фолдам для каждой схемы (медиана, среднее, выбросы); порядок схем совпадает с таблицей;
- автоматический выбор лучшей схемы по `score = RMSE + RMSE_fold_std` (Random K-Fold участвует как контрольная схема);
- детальный разбор лучшей схемы: кросс-плот, аппроксимации, распределение CV-невязок, выбросы.

### Шаг 3 — Диагностика по категориям
- работает для любой категориальной колонки (`horizon_id`, `tectonic_zone`, `reservoir_flag`, `fluid_type` и др.);
- круговая и столбчатая диаграммы количества скважин по категориям;
- описательная статистика по каждой категории;
- boxplot распределения параметров по категориям;
- наложенные гистограммы по категориям для `PARAM_1` и `PARAM_2`;
- общий кросс-плот по всем категориям и отдельно по выбранному подмножеству;
- кросс-плот и аппроксимации отдельно для каждой категории;
- сравнительная таблица метрик аппроксимаций по категориям.

### Шаг 4 — Фильтрация скважин (стресс-тест)
- четыре режима: без фильтра, ручное исключение скважин, случайное исключение по проценту (0–95%, с пересэмплированием) и **выделение точек кликом** на кросс-плоте (клик по точке, Shift+клик — несколько, рамка Box или лассо Lasso; кнопки «Применить исключение» и «Сбросить всё»);
- после фильтрации пересчитываются кросс-плот, аппроксимации и таблица метрик;
- сравнение «до / после»: n скважин, R²_lin, R²_fit, RMSE, MSE, MAE, Bias, Std, Variance, MAPE %.

---

## 📈 Результаты

### Что получает пользователь
- **Честная оценка точности** структурной карты без утечки данных;
- **Диагноз отклонений**: где модель завышает и где занижает глубину, есть ли тренд невязки от глубины;
- **Список аномальных скважин** — кандидатов на перепроверку координат, альтитуд и привязок;
- **Разбивка точности по категориям** — в каких горизонтах и зонах карта «уходит»;
- **Оценка устойчивости** к сокращению выборки и к отдельным наблюдениям;
- **Автоматически выбранная лучшая схема CV** как ориентир для дальнейшей работы — с учётом того, что Random K-Fold участвует в сравнении и может давать оптимистичную оценку.

### Формулы метрик

Здесь Δ — невязка, N — число точек, Δ̄ — средняя невязка (= Bias), y — эталонное значение.

| Метрика | Формула |
|---|---|
| R²_lin | `r²` (квадрат корреляции Пирсона, `linregress`) — мера только линейной связи |
| R²_fit | `1 − Σ(Δ)² / Σ(y − ȳ)²` (корректна для кривой любой формы) |
| RMSE | `√[ (1/N)·Σ(Δ)² ]` |
| MSE | `(1/N)·Σ(Δ)²` |
| MAE | `(1/N)·Σ\|Δ\|` |
| Bias | `(1/N)·Σ(Δ)` |
| Std | `√[ (1/(N−1))·Σ(Δᵢ − Δ̄)² ]` |
| Variance | `(1/(N−1))·Σ(Δᵢ − Δ̄)²` |
| MAPE, % | `(100/N)·Σ[ \|Δ\| / max(\|y\|, 10⁻⁹) ]` |
| P-value | `t = r·√[(N−2)/(1−r²)]`, `p = 2·(1 − F_t,N−2(\|t\|))` — только для базового QC-кросс-плота |

R²_lin и R²_fit — разные величины. Для OLS-прямой они совпадают, для остальных 16 типов кривых — нет, поэтому напрямую сравнивать их нельзя.

### Детекция выбросов

Выбросы проверяются по невязкам Δ в Шагах 1 и 2 четырьмя независимыми правилами. Каждая точка проверяется по каждому правилу отдельно.

| Правило | Условие выброса | Параметр в сайдбаре |
|---|---|---|
| Sigma | `\|Δ\| > σ_thr · σ`, σ — выборочное стандартное отклонение невязок (ddof = 1) | σ_thr: 1.0–6.0, по умолчанию 2.0 |
| IQR | `Δ < Q25 − k·IQR` или `Δ > Q75 + k·IQR` | k: 1.0–3.0, по умолчанию 1.5 |
| Modified z-score | `\|0.6745·(Δ − median)/MAD\| > 3.5`, устойчиво к экстремальным значениям | фиксированный порог 3.5 |
| Квантильное | `Δ < P_low` или `Δ > P_high` | P_low: 0–20%, по умолчанию 5%; P_high: 80–100%, по умолчанию 95% |

Если σ или MAD равны нулю, соответствующее правило не помечает ни одной точки. Границы IQR в таблице статистики фиксированы при k = 1.5; слайдер действует только на детекцию выбросов.

---

## ⚠️ Ограничения и допущения

- **Distance-based exclusion CV** — эвристика по расстояниям. Радиус корреляции равен медианному попарному расстоянию между скважинами, умноженному на `variogram_factor`. Это не полноценная геостатистическая вариограмма (nugget, sill, range) и не подгонка модели вариограммы.
- **Пространственные схемы (Block, Buffered, Spatial Leave-P-Out)** требуют хорошего пространственного покрытия площади. На редкой или сильно кластеризованной сетке скважин отдельные блоки или окна могут оказаться пустыми или содержать мало обучающих скважин.
- **При малом числе скважин (меньше 15–20)** практически единственный осмысленный вариант — Leave-One-Well-Out; остальные схемы дают нестабильные оценки.
- **Random K-Fold** участвует в сравнении как контрольная схема. Он игнорирует пространственную корреляцию, поэтому его результат не следует использовать как оценку реальной точности. Поскольку он участвует в автоматическом выборе, его может выбрать «лучшей» схемой. Это известная проблема, которая намеренно сохранена в текущей версии.
- **Квантильная регрессия** выводится отдельно для каждой τ на графиках и в строках метрик по каждой τ. Общая таблица аппроксимаций (`compute_fit_stats`) учитывает только τ = 0.5. Для τ = 0.1 и 0.9 R²_fit может быть отрицательным; такие кривые оценивайте по уравнению и Bias.
- **Выделение точек кликом** доступно только в интерактивном приложении (`app_full.py`). В Jupyter-ноутбуке этот режим не воспроизводится: там фильтрация задаётся списком скважин или процентом.
- **Выводы зависят от данных.** Инструмент даёт диагностику и ориентиры, но не заменяет геологическую интерпретацию и проверку на реальном наборе скважин.

---

## 📁 Структура репозитория

```text
04_Structural_QC/
│
├── app_full.py                  — главный Streamlit-сценарий (UI, sidebar, шаги 0–4)
├── app_utils.py                 — утилиты расчётов и графики
├── structural_qc_full.py        — ядро расчётов и кросс-валидации
├── translations_full.py         — словари RU / EN
├── ui_examples_full_ru.py       — справка по проекту в приложении (RU)
├── ui_examples_full_en.py       — справка по проекту в приложении (EN)
│
├── data/                        — демонстрационные данные (синтетические)
│   └── *.csv, *.xlsx
├── test_data_import/            — файлы для локальных тестов загрузки
│   └── *.csv, *.xlsx
│
├── structural_qc_demo.ipynb     — демонстрационный ноутбук (читает data/structural_wells_100_linear.csv)
├── structural_qc_demo.pdf       — демонстрационный ноутбук в PDF
├── structural_qc_plan.pdf       — план проекта
├── structural_qc_streamlit_demo.pdf — демонстрация интерфейса
│
├── requirements.txt             — зависимости
└── README.md                    — documentation (RU / EN)
```

---

## 🚀 Запуск

Рекомендуется Python 3.10–3.12.

### Локальный запуск

1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/aspozd-DA-DS/Pet-projects.git
   ```

2. Перейти в папку проекта:
   ```bash
   cd Pet-projects/04_Structural_QC
   ```

3. (Рекомендуется) создать и активировать виртуальное окружение:
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

6. Открыть в браузере `http://localhost:8501`.

### Ноутбук

Для ноутбука нужен Jupyter, он указан в `requirements.txt`:

```bash
jupyter notebook structural_qc_demo.ipynb
```

Запускайте из папки проекта, чтобы относительный путь `data/structural_wells_100_linear.csv` находил файл.

### Онлайн

Приложение также развёрнуто по адресу [structural-qc-cv.streamlit.app](https://structural-qc-cv.streamlit.app/).

---

## 📋 Требования к входному файлу

**Обязательные колонки:**
- `well_id` — идентификатор скважины;
- `X_coord`, `Y_coord` — координаты;
- `PARAM_1` — эталонный параметр;
- `PARAM_2` — сравниваемый параметр.

**Опциональные колонки:**
- `horizon_id` — горизонт;
- `tectonic_zone` — тектоническая зона;
- `reservoir_flag` — тип коллектора;
- `fluid_type` — тип флюида;
- любые дополнительные числовые или категориальные колонки.

**Пример:**

| well_id | X_coord | Y_coord | Z_well | Z_map | horizon_id | tectonic_zone |
|---|---|---|---|---|---|---|
| Well_001 | 125000.0 | 745000.0 | 3120.5 | 3108.7 | H1 | Zone_A |
| Well_002 | 126500.0 | 746200.0 | 3188.2 | 3201.5 | H1 | Zone_A |
| Well_003 | 128200.0 | 748100.0 | 3256.4 | 3242.1 | H1 | Zone_B |

---

## 🔗 Ссылки

- Онлайн-приложение: https://structural-qc-cv.streamlit.app/
- Репозиторий: https://github.com/aspozd-DA-DS/Pet-projects
- Демо-ноутбук: [structural_qc_demo.ipynb](structural_qc_demo.ipynb)
- План проекта: [structural_qc_plan.pdf](structural_qc_plan.pdf)
- Справка по модулям (EN): [ui_examples_full_en.py](ui_examples_full_en.py)
- Справка по модулям (RU): [ui_examples_full_ru.py](ui_examples_full_ru.py)

---

## 🏷 Темы

`Structural QC` `Cross-Plot` `Cross-Validation` `Data Leakage` `Spatial CV`
`LOWO` `Block CV` `Buffered CV` `Spatial Leave-P-Out` `Distance-based CV`
`Geology` `Seismic` `Oil & Gas` `Well Data` `Structural Map`
`Streamlit` `Plotly` `Python` `Pandas` `NumPy` `scikit-learn` `statsmodels`
`Machine Learning` `Data Science` `QC` `Regression` `Residual Analysis`
`Outlier Detection` `Bilingual UI` `RU/EN`

<p align="right"><a href="#top">⬆ наверх</a></p>

---

<a id="en"></a>

# Structural QC — cross-plot & cross-validation (Oil & Gas)

**Links:** [Live app](https://structural-qc-cv.streamlit.app/) 

---

## 📌 Overview

Structural QC is an interactive **Streamlit dashboard** for checking the quality of structural maps against well data from 2D/3D CDP seismic. It compares a **reference parameter** (for example, the horizon depth from a well, `Z_well`) with a **compared parameter** (the depth of the same horizon from a structural map, `Z_map`) at the same points.

The main methodological problem the tool addresses is **data leakage in a geological form**. If wells were used when building the map (velocity-model calibration, VSP tie, kriging with well data), checking the map at those same wells gives an **inflated, optimistic** accuracy. The dashboard applies a full set of cross-validation schemes to estimate the real predictive ability.

The tool is **generic**: it works for any pair of numeric parameters, not only "well ↔ map":

| Scenario | PARAM_1 (Reference) / PARAM_2 (Compared) |
|---|---|
| well ↔ map | Z_well / Z_map, T_well / T_map |
| well ↔ attribute | Z_well / Amplitude |
| map ↔ map | Z_map_v1 / Z_map_v2 |
| attribute ↔ attribute | Amplitude / Sweetness |

### What the tool does

- loads CSV/XLSX and runs a full QC of the input (duplicates, gaps, data types, garbage values);
- builds the basic QC cross-plot with the Y = X line, linear regression and a full metric set;
- overlays **17 approximation curves** (polynomials of degree 1–10, power law, exponential, logarithmic, Huber, Theil-Sen, RANSAC, LOWESS) and **quantile regression** (τ = 0.1 / 0.5 / 0.9);
- estimates predictive ability with **8 cross-validation schemes**: Random K-Fold, Leave-One-Well-Out, Block, Buffered, Leave-P-Out, Spatial Leave-P-Out, Stratified by Zone, Distance-based exclusion CV;
- selects the best CV scheme automatically by `score = RMSE + RMSE_fold_std` across all eight schemes, **including Random K-Fold**, which takes part as a control scheme that is known to be optimistic (see *Limitations*);
- analyses residuals, draws a **residual map** and detects outliers with **4 independent rules**;
- runs category diagnostics (horizons, tectonic zones, reservoir type, fluid type, etc.);
- lets you filter wells interactively for a stability stress test;
- provides a **bilingual RU/EN interface**;
- lets you configure colours, line styles, histogram bins and normalisation in the sidebar.

### Practical value

- an objective estimate of structural-map accuracy without data leakage;
- identification of zones with systematic bias (velocity-model errors, tectonic complications);
- search for anomalous wells to re-check coordinates, elevations and ties;
- evidence for whether the map should be revised before drilling;
- comparison of several interpretation versions;
- assessment of how well the model transfers to new, undrilled areas.

---

## 🧭 Key method decisions

- **Model:** ordinary linear regression (polynomial degree 1) is used in all steps, including cross-validation. Higher-degree fits are available only as approximation curves for visual comparison, not as the model used in CV.
- **Residual sign:** the basic QC cross-plot uses Δ = PARAM_1 − PARAM_2. Approximation metrics and CV metrics use Predicted − Actual. CV residual plots in Step 2 use PARAM_1 − Predicted. These are different conventions by design; the sign is shown on every plot.
- **Best-scheme selection:** `score = RMSE + RMSE_fold_std` is minimised over all eight schemes. Random K-Fold is included on purpose, so that the contrast with spatial schemes stays visible. Its optimistic result can win the selection; interpret the "best" scheme together with the spatial schemes.
- **Distance-based exclusion CV:** the exclusion radius is `excl_radius = median pairwise distance between wells × variogram_factor`. This is a distance heuristic, not a fitted variogram.

---

## 🔧 Technology stack

### Interface and visualisation
- Streamlit
- Plotly (graph_objects, express)
- Seaborn (pairplot)
- Matplotlib (notebook)

### Data processing
- NumPy
- Pandas
- OpenPyXL, xlrd

### ML and statistics
- scikit-learn (LinearRegression, HuberRegressor, TheilSenRegressor, RANSACRegressor, PolynomialFeatures, KFold, LeaveOneGroupOut, r2_score)
- statsmodels (QuantReg, lowess)
- SciPy (stats.linregress, spatial.distance.pdist / cdist, gaussian_kde)

### Architecture
- modular layout: calculation core (`structural_qc_full.py`), helpers and plots (`app_utils.py`), Streamlit scenario (`app_full.py`), translations (`translations_full.py`);
- heavy computations cached with `@st.cache_data` (cross-validation, fits, quantile fits, spacing statistics);
- bilingual interface through the `RU` / `EN` dictionaries.

---

## 📊 Data

The input is a CSV or XLSX file with the following columns.

| Column | Role | Description |
|---|---|---|
| `well_id` | required | Well identifier |
| `X_coord`, `Y_coord` | required | Well coordinates |
| `PARAM_1` (Zprm_well) | required · Reference | Reference parameter, e.g. Z_well (horizon depth from the well) |
| `PARAM_2` (Zprm_map) | required · Compared | Compared parameter, e.g. Z_map (horizon depth from the structural map) |
| `horizon_id` | optional | Horizon identifier, used to filter the analysis |
| `tectonic_zone` | optional | Tectonic zone, used in Stratified by Zone CV and Step 3 |
| `reservoir_flag`, `fluid_type` | optional | Categorical features for Step 3 |
| any other column | optional | Numeric columns go to the correlation analysis; categorical columns go to the category analysis |

> **The demo files in `data/` are synthetic.** They illustrate the tool and do not describe a real field or real wells. For your own data, use files in the format above.

**Automatic cleaning on load:**
- garbage values (`NA`, `NULL`, `---`, `?`, `*`, etc.) are converted to NaN;
- commas are replaced with dots in numeric columns;
- column types are converted;
- rows with gaps in required columns are removed, and the number of removed rows is reported.

**Pipeline outputs:**
- a cleaned dataset with system columns;
- QC cross-plot metrics;
- a metric table for all approximations;
- a summary table of all CV schemes;
- residual distribution and statistics;
- a list of outlier wells from the 4 methods;
- accuracy breakdown by category;
- a before / after comparison for well filtering.

---

## 🧩 Workflow

### Step 0 — Data loading and QC
- load CSV/XLSX or choose a demo file;
- map columns;
- automatic cleaning;
- input QC: full duplicates, duplicate `well_id` (a real duplicate means the same `well_id` and `horizon_id`), coincident coordinates, gaps, removed rows;
- data types and categorical features;
- descriptive statistics of `PARAM_1` and `PARAM_2`;
- statistics for every numeric column: table, histogram with KDE, mean, median and ±3σ, boxplot; bins and normalisation are set per column;
- correlation analysis: Pearson (linear relationships only) and Spearman (any monotonic relationship), plus pairplot (Seaborn or Plotly; at most about 6 parameters recommended for the pairplot).

### Step 1 — Basic QC cross-plot
- cross-plot with the Y = X line and linear regression (degree 1);
- QC metrics: R²_lin, RMSE, MSE, MAE, Bias, Std, Variance, MAPE %, P-value, regression equation;
- 17 approximation curves and quantile regression (τ = 0.1 / 0.5 / 0.9), with a comparison table of R²_fit; metrics are computed separately for each τ;
- distributions of `PARAM_1` and `PARAM_2` (histograms with KDE, boxplots);
- residual analysis Δ = PARAM_1 − PARAM_2: plot against `PARAM_1`, statistics table, histogram, boxplot;
- **residual map** in plan view (X_coord, Y_coord, colour = residual value);
- **outlier detection with 4 methods** (see *Outlier detection* below).

### Step 2 — Cross-validation
- 8 CV schemes, with recommended parameters computed automatically in the sidebar;
- for each scheme: cross-plot Predicted vs Actual, metrics, fold statistics;
- summary table of all schemes, bar charts of RMSE and of mean fold RMSE;
- boxplot of fold RMSE spread for each scheme (median, mean, outliers); the scheme order matches the table;
- automatic best-scheme selection by `score = RMSE + RMSE_fold_std` (Random K-Fold included as a control);
- detailed review of the best scheme: cross-plot, approximations, CV residual distribution, outliers.

### Step 3 — Category diagnostics
- works for any categorical column (`horizon_id`, `tectonic_zone`, `reservoir_flag`, `fluid_type`, etc.);
- pie and bar charts of well counts per category;
- descriptive statistics per category;
- boxplots of parameter distributions per category;
- overlaid histograms per category for `PARAM_1` and `PARAM_2`;
- combined cross-plot for all categories and for a chosen subset;
- cross-plot and approximations for each category separately;
- comparison table of approximation metrics by category.

### Step 4 — Well filtering (stress test)
- four modes: no filter, manual exclusion of wells, random exclusion by percentage (0–95%, with resampling), and **click selection on the cross-plot** (click a point, Shift+click for several, Box or Lasso selection; Apply exclusion / Reset all);
- the cross-plot, approximations and metric table are recalculated after filtering;
- before / after comparison: n wells, R²_lin, R²_fit, RMSE, MSE, MAE, Bias, Std, Variance, MAPE %.

---

## 📈 Results

### What the user gets
- **An honest accuracy estimate** for the structural map without data leakage;
- **A diagnosis of deviations**: where the map overestimates or underestimates depth, and whether the residual trends with depth;
- **A list of anomalous wells** (candidates for re-checking coordinates, elevations and ties);
- **Accuracy by category**: which horizons or zones the map "drifts" in;
- **A stability estimate** for reduced samples and individual observations;
- **The automatically selected best CV scheme** as a reference point, read together with the note on Random K-Fold above.

### Metric formulas

Here Δ is the residual, N the number of points, Δ̄ the mean residual (= Bias), and y the reference value.

| Metric | Formula |
|---|---|
| R²_lin | `r²` (squared Pearson correlation, `linregress`) — a measure of linear strength only |
| R²_fit | `1 − Σ(Δ)² / Σ(y − ȳ)²` (valid for any curve shape) |
| RMSE | `√[ (1/N)·Σ(Δ)² ]` |
| MSE | `(1/N)·Σ(Δ)²` |
| MAE | `(1/N)·Σ\|Δ\|` |
| Bias | `(1/N)·Σ(Δ)` |
| Std | `√[ (1/(N−1))·Σ(Δᵢ − Δ̄)² ]` |
| Variance | `(1/(N−1))·Σ(Δᵢ − Δ̄)²` |
| MAPE, % | `(100/N)·Σ[ \|Δ\| / max(\|y\|, 10⁻⁹) ]` |
| P-value | `t = r·√[(N−2)/(1−r²)]`, `p = 2·(1 − F_t,N−2(\|t\|))` — only for the basic QC cross-plot |

R²_lin and R²_fit are different quantities. They coincide for an OLS line but not for the other 16 curve types, so do not compare them directly.

### Outlier detection

Outliers are checked on the residuals Δ in Steps 1 and 2 by four independent rules. Each point is tested against each rule separately.

| Rule | Outlier condition | Sidebar parameter |
|---|---|---|
| Sigma | `\|Δ\| > σ_thr · σ`, σ = sample standard deviation of residuals (ddof = 1) | σ_thr: 1.0–6.0, default 2.0 |
| IQR | `Δ < Q25 − k·IQR` or `Δ > Q75 + k·IQR` | k: 1.0–3.0, default 1.5 |
| Modified z-score | `\|0.6745·(Δ − median)/MAD\| > 3.5`, robust to extreme values | fixed threshold 3.5 |
| Quantile | `Δ < P_low` or `Δ > P_high` | P_low: 0–20%, default 5%; P_high: 80–100%, default 95% |

If σ or MAD is zero, the corresponding rule flags no points. The IQR bounds in the statistics table are fixed at k = 1.5; the slider applies only to outlier detection.

---

## ⚠️ Limitations and assumptions

- **Distance-based exclusion CV** is a distance heuristic. The correlation radius is the median pairwise distance between wells multiplied by `variogram_factor`. It is not a full geostatistical variogram (nugget, sill, range) and does not fit a variogram model.
- **Spatial CV (Block, Buffered, Spatial Leave-P-Out)** needs good spatial coverage. On sparse or strongly clustered well layouts, some blocks or windows may be empty or contain few training wells.
- **With few wells (fewer than 15–20)**, Leave-One-Well-Out is practically the only meaningful option; the other schemes give unstable estimates.
- **Random K-Fold** is part of the comparison as a control. It ignores spatial correlation, so its result should not be used as an estimate of real accuracy. Because it participates in the automatic selection, it can be selected as the "best" scheme. This is a known issue, kept on purpose in the current version.
- **Quantile regression** is reported per τ in the plots and in the per-τ metric rows. The general approximation table (`compute_fit_stats`) uses only τ = 0.5. For τ = 0.1 and 0.9, R²_fit can be negative; judge those curves by Equation and Bias.
- **Click selection of points** exists only in the interactive app (`app_full.py`). The Jupyter notebook does not reproduce it; there, filtering uses a list of wells or a percentage.
- **Results depend on the data.** The tool gives diagnostics and reference values. It does not replace geological interpretation or validation on a real well set.

---

## 📁 Repository structure

```text
04_Structural_QC/
│
├── app_full.py                  — main Streamlit scenario (UI, sidebar, steps 0–4)
├── app_utils.py                 — calculation helpers and plots
├── structural_qc_full.py        — calculation core and cross-validation
├── translations_full.py         — RU / EN dictionaries
├── ui_examples_full_ru.py       — in-app project reference (RU)
├── ui_examples_full_en.py       — in-app project reference (EN)
│
├── data/                        — demo datasets (synthetic)
│   └── *.csv, *.xlsx
├── test_data_import/            — files for local import tests
│   └── *.csv, *.xlsx
│
├── structural_qc_demo.ipynb     — demo notebook (reads data/structural_wells_100_linear.csv)
├── structural_qc_demo.pdf       — demo notebook as PDF
├── structural_qc_plan.pdf       — project plan
├── structural_qc_streamlit_demo.pdf — interface walkthrough
│
├── requirements.txt             — dependencies
└── README.md                    — documentation (RU / EN)
```

---

## 🚀 Running the project

Python 3.10–3.12 is recommended.

### Local run

1. Clone the repository:
   ```bash
   git clone https://github.com/aspozd-DA-DS/Pet-projects.git
   ```

2. Go to the project folder:
   ```bash
   cd Pet-projects/04_Structural_QC
   ```

3. (Recommended) create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   .venv\Scripts\activate         # Windows
   ```

4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:
   ```bash
   streamlit run app_full.py
   ```

6. Open `http://localhost:8501` in a browser.

### Notebook

The notebook needs Jupyter, which is listed in `requirements.txt`:

```bash
jupyter notebook structural_qc_demo.ipynb
```

Run it from the project folder, so that the relative path `data/structural_wells_100_linear.csv` resolves.

### Online

The app is also deployed at [structural-qc-cv.streamlit.app](https://structural-qc-cv.streamlit.app/).

---

## 📋 Input file requirements

**Required columns:**
- `well_id` — well identifier;
- `X_coord`, `Y_coord` — coordinates;
- `PARAM_1` — reference parameter;
- `PARAM_2` — compared parameter.

**Optional columns:**
- `horizon_id` — horizon;
- `tectonic_zone` — tectonic zone;
- `reservoir_flag` — reservoir type;
- `fluid_type` — fluid type;
- any additional numeric or categorical columns.

**Example:**

| well_id | X_coord | Y_coord | Z_well | Z_map | horizon_id | tectonic_zone |
|---|---|---|---|---|---|---|
| Well_001 | 125000.0 | 745000.0 | 3120.5 | 3108.7 | H1 | Zone_A |
| Well_002 | 126500.0 | 746200.0 | 3188.2 | 3201.5 | H1 | Zone_A |
| Well_003 | 128200.0 | 748100.0 | 3256.4 | 3242.1 | H1 | Zone_B |

---

## 🔗 Links

- Live app: https://structural-qc-cv.streamlit.app/
- Repository: https://github.com/aspozd-DA-DS/Pet-projects
- Demo notebook: [structural_qc_demo.ipynb](structural_qc_demo.ipynb)
- Project plan: [structural_qc_plan.pdf](structural_qc_plan.pdf)
- Module reference (EN): [ui_examples_full_en.py](ui_examples_full_en.py)
- Module reference (RU): [ui_examples_full_ru.py](ui_examples_full_ru.py)

---

## 🏷 Topics

`Structural QC` `Cross-Plot` `Cross-Validation` `Data Leakage` `Spatial CV`
`LOWO` `Block CV` `Buffered CV` `Spatial Leave-P-Out` `Distance-based CV`
`Geology` `Seismic` `Oil & Gas` `Well Data` `Structural Map`
`Streamlit` `Plotly` `Python` `Pandas` `NumPy` `scikit-learn` `statsmodels`
`Machine Learning` `Data Science` `QC` `Regression` `Residual Analysis`
`Outlier Detection` `Bilingual UI` `RU/EN`

<p align="right"><a href="#top">⬆ back to top</a></p>
