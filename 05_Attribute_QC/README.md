<a id="top"></a>

<p align="right">
  <a href="#ru">🇷🇺 RU</a> | <a href="#en">🇬🇧 EN</a>
</p>

---

<a id="ru"></a>

# Attribute QC_CV — кросс-плот и кросс-валидация для атрибутного анализа (Oil & Gas)

**Ссылки:** [Онлайн-приложение](https://attribute-qc-cv.streamlit.app/) 


## 📌 Описание

Attribute QC_CV — интерактивный **Streamlit-дашборд** для оценки качества связи между сейсмическими атрибутами и свойствами пласта по данным скважин. Инструмент сравнивает **эталонный параметр** `PARAM_1` (свойство пласта в скважине: **эффективная толщина H_eff** или **пористость Phi**) с **сравниваемым параметром** `PARAM_2` (сейсмический атрибут в той же скважине) и проверяет, насколько параметр пласта предсказывается по атрибуту.

Главная методологическая проблема — **переоценка качества в выборке, на которой модель обучалась** (in-sample). Модель, проверенная на тех же скважинах, на которых подбиралась, даёт завышенную оценку. Поэтому дашборд разделяет два режима:

- **In-sample** — модель подбирается и проверяется на одних и тех же скважинах (описание того, как модель объясняет известные данные);
- **Out-of-sample** — модель подбирается без части скважин и проверяется на остальных, через схемы кросс-валидации (оценка на новых скважинах).

Подробные определения и таблица «шаг → режим» приведены в разделе [In-sample и out-of-sample в проекте](#in-out-ru).

Инструмент реализован **обобщённо**: он применим к любой паре числовых параметров, а не только к «свойство пласта ↔ сейсмический атрибут»:

| Сценарий | PARAM_1 (Reference) / PARAM_2 (Compared) |
|---|---|
| скважина ↔ атрибут | H_eff / rms_amplitude, Phi / acoustic_impedance |
| атрибут ↔ атрибут | rms_amplitude / sweetness (проверка мультиколлинеарности) |

В README и в приложении сквозной иллюстративный пример — H_eff / rms_amplitude, но все шаги применимы к любой паре.

### Что делает инструмент

- загружает CSV/XLSX, сопоставляет колонки в сайдбаре и проводит QC входных данных (дубликаты, пропуски, типы, мусорные значения);
- строит базовый QC-кросс-плот с линией Y = X, линейной регрессией и набором метрик;
- накладывает **17 типов аппроксимаций** (полиномы 1–10 степени, степенная, экспонента, логарифм, Huber, Theil-Sen, RANSAC, LOWESS) и **квантильную регрессию** (τ = 0.1 / 0.5 / 0.9);
- оценивает предсказательную способность по **схемам кросс-валидации** (до 9): Random K-Fold, Random K-Fold (buffered), Leave-One-Well-Out (LOWO), Block CV, Buffered CV, Leave-P-Out, Spatial Leave-P-Out, Distance-based exclusion CV и Stratified by {колонка};
- автоматически выбирает лучшую схему CV по критерию из сайдбара (по умолчанию `score = RMSE + RMSE_fold_std`);
- выявляет выбросы CV-невязок **четырьмя независимыми методами**;
- проводит диагностику по категориям (горизонты, тектонические зоны, тип коллектора, тип флюида, категория скважины и др.);
- позволяет интерактивно фильтровать скважины для стресс-теста устойчивости;
- на **Шаге 5** строит многоатрибутную линейную регрессию отдельно для H_eff и отдельно для Phi;
- поддерживает **двуязычный интерфейс RU/EN**.

### Практическая ценность

- честная оценка того, насколько сейсмический атрибут предсказывает свойство пласта, без переоценки на обучающих скважинах;
- выбор атрибута для H_eff и для Phi отдельно: у каждого свойства свой оптимальный атрибут и своя чувствительность;
- выявление зон систематических ошибок (по горизонтам и зонам);
- поиск аномальных скважин для перепроверки данных;
- проверка устойчивости связи к сокращению числа скважин;
- обоснование того, стоит ли использовать многоатрибутную модель, или одного атрибута достаточно.

---

## 🧭 Ключевые решения метода

- **Модель.** Во всех шагах используется линейная регрессия (полином степени 1). Аппроксимации более высокой степени и квантильная регрессия доступны как кривые для визуального сравнения, но в кросс-валидации модель — только линейная.
- **Один горизонт.** Для строгой оценки рекомендуется работать с одним горизонтом. При нескольких горизонтах одна скважина может попасть одновременно в обучение и в тест, см. раздел «Ограничения».
- **Знак невязки.** Базовый QC-кросс-плот (Шаг 1): Δ = PARAM_1 − PARAM_2. Аппроксимации и CV (Шаги 1–4): Δ = Predicted − Actual. Графики CV-невязок Шага 2: Actual − Predicted. Соглашения разные намеренно, знак указан на каждом графике.
- **Выбор лучшей схемы CV.** По умолчанию минимизируется `score = RMSE + RMSE_fold_std`. В сайдбаре доступны также RMSE, MAE, максимальный R² (Actual = Predicted), RMSE + |Bias| и ручной выбор. Random K-Fold участвует намеренно как контрольная схема с известно оптимистичной оценкой; «лучшую» схему интерпретируйте вместе с пространственными.
- **Ошибка в процентах — WAPE, а не MAPE.** Для параметров с большим числом нулевых значений (например, H_eff = 0 там, где нет коллектора) MAPE не определена. Используется `WAPE = 100 · Σ|Δ| / Σ|y|`.
- **Шаг 5: отбор модели по RMSE_cv.** Многоатрибутная модель принимается, только если её RMSE_cv меньше RMSE_cv лучшей одноатрибутной модели в той же схеме CV. In-sample R² для отбора не используется.
- **Distance-based exclusion CV.** Радиус исключения `excl_radius = медианное попарное расстояние между скважинами × variogram_factor`. Это эвристика по расстояниям, а не подогнанная вариограмма.

---

<a id="in-out-ru"></a>

## 🔍 In-sample и out-of-sample в проекте

**In-sample** — модель подбирается и проверяется на одних и тех же скважинах. Метрики показывают, насколько хорошо модель описывает данные, которые она уже видела, поэтому обычно они оптимистичны. Уравнение in-sample — обычная линейная регрессия по всем скважинам выбранной выборки.

**Out-of-sample** — модель подбирается без части скважин, а проверяется на тех, которых при подборе не было (кросс-валидация по схемам LOWO, K-fold и др.). Метрики показывают качество на новых скважинах. Разрыв между in-sample и out-of-sample — признак переобучения. В таблицах Out-of-sample метрики и уравнение `Predicted = a + b·Actual` получены по одним и тем же out-of-fold предсказаниям.

| Шаг | Что считается | Режим |
|---|---|---|
| 0 | корреляции Пирсона и Спирмена, описательная статистика | без модели |
| 1 | базовый QC-кросс-плот, аппроксимации, R²_lin, R²_fit, P-value, уравнение регрессии | in-sample |
| 2 | девять схем CV, фолды, выбор лучшей схемы, выбросы CV-невязок, R² (Pearson), R² (Actual = Predicted), уравнение `Predicted = a + b·Actual`, аппроксимации лучшей схемы | out-of-sample |
| 3 | аппроксимации и метрики внутри каждой категории | in-sample (внутри категории) |
| 4 | метрики до и после фильтрации, аппроксимации после фильтрации | in-sample (на отфильтрованной выборке) |
| 4.5 | R² (Pearson) по сериям случайного исключения | in-sample (оставшиеся скважины) и out-of-sample (исключённые) |
| 5 | одноатрибутные модели и топ-5 многоатрибутных моделей | In-sample: RMSE_in, R²_in, r_in, уравнение по всем скважинам; Out-of-sample: RMSE_cv, R²_cv, r_cv, уравнение `Predicted = a + b·Actual` |

---

## 🔧 Стек технологий

### Интерфейс и визуализация
- Streamlit
- Plotly (graph_objects, express)
- Seaborn (pairplot)

### Обработка данных
- NumPy, Pandas
- OpenPyXL (чтение XLSX и выгрузка в Excel)

### Статистика и машинное обучение
- scikit-learn (LinearRegression, HuberRegressor, TheilSenRegressor, RANSACRegressor, PolynomialFeatures, KFold, LeaveOneGroupOut, r2_score)
- statsmodels (QuantReg, lowess)
- SciPy (stats.linregress, spatial.distance pdist / cdist, gaussian_kde)

### Архитектура
- модульная структура: расчёты кросс-валидации и аппроксимаций (`attribute_qc_full.py`), утилиты и графики (`app_utils.py`), сценарий интерфейса (`app_full.py`), словари переводов (`translations_full.py`), справка по проекту в приложении (`ui_examples_full_ru.py`, `ui_examples_full_en.py`);
- тяжёлые вычисления кэшируются через `@st.cache_data` (кросс-валидация, аппроксимации, квантильные подгонки, статистика расстояний, расчёт Шага 5);
- двуязычный интерфейс через словари `RU` / `EN`.

---

## 📊 Данные

На вход принимается CSV или XLSX. Имена колонок произвольные: сопоставление выполняется в сайдбаре.

| Колонка | Роль | Описание |
|---|---|---|
| `well_id` | обязательная | Идентификатор скважины |
| `X`, `Y` | обязательные | Координаты скважины (в коде `X_coord`, `Y_coord`) |
| `H_eff` или `Phi` (PARAM_1) | обязательная · Reference | Эталонный параметр пласта: эффективная толщина коллектора или пористость. Выбирается в начале Шага 1 |
| сейсмический атрибут (PARAM_2) | обязательная · Compared | Сравниваемый атрибут, например `rms_amplitude`, `envelope`, `sweetness`, `coherence`, `inst_frequency`, `avo_intercept`, `avo_gradient`, `spectral_decomp_20Hz`, `acoustic_impedance`, `q_factor`. Выбирается в начале Шага 1 |
| `horizon_id` | опциональная | Горизонт. Позволяет выбрать один или несколько горизонтов; для строгой оценки рекомендуется один |
| `tectonic_zone` | опциональная | Тектоническая зона: категориальная колонка для Stratified by {колонка} CV и Шага 3 |
| `reservoir_flag`, `fluid_type`, `well_category` | опциональные | Категориальные признаки для Шага 3 |
| любая другая числовая | опциональная | Участвует в корреляционном анализе (Шаг 0) и как кандидат в Шаге 5 |

Единицы измерения PARAM_1 и PARAM_2 задаются при сопоставлении колонок и используются только в подписях.

> **Демонстрационные данные синтетические.** Атрибуты не снимались с сейсмического куба, а H_eff и Phi смоделированы. Результаты кросс-плота отражают заложенную структуру, а не реальную геологию. Для собственных данных используйте файлы в формате, описанном выше.

**Автоматическая очистка при загрузке:**
- мусорные значения (`NA`, `NULL`, `—`, `?` и т.п.) превращаются в NaN;
- запятая в числовых колонках заменяется на точку;
- типы колонок приводятся к нужным;
- строки с пропусками в обязательных колонках удаляются, а число удалённых строк выводится в отчёт.

**Результаты пайплайна:**
- очищенный датасет и QC входных данных;
- метрики QC-кросс-плота и таблица аппроксимаций;
- сводная таблица схем CV и детали по фолдам;
- статистика и выбросы CV-невязок;
- разбивка точности по категориям;
- сравнение «до / после» фильтрации скважин;
- многоатрибутные модели для H_eff и Phi (Шаг 5).

---

## 🧩 Ход работы

### Шаг 0 — Загрузка и QC данных
- загрузка CSV/XLSX или выбор демо-файла; сопоставление колонок; автоматическая очистка;
- QC входных данных: полные дубликаты, дубликаты `well_id` (реальный дубликат — совпадение `well_id` и `horizon_id`), совпадающие координаты, пропуски;
- описательная статистика H_eff / Phi и атрибутов: Count, μ, median, mode, σ, variance, CV %, min, Q05, Q25, Q75, Q95, max, range, IQR;
- корреляционный анализ: Пирсон (линейная связь) и Спирмен (монотонная связь), pairplot (Seaborn или Plotly).

### Шаг 1 — Базовый QC-кросс-плот (in-sample)
- выбор PARAM_1 (H_eff или Phi) и PARAM_2 (атрибут), кросс-плот с линией Y = X и линейной регрессией;
- метрики QC: R²_lin, RMSE, MSE, MAE, Bias, Std, Variance, WAPE %, P-value, уравнение регрессии;
- 17 аппроксимаций и квантильная регрессия (τ = 0.1 / 0.5 / 0.9) со сравнительной таблицей R²_fit;
- распределения PARAM_1 и PARAM_2 (гистограммы с KDE, boxplot);

### Шаг 2 — Кросс-валидация (out-of-sample)
- до 9 схем CV с рекомендуемыми параметрами, подбираемыми автоматически в сайдбаре;
- для каждой схемы: кросс-плот Predicted vs Actual, метрики, статистика по фолдам;
- сводная таблица схем, столбчатые диаграммы RMSE, boxplot разброса RMSE по фолдам;
- автоматический выбор лучшей схемы по критерию из сайдбара;
- таблицы фолдов (n_train, n_test, RMSE, MAE, Bias); пропущенные фолды отмечаются строками с n = 0;
- детальный разбор лучшей схемы: кросс-плот, аппроксимации, распределение CV-невязок, выбросы четырьмя методами.

### Шаг 3 — Диагностика по категориям (in-sample внутри категории)
- работает для любой категориальной колонки;
- круговая и столбчатая диаграммы количества скважин;
- статистика PARAM_1 и PARAM_2 по каждой категории; boxplot по категориям;
- наложенные гистограммы по категориям, общий кросс-плот по всем категориям и по выбранному подмножеству;
- кросс-плот и аппроксимации отдельно для каждой категории; сравнительная таблица метрик по категориям.

### Шаг 4 — Фильтрация скважин (стресс-тест)
- четыре режима: без фильтра, ручное исключение скважин, случайное исключение по проценту (0–95 %), выделение точек кликом на кросс-плоте (клик, Shift+клик, рамка Box, лассо Lasso);
- после фильтрации пересчитываются кросс-плот, аппроксимации и таблица метрик;
- сравнение «до / после»: n скважин, R²_lin, R²_fit, RMSE, MSE, MAE, Bias, Std, Variance, WAPE %, P-value;
- **Шаг 4.5 — серия случайного исключения:** уровни 5–95 % с шагом 5 %, по 50–2000 наборов на уровень (по умолчанию 500); для каждого набора R² (Pearson) in-sample (оставшиеся скважины) и out-of-sample (исключённые).

### Шаг 5 — Многоатрибутная регрессия (in-sample и out-of-sample)
- отдельная модель для H_eff и отдельно для Phi; сначала раздел H_eff, затем раздел Phi;
- полный перебор комбинаций атрибутов от 1 до `k_max = min(3, число кандидатов, max(1, ⌊N / 5⌋))`, где N — число скважин; пары атрибутов с |r| > 0.8 в одну модель не попадают;
- две схемы кросс-валидации по скважинам: **Leave-One-Well-Out** (N фолдов) и **K-fold по скважинам** с фиксированным сидом, K = max(3, round(N / 10));
- внутри каждого раздела четыре таблицы:
  1. **Одноатрибутные модели — базовый уровень для сравнения In-sample (RMSE_in, R²_in).** Регрессия параметра на одном атрибуте по всем скважинам, где есть оба значения; одна таблица для обеих схем;
  2. **Одноатрибутные модели — базовый уровень для сравнения Out-of-sample (RMSE_cv, R²_cv).** Отдельно для LOWO и для K-fold;
  3. **Топ-5 многоатрибутных моделей In-sample (RMSE_in, R²_in).**
  4. **Топ-5 многоатрибутных моделей Out-of-sample (RMSE_cv, R²_cv).**
- уравнения: в In-sample — регрессия по всем скважинам; в Out-of-sample — `Predicted = a + b·Actual` по out-of-fold предсказаниям;
- критерий выбора — RMSE_cv; многоатрибутная модель принимается, только если она лучше лучшей одноатрибутной в той же схеме;
- коэффициенты (intercept, коэффициенты, стандартизованные) — в раскрывающемся блоке под таблицей;
- графики: вклад атрибутов лучшей многоатрибутной модели; затем строка из двух графиков для лучшего одноатрибутного (In-sample | Out-of-sample) и строка для лучшего многоатрибутного набора (In-sample | Out-of-sample);
- все таблицы и коэффициенты выгружаются в Excel кнопкой под каждой таблицей.

---

## 📈 Результаты

### Что получает пользователь
- **Оценку связи «атрибут → свойство пласта»** отдельно для H_eff и для Phi, с разделением in-sample и out-of-sample;
- **Выбор атрибута**: одноатрибутный ранг по RMSE_cv и вклад каждого атрибута в многоатрибутную модель;
- **Диагноз ошибок**: систематическое смещение (Bias), наклон и intercept линии `Predicted = a + b·Actual`;
- **Список аномальных скважин** — кандидатов на перепроверку;
- **Разбивку по категориям** — в каких горизонтах и зонах связь работает хуже;
- **Оценку устойчивости** к сокращению выборки.

### Формулы метрик

Здесь Δ — невязка, N — число точек, Δ̄ — средняя невязка (= Bias), y — эталонное значение (PARAM_1).

| Метрика | Формула |
|---|---|
| R²_lin | `r²` (квадрат корреляции Пирсона, `linregress`) — мера только линейной связи |
| R²_fit | `1 − Σ(Δ)² / Σ(y − ȳ)²` (корректна для кривой любой формы) |
| R² (Pearson), Шаг 2 и Шаг 5 | `r²` по out-of-fold предсказаниям: квадрат корреляции Пирсона `Predicted` и `Actual` |
| RMSE | `√[ (1/N)·Σ(Δ)² ]` |
| MSE | `(1/N)·Σ(Δ)²` |
| MAE | `(1/N)·Σ\|Δ\|` |
| Bias | `(1/N)·Σ(Δ)` |
| Std | `√[ (1/(N−1))·Σ(Δᵢ − Δ̄)² ]` |
| Variance | `(1/(N−1))·Σ(Δᵢ − Δ̄)²` |
| WAPE, % | `100 · Σ\|Δ\| / Σ\|y\|` — определена при нулевых значениях y |
| P-value | `t = r·√[(N−2)/(1−r²)]`, `p = 2·(1 − F_t,N−2(\|t\|))`; для Шагов 0, 1, 2, 4 |

R²_lin и R²_fit — разные величины. Для линейной OLS-прямой они совпадают, для остальных 16 типов кривых — нет, поэтому напрямую их сравнивать нельзя.

### Выбросы CV-невязок

Выбросы ищутся на Шаге 2 по CV-невязкам (Actual − Predicted) четырьмя независимыми правилами:

| Правило | Условие выброса | Параметр в сайдбаре |
|---|---|---|
| Sigma | `\|Δ\| > σ_thr · σ`, σ — выборочное стандартное отклонение (ddof = 1) | σ_thr: 1.0–6.0, по умолчанию 2.0 |
| IQR | `Δ < Q25 − k·IQR` или `Δ > Q75 + k·IQR` | k: 1.0–3.0, по умолчанию 1.5 |
| Modified z-score | `\|0.6745·(Δ − median)/MAD\| > 3.5` | фиксированный порог 3.5 |
| Квантильное | `Δ < P_low` или `Δ > P_high` | P_low: 0–20 %, по умолчанию 5 %; P_high: 80–100 %, по умолчанию 95 % |

---

## ⚠️ Ограничения и допущения

- **In-sample оптимистичен.** Метрики и уравнения in-sample описывают обучающие данные. Для оценки на новых скважинах смотрите Out-of-sample.
- **Данные синтетические.** Демонстрационные H_eff и Phi смоделированы, атрибуты рассчитаны по формулам (например, sweetness = envelope / √(instantaneous frequency)). Поэтому часть корреляций между атрибутами — следствие формул, а не независимые связи.
- **Несколько горизонтов одновременно.** Random K-Fold, Random K-Fold (buffered), Leave-P-Out и Spatial Leave-P-Out разбивают записи «скважина — горизонт», а не скважины целиком, и одна скважина может попасть одновременно в обучение и в тест. Для строгой оценки используйте один горизонт или LOWO / Block / Buffered.
- **Random K-Fold** игнорирует пространственную корреляцию и участвует в выборе «лучшей» схемы как контрольная. Его результат не следует считать оценкой реальной точности.
- **Пространственные схемы** (Block, Buffered, Spatial Leave-P-Out) требуют хорошего покрытия площади; на редкой или кластеризованной сетке скважин блоки могут оказаться пустыми.
- **При малом числе скважин** (меньше 15–20) практически единственный осмысленный вариант — LOWO. На малых выборках k_max для Шага 5 мал (при 15 скважинах — 3, при 10 — 2, при 5 — 1), а оценки RMSE_cv нестабильны.
- **Перебор многих атрибутов** увеличивает шанс найти «хороший» тренировочный R² случайно. Поэтому отбор в Шаге 5 делается по RMSE_cv, а каждая многоатрибутная модель должна быть физически осмысленной, а не только статистически лучшей.
- **Квантильная регрессия:** для τ = 0.1 и 0.9 R²_fit может быть отрицательным; такие кривые оценивайте по уравнению и Bias. Общая таблица аппроксимаций учитывает только τ = 0.5.
- **Выделение точек кликом** доступно только в интерактивном приложении (`app_full.py`).
- **Выводы зависят от данных.** Инструмент даёт диагностику и ориентиры, но не заменяет геологическую интерпретацию и проверку на реальном наборе скважин.

---

## 📁 Структура проекта

```text
05_Attribute_QC/
│
├── app_full.py                  — главный Streamlit-сценарий (интерфейс, сайдбар, шаги 0–5)
├── app_utils.py                 — утилиты, графики и вспомогательные функции интерфейса
├── attribute_qc_full.py         — ядро расчётов: аппроксимации, схемы CV, метрики, многоатрибутная регрессия (Шаг 5)
├── translations_full.py         — словари интерфейса RU / EN
├── ui_examples_full_ru.py       — справка по проекту в приложении (RU)
├── ui_examples_full_en.py       — справка по проекту в приложении (EN)
│
├── data/                        — демонстрационные данные (синтетические)
│   └── *.csv, *.xlsx
├── test_data_import/            — файлы для локальных тестов загрузки
│   └── *.csv, *.xlsx
│
├── attribute_qc_demo.ipynb      — демонстрационный ноутбук (читает data/attribute_wells_demo.csv)
├── attribute_qc_demo.pdf        — демонстрационный ноутбук в PDF
├── attribute_qc_plan.pdf        — план проекта
├── attribute_qc_streamlit_demo.pdf — демонстрация интерфейса
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
   cd Pet-projects/05_Attribute_QC
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
jupyter notebook attribute_qc_demo.ipynb
```

Запускайте из папки проекта, чтобы относительный путь `data/attribute_wells_demo.csv` находил файл.

### Онлайн

Приложение также развёрнуто по адресу [attribute-qc-cv.streamlit.app](https://attribute-qc-cv.streamlit.app/).

---

## 📋 Требования к входному файлу

**Обязательные колонки:** `well_id`, координаты `X` и `Y`, эталонный параметр (H_eff или Phi), сейсмический атрибут.

**Опциональные колонки:** `horizon_id`, `tectonic_zone`, `reservoir_flag`, `fluid_type`, `well_category`, любые дополнительные числовые колонки.

Имена колонок произвольные: сопоставление выполняется в сайдбаре. Автоматический выбор колонок по ключевым словам (`well`, `x`, `y`, `H_eff`/`Heff`, `Phi`/`Kp`, `horizon`, `tectonic`/`zone`) работает без учёта регистра.

**Пример:**

| well_id | X | Y | H_eff | rms_amplitude | horizon_id | tectonic_zone |
|---|---|---|---|---|---|---|
| Well_001 | 125000.0 | 745000.0 | 12.4 | 0.082 | H1 | Zone_A |
| Well_002 | 126500.0 | 746200.0 | 3.1 | 0.047 | H1 | Zone_A |
| Well_003 | 128200.0 | 748100.0 | 18.7 | 0.121 | H1 | Zone_B |

*Значения в примере условные и служат только для иллюстрации формата.*

---

## 🔗 Ссылки

- Онлайн-приложение: https://attribute-qc-cv.streamlit.app/
- Репозиторий: https://github.com/aspozd-DA-DS/Pet-projects
- Демо-ноутбук: [attribute_qc_demo.ipynb](attribute_qc_demo.ipynb)
- План проекта: [attribute_qc_plan.pdf](attribute_qc_plan.pdf)
- Справка по модулям (EN): [ui_examples_full_en.py](ui_examples_full_en.py)
- Справка по модулям (RU): [ui_examples_full_ru.py](ui_examples_full_ru.py)

---

## 🏷 Темы

`Attribute QC` `Cross-Plot` `Cross-Validation` `In-sample` `Out-of-sample` `Seismic attributes`
`LOWO` `K-fold` `Block CV` `Buffered CV` `Spatial Leave-P-Out` `Distance-based CV`
`Multi-attribute regression` `Oil & Gas` `Well Data` `H_eff` `Porosity`
`Streamlit` `Plotly` `Python` `Pandas` `NumPy` `scikit-learn` `statsmodels`
`Machine Learning` `Data Science` `QC` `Regression` `Residual Analysis`
`Outlier Detection` `Bilingual UI` `RU/EN`

<p align="right"><a href="#top">⬆ наверх</a></p>

---

<a id="en"></a>

# Attribute QC_CV — cross-plot & cross-validation for seismic attribute analysis (Oil & Gas)

**Links:** [Live app](https://attribute-qc-cv.streamlit.app/) 

## 📌 Overview

Attribute QC_CV is an interactive **Streamlit dashboard** for assessing how well seismic attributes predict reservoir properties at wells. It compares a **reference parameter** `PARAM_1` (a reservoir property at the well: **effective thickness H_eff** or **porosity Phi**) with a **compared parameter** `PARAM_2` (a seismic attribute at the same well), and checks how well the reservoir property is predicted from the attribute.

The main methodological problem is **overestimated quality on the same wells the model was fitted on** (in-sample). A model checked on the wells it was fitted to gives an inflated estimate. The dashboard therefore separates two modes:

- **In-sample** — the model is fitted and checked on the same wells (how well it describes the data it has seen);
- **Out-of-sample** — the model is fitted without some wells and checked on the rest, through cross-validation schemes (the estimate on new wells).

Detailed definitions and the table «step → mode» are in [In-sample and out-of-sample in the project](#in-out-en).

The tool is **generic**: it works for any pair of numeric parameters, not only «reservoir property ↔ seismic attribute»:

| Scenario | PARAM_1 (Reference) / PARAM_2 (Compared) |
|---|---|
| well ↔ attribute | H_eff / rms_amplitude, Phi / acoustic_impedance |
| attribute ↔ attribute | rms_amplitude / sweetness (multicollinearity check) |

In this README and in the app, H_eff / rms_amplitude is the running illustrative example, but every step applies to any pair.

### What the tool does

- loads CSV/XLSX, maps the columns in the sidebar and runs a QC of the input (duplicates, gaps, data types, garbage values);
- builds the basic QC cross-plot with the Y = X line, linear regression and a metric set;
- overlays **17 approximation types** (polynomials of degree 1–10, power law, exponential, logarithmic, Huber, Theil-Sen, RANSAC, LOWESS) and **quantile regression** (τ = 0.1 / 0.5 / 0.9);
- estimates predictive ability with **up to 9 cross-validation schemes**: Random K-Fold, Random K-Fold (buffered), Leave-One-Well-Out (LOWO), Block CV, Buffered CV, Leave-P-Out, Spatial Leave-P-Out, Distance-based exclusion CV and Stratified by {column};
- selects the best CV scheme automatically by the sidebar criterion (default `score = RMSE + RMSE_fold_std`);
- detects outliers of the CV residuals with **four independent rules**;
- runs category diagnostics (horizons, tectonic zones, reservoir type, fluid type, well category, etc.);
- lets you filter wells interactively for a stability stress test;
- runs **Step 5**: multi-attribute linear regression separately for H_eff and for Phi;
- provides a **bilingual RU/EN interface**.

### Practical value

- an honest estimate of how well a seismic attribute predicts a reservoir property, without overestimation on the training wells;
- choice of attributes for H_eff and for Phi separately: each property has its own best attribute and its own sensitivity;
- identification of zones with systematic errors (by horizon and zone);
- search for anomalous wells to re-check the data;
- checking whether the relationship is stable when the number of wells is reduced;
- a basis for deciding whether a multi-attribute model is worth using, or one attribute is enough.

---

## 🧭 Key method decisions

- **Model.** Linear regression (polynomial of degree 1) is used in all steps. Higher-degree fits and quantile regression are available as curves for visual comparison; the model used in cross-validation is always linear.
- **One horizon.** For a strict estimate, work with one horizon. With several horizons one well can fall into the training and the test set at once; see *Limitations*.
- **Residual sign.** Basic QC cross-plot (Step 1): Δ = PARAM_1 − PARAM_2. Fits and CV (Steps 1–4): Δ = Predicted − Actual. CV residual plots in Step 2: Actual − Predicted. The conventions differ on purpose; the sign is shown on every plot.
- **Best-scheme selection.** By default `score = RMSE + RMSE_fold_std` is minimised. The sidebar also offers RMSE, MAE, maximum R² (Actual = Predicted), RMSE + |Bias| and manual choice. Random K-Fold participates on purpose as a control scheme with a known optimistic estimate; interpret the «best» scheme together with the spatial ones.
- **WAPE instead of MAPE.** For parameters with many zero values (for example H_eff = 0 where there is no reservoir) MAPE is undefined. We use `WAPE = 100 · Σ|Δ| / Σ|y|`.
- **Step 5: model selection by RMSE_cv.** A multi-attribute model is accepted only if its RMSE_cv is lower than that of the best single-attribute model in the same CV scheme. In-sample R² is not used for selection.
- **Distance-based exclusion CV.** The exclusion radius is `excl_radius = median pairwise distance between wells × variogram_factor`. This is a distance heuristic, not a fitted variogram.

---

<a id="in-out-en"></a>

## 🔍 In-sample and out-of-sample in the project

**In-sample** — the model is fitted and checked on the same wells. The metrics show how well the model describes the data it has already seen, so they are usually optimistic. The in-sample equation is an ordinary linear regression over all wells of the chosen sample.

**Out-of-sample** — the model is fitted without some wells and checked on the wells it did not see during fitting (cross-validation by LOWO, K-fold and other schemes). The metrics show the quality on new wells. A gap between in-sample and out-of-sample is a sign of overfitting. In the Out-of-sample tables the metrics and the equation `Predicted = a + b·Actual` come from the same out-of-fold predictions.

| Step | What is computed | Mode |
|---|---|---|
| 0 | Pearson and Spearman correlations, descriptive statistics | no model |
| 1 | basic QC cross-plot, fitted curves, R²_lin, R²_fit, P-value, regression equation | in-sample |
| 2 | nine CV schemes, folds, choice of the best scheme, CV residual outliers, R² (Pearson), R² (Actual = Predicted), equation `Predicted = a + b·Actual`, fitted curves of the best scheme | out-of-sample |
| 3 | fitted curves and metrics inside each category | in-sample (within a category) |
| 4 | metrics before and after filtering, fitted curves after filtering | in-sample (on the filtered sample) |
| 4.5 | R² (Pearson) for series of random exclusion | in-sample (remaining wells) and out-of-sample (excluded wells) |
| 5 | single-attribute models and top-5 multi-attribute models | In-sample: RMSE_in, R²_in, r_in, equation over all wells; Out-of-sample: RMSE_cv, R²_cv, r_cv, equation `Predicted = a + b·Actual` |

---

## 🔧 Technology stack

### Interface and visualisation
- Streamlit
- Plotly (graph_objects, express)
- Seaborn (pairplot)

### Data processing
- NumPy, Pandas
- OpenPyXL (reading XLSX and exporting to Excel)

### Statistics and machine learning
- scikit-learn (LinearRegression, HuberRegressor, TheilSenRegressor, RANSACRegressor, PolynomialFeatures, KFold, LeaveOneGroupOut, r2_score)
- statsmodels (QuantReg, lowess)
- SciPy (stats.linregress, spatial.distance pdist / cdist, gaussian_kde)

### Architecture
- modular layout: calculations of approximations and CV schemes (`attribute_qc_full.py`), helpers and plots (`app_utils.py`), Streamlit scenario (`app_full.py`), interface translations (`translations_full.py`), in-app project reference (`ui_examples_full_ru.py`, `ui_examples_full_en.py`);
- heavy computations cached with `@st.cache_data` (cross-validation, fits, quantile fits, spacing statistics, Step 5);
- bilingual interface through the `RU` / `EN` dictionaries.

---

## 📊 Data

The input is a CSV or XLSX file. Column names are free: they are mapped in the sidebar.

| Column | Role | Description |
|---|---|---|
| `well_id` | required | Well identifier |
| `X`, `Y` | required | Well coordinates (`X_coord`, `Y_coord` in the code) |
| `H_eff` or `Phi` (PARAM_1) | required · Reference | Reference reservoir parameter: effective thickness or porosity. Chosen at the start of Step 1 |
| seismic attribute (PARAM_2) | required · Compared | Compared attribute, for example `rms_amplitude`, `envelope`, `sweetness`, `coherence`, `inst_frequency`, `avo_intercept`, `avo_gradient`, `spectral_decomp_20Hz`, `acoustic_impedance`, `q_factor`. Chosen at the start of Step 1 |
| `horizon_id` | optional | Horizon. Lets you choose one or several horizons; one horizon is recommended for a strict estimate |
| `tectonic_zone` | optional | Tectonic zone: a categorical column for Stratified by {column} CV and Step 3 |
| `reservoir_flag`, `fluid_type`, `well_category` | optional | Categorical features for Step 3 |
| any other numeric column | optional | Used in the correlation analysis (Step 0) and as a candidate in Step 5 |

Units of PARAM_1 and PARAM_2 are set when mapping the columns and are used only in labels.

> **The demo data are synthetic.** The attributes were not extracted from a seismic cube, and H_eff and Phi are modelled. Cross-plot results reflect the built-in structure, not real geology. For your own data, use files in the format above.

**Automatic cleaning on load:**
- garbage values (`NA`, `NULL`, `—`, `?`, etc.) are converted to NaN;
- commas are replaced with dots in numeric columns;
- column types are converted;
- rows with gaps in required columns are removed, and the number of removed rows is reported.

**Pipeline outputs:**
- the cleaned dataset and the input QC;
- QC cross-plot metrics and the approximation table;
- the summary table of CV schemes and fold details;
- residual statistics and outliers;
- accuracy breakdown by category;
- before / after comparison of well filtering;
- multi-attribute models for H_eff and Phi (Step 5).

---

## 🧩 Workflow

### Step 0 — Data loading and QC
- load CSV/XLSX or choose a demo file; map columns; automatic cleaning;
- input QC: full duplicates, duplicate `well_id` (a real duplicate means the same `well_id` and `horizon_id`), coincident coordinates, gaps;
- descriptive statistics of H_eff / Phi and the attributes: Count, μ, median, mode, σ, variance, CV %, min, Q05, Q25, Q75, Q95, max, range, IQR;
- correlation analysis: Pearson (linear relationships) and Spearman (monotonic relationships), pairplot (Seaborn or Plotly).

### Step 1 — Basic QC cross-plot (in-sample)
- choice of PARAM_1 (H_eff or Phi) and PARAM_2 (attribute); cross-plot with the Y = X line and linear regression;
- QC metrics: R²_lin, RMSE, MSE, MAE, Bias, Std, Variance, WAPE %, P-value, regression equation;
- 17 approximations and quantile regression (τ = 0.1 / 0.5 / 0.9), with a comparison table of R²_fit;
- distributions of PARAM_1 and PARAM_2 (histograms with KDE, boxplots);

### Step 2 — Cross-validation (out-of-sample)
- up to 9 CV schemes, with recommended parameters computed automatically in the sidebar;
- for each scheme: cross-plot Predicted vs Actual, metrics, fold statistics;
- summary table of all schemes, bar charts of RMSE, boxplot of fold RMSE spread;
- automatic choice of the best scheme by the sidebar criterion;
- fold tables (n_train, n_test, RMSE, MAE, Bias); skipped folds are shown as rows with n = 0;
- detailed review of the best scheme: cross-plot, approximations, CV residual distribution, outliers detected by four methods.

### Step 3 — Category diagnostics (in-sample within a category)
- works for any categorical column;
- pie and bar charts of well counts;
- statistics of PARAM_1 and PARAM_2 per category; boxplots by category;
- overlaid histograms by category, combined cross-plot for all categories and for a chosen subset;
- cross-plot and approximations for each category separately; comparison table of metrics by category.

### Step 4 — Well filtering (stress test)
- four modes: no filter, manual exclusion of wells, random exclusion by percentage (0–95 %), click selection of points on the cross-plot (click, Shift+click, Box or Lasso selection);
- the cross-plot, approximations and metric table are recalculated after filtering;
- before / after comparison: n wells, R²_lin, R²_fit, RMSE, MSE, MAE, Bias, Std, Variance, WAPE %, P-value;
- **Step 4.5 — random exclusion series:** levels 5–95 % in 5 % steps, 50–2000 sets per level (500 by default); for each set, in-sample R² (Pearson) on the remaining wells and out-of-sample R² on the excluded wells.

### Step 5 — Multi-attribute regression (in-sample and out-of-sample)
- a separate model for H_eff and for Phi; the H_eff section comes first, then Phi;
- full enumeration of attribute combinations from 1 to `k_max = min(3, number of candidates, max(1, ⌊N / 5⌋))`, where N is the number of wells; pairs of attributes with |r| > 0.8 never enter the same model;
- two well-based cross-validation schemes: **Leave-One-Well-Out** (N folds) and **K-fold by wells** with a fixed seed, K = max(3, round(N / 10));
- inside each section, four tables:
  1. **Single-attribute models — baseline for comparison In-sample (RMSE_in, R²_in).** Regression of the parameter on one attribute over all wells where both values exist; one table for both CV schemes;
  2. **Single-attribute models — baseline for comparison Out-of-sample (RMSE_cv, R²_cv).** Separately for LOWO and for K-fold;
  3. **Top-5 multi-attribute models In-sample (RMSE_in, R²_in).**
  4. **Top-5 multi-attribute models Out-of-sample (RMSE_cv, R²_cv).**
- equations: In-sample — regression over all wells; Out-of-sample — `Predicted = a + b·Actual` from out-of-fold predictions;
- selection criterion: RMSE_cv; a multi-attribute model is accepted only if it beats the best single-attribute model of the same scheme;
- coefficients (intercept, coefficients, standardised coefficients) are in an expandable block under the table;
- charts: attribute contribution of the best multi-attribute model; then a row of two charts for the best single attribute (In-sample | Out-of-sample) and a row for the best multi-attribute set (In-sample | Out-of-sample);
- all tables and coefficients are exported to Excel with a button under each table.

---

## 📈 Results

### What the user gets
- **An estimate of the «attribute → reservoir property» relationship** for H_eff and for Phi separately, with in-sample and out-of-sample results kept apart;
- **Attribute choice**: single-attribute ranking by RMSE_cv and the contribution of each attribute to a multi-attribute model;
- **A diagnosis of errors**: systematic bias (Bias), slope and intercept of the line `Predicted = a + b·Actual`;
- **A list of anomalous wells** — candidates for re-checking;
- **A breakdown by category** — where the relationship works worse (by horizon or zone);
- **A stability estimate** for reduced samples.

### Metric formulas

Here Δ is the residual, N the number of points, Δ̄ the mean residual (= Bias), and y the reference value (PARAM_1).

| Metric | Formula |
|---|---|
| R²_lin | `r²` (squared Pearson correlation, `linregress`) — a measure of linear strength only |
| R²_fit | `1 − Σ(Δ)² / Σ(y − ȳ)²` (valid for any curve shape) |
| R² (Pearson), Step 2 and Step 5 | `r²` on out-of-fold predictions: squared Pearson correlation of `Predicted` and `Actual` |
| RMSE | `√[ (1/N)·Σ(Δ)² ]` |
| MSE | `(1/N)·Σ(Δ)²` |
| MAE | `(1/N)·Σ\|Δ\|` |
| Bias | `(1/N)·Σ(Δ)` |
| Std | `√[ (1/(N−1))·Σ(Δᵢ − Δ̄)² ]` |
| Variance | `(1/(N−1))·Σ(Δᵢ − Δ̄)²` |
| WAPE, % | `100 · Σ\|Δ\| / Σ\|y\|` — defined even when some y are zero |
| P-value | `t = r·√[(N−2)/(1−r²)]`, `p = 2·(1 − F_t,N−2(\|t\|))`; Steps 0, 1, 2, 4 |

R²_lin and R²_fit are different quantities. They coincide for an OLS line but not for the other 16 curve types, so do not compare them directly.

### Outliers of the CV residuals

Outliers are detected in Step 2 on the CV residuals (Actual − Predicted) by four independent rules:

| Rule | Outlier condition | Sidebar parameter |
|---|---|---|
| Sigma | `\|Δ\| > σ_thr · σ`, σ — sample standard deviation (ddof = 1) | σ_thr: 1.0–6.0, default 2.0 |
| IQR | `Δ < Q25 − k·IQR` or `Δ > Q75 + k·IQR` | k: 1.0–3.0, default 1.5 |
| Modified z-score | `\|0.6745·(Δ − median)/MAD\| > 3.5` | fixed threshold 3.5 |
| Quantile | `Δ < P_low` or `Δ > P_high` | P_low: 0–20 %, default 5 %; P_high: 80–100 %, default 95 % |

---

## ⚠️ Limitations and assumptions

- **In-sample is optimistic.** In-sample metrics and equations describe the training data. For an estimate on new wells, look at Out-of-sample.
- **The data are synthetic.** The demo H_eff and Phi are modelled and the attributes are computed by formulas (for example, sweetness = envelope / √(instantaneous frequency)). Part of the correlations between attributes therefore follows from the formulas, not from independent relationships.
- **Several horizons at once.** Random K-Fold, Random K-Fold (buffered), Leave-P-Out and Spatial Leave-P-Out split well–horizon records rather than whole wells, so one well can end up in both training and test sets. For a strict estimate, use one horizon or LOWO / Block / Buffered.
- **Random K-Fold** ignores spatial correlation and takes part in choosing the «best» scheme as a control. Its result is not an estimate of real accuracy.
- **Spatial schemes** (Block, Buffered, Spatial Leave-P-Out) need good spatial coverage; on sparse or clustered layouts some blocks may be empty.
- **With few wells** (fewer than 15–20), LOWO is practically the only meaningful option. In Step 5, k_max is small (3 at 15 wells, 2 at 10, 1 at 5), and RMSE_cv estimates are unstable.
- **Many attributes raise the chance of a spuriously good training R².** That is why Step 5 selects by RMSE_cv, and every multi-attribute model must be physically meaningful, not only statistically better.
- **Quantile regression:** for τ = 0.1 and 0.9, R²_fit can be negative; judge such curves by the equation and Bias. The general approximation table uses only τ = 0.5.
- **Click selection of points** exists only in the interactive app (`app_full.py`).
- **Results depend on the data.** The tool gives diagnostics and reference values; it does not replace geological interpretation or validation on a real well set.

---

## 📁 Project structure

```text
05_Attribute_QC/
│
├── app_full.py                  — main Streamlit scenario (interface, sidebar, steps 0–5)
├── app_utils.py                 — helpers, plots and interface utilities
├── attribute_qc_full.py         — calculations: approximations, CV schemes, metrics, multi-attribute regression (Step 5)
├── translations_full.py         — RU / EN dictionaries
├── ui_examples_full_ru.py       — in-app project reference (RU)
├── ui_examples_full_en.py       — in-app project reference (EN)
│
├── data/                        — demo datasets (synthetic)
│   └── *.csv, *.xlsx
├── test_data_import/            — files for local import tests
│   └── *.csv, *.xlsx
│
├── attribute_qc_demo.ipynb      — demo notebook (reads data/attribute_wells_demo.csv)
├── attribute_qc_demo.pdf        — demo notebook as PDF
├── attribute_qc_plan.pdf        — project plan
├── attribute_qc_streamlit_demo.pdf — interface walkthrough
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
   cd Pet-projects/05_Attribute_QC
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
jupyter notebook attribute_qc_demo.ipynb
```

Run it from the project folder, so that the relative path `data/attribute_wells_demo.csv` resolves.

### Online

The app is also deployed at [attribute-qc-cv.streamlit.app](https://attribute-qc-cv.streamlit.app/).


---

## 📋 Input file requirements

**Required columns:** `well_id`, coordinates `X` and `Y`, the reference parameter (H_eff or Phi), a seismic attribute.

**Optional columns:** `horizon_id`, `tectonic_zone`, `reservoir_flag`, `fluid_type`, `well_category`, any additional numeric columns.

Column names are free: they are mapped in the sidebar. Automatic column detection by keywords (`well`, `x`, `y`, `H_eff`/`Heff`, `Phi`/`Kp`, `horizon`, `tectonic`/`zone`) is case-insensitive.

**Example:**

| well_id | X | Y | H_eff | rms_amplitude | horizon_id | tectonic_zone |
|---|---|---|---|---|---|---|
| Well_001 | 125000.0 | 745000.0 | 12.4 | 0.082 | H1 | Zone_A |
| Well_002 | 126500.0 | 746200.0 | 3.1 | 0.047 | H1 | Zone_A |
| Well_003 | 128200.0 | 748100.0 | 18.7 | 0.121 | H1 | Zone_B |

*The values in the example are illustrative only and show the format.*

---

## 🔗 Links

- Live app: https://attribute-qc-cv.streamlit.app/
- Repository: https://github.com/aspozd-DA-DS/Pet-projects
- Demo notebook: [attribute_qc_demo.ipynb](attribute_qc_demo.ipynb)
- Project plan: [attribute_qc_plan.pdf](attribute_qc_plan.pdf)
- Module reference (EN): [ui_examples_full_en.py](ui_examples_full_en.py)
- Module reference (RU): [ui_examples_full_ru.py](ui_examples_full_ru.py)

---

## 🏷 Topics

`Attribute QC` `Cross-Plot` `Cross-Validation` `In-sample` `Out-of-sample` `Seismic attributes`
`LOWO` `K-fold` `Block CV` `Buffered CV` `Spatial Leave-P-Out` `Distance-based CV`
`Multi-attribute regression` `Oil & Gas` `Well Data` `H_eff` `Porosity`
`Streamlit` `Plotly` `Python` `Pandas` `NumPy` `scikit-learn` `statsmodels`
`Machine Learning` `Data Science` `QC` `Regression` `Residual Analysis`
`Outlier Detection` `Bilingual UI` `RU/EN`

<p align="right"><a href="#top">⬆ back to top</a></p>
