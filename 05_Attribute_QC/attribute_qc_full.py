"""
attribute_qc_full.py

Модуль статистического анализа и кросс-валидации для оценки качества атрибутного анализа: зависимости эталонного параметра PARAM_1 (H_eff или Phi) в скважинах от сейсмического атрибута PARAM_2 в тех же скважинах. Zprm_well = PARAM_1, Zprm_map = PARAM_2 (внутренние имена колонок, задаются при сопоставлении).
Реализует следующие этапы анализа:
1. Базовые метрики и QC кросс-плот:
   - PARAM_1 vs PARAM_2 (Zprm_well vs Zprm_map), без кросс-валидации.
2. Кросс-валидация зависимости:
   - PARAM_2 → PARAM_1 (x = PARAM_2, y = PARAM_1).

   Используемые схемы:
   - Random K-Fold
   - Random K-Fold (buffered)
   - Leave-One-Well-Out
   - Block CV
   - Buffered CV
   - Distance-based exclusion CV
   - Stratified by Zone
   - Leave-P-Out
   - Spatial Leave-P-Out
3. Диагностика результатов:
   - распределение невязок;
   - карта невязок по координатам скважин (если в данных есть X и Y);
   - cross-plot по фолдам.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import pdist, cdist
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import r2_score
from sklearn.linear_model import (
    HuberRegressor,
    TheilSenRegressor,
    RANSACRegressor,
    LinearRegression
)


@dataclass
class BasicStats:
    n: int
    r2: float
    r: float
    slope: float
    intercept: float
    rmse: float
    mse: float
    mae: float
    bias: float
    std: float
    variance: float
    wape_pct: float
    p_value: float
    residuals: np.ndarray

@dataclass
class FitStats:
    r2: float
    rmse: float
    mse: float
    mae: float
    bias: float
    residuals: np.ndarray
    equation: str
    std: float
    variance: float
    wape_pct: float
    n_fit: int = 0
    r: float = np.nan

# ================================================================================ #
# Basic QC cross-plot — сравнение y vs x. Используется в Шаге 1 и Шаге 2
# Обозначения: x = PARAM_2 (сравниваемое значение) и y = PARAM_1 (эталонное значение)
# В Шаге 1: x = PARAM_2 (атрибут, Zprm_map), y = PARAM_1 (H_eff или Phi, Zprm_well).
# Используется:
# - Шаг 1: PARAM_1 vs PARAM_2
# - Шаг 2: Predicted vs Actual
# Шаг 1:
# - Калибровочное уравнение: PARAM_1 = f(PARAM_2)
# - Невязки: Residual = PARAM_1 - PARAM_2
# - Положительная невязка: PARAM_1 > PARAM_2 (значение эталонного параметра выше сравниваемого в его единицах)
# - Отрицательная невязка: PARAM_1 < PARAM_2
# - Ограничение: Bias, RMSE, MSE, MAE, Std, Variance, WAPE % зависят от масштаба и единиц PARAM_1 и PARAM_2; сравнимы только внутри одной пары. Качество связи оценивать по R² и корреляции (Пирсона, Спирмена)
# - Рассчитываемые метрики:
#   R², RMSE, MSE, MAE, Bias
#   Std, Variance, WAPE %
#   p-value, коэффициенты регрессии
# - Возвращает объект BasicStats
# Шаг 2:
# - Невязки: Residual = y − x = Actual − Predicted (x = Predicted, y = Actual; так вызывает app_full)
# - Кривые аппроксимации (compute_fit_stats) используют pred − y (Predicted − Actual)
# - Возвращает объект BasicStats
# ================================================================================ #

def _validate_xy(x: np.ndarray, y: np.ndarray,) -> None:

    if len(x) != len(y):
        raise ValueError("X and Y must have the same length")

def _compute_basic_stats(x: np.ndarray, y: np.ndarray) -> BasicStats:

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    _validate_xy(x, y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 2:
        raise ValueError("Not enough data")
    
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        slope = intercept = r = p = np.nan
    else:
        slope, intercept, r, p, se = stats.linregress(x, y)

    residuals = y - x
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mse = float(np.mean(residuals ** 2))
    mae = float(np.mean(np.abs(residuals)))
    bias = float(np.mean(residuals))
    std = (float(np.std(residuals, ddof=1)) if n > 1 else 0.0)
    variance = (float(np.var(residuals, ddof=1)) if n > 1 else 0.0)
    wape_pct = float(100 * np.sum(np.abs(residuals)) / np.sum(np.abs(y))) if np.sum(np.abs(y)) > 0 else np.nan

    return BasicStats(
        n=n,
        r2=r ** 2,
        r=r,
        slope=slope,
        intercept=intercept,
        rmse=rmse,
        mse=mse,
        mae=mae,
        bias=bias,
        std=std,
        variance=variance,
        wape_pct=wape_pct,
        p_value=float(p),
        residuals=residuals,
    )

#  Basic QC cross‑plot — сравнение y vs x в шаге 1 
def basic_crossplot_stats(df: pd.DataFrame) -> BasicStats:
    return _compute_basic_stats(_x(df), _y(df),)

#  Basic QC cross‑plot для любых массивов X и Y  (шаг 2)
def basic_crossplot_stats_xy(x: np.ndarray, y: np.ndarray) -> BasicStats:
    return _compute_basic_stats(x, y)

# ================================================================================ #
# Серия случайных исключений скважин (Шаг 4): распределение Pearson r² по уровням
# ================================================================================ #
RAND_SERIES_LEVELS = tuple(range(5, 96, 5))   # 5, 10, ..., 95 %


def _pearson_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r² (= R²_lin шага 4). NaN, если точек < 3 или нет разброса."""
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    r = np.corrcoef(x, y)[0, 1]
    return float(r * r)


def _oos_r2(x_tr, y_tr, x_te, y_te) -> float:
    """Out-of-sample: линия обучена на train, r² (Pearson, pred vs fact) на test."""
    if len(x_tr) < 3 or len(x_te) < 3 or np.ptp(x_tr) == 0:
        return np.nan
    slope, intercept, *_ = stats.linregress(x_tr, y_tr)
    pred = intercept + slope * x_te
    return _pearson_r2(y_te, pred)

def _fmt_count(vals) -> str:
    """Одно число, если у всех наборов совпадает; иначе диапазон 'min–max'."""
    lo, hi = int(min(vals)), int(max(vals))
    return str(lo) if lo == hi else f"{lo}–{hi}"


def _level_summary(vals, pct, n_excl, n_left, mode, n_tr, n_te) -> dict:
    s = pd.Series(vals, dtype=float).dropna()
    ok = len(s) > 0
    return {
        "mode": mode,
        "level_pct": pct,
        "n_excluded": n_excl,
        "n_remaining": n_left,
        "n_train": _fmt_count(n_tr),
        "n_test": _fmt_count(n_te),
        "n_sets": int(len(s)),
        "status": "ok" if ok else "no_valid",
        "min": s.min() if ok else np.nan,
        "max": s.max() if ok else np.nan,
        "mean": s.mean() if ok else np.nan,
        "median": s.median() if ok else np.nan,
    }

def random_exclusion_series(df, levels=RAND_SERIES_LEVELS, n_repeats=500, seed=42, min_remaining=5):
    """
    Для каждого уровня и набора скважин (целиком) считает два r²:
      r2_in  — in-sample: по оставшимся скважинам;
      r2_out — out-of-sample: линия по оставшимся, r² по исключённым.
    Возвращает long_df (по набору) и summary_df (две строки на уровень: mode = "in" / "out").
    """
    codes, uniques = pd.factorize(df["well_id"].astype(str))
    n_wells = len(uniques)
    x_all = _x(df)
    y_all = _y(df)

    long_rows, summary_rows = [], []

    for pct in levels:
        n_excl = int(round(n_wells * pct / 100))
        n_left = n_wells - n_excl

        if n_left < min_remaining:
            for mode in ("in", "out"):
                summary_rows.append({
                    "mode": mode, "level_pct": pct, "n_excluded": n_excl,
                    "n_remaining": n_left, "n_train": "—", "n_test": "—",
                    "n_sets": 0, "status": "skipped",
                    "min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan,
                })
            continue

        rng = np.random.default_rng([seed, int(pct)])
        r2_in_vals, r2_out_vals, n_tr, n_te = [], [], [], []

        for k in range(1, n_repeats + 1):
            excl = rng.choice(n_wells, size=n_excl, replace=False)
            test = np.isin(codes, excl)
            train = ~test

            r2_in = _pearson_r2(x_all[train], y_all[train])
            r2_out = _oos_r2(x_all[train], y_all[train], x_all[test], y_all[test])

            r2_in_vals.append(r2_in)
            r2_out_vals.append(r2_out)
            n_tr.append(int(train.sum()))
            n_te.append(int(test.sum()))

            long_rows.append({
                "level_pct": pct, "set_no": k,
                "n_train": int(train.sum()), "n_test": int(test.sum()),
                "r2_in": r2_in, "r2_out": r2_out,
            })

        summary_rows.append(_level_summary(r2_in_vals, pct, n_excl, n_left, "in", n_tr, n_te))
        summary_rows.append(_level_summary(r2_out_vals, pct, n_excl, n_left, "out", n_tr, n_te))

    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)

# ================================================================================ #
# FitStats — аппроксимации зависимости y(x) в Шагах 1, 2 и 3
# ================================================================================ #
# - принимает: x = сравниваемый параметр (PARAM_2) и y = эталонный параметр (PARAM_1)
# - строит выбранную аппроксимацию: линейная, квадратичная, кубическая, полиномы до 10 степени, степенная, экспонента, логарифм, robust Huber, Theil-Sen, RANSAC, LOWESS, квантильная регрессия
# - рассчитывает метрики: R², RMSE, MSE, MAE, Bias, Std, Variance, WAPE %, уравнение регрессии
# - возвращает объект FitStats

POLY_APPROX = {
    "Linear": 1,
    "Quadratic": 2,
    "Cubic": 3,
    "Polynomial Degree 4": 4,
    "Polynomial Degree 5": 5,
    "Polynomial Degree 6": 6,
    "Polynomial Degree 7": 7,
    "Polynomial Degree 8": 8,
    "Polynomial Degree 9": 9,
    "Polynomial Degree 10": 10,
}


def polynomial_equation(coeffs: np.ndarray) -> str:
    degree = len(coeffs) - 1
    parts = []

    for i, c in enumerate(coeffs):

        power = degree - i
        coef = abs(c)

        if power == 0:
            term = f"{coef:.5f}"
        elif power == 1:
            term = f"{coef:.5f}·x"
        else:
            term = f"{coef:.5f}·x^{power}"

        if i == 0:
            if c < 0:
                parts.append(f"-{term}")
            else:
                parts.append(term)
        else:
            if c < 0:
                parts.append(f"- {term}")
            else:
                parts.append(f"+ {term}")

    return "y = " + " ".join(parts)

def compute_fit_stats(x: np.ndarray, y: np.ndarray, approx_type: str) -> FitStats:

    # ----- ЗАЩИТА ОТ NaN и INF ----- #
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    # -----  ЗАЩИТА ОТ ПУСТЫХ МАССИВОВ ----- #
    if len(x) < 3:
        raise ValueError("Not enough data")
    if len(np.unique(x)) < 2:
        raise ValueError("Not enough unique X values")

    # -----  ПОЛИНОМЫ ----- #
    if approx_type in POLY_APPROX:

        degree = POLY_APPROX[approx_type]

        if len(x) < degree + 1:
            raise ValueError(f"Not enough points for degree {degree} polynomial")

        coeffs = np.polyfit(x, y, degree)

        poly = np.poly1d(coeffs)
        pred = poly(x)
        eq = polynomial_equation(coeffs)

    # -----  СТЕПЕННАЯ ----- #
    elif approx_type == "Power Law":
        mask = (x > 0) & (y > 0)
        if mask.sum() < 3:
            raise ValueError("Not enough data for power-law fit")

        logx = np.log(x[mask])
        logy = np.log(y[mask])

        if np.any(np.isinf(logx)) or np.any(np.isinf(logy)):
            raise ValueError("Log transformation failed")

        b, log_a = np.polyfit(logx, logy, 1)
        a = np.exp(log_a)
        x = x[mask]
        y = y[mask]
        pred = a * (x ** b)
        eq = f"y = {a:.5f}·x^{b:.5f}"

    # ----- ЭКСПОНЕНТА ----- #
    elif approx_type == "Exponential":
        mask = y > 0
        if mask.sum() < 3:
            raise ValueError("Not enough data for exponential fit")

        logy = np.log(y[mask])
        if np.any(np.isinf(logy)):
            raise ValueError("Log transformation failed")

        coeffs = np.polyfit(x[mask], logy, 1)
        a = np.exp(coeffs[1])
        b = coeffs[0]
        x = x[mask]
        y = y[mask]
        pred = a * np.exp(b * x)
        eq = f"y = {a:.5f}·exp({b:.5f}·x)"

    # ----- ЛОГАРИФМ ----- #
    elif approx_type == "Logarithmic":
        mask = x > 0
        if mask.sum() < 3:
            raise ValueError("Not enough data for logarithmic fit")

        logx = np.log(x[mask])
        if np.any(np.isinf(logx)):
            raise ValueError("Cannot apply log transformation")

        coeffs = np.polyfit(logx, y[mask], 1)
        a = coeffs[0]
        b = coeffs[1]
        x = x[mask]
        y = y[mask]
        pred = a * np.log(x) + b
        eq = (
            f"y = {a:.5f}·ln(x) "
            f"{'+' if b >= 0 else '-'} "
            f"{abs(b):.5f}"
              )

    # ----- ROBUST HUBER ----- #
    elif approx_type == "Robust (Huber)":
        model = HuberRegressor().fit(x.reshape(-1, 1), y)
        pred = model.predict(x.reshape(-1, 1))
        eq = f"y = {model.coef_[0]:.5f}·x + {model.intercept_:.5f}"

    # ----- THEIL-SEN ----- #
    elif approx_type == "Theil-Sen":
        model = TheilSenRegressor(random_state=42)
        model.fit(x.reshape(-1, 1), y)
        pred = model.predict(x.reshape(-1, 1))
        eq = (
            f"y = {model.coef_[0]:.5f}·x + "
            f"{model.intercept_:.5f}"
        )

    # ----- RANSAC ----- #
    elif approx_type == "RANSAC":
        model = RANSACRegressor(estimator=LinearRegression(), random_state=42)
        model.fit(x.reshape(-1, 1), y)
        pred = model.predict(x.reshape(-1, 1))
        coef = model.estimator_.coef_[0]
        intercept = model.estimator_.intercept_
        eq = (
            f"y = {coef:.5f}·x + "
            f"{intercept:.5f}"
        )

    # ----- LOWESS ----- #
    elif approx_type == "LOWESS":
        if len(x) < 3:
            raise ValueError("Not enough data for LOWESS")
        lowess = sm.nonparametric.lowess(y, x, frac=0.3)
        pred = np.interp(x, lowess[:, 0], lowess[:, 1])
        eq = "LOWESS (locally weighted regression)"

    # ----- КВАНТИЛЬНАЯ РЕГРЕССИЯ (τ = 0.5) ----- #
    elif approx_type == "Quantile Regression (τ=0.1/0.5/0.9)":

        # считаем метрики только для центральной кривой τ=0.5
        tau = 0.5
        model = sm.QuantReg(y, sm.add_constant(x))
        res = model.fit(q=tau)

        pred = res.predict(sm.add_constant(x))
        eq = f"y = {res.params[1]:.5f}·x + {res.params[0]:.5f}"

    else:
        raise ValueError(f"Unknown approximation type: {approx_type}")

    # ----- МЕТРИКИ  ----- #
    resid = pred - y
    mse = float(np.mean(resid ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    r2 = float(r2_score(y, pred))
    std = (float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0)
    variance = (float(np.var(resid, ddof=1)) if len(resid) > 1 else 0.0)
    wape_pct = float(100 * np.sum(np.abs(resid)) / np.sum(np.abs(y))) if np.sum(np.abs(y)) > 0 else np.nan
    return FitStats(
        r2=r2, r=float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 and np.std(y) > 0 else np.nan, 
        rmse=rmse, mse=mse, mae=mae, bias=bias, residuals=resid, equation=eq,
        std=std, variance=variance, wape_pct=wape_pct, n_fit=len(x),
    )

# ----- Quantile Regression — метрики для отдельной τ ----- 
def compute_quantile_stats(x: np.ndarray, y: np.ndarray, tau: float) -> FitStats:
    """Метрики для одной конкретной τ квантильной регрессии."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        raise ValueError("Not enough data")

    model = sm.QuantReg(y, sm.add_constant(x))
    res = model.fit(q=tau)
    pred = res.predict(sm.add_constant(x))
    eq = f"y = {res.params[1]:.5f}·x + {res.params[0]:.5f}"

    resid = pred - y
    mse = float(np.mean(resid ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    r2 = float(r2_score(y, pred))
    std = (float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0)
    variance = (float(np.var(resid, ddof=1)) if len(resid) > 1 else 0.0)
    wape_pct = float(100 * np.sum(np.abs(resid)) / np.sum(np.abs(y))) if np.sum(np.abs(y)) > 0 else np.nan

    return FitStats(
        r2=r2, r=float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 and np.std(y) > 0 else np.nan,
        rmse=rmse, mse=mse, mae=mae, bias=bias, residuals=resid, equation=eq,
        std=std, variance=variance, wape_pct=wape_pct, n_fit=len(x), 
    )

# ================================================================================
# Calibration Model
# ================================================================================
# - Модель оценивает связь y = f(x)
# - При кросс-валидации значения x считаются фиксированными.
# - Исключаются только наблюдения из обучения модели.
# - Параметр degree задаёт степень полинома по x.

def _fit_calibration(x: np.ndarray, y: np.ndarray, degree: int = 1):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly.fit_transform(x.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X, y)
    model._poly = poly

    return model

def _predict_calibration(model, x: np.ndarray):
    X = model._poly.transform(x.reshape(-1, 1))
    return model.predict(X)

def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:

    resid = y_pred - y_true

    if len(resid) == 0:
        return {
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "r2": np.nan
        }

    if len(y_true) >= 3 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r_pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        r2 = r_pearson ** 2
    else:
        r2 = np.nan

    return {
        "n": len(y_true),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(resid)),
        "r2": float(r2),
    }

# Построение пространственных блоков для Block CV
# - Создаёт регулярную сетку grid_n × grid_n.
# - Каждой точке присваивается идентификатор пространственного блока.
def build_spatial_blocks(df: pd.DataFrame, grid_n: int = 3) -> np.ndarray:

    if grid_n < 1:
        raise ValueError("grid_n must be at least 1")

    x_edges = np.linspace(df["X_coord"].min(), df["X_coord"].max() + 1e-6, grid_n + 1)
    y_edges = np.linspace(df["Y_coord"].min(), df["Y_coord"].max() + 1e-6, grid_n + 1)

    bx = np.clip(
        np.digitize(df["X_coord"].to_numpy(float), x_edges) - 1,
        0,
        grid_n - 1
    )
    by = np.clip(
        np.digitize(df["Y_coord"].to_numpy(float), y_edges) - 1,
        0,
        grid_n - 1
    )

    return bx * grid_n + by

def run_fold(
    x: np.ndarray,
    y: np.ndarray,
    preds: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    degree: int
):
    if len(tr) < degree + 2:
        raise ValueError("Not enough data for training")

    model = _fit_calibration(x[tr], y[tr], degree)
    p = _predict_calibration(model, x[te])
    preds[te] = p

    return p, fold_metrics(y[te], p)

# ================================================================================ #
# Кросс-валидация: схемы CV
# ================================================================================ #
# Реализованы следующие схемы кросс-валидации:
# - Random K-Fold
# - Random K-Fold (buffered)
# - Leave-One-Well-Out
# - Leave-P-Out
# - Block CV
# - Buffered CV
# - Spatial Leave-P-Out
# - Distance-based exclusion CV
# - Stratified by Zone (по категориальной колонке: любая колонка с хотя бы двумя значениями, по умолчанию tectonic_zone)
# Каждая схема:
# - делит данные на train/test
# - обучает калибровочную модель y = f(x)
# - предсказывает y
# - рассчитывает метрики для каждого фолда
# - возвращает preds + таблицу fold‑метрик

def _x(df: pd.DataFrame) -> np.ndarray:
    return df["Zprm_map"].to_numpy(float)

def _y(df: pd.DataFrame) -> np.ndarray:
    return df["Zprm_well"].to_numpy(float)

def _well_group_folds(df: pd.DataFrame, k: int, seed: int):
    """K-Fold по уникальным скважинам: все строки скважины попадают либо в обучение, либо в тест."""
    codes, uniq = pd.factorize(df["well_id"].astype(str))
    k = min(k, len(uniq))
    if k < 2:
        raise ValueError("At least two wells are required for cross-validation")
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for tr_w, te_w in kf.split(np.arange(len(uniq))):
        yield np.where(np.isin(codes, tr_w))[0], np.where(np.isin(codes, te_w))[0]

# -- Random K‑Fold
def cv_random_kfold(df: pd.DataFrame, degree: int = 1, k: int = 5, seed: int = 42):

    # _x: PARAM_2 (сравниваемый атрибут, Zprm_map); _y: PARAM_1 (эталонный параметр, Zprm_well).

    x = _x(df)
    y = _y(df)
    
    preds = np.full(len(df), np.nan)
    rows = []

    for i, (tr, te) in enumerate(_well_group_folds(df, k, seed)):
        if len(tr) < degree + 2:
            rows.append({"fold": f"fold_{i+1}_skipped (n_train={len(tr)} < {degree + 2})",
                         "n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "n_train": len(tr)})
            continue

        p, m = run_fold(x, y, preds, tr, te, degree )
        m["fold"] = f"fold_{i+1}"
        m["n_train"] = len(tr) 
        rows.append(m)

    return preds, pd.DataFrame(rows)

# -- Random K-Fold с буфером (без утечки соседних скважин)
def cv_random_kfold_buffered(df: pd.DataFrame, degree: int = 1, k: int = 5,
                             seed: int = 42, buffer: float = 300.0):

    x = _x(df)
    y = _y(df)
    xy = df[["X_coord", "Y_coord"]].to_numpy(float)

    preds = np.full(len(df), np.nan)
    rows = []

    for i, (tr, te) in enumerate(_well_group_folds(df, k, seed)):
        # расстояние каждой обучающей точки до ближайшей тестовой
        d_min = cdist(xy[tr], xy[te]).min(axis=1)
        tr = tr[d_min > buffer]

        if len(tr) < degree + 2:
            rows.append({"fold": f"buf_fold_{i+1}_skipped (n_train={len(tr)} < {degree + 2})",
                         "n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "n_train": len(tr)})
            continue

        p, m = run_fold(x, y, preds, tr, te, degree)
        m["fold"] = f"buf_fold_{i+1}"
        m["n_train"] = len(tr)
        rows.append(m)

    return preds, pd.DataFrame(rows)

# -- Leave‑One‑Well‑Out
def cv_lowo(df: pd.DataFrame, degree: int = 1):

    x = _x(df)
    y = _y(df)

    groups = df["well_id"].to_numpy()

    logo = LeaveOneGroupOut()
    preds = np.full(len(df), np.nan)
    rows = []

    for tr, te in logo.split(x, y, groups=groups):
        if len(tr) < degree + 2:
            rows.append({"fold": f"{groups[te][0]}_skipped (n_train={len(tr)} < {degree + 2})",
                         "n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "n_train": len(tr)})
            continue
        p, m = run_fold(x, y, preds, tr, te, degree)
        m["fold"] = str(groups[te][0])
        m["n_train"] = len(tr)
        rows.append(m)

    return preds, pd.DataFrame(rows)

# -- Leave-P-Out CV
def cv_leave_p_out(df: pd.DataFrame, degree: int, p_percent: int = 10, repeats: int = 50, seed: int = 42):

    rng = np.random.default_rng(seed)
    n = len(df)
    x = _x(df)
    y = _y(df)

    preds_sum = np.zeros(n, dtype=float)
    preds_count = np.zeros(n, dtype=int)
    fold_rows = []

    codes, uniq = pd.factorize(df["well_id"].astype(str))
    if len(uniq) < 2:
        raise ValueError("At least two wells are required for Leave-P-Out")
    n_wells = len(uniq)
    p_wells = max(1, int(np.ceil(n_wells * p_percent / 100)))
    if p_wells >= n_wells:
        p_wells = n_wells - 1

    for fold in range(repeats):

        test_wells = rng.choice(n_wells, size=p_wells, replace=False)
        test_idx = np.where(np.isin(codes, test_wells))[0]
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        if len(train_idx) < degree + 2:
            fold_rows.append({"fold": f"{fold + 1}_skipped (n_train={len(train_idx)} < {degree + 2})",
                              "n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan,
                              "n_train": len(train_idx), "n_test": len(test_idx)})
            continue
        model = _fit_calibration(x[train_idx], y[train_idx], degree,)
        pred = _predict_calibration(model, x[test_idx],)
        preds_sum[test_idx] += pred
        preds_count[test_idx] += 1
        m = fold_metrics(y[test_idx], pred,)
        m["fold"] = fold + 1
        m["n_train"] = len(train_idx)
        m["n_test"] = len(test_idx)
        fold_rows.append(m)

    preds = np.full(n, np.nan)
    mask = preds_count > 0
    preds[mask] = (preds_sum[mask] / preds_count[mask])

    return preds, pd.DataFrame(fold_rows)

# -- Block CV
def cv_blocked(df: pd.DataFrame, degree: int = 1, grid_n: int = 3):

    x = _x(df)
    y = _y(df)

    block_id = build_spatial_blocks(df, grid_n)

    preds = np.full(len(df), np.nan)
    rows = []

    for b in np.unique(block_id):
        te = np.where(block_id == b)[0]
        tr = np.where(block_id != b)[0]

        if len(te) == 0 or len(tr) < degree + 2:
            if len(te) > 0:
                rows.append({"fold": f"block_{b}_skipped (n_train={len(tr)} < {degree + 2})",
                             "n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "n_train": len(tr)})
            continue

        p, m = run_fold(x, y, preds, tr, te, degree)
        m["fold"] = f"block_{b}"
        m["n_train"] = len(tr)
        rows.append(m)

    return (preds, pd.DataFrame(rows), block_id)

# -- Buffered CV: исключаются точки, расположенные в радиусе buffer вокруг тестового блока
def cv_buffered(df: pd.DataFrame, degree: int = 1, grid_n: int = 3, buffer: float = 300.0):

    x = _x(df)
    y = _y(df)

    block_id = build_spatial_blocks(df, grid_n)

    preds = np.full(len(df), np.nan)
    rows = []

    well_coords = df[["X_coord", "Y_coord"]].to_numpy(float)

    for b in np.unique(block_id):
        te = np.where(block_id == b)[0]

        if len(te) == 0:
            continue

        dist = np.min(
            np.linalg.norm(well_coords[:, None, :] - well_coords[te][None, :, :], axis=2),
            axis=1
        )

        tr = np.where((block_id != b)& (dist > buffer))[0]

        if len(tr) < degree + 2:
            rows.append({"fold": f"buffered_block_{b}_skipped (n_train={len(tr)} < {degree + 2})",
                         "n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "n_train": len(tr)})
            continue

        p, m = run_fold(x, y, preds, tr, te, degree)
        m["fold"] = (f"buffered_block_{b}")
        m["n_train"] = len(tr)
        rows.append(m)

    return (preds, pd.DataFrame(rows), block_id)

# -- Spatial Leave-P-Out
def cv_spatial_leave_p_out(df: pd.DataFrame, degree: int, radius: float = 1000.0, repeats: int = 50, seed: int = 42):

    rng = np.random.default_rng(seed)
    n = len(df)

    x = _x(df)
    y = _y(df)

    coords = df[["X_coord", "Y_coord"]].to_numpy(float)
    preds_sum = np.zeros(n, dtype=float)
    preds_count = np.zeros(n, dtype=int)
    fold_rows = []

    codes, uniq = pd.factorize(df["well_id"].astype(str))
    if len(uniq) < 2:
        raise ValueError("At least two wells are required for Spatial Leave-P-Out")
    for fold in range(repeats):
        center_well = rng.integers(len(uniq))
        center_idx = np.where(codes == center_well)[0][0]
        center = coords[center_idx:center_idx + 1]
        dist = cdist(center, coords)[0]

        test_idx = np.where(dist <= radius)[0]
        train_idx = np.where(dist > radius)[0]
        if len(train_idx) < degree + 2:
            fold_rows.append({"fold": f"{fold + 1}_skipped (n_train={len(train_idx)} < {degree + 2})",
                              "n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "n_train": len(train_idx)})
            continue
        if len(test_idx) < 1:
            continue

        model = _fit_calibration(x[train_idx], y[train_idx], degree,)
        pred = _predict_calibration(model, x[test_idx],)
        preds_sum[test_idx] += pred
        preds_count[test_idx] += 1
        m = fold_metrics(y[test_idx], pred,)
        m["fold"] = fold + 1
        m["n_train"] = len(train_idx)
        m["n_test"] = len(test_idx)
        fold_rows.append(m)

    preds = np.full(n, np.nan)
    mask = preds_count > 0
    preds[mask] = (preds_sum[mask] / preds_count[mask])

    return preds, pd.DataFrame(fold_rows)

# -- Distance-based exclusion CV: для тестовой скважины исключаются все обучающие наблюдения, расположенные ближе радиуса исключения excl_radius

# Медианное попарное расстояние между скважинами (не вариограмма и не диапазон корреляции).
# Используется как база для radius исключения: excl_radius = median_pairwise_distance * variogram_factor
def median_pairwise_distance(coords: np.ndarray) -> float:
    distances = pdist(coords)

    if len(distances) == 0:
        return 0.0

    return float(np.median(distances))

# Distance-based exclusion CV
def cv_distance_exclusion(
    df: pd.DataFrame,
    degree: int = 1,
    variogram_factor: float = 1.0,
    excl_radius_mode: str = "median_factor",     # "median_factor" | "absolute"
    excl_radius_abs: float | None = None,        # метры, если mode == "absolute"
):

    x = _x(df)
    y = _y(df)

    coords = df[["X_coord", "Y_coord"]].to_numpy(float)

    # --- выбор способа задания радиуса исключения ---
    if excl_radius_mode == "absolute":
        if excl_radius_abs is None:
            raise ValueError("excl_radius_abs must be set when excl_radius_mode='absolute'")
        excl_radius = float(excl_radius_abs)
    else:  # "median_factor" — поведение по умолчанию, как раньше
        coords_wells = df.drop_duplicates(subset=["well_id"])[["X_coord", "Y_coord"]].to_numpy(float)
        excl_radius = median_pairwise_distance(coords_wells) * variogram_factor
    preds = np.full(len(df), np.nan)
    rows = []

    for i in range(len(df)):

        dist = np.linalg.norm(coords - coords[i], axis=1)
        tr = np.where(dist > excl_radius)[0]
        te = np.array([i])

        if len(tr) < degree + 2:
            # явный пропуск: прогноз для точки остаётся NaN, в таблице фолдов фиксируется причина
            rows.append({
                "fold": f"dist_excl_{i}_skipped (n_train={len(tr)} < {degree + 2})",
                "n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "n_train": len(tr),
            })
            continue

        p, m = run_fold(x, y, preds, tr, te, degree)
        m["fold"] = f"dist_excl_{i}"
        m["n_train"] = len(tr)
        rows.append(m)

    return preds, pd.DataFrame(rows)

# -- Stratified by Zone (по категориальной колонке, по умолчанию tectonic_zone)
def cv_stratified_zone(
    df: pd.DataFrame,
    degree: int = 1,
    zone_col: str | None = "tectonic_zone",
):
    if zone_col is None or zone_col not in df.columns:
        return None, None

    # Колонка может быть и категориальной, и числовой (horizon_id часто int).
    # Приводим к строке, чтобы NaN и разные типы не ломали groups в LeaveOneGroupOut.

    x = _x(df)
    y = _y(df)

    zones = df[zone_col].astype(str).to_numpy()   # приведение, чтобы NaN/типы не ломали groups
    if len(np.unique(zones)) < 2:
        return None, None

    logo = LeaveOneGroupOut()
    preds = np.full(len(df), np.nan)
    rows = []

    for tr, te in logo.split(x, y, groups=zones):
        if len(tr) < degree + 2:
            rows.append({"fold": f"{zones[te][0]}_skipped (n_train={len(tr)} < {degree + 2})",
                         "n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "n_train": len(tr)})
            continue
        p, m = run_fold(x, y, preds, tr, te, degree)
        m["fold"] = str(zones[te][0])
        m["n_train"] = len(tr)
        rows.append(m)

    return preds, pd.DataFrame(rows)

# ================================================================================ #
# Сводка результатов кросс-валидации CV
# ================================================================================ #
# Считает:
# - RMSE, MAE и Bias по всем прогнозируемым точкам
# Дополнительная статистика по фолдам:
# - средний RMSE по фолдам
# - медианный RMSE по фолдам
# - минимальный RMSE по фолдам
# - максимальный RMSE по фолдам
# - стандартное отклонение Std RMSE по фолдам
# - диапазон Range RMSE по фолдам
# ================================================================================ #

def _n_test_stats(fold_df) -> dict:
    if fold_df is None or fold_df.empty or "n" not in fold_df.columns:
        return {"n_test_min": np.nan, "n_test_max": np.nan}
    n = fold_df["n"].to_numpy(float)
    n = n[n > 0]   # пропущенные фолды с n=0 не учитываем
    if len(n) == 0:
        return {"n_test_min": np.nan, "n_test_max": np.nan}
    return {"n_test_min": float(np.min(n)), "n_test_max": float(np.max(n))}

def _r2_fold_stats(fold_df) -> dict:
    keys = ["r2_fold_mean", "r2_fold_median", "r2_fold_min",
            "r2_fold_max", "r2_fold_std"]
    if fold_df is None or fold_df.empty or "r2" not in fold_df.columns:
        return {k: np.nan for k in keys}
    v = fold_df["r2"].dropna().to_numpy(float)
    if len(v) == 0:
        return {k: np.nan for k in keys}
    return {
        "r2_fold_mean": float(np.mean(v)),
        "r2_fold_median": float(np.median(v)),
        "r2_fold_min": float(np.min(v)),
        "r2_fold_max": float(np.max(v)),
        "r2_fold_std": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
    }

def summarize_cv(preds: np.ndarray, y_true: np.ndarray, fold_df: pd.DataFrame) -> dict:

    mask = ~np.isnan(preds)

    if mask.sum() == 0:
        return {
            "r2_actual_pred": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "rmse_fold_mean": np.nan,
            "rmse_fold_median": np.nan,
            "rmse_fold_min": np.nan,
            "rmse_fold_max": np.nan,
            "rmse_fold_std": np.nan,
            "rmse_fold_range": np.nan,
            "n_pred": 0,
            "r2_pearson_pooled": np.nan,
            **_r2_fold_stats(None),
            **_n_test_stats(None),
        }

    resid = preds[mask] - y_true[mask]
    if fold_df is None or fold_df.empty or "rmse" not in fold_df.columns:
        fold_rmse = np.array([], dtype=float)
    else:
        fold_rmse = fold_df["rmse"].dropna().to_numpy(float)

    if len(fold_rmse) == 0:
        rmse_fold_mean = np.nan
        rmse_fold_median = np.nan
        rmse_fold_min = np.nan
        rmse_fold_max = np.nan
        rmse_fold_std = np.nan
        rmse_fold_range = np.nan

    else:
        rmse_fold_mean = float(np.mean(fold_rmse))
        rmse_fold_median = float(np.median(fold_rmse))
        rmse_fold_min = float(np.min(fold_rmse))
        rmse_fold_max = float(np.max(fold_rmse))

        rmse_fold_std = (
            float(np.std(fold_rmse, ddof=1))
            if len(fold_rmse) > 1
            else 0.0
        )

        rmse_fold_range = (rmse_fold_max - rmse_fold_min)

    # R² для линии Actual = Predicted (через линейную регрессию)
    # r2_score(y_true, y_pred) = 1 - SS_res / SS_tot
    y_true_masked = y_true[mask]
    y_pred_masked = preds[mask]
    try:
        r2_lin = float(r2_score(y_true_masked, y_pred_masked))
    except Exception:
        r2_lin = float("nan")

    try:
        r_pooled = float(np.corrcoef(y_true[mask], preds[mask])[0, 1])
        r2_pearson_pooled = r_pooled ** 2 if mask.sum() >= 3 else np.nan
    except Exception:
        r2_pearson_pooled = np.nan

    return {
        "r2_actual_pred": r2_lin,
        "n_pred": int(mask.sum()),

        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(resid)),

        "rmse_fold_mean": rmse_fold_mean,
        "rmse_fold_median": rmse_fold_median,

        "rmse_fold_min": rmse_fold_min,
        "rmse_fold_max": rmse_fold_max,

        "rmse_fold_std": rmse_fold_std,
        "rmse_fold_range": rmse_fold_range,

        "r2_pearson_pooled": r2_pearson_pooled,
        **_r2_fold_stats(fold_df),
        **_n_test_stats(fold_df),
    }


# - запускает все схемы кросс-валидации
# - собирает предсказания по всем схемам
# - рассчитывает метрики по фолдам
# - формирует сводную статистику
# - строит таблицу сравнения схем
def run_all_cv(df: pd.DataFrame, degree: int = 1, k: int = 5,
               grid_n: int = 3, seed: int = 42,
               buffer: float = 300.0, variogram_factor: float = 1.0,
               leave_p_percent: int = 10, spatial_radius: float = 1000.0,
               cv_repeats: int = 50,
               strat_zone_col: str | None = "tectonic_zone",
               excl_radius_mode: str = "median_factor",      
               excl_radius_abs: float | None = None):        

    y = _y(df)
    results = {}

    # ---------------- Random K-Fold ----------------
    preds_rk, folds_rk = cv_random_kfold(df, degree, k, seed)
    results["Random K-Fold"] = {
        "preds": preds_rk,
        "folds": folds_rk,
        "summary": summarize_cv(preds_rk, y, folds_rk)
    }

    # ---------------- Random K-Fold + buffer----------------
    preds_rkb, folds_rkb = cv_random_kfold_buffered(df, degree, k, seed, buffer)
    results["Random K-Fold (buffered)"] = {
        "preds": preds_rkb,
        "folds": folds_rkb,
        "summary": summarize_cv(preds_rkb, y, folds_rkb)
    }

    # ---------------- Leave-One-Well-Out ----------------
    preds_lowo, folds_lowo = cv_lowo(df, degree)
    results["Leave-One-Well-Out"] = {
        "preds": preds_lowo,
        "folds": folds_lowo,
        "summary": summarize_cv(preds_lowo, y, folds_lowo)
    }

    # ---------------- Block CV ----------------
    preds_blk, folds_blk, block_id = cv_blocked(df, degree, grid_n)
    results["Block CV"] = {
        "preds": preds_blk,
        "folds": folds_blk,
        "summary": summarize_cv(preds_blk, y, folds_blk),
        "block_id": block_id
    }

    # ---------------- Buffered CV ----------------
    preds_buf, folds_buf, block_id_buf = cv_buffered(df, degree, grid_n, buffer)
    results["Buffered CV"] = {
        "preds": preds_buf,
        "folds": folds_buf,
        "summary": summarize_cv(preds_buf, y, folds_buf),
        "block_id": block_id_buf
    }

    # ---------------- Leave-P-Out ----------------
    preds_lpo, folds_lpo = cv_leave_p_out(df, degree, p_percent=leave_p_percent, repeats=cv_repeats, seed=seed)
    results["Leave-P-Out"] = {
        "preds": preds_lpo,
        "folds": folds_lpo,
        "summary": summarize_cv(preds_lpo, y, folds_lpo),
    }

    # ---------------- Spatial Leave-P-Out ----------------
    preds_slpo, folds_slpo = (cv_spatial_leave_p_out(df, degree, radius=spatial_radius, repeats=cv_repeats, seed=seed))
    results["Spatial Leave-P-Out"] = {
        "preds": preds_slpo,
        "folds": folds_slpo,
        "summary": summarize_cv(preds_slpo, y, folds_slpo),
    }

    # ---------------- Stratified by Zone ----------------
    if strat_zone_col is not None:
        preds_zone, folds_zone = cv_stratified_zone(df, degree, zone_col=strat_zone_col)
        if preds_zone is not None:
            zone_name = f"Stratified by {strat_zone_col}"
            results[zone_name] = {
                "preds": preds_zone,
                "folds": folds_zone,
                "summary": summarize_cv(preds_zone, y, folds_zone)
            }

    # ---------------- Distance-based exclusion CV ----------------
    preds_dist, folds_dist = cv_distance_exclusion(
        df, degree, variogram_factor,
        excl_radius_mode=excl_radius_mode,
        excl_radius_abs=excl_radius_abs,
    )
    results["Distance-based exclusion CV"] = {
        "preds": preds_dist,
        "folds": folds_dist,
        "summary": summarize_cv(preds_dist, y, folds_dist)
    }

    # ---------------- Таблица сравнения ----------------
    comparison = pd.DataFrame({
        name: r["summary"]
        for name, r in results.items()
    }).T[[
        "n_pred",
        "n_test_min",
        "n_test_max",
        "r2_actual_pred",
        "rmse",
        "mae",
        "bias",

        "rmse_fold_mean",
        "rmse_fold_median",
        "rmse_fold_min",
        "rmse_fold_max",
        "rmse_fold_std",
        "rmse_fold_range",

    ]]

    comparison.index.name = "CV scheme"

    return results, comparison


# ================================================================================================================================ #
#   ШАГ 5 · МНОГОАТРИБУТНАЯ РЕГРЕССИЯ (H_eff и Phi отдельно)
#   Перебор всех комбинаций атрибутов от 1 до k_max. Оценка: LOWO и K-fold по скважинам. Критерий: RMSE_cv.
# ================================================================================================================================ #
import itertools as _itertools

S5_SEED = 42
S5_CORR_THRESHOLD = 0.8
S5_MAX_ATTRS = 3
S5_SCHEMES = ("LOWO", "K-fold")


def step5_kmax(n_wells, n_candidates):
    """k_max = min(3, число кандидатов, max(1, floor(N / 5)))."""
    return int(min(S5_MAX_ATTRS, n_candidates, max(1, n_wells // 5)))


def step5_kfold_k(n_wells):
    """K = max(3, round(N / 10)), но не больше числа скважин."""
    return int(min(n_wells, max(3, round(n_wells / 10))))


def step5_excluded_pairs(df, cands, threshold=S5_CORR_THRESHOLD):
    """Пары атрибутов с |r| > threshold не могут входить в одну модель."""
    bad = set()
    if len(cands) < 2:
        return bad
    corr = df[cands].corr().abs()
    for a, b in _itertools.combinations(cands, 2):
        r = corr.loc[a, b]
        if np.isfinite(r) and r > threshold:
            bad.add(frozenset((a, b)))
    return bad


def step5_combinations(cands, kmax, bad_pairs):
    """Все комбинации от 1 до kmax атрибутов без запрещённых пар."""
    out = []
    for k in range(1, kmax + 1):
        for combo in _itertools.combinations(cands, k):
            if any(frozenset(p) in bad_pairs for p in _itertools.combinations(combo, 2)):
                continue
            out.append(combo)
    return out


def _step5_folds(well_codes, scheme, seed):
    """Список масок тестовых строк. LOWO — по одной скважине; K-fold — K групп скважин с фиксированным сидом."""
    uniq = np.unique(well_codes)
    if scheme == "LOWO":
        groups = [[w] for w in uniq]
    else:
        K = step5_kfold_k(len(uniq))
        perm = np.random.default_rng(seed).permutation(uniq)
        groups = [list(g) for g in np.array_split(perm, K)]
    return [np.isin(well_codes, g) for g in groups]


def _step5_design(X):
    return np.column_stack([np.ones(len(X)), X])


def _step5_fit(X, y):
    beta, *_ = np.linalg.lstsq(_step5_design(X), y, rcond=None)
    return beta


def _step5_metrics(pred, actual):
    res = pred - actual
    rmse = float(np.sqrt(np.mean(res ** 2)))
    sy = np.std(actual)
    sp = np.std(pred)
    r = float(np.corrcoef(pred, actual)[0, 1]) if sy > 0 and sp > 0 else np.nan
    sum_abs_y = float(np.sum(np.abs(actual)))
    wape = float(100 * np.sum(np.abs(res)) / sum_abs_y) if sum_abs_y > 0 else np.nan
    return {"rmse_cv": rmse, "r2_cv": r ** 2 if np.isfinite(r) else np.nan, "r_cv": r if np.isfinite(r) else np.nan,
            "wape_cv": wape, "bias_cv": float(np.mean(res))}


def _step5_equation(param, intercept, attrs, coefs):
    """Текстовое уравнение модели: H_eff = b0 + b1·a1 + ..."""
    parts = [f"{param} = {intercept:.4g}"]
    for a, c in zip(attrs, coefs):
        sign = "+" if c >= 0 else "-"
        parts.append(f" {sign} {abs(c):.4g}·{a}")
    return "".join(parts)


def _step5_insample(pred, actual):
    """Метрики in-sample: RMSE_in, R²_in, r_in, Bias_in (pred − actual)."""
    res = pred - actual
    rmse = float(np.sqrt(np.mean(res ** 2)))
    r = float(np.corrcoef(pred, actual)[0, 1]) if np.std(pred) > 0 and np.std(actual) > 0 else np.nan
    return {"RMSE_in": rmse, "R2_in": r ** 2 if np.isfinite(r) else np.nan, "r_in": r,
            "Bias_in": float(np.mean(res))}


def _step5_line(x, y):
    """Коэффициенты (a, b) линии y = a + b·x. Возвращает (nan, nan), если линия не определена."""
    if len(x) < 2 or np.std(x) == 0:
        return np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def _step5_lowo_oof(X, y, codes, lower, upper):
    """LOWO без переобучения для линейной OLS-модели (замкнутая форма).
    При исключении группы (скважины) g предсказание: oof_g = y_g − (I − H_gg)⁻¹ e_g,
    где H — матрица проекции полной модели, e — её остатки. Для одной строки на скважину
    это обычная формула e_i / (1 − h_ii). Совпадает с переобучением по фолдам.
    Средние коэффициенты по фолдам (mb) не считаются: нигде не используются."""
    n = len(y)
    p = X.shape[1] + 1
    D = _step5_design(X)
    H = D @ np.linalg.pinv(D)
    e = y - H @ y
    oof = np.full(n, np.nan)
    for g in np.unique(codes):
        idx = np.where(codes == g)[0]
        if n - len(idx) < p + 1:          # та же проверка, что в _step5_cv_eval
            continue
        try:
            corr = np.linalg.solve(np.eye(len(idx)) - H[np.ix_(idx, idx)], e[idx])
        except np.linalg.LinAlgError:
            continue
        oof[idx] = y[idx] - corr
    oof = np.clip(oof, lower, upper) if upper is not None else np.maximum(oof, lower)
    return oof, np.full(p, np.nan)


def _step5_cv_eval(X, y, folds, lower, upper):
    """Out-of-fold предсказания (с ограничением диапазона) и средние коэффициенты по фолдам."""
    p = X.shape[1] + 1
    oof = np.full(len(y), np.nan)
    beta_sum = np.zeros(p)
    beta_cnt = 0
    for te in folds:
        tr = ~te
        if tr.sum() < p + 1 or te.sum() == 0:
            continue
        beta = _step5_fit(X[tr], y[tr])
        oof[te] = _step5_design(X[te]) @ beta
        beta_sum += beta
        beta_cnt += 1
    oof = np.clip(oof, lower, upper) if upper is not None else np.maximum(oof, lower)
    mean_beta = beta_sum / beta_cnt if beta_cnt else np.full(p, np.nan)
    return oof, mean_beta


def _step5_cv_equation(a, b):
    """Калибровочная линия по out-of-fold предсказаниям: Predicted = a + b·Actual."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return "Predicted = n/a"
    return f"Predicted = {a:.4g} + {b:.4g}·Actual"


def step5_run(df, candidates, phi_upper=1.0, seed=S5_SEED, params=("H_eff", "Phi"), min_wells=3, top_n=5):
    """
    Шаг 5. Выборки:
      - одноатрибутная модель: скважины, где есть параметр и этот атрибут (попарно, как в матрице корреляции);
      - многоатрибутная модель: скважины, где есть параметр и все кандидаты (общая выборка для ранжирования).
    In-sample (вся выборка, OLS без ограничений) — не зависит от схемы CV. Вывод: singles_in (одинаков для схем),
    top_in (модели top-N схемы). Out-of-sample (CV): singles, table; уравнение = среднее коэффициентов по фолдам.
    """
    cands = [c for c in candidates if c in df.columns]

    def _fmt4(v):
        return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.4f}"

    out = {"ok": False, "n_wells": 0, "kmax": 0, "excluded_pairs": 0, "kfold_K": None, "schemes": {}}
    base = df.dropna(subset=["well_id"]).reset_index(drop=True)
    n_wells = int(base["well_id"].nunique())
    out["n_wells"] = n_wells
    if n_wells < min_wells or not cands:
        return out

    kmax = step5_kmax(n_wells, len(cands))
    bad = step5_excluded_pairs(base.dropna(subset=cands), cands)
    combos = step5_combinations(cands, kmax, bad)
    out.update(ok=True, kmax=kmax, excluded_pairs=len(bad), kfold_K=step5_kfold_k(n_wells))
    for scheme in S5_SCHEMES:
        out["schemes"][scheme] = {"singles_rows": [], "singles_in_rows": [], "top_rows": [], "top_in_rows": [],
                                  "coef_rows": [], "details": {}, "single_best": {}}

    for param in params:
        lower = 0.0
        upper = None if param == "H_eff" else float(phi_upper)

        # ---------- одноатрибутные: in-sample на попарной выборке (совпадает с матрицей корреляции) ----------
        single_sample = {}
        ins = []
        for combo in combos:
            if len(combo) != 1:
                continue
            a = combo[0]
            dd = base.dropna(subset=[param, a]).reset_index(drop=True)
            if len(dd) < 3:
                continue
            single_sample[a] = dd
            y = dd[param].to_numpy(float)
            X = dd[[a]].to_numpy(float)
            beta = _step5_fit(X, y)
            m = _step5_insample(_step5_design(X) @ beta, y)
            m["r_in"] = float(np.corrcoef(X[:, 0], y)[0, 1]) if np.std(X[:, 0]) > 0 and np.std(y) > 0 else np.nan
            m["R2_in"] = m["r_in"] ** 2 if np.isfinite(m["r_in"]) else np.nan
            ins.append({"attr": a, "n": len(dd), "beta": beta, "m": m})
        ins.sort(key=lambda e: e["m"]["RMSE_in"])
        ins_by_attr = {e["attr"]: e for e in ins}

        singles_in_param = [
            {"Parameter": param, "Rank": r, "Attributes": e["attr"], "k": 1, "n": e["n"],
             "RMSE_in": _fmt4(e["m"]["RMSE_in"]),
             "R2_in": _fmt4(e["m"]["R2_in"]),
             "r_in": _fmt4(e["m"]["r_in"]),
             "Bias_in": _fmt4(e["m"]["Bias_in"]),
             "Equation": _step5_equation(param, e["beta"][0], [e["attr"]], e["beta"][1:])}
            for r, e in enumerate(ins, start=1)
        ]

        # ---------- многоатрибутные: in-sample на общей выборке ----------
        multi_combos = [c for c in combos if len(c) > 1]
        dp = base.dropna(subset=[param, *cands]).reset_index(drop=True)
        yp = dp[param].to_numpy(float) if len(dp) else np.array([])
        codes_p = pd.factorize(dp["well_id"].astype(str))[0] if len(dp) else np.array([], dtype=int)
        multi_ins = {}
        if len(dp) >= 3:
            for combo in multi_combos:
                X = dp[list(combo)].to_numpy(float)
                beta = _step5_fit(X, yp)
                multi_ins[combo] = {"beta": beta, "m": _step5_insample(_step5_design(X) @ beta, yp), "X": X}

        for scheme in S5_SCHEMES:
            sc = out["schemes"][scheme]
            sc["singles_in_rows"].extend(singles_in_param)

            # ---------- out-of-sample (CV) ----------
            evaluated = []
            for a, dd in single_sample.items():
                y = dd[param].to_numpy(float)
                X = dd[[a]].to_numpy(float)
                codes = pd.factorize(dd["well_id"].astype(str))[0]
                if scheme == "LOWO":
                    oof, mb = _step5_lowo_oof(X, y, codes, -np.inf, None)
                else:
                    oof, mb = _step5_cv_eval(X, y, _step5_folds(codes, scheme, seed), -np.inf, None)
                ok = np.isfinite(oof)
                if ok.sum() < 3:
                    continue
                evaluated.append({"combo": (a,), "k": 1, **_step5_metrics(oof[ok], y[ok]),
                                  "oof": oof[ok], "y": y[ok], "mb": mb, "n": len(dd), "X": X, "dd": dd})
            if multi_combos and len(dp) >= 3:
                folds_p = _step5_folds(codes_p, scheme, seed)
                for combo in multi_combos:
                    X = dp[list(combo)].to_numpy(float)
                    if scheme == "LOWO":
                        oof, mb = _step5_lowo_oof(X, yp, codes_p, lower, upper)
                    else:
                        oof, mb = _step5_cv_eval(X, yp, folds_p, lower, upper)
                    ok = np.isfinite(oof)
                    if ok.sum() < 3:
                        continue
                    evaluated.append({"combo": combo, "k": len(combo), **_step5_metrics(oof[ok], yp[ok]),
                                      "oof": oof[ok], "y": yp[ok], "mb": mb, "n": len(dp), "X": X, "dd": dp})
            if not evaluated:
                continue

            singles_cv = sorted([e for e in evaluated if e["k"] == 1], key=lambda e: e["rmse_cv"])
            base_rmse = singles_cv[0]["rmse_cv"] if singles_cv else np.nan

            # таблица одноатрибутных (out-of-sample)
            for r, e in enumerate(singles_cv, start=1):
                cv_a, cv_b = _step5_line(e["y"], e["oof"])
                sc["singles_rows"].append({
                    "Parameter": param, "Rank": r, "Attributes": e["combo"][0], "k": 1, "n": e["n"],
                    "RMSE_cv": _fmt4(e["rmse_cv"]), "R2_cv": _fmt4(e["r2_cv"]), "r_cv": _fmt4(e["r_cv"]),
                    "WAPE_cv": _fmt4(e["wape_cv"]), "Bias_cv": _fmt4(e["bias_cv"]),
                    "DeltaRMSE": _fmt4(e["rmse_cv"] - base_rmse), "Accepted": "single",
                    "Equation": _step5_cv_equation(cv_a, cv_b),
                })
            # топ-N (out-of-sample) и их in-sample
            evaluated.sort(key=lambda e: e["rmse_cv"])
            for rank, e in enumerate(evaluated[:top_n], start=1):
                if e["k"] == 1:
                    accepted = "single"
                else:
                    accepted = "yes" if (np.isfinite(base_rmse) and e["rmse_cv"] < base_rmse) else "no"
                delta = e["rmse_cv"] - base_rmse if np.isfinite(base_rmse) else np.nan
                cv_a, cv_b = _step5_line(e["y"], e["oof"])
                attrs_txt = ", ".join(e["combo"])
                sc["top_rows"].append({
                    "Parameter": param, "Rank": rank, "Attributes": attrs_txt, "k": e["k"], "n": e["n"],
                    "RMSE_cv": _fmt4(e["rmse_cv"]), "R2_cv": _fmt4(e["r2_cv"]), "r_cv": _fmt4(e["r_cv"]),
                    "WAPE_cv": _fmt4(e["wape_cv"]), "Bias_cv": _fmt4(e["bias_cv"]),
                    "DeltaRMSE": _fmt4(delta), "Accepted": accepted,
                    "Equation": _step5_cv_equation(cv_a, cv_b),
                })
                # in-sample для той же модели (полная выборка модели)
                if e["k"] == 1:
                    src = ins_by_attr[e["combo"][0]]
                    X_in = e["X"]
                    y_in = e["dd"][param].to_numpy(float)
                    beta_in = src["beta"]
                    m_in = src["m"]
                    n_in = src["n"]
                else:
                    src = multi_ins[e["combo"]]
                    X_in = src["X"]
                    y_in = yp
                    beta_in = src["beta"]
                    m_in = src["m"]
                    n_in = len(dp)
                eq_in = _step5_equation(param, beta_in[0], e["combo"], beta_in[1:])
                sc["top_in_rows"].append({
                    "Parameter": param, "Rank": rank, "Attributes": attrs_txt, "k": e["k"], "n": n_in,
                    "RMSE_in": _fmt4(m_in["RMSE_in"]),
                    "R2_in": _fmt4(m_in["R2_in"]),
                    "r_in": _fmt4(m_in["r_in"]),
                    "Bias_in": _fmt4(m_in["Bias_in"]),
                    "Equation": eq_in,
                })
                # коэффициенты (in-sample) и данные для графиков
                sx = X_in.std(axis=0)
                sy = y_in.std()
                std_coef = beta_in[1:] * sx / sy if sy > 0 else np.full(len(e["combo"]), np.nan)
                coef_df = pd.DataFrame({
                    "Term": ["intercept", *e["combo"]],
                    "Coefficient": [_fmt4(v) for v in beta_in],
                    "StdCoefficient": ["" if not np.isfinite(v) else f"{v:.4f}" for v in np.concatenate([[np.nan], std_coef])],
                })
                coef_rows_item = coef_df.copy()
                coef_rows_item.insert(0, "Rank", rank)
                coef_rows_item.insert(0, "Parameter", param)
                sc["coef_rows"].append(coef_rows_item)
                fit_pred = _step5_design(X_in) @ beta_in
                sc["details"][(param, rank)] = {
                    "attrs": list(e["combo"]),
                    "intercept": float(beta_in[0]),
                    "coef": beta_in[1:],
                    "std_coef": std_coef,
                    "pred": e["oof"],
                    "actual": e["y"],
                    "fit_pred": fit_pred,
                    "fit_actual": y_in,
                    "equation": eq_in,
                    "coef_df": coef_df.reset_index(drop=True),
                }

            # одноатрибутная лучшая для графика 1 (уравнение — из in-sample строки этого атрибута)
            best = singles_cv[0] if singles_cv else None
            if best is not None:
                src = ins_by_attr[best["combo"][0]]
                sc["single_best"][param] = {
                    "attr": best["combo"][0],
                    "x": best["dd"][best["combo"][0]].to_numpy(float),
                    "y": best["dd"][param].to_numpy(float),
                    "equation": _step5_equation(param, src["beta"][0], [best["combo"][0]], src["beta"][1:]),
                    "oof": best["oof"],
                    "y_cv": best["y"],
                }

    for scheme in S5_SCHEMES:
        sc = out["schemes"][scheme]
        sc["table"] = pd.DataFrame(sc["top_rows"])
        sc["singles"] = pd.DataFrame(sc["singles_rows"])
        sc["singles_in"] = pd.DataFrame(sc["singles_in_rows"])
        sc["top_in"] = pd.DataFrame(sc["top_in_rows"])
        sc["coef_table"] = pd.concat(sc["coef_rows"], ignore_index=True) if sc["coef_rows"] else pd.DataFrame()
    return out
