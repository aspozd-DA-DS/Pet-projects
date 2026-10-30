"""
structural_qc_full.py

Модуль статистического анализа и кросс-валидации для оценки качества структурных поверхностей (карт), построенных по данным сейсморазведки МОГТ 2D/3D, по отношению к фактическим глубинам геологических маркеров в скважинах.
Реализует следующие этапы анализа:
1. Базовые метрики и QC кросс-плот:
   - Zprm_well vs Zprm_map (без кросс-валидации).
2. Кросс-валидация зависимости:
   - Zprm_map → Zprm_well.

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
   - карта невязок;
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
    slope: float
    intercept: float
    rmse: float
    mse: float
    mae: float
    bias: float
    std: float
    variance: float
    mape_pct: float
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
    mape_pct: float
    r: float = float("nan")

# ================================================================================ #
# Basic QC cross-plot — сравнение y vs x. Используется в Шаге 1 и Шаге 2
# Обозначения: x = PARAM_2 (сравниваемое значение) и y = PARAM_1 (эталонное значение)
# Для структурного QC в Шаге 1: x = Zprm_map и  y = Zprm_well
# Используется:
# - Шаг 1: PARAM_1 vs PARAM_2
# - Шаг 2: Predicted vs Actual
# Шаг 1:
# - Калибровочное уравнение:PARAM_1 = f(PARAM_2)  или Zprm_well = f(Zprm_map)
# - Невязки: Residual = PARAM_1 - PARAM_2
# - Для структурного QC: Residual = Zprm_well - Zprm_map
# - Положительная невязка: Zprm_well > Zprm_map
# - Отрицательная невязка: Zprm_well < Zprm_map
# - Рассчитываемые метрики:
#   R², RMSE, MSE, MAE, Bias
#   Std, Variance, MAPE %
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
    mape_pct = float(np.mean(np.abs(residuals) / np.maximum(np.abs(y), 1e-9)) * 100)

    return BasicStats(
        n=n,
        r2=r ** 2,
        slope=slope,
        intercept=intercept,
        rmse=rmse,
        mse=mse,
        mae=mae,
        bias=bias,
        std=std,
        variance=variance,
        mape_pct=mape_pct,
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
# - рассчитывает метрики: R², RMSE, MSE, MAE, Bias, Std, Variance, MAPE %, уравнение регрессии
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
    r_corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 2 else float("nan")
    std = (float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0)
    variance = (float(np.var(resid, ddof=1)) if len(resid) > 1 else 0.0)
    mape_pct = float(np.mean(np.abs(resid)/ np.maximum(np.abs(y), 1e-9)) * 100)
    return FitStats(
        r2=r2, r=r_corr, rmse=rmse, mse=mse, mae=mae, bias=bias, residuals=resid, equation=eq,
        std=std, variance=variance, mape_pct=mape_pct
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
    r_corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 2 else float("nan")
    std = (float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0)
    variance = (float(np.var(resid, ddof=1)) if len(resid) > 1 else 0.0)
    mape_pct = float(np.mean(np.abs(resid) / np.maximum(np.abs(y), 1e-9)) * 100)

    return FitStats(
        r2=r2, r=r_corr, rmse=rmse, mse=mse, mae=mae, bias=bias, residuals=resid, equation=eq,
        std=std, variance=variance, mape_pct=mape_pct,
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
# - Stratified by Zone (по тектоническим зонам)
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

# -- Random K‑Fold
def cv_random_kfold(df: pd.DataFrame, degree: int = 1, k: int = 5, seed: int = 42):

    x = _x(df)
    y = _y(df)
    
    k = min(k, len(df))
    if k < 2:
        raise ValueError("At least two samples are required for cross-validation")

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    preds = np.full(len(df), np.nan)
    rows = []

    for i, (tr, te) in enumerate(kf.split(x)):
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

    k = min(k, len(df))
    if k < 2:
        raise ValueError("At least two samples are required for cross-validation")

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    preds = np.full(len(df), np.nan)
    rows = []

    for i, (tr, te) in enumerate(kf.split(x)):
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

    p_count = max(1, int(np.ceil(n * p_percent / 100)))
    if p_count >= n:
        p_count = n - 1

    for fold in range(repeats):

        test_idx = rng.choice(n, size=p_count, replace=False)

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

    for fold in range(repeats):
        center_idx = rng.integers(n)
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
        excl_radius = median_pairwise_distance(coords) * variogram_factor

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

# -- Stratified by Zone (по тектоническим зонам)
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
            "r2_lin": np.nan,
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
            "r_pooled": np.nan,
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
        "r2_lin": r2_lin,
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
        "r_pooled": r_pooled if mask.sum() >= 3 else np.nan,
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
        "r2_lin",
        "r2_pearson_pooled",
        "r_pooled",
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