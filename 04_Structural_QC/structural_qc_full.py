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
   - Leave-One-Well-Out
   - Block CV
   - Buffered CV
   - Distance-based exclusion CV
   - Stratified by Zone
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
# - Невязки: Residual = y - x
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
    std = (float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0)
    variance = (float(np.var(resid, ddof=1)) if len(resid) > 1 else 0.0)
    mape_pct = float(np.mean(np.abs(resid)/ np.maximum(np.abs(y), 1e-9)) * 100)
    return FitStats(
        r2=r2, rmse=rmse, mse=mse, mae=mae, bias=bias, residuals=resid, equation=eq,
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
    std = (float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0)
    variance = (float(np.var(resid, ddof=1)) if len(resid) > 1 else 0.0)
    mape_pct = float(np.mean(np.abs(resid) / np.maximum(np.abs(y), 1e-9)) * 100)

    return FitStats(
        r2=r2, rmse=rmse, mse=mse, mae=mae, bias=bias, residuals=resid, equation=eq,
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
        }

    return {
        "n": len(y_true),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(resid)),
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
def cv_random_kfold(df: pd.DataFrame, degree: int = 1, k: int = 5, seed: int = 0):

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
            continue
        p, m = run_fold(x, y, preds, tr, te, degree )
        m["fold"] = f"fold_{i+1}"
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
            continue
        p, m = run_fold(x, y, preds, tr, te, degree)
        m["fold"] = str(groups[te][0])
        rows.append(m)

    return preds, pd.DataFrame(rows)

# -- Leave-P-Out CV
def cv_leave_p_out(df: pd.DataFrame, degree: int, p_percent: int = 10, repeats: int = 50, seed: int = 0):

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

        if len(te) == 0 or len(tr) < 3:
            continue

        p, m = run_fold(x, y, preds, tr, te, degree)
        m["fold"] = f"block_{b}"
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

        if len(tr) < 3:
            continue

        p, m = run_fold(x, y, preds, tr, te, degree)
        m["fold"] = (f"buffered_block_{b}")
        rows.append(m)

    return (preds, pd.DataFrame(rows), block_id)

# -- Spatial Leave-P-Out
def cv_spatial_leave_p_out(df: pd.DataFrame, degree: int, radius: float = 1000.0, repeats: int = 50, seed: int = 0):

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

# -- Distance-based exclusion CV: для тестовой скважины исключаются все обучающие наблюдения, расположенные ближе радиуса пространственной корреляции range_corr

# Оценка радиуса пространственной корреляции по медианному межскважинному расстоянию
def estimate_correlation_range(coords: np.ndarray) -> float:
    distances = pdist(coords)

    if len(distances) == 0:
        return 0.0

    return float(np.median(distances))

# Distance-based exclusion CV
def cv_variogram(df: pd.DataFrame, degree: int = 1, variogram_factor: float = 1.0):

    x = _x(df)
    y = _y(df)

    coords = df[["X_coord", "Y_coord"]].to_numpy(float)
    range_corr = (estimate_correlation_range(coords) * variogram_factor)

    preds = np.full(len(df), np.nan)
    rows = []

    for i in range(len(df)):

        dist = np.linalg.norm(coords - coords[i], axis=1)
        tr = np.where(dist > range_corr)[0]
        te = np.array([i])

        if len(tr) < degree + 2:
            continue

        p, m = run_fold(x, y, preds, tr, te, degree)
        m["fold"] = f"vario_{i}"
        rows.append(m)

    return preds, pd.DataFrame(rows)

# -- Stratified by Zone (по тектоническим зонам)
def cv_stratified_zone(df: pd.DataFrame, degree: int = 1, zone_col: str = "tectonic_zone"):

    if zone_col not in df.columns:
        return None, None

    x = _x(df)
    y = _y(df)

    zones = df[zone_col].to_numpy()
    if len(np.unique(zones)) < 2:
        return None, None
    logo = LeaveOneGroupOut()
    preds = np.full(len(df), np.nan)
    rows = []

    for tr, te in logo.split(x, y, groups=zones):
        if len(tr) < degree + 2:
            continue
        p, m = run_fold(x, y, preds, tr, te, degree)
        m["fold"] = str(zones[te][0])
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

def summarize_cv(preds: np.ndarray, y_true: np.ndarray, fold_df: pd.DataFrame) -> dict:

    mask = ~np.isnan(preds)

    if mask.sum() == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "rmse_fold_mean": np.nan,
            "rmse_fold_median": np.nan,
            "rmse_fold_min": np.nan,
            "rmse_fold_max": np.nan,
            "rmse_fold_std": np.nan,
            "rmse_fold_range": np.nan,
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

    return {
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(resid)),

        "rmse_fold_mean": rmse_fold_mean,
        "rmse_fold_median": rmse_fold_median,

        "rmse_fold_min": rmse_fold_min,
        "rmse_fold_max": rmse_fold_max,

        "rmse_fold_std": rmse_fold_std,
        "rmse_fold_range": rmse_fold_range,
    }

# - запускает все схемы кросс-валидации
# - собирает предсказания по всем схемам
# - рассчитывает метрики по фолдам
# - формирует сводную статистику
# - строит таблицу сравнения схем
def run_all_cv(df: pd.DataFrame, degree: int = 1, k: int = 5,
               grid_n: int = 3, seed: int = 0,
               buffer: float = 300.0, variogram_factor: float = 1.0,
               leave_p_percent: int = 10, spatial_radius: float = 1000.0, cv_repeats: int = 50,):

    y = _y(df)
    results = {}

    # ---------------- Random K-Fold ----------------
    preds_rk, folds_rk = cv_random_kfold(df, degree, k, seed)
    results["Random K-Fold"] = {
        "preds": preds_rk,
        "folds": folds_rk,
        "summary": summarize_cv(preds_rk, y, folds_rk)
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
    preds_zone, folds_zone = cv_stratified_zone(df, degree)
    if preds_zone is not None:
        results["Stratified by Zone"] = {
            "preds": preds_zone,
            "folds": folds_zone,
            "summary": summarize_cv(preds_zone, y, folds_zone)
        }

    # ---------------- Distance-based exclusion CV ----------------
    preds_vario, folds_vario = cv_variogram(df, degree, variogram_factor)
    results["Distance-based exclusion CV"] = {
        "preds": preds_vario,
        "folds": folds_vario,
        "summary": summarize_cv(preds_vario, y, folds_vario)
    }

    # ---------------- Таблица сравнения ----------------
    comparison = pd.DataFrame({
        name: r["summary"]
        for name, r in results.items()
    }).T[[
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