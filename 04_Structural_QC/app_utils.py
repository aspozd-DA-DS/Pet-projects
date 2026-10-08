"""
app_utils.py — вспомогательные функции для Structural QC.
Автономные утилиты без зависимости от глобалов app_full.py.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from sklearn.linear_model import (
    HuberRegressor,
    TheilSenRegressor,
    RANSACRegressor,
    LinearRegression,
)

import structural_qc_full as sq

# ------------------------------------------------------------- #
# FUNC: КОНСТАНТЫ — палитры
# ------------------------------------------------------------- #
PALETTE_DICT = {
    "Alphabet": px.colors.qualitative.Alphabet,
    "Bold": px.colors.qualitative.Bold,
    "Cividis": px.colors.sequential.Cividis,
    "Dark24": px.colors.qualitative.Dark24,
    "Light24": px.colors.qualitative.Light24,
    "Pastel": px.colors.qualitative.Pastel,
    "Plotly": px.colors.qualitative.Plotly,
    "Portland": px.colors.diverging.Portland,
    "Set1": px.colors.qualitative.Set1,
    "Set2": px.colors.qualitative.Set2,
    "Set3": px.colors.qualitative.Set3,
    "Spectral": px.colors.diverging.Spectral,
    "Turbo": px.colors.sequential.Turbo,
    "Viridis": px.colors.sequential.Viridis,
}

# ------------------------------------------------------------- #
# FUNC: РАСЧЁТ СТАТИСТИКИ
# ------------------------------------------------------------- #
# ------ Универсальная функция расчёта статистики параметров для Шагов 1 и 3 ------
def show_depth_stats(series, title):

    data = pd.Series(series).dropna()
    if len(data) == 0:
        st.warning("No data available for statistics")
        return

    mean = data.mean()
    median = data.median()

    try:
        mode = data.mode().iloc[0]
    except Exception:
        mode = np.nan

    std = data.std(ddof=1)
    variance = data.var(ddof=1)

    cv_pct = (100 * std / abs(mean) if mean != 0 else np.nan)

    q05 = data.quantile(0.05)
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    q95 = data.quantile(0.95)

    min_val = data.min()
    max_val = data.max()

    data_range = max_val - min_val
    iqr = q75 - q25

    stats = {
        "Count": len(data),
        "μ, mean": mean,
        "median": median,
        "mode": mode,
        "σ, std": std,
        "variance": variance,
        "CV %": cv_pct,
        "min": min_val,
        "Q05": q05,
        "Q25": q25,
        "Q75": q75,
        "Q95": q95,
        "max": max_val,
        "range": data_range,
        "IQR": iqr,
    }
    stats_df = pd.DataFrame.from_dict(stats, orient="index", columns=[title])

    st.markdown(f"#### {title}")
    st.dataframe(stats_df.style.format("{:,.2f}"), use_container_width=True)

def compute_stats(data):
    """Универсальная функция для таблицы статистики невязок (Шаги 1 и 2)."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)
    if n == 0:
        return pd.DataFrame({"Metrics": ["Error"], "Value": ["No data"]})

    mean = np.mean(data)
    sigma = (np.std(data, ddof=1) if len(data) > 1 else 0.0)
    variance = (np.var(data, ddof=1) if len(data) > 1 else 0.0)
    cv_pct = (100 * sigma / abs(mean) if mean != 0 else np.nan)

    rmse = np.sqrt(np.mean(data ** 2))
    median = np.median(data)

    q05 = np.percentile(data, 5)
    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)
    q95 = np.percentile(data, 95)

    iqr = q75 - q25
    iqr_lower = q25 - 1.5 * iqr
    iqr_upper = q75 + 1.5 * iqr

    data_range = np.max(data) - np.min(data)

    try:
        mode = pd.Series(data).mode().iloc[0]
    except Exception:
        mode = np.nan

    stats_dict = {
        "Count": n,
        "μ, mean": mean,
        "median": median,
        "mode": mode,
        "σ, std": sigma,
        "variance": variance,
        "CV %": cv_pct,
        "RMSE": rmse,
        "min": np.min(data),
        "Q05": q05,
        "Q25": q25,
        "Q75": q75,
        "Q95": q95,
        "max": np.max(data),
        "range": data_range,
        "IQR": iqr,
        "IQR lower bound": iqr_lower,
        "IQR upper bound": iqr_upper,
        "μ+3σ": mean + 3 * sigma,
        "μ-3σ": mean - 3 * sigma,
    }

    return pd.DataFrame({
        "Metrics": list(stats_dict.keys()),
        "Value": list(stats_dict.values())
    })

# ------ Универсальная функция подготовки метрик кросс-плота для Шагов 1 и 2 ------
def basicstats_to_dict(stats, equation):
    return {
        "R²_lin": stats.r2,
        "RMSE": stats.rmse,
        "MSE": stats.mse,
        "MAE": stats.mae,
        "Bias": stats.bias,
        "Std": stats.std,
        "Variance": stats.variance,
        "MAPE %": stats.mape_pct,
        "P-value": f"{stats.p_value:.5e}",
        "Equation": equation,
    }
# ------ Универсальная функция подготовки метрик аппроксимации для Шагов 1, 2 и 3 ------
def fitstats_to_dict(stats):
    return {
        "R²_fit": stats.r2,
        "RMSE": stats.rmse,
        "MSE": stats.mse,
        "MAE": stats.mae,
        "Bias": stats.bias,
        "Std": stats.std,
        "Variance": stats.variance,
        "MAPE %": stats.mape_pct,
        "Equation": stats.equation,
    }

# ------------------------------------------------------------- #
# FUNC: КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# ------------------------------------------------------------- #
def plot_pairplot(
    df, columns, diag_kind="kde",
    height=1.8,
    corner=False,
    point_color="steelblue", point_size=5, point_opacity=0.7,
    hist_color="orange", kde_color="black", scale=1.0,
    category_col=None,
    category_colors=None,
):
    pair_df = (df[columns].select_dtypes(include=np.number).dropna())

    # Защита от одинаковых имён колонок
    if len(pair_df.columns) != len(set(pair_df.columns)):
        counts = {}
        new_cols = []
        for col in pair_df.columns:
            counts[col] = counts.get(col, 0) + 1
            if counts[col] == 1:
                new_cols.append(col)
            else:
                new_cols.append(f"{col} ({counts[col]})")

        pair_df.columns = new_cols

    if len(pair_df) < 2:
        raise ValueError("Not enough data for Pairplot")

    # Если задана категориальная колонка — раскрашиваем точки по категории
    hue_series = None
    if category_col is not None and category_col in df.columns:
        # выравниваем по тем же строкам, что и pair_df
        hue_series = (
            df.loc[pair_df.index, category_col]
            .fillna("(not specified)")
            .astype(str)
        )

        # добавляем категорию как отдельную колонку в pair_df (в конец)
        pair_df = pair_df.copy()
        pair_df["_pairplot_category_"] = hue_series.values

        # Палитра: если задан словарь — используем его, иначе seaborn сам подберёт
        palette_arg = category_colors if category_colors else None

        g = sns.pairplot(
            pair_df,
            hue="_pairplot_category_",
            diag_kind=diag_kind,
            corner=corner,
            height=height,
            palette=palette_arg,
            plot_kws=dict(alpha=point_opacity, s=point_size * 8,),
            diag_kws=dict(color=(kde_color if diag_kind == "kde" else hist_color)),
        )

        # Легенда горизонтально НАД графиком, по центру
        n_cats = len(pair_df["_pairplot_category_"].unique())
        g.figure.subplots_adjust(top=0.88)
        if g.legend is not None:
            sns.move_legend(
                g,
                "lower center",
                bbox_to_anchor=(0.5, 0.98),
                ncol=min(4, n_cats),
                frameon=False,
                title=None,
            )
    else:
        g = sns.pairplot(
            pair_df,
            diag_kind=diag_kind,
            corner=corner,
            height=height,
            plot_kws=dict(color=point_color, alpha=point_opacity, s=point_size * 8,),
            diag_kws=dict(color=(kde_color if diag_kind == "kde" else hist_color)),
        )

    g.figure.set_size_inches(g.figure.get_size_inches() * scale)
    g.figure.suptitle(f"Seaborn Pairplot ({len(pair_df.columns)} parameters)", fontsize=14, y=1.02,)
    g.figure.tight_layout()

    return g.figure

# ------------------------------------------------------------- #
# FUNC: ОФОРМЛЕНИЕ КРОСС-ПЛОТОВ
# ------------------------------------------------------------- #
# ------ Универсальная функция добавления линии «Y = X» ------
def add_identity_line(
    fig, lo, hi,
    color="lightgray", width=3, dash="dash",
):
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Y = X",
            visible="legendonly",
            line=dict(color=color, width=width, dash=dash,)
        )
    )

# ------ Универсальная функция оформления cross-plot ------
def apply_crossplot_layout(fig, x_title, y_title, height=420, title=None):

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=70, b=10),
        xaxis_title=x_title,
        yaxis_title=y_title,
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.00, yanchor="bottom")
    )

    if title is not None:
        fig.update_layout(title=dict(text=title, x=0.00, y=0.999,))

# ------ Универсальная функция линейной регрессии ------
def add_regression_line(
    fig, x, y, name,
    color="orange", width=3, dash="solid", x_line=None,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Удаляем нечисловые значения (NaN и Inf)
    mask = (np.isfinite(x) & np.isfinite(y))

    x = x[mask]
    y = y[mask]

    # Недостаточно данных для регрессии
    if len(x) < 2:
        return np.array([np.nan, np.nan]), None

    # Если x_line не задан
    if x_line is None:
        x_line = np.linspace(np.min(x), np.max(x), 300)

    # Все значения X одинаковы
    if np.ptp(x) == 0:
        level = np.mean(y)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=np.full_like(x_line, level),
                mode="lines",
                name=name,
                line=dict(color=color, width=width, dash=dash,)
            )
        )
        coeffs = np.array([0.0, level])
        return coeffs, None

    try:
        coeffs = np.polyfit(x, y, 1)
        poly = np.poly1d(coeffs)

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=poly(x_line),
                mode="lines",
                name=name,
                line=dict(color=color, width=width, dash=dash,)
            )
        )
        return coeffs, poly

    except Exception:
        level = np.mean(y)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=np.full_like(x_line, level),
                mode="lines",
                name=name,
                line=dict(color=color, width=width)
            )
        )
        coeffs = np.array([0.0, level])
        return coeffs, None

# ------ Универсальная функция таблицы метрик кросс-плота ------
def build_qc_metrics_table(basic, coeffs_qc, left_name, right_name):
    eq_qc = (
        f"{left_name} = "
        f"{coeffs_qc[0]:.5f}·{right_name} + "
        f"{coeffs_qc[1]:.5f}"
    )

    return pd.DataFrame([basicstats_to_dict(basic, eq_qc)])

# ------ Универсальная функция qc_crossplot ------
def build_qc_crossplot(
    df, x_ref, y_ref, title, x_label, y_label,
    point_color, identity_color, regression_color,
    wells_label, well_label, regression_name,
    regression_width=3,
    regression_dash="solid",
    identity_width=3,
    identity_dash="dash",
    height=420,
    marker_size=9,
    marker_opacity=0.8,
):
    fig_qc = go.Figure()

    # Линия Y = X (идеальная карта)
    lo = min(y_ref.min(), x_ref.min())
    hi = max(y_ref.max(), x_ref.max())
    add_identity_line(fig_qc, lo, hi, color=identity_color, width=identity_width, dash=identity_dash,)

    # Точки скважин
    fig_qc.add_trace(
        go.Scatter(
            x=x_ref,
            y=y_ref,
            mode="markers",
            name=wells_label,
            text=df["well_id"],
            customdata=np.column_stack([df["well_id"], x_ref, y_ref,]),
            marker=dict(color=point_color, size=marker_size, opacity=marker_opacity,),
            hovertemplate=(
                f"{well_label}: %{{text}}"
                f"<br>{x_label}=%{{x}}"
                f"<br>{y_label}=%{{y}}"
                f"<extra></extra>"
            )
        )
    )

    # ----- Линейная регрессия для QC CROSS-PLOT -----
    z_line_qc = np.linspace(x_ref.min(), x_ref.max(), 300,)

    coeffs_qc, _ = add_regression_line(
        fig=fig_qc,
        x=x_ref,
        y=y_ref,
        name=regression_name,
        color=regression_color,
        width=regression_width,
        dash=regression_dash,
        x_line=z_line_qc,
    )

    # Настройка оформления cross-plot
    apply_crossplot_layout(fig_qc, x_label, y_label, title=title, height=height,)

    return fig_qc, coeffs_qc

# ------ Универсальная функция crossplot линий аппроксимации ------
def build_approximation_plot(
    df, x_ref, y_ref, z_line, title, x_label, y_label, x_name, y_name, approx_types,
    point_color, identity_color,
    wells_label, well_label, error_label,
    approx_colors, quantile_colors,
    approx_dashes=None, quantile_dashes=None,
    approx_widths=None, quantile_widths=None,       # ← новое
    marker_size=9, marker_opacity=0.8,
    identity_width=3, identity_dash="dash",
):
    if approx_dashes is None:     approx_dashes = {}
    if quantile_dashes is None:   quantile_dashes = {}
    if approx_widths is None:     approx_widths = {}
    if quantile_widths is None:   quantile_widths = {}
    
    fig_fit = go.Figure()

    # Линия Y = X (идеальная карта)
    lo = min(y_ref.min(), x_ref.min())
    hi = max(y_ref.max(), x_ref.max())
    add_identity_line(fig_fit, lo, hi, color=identity_color, width=identity_width, dash=identity_dash,)

    # Точки скважин
    fig_fit.add_trace(
        go.Scatter(
            x=x_ref,
            y=y_ref,
            mode="markers",
            name=wells_label,
            text=df["well_id"],
            customdata=np.column_stack([df["well_id"], x_ref, y_ref,]),
            marker=dict(color=point_color, size=marker_size, opacity=marker_opacity,),
            hovertemplate=(
                f"{well_label}: %{{text}}"
                f"<br>{x_name}=%{{x}}"
                f"<br>{y_name}=%{{y}}"
                f"<extra></extra>"
            )
        )
    )

    fit_rows = []

    for approx in approx_types:
        try:
            # --- вычисление кривой ---
            result = compute_curve(approx, x_ref, y_ref, z_line, approx_colors, quantile_colors)

            # --- квантильная регрессия ---
            if approx == "Quantile Regression (τ=0.1/0.5/0.9)":
                curves, eqs, _ = result
                for tau, y_pred, col in curves:
                    dash_tau = quantile_dashes.get(tau, "solid")
                    width_tau = quantile_widths.get(tau, 3)      
                    fig_fit.add_trace(
                        go.Scatter(
                            x=z_line,
                            y=y_pred,
                            mode="lines",
                            name=f"{approx} τ={tau}",
                            line=dict(color=col, width=width_tau, dash=dash_tau),
                        )
                    )

                # Метрики считаем отдельно для каждой τ
                for tau in [0.1, 0.5, 0.9]:
                    try:
                        stats_tau = sq.compute_quantile_stats(x_ref, y_ref, tau)
                        fit_rows.append({
                            "Approximation": f"Quantile Regression (τ={tau})",
                            **fitstats_to_dict(stats_tau),
                        })
                    except Exception as e:
                        fit_rows.append({
                            "Approximation": f"Quantile Regression (τ={tau})",
                            error_label: str(e),
                        })

            # --- обычные аппроксимации ---
            else:
                z_fit, eq, color = result
                if z_fit is not None:
                    dash_approx = approx_dashes.get(approx, "solid")
                    width_approx = approx_widths.get(approx, 3)
                    fig_fit.add_trace(
                        go.Scatter(
                            x=z_line,
                            y=z_fit,
                            mode="lines",
                            name=approx,
                            line=dict(color=color, width=width_approx, dash=dash_approx),
                        )
                    )
                stats_fit = sq.compute_fit_stats(x_ref, y_ref, approx,)
                fit_rows.append({"Approximation": approx, **fitstats_to_dict(stats_fit),})
        except Exception as e:
            fit_rows.append({"Approximation": approx, error_label: str(e),})

    # Настройка оформления cross-plot
    apply_crossplot_layout(fig_fit, x_label, y_label, title=title,)
    return fig_fit, fit_rows

# ------------------------------------------------------------- #
# FUNC: ОФОРМЛЕНИЕ Bar Chart, ГИСТОГРАММ И BOXPLOT
# ------------------------------------------------------------- #
# ------ Универсальная функция столбчатой диаграммы bar chart ------
def plot_bar_chart(x, y, palette, title=None, xaxis_title=None, yaxis_title=None, show_percent=False, orientation="v", legend_title=None,):

    colors = PALETTE_DICT.get(palette, px.colors.qualitative.Plotly)
    bar_colors = [
        colors[i % len(colors)]
        for i in range(len(x))
    ]

    if show_percent:
        total = np.sum(y)
        if total == 0:
            text = [f"{v:.0f}" for v in y]
        else:
            text = [f"{v:.0f}<br>({100 * v / total:.1f}%)" for v in y]
    else:
        text = [f"{v:.1f}" for v in y]

    fig = go.Figure()

    if orientation == "h":
        for xi, yi, txt, col in zip(x, y, text, bar_colors):
            fig.add_trace(
                go.Bar(
                    x=[yi],
                    y=[xi],
                    orientation="h",
                    name=str(xi),
                    text=[txt],
                    textposition="outside",
                    marker_color=col,
                    showlegend=True,
                )
            )
    else:
        for xi, yi, txt, col in zip(x, y, text, bar_colors):
            fig.add_trace(
                go.Bar(
                    x=[xi],
                    y=[yi],
                    name=str(xi),
                    text=[txt],
                    textposition="outside",
                    marker_color=col,
                    showlegend=True,
                )
            )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=True,
        legend_title=legend_title,
    )

    return fig

# ------ Универсальная функция для гистограмм (Шаги 1 и 2) ------
def plot_hist_kde(
    data,
    bins,
    histnorm,
    color,
    title,
    xlabel,
    show_sigma=True,
    kde_color="black",    kde_dash="solid",     kde_width=3,      
    mean_color="green",   mean_dash="dash",     mean_width=2,     
    median_color="red",   median_dash="dot",    median_width=2,   
    sigma_color="orange", sigma_dash="dash",    sigma_width=2,    
):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return go.Figure()

    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data, ddof=1) if len(data) > 1 else 0.0
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins, histnorm=histnorm,
        marker_color=color,
        opacity=0.85,
        name="Histogram"
    ))

    # KDE с защитой от константного массива
    if len(np.unique(data)) < 2:
        ymax = 1.0
        fig.add_annotation(
            text="KDE cannot be computed (all values are identical)",
            xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False
        )
    else:
        kde = gaussian_kde(data)
        x_kde = np.linspace(np.min(data), np.max(data), 300)
        kde_y = kde(x_kde)
        fig.add_trace(
            go.Scatter(
                x=x_kde, y=kde_y, mode="lines", name="KDE",
                line=dict(color=kde_color, width=kde_width, dash=kde_dash),
                yaxis="y2",
            )
        )
        ymax = float(np.max(kde_y))

    # Mean
    fig.add_trace(go.Scatter(
        x=[mean, mean], y=[0, ymax], mode="lines",
        name=f"Mean = {mean:.1f}",
        line=dict(color=mean_color, width=mean_width, dash=mean_dash),
        yaxis="y2"
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=[median, median], y=[0, ymax], mode="lines",
        name=f"Median = {median:.1f}",
        line=dict(color=median_color, width=median_width, dash=median_dash),
        yaxis="y2"
    ))

    # μ±3σ
    if show_sigma:
        fig.add_trace(go.Scatter(
            x=[mean + 3*std, mean + 3*std], y=[0, ymax], mode="lines",
            name=f"+3σ = {mean + 3*std:.1f}",
            line=dict(color=sigma_color, width=sigma_width, dash=sigma_dash),
            yaxis="y2"
        ))
        fig.add_trace(go.Scatter(
            x=[mean - 3*std, mean - 3*std], y=[0, ymax], mode="lines",
            name=f"-3σ = {mean - 3*std:.1f}",
            line=dict(color=sigma_color, width=sigma_width, dash=sigma_dash),
            yaxis="y2"
        ))

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=70, b=10),
        title=dict(text=title, x=0.01, y=0.995),
        xaxis_title=xlabel,
        yaxis_title="Histogram",
        yaxis2=dict(overlaying="y", side="right", title="KDE", showgrid=False),
        legend=dict(orientation="h", x=0, yanchor="bottom", y=1.02)
    )

    return fig

# ------ Универсальная функция для boxplot в шаге 1 и шаге 2 ------
def plot_box(
    data, color, ylabel, well_label,
    title=None, well_ids=None, value_label=None,
    mean_color="green",   mean_dash="dash",   mean_width=2,     
    median_color="red",   median_dash="dot",  median_width=2,   
):
    data = np.asarray(data)
    if well_ids is not None:
        well_ids = np.asarray(well_ids)
    mask = np.isfinite(data)
    data = data[mask]
    if well_ids is not None:
        well_ids = well_ids[mask]
    if len(data) == 0:
        return go.Figure()

    mean = np.mean(data)
    median = np.median(data)

    fig = go.Figure()
    if value_label is None:
        value_label = ylabel

    # Основной boxplot — фиксируем его в x=0, чтобы числовая ось X
    # была согласована с линиями Mean/Median ниже
    fig.add_trace(
        go.Box(
            y=data,
            x=[0] * len(data),                 # ← box в x=0
            name="Boxplot",
            marker_color=color,
            fillcolor=color,
            opacity=0.6,
            boxmean=True,
            boxpoints="outliers",
            text=well_ids,
            hovertemplate=(
                f"{well_label}: %{{text}}"
                f"<br>{value_label}: %{{y:.2f}}"
                f"<extra></extra>"
            ),
        )
    )

    # Линии Mean и Median — как trace'ы, чтобы клик по легенде скрывал/показывал их
    fig.add_trace(
        go.Scatter(
            x=[-0.5, 0.5],
            y=[mean, mean],
            mode="lines",
            name=f"Mean = {mean:.2f}",
            line=dict(color=mean_color, width=mean_width, dash=mean_dash),
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[-0.5, 0.5],
            y=[median, median],
            mode="lines",
            name=f"Median = {median:.2f}",
            line=dict(color=median_color, width=median_width, dash=median_dash),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=70, b=10),
        title=dict(text=title, x=0.01, y=0.995),
        yaxis_title=ylabel,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(
            showticklabels=False,     # скрываем числовые tick'и
            showgrid=False,
            zeroline=False,
            range=[-0.75, 0.75],      # чуть шире линий, чтобы box не «прилипал» к краям
        ),
    )

    return fig

# ------------------------------------------------------------- #
# FUNC: Расчёт кривых аппроксимации
# ------------------------------------------------------------- #
def compute_polynomial_curve(x, y, x_line, degree):
    """Универсальная функция расчёта полиномиальной кривой."""
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    return (poly(x_line), sq.polynomial_equation(coeffs))


def get_equation(approx, x, y):
    """Универсальная функция получения текста уравнения."""
    try:
        return sq.compute_fit_stats(x, y, approx).equation
    except Exception:
        return f"{approx} (equation unavailable)"

# ------ Универсальная функция расчёта кривой аппроксимации ------
def compute_curve(approx, x, y, x_line, approx_colors, quantile_colors):

    # --- Защита от NaN и Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return None, "Not enough data", "gray"

    # ---- Полиномиальные аппроксимации
    if approx in sq.POLY_APPROX:
        if np.std(x) == 0:
            return None, "All X values are identical", "gray"
        z_fit, eq = compute_polynomial_curve(x, y, x_line, sq.POLY_APPROX[approx])
        return (z_fit, eq, approx_colors[approx])

    # ----- Степенная
    elif approx == "Power Law":
        mask = (x > 0) & (y > 0)
        if mask.sum() < 3:
            return None, "Not enough data for power-law approximation", approx_colors.get(approx, "gray")

        logx = np.log(x[mask])
        logy = np.log(y[mask])

        if np.any(np.isinf(logx)) or np.any(np.isinf(logy)):
            return None, "Logarithmic transformation failed", approx_colors.get(approx, "gray")

        b, log_a = np.polyfit(logx, logy, 1)
        a = np.exp(log_a)
        x_line_safe = np.where(x_line > 0, x_line, np.nan)
        return (a * (x_line_safe ** b), get_equation(approx, x, y), approx_colors[approx])

    # ----- Экспоненциальная
    elif approx == "Exponential":
        mask = y > 0
        if mask.sum() < 3:
            return None, "Not enough data for exponential approximation", approx_colors.get(approx, "gray")

        logy = np.log(y[mask])
        if np.any(np.isinf(logy)):
            return None, "Logarithmic transformation failed", approx_colors.get(approx, "gray")

        coeffs = np.polyfit(x[mask], logy, 1)
        a = np.exp(coeffs[1])
        b = coeffs[0]
        return (a * np.exp(b * x_line), get_equation(approx, x, y), approx_colors[approx])

    # ----- Логарифмическая
    elif approx == "Logarithmic":
        mask = x > 0
        if mask.sum() < 3:
            return None, "Not enough data for logarithmic approximation", approx_colors.get(approx, "gray")

        logx = np.log(x[mask])
        if np.any(np.isinf(logx)):
            return None, "Logarithmic transformation failed", approx_colors.get(approx, "gray")

        coeffs = np.polyfit(logx, y[mask], 1)
        a = coeffs[0]
        b = coeffs[1]
        x_line_safe = np.where(x_line > 0, x_line, np.nan)
        return (a * np.log(x_line_safe) + b, get_equation(approx, x, y), approx_colors[approx])

    # ----- Robust (Huber)
    elif approx == "Robust (Huber)":
        model = HuberRegressor().fit(x.reshape(-1, 1), y)
        return (model.predict(x_line.reshape(-1, 1)), get_equation(approx, x, y), approx_colors[approx])

    # ----- Theil-Sen
    elif approx == "Theil-Sen":
        model = TheilSenRegressor(random_state=42)
        model.fit(x.reshape(-1, 1), y)
        return (model.predict(x_line.reshape(-1, 1)), get_equation(approx, x, y), approx_colors[approx])

    # ----- RANSAC
    elif approx == "RANSAC":
        model = RANSACRegressor(estimator=LinearRegression(), random_state=42)
        model.fit(x.reshape(-1, 1), y)
        return (model.predict(x_line.reshape(-1, 1)), get_equation(approx, x, y), approx_colors[approx])

    # ----- LOWESS
    elif approx == "LOWESS":
        lowess = sm.nonparametric.lowess(y, x, frac=0.3)
        return (np.interp(x_line, lowess[:, 0], lowess[:, 1]), get_equation(approx, x, y), approx_colors[approx])

    # ----- Квантильная регрессия
    elif approx == "Quantile Regression (τ=0.1/0.5/0.9)":
        curves = []
        colors = quantile_colors
        for tau in [0.1, 0.5, 0.9]:
            model = sm.QuantReg(y, sm.add_constant(x))
            res = model.fit(q=tau)
            y_pred = res.predict(sm.add_constant(x_line))
            curves.append((tau, y_pred, colors[tau]))

        eq_all = get_equation(approx, x, y)
        eqs = {tau: eq_all for tau in [0.1, 0.5, 0.9]}
        return curves, eqs, None

    return None, None, None

# ------------------------------------------------------------- #
# FUNC: ОФОРМЛЕНИЕ ГРАФИКОВ И КАРТ НЕВЯЗОК
# ------------------------------------------------------------- #
# ------ Универсальная функция для графика невязок ------
def plot_residual_vs_depth(
    z_values, residuals, well_ids, color,
    wells_label, well_label,
    point_size=8, point_opacity=0.85,
    xlabel="", ylabel="", hover_label="Residual", title=None,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=z_values,
            y=residuals,
            mode="markers",
            name=wells_label,
            showlegend=True,
            marker=dict(color=color, size=point_size, opacity=point_opacity,),
            text=well_ids,
            hovertemplate=(
                f"{well_label}: %{{text}}"
                f"<br>{xlabel}=%{{x}}"
                f"<br>{hover_label}=%{{y}}"
                f"<extra></extra>"
            )
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    # Настройка оформления cross-plot
    apply_crossplot_layout(fig, xlabel, ylabel)

    if title is not None:
        fig.update_layout(title=dict(text=title, x=0.01, y=0.98))

    return fig

# ------ Универсальная функция для карты невязок ------
def plot_residual_map(
    df, residuals, palette, wells_label, well_label,
    residual_label="Residual", point_size=12, point_opacity=0.9, title=None,
):
    if len(residuals) == 0:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["X_coord"],
            y=df["Y_coord"],
            mode="markers",
            name=wells_label,
            marker=dict(
                size=point_size,
                color=residuals,
                opacity=point_opacity,
                colorscale=palette,
                cmin=np.min(residuals),
                cmax=np.max(residuals),
                colorbar=dict(title=residual_label, x=1.02, y=0.5, len=0.9,),
            ),
            text=df["well_id"],
            hovertemplate=(
                f"{well_label}: %{{text}}"
                "<br>X=%{x}"
                "<br>Y=%{y}"
                f"<br>{residual_label}=%{{marker.color:.2f}}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )

    if title is not None:
        fig.update_layout(
            title=dict(text=title, x=0.01, y=0.98,)
        )

    return fig

# ------------------------------------------------------------- #
# FUNC: АНАЛИЗ ВЫБРОСОВ (ШАГИ 1 И 2)
# ------------------------------------------------------------- #
def detect_outliers(residuals: np.ndarray, sigma_thr: float) -> tuple:
    """Универсальная функция поиска выбросов по невязкам (Шаги 1 и 2)."""
    residuals = np.asarray(residuals)

    if len(residuals) == 0:
        empty = np.array([], dtype=bool)
        return (empty, empty, empty, empty, 0.0,)

    sigma = (
        np.std(residuals, ddof=1)
        if len(residuals) > 1
        else 0.0
    )

    sigma_mask = (np.abs(residuals) > sigma_thr * sigma)

    q1 = np.percentile(residuals, 25)
    q3 = np.percentile(residuals, 75)
    iqr = q3 - q1
    iqr_lower = q1 - 1.5 * iqr
    iqr_upper = q3 + 1.5 * iqr
    iqr_mask = ((residuals < iqr_lower) | (residuals > iqr_upper))

    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    if mad > 0:
        modified_z = (0.6745 * (residuals - median) / mad)
        modz_mask = (np.abs(modified_z) > 3.5)
    else:
        modz_mask = np.zeros_like(residuals, dtype=bool)

    q05 = np.percentile(residuals, 5)
    q95 = np.percentile(residuals, 95)
    q05q95_mask = ((residuals < q05) | (residuals > q95))

    return (sigma_mask, iqr_mask, modz_mask, q05q95_mask, sigma)

# ------ Универсальная функция отображения выбросов в шаге 1 ------
def show_outlier_table(title, df, mask, residuals, residual_column_name, columns, no_outliers_label):
    st.markdown(title)
    out = df.loc[mask].copy()
    if len(out) > 0:
        out[residual_column_name] = residuals[mask]
        cols = [
            c
            for c in columns
            if c in out.columns
        ]

        st.dataframe(out[cols], use_container_width=True,)

    else:
        st.caption(no_outliers_label)


def show_outlier_tables(df, residuals, sigma_thr, param_1, param_2,
                        residual_column_name="Residual",
                        no_outliers_label="No outliers detected."):

    (
        sigma_mask,
        iqr_mask,
        modz_mask,
        q05q95_mask,
        sigma,
    ) = detect_outliers(
        residuals,
        sigma_thr,
    )

    base_cols = ["horizon_id", "well_id", "X_coord", "Y_coord", param_1, param_2, residual_column_name,]
    # Sigma
    sigma_cols = base_cols.copy()
    out_sigma = df.loc[sigma_mask].copy()
    if len(out_sigma) > 0:
        out_sigma["sigma"] = (residuals[sigma_mask] / sigma if sigma > 0 else 0.0)

        sigma_cols.append("sigma")
        st.markdown(f"#### Sigma rule (|{residual_column_name}| > {sigma_thr}σ)")

        show_outlier_table(
            "",
            out_sigma,
            np.ones(len(out_sigma), dtype=bool),
            residuals[sigma_mask],
            residual_column_name,
            sigma_cols,
            no_outliers_label,
        )

    else:
        st.markdown(f"#### Sigma rule (|{residual_column_name}| > {sigma_thr}σ)")
        st.caption(no_outliers_label)

    # IQR
    show_outlier_table(
        "#### IQR rule",
        df,
        iqr_mask,
        residuals,
        residual_column_name,
        base_cols,
        no_outliers_label,
    )

    # Modified Z
    show_outlier_table(
        "#### Modified Z-score rule (|Modified Z| > 3.5)",
        df,
        modz_mask,
        residuals,
        residual_column_name,
        base_cols,
        no_outliers_label,
    )

    # Percentile
    show_outlier_table(
        "#### Percentile rule (Q05-Q95)",
        df,
        q05q95_mask,
        residuals,
        residual_column_name,
        base_cols,
        no_outliers_label,
    )

# ------------------------------------------------------------- #
# FUNC: КАТЕГОРИИ В ШАГЕ 3 - Plot_Pie, Boxplot
# ------------------------------------------------------------- #
# ------ Круговая диаграмма для категорий в шаге 3 ------
def plot_zone_pie(df, category_col, palette="Plotly", title=""):

    zone_counts = (df[category_col].fillna("Not specified").value_counts().reset_index())
    zone_counts.columns = ["Category", "Count"]

    fig = px.pie(
        zone_counts,
        names="Category",
        values="Count",
        hole=0.35,
        color_discrete_sequence=PALETTE_DICT.get(palette, px.colors.qualitative.Plotly)
    )

    total = zone_counts["Count"].sum()

    custom_text = [
        f"{cat}<br>{cnt}<br>{100 * cnt / total:.1f}%"
        for cat, cnt in zip(zone_counts["Category"], zone_counts["Count"])
    ]

    fig.update_traces(
        text=custom_text,
        textinfo="text",
        textposition="inside"
    )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=70, b=10),
        title=dict(text=title, x=0.01, y=0.995),
        legend=dict(orientation="v")
    )

    return fig

# ------ Диаграмма Boxplot для категорий в шаге 3 ------
def plot_zone_boxplot(
    df, value_col, well_label,
    zone_col="tectonic_zone", zone_label="Category",
    title="", yaxis_title="", zone_colors=None, value_label=None,
):
    fig = go.Figure()
    if value_label is None:
        value_label = yaxis_title

    zones = sorted(df[zone_col].dropna().unique())

    for zone in zones:
        zone_part = df[df[zone_col] == zone]
        fig.add_trace(
            go.Box(
                y=zone_part[value_col],
                name=str(zone),
                boxmean=True,
                boxpoints="outliers",
                text=zone_part["well_id"],
                marker_color=(zone_colors.get(zone, "gray") if zone_colors else None),
                hovertemplate=(
                    f"{well_label}: %{{text}}"
                    f"<br>{zone_label}: {zone}"
                    f"<br>{value_label}: %{{y:.2f}}"
                    f"<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=450,
        title=title,
        xaxis_title=zone_label,
        yaxis_title=yaxis_title,
        showlegend=True,
        legend_title=zone_label,
    )

    return fig

# ------------------------------------------------------------- #
# FUNC: ОФОРМЛЕНИЕ КРОСС-ПЛОТОВ КАТЕГОРИЙ (ШАГ 3)
# ------------------------------------------------------------- #
# ------ Универсальная функция crossplot по категориям ------
def build_zone_overview_plot(
    df_source, zones_to_plot, category_col,
    zone_marker_colors, zone_curve_colors, approx_types,
    param_1, param_2, well_label,
    approx_colors, quantile_colors,
    approx_dashes=None, quantile_dashes=None,
    approx_widths=None,                            
    point_size=9, point_opacity=0.8,
):
    if approx_dashes is None:
        approx_dashes = {}
    if quantile_dashes is None:
        quantile_dashes = {}

    fig = go.Figure()

    for zone in zones_to_plot:
        zone_data = prepare_zone_data(df_source, category_col, zone,)

        if zone_data is None:
            continue

        zone_part, xz, yz, x_line_z = zone_data

        # Точки скважин
        fig.add_trace(
            go.Scatter(
                x=xz,
                y=yz,
                mode="markers",
                name=f"{category_col}: {zone}",
                marker=dict(size=point_size, color=zone_marker_colors[zone], opacity=point_opacity,),
                text=zone_part["well_id"],
                hovertemplate=(
                    f"{well_label}: %{{text}}"
                    f"<br>{param_2}=%{{x}}"
                    f"<br>{param_1}=%{{y}}"
                    f"<extra></extra>"
                )
            )
        )

        # Линии аппроксимации
        for approx in approx_types:
            try:
                z_fit, eq, color = compute_curve(
                    approx, xz, yz, x_line_z, approx_colors, quantile_colors,
                )
                if z_fit is not None:
                    dash_approx = approx_dashes.get(approx, "solid")
                    width_approx = approx_widths.get(approx, 2)
                    fig.add_trace(
                        go.Scatter(
                            x=x_line_z,
                            y=z_fit,
                            mode="lines",
                            name=f"{approx} — {zone}",
                            line=dict(width=width_approx, color=zone_curve_colors[zone], dash=dash_approx),
                        )
                    )
            except Exception:
                continue

    return fig

# ------ Универсальная функция добавления QC-линии регрессии на zone-plot ------
def add_qc_line_to_zone_plot(
    fig,
    df_source,
    param_1,
    param_2,
    regression_qc_name,
    regression_color,
    identity_color,
    regression_width=3,
    regression_dash="solid",
    identity_width=3,
    identity_dash="dash",
):
    if len(df_source) < 2:
        return fig

    x_qc = np.linspace(df_source["Zprm_map"].min(), df_source["Zprm_map"].max(), 300,)
    # линия регрессии
    coeffs_qc, _ = add_regression_line(
        fig=fig,
        x=df_source["Zprm_map"].to_numpy(float),
        y=df_source["Zprm_well"].to_numpy(float),
        name="_nolegend_",
        color=regression_color,
        width=regression_width,
        dash=regression_dash,
        x_line=x_qc,
    )

    sign = "+" if coeffs_qc[1] >= 0 else "-"
    eq_qc = (
        f"{regression_qc_name}: "
        f"{param_1} = {coeffs_qc[0]:.5f}·{param_2} "
        f"{sign} "
        f"{abs(coeffs_qc[1]):.5f}"
    )

    fig.data[-1].name = eq_qc

    # Линия Y = X (идеальная карта)
    lo = min(df_source["Zprm_well"].min(), df_source["Zprm_map"].min(),)
    hi = max(df_source["Zprm_well"].max(), df_source["Zprm_map"].max(),)
    add_identity_line(fig, lo, hi, color=identity_color, width=identity_width, dash=identity_dash,)

    return fig

def prepare_zone_data(df, category_col, category):
    """Подготовка данных по выбранной категории (Шаг 3)."""
    zone_part = df[df[category_col] == category]

    xz = zone_part["Zprm_map"].to_numpy(float)
    yz = zone_part["Zprm_well"].to_numpy(float)

    if len(xz) < 3:
        return None
    if len(np.unique(xz)) < 2:
        return None

    x_line_z = np.linspace(xz.min(), xz.max(), 200)

    return (zone_part, xz, yz, x_line_z)