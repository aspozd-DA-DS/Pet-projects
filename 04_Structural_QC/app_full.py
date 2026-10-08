"""
app_full.py — интерактивный Streamlit-дашборд для контроля качества структурных построений по данным МОГТ 2D/3D.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import os
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde
from sklearn.linear_model import (
    HuberRegressor,
    TheilSenRegressor,
    RANSACRegressor,
    LinearRegression
)

import structural_qc_full as sq
import ui_examples_full_ru
import ui_examples_full_en
from translations_full import RU, EN

from app_utils import (
    compute_stats,
    prepare_zone_data,
    detect_outliers,
    compute_polynomial_curve,
    get_equation,
    plot_pairplot,
    plot_bar_chart,
    plot_zone_pie,
    add_identity_line,
    apply_crossplot_layout,
    basicstats_to_dict,
    fitstats_to_dict,
    add_regression_line,
    build_qc_metrics_table,
    compute_curve,
    plot_hist_kde,
    plot_box,
    plot_zone_boxplot,
    plot_residual_vs_depth, 
    plot_residual_map, 
    build_qc_crossplot,
    build_approximation_plot,
    build_zone_overview_plot, 
    add_qc_line_to_zone_plot,
    show_depth_stats,
    show_outlier_table,
    show_outlier_tables,
)

# ================================================================================================================================ #
# ----------------------------------------------------------  SIDEBAR   ----------------------------------------------------------
# ================================================================================================================================ #
# ------ SESSION_STATE ПРАВИЛА ------
# Все ключи st.session_state используют префикс по принадлежности:
#   data_*    — данные и результаты загрузки (df, removed_rows, raw_*)
#   param_*   — выбранные параметры и единицы (PARAM_1, PARAM_2, *_UNIT)
#   sidebar_* — настройки сайдбара (язык, палитры, цвета)
#   step0_*   — Шаг 0 (корреляционный анализ, pairplot)
#   step1_*   — Шаг 1 (QC-кросс-плот)
#   step2_*   — Шаг 2 (кросс-валидация)
#   step3_*   — Шаг 3 (анализ по категориям)
#   step4_*   — Шаг 4 (фильтрация скважин)
#
# Исключения (уже используются и не переименовываются):
#   - data: df, removed_rows, df_filtered, df_for_cv
#   - param: PARAM_1, PARAM_2, PARAM_1_UNIT, PARAM_2_UNIT, zone_col

# ------ PAGE CONFIG ------
st.set_page_config(page_title="Structural QC — cross-plot & cross-validation", layout="wide", page_icon="✳",)

# ------ SIDEBAR: Language ------
with st.sidebar:
    if "sidebar_language" not in st.session_state:
        st.session_state.sidebar_language = "RU"
    language = st.selectbox(
        "Language",
        ["RU", "EN"],
        index=["RU", "EN"].index(st.session_state.sidebar_language),
        key="sidebar_language_select",
    )
    st.session_state.sidebar_language = language

T = RU if language == "RU" else EN

ui_examples = (ui_examples_full_ru if language == "RU" else ui_examples_full_en)

# ------------------------------------------------------------- #
# SIDEBAR: LABELS
# ------------------------------------------------------------- #
sidebar_data_title = T["sidebar_data_title"]
csv_encoding_error = T["csv_encoding_error"]
required_columns_not_mapped = T["required_columns_not_mapped"]
no_valid_rows_after_cleaning = T["no_valid_rows_after_cleaning"]
no_columns_for_analysis = T["no_columns_for_analysis"]
param1_reference_label = T["param1_reference_label"]
param2_reference_label = T["param2_reference_label"]
param1_units_label = T["param1_units_label"]
param2_units_label = T["param2_units_label"]
not_used_label = T["not_used_label"]
no_demo_label = T["no_demo_label"]

demo_file_select_label = T["demo_file_select_label"]
demo_column_mapping_title = T["demo_column_mapping_title"]
demo_prefix_label = T["demo_prefix_label"]
demo_removed_rows_warning = T["demo_removed_rows_warning"]
demo_loading_error = T["demo_loading_error"]

upload_file_label = T["upload_file_label"]
column_mapping_title = T["column_mapping_title"]
removed_rows_warning = T["removed_rows_warning"]
file_loading_error = T["file_loading_error"]

sidebar_contents_title = T["sidebar_contents_title"]
contents_data = T["contents_data"]
contents_step1 = T["contents_step1"]
contents_step1_cp = T["contents_step1_cp"]
contents_step1_hb = T["contents_step1_hb"]
contents_step1_res = T["contents_step1_res"]
contents_step1_otl = T["contents_step1_otl"]
contents_step2 = T["contents_step2"]
contents_step2_bcv = T["contents_step2_bcv"]
contents_step2_cp = T["contents_step2_cp"]
contents_step2_res = T["contents_step2_res"]
contents_step2_otl = T["contents_step2_otl"]
contents_step3 = T["contents_step3"]
contents_step3_pb = T["contents_step3_pb"]
contents_step3_cp = T["contents_step3_cp"]
contents_step4 = T["contents_step4"]

horizon_expander_title = T["horizon_expander_title"]
all_horizons_label = T["all_horizons_label"]
horizons_label = T["horizons_label"]
selected_horizons_label = T["selected_horizons_label"]
single_horizon_message = T["single_horizon_message"]

corr_pairplot_expander = T["corr_pairplot_expander"]
corr_palette_label = T["corr_palette_label"]
pairplot_point_color_label = T["pairplot_point_color_label"]
pairplot_point_size_label = T["pairplot_point_size_label"]
pairplot_point_opacity_label = T["pairplot_point_opacity_label"]
pairplot_hist_color_label = T["pairplot_hist_color_label"]
pairplot_kde_color_label = T["pairplot_kde_color_label"]
pairplot_category_label = T["pairplot_category_label"]
pairplot_category_none = T["pairplot_category_none"]
pairplot_category_colors_title = T["pairplot_category_colors_title"]
crossplot_expander = T["crossplot_expander"]
crossplot_point_color_label = T["crossplot_point_color_label"]
point_size_label = T["point_size_label"]
point_opacity_label = T["point_opacity_label"]
approx_point_color_label = T["approx_point_color_label"]
approx_point_size_label = T["approx_point_size_label"]
approx_point_opacity_label = T["approx_point_opacity_label"]
regression_expander = T["regression_expander"]
regression_color_label = T["regression_color_label"]
regression_width_label = T["regression_width_label"]
regression_dash_label = T["regression_dash_label"]
identity_line_expander = T["identity_line_expander"]
identity_line_color_label = T["identity_line_color_label"]
identity_line_width_label = T["identity_line_width_label"]
identity_line_dash_label = T["identity_line_dash_label"]
histogram_expander = T["histogram_expander"]
hist_line_style_label = T["hist_line_style_label"]
hist_line_width_label = T["hist_line_width_label"]
hist_fill_color_label = T["hist_fill_color_label"]
kde_curve_label = T["kde_curve_label"]
mean_line_label = T["mean_line_label"]
median_line_label = T["median_line_label"]
sigma_line_label = T["sigma_line_label"]
boxplot_expander = T["boxplot_expander"]
box_fill_color_label = T["box_fill_color_label"]
box_line_width_label = T["box_line_width_label"]
box_mean_label = T["box_mean_label"]
box_median_label = T["box_median_label"]
residual_expander = T["residual_expander"]
residual_color_label = T["residual_color_label"]
residual_map_expander = T["residual_map_expander"]
residual_map_palette_label = T["residual_map_palette_label"]
residual_map_point_size_label = T["residual_map_point_size_label"]
residual_map_point_opacity_label = T["residual_map_point_opacity_label"]
bar_chart_expander = T["bar_chart_expander"]
rmse_palette_label = T["rmse_palette_label"]
category_bar_palette_label = T["category_bar_palette_label"]
pie_chart_expander = T["pie_chart_expander"]
pie_palette_label = T["pie_palette_label"]

approx_expander = T["approx_expander"]
all_approximations_label_step1 = T["all_approximations_label_step1"]
all_approximations_label_step2 = T["all_approximations_label_step2"]
all_approximations_label_step3 = T["all_approximations_label_step3"]
approx_types_step1_label = T["approx_types_step1_label"]
approx_types_step2_label = T["approx_types_step2_label"]
approx_types_step3_label = T["approx_types_step3_label"]
approx_colors_expander = T["approx_colors_expander"]
quantile_regression_title = T["quantile_regression_title"]
approx_line_width_label = T["approx_line_width_label"]

correlation_expander = T["correlation_expander"]
all_parameters_label = T["all_parameters_label"]
corr_parameters_label = T["corr_parameters_label"]
corr_warning = T["corr_warning"]
pairplot_type_label = T["pairplot_type_label"]
pairplot_diag_label = T["pairplot_diag_label"]
pairplot_scale_label = T["pairplot_scale_label"]
pairplot_size_label = T["pairplot_size_label"]
pairplot_corner_label = T["pairplot_corner_label"]

outlier_expander = T["outlier_expander"]
sigma_threshold_label = T["sigma_threshold_label"]

category_expander = T["category_expander"]
category_parameter_label = T["category_parameter_label"]
category_point_color_label = T["category_point_color_label"]
category_line_color_label =  T["category_line_color_label"]
category_point_size_label =  T["category_point_size_label"]
category_point_opacity_label = T["category_point_opacity_label"]
category_wells_suffix = T["category_wells_suffix"]

well_filter_expander = T["well_filter_expander"]
filter_mode_label = T["filter_mode_label"]
filter_mode_none = T["filter_mode_none"]
filter_mode_manual = T["filter_mode_manual"]
filter_mode_random = T["filter_mode_random"]
current_filter_mode = T["current_filter_mode"]
exclude_wells_label = T["exclude_wells_label"]
excluded_wells_count = T["excluded_wells_count"]
percent_exclude_label = T["percent_exclude_label"]
new_random_set_button = T["new_random_set_button"]
random_exclusion_caption = T["random_exclusion_caption"]
current_random_set = T["current_random_set"]
will_be_excluded = T["will_be_excluded"]
wells_suffix = T["wells_suffix"]
filter_mode_click = T["filter_mode_click"]
step4_click_title = T["step4_click_title"]
step4_click_hint = T["step4_click_hint"]
step4_click_selected_count = T["step4_click_selected_count"]
step4_click_excluded_count = T["step4_click_excluded_count"]
step4_click_apply_button = T["step4_click_apply_button"]
step4_click_reset_button = T["step4_click_reset_button"]
step4_click_excluded_expander = T["step4_click_excluded_expander"]
step4_click_excluded_wells = T["step4_click_excluded_wells"]
step4_summary_click = T["step4_summary_click"]

cv_validation_expander = T["cv_validation_expander"]
random_kfold_title = T["random_kfold_title"]
recommended_k_text = T["recommended_k_text"]
wells_count_text = T["wells_count_text"]
kfold_caption = T["kfold_caption"]
kfold_slider_label = T["kfold_slider_label"]
block_cv_title = T["block_cv_title"]
recommended_grid_text = T["recommended_grid_text"]
median_distance_text = T["median_distance_text"]
grid_caption = T["grid_caption"]
grid_slider_label = T["grid_slider_label"]
recommended_buffer_text = T["recommended_buffer_text"]
buffer_based_on_median = T["buffer_based_on_median"]
buffer_caption = T["buffer_caption"]
buffer_slider_label = T["buffer_slider_label"]
leave_p_out_title = T["leave_p_out_title"]
recommended_repeats_text = T["recommended_repeats_text"]
cv_repeats_label = T["cv_repeats_label"]
cv_repeats_caption = T["cv_repeats_caption"]
leave_p_percent_label = T["leave_p_percent_label"]
leave_p_percent_caption = T["leave_p_percent_caption"]
spatial_leave_p_out_title = T["spatial_leave_p_out_title"]
recommended_radius_text = T["recommended_radius_text"]
spatial_radius_caption = T["spatial_radius_caption"]
spatial_radius_label = T["spatial_radius_label"]
distance_cv_title = T["distance_cv_title"]
variogram_factor_label = T["variogram_factor_label"]
variogram_factor_caption = T["variogram_factor_caption"]

# ------------------------------------------------------------- #
# SIDEBAR: Подготовка данных
# ------------------------------------------------------------- #

# ------ Проверка существования колонки ------
def has_col(df, col):
    return col in df.columns and df[col].notna().any()

# ------ Чтение CSV/XLSX ------
def read_input_file(path_or_file):

    filename = (path_or_file.name if hasattr(path_or_file, "name") else str(path_or_file)).lower()
    if filename.endswith((".xls", ".xlsx")):
        return pd.read_excel(path_or_file)

    encodings = ["utf-8", "cp1251", "windows-1251", "latin1",]
    for encoding in encodings:
        try:
            return pd.read_csv(path_or_file, encoding=encoding, sep=None, engine="python",)
        except Exception:
            pass

    raise ValueError(csv_encoding_error)

def save_dataset_to_session(df, param_1, param_2, unit_1, unit_2, zone_col=None,):
    # Определяем "отпечаток" — если данные и параметры не изменились, ничего не сбрасываем
    fingerprint = (
        len(df) if df is not None else None,
        tuple(df.columns.tolist()) if df is not None else None,
        param_1, param_2, unit_1, unit_2, zone_col,
    )
    prev_fingerprint = st.session_state.get("_dataset_fingerprint")

    st.session_state["df"] = df
    st.session_state["PARAM_1"] = param_1
    st.session_state["PARAM_2"] = param_2
    st.session_state["PARAM_1_UNIT"] = unit_1
    st.session_state["PARAM_2_UNIT"] = unit_2
    st.session_state["zone_col"] = zone_col

    # Если набор данных тот же — выходим и не трогаем производные
    if prev_fingerprint == fingerprint:
        return

    st.session_state["_dataset_fingerprint"] = fingerprint

    # Данные изменились — сбрасываем производные
    st.session_state["df_filtered"] = None
    st.session_state["df_for_cv"] = None

    for key in [
        "step0_corr_columns",
        "step0_all_parameters",
        "sidebar_all_horizons",
        "sidebar_picked_horizons",
        "category_col",
        "pairplot_category", 
        "step3_selected_zones",
    ]:
        st.session_state.pop(key, None)


# ------ PREPARE INPUT DF ------
def prepare_input_dataframe(raw_df, well_col, x_col, y_col, param_1_col,  param_2_col, horizon_col=None, zone_col=None,):

    # сохраняем все колонки исходного файла
    df = raw_df.copy()
    df.drop(columns=[well_col, x_col, y_col], inplace=True, errors="ignore",)

    # Страховка: убираем одноимённые «чужие» системные колонки, чтобы присваивание ниже не затирало данные
    reserved = {"well_id", "X_coord", "Y_coord", "Zprm_well", "Zprm_map", "horizon_id", "tectonic_zone"}
    sources = {well_col, x_col, y_col, param_1_col, param_2_col, horizon_col, zone_col}
    df.drop(columns=[c for c in reserved if c in df.columns and c not in sources],
            inplace=True, errors="ignore")

    # Системные колонки
    df["well_id"] = raw_df[well_col]
    df["X_coord"] = raw_df[x_col]
    df["Y_coord"] = raw_df[y_col]

    df["Zprm_well"] = raw_df[param_1_col]
    df["Zprm_map"] = raw_df[param_2_col]

    # Необязательные колонки: создаём системные алиасы
    # и убираем исходную колонку, чтобы не было дублей в категориальных списках
    if horizon_col:
        df["horizon_id"] = raw_df[horizon_col]
        if horizon_col != "horizon_id":
            df.drop(columns=[horizon_col], inplace=True, errors="ignore")

    if zone_col:
        df["tectonic_zone"] = raw_df[zone_col]
        if zone_col != "tectonic_zone":
            df.drop(columns=[zone_col], inplace=True, errors="ignore")

    # Проверка обязательных колонок
    required = ["well_id", "X_coord", "Y_coord", "Zprm_well", "Zprm_map",]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{required_columns_not_mapped} {missing}")

    # Очистка специальных и мусорных значений
    bad_values = [
        "", " ", "  ", "   ",
        "NA", "N/A",
        "NULL", "null",
        "None", "none",
        "NaN", "nan",
        "-", "--", "---", "—",
        "?", "*",
    ]

    df = df.replace(bad_values, np.nan,)

    # Числовые колонки
    numeric_cols = ["X_coord", "Y_coord", "Zprm_well", "Zprm_map",]

    for c in numeric_cols:

        # Если колонка уже numeric — только чистим NaN/Inf
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
            continue

        # пустые строки -> NaN
        df[c] = df[c].replace(r"^\s*$", np.nan, regex=True,)

        # запятая -> точка
        df[c] = (df[c].astype(str).str.strip().str.replace(",", ".", regex=False))

        # перевод в число
        df[c] = pd.to_numeric(df[c], errors="coerce",)

    # Очистка строковых полей
    text_cols = [
        c
        for c in ["well_id", "horizon_id", "tectonic_zone",]
        if c in df.columns
    ]

    for c in text_cols:
        df[c] = (df[c].astype(str).str.strip())
        df[c] = df[c].replace(
            {
                "nan": np.nan,
                "None": np.nan,
                "NULL": np.nan,
            }
        )

    # Удаление строк без обязательных данных
    required_numeric_cols = ["X_coord", "Y_coord", "Zprm_well", "Zprm_map",]
    removed_rows = df[df[required_numeric_cols].isna().any(axis=1)].copy()
    before = len(df)
    df = (df.dropna(subset=["X_coord", "Y_coord", "Zprm_well", "Zprm_map",]).reset_index(drop=True))
    dropped = before - len(df)

    # Контроль результата очистки
    if len(df) == 0:
        raise ValueError(no_valid_rows_after_cleaning)

    return df, dropped, removed_rows
     
# ------ SIDEBAR — данные и настройки ------
st.sidebar.title(sidebar_data_title)
UNIT_OPTIONS = [
    "m", "km", "ft", "ms", "sec", "m/s", "ft/s", "units", "%", "Hz","dB",
    "°C", "MPa", "psi", "ppm", "g/cm³"
]

# ------ SESSION STATE — данные ------
defaults = {
    "df": None,
    "PARAM_1": "PARAM_1",
    "PARAM_2": "PARAM_2",
    "PARAM_1_UNIT": "units",
    "PARAM_2_UNIT": "units",
}

for key, value in defaults.items():
    st.session_state.setdefault(key, value)

def show_column_mapping_sidebar(cols, prefix_label="", prefix_key="",):

    well_col = st.sidebar.selectbox(f"{prefix_label}well_id", cols, key=f"{prefix_key}_well",)
    x_col = st.sidebar.selectbox(f"{prefix_label}X_coord", cols, key=f"{prefix_key}_x",)
    y_col = st.sidebar.selectbox(f"{prefix_label}Y_coord", cols, key=f"{prefix_key}_y",)

    excluded_cols = {well_col, x_col, y_col}
    analysis_cols = [c for c in cols if c not in excluded_cols]

    if len(analysis_cols) == 0:
        raise ValueError(no_columns_for_analysis)

    param_1_col = st.sidebar.selectbox(f"{prefix_label}{param1_reference_label}", analysis_cols, key=f"{prefix_key}_p1",)
    param_2_col = st.sidebar.selectbox(f"{prefix_label}{param2_reference_label}", analysis_cols, index=min(1, len(analysis_cols) - 1), key=f"{prefix_key}_p2",)
    unit_1 = st.sidebar.selectbox(f"{prefix_label}{param1_units_label}", UNIT_OPTIONS, key=f"{prefix_key}_unit1",)
    unit_2 = st.sidebar.selectbox(f"{prefix_label}{param2_units_label}", UNIT_OPTIONS, index=UNIT_OPTIONS.index(unit_1), key=f"{prefix_key}_unit2",)
    optional_cols = [NOT_USED] + cols
    horizon_col = st.sidebar.selectbox(f"{prefix_label}horizon_id", optional_cols, key=f"{prefix_key}_horizon",)
    zone_col = st.sidebar.selectbox(f"{prefix_label}tectonic_zone", optional_cols, key=f"{prefix_key}_zone",)

    return (
        well_col,
        x_col,
        y_col,
        param_1_col,
        param_2_col,
        unit_1,
        unit_2,
        horizon_col,
        zone_col,
    )

# ------------------------------------------------------------- #
# SIDEBAR: DEMO-файл (пример данных)
# ------------------------------------------------------------- #
NOT_USED = not_used_label
NO_DEMO = no_demo_label

# ------ Автоматический поиск демонстрационных файлов ------
# demo_dir = "data"
demo_dir = Path(__file__).parent / "data"

if os.path.isdir(demo_dir):
    demo_files = [
        f for f in os.listdir(demo_dir)
        if f.lower().endswith((".csv", ".xls", ".xlsx"))
    ]
else:
    demo_files = []

picked_demo = st.sidebar.selectbox(demo_file_select_label, [NO_DEMO] + demo_files, key="demo_file_select",)

# ------ Отслеживание активного источника (demo / uploaded) ------
st.session_state.setdefault("_last_demo_choice", None)
st.session_state.setdefault("_last_uploaded_name", None)

if picked_demo != st.session_state["_last_demo_choice"]:
    # пользователь явно сменил demo-файл
    if picked_demo != NO_DEMO:
        st.session_state["active_source"] = "demo"
    else:
        # выбрал NO_DEMO — сбрасываем demo, но файл (если есть) остаётся активным
        if st.session_state.get("_last_uploaded_name") is not None:
            st.session_state["active_source"] = "uploaded"
        else:
            st.session_state["active_source"] = None

st.session_state["_last_demo_choice"] = picked_demo

# ------ Загрузка демонстрационного файла ------
active_source = st.session_state.get("active_source")

if active_source == "demo" and picked_demo != NO_DEMO:
    try:
        # Кэшируем прочитанный demo только при смене файла
        if st.session_state.get("picked_demo") != picked_demo:
            demo_path = os.path.join(demo_dir, picked_demo)
            raw_demo = read_input_file(demo_path)
            st.session_state["raw_demo_df"] = raw_demo
            st.session_state["demo_cols"] = raw_demo.columns.tolist()
            st.session_state["picked_demo"] = picked_demo
        else:
            raw_demo = st.session_state["raw_demo_df"]

        demo_cols = st.session_state["demo_cols"]

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {demo_column_mapping_title}")
    
        (
         demo_well_col, demo_x_col, demo_y_col, demo_param_1_col, demo_param_2_col,
         demo_unit_1, demo_unit_2, demo_horizon_col, demo_zone_col,
        ) = show_column_mapping_sidebar(demo_cols, prefix_label=demo_prefix_label, prefix_key="demo",)

        # Подготовка данных
        df_demo, dropped_demo, removed_demo_rows = prepare_input_dataframe(
            raw_demo,
            demo_well_col,
            demo_x_col,
            demo_y_col,
            demo_param_1_col,
            demo_param_2_col,
            None if demo_horizon_col == NOT_USED else demo_horizon_col,
            None if demo_zone_col == NOT_USED else demo_zone_col,
        )
        st.session_state["removed_rows"] = removed_demo_rows

        if dropped_demo:
            st.sidebar.warning(
            f"{demo_removed_rows_warning} "
            f"{dropped_demo}"
            )
        save_dataset_to_session(
            df_demo,
            demo_param_1_col,
            demo_param_2_col,
            demo_unit_1,
            demo_unit_2,
            None if demo_zone_col == NOT_USED else demo_zone_col,
        )

    except Exception as e:
        st.sidebar.error(f"{demo_loading_error} {e}")

# ------------------------------------------------------------- #
# SIDEBAR: Загрузить CSV/XLSX
# ------------------------------------------------------------- #
# ------ Загрузка пользовательского файла + сопоставление колонок ------
uploaded = st.sidebar.file_uploader(upload_file_label, type=["csv", "xls", "xlsx"])

# Отслеживаем изменение загруженного файла
current_uploaded_name = uploaded.name if uploaded is not None else None

if current_uploaded_name != st.session_state["_last_uploaded_name"]:
    if current_uploaded_name is not None:
        # пользователь загрузил файл — источник = uploaded
        st.session_state["active_source"] = "uploaded"
    elif picked_demo != NO_DEMO:
        # очистил файл, demo выбран — возвращаемся к demo
        st.session_state["active_source"] = "demo"
    else:
        # очистил файл, demo не выбран
        st.session_state["active_source"] = None

st.session_state["_last_uploaded_name"] = current_uploaded_name

active_source = st.session_state.get("active_source")

if active_source == "uploaded" and uploaded is not None:
    try:

        # Читаем исходный файл (uploaded всегда не None внутри этого блока)
        raw_df = read_input_file(uploaded)
        cols = raw_df.columns.tolist()
        st.session_state["raw_uploaded_df"] = raw_df
        st.session_state["uploaded_cols"] = cols

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {column_mapping_title}")

        (
         well_col, x_col, y_col, param_1_col, param_2_col,
         unit_1, unit_2, horizon_col, zone_col,
        ) = show_column_mapping_sidebar(cols, prefix_label="", prefix_key="user",)

        # Подготовка данных
        df_user, dropped, removed_rows = prepare_input_dataframe(
            raw_df,
            well_col,
            x_col,
            y_col,
            param_1_col,
            param_2_col,
            None if horizon_col == NOT_USED else horizon_col,
            None if zone_col == NOT_USED else zone_col,
        )
        st.session_state["removed_rows"] = removed_rows

        if dropped:
            st.sidebar.warning(
                f"{removed_rows_warning} "
                f"{dropped}"
            )
        # Сохранение в Session State
        save_dataset_to_session(
            df_user,
            param_1_col,
            param_2_col,
            unit_1,
            unit_2,
            None if zone_col == NOT_USED else zone_col,
        )

    except Exception as e:
        st.sidebar.error(f"{file_loading_error} {e}")

# ------------------------------------------------------------- #
# Текущие параметры проекта: PARAM
# ------------------------------------------------------------- #

df = st.session_state["df"]

PARAM_1 = st.session_state.get("PARAM_1", "PARAM_1")
PARAM_2 = st.session_state.get("PARAM_2", "PARAM_2")

PARAM_1_UNIT = st.session_state.get("PARAM_1_UNIT", "units")
PARAM_2_UNIT = st.session_state.get("PARAM_2_UNIT", "units")

PARAM_1_LABEL = f"{PARAM_1}, {PARAM_1_UNIT}"
PARAM_2_LABEL = f"{PARAM_2}, {PARAM_2_UNIT}"

DELTA_LABEL = f"Δ({PARAM_1} − {PARAM_2})"

# Флаги наличия необязательных колонок
has_horizon = df is not None and "horizon_id" in df.columns
has_zone = df is not None and "tectonic_zone" in df.columns

# Отображаемые названия служебных параметров
DISPLAY_NAMES = {
    "Zprm_well": PARAM_1,
    "Zprm_map": PARAM_2,
}

def display_name(col):
    return DISPLAY_NAMES.get(col, col)

# ------------------------------------------------------------- #
# SIDEBAR: Содержание
# ------------------------------------------------------------- #
st.sidebar.markdown("---")
st.sidebar.markdown(f"### {sidebar_contents_title}")

st.sidebar.markdown(f"""
- [{contents_data}](#step-0)
- [{contents_step1}](#step-1)
    - [{contents_step1_cp}](#step-1-cp)
    - [{contents_step1_hb}](#step-1-hb)
    - [{contents_step1_res}](#step-1-res) 
    - [{contents_step1_otl}](#step-1-otl)
- [{contents_step2}](#step-2)
    - [{contents_step2_bcv}](#step-2-bcv)
    - [{contents_step2_cp}](#step-2-cp)
    - [{contents_step2_res}](#step-2-res) 
    - [{contents_step2_otl}](#step-2-otl)
- [{contents_step3}](#step-3)
    - [{contents_step3_pb}](#step-3-pb)
    - [{contents_step3_cp}](#step-3-cp)
- [{contents_step4}](#step-4)
""")

# ------------------------------------------------------------- #
# SIDEBAR: Горизонт
# ------------------------------------------------------------- #
st.sidebar.markdown("---")

with st.sidebar.expander(horizon_expander_title, expanded=False):

    if (df is not None and has_col(df, "horizon_id") and df["horizon_id"].nunique() > 1):

        horizons = sorted(df["horizon_id"].dropna().unique().tolist())
        all_horizons = st.checkbox(all_horizons_label, value=True, key="sidebar_all_horizons")
        picked_horizons = st.multiselect(horizons_label, options=horizons, default=(horizons if all_horizons else []), key="sidebar_picked_horizons",)
        st.caption(
            f"{selected_horizons_label} "
            f"{len(picked_horizons)} из {len(horizons)}"
        )

        if len(picked_horizons) > 0:
            df = df[df["horizon_id"].isin(picked_horizons)].reset_index(drop=True)
            st.session_state["df_filtered"] = df

    else:
        picked_horizons = []
        st.multiselect(horizons_label, options=[], default=[], disabled=True)
        st.caption(single_horizon_message)
        st.session_state["df_filtered"] = df

# --- фиксируем отфильтрованный df, чтобы при смене языка не пересобирался ---
if st.session_state.get("df_filtered") is not None:
    df = st.session_state["df_filtered"]

# --- замораживаем df для кэша CV, чтобы смена языка не инвалидировала кэш ---
if df is not None:
    if "df_for_cv" not in st.session_state or st.session_state["df_for_cv"] is None:
        st.session_state["df_for_cv"] = df.copy()
    else:
        cached = st.session_state["df_for_cv"]
        if (
            len(cached) != len(df)
            or not cached["well_id"].equals(df["well_id"])
        ):
            st.session_state["df_for_cv"] = df.copy()
else:
    st.session_state["df_for_cv"] = None

# ------------------------------------------------------------- #
# SIDEBAR: Цвета графиков
# ------------------------------------------------------------- #
st.sidebar.markdown("---")

ALL_COLORS = [
    "black", "blue", "brown", "cadetblue", "chocolate", "coral", "cornflowerblue", "crimson","cyan",
    "darkblue", "darkcyan", "darkgoldenrod", "darkgreen", "darkmagenta", "darkorange", "darkorchid", "darkslateblue", "darkslategray",
    "darkred", "deepskyblue", "dodgerblue","firebrick", "forestgreen", "gold", "goldenrod", "gray", "green", "indianred",
    "indigo", "limegreen","lightgray", "magenta", "mediumblue", "mediumorchid", "mediumseagreen", "midnightblue","navy", "olive", "orange",
    "orangered", "peru", "purple", "red", "royalblue", "saddlebrown", "seagreen", "slateblue",
    "steelblue", "teal", "tomato"
]

DASH_OPTIONS = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

MAP_PALETTES  = [
    # карты
    "Aggrnyl", "Balance", "Blackbody", "Bluered", "Blues", "Cividis", "Earth", "Electric", "Greens", "Greys",
    "Hot", "IceFire", "Inferno", "Jet", "Magma", "Oranges", "Picnic", "Plasma", "Portland", "Purples",
    "Rainbow", "RdBu", "RdYlBu", "RdYlGn", "Reds", "Spectral", "Teal", "Tealgrn", "Temps", "Turbo",
    "Viridis", "YlGnBu", "YlOrRd",
]

CATEGORY_PALETTES = [
    # категориальные
    "Alphabet", "Bold", "Cividis", "Dark24", "Light24", "Pastel", "Plotly", "Portland", "Set1", "Set2",
    "Set3", "Spectral", "Turbo", "Viridis"
]

with st.sidebar.expander(corr_pairplot_expander, expanded=False):
    corr_palette = st.selectbox(corr_palette_label, MAP_PALETTES, index=MAP_PALETTES.index("RdBu"), key="corr_palette")
    pairplot_point_color = st.selectbox(pairplot_point_color_label, ALL_COLORS, index=ALL_COLORS.index("steelblue"), key="pairplot_point_color")
    pairplot_point_size = st.slider(pairplot_point_size_label, 2, 20, 5, 1, key="pairplot_point_size",)
    pairplot_point_opacity = st.slider(pairplot_point_opacity_label, 0.1, 1.0, 0.7, 0.05, key="pairplot_point_opacity",)

    # Категория для цвета точек Pairplot (опционально) — работает и для Seaborn, и для Plotly
    pairplot_category_candidates = []
    if df is not None:
        pairplot_category_candidates = [
            c for c in df.columns
            if (
                c not in ["well_id", "X_coord", "Y_coord", "Zprm_well", "Zprm_map"]
                and not pd.api.types.is_numeric_dtype(df[c])
            )
        ]

    pairplot_category_options = [pairplot_category_none] + pairplot_category_candidates

    # Умный default:
    #   1. zone_col (если пользователь явно выбрал его при загрузке данных)
    #   2. первая доступная категория
    #   3. "— не использовать —" (если категорий нет)
    default_pairplot_category = pairplot_category_none
    zone_col_default = st.session_state.get("zone_col")

    if (
        zone_col_default is not None
        and zone_col_default in pairplot_category_candidates
    ):
        default_pairplot_category = zone_col_default
    elif pairplot_category_candidates:
        default_pairplot_category = pairplot_category_candidates[0]

    pairplot_category = st.selectbox(
        pairplot_category_label,
        pairplot_category_options,
        index=pairplot_category_options.index(default_pairplot_category),
        key="pairplot_category",
    )
    if pairplot_category == pairplot_category_none:
        pairplot_category = None

    # Цвета категорий — как в Шаге 3, но со своим префиксом ключей
    pairplot_category_colors = {}
    if pairplot_category is not None and df is not None and pairplot_category in df.columns:
        st.markdown(f"**{pairplot_category_colors_title}**")

        # Собираем список категорий в том же виде, как их увидит Plotly/Seaborn:
        # str, без NaN (NaN заменён на "(not specified)" в Plotly-ветке)
        cats_series = (
            df[pairplot_category]
            .fillna("(not specified)")
            .astype(str)
        )
        pairplot_categories_list = sorted(cats_series.unique())

        for i, cat in enumerate(pairplot_categories_list):
            pairplot_category_colors[cat] = st.selectbox(
                cat,
                ALL_COLORS,
                index=i % len(ALL_COLORS),
                key=f"pairplot_cat_color_{pairplot_category}_{cat}",
            )
    # ----- Настройки типа Pairplot и отображения -----
    st.markdown("---")

    pairplot_options = ["Seaborn Pairplot (static)", "Plotly Pairplot (interactive)"]
    pairplot_engine = st.selectbox(
        pairplot_type_label,
        pairplot_options,
        index=pairplot_options.index("Plotly Pairplot (interactive)"),
        key="pairplot_engine",
    )

    if pairplot_engine == "Seaborn Pairplot (static)":
        # Настройки Seaborn Pairplot
        diag_kind = st.selectbox(pairplot_diag_label, ["kde", "hist"], index=0, key="diag_kind",)
        pairplot_scale = st.slider(pairplot_scale_label, 0.5, 6.0, 1.0, 0.1, key="pairplot_scale",)
        pairplot_size = st.slider(pairplot_size_label, 0.8, 6.0, 1.8, 0.1, key="pairplot_size",)
        pairplot_corner = st.checkbox(pairplot_corner_label, value=False, key="pairplot_corner",)

        # ← Цвета Seaborn переносим сюда
        pairplot_hist_color = st.selectbox(pairplot_hist_color_label, ALL_COLORS, index=ALL_COLORS.index("orange"), key="pairplot_hist_color")
        pairplot_kde_color = st.selectbox(pairplot_kde_color_label, ALL_COLORS, index=ALL_COLORS.index("black"), key="pairplot_kde_color")

    else:
        diag_kind = "kde"
        pairplot_scale = 1.0
        pairplot_size = 1.8
        pairplot_corner = False

        # Дефолты для Plotly-режима, чтобы переменные не были undefined
        pairplot_hist_color = "orange"
        pairplot_kde_color = "black"

with st.sidebar.expander(crossplot_expander, expanded=False):
    crossplot_point_color = st.selectbox(crossplot_point_color_label, ALL_COLORS, index=ALL_COLORS.index("black"), key="crossplot_point_color")
    crossplot_point_size = st.slider(point_size_label, 3, 20, 7, key="crossplot_point_size",)
    crossplot_point_opacity = st.slider(point_opacity_label, 0.1, 1.0, 0.8, 0.05, key="crossplot_point_opacity",)
    approx_point_color = st.selectbox(approx_point_color_label, ALL_COLORS, index=ALL_COLORS.index("brown"), key="approx_point_color")
    approx_point_size = st.slider(approx_point_size_label, 3, 20, 7, key="approx_point_size",)
    approx_point_opacity = st.slider(approx_point_opacity_label, 0.1, 1.0, 0.8, 0.05, key="approx_point_opacity",)

with st.sidebar.expander(regression_expander, expanded=False):
    qc_regression_color = st.selectbox(regression_color_label, ALL_COLORS, index=ALL_COLORS.index("orange"), key="qc_regression_color")
    qc_regression_width = st.slider(regression_width_label, 1, 8, 3, key="qc_regression_width",)
    qc_regression_dash = st.selectbox(regression_dash_label, ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot",], index=0, key="qc_regression_dash")

with st.sidebar.expander(identity_line_expander, expanded=False):
    identity_line_color = st.selectbox(identity_line_color_label, ALL_COLORS, index=ALL_COLORS.index("lightgray"), key="identity_line_color")
    identity_line_width = st.slider(identity_line_width_label, 1, 8, 3, key="identity_line_width",)
    identity_line_dash = st.selectbox(identity_line_dash_label, ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot",], index=1, key="identity_line_dash")

with st.sidebar.expander(histogram_expander, expanded=False):

    # Заливка гистограммы — только цвет
    hist_color = st.selectbox(hist_fill_color_label, ALL_COLORS, index=ALL_COLORS.index("orange"), key="hist_color")

    st.markdown("---")

    # ---------- KDE ----------
    st.markdown(f"**{kde_curve_label}**")
    kde_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("black"), key="kde_color", label_visibility="collapsed",)
    kde_dash = st.selectbox(hist_line_style_label, DASH_OPTIONS, index=DASH_OPTIONS.index("solid"), key="kde_dash", label_visibility="collapsed",)
    kde_width = st.slider(hist_line_width_label, 1, 10, 3, key="kde_width", label_visibility="collapsed",)

    # ---------- Mean ----------
    st.markdown(f"**{mean_line_label}**")
    mean_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("green"), key="mean_color", label_visibility="collapsed",)
    mean_dash = st.selectbox(hist_line_style_label, DASH_OPTIONS, index=DASH_OPTIONS.index("dash"), key="mean_dash", label_visibility="collapsed",)
    mean_width = st.slider(hist_line_width_label, 1, 10, 2, key="mean_width", label_visibility="collapsed",)

    # ---------- Median ----------
    st.markdown(f"**{median_line_label}**")
    median_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("red"), key="median_color", label_visibility="collapsed",)
    median_dash = st.selectbox(hist_line_style_label, DASH_OPTIONS, index=DASH_OPTIONS.index("dot"), key="median_dash", label_visibility="collapsed",)
    median_width = st.slider(hist_line_width_label, 1, 10, 2, key="median_width", label_visibility="collapsed",)

    # ---------- ±3σ ----------
    st.markdown(f"**{sigma_line_label}**")
    sigma_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("orange"), key="sigma_color", label_visibility="collapsed",)
    sigma_dash = st.selectbox(hist_line_style_label, DASH_OPTIONS, index=DASH_OPTIONS.index("dash"), key="sigma_dash", label_visibility="collapsed",)
    sigma_width = st.slider(hist_line_width_label, 1, 10, 2, key="sigma_width", label_visibility="collapsed",)

with st.sidebar.expander(boxplot_expander, expanded=False):

    # Заливка боксплота — только цвет
    box_color = st.selectbox(box_fill_color_label, ALL_COLORS, index=ALL_COLORS.index("royalblue"), key="box_color")

    st.markdown("---")

    # ---------- Mean ----------
    st.markdown(f"**{box_mean_label}**")
    box_mean_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("green"), key="box_mean_color", label_visibility="collapsed",)
    box_mean_dash = st.selectbox(hist_line_style_label, DASH_OPTIONS, index=DASH_OPTIONS.index("dash"), key="box_mean_dash", label_visibility="collapsed",)
    box_mean_width = st.slider(hist_line_width_label, 1, 10, 2, key="box_mean_width", label_visibility="collapsed",)

    # ---------- Median ----------
    st.markdown(f"**{box_median_label}**")
    box_median_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("red"), key="box_median_color", label_visibility="collapsed",)
    box_median_dash = st.selectbox(hist_line_style_label, DASH_OPTIONS, index=DASH_OPTIONS.index("dot"), key="box_median_dash", label_visibility="collapsed",)
    box_median_width = st.slider(hist_line_width_label, 1, 10, 2, key="box_median_width", label_visibility="collapsed",)

with st.sidebar.expander(residual_expander, expanded=False):
    residual_color = st.selectbox(residual_color_label, ALL_COLORS, index=ALL_COLORS.index("black"), key="residual_color")       # black
    residual_point_size = st.slider(point_size_label, 3, 20, 8, key="residual_point_size",)
    residual_point_opacity = st.slider(point_opacity_label, 0.1, 1.0, 0.85, 0.05, key="residual_point_opacity",)

with st.sidebar.expander(residual_map_expander, expanded=False):
    map_palette = st.selectbox(residual_map_palette_label, MAP_PALETTES, index=MAP_PALETTES.index("RdBu"), key="map_palette")                   # RdBu
    residual_map_point_size = st.slider(residual_map_point_size_label, 3, 30, 12,  key="residual_map_point_size",)
    residual_map_point_opacity = st.slider(residual_map_point_opacity_label, 0.1, 1.0, 0.9, 0.05, key="residual_map_point_opacity",)

with st.sidebar.expander(bar_chart_expander, expanded=False):
    rmse_palette = st.selectbox(rmse_palette_label, CATEGORY_PALETTES, index=CATEGORY_PALETTES.index("Plotly"), key="rmse_palette")              # Plotly
    cat_palette = st.selectbox(category_bar_palette_label, CATEGORY_PALETTES, index=CATEGORY_PALETTES.index("Plotly"), key="cat_palette")              # Plotly

with st.sidebar.expander(pie_chart_expander, expanded=False):
    pie_palette = st.selectbox(pie_palette_label, CATEGORY_PALETTES, index=CATEGORY_PALETTES.index("Plotly"), key="pie_palette")

# ------------------------------------------------------------- #
# SIDEBAR: Выбор типов и цветов аппроксимаций в Шагах 1–3
# ------------------------------------------------------------- #
DEFAULT_APPROX_COLORS = {
    "Linear": "orange",
    "Quadratic": "green",
    "Cubic": "purple",
    "Polynomial Degree 4": "darkgreen",
    "Polynomial Degree 5": "darkred",
    "Polynomial Degree 6": "black",
    "Polynomial Degree 7": "blue",
    "Polynomial Degree 8": "brown",
    "Polynomial Degree 9": "cadetblue",
    "Polynomial Degree 10": "coral",
    "Power Law": "magenta",
    "Exponential": "red",
    "Logarithmic": "brown",
    "Robust (Huber)": "black",
    "Theil-Sen": "darkcyan",
    "RANSAC": "darkmagenta",
    "LOWESS": "cyan"
}

DEFAULT_QUANTILE_COLORS = {
    0.1: "darkred",
    0.5: "black",
    0.9: "darkgreen",
}


COLOR_OPTIONS = ALL_COLORS

DEFAULT_APPROX_DASHES = {
    "Linear": "solid",
    "Quadratic": "solid",
    "Cubic": "solid",
    "Polynomial Degree 4": "solid",
    "Polynomial Degree 5": "solid",
    "Polynomial Degree 6": "solid",
    "Polynomial Degree 7": "solid",
    "Polynomial Degree 8": "solid",
    "Polynomial Degree 9": "solid",
    "Polynomial Degree 10": "solid",
    "Power Law": "solid",
    "Exponential": "solid",
    "Logarithmic": "solid",
    "Robust (Huber)": "solid",
    "Theil-Sen": "solid",
    "RANSAC": "solid",
    "LOWESS": "solid",
}

APPROX_OPTIONS = [
    "Linear",
    "Quadratic",
    "Cubic",
    "Polynomial Degree 4",
    "Polynomial Degree 5",
    "Polynomial Degree 6",
    "Polynomial Degree 7",
    "Polynomial Degree 8",
    "Polynomial Degree 9",
    "Polynomial Degree 10",
    "Power Law",
    "Exponential",
    "Logarithmic",
    "Robust (Huber)",
    "Theil-Sen",
    "RANSAC",
    "LOWESS",
    "Quantile Regression (τ=0.1/0.5/0.9)",
]

NON_QUANTILE_APPROX_OPTIONS = [
    a
    for a in APPROX_OPTIONS
    if a != "Quantile Regression (τ=0.1/0.5/0.9)"
]

st.sidebar.markdown("---")

with st.sidebar.expander(approx_expander, expanded=False):

    # --- Шаг 1 ---
    st.markdown(f"**{approx_types_step1_label}**")
    all_approximations_step1 = st.checkbox(
        all_approximations_label_step1,
        key="all_approximations_step1",
    )
    if all_approximations_step1:
        approx_types = APPROX_OPTIONS
    else:
        approx_types = st.multiselect(
            approx_types_step1_label,
            APPROX_OPTIONS,
            default=["Linear"],
            key="approx_types",
        )

    st.markdown("---")

    # --- Шаг 2 ---
    st.markdown(f"**{approx_types_step2_label}**")
    all_approximations_step2 = st.checkbox(
        all_approximations_label_step2,
        key="all_approximations_step2",
    )
    if all_approximations_step2:
        cv_approx_types = NON_QUANTILE_APPROX_OPTIONS
    else:
        cv_approx_types = st.multiselect(
            approx_types_step2_label,
            NON_QUANTILE_APPROX_OPTIONS,
            default=["Linear"],
            key="cv_approx_types",
        )

    st.markdown("---")

    # --- Шаг 3 ---
    st.markdown(f"**{approx_types_step3_label}**")
    all_approximations_step3 = st.checkbox(
        all_approximations_label_step3,
        key="all_approximations_step3",
    )
    if all_approximations_step3:
        zone_approx_types = NON_QUANTILE_APPROX_OPTIONS
    else:
        zone_approx_types = st.multiselect(
            approx_types_step3_label,
            NON_QUANTILE_APPROX_OPTIONS,
            default=["Linear"],
            key="zone_approx_types",
        )

with st.sidebar.expander(approx_colors_expander, expanded=False):
    user_approx_colors = {}
    user_approx_dashes = {}
    user_approx_widths = {}

    for approx in NON_QUANTILE_APPROX_OPTIONS:
        default_color = DEFAULT_APPROX_COLORS.get(approx, "black")
        default_dash = DEFAULT_APPROX_DASHES.get(approx, "solid")

        st.markdown(f"**{approx}**")

        user_approx_colors[approx] = st.selectbox(
            "Color",
            COLOR_OPTIONS,
            index=COLOR_OPTIONS.index(default_color),
            key=f"approx_color_{approx}",
            label_visibility="collapsed",
        )
        user_approx_dashes[approx] = st.selectbox(
            hist_line_style_label,
            DASH_OPTIONS,
            index=DASH_OPTIONS.index(default_dash),
            key=f"approx_dash_{approx}",
            label_visibility="collapsed",
        )
        user_approx_widths[approx] = st.slider(
            approx_line_width_label,
            1, 10, 3,
            key=f"approx_width_{approx}",
            label_visibility="collapsed",
        )

    st.markdown("---")
    st.markdown(f"**{quantile_regression_title}**")

    quantile_colors = {}
    quantile_dashes = {}
    quantile_widths = {}

    for tau in [0.1, 0.5, 0.9]:
        default = DEFAULT_QUANTILE_COLORS[tau]

        st.markdown(f"**τ={tau}**")

        quantile_colors[tau] = st.selectbox(
            "Color",
            COLOR_OPTIONS,
            index=COLOR_OPTIONS.index(default),
            key=f"quantile_color_{tau}",
            label_visibility="collapsed",
        )
        quantile_dashes[tau] = st.selectbox(
            hist_line_style_label,
            DASH_OPTIONS,
            index=DASH_OPTIONS.index("solid"),
            key=f"quantile_dash_{tau}",
            label_visibility="collapsed",
        )
        quantile_widths[tau] = st.slider(
            approx_line_width_label,
            1, 10, 3,
            key=f"quantile_width_{tau}",
            label_visibility="collapsed",
        )

APPROX_COLORS  = user_approx_colors
APPROX_DASHES  = user_approx_dashes
APPROX_WIDTHS  = user_approx_widths       # для неквантильных кривых
QUANTILE_DASHES = quantile_dashes
QUANTILE_WIDTHS = quantile_widths         # для квантильных кривых (τ)

# ------------------------------------------------- #
# SIDEBAR: STEP 0 - корреляционный анализ
# ------------------------------------------------- #
st.sidebar.markdown("---")

with st.sidebar.expander(correlation_expander, expanded=False):

    if df is not None:
        numeric_columns = [
            c
            for c in df.select_dtypes(include=np.number).columns
            if c not in ["Zprm_well", "Zprm_map"]
        ]
    else:
        numeric_columns = []

    default_corr_cols = [
        c
        for c in [PARAM_1, PARAM_2]
        if c in numeric_columns
    ]

    all_parameters = st.checkbox(all_parameters_label, value=False, key="step0_all_parameters",)
    corr_columns = st.multiselect(corr_parameters_label, options=numeric_columns,
        default=(numeric_columns if all_parameters else default_corr_cols), key="step0_corr_columns",
    )

    if len(corr_columns) > 10:
        st.warning(corr_warning)

# ------------------------------------------------- #
# SIDEBAR: Предполагаемые выбросы в шагах 1 и 2
# ------------------------------------------------- #
st.sidebar.markdown("---")
with st.sidebar.expander(outlier_expander, expanded=False):
    # ------ рекомендуемый Порог выброса, σ ------
    sigma_thr = st.slider(sigma_threshold_label, 1.0, 6.0, 2.0, 0.1, key="sigma_thr",)

# ------------------------------------------------- #
# SIDEBAR: STEP 2 - параметры валидации CV 
# ------------------------------------------------- #
# Анализ расстояний между скважинами для рекомендаций по CV
def compute_spacing_stats(df):
    if df is None or len(df) < 2:
        return {
            "median_dist": None,
            "mean_dist": None,
            "min_dist": None,
            "max_dist": None,
            "n_wells": None,
        }

    # Уникальные скважины — по координатам, т.к. одна скважина = одна точка (X,Y)
    coords_df = df[["well_id", "X_coord", "Y_coord"]].drop_duplicates(subset=["well_id"])
    coords = coords_df[["X_coord", "Y_coord"]].to_numpy(float)
    dists = pdist(coords)

    return {
        "median_dist": np.median(dists) if len(dists) else None,
        "mean_dist": np.mean(dists) if len(dists) else None,
        "min_dist": np.min(dists) if len(dists) else None,
        "max_dist": np.max(dists) if len(dists) else None,
        "n_wells": coords_df["well_id"].nunique(),
    }

spacing = compute_spacing_stats(df)
median_dist = spacing["median_dist"]
mean_dist = spacing["mean_dist"]
min_dist = spacing["min_dist"]
max_dist = spacing["max_dist"]
n_wells = spacing["n_wells"]

st.sidebar.markdown("---")
with st.sidebar.expander(cv_validation_expander, expanded=False):

    # ------ рекомендуемое k ------
    st.markdown(f"#### {random_kfold_title}")
    if n_wells is not None:
        if n_wells < 20:
            recommended_k = 5
        elif n_wells < 50:
            recommended_k = 7
        elif n_wells < 150:
            recommended_k = 10
        else:
            recommended_k = 12
        st.markdown(
            f"**{recommended_k_text}: {recommended_k}** "
            f"({wells_count_text}: {n_wells})"
        )

        st.caption(kfold_caption)
    else:
        recommended_k = 5
    k_folds = st.slider(kfold_slider_label, 3, 20, recommended_k, key="k_folds")

    # ------ рекомендуемая сетка N×N ------
    st.markdown(f"#### {block_cv_title}")
    if median_dist is not None:
        if median_dist < 300:
            recommended_grid = 5
        elif median_dist < 600:
            recommended_grid = 4
        elif median_dist < 1000:
            recommended_grid = 3
        else:
            recommended_grid = 2

        st.markdown(
            f"**{recommended_grid_text}: "
            f"{recommended_grid}×{recommended_grid}** "
            f"({median_distance_text}: "
            f"{median_dist:.0f} м)"
        )
        st.caption(grid_caption)
    else:
        recommended_grid = 3
    grid_n = st.slider(grid_slider_label, 2, 10, recommended_grid, key="grid_n")

    # ------ рекомендуемый buffer ------
    if median_dist is not None:
        recommended_buffer = median_dist * 0.5

        st.markdown(
            f"**{recommended_buffer_text}: "
            f"{recommended_buffer:.0f} м** "
            f"({buffer_based_on_median})"
        )

        st.caption(buffer_caption)

        buffer = st.slider(
            buffer_slider_label,
            max(1, int(min_dist)),
            int(median_dist * 2),
            int(recommended_buffer),
            key="buffer",
        )

    else:
        buffer = 300

    # ------ Количество повторов CV repeats ------
    st.markdown(f"#### {leave_p_out_title}")
    if n_wells is not None:

        if n_wells < 30:
            recommended_repeats = 100
        elif n_wells < 100:
            recommended_repeats = 50
        elif n_wells < 300:
            recommended_repeats = 30
        else:
            recommended_repeats = 20
    else:
        recommended_repeats = 50

    st.markdown(
        f"**{recommended_repeats_text}: "
        f"{recommended_repeats}**"
    )

    cv_repeats = st.slider(cv_repeats_label, 5, 200, recommended_repeats, key="cv_repeats")

    st.caption(cv_repeats_caption)   
    # ------ Leave-P-Out ------
    leave_p_percent = st.slider(leave_p_percent_label, 1, 50, 10, key="leave_p_percent")
    st.caption(leave_p_percent_caption)
    
    # ------ Spatial Leave-P-Out ------
    st.markdown(f"#### {spatial_leave_p_out_title}")
    if median_dist is not None:
        recommended_radius = int(median_dist)
        st.markdown( 
            f"**{recommended_radius_text}: "
            f"{recommended_radius:.0f} м**"
        )
        st.caption(spatial_radius_caption)
        spatial_radius = st.slider(spatial_radius_label, int(max(1, min_dist)), max(int(max_dist), int(max(1, min_dist)) + 1), int(recommended_radius), key="spatial_radius")

    else:
        spatial_radius = 1000

    # ------ variogram factor ------
    st.markdown(f"#### {distance_cv_title}")
    variogram_factor = st.slider(variogram_factor_label, 0.25, 3.00, 1.00, 0.25, key="variogram_factor")

    st.caption(variogram_factor_caption)

# ------------------------------------------------------------- #
# SIDEBAR: STEP 3 — анализ по категориям
# ------------------------------------------------------------- #
# Настройка цветов категорий для Шага 3
def build_category_colors(categories, category_col, color_choices,):
    point_colors = {}
    line_colors = {}

    for i, category in enumerate(categories):
        st.markdown(
            f"**{category} "
            f"({(df[category_col] == category).sum()} "
            f"{category_wells_suffix})**"
        )
        point_colors[category] = st.selectbox(
            category_point_color_label,
            color_choices,
            index=i % len(color_choices),
            key=f"zone_point_color_{category_col}_{category}",
        )

        line_colors[category] = st.selectbox(
            category_line_color_label,
            color_choices,
            index=i % len(color_choices),
            key=f"zone_line_color_{category_col}_{category}",
        )

    return point_colors, line_colors

category_col = None

category_color_choices = ALL_COLORS
zone_point_colors = {}
zone_line_colors = {}

if df is not None:

    category_candidates = [
        c
        for c in df.columns
        if (
            c not in [
                "well_id",
                "X_coord",
                "Y_coord",
                "Zprm_well",
                "Zprm_map",
            ]
            and not pd.api.types.is_numeric_dtype(df[c])
        )
    ]

    default_category = st.session_state.get("zone_col")
    default_index = 0

    if (
        default_category is not None
        and default_category in category_candidates
    ):
        default_index = category_candidates.index(default_category)

    if len(category_candidates) > 0:
        st.sidebar.markdown("---")
        with st.sidebar.expander(category_expander,  expanded=False):
            category_col = st.selectbox(category_parameter_label, category_candidates, index=default_index, key="category_col",)
            # При смене категориального параметра — сбросить выбор зон в Шаге 3,
            # чтобы "Категории для общего графика" переинициализировались из zones
            if st.session_state.get("_last_category_col") != category_col:
                st.session_state.pop("step3_selected_zones", None)
                st.session_state["_last_category_col"] = category_col
            zone_point_size = st.slider(category_point_size_label, 3, 20, 7, key="zone_point_size",)
            zone_point_opacity = st.slider(category_point_opacity_label, 0.1, 1.0, 0.8, 0.05, key="zone_point_opacity",)
            categories = sorted(df[category_col].dropna().unique())
            zone_point_colors, zone_line_colors = (build_category_colors(categories, category_col, category_color_choices,))

# ------------------------------------------------------------- #
# SIDEBAR: STEP 4 — фильтрация скважин
# ------------------------------------------------------------- #

excluded_wells_manual = []
excluded_wells_random = []

percent_exclude = 0
n_excluded = 0   # используется в FILTER_RANDOM (Шаг 4)

if df is not None:
    st.sidebar.markdown("---")
    with st.sidebar.expander(well_filter_expander, expanded=False):

        FILTER_NONE = "none"
        FILTER_MANUAL = "manual"
        FILTER_RANDOM = "random"
        FILTER_CLICK = "click"
        filter_mode_labels = {
            FILTER_NONE: filter_mode_none,
            FILTER_MANUAL: filter_mode_manual,
            FILTER_RANDOM: filter_mode_random,
            FILTER_CLICK: filter_mode_click,
        }
        if "step4_filter_mode" not in st.session_state:
            st.session_state.step4_filter_mode = FILTER_NONE

        filter_mode = st.radio(
            filter_mode_label,
            [FILTER_NONE, FILTER_MANUAL, FILTER_RANDOM, FILTER_CLICK],
            format_func=lambda x: filter_mode_labels[x],
            index=[FILTER_NONE, FILTER_MANUAL, FILTER_RANDOM, FILTER_CLICK,].index(st.session_state.step4_filter_mode), key="step4_filter_mode",)
        st.caption(
            f"{current_filter_mode} "
            f"{filter_mode_labels[filter_mode]}"
        )

        # Ручное исключение
        if filter_mode == FILTER_MANUAL:
            excluded_wells_manual = st.multiselect(
                exclude_wells_label,
                options=sorted(df["well_id"].astype(str).unique()),
                default=[], key="step4_excluded_wells",
            )

            st.markdown(
                f"**{excluded_wells_count} "
                f"{len(excluded_wells_manual)}**"
            )

        # Процентное исключение
        elif filter_mode == FILTER_RANDOM:
            percent_exclude = st.slider(percent_exclude_label, 0, 95, 10,key="step4_percent_exclude",)
            if "step4_random_seed" not in st.session_state:
                st.session_state.step4_random_seed = np.random.randint(0, 10_000_000)
            if st.button(new_random_set_button, key="step4_new_random_set"):
                st.session_state.step4_random_seed = np.random.randint(0, 10_000_000)

            st.caption(random_exclusion_caption)

            if percent_exclude > 0:
                n_excluded = max(1, int(len(df) * percent_exclude / 100))
                n_excluded = min(n_excluded, len(df) - 1)
            else:
                n_excluded = 0

            st.caption(
                f"{current_random_set} "
                f"{st.session_state.step4_random_seed}"
            )
            st.markdown(
                f"**{will_be_excluded} "
                f"{n_excluded} "
                f"{wells_suffix}**" 
            )

        # Выделение кликом по точке
        elif filter_mode == FILTER_CLICK:
            st.caption(step4_click_hint)

            if "step4_click_excluded" not in st.session_state:
                st.session_state.step4_click_excluded = []

            st.markdown(
                f"**{step4_click_excluded_count}** "
                f"{len(st.session_state.step4_click_excluded)}"
            )

            if st.button(step4_click_reset_button, key="step4_click_reset_sidebar"):
                st.session_state.step4_click_excluded = []
                st.rerun()

# --------------
# FUNC: LABELS
# ---------------
wells_legend = T["wells_legend"]
well_label = T["well_label"]
regression_qc_label = T["regression_qc_label"]
regression_cv_label = T["regression_cv_label"]
well_count_label = T["well_count_label"]
error_column_label = T["error_column_label"]
no_outliers_label = T["no_outliers_label"]

# ================================================================================================================================ #
# -----------------------------------------------   ЗАГРУЗКА ДАННЫХ   --------------------------------------------------------------
# ================================================================================================================================ #

# --------------
# STEP 0 LABELS
# ---------------
step0_title = T["step0_title"]
step0_load_data_message = T["step0_load_data_message"]
step0_loaded_data_title = T["step0_loaded_data_title"]
step0_input_qc_title = T["step0_input_qc_title"]
step0_full_duplicates_title = T["step0_full_duplicates_title"]
step0_full_duplicates_notfound = T["step0_full_duplicates_notfound"]
step0_found_full_duplicates = T["step0_found_full_duplicates"]
step0_duplicate_wells_title = T["step0_duplicate_wells_title"]
step0_duplicate_wells_found = T["step0_duplicate_wells_found"]
step0_duplicate_wells_notfound = T["step0_duplicate_wells_notfound"]
step0_duplicate_coords_title = T["step0_duplicate_coords_title"]
step0_duplicate_coords_found = T["step0_duplicate_coords_found"]
step0_duplicate_coords_notfound = T["step0_duplicate_coords_notfound"]
step0_missing_values_title = T["step0_missing_values_title"]
step0_removed_rows_title = T["step0_removed_rows_title"]
step0_removed_rows_count = T["step0_removed_rows_count"]
step0_removed_rows_show = T["step0_removed_rows_show"]
step0_removed_rows_deleted = T["step0_removed_rows_deleted"]
step0_notremoved_rows = T["step0_notremoved_rows"]
step0_well_count_title = T["step0_well_count_title"]
step0_columns_title = T["step0_columns_title"]
step0_data_types_title = T["step0_data_types_title"]
step0_categorical_features_title = T["step0_categorical_features_title"]
step0_first_rows_title = T["step0_first_rows_title"]
step0_last_rows_title = T["step0_last_rows_title"]
step0_parameter_stats_title = T["step0_parameter_stats_title"]
step0_selected_params_title = T["step0_selected_params_title"]
step0_category_values_preview = T["step0_category_values_preview"]
step0_corr_matrix_title = T["step0_corr_matrix_title"]
step0_corr_matrix_plot_title = T["step0_corr_matrix_plot_title"]
step0_plotly_pairplot_title = T["step0_plotly_pairplot_title"]
step0_pairplot_title = T["step0_pairplot_title"]
step0_pairplot_warning = T["step0_pairplot_warning"]
step0_plotly_pairplot_info = T["step0_plotly_pairplot_info"]
step0_selected_stats_title = T["step0_selected_stats_title"]
step0_stats_prefix = T["step0_stats_prefix"]
step0_corr_spearman_title = T["step0_corr_spearman_title"]
step0_corr_hint = T["step0_corr_hint"]
step0_rows_count_title = T["step0_rows_count_title"]
step0_unique_wells_title = T["step0_unique_wells_title"]
step0_duplicate_wells_hint = T["step0_duplicate_wells_hint"]

st.markdown('<div id="step-0"></div>', unsafe_allow_html=True)
# ------------------------------------------------- #
# STEP 0.1.: SHOW SAMPLE FORMAT
# ------------------------------------------------- #
st.title(step0_title)
ui_examples.show_project_introduction()
ui_examples.show_input_file_example()

# ------------------------------------------------- #
# STEP 0.2.: QC входных данных
# ------------------------------------------------- #
if df is None:
    st.info(step0_load_data_message)
    st.stop()

# --- Основные сведения о данных ---
n_rows = len(df)
cols_list = df.columns.tolist()

cols_for_display = [
    c
    for c in cols_list
    if c not in ["Zprm_well", "Zprm_map"]
]

required_cols = ["well_id", "X_coord", "Y_coord", "Zprm_well", "Zprm_map"]

# --- Цветовые бейджи колонок ---
def badge(col):
    display_col = display_name(col)
    if col in required_cols:
        return f"🟩 {display_col}"
    else:
        return f"🟦 {display_col}"

badges = " · ".join([badge(c) for c in cols_for_display])

# --- Сводная информация о загруженных данных ---
st.markdown(f"## {step0_loaded_data_title}")
# QC входных данных

with st.expander(f"✅ {step0_input_qc_title}",  expanded=False):

    # Полные дубликаты строк
    full_duplicates = df[df.duplicated(keep=False)]
    st.markdown(f"### {step0_full_duplicates_title}")
    if len(full_duplicates) > 0:
        st.warning(
            f"{step0_found_full_duplicates} "
            f"{len(full_duplicates)}"
        ) 
        st.dataframe(full_duplicates, use_container_width=True)
    else:
        st.caption(step0_full_duplicates_notfound)

    # Дубликаты скважин
    st.markdown(f"### {step0_duplicate_wells_title}")
    st.caption(step0_duplicate_wells_hint)
    dup_wells = (df.groupby("well_id").size().reset_index(name="count"))
    dup_wells = dup_wells[dup_wells["count"] > 1]
    if len(dup_wells) > 0:
        st.warning(f"{step0_duplicate_wells_found}"
            f"{len(dup_wells)}"
        )
        st.dataframe(dup_wells, use_container_width=True)
    else:
        st.caption(step0_duplicate_wells_notfound)

    # Совпадающие координаты (в пределах одного горизонта)
    st.markdown(f"### {step0_duplicate_coords_title}")

    coords_group_cols = ["X_coord", "Y_coord"]
    if "horizon_id" in df.columns:
        coords_group_cols.append("horizon_id")

    dup_coords = (
        df.groupby(coords_group_cols)
          .size()
          .reset_index(name="count")
    )
    dup_coords = dup_coords[dup_coords["count"] > 1]

    if len(dup_coords) > 0:
        st.warning(f"{step0_duplicate_coords_found}"
            f"{len(dup_coords)}"
        )
        st.dataframe(dup_coords, use_container_width=True)
    else:
        st.caption(step0_duplicate_coords_notfound)

    # Пропуски
    st.markdown(f"### {step0_missing_values_title}")
    na_summary = pd.DataFrame({
        "column": [display_name(c) for c in cols_for_display],
        "missing values": [int(df[c].isna().sum()) for c in cols_for_display],
    })
    st.dataframe(na_summary, use_container_width=True)

    # Строки удалённые при очистке
    st.markdown(f"### {step0_removed_rows_title}")
    removed_rows = st.session_state.get("removed_rows", pd.DataFrame() )
    if len(removed_rows) > 0:
        st.warning(
            f"{step0_removed_rows_count}"
            f"{len(removed_rows)}"
        )
        removed_display = removed_rows.drop(columns=["Zprm_well", "Zprm_map"], errors="ignore")
        st.dataframe(removed_display.head(20), use_container_width=True)
        if len(removed_display) > 20:
            st.caption(
                f"{step0_removed_rows_show} "
                f"{len(removed_display)} "
                f"{step0_removed_rows_deleted}"
            )
    else:
        st.caption(step0_notremoved_rows)

n_unique_wells = df["well_id"].nunique()

st.write(
    f"**{step0_rows_count_title}:** "
    f"{n_rows}"
)

st.write(
    f"**{step0_unique_wells_title}:** "
    f"{n_unique_wells}"
)

st.write(
    f"**{step0_columns_title}:** "
    f"{badges}"
)

# ------------------------------------------------- #
# STEP 0.3.: Типы данных по колонкам
# ------------------------------------------------- #
st.markdown(f"### {step0_data_types_title}")

dtype_rows = []

for col in cols_for_display:

    dtype = df[col].dtype

    if pd.api.types.is_numeric_dtype(df[col]):
        data_type = "Numeric"

    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        data_type = "Date & Time"

    elif pd.api.types.is_bool_dtype(df[col]):
        data_type = "Boolean"

    elif (
        pd.api.types.is_string_dtype(df[col])
        or dtype == "object"
    ):
        data_type = "String"

    else:
        data_type = "Other"

    dtype_rows.append(
        {
            "Column": display_name(col),
            "Data Types": data_type,
            "Pandas dtype": str(dtype),
            "Not NaN": int(df[col].notna().sum()),
            "NaN": int(df[col].isna().sum()),
            "Unique Values": int(df[col].nunique(dropna=True)),
        }
    )

dtype_df = pd.DataFrame(dtype_rows)

st.dataframe(dtype_df, use_container_width=True)

st.markdown(f"### {step0_categorical_features_title}")

category_cols = [
    c
    for c in cols_for_display
    if (
        not pd.api.types.is_numeric_dtype(df[c])
        and c != "well_id"
    )
]

for col in category_cols:
    unique_vals = sorted(map(str, df[col].dropna().unique()))
    max_show = 20
    show_vals = unique_vals[:max_show]

    with st.expander(
        f"{display_name(col)} ({len(unique_vals)})",
        expanded=False,
    ):
        # Список значений (как было)
        st.write(", ".join(show_vals))
        if len(unique_vals) > max_show:
            st.caption(
                f"{step0_category_values_preview}: "
                f"{max_show} : {len(unique_vals)}"
            )

        # Пирог + бар по этой категории
        col_pie, col_bar = st.columns(2)

        with col_pie:
            fig_pie = plot_zone_pie(
                df,
                category_col=col,
                palette=pie_palette,
                title=f"{step0_well_count_title} : {display_name(col)}",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            zone_counts_local = df[col].value_counts().sort_index()
            fig_bar = plot_bar_chart(
                x=zone_counts_local.index,
                y=zone_counts_local.values,
                palette=cat_palette,
                orientation="h",
                title=f"{step0_well_count_title} : {display_name(col)}",
                yaxis_title=well_count_label,
                show_percent=True,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# --- отображаем пользователю исходные названия параметров ---
display_df = df.copy()
display_df = display_df.drop(columns=["Zprm_well", "Zprm_map"], errors="ignore")
front_cols = ["well_id", "X_coord", "Y_coord"]

other_cols = [
    c
    for c in display_df.columns
    if c not in front_cols
]

display_df = display_df[front_cols + other_cols]

# --- Просмотр данных ---
st.markdown(f"### {step0_first_rows_title}")
st.dataframe(display_df.head(5), use_container_width=True)

st.markdown(f"### {step0_last_rows_title}")
st.dataframe(display_df.tail(5), use_container_width=True)

# --- Статистика параметров
st.markdown(
    f"### {step0_parameter_stats_title} "
    f"{PARAM_1}, {PARAM_2}"
)
col_desc1, col_desc2 = st.columns(2)
with col_desc1:
    show_depth_stats(df["Zprm_well"], PARAM_1_LABEL)
with col_desc2:
    show_depth_stats(df["Zprm_map"], PARAM_2_LABEL)

# ------------------------------------------------- #
# STEP 0.4.: Корреляционный анализ выбранных параметров
# ------------------------------------------------- #
if (df is not None and len(corr_columns) >= 2):
    st.markdown(f"## {step0_selected_params_title}")

    if len(corr_columns) > 6:
        st.warning(step0_pairplot_warning)

    # Матрица корреляции
    st.markdown(f"### {step0_corr_matrix_title}")

    # специальный случай: выбран один и тот же параметр
    if len(set(corr_columns)) == 1:
        col = corr_columns[0]
        corr_df = pd.DataFrame([[1.0]], index=[col], columns=[col],)
    else:
        corr_df = df[corr_columns].corr()

    corr_df = corr_df.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)
    fig_corr = px.imshow(
    corr_df, text_auto=".2f", color_continuous_scale=corr_palette, zmin=-1, zmax=1, aspect="auto",
    title=f"{step0_corr_matrix_plot_title} "
      f"{len(corr_columns)}"
    )
    fig_corr.update_coloraxes(colorbar_title="Correlation<br>coefficient") 
    fig_corr.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10),)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Пояснение: чем Пирсон отличается от Спирмена
    st.caption(step0_corr_hint)

    # Матрица корреляции Спирмена
    st.markdown(f"### {step0_corr_spearman_title}")
    if len(set(corr_columns)) == 1:
        col = corr_columns[0]
        corr_df_spearman = pd.DataFrame([[1.0]], index=[col], columns=[col],)
    else:
        corr_df_spearman = df[corr_columns].corr(method="spearman")
    corr_df_spearman = corr_df_spearman.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)
    fig_corr_sp = px.imshow(
        corr_df_spearman,
        text_auto=".2f",
        color_continuous_scale=corr_palette,
        zmin=-1, zmax=1,
        aspect="auto",
        title=f"{step0_corr_spearman_title}: {len(corr_columns)}"
    )
    fig_corr_sp.update_coloraxes(colorbar_title="Spearman<br>ρ")
    fig_corr_sp.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10),)
    st.plotly_chart(fig_corr_sp, use_container_width=True)

    # Pairplot и Seaborn Pairplot
    st.markdown(f"### {step0_pairplot_title}")
    if pairplot_engine == "Seaborn Pairplot (static)":
        # Pairplot - корреляционный анализ
        fig_pair = plot_pairplot(
            df=df, columns=corr_columns, diag_kind=diag_kind, height=pairplot_size, corner=pairplot_corner, 
            point_color=pairplot_point_color, point_size=pairplot_point_size, point_opacity=pairplot_point_opacity,
            hist_color=pairplot_hist_color, kde_color=pairplot_kde_color, scale=pairplot_scale,
            category_col=pairplot_category,
            category_colors=pairplot_category_colors if pairplot_category_colors else None,
        )
        st.pyplot(fig_pair, use_container_width=True)
    
    elif pairplot_engine == "Plotly Pairplot (interactive)":
        st.caption(step0_plotly_pairplot_info) 

        extra = [c for c in corr_columns if c != "well_id"]

        # Если задана категория — добавляем её в выборку и используем как color
        cols_for_pair = ["well_id"] + extra
        use_category_for_color = (
            pairplot_category is not None
            and pairplot_category in df.columns
            and pairplot_category not in cols_for_pair
        )

        if use_category_for_color:
            cols_for_pair = cols_for_pair + [pairplot_category]

        # Копируем, чтобы не менять исходный df
        pair_source = df[cols_for_pair].copy()
        if use_category_for_color:
            # NaN в категории заменяем на строку, чтобы не терять строки при dropna
            pair_source[pairplot_category] = (
                pair_source[pairplot_category].fillna("(not specified)").astype(str)
            )

        pair_df = pair_source.dropna()

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

        pair_df = pair_df.rename(columns=DISPLAY_NAMES)

        # Для dimensions исключаем well_id и (если используется) категорию
        excluded_from_dims = {"well_id"}
        if use_category_for_color:
            # после rename имя категории могло измениться — берём переименованное
            cat_display = DISPLAY_NAMES.get(pairplot_category, pairplot_category)
            excluded_from_dims.add(cat_display)

        dimensions = [c for c in pair_df.columns if c not in excluded_from_dims]

        if use_category_for_color:
            cat_display = DISPLAY_NAMES.get(pairplot_category, pairplot_category)

            # Готовим color_discrete_map из словаря цветов сайдбара
            color_map = None
            if pairplot_category_colors:
                color_map = {str(k): v for k, v in pairplot_category_colors.items()}

            fig_pair = px.scatter_matrix(
                pair_df,
                dimensions=dimensions,
                color=cat_display,
                hover_name="well_id",
                color_discrete_map=color_map,
                title=(
                    f"{step0_plotly_pairplot_title} "
                    f"({len(corr_columns)})"
                ),
            )

        else:
            fig_pair = px.scatter_matrix(
                pair_df,
                dimensions=dimensions,
                hover_name="well_id",
                title=(
                    f"{step0_plotly_pairplot_title} "
                    f"({len(corr_columns)})"
                ),
            )

        # Если категория НЕ используется — красим точки в один цвет, как раньше
        if not use_category_for_color:
            fig_pair.update_traces(
                marker=dict(
                    color=pairplot_point_color,
                    size=pairplot_point_size,
                    opacity=pairplot_point_opacity,
                )
            )
        else:
            fig_pair.update_traces(
                marker=dict(
                    size=pairplot_point_size,
                    opacity=pairplot_point_opacity,
                )
            )

        fig_pair.update_traces(diagonal_visible=False)
        fig_pair.update_layout(height=850)

        st.plotly_chart(fig_pair, use_container_width=True)

    # Статистика выбранных параметров
    st.markdown(f"### {step0_selected_stats_title}")
    for col in corr_columns:
        display_col = display_name(col)
        with st.expander(f"{step0_stats_prefix}: {display_col}", expanded=False):
            show_depth_stats(df[col], display_col)

# ================================================================================================================================ #
# ---------------------------------------   ШАГ 1: QC‑кросс‑плот + аппроксимации + таблицы   ---------------------------------------
# ================================================================================================================================ #
# --------------
# STEP 1 LABELS
# ---------------
step1_title = T["step1_title"]
step1_not_enough_data = T["step1_not_enough_data"]
step1_crossplot_title = T["step1_crossplot_title"]
step1_qc_metrics_title = T["step1_qc_metrics_title"]
step1_approximations_title = T["step1_approximations_title"]
step1_approx_comparison_title = T["step1_approx_comparison_title"]
step1_crossplot_build_error = T["step1_crossplot_build_error"]
step1_distribution_title = T["step1_distribution_title"]
residual_distribution_title = T["residual_distribution_title"]
residual_vs_param_title = T["residual_vs_param_title"]
residual_stats_title = T["residual_stats_title"]
spatial_residuals_title = T["spatial_residuals_title"]
residual_map_title = T["residual_map_title"]
outlier_analysis_title = T["outlier_analysis_title"]
hist_bins_label = T["hist_bins_label"]
hist_norm_label = T["hist_norm_label"]
hist_title_t = T["hist_title"]
boxplot_title_t = T["boxplot_title"]
hist_norm_map = {
    "none": T["hist_norm_none"],
    "density": T["hist_norm_density"],
    "probability": T["hist_norm_probability"],
    "percent": T["hist_norm_percent"],
}
step1_data_hint = T["step1_data_hint"]

st.markdown('<div id="step-1"></div>', unsafe_allow_html=True)
# ---------------------------------------------------------------------- #
# STEP 1.0
# ---------------------------------------------------------------------- #
st.markdown(
    f"## {step1_title} "
    f"{PARAM_1} vs {PARAM_2}"
)

if len(df) < 2:
    st.warning(step1_not_enough_data)
    st.info(step1_data_hint)
    st.stop()

# Basic QC cross‑plot step 1
basic = sq.basic_crossplot_stats(df)
# Δ = PARAM_1 − PARAM_2
resid = basic.residuals

z_m = df["Zprm_map"].to_numpy(float)
z_w = df["Zprm_well"].to_numpy(float)
x_ref = z_m      # PARAM_2
y_ref = z_w      # PARAM_1
if len(y_ref) < 2:
    st.warning(step1_crossplot_build_error)
    st.stop()
z_line = np.linspace(x_ref.min(), x_ref.max(), 300)

# ---------------------------------------------------------------------- #
# STEP 1.1. LAYOUT: LEFT (QC CROSSPLOT + METRICS) / RIGHT (APPROX CURVES + METRICS)
# ---------------------------------------------------------------------- #
st.markdown('<div id="step-1-cp"></div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1])

# --- QC CROSSPLOT + METRICS
with col_left:
    st.markdown(
        f"### {step1_crossplot_title}: "
        f"{PARAM_1} vs {PARAM_2}"
    )
    fig_qc, coeffs_qc = build_qc_crossplot(
        df=df, x_ref=x_ref, y_ref=y_ref,
        title=f"QC cross-plot: {PARAM_1} vs {PARAM_2}",
        x_label=PARAM_2_LABEL,
        y_label=PARAM_1_LABEL,
        point_color=crossplot_point_color,
        regression_color=qc_regression_color,
        identity_color=identity_line_color,
        identity_width=identity_line_width,
        identity_dash=identity_line_dash,
        wells_label=wells_legend,
        well_label=well_label,
        regression_name=regression_qc_label,
        regression_width=qc_regression_width,
        regression_dash=qc_regression_dash,
        marker_size=crossplot_point_size,
        marker_opacity=crossplot_point_opacity,
    )
    st.plotly_chart(fig_qc, use_container_width=True)

# --- QC METRICS TABLE
qc_metrics_df = build_qc_metrics_table(basic=basic, coeffs_qc=coeffs_qc, left_name=PARAM_1, right_name=PARAM_2,)
st.markdown(
    f"### {step1_qc_metrics_title}: "
    f"{PARAM_1} vs {PARAM_2}"
)
st.dataframe(qc_metrics_df, use_container_width=True)

# --- APPROX CURVES + METRICS
with col_right:
    st.markdown(
        f"### {step1_approximations_title}: "
        f"{PARAM_1} vs {PARAM_2}"
    )
    fig_fit, fit_rows = build_approximation_plot(
        df=df, x_ref=x_ref, y_ref=y_ref,
        z_line=z_line, approx_types=approx_types,
        title=f"QC cross-plot approx: {PARAM_1} vs {PARAM_2}",
        x_label=PARAM_2_LABEL,
        y_label=PARAM_1_LABEL,
        x_name=PARAM_2, y_name=PARAM_1,
        point_color=approx_point_color,
        identity_color=identity_line_color,
        wells_label=wells_legend,
        well_label=well_label,
        error_label=error_column_label,
        approx_colors=APPROX_COLORS,
        quantile_colors=quantile_colors,
        approx_dashes=APPROX_DASHES,          
        quantile_dashes=QUANTILE_DASHES,    
        approx_widths=APPROX_WIDTHS,        
        quantile_widths=QUANTILE_WIDTHS,    
        identity_width=identity_line_width,
        identity_dash=identity_line_dash,
        marker_size=approx_point_size,
        marker_opacity=approx_point_opacity,
    )
    st.plotly_chart(fig_fit, use_container_width=True)

# --- TABLE: COMPARISON OF ALL APPROXIMATIONS
st.markdown(
    f"### {step1_approx_comparison_title}: "
    f"{PARAM_1} vs {PARAM_2}"
)
st.dataframe(pd.DataFrame(fit_rows), use_container_width=True,)

# -------------------------------------------------------------------- #
# STEP 1.2.: Histogram и Boxplot выбранных параметров
# -------------------------------------------------------------------- #
st.markdown('<div id="step-1-hb"></div>', unsafe_allow_html=True)

st.markdown(
    f"### {step1_distribution_title} "
    f"{PARAM_1} , {PARAM_2}"
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    bins_step1 = st.slider(f"{hist_bins_label} {PARAM_1}, {PARAM_2}", 10, 120, 40, key="bins_step1")
with col_ctrl2:
    histnorm_step1 = st.selectbox(
        f"{hist_norm_label} {PARAM_1}, {PARAM_2}",
        ["none", "density", "probability", "percent"],
        format_func=lambda x: hist_norm_map[x],
        index=0, key="histnorm_step1"
    )

histnorm_value = "" if histnorm_step1 == "none" else histnorm_step1

# Histogram в шаге 1
col_h1, col_h2 = st.columns(2)
with col_h1:
    st.markdown(f"#### {hist_title_t} {PARAM_1}")
    fig_hw = plot_hist_kde(
        data=y_ref,
        bins=bins_step1,
        histnorm=histnorm_value,
        color=hist_color,
        title=f"{hist_title_t} {PARAM_1}",
        xlabel=PARAM_1_LABEL,
        show_sigma=True,
        kde_color=kde_color,       kde_dash=kde_dash,       kde_width=kde_width,
        mean_color=mean_color,     mean_dash=mean_dash,     mean_width=mean_width,
        median_color=median_color, median_dash=median_dash, median_width=median_width,
        sigma_color=sigma_color,   sigma_dash=sigma_dash,   sigma_width=sigma_width,
    )
    st.plotly_chart(fig_hw, use_container_width=True)
with col_h2:
    st.markdown(f"#### {hist_title_t} {PARAM_2}")
    fig_hm = plot_hist_kde(
        data=x_ref,
        bins=bins_step1,
        histnorm=histnorm_value,
        color=hist_color,
        title=f"{hist_title_t} {PARAM_2}",
        xlabel=PARAM_2_LABEL,
        show_sigma=True,
        kde_color=kde_color,       kde_dash=kde_dash,       kde_width=kde_width,
        mean_color=mean_color,     mean_dash=mean_dash,     mean_width=mean_width,
        median_color=median_color, median_dash=median_dash, median_width=median_width,
        sigma_color=sigma_color,   sigma_dash=sigma_dash,   sigma_width=sigma_width,
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# Boxplot в шаге 1
col_b1, col_b2 = st.columns(2)
with col_b1:
    st.markdown(f"#### {boxplot_title_t} {PARAM_1}")
    fig_bw = plot_box(
        y_ref, box_color, PARAM_1_LABEL, well_label,
        title=f"{boxplot_title_t} {PARAM_1}",
        well_ids=df["well_id"], value_label=PARAM_1_LABEL,
        mean_color=box_mean_color,     mean_dash=box_mean_dash,     mean_width=box_mean_width,
        median_color=box_median_color, median_dash=box_median_dash, median_width=box_median_width,
    )
    st.plotly_chart(fig_bw, use_container_width=True)
with col_b2:
    st.markdown(f"#### {boxplot_title_t} {PARAM_2}")
    fig_bm = plot_box(
        x_ref, box_color, PARAM_2_LABEL, well_label,
        title=f"{boxplot_title_t} {PARAM_2}",
        well_ids=df["well_id"], value_label=PARAM_2_LABEL,
        mean_color=box_mean_color,     mean_dash=box_mean_dash,     mean_width=box_mean_width,
        median_color=box_median_color, median_dash=box_median_dash, median_width=box_median_width,
    )
    st.plotly_chart(fig_bm, use_container_width=True)

# -------------------------------------------------------------------- #
# STEP 1.3.: Анализ невязок Residuals vs PARAM_1 + statistics
# -------------------------------------------------------------------- #
st.markdown('<div id="step-1-res"></div>', unsafe_allow_html=True)
st.markdown(
    f"### {residual_distribution_title} "
    f"{DELTA_LABEL}"
)
st.markdown(
    f"#### {residual_vs_param_title} "
    f"{DELTA_LABEL} "
    f" vs  {PARAM_1}"
)

col_resid_plot, col_resid_stats = st.columns(2)

# График невязок (Шаг 1)
with col_resid_plot:
    fig_resid = plot_residual_vs_depth(
        z_values=y_ref,
        residuals=resid,
        well_ids=df["well_id"],
        color=residual_color,
        wells_label=wells_legend,
        well_label=well_label,
        point_size=residual_point_size,
        point_opacity=residual_point_opacity,
        xlabel=PARAM_1_LABEL,
        ylabel=DELTA_LABEL,
        hover_label=DELTA_LABEL,
        title=f"{DELTA_LABEL} vs {PARAM_1}"
    )
    st.plotly_chart(fig_resid, use_container_width=True)

with col_resid_stats:
    st.markdown(
        f"#### {residual_stats_title} "
        f"{DELTA_LABEL}"
    )
    # Статистика невязок (Шаг 1)
    st.dataframe(compute_stats(resid), use_container_width=True)

# HISTOGRAM, KDE и Boxplot невязок (Шаг 1)
col_ctrl_r1, col_ctrl_r2 = st.columns(2)
with col_ctrl_r1:
    bins_resid = st.slider(f"{hist_bins_label} {DELTA_LABEL}", 10, 120, 40, key="bins_resid")
with col_ctrl_r2:
    histnorm_resid = st.selectbox(
        f"{hist_norm_label} {DELTA_LABEL}",
        ["none", "density", "probability", "percent"],
        format_func=lambda x: hist_norm_map[x],
        index=0, key="histnorm_resid"
    )

histnorm_resid_value = "" if histnorm_resid == "none" else histnorm_resid

# Histogram невязок (Шаг 1)
col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown(f"#### {hist_title_t} {DELTA_LABEL}")
    fig_hr = plot_hist_kde(
        data=resid,
        bins=bins_resid,
        histnorm=histnorm_resid_value,
        color=hist_color,
        title=f"{hist_title_t} {DELTA_LABEL}",
        xlabel=DELTA_LABEL,
        show_sigma=True,
        kde_color=kde_color,       kde_dash=kde_dash,       kde_width=kde_width,
        mean_color=mean_color,     mean_dash=mean_dash,     mean_width=mean_width,
        median_color=median_color, median_dash=median_dash, median_width=median_width,
        sigma_color=sigma_color,   sigma_dash=sigma_dash,   sigma_width=sigma_width,
    )
    st.plotly_chart(fig_hr, use_container_width=True)

# Boxplot невязок (Шаг 1)
with col_r2:
    st.markdown(f"#### {boxplot_title_t} {DELTA_LABEL}")
    fig_br = plot_box(
        resid, box_color, DELTA_LABEL, well_label,
        title=f"{boxplot_title_t} {DELTA_LABEL}",
        well_ids=df["well_id"], value_label=DELTA_LABEL,
        mean_color=box_mean_color,     mean_dash=box_mean_dash,     mean_width=box_mean_width,
        median_color=box_median_color, median_dash=box_median_dash, median_width=box_median_width,
    )
    st.plotly_chart(fig_br, use_container_width=True)

# Карта невязок (Шаг 1)
st.markdown(
    f"### {spatial_residuals_title} "
    f"{DELTA_LABEL}"
)


fig_map = plot_residual_map(
    df=df,
    residuals=resid,
    palette=map_palette,
    wells_label=wells_legend,
    well_label=well_label,
    residual_label=DELTA_LABEL,
    point_size=residual_map_point_size,
    point_opacity=residual_map_point_opacity,
    title=f"{residual_map_title} {DELTA_LABEL}",
)
st.plotly_chart(fig_map, use_container_width=True)

# ------ Анализ выбросов по результатам невязок (Шаг 1)
st.markdown('<div id="step-1-otl"></div>', unsafe_allow_html=True)
st.markdown(
    f"### {outlier_analysis_title} "
    f"{DELTA_LABEL}"
)
show_outlier_tables(
    df=df,
    residuals=resid,
    sigma_thr=sigma_thr,
    param_1=PARAM_1,
    param_2=PARAM_2,
    residual_column_name=DELTA_LABEL,
    no_outliers_label=no_outliers_label,
)

# ================================================================================================================================ #
# ---------------------------------------------   Шаг 2: Кросс-валидация и таблицы метрик ------------------------------------------
# ================================================================================================================================ #
# --------------
# STEP 2 LABELS
# ---------------
step2_title = T["step2_title"]
cv_not_enough_data_msg = T["cv_not_enough_data_msg"]
cv_best_scheme_error_msg = T["cv_best_scheme_error_msg"]
step2_crossplot_all_title = T["step2_crossplot_all_title"]
step2_crossplot_all_caption = T["step2_crossplot_all_caption"]
step2_metrics_all_title = T["step2_metrics_all_title"]
step2_cv_comparison_title = T["step2_cv_comparison_title"]
step2_rmse_title = T["step2_rmse_title"]
step2_best_scheme_title = T["step2_best_scheme_title"]
step2_cv_crossplot_title = T["step2_cv_crossplot_title"]
step2_approximations_title = T["step2_approximations_title"]
step2_cv_metrics_title = T["step2_cv_metrics_title"]
step2_approx_comparison_title = T["step2_approx_comparison_title"]
step2_residual_distribution_title = T["step2_residual_distribution_title"]
step2_residuals_title = T["step2_residuals_title"]
step2_residual_stats_title = T["step2_residual_stats_title"]
step2_outlier_analysis_title = T["step2_outlier_analysis_title"]
cv_residual_col_label = T["cv_residual_col_label"]


st.markdown('<div id="step-2"></div>', unsafe_allow_html=True)
st.markdown('<div id="step-2-bcv"></div>', unsafe_allow_html=True)
# ------------------------------------------------- #
# STEP 2.0.: 
# ------------------------------------------------- #
@st.cache_data(
    show_spinner="Running cross-validation...",
    hash_funcs={pd.DataFrame: lambda df: (len(df), tuple(df.columns))},
)
def cached_run_all_cv(df, degree, k, grid_n, buffer, variogram_factor,
                       leave_p_percent, spatial_radius, cv_repeats):
    return sq.run_all_cv(
        df, degree=degree, k=k, grid_n=grid_n, buffer=buffer,
        variogram_factor=variogram_factor, leave_p_percent=leave_p_percent,
        spatial_radius=spatial_radius, cv_repeats=cv_repeats,
    )

st.markdown(
    f"## {step2_title}: "
    f"{PARAM_1} vs {PARAM_2}"
)

if st.session_state.get("df_for_cv") is None:
    st.info(cv_not_enough_data_msg)
    st.stop()

results, comparison = cached_run_all_cv(
    st.session_state["df_for_cv"], 1, k_folds, grid_n, buffer,
    variogram_factor, leave_p_percent, spatial_radius, cv_repeats,
)

if comparison.empty:
    st.warning(cv_not_enough_data_msg)
    st.stop()

# --- Удаляем схему Stratified by Zone при отсутствии tectonic_zone ---
if "Stratified by Zone" in comparison.index and "tectonic_zone" not in df.columns:
    comparison = comparison.drop(index="Stratified by Zone")
    results.pop("Stratified by Zone", None)

# --- Выбор лучшей схемы CV по двум критериям ---
comparison["score"] = (comparison["rmse"] + comparison["rmse_fold_std"])
valid_scores = comparison["score"].dropna()
if len(valid_scores) == 0:
    st.warning(cv_best_scheme_error_msg)
    st.stop()

best_scheme = valid_scores.idxmin()
best_preds = results[best_scheme]["preds"]

# "Замороженный" датасет, на котором обучалась CV — используется далее во всём Шаге 2
df_cv = st.session_state["df_for_cv"]

# ------------------------------------------------- #
# STEP 2.1.: Сross-plot Predicted vs Actual для схем CV
# ------------------------------------------------- #
st.markdown(f"### {step2_crossplot_all_title}")
st.caption(step2_crossplot_all_caption)
cv_summary_rows = []

scheme_names = list(results.keys())

for i in range(0, len(scheme_names), 3):
    cols = st.columns(3)
    for j, scheme_name in enumerate(scheme_names[i:i+3]):
        with cols[j]:

            scheme_result = results[scheme_name]
            preds_tmp = scheme_result["preds"]
            valid_tmp = np.isfinite(preds_tmp)
            if valid_tmp.sum() < 3:
                continue

            z_w_tmp = df_cv.loc[valid_tmp, "Zprm_well"].to_numpy(float)
            z_p_tmp = preds_tmp[valid_tmp]

            # Помечаем лучшую схему звёздочкой
            plot_title = f"⭐ {scheme_name}" if scheme_name == best_scheme else scheme_name

            fig_tmp, coeffs_tmp = build_qc_crossplot(
                df=df_cv.loc[valid_tmp],
                x_ref=z_w_tmp,
                y_ref=z_p_tmp,
                title=plot_title,
                x_label=PARAM_1_LABEL,
                y_label=f"Predicted, {PARAM_1_UNIT}",
                point_color=crossplot_point_color,
                regression_color=qc_regression_color,
                identity_color=identity_line_color,
                identity_width=identity_line_width,
                identity_dash=identity_line_dash,
                wells_label=wells_legend,
                well_label=well_label,
                regression_name=regression_cv_label,
                regression_width=qc_regression_width,
                regression_dash=qc_regression_dash,
                marker_size=crossplot_point_size,
                marker_opacity=crossplot_point_opacity,
                height=280,
            )

            # Basic QC cross‑plot step 2
            st.plotly_chart(fig_tmp, use_container_width=True )
            stats_tmp = sq.basic_crossplot_stats_xy(
                z_p_tmp,   # Predicted
                z_w_tmp    # Actual
            )

            eq_tmp = (
                f"Predicted = "
                f"{coeffs_tmp[0]:.5f}·Actual + "
                f"{coeffs_tmp[1]:.5f}"
            )
            # Расчёт метрик кросс-плота для Шага 2
            cv_summary_rows.append({
                "CV scheme": scheme_name,
                **basicstats_to_dict(
                stats_tmp,
                eq_tmp
            )
})

# --- Таблица по всем схемам CV
st.markdown(f"### {step2_metrics_all_title}")
st.dataframe(pd.DataFrame(cv_summary_rows), use_container_width=True)

st.markdown(f"### {step2_cv_comparison_title}")
st.dataframe(
    comparison.style.format({
        "rmse": "{:.2f}",
        "mae": "{:.2f}",
        "bias": "{:+.2f}",

        "rmse_fold_mean": "{:.2f}",
        "rmse_fold_median": "{:.2f}",
        "rmse_fold_min": "{:.2f}",
        "rmse_fold_max": "{:.2f}",
        "rmse_fold_std": "{:.2f}",
        "rmse_fold_range": "{:.2f}",
        "score": "{:.2f}",
    }),
    use_container_width=True
)

# RMSE bar chart в шаге 2

fig_rmse = plot_bar_chart(
    x=comparison.index,
    y=comparison["rmse"],
    palette=rmse_palette,
    orientation="v",
    title=f"{step2_rmse_title}",
    yaxis_title=f"RMSE, {PARAM_1_UNIT}",
    show_percent= False,
    legend_title="CV scheme",
)

st.plotly_chart(fig_rmse, use_container_width=True)

st.success(
    f"{step2_best_scheme_title}: **{best_scheme}** "
    f"(RMSE={comparison.loc[best_scheme,'rmse']:.2f}, "
    f"Std={comparison.loc[best_scheme,'rmse_fold_std']:.2f})"
)

# ------------------------------------------------- #
# STEP 2.2.: Графики кросс-валидации для лучшей модели
# ------------------------------------------------- #
st.markdown('<div id="step-2-cp"></div>', unsafe_allow_html=True)

scheme_for_plot = best_scheme
preds = best_preds
valid = np.isfinite(preds)

base_cols_step2 = ["well_id", "Zprm_well"]
for extra_col in ["X_coord", "Y_coord", "horizon_id"]:
    if extra_col in df_cv.columns:
        base_cols_step2.append(extra_col)

plot_df = df_cv.loc[valid, base_cols_step2].copy()
plot_df["Predicted"] = preds[valid]

# Категориальная колонка для отображения в таблицах выбросов
category_for_plot = None
if "tectonic_zone" in df_cv.columns and not pd.api.types.is_numeric_dtype(df_cv["tectonic_zone"]):
    category_for_plot = "tectonic_zone"
elif category_col is not None and category_col in df_cv.columns:
    category_for_plot = category_col

plot_df["category_plot"] = (
    df_cv.loc[valid, category_for_plot].values
    if category_for_plot is not None
    else "—"
)

# --- Вычисляем невязки один раз ---
# CV residual = Actual - Predicted
resid_cv = plot_df["Zprm_well"] - plot_df["Predicted"]
# --- Разметка: слева Predicted vs Actual, справа кривые аппроксимации
col_pred, col_fit = st.columns(2)

# -- CV PLOT 1 — Predicted vs Actual + регрессия
with col_pred:
    st.markdown(
        f"### {step2_cv_crossplot_title} "
        f"— {scheme_for_plot}"
    )
    # Линия Y = X (идеальная карта)
    z_w_cv = plot_df["Zprm_well"].to_numpy(float)
    z_p_cv = plot_df["Predicted"].to_numpy(float)

    fig_cv1, coeffs_cv = build_qc_crossplot(
        df=plot_df,
        x_ref=z_w_cv,
        y_ref=z_p_cv,
        title=f"Cross-plot Predicted vs Actual ({scheme_for_plot})",
        x_label=PARAM_1_LABEL,
        y_label=f"Predicted, {PARAM_1_UNIT}",
        point_color=crossplot_point_color,
        regression_color=qc_regression_color,
        identity_color=identity_line_color,
        identity_width=identity_line_width,
        identity_dash=identity_line_dash,
        wells_label=wells_legend,
        well_label=well_label,
        regression_name=regression_cv_label,
        regression_width=qc_regression_width,
        regression_dash=qc_regression_dash,
        marker_size=crossplot_point_size,
        marker_opacity=crossplot_point_opacity,
        height=380,
    )
    st.plotly_chart(fig_cv1, use_container_width=True)

# ---- CV PLOT 2 — кривые аппроксимации
with col_fit:
    st.markdown(
        f"### {step2_approximations_title} "
        f"— {scheme_for_plot}"
    )
    # -----  ЛИНЕЙНАЯ РЕГРЕССИЯ ДЛЯ CV -----
    z_line_cv = np.linspace(z_w_cv.min(), z_w_cv.max(), 300)

    fig_cv2, approx_rows = build_approximation_plot(
        df=plot_df,
        x_ref=z_w_cv,
        y_ref=z_p_cv,
        z_line=z_line_cv,
        approx_types=cv_approx_types,
        title=f"Cross-plot approx ({scheme_for_plot})",
        x_label=PARAM_1_LABEL,
        y_label=f"Predicted, {PARAM_1_UNIT}",
        x_name="Actual",
        y_name="Predicted",
        point_color=approx_point_color,
        identity_color=identity_line_color,
        wells_label=wells_legend,
        well_label=well_label,
        error_label=error_column_label,
        approx_colors=APPROX_COLORS,
        quantile_colors=quantile_colors,
        approx_dashes=APPROX_DASHES,          
        quantile_dashes=QUANTILE_DASHES, 
        approx_widths=APPROX_WIDTHS,        
        quantile_widths=QUANTILE_WIDTHS,       
        identity_width=identity_line_width,
        identity_dash=identity_line_dash,
        marker_size=approx_point_size,
        marker_opacity=approx_point_opacity,
    )
    st.plotly_chart(fig_cv2, use_container_width=True)

# --- Метрики Predicted vs Actual

cv_fit_stats = sq.basic_crossplot_stats_xy(
    z_p_cv,   # Predicted
    z_w_cv    # Actual
)

cv_metrics_df = build_qc_metrics_table(
    basic=cv_fit_stats,
    coeffs_qc=coeffs_cv,
    left_name="Predicted",
    right_name="Actual",
)

# Метрики Cross-Plot (Predicted vs Actual) для Шага 2
st.markdown(
    f"### {step2_cv_metrics_title} "
    f"— {scheme_for_plot}"
)
st.dataframe(cv_metrics_df, use_container_width=True,)

st.markdown(
    f"### {step2_approx_comparison_title} "
    f"— {scheme_for_plot}"
)
st.dataframe(pd.DataFrame(approx_rows), use_container_width=True)

# -------------------------------------------------------------------- #
# STEP 2.3.: CV-невязки (Actual − Predicted) vs PARAM_1 + статистика
# -------------------------------------------------------------------- #
st.markdown('<div id="step-2-res"></div>', unsafe_allow_html=True)

CV_RESIDUAL_LABEL = f"{PARAM_1} − Predicted"

st.markdown(f"### {step2_residual_distribution_title} "
    f"({CV_RESIDUAL_LABEL})")
st.markdown(
    f"#### {step2_residuals_title} "
    f"({CV_RESIDUAL_LABEL}) vs {PARAM_1}"
)

col_cv_resid_plot, col_cv_resid_stats = st.columns(2)

# График невязок
with col_cv_resid_plot:
    fig_cv_resid = plot_residual_vs_depth(
        z_values=z_w_cv,
        residuals=resid_cv,
        well_ids=plot_df["well_id"],
        color=residual_color,
        wells_label=wells_legend,
        well_label=well_label,
        point_size=residual_point_size,
        point_opacity=residual_point_opacity,
        xlabel=PARAM_1_LABEL,
        ylabel=f"{CV_RESIDUAL_LABEL}, {PARAM_1_UNIT}",
        hover_label=f"{CV_RESIDUAL_LABEL} ({PARAM_1_UNIT})",
        title=f"CV Δ({CV_RESIDUAL_LABEL}) vs {PARAM_1}"
    )
    st.plotly_chart(fig_cv_resid, use_container_width=True)

with col_cv_resid_stats:
    st.markdown(f"#### {step2_residual_stats_title} ({CV_RESIDUAL_LABEL})")
    # Статистика невязок в шаге 2
    cv_stats = compute_stats(resid_cv)
    st.dataframe(cv_stats, use_container_width=True)

# --- Histogram и boxplot CV-невязок 
col_ctrl_r1, col_ctrl_r2 = st.columns(2)

with col_ctrl_r1:
    cv_bins = st.slider(f"{hist_bins_label} CV", 10, 120, 40, key="cv_bins")
with col_ctrl_r2:
    cv_histnorm = st.selectbox(
        f"{hist_norm_label} CV",
        ["none", "density", "probability", "percent"],
        format_func=lambda x: hist_norm_map[x],
        index=0, key="cv_histnorm"
    )

cv_histnorm_value = "" if cv_histnorm == "none" else cv_histnorm
col_hcv, col_bcv = st.columns(2)

# Histogram в шаге 2
with col_hcv:
    st.markdown(f"#### {hist_title_t} CV ({CV_RESIDUAL_LABEL})")
    fig_hcv = plot_hist_kde(
        data=resid_cv,
        bins=cv_bins,
        histnorm=cv_histnorm_value,
        color=hist_color,
        title=f"{hist_title_t} CV ({CV_RESIDUAL_LABEL})",
        xlabel=f"{CV_RESIDUAL_LABEL}, {PARAM_1_UNIT}",
        show_sigma=True,
        kde_color=kde_color,       kde_dash=kde_dash,       kde_width=kde_width,
        mean_color=mean_color,     mean_dash=mean_dash,     mean_width=mean_width,
        median_color=median_color, median_dash=median_dash, median_width=median_width,
        sigma_color=sigma_color,   sigma_dash=sigma_dash,   sigma_width=sigma_width,
    )
    st.plotly_chart(fig_hcv, use_container_width=True)

# Boxplot в шаге 2
with col_bcv:
    st.markdown(f"#### {boxplot_title_t} CV ({CV_RESIDUAL_LABEL})")
    fig_bcv = plot_box(
        data=resid_cv,
        color=box_color,
        well_label=well_label,
        title=f"{boxplot_title_t} CV ({CV_RESIDUAL_LABEL})",
        ylabel=f"{CV_RESIDUAL_LABEL}, {PARAM_1_UNIT}",
        well_ids=plot_df["well_id"],
        value_label=f"{CV_RESIDUAL_LABEL}, {PARAM_1_UNIT}",
        mean_color=box_mean_color,     mean_dash=box_mean_dash,     mean_width=box_mean_width,
        median_color=box_median_color, median_dash=box_median_dash, median_width=box_median_width,
    )
    st.plotly_chart(fig_bcv, use_container_width=True)

# --- Анализ выбросов по результатам CV-невязок
st.markdown('<div id="step-2-otl"></div>', unsafe_allow_html=True)
st.markdown(f"### {step2_outlier_analysis_title}")
show_outlier_tables(
    df=plot_df,
    residuals=resid_cv,
    sigma_thr=sigma_thr,
    param_1=PARAM_1,
    param_2=PARAM_2,
    residual_column_name=cv_residual_col_label.format(param=PARAM_1),
    no_outliers_label=no_outliers_label,
)

# ================================================================================================================================ #
# --------------------------------------------------   Шаг 3: Кривые по зонам + метрики   ------------------------------------------
# ================================================================================================================================ #
# --------------
# STEP 3 LABELS
# ---------------
step3_title = T["step3_title"]
step3_no_categories = T["step3_no_categories"]
step3_distribution_title = T["step3_distribution_title"]
step3_well_count_title = T["step3_well_count_title"]
step3_stats_title = T["step3_stats_title"]
step3_boxplot_caption = T["step3_boxplot_caption"]
step3_relationship_title = T["step3_relationship_title"]
step3_select_zones_title = T["step3_select_zones_title"]
step3_all_categories_title = T["step3_all_categories_title"]
step3_selected_categories_title = T["step3_selected_categories_title"]
step3_no_zone_selected = T["step3_no_zone_selected"]
step3_individual_categories_title = T["step3_individual_categories_title"]
step3_metrics_comparison_title = T["step3_metrics_comparison_title"]

st.markdown('<div id="step-3"></div>', unsafe_allow_html=True)
# ------------------------------------------------- #
# STEP 3.0.: 
# ------------------------------------------------- #
st.markdown(
    f"## {step3_title}: "
    f"{category_col} "
    f"({PARAM_1} vs {PARAM_2})"
)

if category_col is None:
    st.info(step3_no_categories)
else:

    # ------------------------------------------------- #
    # STEP 3.1.: РАСПРЕДЕЛЕНИЕ СКВАЖИН ПО ЗОНАМ
    # ------------------------------------------------- #
    st.markdown('<div id="step-3-pb"></div>', unsafe_allow_html=True)
    st.markdown(
        f"### {step3_distribution_title}: "
        f"{category_col} "
        f"({PARAM_1} vs {PARAM_2})"
    )

    # Круговая диаграмма в шаге 3
    fig_zone_pie = plot_zone_pie(df, category_col=category_col, palette=pie_palette, title=f"{step3_well_count_title} : {category_col}")
    st.plotly_chart(fig_zone_pie, use_container_width=True)
    
    # Bar chart в Шаге 3
    zones = sorted(df[category_col].dropna().unique())
    zone_counts = (df[category_col].value_counts().sort_index())
    fig_zones = plot_bar_chart(
        x=zone_counts.index,
        y=zone_counts.values,
        palette=cat_palette,
        orientation="h",
        title=f"{step3_well_count_title} : {category_col}",
        yaxis_title= well_count_label,
        show_percent=True
    )

    st.plotly_chart(fig_zones, use_container_width=True)

    # Цвета категорий из Sidebar
    zone_marker_colors = {zone: zone_point_colors.get(zone, "gray") for zone in zones}
    zone_curve_colors = {zone: zone_line_colors.get(zone, "gray") for zone in zones}

    # Статистика параметров по категориям
    st.markdown(
        f"### {step3_stats_title}: "
        f"{category_col} "
        f"({PARAM_1} vs {PARAM_2})"
    )    
    for zone in zones:
        zone_part = df[df[category_col] == zone]
        with st.expander(f"{category_col}: {zone}", expanded=False):
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                show_depth_stats(zone_part["Zprm_well"], f"{PARAM_1_LABEL} ({zone})")
            with col_stat2:
               show_depth_stats(zone_part["Zprm_map"], f"{PARAM_2_LABEL} ({zone})")

    st.caption(step3_boxplot_caption)

    # Boxplot для категорий в шаге 3
    col_box1, col_box2 = st.columns(2)

    with col_box1:
        fig_box_p1 = plot_zone_boxplot(
            df=df,
            value_col="Zprm_well",
            well_label=well_label,
            zone_col=category_col,
            zone_label=category_col,
            title=f"Boxplot {PARAM_1}",
            yaxis_title=PARAM_1_LABEL,
            zone_colors=zone_marker_colors,
            value_label=PARAM_1_LABEL,
        )
        st.plotly_chart(fig_box_p1, use_container_width=True)

    with col_box2:
        fig_box_p2 = plot_zone_boxplot(
            df=df,
            value_col="Zprm_map",
            well_label=well_label,
            zone_col=category_col,
            zone_label=category_col,
            title=f"Boxplot {PARAM_2}",
            yaxis_title=PARAM_2_LABEL,
            zone_colors=zone_marker_colors,
            value_label=PARAM_2_LABEL,
        )
        st.plotly_chart(fig_box_p2, use_container_width=True)

    zone_fit_rows = []

    # ------------------------------------------------- #
    # # STEP 3.2.: ОБЩИЙ ГРАФИК ПО ВСЕМ ЗОНАМ
    # ------------------------------------------------- #
    st.markdown('<div id="step-3-cp"></div>', unsafe_allow_html=True)
    st.markdown(
        f"### {step3_relationship_title}: "
        f"{category_col} "
        f"({PARAM_1} vs {PARAM_2})"
    )
    selected_zones = st.multiselect(f"{step3_select_zones_title} ({category_col})", options=zones, default=zones, key="step3_selected_zones")
    df_selected = df[df[category_col].isin(selected_zones)].copy()
    has_selected_zones = not df_selected.empty

    fig_zone_all_full = build_zone_overview_plot(
        df_source=df,
        zones_to_plot=zones,
        category_col=category_col,
        zone_marker_colors=zone_marker_colors,
        zone_curve_colors=zone_curve_colors,
        approx_types=zone_approx_types,
        param_1=PARAM_1,
        param_2=PARAM_2,
        well_label=well_label,
        approx_colors=APPROX_COLORS,
        quantile_colors=quantile_colors,
        approx_dashes=APPROX_DASHES,          
        quantile_dashes=QUANTILE_DASHES,    
        approx_widths=APPROX_WIDTHS,   
        point_size=zone_point_size,
        point_opacity=zone_point_opacity,
    )
    fig_zone_all_full = add_qc_line_to_zone_plot(
        fig=fig_zone_all_full,
        df_source=df,
        param_1=PARAM_1, param_2=PARAM_2,
        regression_qc_name=regression_qc_label,
        regression_color=qc_regression_color,
        identity_color=identity_line_color,
        regression_width=qc_regression_width,
        regression_dash=qc_regression_dash,
        identity_width=identity_line_width,
        identity_dash=identity_line_dash,
    )   

    if has_selected_zones:
        fig_zone_all_selected = build_zone_overview_plot(
            df_source=df,
            zones_to_plot=selected_zones,
            category_col=category_col,
            zone_marker_colors=zone_marker_colors,
            zone_curve_colors=zone_curve_colors,
            approx_types=zone_approx_types,
            param_1=PARAM_1,
            param_2=PARAM_2,
            well_label=well_label,
            approx_colors=APPROX_COLORS,
            quantile_colors=quantile_colors,
            approx_dashes=APPROX_DASHES,          
            quantile_dashes=QUANTILE_DASHES,  
            approx_widths=APPROX_WIDTHS, 
            point_size=zone_point_size,
            point_opacity=zone_point_opacity,
        )
        fig_zone_all_selected = add_qc_line_to_zone_plot(
            fig=fig_zone_all_selected,
            df_source=df_selected,
            param_1=PARAM_1, param_2=PARAM_2,
            regression_qc_name=regression_qc_label,
            regression_color=qc_regression_color,
            identity_color=identity_line_color,
            regression_width=qc_regression_width,
            regression_dash=qc_regression_dash,
            identity_width=identity_line_width,
            identity_dash=identity_line_dash,
        )

    col_all, col_selected = st.columns(2)
    with col_all:
        st.caption(step3_all_categories_title)
        apply_crossplot_layout(fig_zone_all_full, PARAM_2_LABEL, PARAM_1_LABEL, title=f"{step3_all_categories_title} ({category_col})",)
        st.plotly_chart(fig_zone_all_full, use_container_width=True,)
    with col_selected:
        st.caption(step3_selected_categories_title)
        if has_selected_zones:
            apply_crossplot_layout(fig_zone_all_selected, PARAM_2_LABEL, PARAM_1_LABEL, title=f"{step3_selected_categories_title} ({category_col})",)
            st.plotly_chart(fig_zone_all_selected, use_container_width=True,)
        else:
            st.info(step3_no_zone_selected)

    # ------------------------------------------------- #
    # STEP 3.3.: ОТДЕЛЬНЫЕ ЗОНЫ
    # ------------------------------------------------- #
    st.markdown(f"### {step3_individual_categories_title} ({category_col})")

    for zone in zones:

        # определение категорий в Шаге 3
        zone_data = prepare_zone_data(df,category_col, zone)
        if zone_data is None:
            continue
        zone_part, xz, yz, x_line_z = zone_data

        zone_marker_color = zone_marker_colors.get(zone, "gray")

        fig_zone = go.Figure()

        # Точки скважин
        fig_zone.add_trace(
            go.Scatter(
                x=xz,
                y=yz,
                mode="markers",
                name=f"{category_col}: {zone}",
                marker=dict(size=zone_point_size, color=zone_marker_color, opacity=zone_point_opacity,),
                text=zone_part["well_id"],
                hovertemplate=(
                    f"{well_label}: %{{text}}"
                    f"<br>{PARAM_2}=%{{x}}"
                    f"<br>{PARAM_1}=%{{y}}"
                    f"<extra></extra>"
                )
            )
        )

        # Линии аппроксимации
        zone_line_color = zone_curve_colors.get(zone, "gray")
        for approx in zone_approx_types:
            try:
                # --- вычисление кривой ---
                z_fit, eq, color = compute_curve(approx, xz, yz, x_line_z, APPROX_COLORS, quantile_colors)
                approx_dash = APPROX_DASHES.get(approx, "solid")
                approx_width = APPROX_WIDTHS.get(approx, 2)
                if z_fit is not None:
                    fig_zone.add_trace(
                        go.Scatter(
                            x=x_line_z,
                            y=z_fit,
                            mode="lines",
                            name=approx,
                            line=dict(width=approx_width, color=zone_line_color, dash=approx_dash)
                        )
                    )

                stats_zone = sq.compute_fit_stats(xz, yz, approx)

                # статистика метрик выбранных кривых аппроксимации в шаге 3
                zone_fit_rows.append({category_col: zone, "Approximation": approx, **fitstats_to_dict(stats_zone)})
            except Exception as e:
                zone_fit_rows.append({category_col: zone, "Approximation": approx, error_column_label: str(e)})

        # Линия Y = X (идеальная карта)
        lo = min(xz.min(), yz.min())
        hi = max(xz.max(), yz.max())
        add_identity_line(fig_zone, lo, hi, color=identity_line_color, width=identity_line_width, dash=identity_line_dash,)
        
        # Настройка оформления cross-plot
        apply_crossplot_layout(fig_zone, PARAM_2_LABEL, PARAM_1_LABEL, height=360, title=f"{category_col}: {zone}")
        st.plotly_chart(fig_zone, use_container_width=True)

    st.markdown(
        f"### {step3_metrics_comparison_title}: "
        f"{category_col}"
    )
    st.dataframe(pd.DataFrame(zone_fit_rows), use_container_width=True)

# ================================================================================================================================ #
# --------------------------------------------------   Шаг 4: РУЧНОЕ ИСКЛЮЧЕНИЕ СКВАЖИН   ------------------------------------------
# ================================================================================================================================ #
# --------------
# STEP 4 LABELS
# ---------------
step4_title = T["step4_title"]
step4_less_than_3_wells = T["step4_less_than_3_wells"]
step4_summary_initial = T["step4_summary_initial"]
step4_summary_random = T["step4_summary_random"]
step4_summary_manual = T["step4_summary_manual"]
step4_summary_remaining = T["step4_summary_remaining"]
step4_excluded_wells_title = T["step4_excluded_wells_title"]
step4_excluded_manual_title = T["step4_excluded_manual_title"]
step4_excluded_random_title = T["step4_excluded_random_title"]
step4_crossplot_title = T["step4_crossplot_title"]
step4_cp_title = T["step4_cp_title"]
step4_approximations_title = T["step4_approximations_title"]
step4_appr_title = T["step4_appr_title"]
step4_qc_metrics_title = T["step4_qc_metrics_title"]
step4_approx_comparison_title = T["step4_approx_comparison_title"]
step4_before_after_metrics_title = T["step4_before_after_metrics_title"]
step4_filter_hint = T["step4_filter_hint"]
step4_before_filter_plots = T["step4_before_filter_plots"]
step4_after_filter_plots = T["step4_after_filter_plots"]
step4_crossplot_title_before = T["step4_crossplot_title_before"]
step4_cp_title_before = T["step4_cp_title_before"]
step4_approximations_title_before = T["step4_approximations_title_before"]
step4_appr_title_before = T["step4_appr_title_before"]

st.markdown('<div id="step-4"></div>', unsafe_allow_html=True)
# --------------------------------------------------
# STEP 4.1.: ФИЛЬТРАЦИЯ СКВАЖИН
# ------------------------------------------------- #
st.markdown(f"## {step4_title}") 

df_step4 = df.copy()
excluded_wells_random = []

# ------ Ручное исключение ------
if filter_mode == FILTER_MANUAL:
    if len(excluded_wells_manual) > 0:
        df_step4 = df_step4[~df_step4["well_id"].astype(str).isin(excluded_wells_manual)]

# ------ Случайное исключение ------
elif filter_mode == FILTER_RANDOM:
    rng = np.random.default_rng(st.session_state.step4_random_seed)
    available_wells = sorted(df_step4["well_id"].astype(str).unique())
    n_random = n_excluded
    if n_random > 0:
        excluded_wells_random = rng.choice(available_wells, size=n_random, replace=False,).tolist()
    else:
        excluded_wells_random = []
    df_step4 = df_step4[~df_step4["well_id"].astype(str).isin(excluded_wells_random)]

# ------ Выделение кликом по точке ------
excluded_wells_click = []
if filter_mode == FILTER_CLICK:

    if "step4_click_excluded" not in st.session_state:
        st.session_state.step4_click_excluded = []

    st.markdown(f"### {step4_click_title}")
    st.caption(step4_click_hint)

    df_click_source = df.copy()
    is_excluded_mask = (df_click_source["well_id"].astype(str).isin(st.session_state.step4_click_excluded))

    fig_click = go.Figure()

    # Обычные точки
    fig_click.add_trace(
        go.Scatter(
            x=df_click_source.loc[~is_excluded_mask, "Zprm_map"],
            y=df_click_source.loc[~is_excluded_mask, "Zprm_well"],
            mode="markers",
            name="Wells",
            marker=dict(color=crossplot_point_color, size=crossplot_point_size, opacity=crossplot_point_opacity,),
            text=df_click_source.loc[~is_excluded_mask, "well_id"],
            hovertemplate=(
                f"{well_label}: %{{text}}"
                f"<br>{PARAM_2_LABEL}=%{{x}}"
                f"<br>{PARAM_1_LABEL}=%{{y}}"
                "<extra></extra>"
            ),
        )
    )

    # Исключённые точки — серые
    fig_click.add_trace(
        go.Scatter(
            x=df_click_source.loc[is_excluded_mask, "Zprm_map"],
            y=df_click_source.loc[is_excluded_mask, "Zprm_well"],
            mode="markers",
            name=step4_click_excluded_wells,
            marker=dict(color="lightgray", size=crossplot_point_size, opacity=0.6,),
            text=df_click_source.loc[is_excluded_mask, "well_id"],
            hovertemplate=(
                f"{well_label}: %{{text}}"
                f"<br>{PARAM_2_LABEL}=%{{x}}"
                f"<br>{PARAM_1_LABEL}=%{{y}}"
                "<extra></extra>"
            ),
        )
    )

    # Линия регрессии QC — визуальный ориентир для выделения точек
    x_line_qc = np.linspace(
        df_click_source["Zprm_map"].min(),
        df_click_source["Zprm_map"].max(),
        300,
    )
    add_regression_line(
        fig=fig_click,
        x=df_click_source["Zprm_map"].to_numpy(float),
        y=df_click_source["Zprm_well"].to_numpy(float),
        name=regression_qc_label,
        color=qc_regression_color,
        width=qc_regression_width,
        dash=qc_regression_dash,
        x_line=x_line_qc,
    )

    # Линия Y = X — второй ориентир
    lo = min(
        df_click_source["Zprm_map"].min(),
        df_click_source["Zprm_well"].min(),
    )
    hi = max(
        df_click_source["Zprm_map"].max(),
        df_click_source["Zprm_well"].max(),
    )
    add_identity_line(
        fig_click, lo, hi,
        color=identity_line_color,
        width=identity_line_width,
        dash=identity_line_dash,
    )

    apply_crossplot_layout(fig_click, PARAM_2_LABEL, PARAM_1_LABEL, title=step4_click_title,)
    fig_click.update_layout(clickmode="event+select")

    # --- Только точечное выделение ---
    sel = st.plotly_chart(
        fig_click,
        key="step4_click_plot",
        on_select="rerun",
        selection_mode=["points", "box", "lasso"],
        use_container_width=True,
    )

    selected_wells = []
    if sel and sel.get("selection", {}).get("points"):
        for p in sel["selection"]["points"]:
            wid = p.get("text")
            if wid is not None:
                selected_wells.append(str(wid))

    col_apply, col_reset, col_count = st.columns([2, 2, 3])
    with col_apply:
        apply_clicked = st.button(
            step4_click_apply_button,
            key="step4_click_apply",
            disabled=(len(selected_wells) == 0),
        )
    with col_reset:
        reset_clicked = st.button(
            step4_click_reset_button,
            key="step4_click_reset_main",
        )
    with col_count:
        st.markdown(
            f"**{step4_click_selected_count}** {len(selected_wells)}  \n"
            f"**{step4_click_excluded_count}** {len(st.session_state.step4_click_excluded)}"
        )

    if reset_clicked:
        st.session_state.step4_click_excluded = []
        st.rerun()

    if apply_clicked and selected_wells:
        # каждое новое выделение ЗАМЕНЯЕТ список исключённых
        st.session_state.step4_click_excluded = list(dict.fromkeys(selected_wells))
        st.rerun()

    if st.session_state.step4_click_excluded:
        with st.expander(step4_click_excluded_expander, expanded=False):
            kept = st.multiselect(
                step4_click_excluded_wells,
                options=st.session_state.step4_click_excluded,
                default=st.session_state.step4_click_excluded,
                key="step4_click_kept",
            )
            if set(kept) != set(st.session_state.step4_click_excluded):
                st.session_state.step4_click_excluded = list(kept)
                st.rerun()

    excluded_wells_click = st.session_state.step4_click_excluded
    if excluded_wells_click:
        df_step4 = df_step4[
            ~df_step4["well_id"].astype(str).isin(excluded_wells_click)
        ]

# ------ Контроль ------
step4_ok = len(df_step4) >= 3
if not step4_ok:
    st.warning(step4_less_than_3_wells)
    st.info(step4_filter_hint)

# ------ Сводка ------
n_wells_init = df["well_id"].nunique() if len(df) > 0 else 0
n_wells_step4 = df_step4["well_id"].nunique() if len(df_step4) > 0 else 0
remaining_pct = (100 * n_wells_step4 / n_wells_init if n_wells_init > 0 else 0)

st.markdown(
    f"""
**{step4_summary_initial}:** {n_wells_init}

**{step4_summary_random}:** {len(excluded_wells_random)}

**{step4_summary_manual}:** {len(excluded_wells_manual)}

**{step4_summary_click}:** {len(excluded_wells_click)}

**{step4_summary_remaining}:** {n_wells_step4} ({remaining_pct:.1f}%)
"""
)

# Детализация по запросу
with st.expander(step4_excluded_wells_title):
    if len(excluded_wells_manual) > 0:
        st.markdown(f"#### {step4_excluded_manual_title}")
        st.write(", ".join(excluded_wells_manual))

    if len(excluded_wells_random) > 0:
        st.markdown(f"#### {step4_excluded_random_title}")
        st.write(", ".join(excluded_wells_random))

if step4_ok:
    # --------------------------------------------------
    # STEP 4.2.: Расчёт статистики
    # --------------------------------------------------
    basic_step4 = sq.basic_crossplot_stats(df_step4)

    z_m_step4 = df_step4["Zprm_map"].to_numpy(float)
    z_w_step4 = df_step4["Zprm_well"].to_numpy(float)

    x_ref_step4 = z_m_step4
    y_ref_step4 = z_w_step4

    z_line_step4 = np.linspace(x_ref_step4.min(), x_ref_step4.max(), 300,)

    # --------------------------------------------------
    # STEP 4.3a.: QC Cross-Plot и аппроксимации ДО фильтрации
    # --------------------------------------------------
    st.markdown(f"### {step4_before_filter_plots}")

    col_left_before, col_right_before = st.columns([1, 1])

    with col_left_before:
        st.markdown(f"#### {step4_crossplot_title_before}: {PARAM_1} vs {PARAM_2}")
        fig_qc_before, coeffs_qc_before = build_qc_crossplot(
            df=df,
            x_ref=x_ref,
            y_ref=y_ref,
            title=step4_cp_title_before,
            x_label=PARAM_2_LABEL,
            y_label=PARAM_1_LABEL,
            point_color=crossplot_point_color,
            regression_color=qc_regression_color,
            identity_color=identity_line_color,
            identity_width=identity_line_width,
            identity_dash=identity_line_dash,
            wells_label=wells_legend,
            well_label=well_label,
            regression_name=regression_qc_label,
            regression_width=qc_regression_width,
            regression_dash=qc_regression_dash,
            marker_size=crossplot_point_size,
            marker_opacity=crossplot_point_opacity,
        )
        st.plotly_chart(fig_qc_before, use_container_width=True)

    with col_right_before:
        st.markdown(f"#### {step4_approximations_title_before}: {PARAM_1} vs {PARAM_2}")
        fig_fit_before, fit_rows_before = build_approximation_plot(
            df=df,
            x_ref=x_ref,
            y_ref=y_ref,
            z_line=z_line,
            approx_types=approx_types,
            title=step4_appr_title_before,
            x_label=PARAM_2_LABEL,
            y_label=PARAM_1_LABEL,
            x_name=PARAM_2,
            y_name=PARAM_1,
            point_color=approx_point_color,
            identity_color=identity_line_color,
            wells_label=wells_legend,
            well_label=well_label,
            error_label=error_column_label,
            approx_colors=APPROX_COLORS,
            quantile_colors=quantile_colors,
            approx_dashes=APPROX_DASHES,
            quantile_dashes=QUANTILE_DASHES,
            approx_widths=APPROX_WIDTHS,        
            quantile_widths=QUANTILE_WIDTHS,  
            identity_width=identity_line_width,
            identity_dash=identity_line_dash,
            marker_size=approx_point_size,
            marker_opacity=approx_point_opacity,
        )
        st.plotly_chart(fig_fit_before, use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------
    # STEP 4.3b.: QC Cross-Plot
    # --------------------------------------------------
    st.markdown(f"### {step4_after_filter_plots}")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(
            f"#### {step4_crossplot_title}: "
            f"{PARAM_1} vs {PARAM_2}"
        )

        fig_qc_step4, coeffs_qc_step4 = build_qc_crossplot(
            df=df_step4,
            x_ref=x_ref_step4,
            y_ref=y_ref_step4,
            title=step4_cp_title,
            x_label=PARAM_2_LABEL,
            y_label=PARAM_1_LABEL,
            point_color=crossplot_point_color,
            regression_color=qc_regression_color,
            identity_color=identity_line_color,
            identity_width=identity_line_width,
            identity_dash=identity_line_dash,
            wells_label=wells_legend, 
            well_label=well_label,
            regression_name=regression_qc_label,
            regression_width=qc_regression_width,
            regression_dash=qc_regression_dash,
            marker_size=crossplot_point_size,
            marker_opacity=crossplot_point_opacity,
        )

        st.plotly_chart(fig_qc_step4, use_container_width=True)

    # STEP 4.3.: Аппроксимации
    with col_right:
        st.markdown(
            f"#### {step4_approximations_title}: "
            f"{PARAM_1} vs {PARAM_2}"
        )
        fig_fit_step4, fit_rows_step4 = build_approximation_plot(
            df=df_step4,
            x_ref=x_ref_step4,
            y_ref=y_ref_step4,
            z_line=z_line_step4,
            approx_types=approx_types,
            title=step4_appr_title,
            x_label=PARAM_2_LABEL,
            y_label=PARAM_1_LABEL,
            x_name=PARAM_2,
            y_name=PARAM_1,
            point_color=approx_point_color,
            identity_color=identity_line_color,
            wells_label=wells_legend,
            well_label=well_label,
            error_label=error_column_label,
            approx_colors=APPROX_COLORS,
            quantile_colors=quantile_colors,
            approx_dashes=APPROX_DASHES,          
            quantile_dashes=QUANTILE_DASHES,    
            approx_widths=APPROX_WIDTHS,        
            quantile_widths=QUANTILE_WIDTHS,    
            identity_width=identity_line_width,
            identity_dash=identity_line_dash,
            marker_size=approx_point_size,
            marker_opacity=approx_point_opacity,
        )

        st.plotly_chart(fig_fit_step4, use_container_width=True)

    # STEP 4.3.: Таблица метрик QC Cross-Plot
    qc_metrics_step4 = build_qc_metrics_table(basic=basic_step4, coeffs_qc=coeffs_qc_step4, left_name=PARAM_1, right_name=PARAM_2,)
    st.markdown(f"### {step4_qc_metrics_title}")
    st.dataframe(qc_metrics_step4, use_container_width=True,)

    # STEP 4.3.: Таблица метрик кривых аппроксимации
    st.markdown(f"### {step4_approx_comparison_title}")
    st.dataframe(pd.DataFrame(fit_rows_step4), use_container_width=True,)

    # --------------------------------------------------
    # STEP 4.4.: Таблица метрик до и после фильтрации
    # --------------------------------------------------
    comparison_df = pd.DataFrame({
        "Metrics": [
            "Well",
            "R²_lin",
            "RMSE",
            "MSE",
            "MAE",
            "Bias",
            "Std",
            "Variance",
            "MAPE %",
        ],
        "before filt": [
            len(df),
            basic.r2,
            basic.rmse,
            basic.mse,
            basic.mae,
            basic.bias,
            basic.std,
            basic.variance,
            basic.mape_pct,
        ],
        "after filt": [
            len(df_step4),
            basic_step4.r2,
            basic_step4.rmse,
            basic_step4.mse,
            basic_step4.mae,
            basic_step4.bias,
            basic_step4.std,
            basic_step4.variance,
            basic_step4.mape_pct,
        ]
    })

    st.markdown(f"### {step4_before_after_metrics_title}")
    st.dataframe(comparison_df, use_container_width=True,)

st.markdown("---")
st.caption("CROSSPL✳T.structural · данные не сохраняются на сервере")
st.caption("© A.S.Pozdniakov 2026")