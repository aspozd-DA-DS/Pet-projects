"""
#app_full.py — интерактивный Streamlit-дашборд для анализа связи эталонного параметра (H_eff или Phi) с сейсмическими атрибутами по данным скважин.
"""
import numpy as np
import pandas as pd
import io
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

import attribute_qc_full as sq
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
    cached_plotly_pairplot,
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
    plot_zone_overlay_hist,
    plot_residual_vs_depth, 
    plot_residual_map, 
    build_qc_crossplot,
    build_approximation_plot,
    build_zone_overview_plot, 
    add_qc_line_to_zone_plot,
    show_depth_stats,
    show_outlier_table,
    show_outlier_tables,
    PALETTE_DICT,
    corr_pairs_table,
    plot_category_crosstab,
    plot_category_pies,
    crosstab_stats_table,
    category_stats_table,
    excel_download_button,
    cached_fit_stats,
    cached_quantile_stats,
    cached_curve_numeric,
    add_cv_residual_band,
    corr_pairs_grouped,
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
st.set_page_config(page_title="Attribute QC — cross-plot & cross-validation", layout="wide", page_icon="✳",)

# ------ SIDEBAR: Language ------
DEFAULT_LANGUAGE = "EN"   # язык интерфейса при первом открытии страницы

with st.sidebar:
    if "sidebar_language" not in st.session_state:
        st.session_state.sidebar_language = DEFAULT_LANGUAGE
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

heff_col_label = T["heff_col_label"]
phi_col_label = T["phi_col_label"]
heff_units_label = T["heff_units_label"]
phi_units_label = T["phi_units_label"]
step1_select_title = T["step1_select_title"]
step1_no_attributes_msg = T["step1_no_attributes_msg"]
select_horizon_warning = T["select_horizon_warning"]
step0_corr_need_two_msg = T["step0_corr_need_two_msg"]
corr_pick_attrs_label = T["corr_pick_attrs_label"]

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
contents_data_pr = T["contents_data_pr"]
contents_data_res = T["contents_data_res"]
contents_data_cat = T["contents_data_cat"]
contents_data_num = T["contents_data_num"]
contents_step1 = T["contents_step1"]
contents_step1_cp = T["contents_step1_cp"]
contents_step1_hb = T["contents_step1_hb"]
contents_step2 = T["contents_step2"]
contents_step2_bcv = T["contents_step2_bcv"]
contents_step2_cp = T["contents_step2_cp"]
contents_step2_res = T["contents_step2_res"]
contents_step2_otl = T["contents_step2_otl"]
contents_step3 = T["contents_step3"]
contents_step3_pb = T["contents_step3_pb"]
contents_step3_cp = T["contents_step3_cp"]
contents_step4 = T["contents_step4"]
contents_step5 = T["contents_step5"]
contents_step5_lowo = T["contents_step5_lowo"]
contents_step5_kfold = T["contents_step5_kfold"]
contents_step5_heff = T["contents_step5_heff"]
contents_step5_phi = T["contents_step5_phi"]

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
hist_outline_label = T["hist_outline_label"]
hist_outline_on_label = T["hist_outline_on_label"]
hist_opacity_label = T["hist_opacity_label"]
hist_bins_label = T["hist_bins_label"]
zone_hist_opacity_label = T["zone_hist_opacity_label"]
kde_curve_label = T["kde_curve_label"]
mean_line_label = T["mean_line_label"]
median_line_label = T["median_line_label"]
sigma_line_label = T["sigma_line_label"]
boxplot_expander = T["boxplot_expander"]
box_fill_color_label = T["box_fill_color_label"]
box_line_width_label = T["box_line_width_label"]
box_mean_label = T["box_mean_label"]
box_median_label = T["box_median_label"]
box_outline_label = T["box_outline_label"]
box_opacity_label = T["box_opacity_label"]
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

approx_group_title = T["approx_group_title"]
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
sigma_k_label = T["sigma_k_label"]
outlier_iqr_k_label = T["outlier_iqr_k_label"]
outlier_pct_low_label = T["outlier_pct_low_label"]
outlier_pct_high_label = T["outlier_pct_high_label"]

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
buffer_cv_title = T["buffer_cv_title"]
recommended_grid_text = T["recommended_grid_text"]
median_distance_text = T["median_distance_text"]
grid_caption = T["grid_caption"]
grid_slider_label = T["grid_slider_label"]
recommended_buffer_text = T["recommended_buffer_text"]
buffer_based_on_median = T["buffer_based_on_median"]
buffer_caption = T["buffer_caption"]
buffer_slider_label = T["buffer_slider_label"]
leave_p_sp_out_title = T["leave_p_sp_out_title"]
recommended_repeats_text = T["recommended_repeats_text"]
cv_repeats_label = T["cv_repeats_label"]
cv_repeats_caption = T["cv_repeats_caption"]
leave_p_out_title = T["leave_p_out_title"]
leave_p_percent_label = T["leave_p_percent_label"]
leave_p_percent_caption = T["leave_p_percent_caption"]
spatial_leave_p_out_title = T["spatial_leave_p_out_title"]
recommended_radius_text = T["recommended_radius_text"]
spatial_radius_caption = T["spatial_radius_caption"]
spatial_radius_label = T["spatial_radius_label"]
distance_cv_title = T["distance_cv_title"]
variogram_factor_label = T["distance_factor_label"]
variogram_factor_caption = T["distance_factor_caption"]
corr_threshold_label = T["corr_threshold_label"]
excl_radius_mode_label = T["excl_radius_mode_label"]
excl_radius_mode_median_factor = T["excl_radius_mode_median_factor"]
excl_radius_mode_absolute = T["excl_radius_mode_absolute"]
excl_radius_abs_label = T["excl_radius_abs_label"]
excl_radius_abs_caption = T["excl_radius_abs_caption"]
cv_seed_title = T["cv_seed_title"]
cv_seed_current = T["cv_seed_current"]
cv_seed_new_button = T["cv_seed_new_button"]
cv_seed_caption = T["cv_seed_caption"]
cv_seed_reset_button = T["cv_seed_reset_button"]
step4_random_seed_reset_button = T["step4_random_seed_reset_button"]

crosstab_expander = T["crosstab_expander"]
multi_pie_expander = T["multi_pie_expander"]
multi_pie_opacity_label = T["multi_pie_opacity_label"]
multi_pie_outline_label = T["multi_pie_outline_label"]
crosstab_orientation_label = T["crosstab_orientation_label"]
crosstab_orient_v = T["crosstab_orient_v"]
crosstab_orient_h = T["crosstab_orient_h"]
crosstab_outline_label = T["crosstab_outline_label"]
step0_crosstab_title = T["step0_crosstab_title"]
step0_multi_pie_title = T["step0_multi_pie_title"]
step0_crosstab_x_label = T["step0_crosstab_x_label"]
step0_crosstab_group_label = T["step0_crosstab_group_label"]
step0_crosstab_need_two = T["step0_crosstab_need_two"]
crosstab_agg_label = T["crosstab_agg_label"]
crosstab_agg_count = T["crosstab_agg_count"]
crosstab_agg_min = T["crosstab_agg_min"]
crosstab_agg_max = T["crosstab_agg_max"]
crosstab_agg_median = T["crosstab_agg_median"]
crosstab_agg_mean = T["crosstab_agg_mean"]
crosstab_value_label = T["crosstab_value_label"]
crosstab_agg_std = T["crosstab_agg_std"]

bar_orientation_label = T["bar_orientation_label"]
bar_orient_v = T["bar_orient_v"]
bar_orient_h = T["bar_orient_h"]
bar_outline_label = T["bar_outline_label"]
bar_group_load_title = T["bar_group_load_title"]
bar_group_fold_title = T["bar_group_fold_title"]
bar_group_rmse_title = T["bar_group_rmse_title"]
bar_group_step3_title = T["bar_group_step3_title"]
bar_group_step5_title = T["bar_group_step5_title"]


cat_stats_wells = T["cat_stats_wells"]
cat_stats_records = T["cat_stats_records"]
cat_stats_share = T["cat_stats_share"]
crosstab_table_title = T["crosstab_table_title"]
crosstab_table_total = T["crosstab_table_total"]

residual_map_category_label = T["residual_map_category_label"]
residual_map_no_category = T["residual_map_no_category"]
residual_map_highlight_label = T["residual_map_highlight_label"]
residual_map_no_highlight = T["residual_map_no_highlight"]
residual_map_highlight_opacity_label = T["residual_map_highlight_opacity_label"]
residual_map_dim_opacity_label = T["residual_map_dim_opacity_label"]

strat_zone_col_label = T["strat_zone_col_label"]
strat_zone_col_none_hint = T["strat_zone_col_none_hint"]
strat_zone_col_single_value_hint = T["strat_zone_col_single_value_hint"]

step4_n_sets_label = T["step4_n_sets_label"]

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

def save_dataset_to_session(df, unit_heff, unit_phi, zone_col=None,):
    fingerprint = (
        len(df) if df is not None else None,
        tuple(df.columns.tolist()) if df is not None else None,
        unit_heff, unit_phi, zone_col,
    )
    prev_fingerprint = st.session_state.get("_dataset_fingerprint")

    st.session_state["df"] = df
    st.session_state["UNITS_BY_PARAM"] = {"H_eff": unit_heff, "Phi": unit_phi}
    st.session_state["zone_col"] = zone_col

    if prev_fingerprint == fingerprint:
        return

    st.session_state["_dataset_fingerprint"] = fingerprint
    st.session_state["df_filtered"] = None
    st.session_state["df_for_cv"] = None

    for key in [
        "corr_attrs_attr", "corr_attrs_heff", "corr_attrs_phi",
        "sidebar_all_horizons",
        "sidebar_picked_horizons",
        "category_col",
        "pairplot_category",
        "step3_selected_zones",
        "cv_seed",
        "step4_random_seed",
        "step4_click_excluded",
        "step1_param1", "step1_param2", "step1_unit2",
        "best_scheme_criterion",
        "best_scheme_manual",
        "df_for_cv_sig",
    ]:
        st.session_state.pop(key, None)


# ------ PREPARE INPUT DF ------
@st.cache_data(
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df, index=True).values.tobytes()},
)
def prepare_input_dataframe(raw_df, well_col, x_col, y_col, h_col, phi_col, horizon_col=None, zone_col=None,):

    # сохраняем все колонки исходного файла
    df = raw_df.copy()
    df.drop(columns=[well_col, x_col, y_col], inplace=True, errors="ignore",)

    # Страховка: убираем одноимённые «чужие» системные колонки, чтобы присваивание ниже не затирало данные
    reserved = {"well_id", "X_coord", "Y_coord", "Zprm_well", "Zprm_map", "horizon_id", "tectonic_zone", "H_eff", "Phi"}
    sources = {well_col, x_col, y_col, h_col, phi_col, horizon_col, zone_col}
    df.drop(columns=[c for c in reserved if c in df.columns and c not in sources],
            inplace=True, errors="ignore")

    # Системные колонки
    df["well_id"] = raw_df[well_col]
    df["X_coord"] = raw_df[x_col]
    df["Y_coord"] = raw_df[y_col]
    df["H_eff"] = raw_df[h_col]
    df["Phi"] = raw_df[phi_col]

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
    required = ["well_id", "X_coord", "Y_coord", "H_eff", "Phi",]
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
    numeric_cols = ["X_coord", "Y_coord", "H_eff", "Phi",]

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
    required_numeric_cols = ["X_coord", "Y_coord", "H_eff", "Phi",]
    removed_rows = df[df[required_numeric_cols].isna().any(axis=1)].copy()
    before = len(df)
    df = (df.dropna(subset=["X_coord", "Y_coord", "H_eff", "Phi",]).reset_index(drop=True))
    dropped = before - len(df)

    # Контроль результата очистки
    if len(df) == 0:
        raise ValueError(no_valid_rows_after_cleaning)

    return df, dropped, removed_rows
     
# ------ SIDEBAR — данные и настройки ------
st.sidebar.title(sidebar_data_title)
UNIT_OPTIONS = [
    "m", "km", "ft", "ms", "sec", "m/s", "ft/s", "units", "%", "Hz","dB",
    "°C", "MPa", "psi", "ppm", "g/cm³", "fractions"
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

def _guess_col(cols, keywords):
    """Находит колонку по ключевым словам: без регистра и без _ - пробелов.
    Порядок: точное совпадение -> начинается с ключевого слова -> содержит (для слов от 3 букв)."""

    if isinstance(keywords, str):
        keywords = [keywords]

    norm = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
    ncols = [(c, norm(c)) for c in cols]
    for kw in keywords:
        for c, n in ncols:
            if n == norm(kw):
                return c
    for kw in keywords:
        k = norm(kw)
        for c, n in ncols:
            if n.startswith(k):
                return c
    for kw in keywords:
        k = norm(kw)
        if len(k) >= 3:
            for c, n in ncols:
                if k in n:
                    return c
    return None

def show_column_mapping_sidebar(cols, prefix_label="", prefix_key="",):

    # Сбрасываем выбор horizon/zone только если сменился набор колонок (другой файл)
    sig_key = f"{prefix_key}_cols_signature"
    cols_signature = tuple(cols)
    if st.session_state.get(sig_key) != cols_signature:
        st.session_state.pop(f"{prefix_key}_horizon", None)
        st.session_state.pop(f"{prefix_key}_zone", None)
        st.session_state[sig_key] = cols_signature

    def _idx(keywords, fallback=0):
        hit = _guess_col(cols, keywords)
        return cols.index(hit) if hit is not None else min(fallback, len(cols) - 1)

    well_col = st.sidebar.selectbox(f"{prefix_label}well_id", cols, index=_idx(["well_id", "well"], 0), key=f"{prefix_key}_well",)
    x_col = st.sidebar.selectbox(f"{prefix_label}X", cols, index=_idx(["x"], 0), key=f"{prefix_key}_x",)
    y_col = st.sidebar.selectbox(f"{prefix_label}Y", cols, index=_idx(["y"], min(1, len(cols) - 1)), key=f"{prefix_key}_y",)

    h_col = st.sidebar.selectbox(f"{prefix_label}{heff_col_label}", cols, index=_idx(["H_eff", "Heff"], 0), key=f"{prefix_key}_heff",)
    phi_col = st.sidebar.selectbox(f"{prefix_label}{phi_col_label}", cols, index=_idx(["Phi", "Kp"], 1), key=f"{prefix_key}_phi",)

    unit_h = st.sidebar.selectbox(f"{prefix_label}{heff_units_label}", UNIT_OPTIONS, index=UNIT_OPTIONS.index("m"), key=f"{prefix_key}_unit_heff",)
    unit_phi = st.sidebar.selectbox(f"{prefix_label}{phi_units_label}", UNIT_OPTIONS, index=UNIT_OPTIONS.index("fractions"), key=f"{prefix_key}_unit_phi",)

    optional_cols = [NOT_USED] + cols

    def _opt_idx(keywords):
        hit = _guess_col(cols, keywords)
        return optional_cols.index(hit) if hit is not None else 0

    horizon_col = st.sidebar.selectbox(f"{prefix_label}horizon_id", optional_cols,
                                       index=_opt_idx(["horizon_id", "horizon"]), key=f"{prefix_key}_horizon",)
    zone_col = st.sidebar.selectbox(f"{prefix_label}tectonic_zone", optional_cols,
                                    index=_opt_idx(["tectonic_zone", "tectonic", "zone"]), key=f"{prefix_key}_zone",)

    detected = {
        "well_id": _guess_col(cols, ["well_id", "well"]),
        "X_coord": _guess_col(cols, ["x"]),
        "Y_coord": _guess_col(cols, ["y"]),
        "horizon_id": _guess_col(cols, ["horizon_id", "horizon"]),
        "tectonic_zone": _guess_col(cols, ["tectonic_zone", "tectonic", "zone"]),
    }
    detected_text = ", ".join(f"{k_} ← {v}" for k_, v in detected.items() if v is not None)
    if detected_text:
        st.sidebar.caption(f"{T['auto_detect_caption']} {detected_text}")

    return (well_col, x_col, y_col, h_col, phi_col, unit_h, unit_phi, horizon_col, zone_col)

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
        demo_well_col, demo_x_col, demo_y_col, demo_h_col, demo_phi_col, demo_unit_h, demo_unit_phi, demo_horizon_col, demo_zone_col
        ) = show_column_mapping_sidebar(demo_cols, prefix_label=demo_prefix_label, prefix_key="demo",)

        # Подготовка данных
        df_demo, dropped_demo, removed_demo_rows = prepare_input_dataframe(
            raw_demo,
            demo_well_col,
            demo_x_col,
            demo_y_col,
            demo_h_col,
            demo_phi_col,
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
            demo_unit_h,
            demo_unit_phi,
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

        (well_col, x_col, y_col, h_col, phi_col, unit_h, unit_phi, horizon_col, zone_col) = show_column_mapping_sidebar(cols, prefix_label="", prefix_key="user",)

        # Подготовка данных
        df_user, dropped, removed_rows = prepare_input_dataframe(
            raw_df,
            well_col,
            x_col,
            y_col,
            h_col,
            phi_col,
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
            unit_h,
            unit_phi,
            None if zone_col == NOT_USED else zone_col,
        )
    except Exception as e:
        st.sidebar.error(f"{file_loading_error} {e}")

# ------------------------------------------------------------- #
# Текущие параметры проекта: PARAM
# ------------------------------------------------------------- #

df = st.session_state["df"]

EXCLUDED_ATTR_COLS = {"well_id", "X_coord", "Y_coord", "Zprm_well", "Zprm_map", "H_eff", "Phi"}
if df is not None:
    ATTR_CANDIDATES = [c for c in df.select_dtypes(include=np.number).columns if c not in EXCLUDED_ATTR_COLS]
else:
    ATTR_CANDIDATES = []


step1_crossplot_title = T["step1_crossplot_title"]
step1_qc_metrics_title = T["step1_qc_metrics_title"]
step1_approximations_title = T["step1_approximations_title"]
step1_approx_comparison_title = T["step1_approx_comparison_title"]

_units_by_param = st.session_state.get("UNITS_BY_PARAM", {})
HEFF_LABEL = f"H_eff, {_units_by_param.get('H_eff', 'm')}"
PHI_LABEL = f"Phi, {_units_by_param.get('Phi', 'fractions')}"

def render_qc_pair(x_ref, y_ref, x_name, y_name, x_label, y_label, key_prefix, file_prefix):
    """Полный набор Шага 1 для одной пары: кросс-плот, метрики QC, аппроксимации, таблица аппроксимаций."""
    x_arr = np.asarray(x_ref, dtype=float)
    y_arr = np.asarray(y_ref, dtype=float)
    basic_pair = sq.basic_crossplot_stats_xy(x_arr, y_arr)
    z_line_pair = np.linspace(x_arr.min(), x_arr.max(), 300)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(f"#### {step1_crossplot_title}: {y_name} vs {x_name}")
        fig_qc, coeffs_qc = build_qc_crossplot(
            df=df, x_ref=x_arr, y_ref=y_arr,
            title=f"QC cross-plot: {y_name} vs {x_name}",
            x_label=x_label, y_label=y_label,
            point_color=crossplot_point_color,
            regression_color=qc_regression_color,
            identity_color=identity_line_color,
            identity_width=identity_line_width,
            identity_dash=identity_line_dash,
            wells_label=wells_legend, well_label=well_label,
            regression_name=regression_qc_label,
            regression_width=qc_regression_width,
            regression_dash=qc_regression_dash,
            marker_size=crossplot_point_size,
            marker_opacity=crossplot_point_opacity,
        )
        st.plotly_chart(fig_qc, use_container_width=True)

    qc_metrics_pair = build_qc_metrics_table(basic=basic_pair, coeffs_qc=coeffs_qc,
                                             left_name=y_name, right_name=x_name)
    st.markdown(f"#### {step1_qc_metrics_title}: {y_name} vs {x_name}")
    st.dataframe(qc_metrics_pair, use_container_width=True)
    st.caption(T["delta_help_step1"])
    excel_download_button(qc_metrics_pair, f"{file_prefix}_metrics",
                          key=f"download_excel_{key_prefix}_metrics", index=True)

    with col_right:
        st.markdown(f"#### {step1_approximations_title}: {y_name} vs {x_name}")
        fig_fit, fit_rows_pair = build_approximation_plot(
            df=df, x_ref=x_arr, y_ref=y_arr,
            z_line=z_line_pair, approx_types=approx_types,
            title=f"QC cross-plot approx: {y_name} vs {x_name}",
            x_label=x_label, y_label=y_label,
            x_name=x_name, y_name=y_name,
            point_color=approx_point_color,
            identity_color=identity_line_color,
            wells_label=wells_legend, well_label=well_label,
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
            show_quantile_band=True,
        )
        st.plotly_chart(fig_fit, use_container_width=True)

    st.markdown(f"#### {step1_approx_comparison_title}: {y_name} vs {x_name}")
    fit_table_pair = pd.DataFrame(fit_rows_pair)
    st.dataframe(fit_table_pair, use_container_width=True)
    excel_download_button(fit_table_pair, f"{file_prefix}_approx_comparison",
                          key=f"download_excel_{key_prefix}_approx")

def render_corr_block(fixed_cols, block_title, key_suffix, pick_attrs=True):
    st.markdown(f"#### {block_title}")

    if pick_attrs:
        picked_attrs = st.multiselect(
            corr_pick_attrs_label,
            options=ATTR_CANDIDATES,
            default=ATTR_CANDIDATES,
            key=f"corr_attrs_{key_suffix}",
        )
        corr_columns = list(picked_attrs) + list(fixed_cols)
    else:
        corr_columns = list(fixed_cols)

    if len(corr_columns) < 2:
        st.info(step0_corr_need_two_msg)
        return
    if len(corr_columns) > 10:
        st.warning(step0_pairplot_warning)

    # Матрица корреляции Пирсона
    st.markdown(f"#### {step0_corr_matrix_title}")
    corr_df = df[corr_columns].corr()
    corr_df = corr_df.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)
    fig_corr = px.imshow(
        corr_df, text_auto=".2f", color_continuous_scale=corr_palette,
        zmin=-1, zmax=1, aspect="auto",
        title=f"{step0_corr_matrix_plot_title} {len(corr_columns)}",
    )
    fig_corr.update_coloraxes(colorbar_title="Correlation<br>coefficient")
    fig_corr.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10))
    excel_download_button(corr_df, "attribute_qc_corr_matrix_pearson",
                          key=f"download_excel_corr_matrix_pearson_{key_suffix}", index=True)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.caption(step0_corr_hint)
    st.markdown(f"#### {corr_pairs_title.format(threshold=f'{corr_threshold:.2f}')}")
    pairs_pearson = corr_pairs_table(corr_df, corr_threshold, corr_pair_labels)
    if pairs_pearson.empty:
        st.caption(corr_pairs_empty)
    else:
        st.dataframe(pairs_pearson, use_container_width=True)
        excel_download_button(pairs_pearson, "attribute_qc_corr_pairs_pearson",
                              key=f"download_excel_corr_pairs_pearson_{key_suffix}")

    # Матрица корреляции Спирмена
    st.markdown(f"#### {step0_corr_spearman_title}")
    corr_df_spearman = df[corr_columns].corr(method="spearman")
    corr_df_spearman = corr_df_spearman.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)
    fig_corr_sp = px.imshow(
        corr_df_spearman, text_auto=".2f", color_continuous_scale=corr_palette,
        zmin=-1, zmax=1, aspect="auto",
        title=f"{step0_corr_spearman_title}: {len(corr_columns)}",
    )
    fig_corr_sp.update_coloraxes(colorbar_title="Spearman<br>ρ")
    fig_corr_sp.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10))
    excel_download_button(corr_df_spearman, "attribute_qc_corr_matrix_spearman",
                          key=f"download_excel_corr_matrix_spearman_{key_suffix}", index=True)
    st.plotly_chart(fig_corr_sp, use_container_width=True)
    st.markdown(f"#### {corr_pairs_title.format(threshold=f'{corr_threshold:.2f}')}")
    pairs_spearman = corr_pairs_table(corr_df_spearman, corr_threshold, corr_pair_labels)
    if pairs_spearman.empty:
        st.caption(corr_pairs_empty)
    else:
        st.dataframe(pairs_spearman, use_container_width=True)
        excel_download_button(pairs_spearman, "attribute_qc_corr_pairs_spearman",
                              key=f"download_excel_corr_pairs_spearman_{key_suffix}")

    render_pairplot_block(corr_columns, step0_pairplot_title, key_suffix)


def render_pairplot_block(cols, block_title, key_suffix):
    st.markdown(f"#### {block_title}")
    if len(cols) < 2:
        st.info(step0_corr_need_two_msg)
        return
    if len(cols) > 10:
        st.warning(step0_pairplot_warning)

    show_pairplot = st.toggle(T["show_pairplot_label"], value=False, key=f"show_pairplot_{key_suffix}")
    if show_pairplot and pairplot_engine == "Seaborn Pairplot (static)":
        fig_pair = plot_pairplot(
            df=df, columns=cols, diag_kind=diag_kind, height=pairplot_size, corner=pairplot_corner,
            point_color=pairplot_point_color, point_size=pairplot_point_size, point_opacity=pairplot_point_opacity,
            hist_color=pairplot_hist_color, kde_color=pairplot_kde_color, scale=pairplot_scale,
            category_col=pairplot_category,
            category_colors=pairplot_category_colors if pairplot_category_colors else None,
            kde_linewidth=pairplot_kde_width, kde_linestyle=pairplot_kde_linestyle,
            hist_bins=pairplot_hist_bins,
            hist_outline_color=pairplot_hist_outline_color,
            hist_outline_width=pairplot_hist_outline_width_eff,
        )
        st.pyplot(fig_pair, use_container_width=True)
    elif show_pairplot and pairplot_engine == "Plotly Pairplot (interactive)":
        st.caption(step0_plotly_pairplot_info)

        extra = [c for c in cols if c != "well_id"]
        cols_for_pair = ["well_id"] + extra
        use_category_for_color = (
            pairplot_category is not None
            and pairplot_category in df.columns
            and pairplot_category not in cols_for_pair
        )
        if use_category_for_color:
            cols_for_pair = cols_for_pair + [pairplot_category]

        pair_source = df[cols_for_pair].copy()
        if use_category_for_color:
            pair_source[pairplot_category] = (
                pair_source[pairplot_category].fillna("(not specified)").astype(str)
            )
        pair_df = pair_source.dropna()

        if len(pair_df.columns) != len(set(pair_df.columns)):
            counts = {}
            new_cols = []
            for col in pair_df.columns:
                counts[col] = counts.get(col, 0) + 1
                new_cols.append(col if counts[col] == 1 else f"{col} ({counts[col]})")
            pair_df.columns = new_cols

        pair_df = pair_df.rename(columns=DISPLAY_NAMES)

        excluded_from_dims = {"well_id"}
        cat_display = (
            DISPLAY_NAMES.get(pairplot_category, pairplot_category)
            if use_category_for_color else None
        )
        if cat_display is not None:
            excluded_from_dims.add(cat_display)
        dimensions = [c for c in pair_df.columns if c not in excluded_from_dims]

        color_map = (
            {str(k): v for k, v in pairplot_category_colors.items()}
            if (use_category_for_color and pairplot_category_colors) else None
        )

        fig_pair = cached_plotly_pairplot(
            pair_df, dimensions, cat_display, color_map,
            pairplot_point_color, pairplot_point_size, pairplot_point_opacity,
            f"{step0_plotly_pairplot_title} ({len(cols)})",
        )
        st.plotly_chart(fig_pair, use_container_width=True)

def render_corr_overview():
    st.markdown(f"#### {T['corr_overview_title']}")

    picked_attrs = st.multiselect(
        corr_pick_attrs_label,
        options=ATTR_CANDIDATES,
        default=ATTR_CANDIDATES,
        key="corr_attrs_overview",
    )
    cols = list(picked_attrs) + ["H_eff", "Phi"]
    if len(cols) < 3:
        st.info(step0_corr_need_two_msg)
        return

    corr_p = df[cols].corr().rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)
    corr_s = df[cols].corr(method="spearman").rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)

    st.markdown(f"##### {step0_corr_matrix_title}")
    fig_p = px.imshow(corr_p, text_auto=".2f", color_continuous_scale=corr_palette,
                      zmin=-1, zmax=1, aspect="auto")
    fig_p.update_coloraxes(colorbar_title="Correlation<br>coefficient")
    fig_p.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10))
    excel_download_button(corr_p, "attribute_qc_corr_matrix_pearson",
                          key="download_excel_overview_pearson", index=True)
    st.plotly_chart(fig_p, use_container_width=True)

    st.markdown(f"##### {step0_corr_spearman_title}")
    fig_s = px.imshow(corr_s, text_auto=".2f", color_continuous_scale=corr_palette,
                      zmin=-1, zmax=1, aspect="auto")
    fig_s.update_coloraxes(colorbar_title="Spearman<br>ρ")
    fig_s.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10))
    excel_download_button(corr_s, "attribute_qc_corr_matrix_spearman",
                          key="download_excel_overview_spearman", index=True)
    st.plotly_chart(fig_s, use_container_width=True)

    st.caption(step0_corr_hint)
    st.markdown(f"#### {corr_pairs_title.format(threshold=f'{corr_threshold:.2f}')}")
    labels = (
        T["corr_group_label"],
        corr_pair_labels[0],
        corr_pair_labels[1],
        corr_pair_labels[2],
        T["corr_spearman_col_label"],
    )
    group_names = (
        T["corr_group_attr_attr"],
        T["corr_group_heff_attr"],
        T["corr_group_phi_attr"],
        T["corr_group_heff_phi"],
    )
    pairs = corr_pairs_grouped(corr_p, corr_s, corr_threshold, labels, group_names)
    if pairs.empty:
        st.caption(corr_pairs_empty)
    else:
        st.dataframe(pairs, use_container_width=True)
        excel_download_button(pairs, "attribute_qc_corr_pairs",
                              key="download_excel_overview_pairs")

    st.markdown(f"#### {T['corr_pairs_weak_title'].format(threshold=f'{corr_weak_threshold:.2f}')}")
    pairs_weak = corr_pairs_grouped(corr_p, corr_s, corr_weak_threshold, labels, group_names, below=True)
    if pairs_weak.empty:
        st.caption(corr_pairs_empty)
    else:
        st.dataframe(pairs_weak, use_container_width=True)
        excel_download_button(pairs_weak, "attribute_qc_corr_pairs_weak",
                              key="download_excel_overview_pairs_weak")

PARAM_1 = st.session_state.get("step1_param1", "H_eff")
PARAM_2 = st.session_state.get("step1_param2", ATTR_CANDIDATES[0] if ATTR_CANDIDATES else "PARAM_2")
PARAM_1_UNIT = st.session_state.get("UNITS_BY_PARAM", {}).get(PARAM_1, "units")
PARAM_2_UNIT = st.session_state.get("step1_unit2", "units")

def add_zprm(d, p1, p2):
    if d is None:
        return None
    out = d.copy()
    out["Zprm_well"] = d[p1]
    out["Zprm_map"] = d[p2]
    return out

df = add_zprm(df, PARAM_1, PARAM_2)

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
    - [{contents_data_pr}](#step-0-pr)
    - [{contents_data_res}](#step-0-res)
    - [{contents_data_cat}](#step-0-cat)
    - [{contents_data_num}](#step-0-num)
- [{contents_step1}](#step-1)
    - [{contents_step1_cp}](#step-1-cp)
    - [{contents_step1_hb}](#step-1-hb)
- [{contents_step2}](#step-2)
    - [{contents_step2_bcv}](#step-2-bcv)
    - [{contents_step2_cp}](#step-2-cp)
    - [{contents_step2_res}](#step-2-res) 
    - [{contents_step2_otl}](#step-2-otl)
- [{contents_step3}](#step-3)
    - [{contents_step3_pb}](#step-3-pb)
    - [{contents_step3_cp}](#step-3-cp)
- [{contents_step4}](#step-4)
- [{contents_step5}](#step-5)
    - [{contents_step5_heff}](#step-5-heff)
        - [{contents_step5_lowo}](#step-5-heff-lowo)
        - [{contents_step5_kfold}](#step-5-heff-kfold)
    - [{contents_step5_phi}](#step-5-phi)
        - [{contents_step5_lowo}](#step-5-phi-lowo)
        - [{contents_step5_kfold}](#step-5-phi-kfold)
""")

# ------------------------------------------------------------- #
# SIDEBAR: Горизонт
# ------------------------------------------------------------- #
st.sidebar.markdown("---")

def _on_picked_horizons_change(all_options):
    picked = st.session_state.get("sidebar_picked_horizons", [])
    st.session_state["sidebar_all_horizons"] = set(picked) == set(all_options)

def _on_all_horizons_toggle(all_options):
    # галочка включена — выбираем все горизонты, выключена — оставляем первый
    if st.session_state.get("sidebar_all_horizons"):
        st.session_state["sidebar_picked_horizons"] = list(all_options)
    else:
        st.session_state["sidebar_picked_horizons"] = list(all_options[:1])

with st.sidebar.expander(horizon_expander_title, expanded=False):

    if (df is not None and has_col(df, "horizon_id") and df["horizon_id"].nunique() > 1):

        horizons = sorted(df["horizon_id"].dropna().unique().tolist())

        # первичная инициализация (без default= у мультиселекта)
        if "sidebar_picked_horizons" not in st.session_state:
            st.session_state["sidebar_picked_horizons"] = horizons[:1]

        all_horizons = st.checkbox(
            all_horizons_label,
            value=False,
            key="sidebar_all_horizons",
            on_change=_on_all_horizons_toggle,
            args=(horizons,),
        )
        picked_horizons = st.multiselect(
            horizons_label,
            options=horizons,
            key="sidebar_picked_horizons",
            on_change=_on_picked_horizons_change,
            args=(horizons,),

        )

        st.caption(
            f"{selected_horizons_label} "
            f"{len(picked_horizons)} из {len(horizons)}"
        )

        if len(picked_horizons) == 0:
            st.warning(select_horizon_warning)
            st.stop()

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
    cache_sig = (
        PARAM_1,
        PARAM_2,
        len(df),
        tuple(df["well_id"].astype(str).unique()),
    )
    if (
        st.session_state.get("df_for_cv") is None
        or st.session_state.get("df_for_cv_sig") != cache_sig
    ):
        st.session_state["df_for_cv"] = df.copy()
        st.session_state["df_for_cv_sig"] = cache_sig
else:
    st.session_state["df_for_cv"] = None
    st.session_state["df_for_cv_sig"] = None

# ------------------------------------------------------------- #
# SIDEBAR: Цвета графиков
# ------------------------------------------------------------- #
st.sidebar.markdown("---")

ALL_COLORS = [
    "black", "blue", "brown", "cadetblue", "chocolate", "coral", "cornflowerblue", "crimson","cyan",
    "darkblue", "darkcyan", "darkgoldenrod", "darkgreen", "darkmagenta", "darkorange", "darkorchid", "darkslateblue", "darkslategray",
    "darkred", "deepskyblue", "dodgerblue","firebrick", "forestgreen", "gold", "goldenrod", "gray", "green", "indianred",
    "indigo", "limegreen","lightgray", "lightblue", "lightgreen", "magenta", "mediumblue", "mediumorchid", "mediumseagreen", "midnightblue","navy", "olive", "orange",
    "orangered", "peru", "purple", "red", "royalblue", "saddlebrown", "seagreen", "slateblue", "skyblue",
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

# Круговая диаграмма
with st.sidebar.expander(pie_chart_expander, expanded=False):
    pie_palette = st.selectbox(pie_palette_label, CATEGORY_PALETTES, index=CATEGORY_PALETTES.index("Plotly"), key="pie_palette")
    pie_opacity = st.slider(multi_pie_opacity_label, 0.1, 1.0, 1.0, 0.05, key="pie_opacity")
    st.markdown(f"**{multi_pie_outline_label}**")
    st.caption(hist_line_width_label)
    pie_outline_width = st.slider(hist_line_width_label, 0, 5, 0, key="pie_outline_width", label_visibility="collapsed")
    pie_outline_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("black"), key="pie_outline_color", label_visibility="collapsed")

# Столбчатая диаграмма
def bar_style_controls(prefix, group_title, palette_label, default_orientation="v"):
    """Настройки одного столбчатого графика: палитра, направление, граница столбцов."""
    st.markdown(f"**{group_title}**")
    palette = st.selectbox(palette_label, CATEGORY_PALETTES,
                           index=CATEGORY_PALETTES.index("Plotly"), key=f"{prefix}_palette")
    orientation = st.radio(
        T["bar_orientation_label"], ["v", "h"],
        index=0 if default_orientation == "v" else 1, horizontal=True,
        format_func=lambda x: T["bar_orient_v"] if x == "v" else T["bar_orient_h"],
        key=f"{prefix}_orientation",
    )
    st.markdown(f"*{T['bar_outline_label']}*")
    st.caption(hist_line_width_label)
    outline_width = st.slider(hist_line_width_label, 0, 5, 0,
                              key=f"{prefix}_outline_width", label_visibility="collapsed")
    outline_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("black"),
                                 key=f"{prefix}_outline_color", label_visibility="collapsed")
    
    opacity = st.slider(multi_pie_opacity_label, 0.1, 1.0, 1.0, 0.05,
                        key=f"{prefix}_opacity")
    st.markdown("---")
    return dict(palette=palette, orientation=orientation,
                outline_color=outline_color, outline_width=outline_width,
                opacity=opacity)

crosstab_value_options = [
    c for c in df.select_dtypes(include=np.number).columns
    if c not in ("well_id", "X_coord", "Y_coord")
] if df is not None else []

with st.sidebar.expander(bar_chart_expander, expanded=False):
    # Что считать на столбцах для категориальных признаков (Загрузка данных)
    bar_load_agg = st.selectbox(
        crosstab_agg_label,
        ["count", "min", "max", "median", "mean", "std"],
        format_func=lambda x: T[f"crosstab_agg_{x}"],
        index=0, key="bar_load_agg",
    )
    bar_load_value_col = None
    if bar_load_agg != "count" and crosstab_value_options:
        bar_load_value_col = st.selectbox(
            crosstab_value_label, crosstab_value_options,
            format_func=display_name, key="bar_load_value_col",
        )
    style_load = bar_style_controls("bar_load", bar_group_load_title, category_bar_palette_label, "h")
    style_fold = bar_style_controls("bar_fold", bar_group_fold_title, rmse_palette_label, "v")
    style_rmse = bar_style_controls("bar_rmse", bar_group_rmse_title, rmse_palette_label, "v")

    
    # Что считать на столбцах для категориальных признаков (Шаг 3)
    step3_bar_agg = st.selectbox(
        crosstab_agg_label,
        ["count", "min", "max", "median", "mean", "std"],
        format_func=lambda x: T[f"crosstab_agg_{x}"],
        index=0, key="step3_bar_agg",
    )
    step3_bar_value_options = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in ("well_id", "X_coord", "Y_coord")
    ] if df is not None else []
    step3_bar_value_col = None
    if step3_bar_agg != "count" and step3_bar_value_options:
        step3_bar_value_col = st.selectbox(
            crosstab_value_label, step3_bar_value_options,
            format_func=display_name, key="step3_bar_value_col",
        )
    style_step3 = bar_style_controls("bar_step3", bar_group_step3_title, category_bar_palette_label, "h")              # Plotly
    style_step5 = bar_style_controls("bar_step5", bar_group_step5_title, category_bar_palette_label, "v")

# Круговые диаграммы
with st.sidebar.expander(multi_pie_expander, expanded=False):
    multi_pie_palette = st.selectbox(category_bar_palette_label, CATEGORY_PALETTES, index=CATEGORY_PALETTES.index("Plotly"), key="multi_pie_palette",)
    multi_pie_opacity = st.slider(multi_pie_opacity_label, 0.1, 1.0, 1.0, 0.05, key="multi_pie_opacity",)
    st.markdown(f"**{multi_pie_outline_label}**")
    st.caption(hist_line_width_label)
    multi_pie_outline_width = st.slider(hist_line_width_label, 0, 5, 0, key="multi_pie_outline_width",label_visibility="collapsed",)
    multi_pie_outline_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("black"), key="multi_pie_outline_color", label_visibility="collapsed",)

# Столбчатые диаграммы
with st.sidebar.expander(crosstab_expander, expanded=False):
    crosstab_palette = st.selectbox(category_bar_palette_label, CATEGORY_PALETTES, index=CATEGORY_PALETTES.index("Plotly"), key="crosstab_palette",)
    crosstab_agg = st.selectbox(
        crosstab_agg_label,
        ["count", "min", "max", "median", "mean", "std"],
        format_func=lambda x: T[f"crosstab_agg_{x}"],
        index=0, key="crosstab_agg",
    )

    crosstab_value_col = None
    if crosstab_agg != "count" and crosstab_value_options:
        crosstab_value_col = st.selectbox(crosstab_value_label, crosstab_value_options, format_func=display_name, key="crosstab_value_col",)

    crosstab_orientation = st.radio(
        crosstab_orientation_label, ["v", "h"], index=0, horizontal=True,
        format_func=lambda x: crosstab_orient_v if x == "v" else crosstab_orient_h,
        key="crosstab_orientation",
    )
    crosstab_opacity = st.slider(multi_pie_opacity_label, 0.1, 1.0, 1.0, 0.05, key="crosstab_opacity",)
    st.markdown(f"**{crosstab_outline_label}**")
    st.caption(hist_line_width_label)
    crosstab_outline_width = st.slider(hist_line_width_label, 0, 5, 0, key="crosstab_outline_width", label_visibility="collapsed")
    crosstab_outline_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("black"), key="crosstab_outline_color", label_visibility="collapsed")

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

    if "tectonic_zone" in pairplot_category_candidates:
        default_pairplot_category = "tectonic_zone"
    elif zone_col_default is not None and zone_col_default in pairplot_category_candidates:
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

        if diag_kind == "kde":
            st.caption(hist_line_width_label)
            pairplot_kde_width = st.slider(hist_line_width_label, 1.0, 10.0, 1.5, 0.5, key="pairplot_kde_width", label_visibility="collapsed",)
            pairplot_kde_linestyle = st.selectbox(hist_line_style_label, DASH_OPTIONS, index=DASH_OPTIONS.index("solid"), key="pairplot_kde_linestyle",)
            pairplot_hist_bins = 30
            pairplot_hist_outline_color = "black"
            pairplot_hist_outline_width_eff = 0
        else:
            pairplot_kde_width = 1.5
            pairplot_kde_linestyle = "solid"
            pairplot_hist_bins = st.slider(hist_bins_label, 5, 120, 30, key="pairplot_hist_bins",)

            st.markdown(f"**{hist_outline_label}**")
            pairplot_hist_outline_on = st.checkbox(hist_outline_on_label, value=False, key="pairplot_hist_outline_on")
            pairplot_hist_outline_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("black"), key="pairplot_hist_outline_color",
                                                       label_visibility="collapsed", disabled=not pairplot_hist_outline_on)
            st.caption(hist_line_width_label)
            pairplot_hist_outline_width = st.slider(hist_line_width_label, 0, 5, 1,
                                                    key="pairplot_hist_outline_width",
                                                    label_visibility="collapsed",
                                                    disabled=not pairplot_hist_outline_on)
            pairplot_hist_outline_width_eff = (pairplot_hist_outline_width if pairplot_hist_outline_on else 0)

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

# Гисторгамма
with st.sidebar.expander(histogram_expander, expanded=False):

    # Заливка гистограммы — только цвет
    hist_color = st.selectbox(hist_fill_color_label, ALL_COLORS, index=ALL_COLORS.index("orange"), key="hist_color")
    # ---------- Прозрачность ----------
    hist_opacity = st.slider(hist_opacity_label, 0.1, 1.0, 0.85, 0.05, key="hist_opacity")

    # ---------- Границы блоков ----------
    st.markdown(f"**{hist_outline_label}**")
    hist_outline_on = st.checkbox(hist_outline_on_label, value=False, key="hist_outline_on")
    hist_outline_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("black"), key="hist_outline_color", label_visibility="collapsed", disabled=not hist_outline_on,)
    st.caption(hist_line_width_label)
    hist_outline_width = st.slider(hist_line_width_label, 0, 5, 1, key="hist_outline_width", label_visibility="collapsed", disabled=not hist_outline_on,)
    # итоговая толщина обводки: 0 означает «без обводки» (вариант как сейчас)
    hist_outline_width_eff = hist_outline_width if hist_outline_on else 0

    st.markdown("---")

    # ---------- KDE ----------
    st.markdown(f"**{kde_curve_label}**")
    kde_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("black"), key="kde_color", label_visibility="collapsed",)
    kde_dash = st.selectbox(hist_line_style_label, DASH_OPTIONS, index=DASH_OPTIONS.index("solid"), key="kde_dash", label_visibility="collapsed",)
    st.caption(hist_line_width_label)
    kde_width = st.slider(hist_line_width_label, 1, 10, 3, key="kde_width", label_visibility="collapsed",)

    # ---------- Mean ----------
    st.markdown(f"**{mean_line_label}**")
    mean_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("green"), key="mean_color", label_visibility="collapsed",)
    mean_dash = st.selectbox(hist_line_style_label, DASH_OPTIONS, index=DASH_OPTIONS.index("dash"), key="mean_dash", label_visibility="collapsed",)
    st.caption(hist_line_width_label)
    mean_width = st.slider(hist_line_width_label, 1, 10, 2, key="mean_width", label_visibility="collapsed",)

    # ---------- Median ----------
    st.markdown(f"**{median_line_label}**")
    median_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("red"), key="median_color", label_visibility="collapsed",)
    median_dash = st.selectbox(hist_line_style_label, DASH_OPTIONS, index=DASH_OPTIONS.index("dot"), key="median_dash", label_visibility="collapsed",)
    st.caption(hist_line_width_label)
    median_width = st.slider(hist_line_width_label, 1, 10, 2, key="median_width", label_visibility="collapsed",)

    # ---------- ±3σ ----------
    st.markdown(f"**{sigma_line_label}**")
    sigma_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("orange"), key="sigma_color", label_visibility="collapsed",)
    sigma_dash = st.selectbox(hist_line_style_label, DASH_OPTIONS, index=DASH_OPTIONS.index("dash"), key="sigma_dash", label_visibility="collapsed",)
    st.caption(hist_line_width_label)
    sigma_width = st.slider(hist_line_width_label, 1, 10, 2, key="sigma_width", label_visibility="collapsed",)
    sigma_k = st.slider(sigma_k_label, 1, 3, 3, key="sigma_k")

# Boxplot
with st.sidebar.expander(boxplot_expander, expanded=False):

    # Заливка боксплота — только цвет
    box_color = st.selectbox(box_fill_color_label, ALL_COLORS, index=ALL_COLORS.index("royalblue"), key="box_color")
    box_opacity = st.slider(box_opacity_label, 0.1, 1.0, 0.6, 0.05, key="box_opacity")

    st.markdown("---")
    box_orientation = st.radio(bar_orientation_label, ["v", "h"], index=0, horizontal=True,
    format_func=lambda x: bar_orient_v if x == "v" else bar_orient_h, key="box_orientation",)

    st.markdown(f"**{box_outline_label}**")
    st.caption(hist_line_width_label)
    box_outline_width = st.slider(hist_line_width_label, 0, 6, 2, key="box_outline_width", label_visibility="collapsed",)
    box_outline_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("royalblue"), key="box_outline_color", label_visibility="collapsed",)

    st.markdown("---")

    # ---------- Mean ----------
    st.markdown(f"**{box_mean_label}**")
    box_mean_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("green"), key="box_mean_color", label_visibility="collapsed",)
    box_mean_dash = st.selectbox(hist_line_style_label, DASH_OPTIONS, index=DASH_OPTIONS.index("dash"), key="box_mean_dash", label_visibility="collapsed",)
    st.caption(hist_line_width_label)
    box_mean_width = st.slider(hist_line_width_label, 1, 10, 2, key="box_mean_width", label_visibility="collapsed",)

    # ---------- Median ----------
    st.markdown(f"**{box_median_label}**")
    box_median_color = st.selectbox("Color", ALL_COLORS, index=ALL_COLORS.index("red"), key="box_median_color", label_visibility="collapsed",)
    box_median_dash = st.selectbox(hist_line_style_label, DASH_OPTIONS, index=DASH_OPTIONS.index("dot"), key="box_median_dash", label_visibility="collapsed",)
    st.caption(hist_line_width_label)
    box_median_width = st.slider(hist_line_width_label, 1, 10, 2, key="box_median_width", label_visibility="collapsed",)

    # общие стили боксплота: передаются во все вызовы plot_box
    BOX_STYLE = dict(
        box_opacity=box_opacity,
        box_outline_color=box_outline_color,
        box_outline_width=box_outline_width,
        box_orientation=box_orientation,
    )


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


with st.sidebar.expander(residual_expander, expanded=False):
    residual_color = st.selectbox(residual_color_label, ALL_COLORS, index=ALL_COLORS.index("black"), key="residual_color")       # black
    residual_point_size = st.slider(point_size_label, 3, 20, 8, key="residual_point_size",)
    residual_point_opacity = st.slider(point_opacity_label, 0.1, 1.0, 0.85, 0.05, key="residual_point_opacity",)

with st.sidebar.expander(residual_map_expander, expanded=False):
    map_palette = st.selectbox(residual_map_palette_label, MAP_PALETTES, index=MAP_PALETTES.index("RdBu"), key="map_palette")                   # RdBu
    residual_map_point_size = st.slider(residual_map_point_size_label, 3, 30, 12,  key="residual_map_point_size",)
    residual_map_point_opacity = st.slider(residual_map_point_opacity_label, 0.1, 1.0, 0.9, 0.05, key="residual_map_point_opacity",)

    if df is not None:
        residual_map_category_options = [
            c for c in df.columns
            if c not in ("Zprm_well", "Zprm_map")
            and not pd.api.types.is_numeric_dtype(df[c])
            and c != "well_id"
        ]
    else:
        residual_map_category_options = []

    residual_map_options_all = [None] + residual_map_category_options
    rm_default = "tectonic_zone" if "tectonic_zone" in residual_map_category_options else None
    rm_sig = tuple(residual_map_options_all)
    if st.session_state.get("_rm_cat_sig") != rm_sig:
        st.session_state["residual_map_category"] = rm_default
        st.session_state["_rm_cat_sig"] = rm_sig

    residual_map_category = st.selectbox(residual_map_category_label, residual_map_options_all,
        format_func=lambda x: residual_map_no_category if x is None else display_name(x),
        key="residual_map_category",)

    if residual_map_category is not None and df is not None:
        highlight_options = sorted(
            df[residual_map_category].fillna("(not specified)").astype(str).unique()
        )
    else:
        highlight_options = []

    residual_map_highlight = st.selectbox(residual_map_highlight_label, [None] + highlight_options,
        format_func=lambda x: residual_map_no_highlight if x is None else x,
        index=0, key=f"residual_map_highlight_{residual_map_category}", disabled=not highlight_options,)
    residual_map_highlight_opacity = st.slider(residual_map_highlight_opacity_label, 0.1, 1.0, 1.0, 0.05, key="residual_map_highlight_opacity",)
    residual_map_dim_opacity = st.slider(residual_map_dim_opacity_label, 0.05, 1.0, 0.15, 0.05, key="residual_map_dim_opacity",)


# ------------------------------------------------- #
# SIDEBAR: STEP 0 - корреляционный анализ
# ------------------------------------------------- #
st.sidebar.markdown("---")

with st.sidebar.expander(correlation_expander, expanded=False):
    corr_threshold = st.slider(
        corr_threshold_label,
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        key="corr_threshold",
    )
    corr_weak_threshold = st.slider(
        T["corr_weak_threshold_label"],
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="corr_weak_threshold",
    )
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
st.sidebar.markdown(f"### {approx_group_title}")
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
        zone_approx_types = APPROX_OPTIONS
    else:
        zone_approx_types = st.multiselect(
            approx_types_step3_label,
            APPROX_OPTIONS,
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
        st.caption(hist_line_width_label)
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
        st.caption(hist_line_width_label)
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
# SIDEBAR: Предполагаемые выбросы в шагах 1 и 2
# ------------------------------------------------- #
st.sidebar.markdown("---")
with st.sidebar.expander(outlier_expander, expanded=False):
    # ------ рекомендуемый Порог выброса, σ ------
    sigma_thr = st.slider(sigma_threshold_label, 1.0, 6.0, 2.0, 0.1, key="sigma_thr",)

    iqr_k = st.slider(outlier_iqr_k_label, 1.0, 3.0, 1.5, 0.1, key="iqr_k",)
    pct_low = st.slider(outlier_pct_low_label, 0.0, 20.0, 5.0, 0.5, key="pct_low",)
    pct_high = st.slider(outlier_pct_high_label, 80.0, 100.0, 95.0, 0.5, key="pct_high",)

# ------------------------------------------------- #
# SIDEBAR: STEP 2 - параметры валидации CV 
# ------------------------------------------------- #
# Анализ расстояний между скважинами для рекомендаций по CV
@st.cache_data(show_spinner=False, max_entries=64)
def cached_spacing_stats(coords):
    """Статистика расстояний между скважинами по массиву координат (N×2)."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2:
        return {
            "median_dist": None,
            "mean_dist": None,
            "min_dist": None,
            "max_dist": None,
            "n_wells": n,
        }

    dists = pdist(coords)
    return {
        "median_dist": float(np.median(dists)),
        "mean_dist": float(np.mean(dists)),
        "min_dist": float(np.min(dists)),
        "max_dist": float(np.max(dists)),
        "n_wells": n,
    }


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
    return cached_spacing_stats(coords)

spacing = compute_spacing_stats(df)
median_dist = spacing["median_dist"]
mean_dist = spacing["mean_dist"]
min_dist = spacing["min_dist"]
max_dist = spacing["max_dist"]
n_wells = spacing["n_wells"]

st.sidebar.markdown("---")
with st.sidebar.expander(cv_validation_expander, expanded=False):
    # ------ Random seed (влияет на Random K-Fold, Leave-P-Out, Spatial Leave-P-Out) ------
    st.markdown(f"#### {cv_seed_title}")

    if "cv_seed" not in st.session_state:
        st.session_state["cv_seed"] = 42

    st.markdown(f"**{cv_seed_current}** `{st.session_state['cv_seed']}`")

    if st.button(cv_seed_new_button, key="cv_new_seed", use_container_width=True):
        st.session_state["cv_seed"] += 1
    if st.button(cv_seed_reset_button, key="cv_reset_seed", use_container_width=True):
        st.session_state["cv_seed"] = 42

    st.caption(cv_seed_caption)

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
        st.markdown(f"#### {buffer_cv_title}")
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
    st.markdown(f"#### {leave_p_sp_out_title}")
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
    st.markdown(f"#### {leave_p_out_title}")
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

    # ------ Stratified by Zone: выбор колонки ------
    st.markdown(f"#### {T['strat_zone_col_label']}")
    strat_zone_col_options = [None]
    if df is not None:
        for c in df.columns:
            if c in ("well_id", "X_coord", "Y_coord", "Zprm_well", "Zprm_map"):
                continue
            # категориальные ИЛИ числовые с небольшим числом уникальных (horizon_id)
            if (
                not pd.api.types.is_numeric_dtype(df[c])
                or df[c].nunique(dropna=True) <= 20
            ):
                strat_zone_col_options.append(c)

    # Умный default: tectonic_zone, если есть; иначе horizon_id; иначе None.
    # Пересчитываем, когда меняется набор доступных колонок (новый файл или новый маппинг).
    strat_default = (
        "tectonic_zone" if "tectonic_zone" in strat_zone_col_options
        else "horizon_id" if "horizon_id" in strat_zone_col_options
        else None
    )
    strat_sig = tuple(strat_zone_col_options)
    if st.session_state.get("_strat_opts_sig") != strat_sig:
        st.session_state["strat_zone_col"] = strat_default
        st.session_state["_strat_opts_sig"] = strat_sig

    # если выбранного ранее значения больше нет в списке — сброс
    if st.session_state["strat_zone_col"] not in strat_zone_col_options:
        st.session_state["strat_zone_col"] = None

    strat_zone_col = st.selectbox(
        strat_zone_col_label,
        strat_zone_col_options,
        format_func=lambda x: T["not_used_label"] if x is None else display_name(x),
        key="strat_zone_col",
    )

    # --- предупреждение: колонка выбрана, но в ней < 2 уникальных значений ---
    if strat_zone_col is not None and df is not None:
        n_unique = df[strat_zone_col].astype(str).nunique()
        if n_unique < 2:
            st.caption(
                strat_zone_col_single_value_hint.format(
                    col=strat_zone_col, n=n_unique,
                )
            )
    elif strat_zone_col is None:
        st.caption(strat_zone_col_none_hint)

    # ------ Distance-based exclusion CV ------
    st.markdown(f"#### {distance_cv_title}")

    # --- выбор способа задания радиуса ---
    excl_radius_mode = st.radio(
        excl_radius_mode_label,
        ["median_factor", "absolute"],
        format_func=lambda x: (
            excl_radius_mode_median_factor if x == "median_factor"
            else excl_radius_mode_absolute
        ),
        index=0,
        horizontal=True,
        key="excl_radius_mode",
    )

    # --- параметры в зависимости от режима ---
    if excl_radius_mode == "median_factor":
        variogram_factor = st.slider(
            variogram_factor_label, 0.25, 3.00, 1.00, 0.25,
            key="variogram_factor",
        )
        st.caption(variogram_factor_caption)

        # абсолютный радиус не используется — передадим None
        excl_radius_abs = None
    else:
        # режим absolute: множитель не нужен, оставляем дефолт (для совместимости с кэшем)
        variogram_factor = 1.0

        # рекомендуемый диапазон — как в Spatial Leave-P-Out (по медианному расстоянию)
        if median_dist is not None and median_dist > 0:
            abs_min = max(1, int(min_dist))
            abs_max = max(abs_min + 1, int(max_dist))
            abs_default = max(abs_min, min(int(median_dist), abs_max))
        else:
            abs_min, abs_max, abs_default = 1, 5000, 500

        excl_radius_abs = st.slider(
            excl_radius_abs_label,
            abs_min, abs_max, abs_default,
            key="excl_radius_abs",
        )
        st.caption(excl_radius_abs_caption)

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

    default_index = 0
    if "tectonic_zone" in category_candidates:
        default_index = category_candidates.index("tectonic_zone")
    else:
        default_category = st.session_state.get("zone_col")
        if default_category is not None and default_category in category_candidates:
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

        # Seed для случайных выборок (шаг 4) — инициализируется независимо от режима фильтрации
        if "step4_random_seed" not in st.session_state:
            st.session_state.step4_random_seed = 42

        filter_mode = st.radio(
            filter_mode_label,
            [FILTER_NONE, FILTER_MANUAL, FILTER_RANDOM, FILTER_CLICK],
            format_func=lambda x: filter_mode_labels[x],
            index=[FILTER_NONE, FILTER_MANUAL, FILTER_RANDOM, FILTER_CLICK,].index(st.session_state.step4_filter_mode), key="step4_filter_mode",)
        st.caption(
            f"{current_filter_mode} "
            f"{filter_mode_labels[filter_mode]}"
        )

        # Seed общий для случайного исключения и серии шага 4.5
        if st.button(new_random_set_button, key="step4_new_random_set", use_container_width=True):
            st.session_state.step4_random_seed += 1
        if st.button(step4_random_seed_reset_button, key="step4_reset_random_set", use_container_width=True):
            st.session_state.step4_random_seed = 42
        st.caption(f"{current_random_set} {st.session_state.step4_random_seed}")

        # Число случайных наборов для серии шага 4.5
        st.slider(
            step4_n_sets_label,
            min_value=50, max_value=2000, value=500, step=50,
            key="step4_n_sets",
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

            st.caption(random_exclusion_caption)

            if percent_exclude > 0:
                n_wells_all = df["well_id"].nunique()
                n_excluded = max(1, int(n_wells_all * percent_exclude / 100))
                n_excluded = min(n_excluded, n_wells_all - 1)
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

    # SIDEBAR: STEP 5 — выбор атрибутов для многоатрибутной регрессии
    step5_sidebar_expander = T["step5_sidebar_expander"]
    step5_mode_label = T["step5_mode_label"]
    step5_mode_auto = T["step5_mode_auto"]
    step5_mode_manual = T["step5_mode_manual"]
    step5_manual_label = T["step5_manual_label"]
    step5_manual_empty = T["step5_manual_empty"]
    st.sidebar.markdown("---")
    with st.sidebar.expander(step5_sidebar_expander, expanded=False):
        step5_mode = st.radio(
            step5_mode_label,
            ["auto", "manual"],
            format_func=lambda x: step5_mode_auto if x == "auto" else step5_mode_manual,
            key="step5_attr_mode",
        )
        if step5_mode == "manual":
            if "step5_manual_attrs" not in st.session_state:
                st.session_state["step5_manual_attrs"] = list(ATTR_CANDIDATES)
            st.session_state["step5_manual_attrs"] = [
                c for c in st.session_state["step5_manual_attrs"] if c in ATTR_CANDIDATES
            ]
            st.multiselect(step5_manual_label, options=ATTR_CANDIDATES, key="step5_manual_attrs")
            if not st.session_state["step5_manual_attrs"]:
                st.warning(step5_manual_empty)

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
step0_loaded_data_pr = T["step0_loaded_data_pr"]
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
corr_pair_labels = (
        T["corr_pairs_col_param1"],
        T["corr_pairs_col_param2"],
        T["corr_pairs_col_cc"],
    )
corr_pairs_title = T["corr_pairs_title"]
hist_norm_label = T["hist_norm_label"]
hist_title = T["hist_title"]
boxplot_title = T["boxplot_title"]
corr_pairs_empty = T["corr_pairs_empty"]

def back_to_data_load():
    st.markdown(f'<a href="#data-load">{T["back_to_data_load_label"]}</a>', unsafe_allow_html=True)

st.markdown('<div id="data-load"></div>', unsafe_allow_html=True)

st.markdown('<div id="step-0"></div>', unsafe_allow_html=True)

# ------------------------------------------------- #
# STEP 0.0.: SHOW SAMPLE FORMAT
# ------------------------------------------------- #
st.title(step0_title)
st.caption("CROSSPL✳T↔CROSS⊗VALIDATION")
st.caption(T["footer_data_note"])
st.caption(T["footer_copyright"])

ui_examples.show_project_introduction()
ui_examples.show_input_file_example()

st.markdown('<div id="step-0-pr"></div>', unsafe_allow_html=True)
st.markdown("---")
# ------------------------------------------------- #
# STEP 0.1.: QC входных данных
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
st.markdown(f"### {step0_loaded_data_pr}")

# QC входных данных
with st.expander(f"✅ {step0_input_qc_title}",  expanded=False):

    # Полные дубликаты строк
    full_duplicates = df[df.duplicated(keep=False)]
    st.markdown(f"#### {step0_full_duplicates_title}")
    if len(full_duplicates) > 0:
        st.warning(
            f"{step0_found_full_duplicates} "
            f"{len(full_duplicates)}"
        ) 
        st.dataframe(full_duplicates, use_container_width=True)
        excel_download_button(full_duplicates, "attribute_qc_dup_rows", key="download_excel_dup_rows")
    else:
        st.caption(step0_full_duplicates_notfound)

    # Дубликаты скважин
    st.markdown(f"#### {step0_duplicate_wells_title}")
    st.caption(step0_duplicate_wells_hint)
    dup_wells = (df.groupby("well_id").size().reset_index(name="count"))
    dup_wells = dup_wells[dup_wells["count"] > 1]
    if len(dup_wells) > 0:
        st.warning(f"{step0_duplicate_wells_found}"
            f"{len(dup_wells)}"
        )
        st.dataframe(dup_wells, use_container_width=True)
        excel_download_button(dup_wells, "attribute_qc_dup_wells", key="download_excel_dup_wells")
    else:
        st.caption(step0_duplicate_wells_notfound)

    # Совпадающие координаты (в пределах одного горизонта)
    st.markdown(f"#### {step0_duplicate_coords_title}")

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
        excel_download_button(dup_coords, "attribute_qc_dup_coords", key="download_excel_dup_coords")
    else:
        st.caption(step0_duplicate_coords_notfound)

    # Пропуски
    st.markdown(f"#### {step0_missing_values_title}")
    na_summary = pd.DataFrame({
        "column": [display_name(c) for c in cols_for_display],
        "missing values": [int(df[c].isna().sum()) for c in cols_for_display],
    })
    st.dataframe(na_summary, use_container_width=True)
    excel_download_button(na_summary, "attribute_qc_missing_values", key="download_excel_na_summary")

    # Строки удалённые при очистке
    st.markdown(f"#### {step0_removed_rows_title}")
    removed_rows = st.session_state.get("removed_rows", pd.DataFrame() )
    if len(removed_rows) > 0:
        st.warning(
            f"{step0_removed_rows_count}"
            f"{len(removed_rows)}"
        )
        removed_display = removed_rows.drop(columns=["Zprm_well", "Zprm_map"], errors="ignore")
        st.dataframe(removed_display.head(20), use_container_width=True)
        excel_download_button(removed_display, "attribute_qc_removed_rows", key="download_excel_removed_rows")
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
# STEP 0.2.: Типы данных по колонкам
# ------------------------------------------------- #
st.markdown(f"#### {step0_data_types_title}")

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
excel_download_button(dtype_df, "attribute_qc_data_type", key="download_excel_data_type")

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
st.markdown(f"#### {step0_first_rows_title}")
st.dataframe(display_df.head(5), use_container_width=True)

st.markdown(f"#### {step0_last_rows_title}")
st.dataframe(display_df.tail(5), use_container_width=True)
excel_download_button(display_df, "attribute_qc_data", key="download_excel_display")

st.markdown('<div id="step-0-res"></div>', unsafe_allow_html=True)
st.markdown(f"### {T['step0_reservoir_params_title']}")

# --- Статистика параметров
st.markdown(
    f"#### {step0_parameter_stats_title} "
    f"H_eff, Phi"
)

col_desc1, col_desc2 = st.columns(2)
with col_desc1:
    stats_well_df = show_depth_stats(df["H_eff"], "H_eff")
with col_desc2:
    stats_map_df = show_depth_stats(df["Phi"], "Phi")

if stats_well_df is not None and stats_map_df is not None:
    stats_export_df = pd.concat([stats_well_df, stats_map_df], axis=1)
    excel_download_button(stats_export_df, "attribute_qc_parameter_stats",
                          key="download_excel_param_stats", index=True)

st.markdown(f"### {T['heff_phi_crossplot_title']}")
render_qc_pair(x_ref=df["Phi"].to_numpy(float), y_ref=df["H_eff"].to_numpy(float),
               x_name="Phi", y_name="H_eff",
               x_label=PHI_LABEL, y_label=HEFF_LABEL,
               key_prefix="heffphi", file_prefix="attribute_qc_heffphi")

render_corr_block(["H_eff", "Phi"], T["corr_block_heffphi_title"], "heffphi", pick_attrs=False)

# ------------------------------------------------- #
# STEP 0.3.: Категориальные параметры
# ------------------------------------------------- #
st.markdown('<div id="step-0-cat"></div>', unsafe_allow_html=True)
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
                opacity=pie_opacity,
                outline_color=pie_outline_color,
                outline_width=pie_outline_width,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            if bar_load_agg == "count" or bar_load_value_col is None:
                # число уникальных скважин в каждой категории
                bar_series = df.groupby(col)["well_id"].nunique().sort_index()
                bar_y_label = well_count_label
                bar_title = f"{step0_well_count_title} : {display_name(col)}"
                bar_show_percent = True
            else:
                vals_local = pd.to_numeric(df[bar_load_value_col], errors="coerce")
                bar_series = (
                    getattr(vals_local.groupby(df[col]), bar_load_agg)()
                    .dropna().sort_index()
                )
                bar_y_label = f"{T['crosstab_agg_' + bar_load_agg]}, {display_name(bar_load_value_col)}"
                bar_title = f"{T['crosstab_agg_' + bar_load_agg]} {display_name(bar_load_value_col)} : {display_name(col)}"
                bar_show_percent = False
            fig_bar = plot_bar_chart(
                x=bar_series.index,
                y=bar_series.values,
                **style_load,
                title=bar_title,
                yaxis_title=bar_y_label,
                show_percent=bar_show_percent,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Таблица статистики по категории: число скважин, записей и доля скважин
        st.markdown(f"**{crosstab_table_title}: {display_name(col)}**")
        cat_stats_table = category_stats_table(
            df, col,
            T["cat_stats_wells"], T["cat_stats_records"], T["cat_stats_share"],
            display_name(col),
        )
        st.dataframe(cat_stats_table, use_container_width=True, hide_index=True)
        excel_download_button(cat_stats_table, f"attribute_qc_category_{col}",
                              key=f"download_excel_cat_stats_{col}")

        # ---- Таблица агрегаций по категории ----
        well_counts = df.groupby(col)["well_id"].nunique().sort_index()

        agg_table = well_counts.to_frame(name=well_count_label)

        if bar_load_value_col is not None:
            vals_all = pd.to_numeric(df[bar_load_value_col], errors="coerce")
            agg_extra = (
                vals_all.groupby(df[col])
                        .agg(["min", "max", "median", "mean", "std"])
                        .sort_index()
            )
            agg_table = agg_table.join(agg_extra)
            table_title = f"{T['crosstab_table_title']}: {display_name(bar_load_value_col)}"
        else:
            table_title = f"{T['crosstab_table_title']}: {display_name(col)}"

        agg_table = agg_table.rename_axis(display_name(col)).reset_index()

        st.markdown(f"**{table_title}**")
        st.dataframe(agg_table, use_container_width=True, hide_index=True)
        excel_download_button(agg_table, f"attribute_qc_bar_aggs_{col}",
                              key=f"download_excel_bar_aggs_{col}")

# Круговые диаграммы по нескольким категориям (перед сгруппированной)
# ------------------------------------------------- #

# Сброс выбора осей и групп при смене набора категорий (другой файл)
_cat_sig = tuple(category_cols)
if st.session_state.get("_cat_pies_sig") != _cat_sig:
    for _k in ["pies_x_col", "pies_group_col", "crosstab_x_col", "crosstab_group_col"]:
        st.session_state.pop(_k, None)
    st.session_state["_cat_pies_sig"] = _cat_sig

if len(category_cols) >= 2:
    st.markdown(f"#### {step0_multi_pie_title}")

    col_pie_x, col_pie_g = st.columns(2)
    pies_x_default = category_cols.index("tectonic_zone") if "tectonic_zone" in category_cols else 0

    with col_pie_x:
        pies_x_col = st.selectbox(
            step0_crosstab_x_label, category_cols,
            index=pies_x_default,
            format_func=display_name, key="pies_x_col",
        )
    pies_group_options = sorted(
        [c for c in category_cols if c != pies_x_col],
        key=lambda c: display_name(c).lower(),
    )
    with col_pie_g:
        pies_group_col = st.selectbox(
            step0_crosstab_group_label,
            pies_group_options,
            index=0,
            format_func=display_name, key="pies_group_col",
        )

    fig_pies = plot_category_pies(
        df=df,
        x_col=pies_x_col,
        group_col=pies_group_col,
        title=f"{step0_multi_pie_title}: {display_name(pies_x_col)} × {display_name(pies_group_col)}",
        x_label=display_name(pies_x_col),
        group_label=display_name(pies_group_col),
        palette=multi_pie_palette,
        opacity=multi_pie_opacity,
        outline_color=multi_pie_outline_color,
        outline_width=multi_pie_outline_width,
    )
    st.plotly_chart(fig_pies, use_container_width=True)

# Сгруппированная диаграмма по двум категориям
# ------------------------------------------------- #
if category_cols:
    st.markdown(f"#### {step0_crosstab_title}")
    if len(category_cols) < 2:
        st.caption(step0_crosstab_need_two)
    else:
        col_x_sel, col_g_sel = st.columns(2)
        crosstab_x_default = category_cols.index("tectonic_zone") if "tectonic_zone" in category_cols else 0
        with col_x_sel:
            crosstab_x_col = st.selectbox(
                step0_crosstab_x_label, category_cols,
                index=crosstab_x_default,
                format_func=display_name, key="crosstab_x_col",
            )
        group_options = sorted(
            [c for c in category_cols if c != crosstab_x_col],
            key=lambda c: display_name(c).lower(),
        )
        with col_g_sel:
            crosstab_group_col = st.selectbox(
                step0_crosstab_group_label, group_options,
                index=0,
                format_func=display_name, key="crosstab_group_col",
            )

        if crosstab_agg == "count" or crosstab_value_col is None:
            crosstab_y_label = well_count_label
        else:
            crosstab_y_label = f"{T['crosstab_agg_' + crosstab_agg]}, {display_name(crosstab_value_col)}"

        fig_crosstab = plot_category_crosstab(
            df=df,
            x_col=crosstab_x_col,
            group_col=crosstab_group_col,
            title=f"{step0_crosstab_title}: {display_name(crosstab_x_col)} × {display_name(crosstab_group_col)}",
            x_label=display_name(crosstab_x_col),
            y_label=crosstab_y_label,
            legend_title=display_name(crosstab_group_col),
            palette=crosstab_palette,
            orientation=crosstab_orientation,
            outline_color=crosstab_outline_color,
            outline_width=crosstab_outline_width,
            agg=crosstab_agg,
            value_col=crosstab_value_col,
            opacity=crosstab_opacity,
        )
        st.plotly_chart(fig_crosstab, use_container_width=True)

        # ---- Общий заголовок для обеих таблиц ----
        st.markdown(f"#### {crosstab_table_title}")

        # ---- 1. Таблица к мульти-пирогам (не зависит от crosstab_agg) ----
        st.markdown(
            f"**{T['step0_multi_pie_title']}: "
            f"{display_name(pies_x_col)} × {display_name(pies_group_col)}**"
        )
        table_pies = crosstab_stats_table(
            df=df,
            x_col=pies_x_col,
            group_col=pies_group_col,
            x_label=display_name(pies_x_col),
            group_label=display_name(pies_group_col),
            agg="count",
            value_col=None,
            total_label=crosstab_table_total,
        )
        st.dataframe(table_pies, use_container_width=True)
        excel_download_button(table_pies, "attribute_qc_crosstab_pies",
                              key="download_excel_crosstab_pies", index=True)

        # ---- 2. Таблица к столбчатой диаграмме (зависит от crosstab_agg / crosstab_value_col) ----
        if crosstab_agg == "count" or crosstab_value_col is None:
            bar_agg_label = well_count_label
        else:
            bar_agg_label = f"{T['crosstab_agg_' + crosstab_agg]}, {display_name(crosstab_value_col)}"

        st.markdown(
            f"**{step0_crosstab_title}: "
            f"{display_name(crosstab_x_col)} × {display_name(crosstab_group_col)} "
            f"({bar_agg_label})**"
        )
        table_crosstab = crosstab_stats_table(
            df=df,
            x_col=crosstab_x_col,
            group_col=crosstab_group_col,
            x_label=display_name(crosstab_x_col),
            group_label=display_name(crosstab_group_col),
            agg=crosstab_agg,
            value_col=crosstab_value_col,
            total_label=crosstab_table_total,
        )
        st.dataframe(table_crosstab, use_container_width=True)
        excel_download_button(table_crosstab, "attribute_qc_crosstab",
                              key="download_excel_crosstab", index=True)

# ------------------------------------------------- #
# STEP 0.4.: Корреляционный анализ выбранных параметров
# ------------------------------------------------- #
st.markdown('<div id="step-0-num"></div>', unsafe_allow_html=True)
st.markdown(f"### {step0_selected_params_title}")

render_corr_overview()
render_pairplot_block(ATTR_CANDIDATES + ["H_eff"], T["corr_pairplot_heff_title"], "heff")
render_pairplot_block(ATTR_CANDIDATES + ["Phi"], T["corr_pairplot_phi_title"], "phi")

# Статистика, гистограмма и boxplot по всем числовым параметрам (стиль графиков и элементы управления — как в Шаге 1, отдельно для каждого параметра)
numeric_cols = [
    c for c in df.select_dtypes(include="number").columns
    if c not in ("well_id", "Zprm_well", "Zprm_map")
]

# Словарь подписей нормировки (в Шаге 1 он объявлен ниже, поэтому здесь свой)
hist_norm_map_step0 = {
    "none": T["hist_norm_none"],
    "density": T["hist_norm_density"],
    "probability": T["hist_norm_probability"],
    "percent": T["hist_norm_percent"],
}

st.markdown(f"#### {step0_selected_stats_title}")

for col in numeric_cols:
    display_col = display_name(col)
    values = df[col].to_numpy(float)

    with st.expander(f"{step0_stats_prefix}: {display_col}", expanded=False):

        # Элементы управления гистограммой — свои для каждого параметра
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            bins_step0 = st.slider(
                f"{hist_bins_label} {display_col}", 10, 120, 40,
                key=f"bins_step0_{col}",
            )
        with col_ctrl2:
            histnorm_step0 = st.selectbox(
                f"{hist_norm_label} {display_col}",
                ["none", "density", "probability", "percent"],
                format_func=lambda x: hist_norm_map_step0[x],
                index=0, key=f"histnorm_step0_{col}",
            )
        histnorm_value_step0 = "" if histnorm_step0 == "none" else histnorm_step0

        col_stats, col_plots = st.columns([1, 2])

        with col_stats:
            stats_col_df = show_depth_stats(df[col], display_col)
            if stats_col_df is not None:
                excel_download_button(stats_col_df, f"attribute_qc_stats_{col}",
                                      key=f"download_excel_stats_{col}", index=True)

        with col_plots:
            fig_hist = plot_hist_kde(
                data=values,
                bins=bins_step0,
                histnorm=histnorm_value_step0,
                color=hist_color,
                title=f"{hist_title} {display_col}",
                xlabel=display_col,
                show_sigma=True,
                sigma_k=sigma_k,
                opacity=hist_opacity,
                hist_outline_color=hist_outline_color,
                hist_outline_width=hist_outline_width_eff,
                kde_color=kde_color,       kde_dash=kde_dash,       kde_width=kde_width,
                mean_color=mean_color,     mean_dash=mean_dash,     mean_width=mean_width,
                median_color=median_color, median_dash=median_dash, median_width=median_width,
                sigma_color=sigma_color,   sigma_dash=sigma_dash,   sigma_width=sigma_width,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_plots:
            fig_box = plot_box(
                data=values,
                color=box_color,
                ylabel=display_col,
                well_label=well_label,
                title=f"{boxplot_title} {display_col}",
                well_ids=df["well_id"],
                value_label=display_col,
                mean_color=box_mean_color,     mean_dash=box_mean_dash,     mean_width=box_mean_width,
                median_color=box_median_color, median_dash=box_median_dash, median_width=box_median_width,
                **BOX_STYLE,
            )
            st.plotly_chart(fig_box, use_container_width=True)

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
hist_norm_map = {
    "none": T["hist_norm_none"],
    "density": T["hist_norm_density"],
    "probability": T["hist_norm_probability"],
    "percent": T["hist_norm_percent"],
}
step1_data_hint = T["step1_data_hint"]

back_to_data_load()
st.markdown('<div id="step-1"></div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    f"## {step1_title} "
    f"{PARAM_1} vs {PARAM_2}"
)

# ---------------------------------------------------------------------- #
# STEP 1.0
# ---------------------------------------------------------------------- #

st.markdown(f"### {step1_select_title}")
if len(ATTR_CANDIDATES) == 0:
    st.warning(step1_no_attributes_msg)
    st.stop()
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.selectbox(param1_reference_label, ["H_eff", "Phi"], key="step1_param1")
with col_s2:
    st.selectbox(param2_reference_label, ATTR_CANDIDATES, key="step1_param2")
with col_s3:
    st.selectbox(param2_units_label, UNIT_OPTIONS, index=UNIT_OPTIONS.index("units"), key="step1_unit2")

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
st.markdown(f"### {contents_step1_cp}")
render_qc_pair(x_ref=x_ref, y_ref=y_ref, x_name=PARAM_2, y_name=PARAM_1,
               x_label=PARAM_2_LABEL, y_label=PARAM_1_LABEL,
               key_prefix="step1", file_prefix="attribute_qc_step1")

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
    st.markdown(f"#### {hist_title} {PARAM_1}")
    fig_hw = plot_hist_kde(
        data=y_ref,
        bins=bins_step1,
        histnorm=histnorm_value,
        color=hist_color,
        title=f"{hist_title} {PARAM_1}",
        xlabel=PARAM_1_LABEL,
        show_sigma=True,
        sigma_k=sigma_k,
        opacity=hist_opacity,
        hist_outline_color=hist_outline_color,
        hist_outline_width=hist_outline_width_eff,
        kde_color=kde_color,       kde_dash=kde_dash,       kde_width=kde_width,
        mean_color=mean_color,     mean_dash=mean_dash,     mean_width=mean_width,
        median_color=median_color, median_dash=median_dash, median_width=median_width,
        sigma_color=sigma_color,   sigma_dash=sigma_dash,   sigma_width=sigma_width,
    )
    st.plotly_chart(fig_hw, use_container_width=True)
with col_h2:
    st.markdown(f"#### {hist_title} {PARAM_2}")
    fig_hm = plot_hist_kde(
        data=x_ref,
        bins=bins_step1,
        histnorm=histnorm_value,
        color=hist_color,
        title=f"{hist_title} {PARAM_2}",
        xlabel=PARAM_2_LABEL,
        show_sigma=True,
        sigma_k=sigma_k,
        opacity=hist_opacity,
        hist_outline_color=hist_outline_color,
        hist_outline_width=hist_outline_width_eff,
        kde_color=kde_color,       kde_dash=kde_dash,       kde_width=kde_width,
        mean_color=mean_color,     mean_dash=mean_dash,     mean_width=mean_width,
        median_color=median_color, median_dash=median_dash, median_width=median_width,
        sigma_color=sigma_color,   sigma_dash=sigma_dash,   sigma_width=sigma_width,
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# Boxplot в шаге 1
col_b1, col_b2 = st.columns(2)
with col_b1:
    st.markdown(f"#### {boxplot_title} {PARAM_1}")
    fig_bw = plot_box(
        y_ref, box_color, PARAM_1_LABEL, well_label,
        title=f"{boxplot_title} {PARAM_1}",
        well_ids=df["well_id"], value_label=PARAM_1_LABEL,
        mean_color=box_mean_color,     mean_dash=box_mean_dash,     mean_width=box_mean_width,
        median_color=box_median_color, median_dash=box_median_dash, median_width=box_median_width,
        **BOX_STYLE,
    )
    st.plotly_chart(fig_bw, use_container_width=True)
with col_b2:
    st.markdown(f"#### {boxplot_title} {PARAM_2}")
    fig_bm = plot_box(
        x_ref, box_color, PARAM_2_LABEL, well_label,
        title=f"{boxplot_title} {PARAM_2}",
        well_ids=df["well_id"], value_label=PARAM_2_LABEL,
        mean_color=box_mean_color,     mean_dash=box_mean_dash,     mean_width=box_mean_width,
        median_color=box_median_color, median_dash=box_median_dash, median_width=box_median_width,
        **BOX_STYLE,
    )
    st.plotly_chart(fig_bm, use_container_width=True)

# ================================================================================================================================ #
# ---------------------------------------------   Шаг 2: Кросс-валидация и таблицы метрик ------------------------------------------
# ================================================================================================================================ #
# --------------
# STEP 2 LABELS
# ---------------
step2_title = T["step2_title"]
cv_not_enough_data_msg = T["cv_not_enough_data_msg"]
cv_best_scheme_error_msg = T["cv_best_scheme_error_msg"]
cv_npred_mismatch_warning = T["cv_npred_mismatch_warning"]
cv_pearson_r2_col_label = T["cv_pearson_r2_col_label"]
cv_r_col_label = T["cv_r_col_label"]
cv_r2_col_label = T["cv_r2_col_label"]
step2_crossplot_all_title = T["step2_crossplot_all_title"]
step2_crossplot_all_caption = T["step2_crossplot_all_caption"]
step2_metrics_all_title = T["step2_metrics_all_title"]
step2_cv_comparison_title = T["step2_cv_comparison_title"]
step2_fold_r2_table_title = T["step2_fold_r2_table_title"]
step2_fold_r2_boxplot_title = T["step2_fold_r2_boxplot_title"]
step2_fold_r2_nan_note = T["step2_fold_r2_nan_note"]
step2_rmse_title = T["step2_rmse_title"]
step2_fold_rmse_title = T["step2_fold_rmse_title"]
step2_fold_rmse_boxplot_title = T["step2_fold_rmse_boxplot_title"]
step2_fold_table_title = T["step2_fold_table_title"]
step2_fold_table_empty = T["step2_fold_table_empty"]
step2_fold_boxplot_empty = T["step2_fold_boxplot_empty"]
step2_boxplot_caption = T["step2_boxplot_caption"]
best_scheme_criterion_label = T["best_scheme_criterion_label"]
best_scheme_criterion_rmse_plus_std = T["best_scheme_criterion_rmse_plus_std"]
best_scheme_criterion_rmse = T["best_scheme_criterion_rmse"]
best_scheme_criterion_mae = T["best_scheme_criterion_mae"]
best_scheme_criterion_r2 = T["best_scheme_criterion_r2"]
best_scheme_criterion_rmse_plus_abs_bias = T["best_scheme_criterion_rmse_plus_abs_bias"]
best_scheme_criterion_manual = T["best_scheme_criterion_manual"]
best_scheme_manual_label = T["best_scheme_manual_label"]
best_scheme_score_caption = T["best_scheme_score_caption"]
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

back_to_data_load()
st.markdown('<div id="step-2"></div>', unsafe_allow_html=True)
st.markdown('<div id="step-2-bcv"></div>', unsafe_allow_html=True)
st.markdown("---")
# ------------------------------------------------- #
# STEP 2.0.: 
# ------------------------------------------------- #

@st.cache_data(
    show_spinner="Running cross-validation...",
    hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df, index=True).values.tobytes()},
)
def cached_run_all_cv(df, degree, k, grid_n, buffer, variogram_factor,
                       leave_p_percent, spatial_radius, cv_repeats,
                       strat_zone_col,
                       excl_radius_mode, excl_radius_abs,
                       seed):                                              
    return sq.run_all_cv(
        df, degree=degree, k=k, grid_n=grid_n, buffer=buffer,
        variogram_factor=variogram_factor, leave_p_percent=leave_p_percent,
        spatial_radius=spatial_radius, cv_repeats=cv_repeats,
        seed=seed,                                                     
        strat_zone_col=strat_zone_col,
        excl_radius_mode=excl_radius_mode,
        excl_radius_abs=excl_radius_abs,
    )

@st.cache_data(show_spinner=False)
def cached_random_exclusion_series(df_in, levels, n_repeats, seed, min_remaining):
    return sq.random_exclusion_series(
        df_in,
        levels=levels,
        n_repeats=n_repeats,
        seed=seed,
        min_remaining=min_remaining,
    )

st.markdown(
    f"## {step2_title}: "
    f"{PARAM_1} vs {PARAM_2}"
)

st.markdown(f"### {contents_step2_bcv}")

if st.session_state.get("df_for_cv") is None:
    st.info(cv_not_enough_data_msg)
    st.stop()

results, comparison = cached_run_all_cv(
    st.session_state["df_for_cv"], 1, k_folds, grid_n, buffer,
    variogram_factor, leave_p_percent, spatial_radius, cv_repeats,
    strat_zone_col,
    excl_radius_mode, excl_radius_abs,
    st.session_state["cv_seed"],                                      
)

if comparison.empty:
    st.warning(cv_not_enough_data_msg)
    st.stop()

# --- Если схема Stratified не построилась (нет колонки / один класс) ---
for key in list(results.keys()):
    if key.startswith("Stratified by ") and results[key].get("preds") is None:
        comparison = comparison.drop(index=key, errors="ignore")
        results.pop(key, None)

# --- Предупреждение: схемы CV покрывают разное число точек ---
if comparison["n_pred"].nunique() > 1:
    st.warning(cv_npred_mismatch_warning.format(
        n_min=int(comparison["n_pred"].min()),
        n_max=int(comparison["n_pred"].max()),
    ))

# --- Выбор лучшей схемы CV: критерий + ручной override ---
st.markdown(f"#### {best_scheme_criterion_label}")

criterion_options = [
    "rmse_plus_std",
    "rmse",
    "mae",
    "r2",
    "rmse_plus_abs_bias",
    "manual",
]

criterion_labels = {
    "rmse_plus_std": best_scheme_criterion_rmse_plus_std,
    "rmse": best_scheme_criterion_rmse,
    "mae": best_scheme_criterion_mae,
    "r2": best_scheme_criterion_r2,
    "rmse_plus_abs_bias": best_scheme_criterion_rmse_plus_abs_bias,
    "manual": best_scheme_criterion_manual,
}

if "best_scheme_criterion" not in st.session_state:
    st.session_state["best_scheme_criterion"] = "rmse_plus_std"

criterion = st.selectbox(
    best_scheme_criterion_label,
    criterion_options,
    format_func=lambda x: criterion_labels[x],
    index=criterion_options.index(st.session_state["best_scheme_criterion"]),
    key="best_scheme_criterion",
    label_visibility="collapsed",
)

# --- Считаем score по выбранному критерию ---
# Все критерии приводим к виду "чем меньше — тем лучше"
comparison["score"] = np.nan

if criterion == "rmse_plus_std":
    comparison["score"] = comparison["rmse"] + comparison["rmse_fold_std"]
elif criterion == "rmse":
    comparison["score"] = comparison["rmse"]
elif criterion == "mae":
    comparison["score"] = comparison["mae"]
elif criterion == "r2":
    comparison["score"] = -comparison["r2_actual_pred"]           # инверсия: max R² → min −R²
elif criterion == "rmse_plus_abs_bias":
    comparison["score"] = comparison["rmse"] + comparison["bias"].abs()

# --- Ручной выбор override ---
if criterion == "manual":
    st.session_state.setdefault("best_scheme_manual", comparison.index[0])

    # если сохранённое значение больше не в списке — сброс
    if st.session_state["best_scheme_manual"] not in comparison.index:
        st.session_state["best_scheme_manual"] = comparison.index[0]

    best_scheme = st.selectbox(
        best_scheme_manual_label,
        list(comparison.index),
        index=list(comparison.index).index(st.session_state["best_scheme_manual"]),
        key="best_scheme_manual",
    )
else:
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
st.markdown(f"#### {step2_crossplot_all_title}")
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
            stats_tmp.bias = -stats_tmp.bias   # Bias = Predicted − Actual
            eq_tmp = (
                f"Predicted = "
                f"{coeffs_tmp[0]:.5f}·Actual + "
                f"{coeffs_tmp[1]:.5f}"
            )
            # Расчёт метрик кросс-плота для Шага 2
            cv_row = {
                (cv_pearson_r2_col_label if k == "R²_lin" else k): v
                for k, v in basicstats_to_dict(stats_tmp, eq_tmp, r2_label=cv_pearson_r2_col_label, r_label=cv_r_col_label).items()
            }
            cv_summary_rows.append({
                "CV scheme": scheme_name,
                **cv_row,
            })

# --- Таблица по всем схемам CV
st.markdown(f"#### {step2_metrics_all_title}")
cv_summary_df = pd.DataFrame(cv_summary_rows)
st.dataframe(cv_summary_df, use_container_width=True)
excel_download_button(cv_summary_df, "attribute_qc_step2_cv_metrics_all",
                      key="download_excel_step2_cv_metrics_all")

# --- caption-связка: score пересчитан по выбранному критерию ---
st.caption(
    best_scheme_score_caption.format(
        criterion=criterion_labels[st.session_state["best_scheme_criterion"]]
    )
)

summary_df = pd.DataFrame({
    name: results[name]["summary"] for name in comparison.index
}).T

# r (Pearson, Predicted vs Actual) уже посчитан в summarize_cv (r_pooled)
summary_df["r_pearson_pooled"] = summary_df["r_pooled"]

r2_table_cols = [
    "n_pred", "n_test_min", "n_test_max",
    "r2_pearson_pooled", "r_pearson_pooled",
    "r2_fold_min", "r2_fold_max", "r2_fold_std",
    "r2_fold_mean", "r2_fold_median",
]
r2_table = summary_df[r2_table_cols].rename(columns={
    "r2_pearson_pooled": "R² (Pearson, Predicted vs Actual)",
    "r_pearson_pooled":  "r (Pearson, Predicted vs Actual)",
})
r2_table = r2_table.rename_axis("CV scheme").reset_index()

st.markdown(f"#### {step2_fold_r2_table_title}")
st.dataframe(
    r2_table.style.format({
        "n_pred": "{:.0f}",
        "n_test_min": "{:.0f}",
        "n_test_max": "{:.0f}",
        **{c: "{:.4f}" for c in [
            "r2_fold_min", "r2_fold_max", "r2_fold_std",
            "r2_fold_mean", "r2_fold_median",
        ]},
        "R² (Pearson, Predicted vs Actual)": "{:.4f}",
        "r (Pearson, Predicted vs Actual)": "{:.4f}",
    }),
    use_container_width=True,
)
excel_download_button(r2_table, "attribute_qc_step2_cv_r2_folds",
                      key="download_excel_step2_cv_r2_folds", index=True)

rows_r2 = []
for scheme_name in comparison.index:
    fold_df = results[scheme_name].get("folds")
    if fold_df is None or len(fold_df) == 0 or "r2" not in fold_df.columns:
        continue
    for fold_idx, v in enumerate(fold_df["r2"].dropna().to_numpy(float), start=1):
        rows_r2.append({
            "cv_scheme": scheme_name,
            "fold_r2": v,
            "well_id": f"{scheme_name} · фолд {fold_idx}",
        })
df_fold_r2 = pd.DataFrame(rows_r2)

if not df_fold_r2.empty:
    base_colors = PALETTE_DICT.get(style_fold["palette"], px.colors.qualitative.Plotly)
    cv_colors_r2 = {
        name: base_colors[i % len(base_colors)]
        for i, name in enumerate(comparison.index)
    }
    fig_fold_r2 = plot_zone_boxplot(
        df=df_fold_r2,
        value_col="fold_r2",
        well_label="fold",
        zone_col="cv_scheme",
        zone_label="CV scheme",
        box_opacity=box_opacity,
        box_outline_width=box_outline_width,
        box_orientation=box_orientation,
        title=f"{step2_fold_r2_boxplot_title}",
        yaxis_title="R² fold",
        zone_colors=cv_colors_r2,
        value_label="R² fold",
        zone_order=list(comparison.index),
    )
    st.plotly_chart(fig_fold_r2, use_container_width=True)
    st.caption(step2_fold_r2_nan_note)
else:
    st.info(step2_fold_boxplot_empty)

st.markdown(f"#### {step2_cv_comparison_title}")
# r (Actual = Predicted): честный corr(Actual, Predicted) из summarize_cv
comparison["r_actual_pred"] = comparison["r_pooled"]

r_ap_col = "r (Actual = Predicted)"
comparison_display = comparison.rename(columns={
    "r2_actual_pred": cv_r2_col_label,
    "r_actual_pred":  r_ap_col,
}).reset_index()

# порядок: вставить r сразу после R² (Actual = Predicted)
cols_order = list(comparison_display.columns)
if cv_r2_col_label in cols_order and r_ap_col in cols_order:
    cols_order.remove(r_ap_col)
    cols_order.insert(cols_order.index(cv_r2_col_label) + 1, r_ap_col)
    comparison_display = comparison_display[cols_order]
st.dataframe(
    comparison_display.style.format({
        "n_pred": "{:.0f}",
        "n_test_min": "{:.0f}",
        "n_test_max": "{:.0f}",

        cv_r2_col_label: "{:.4f}",
        r_ap_col: "{:.4f}",
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
excel_download_button(comparison_display, "attribute_qc_step2_cv_comparison",
                      key="download_excel_step2_cv_comparison", index=True)

st.caption(step2_boxplot_caption)

# Boxplot разброса RMSE по фолдам для каждой схемы CV (Шаг 2) — в стиле Шага 3
# Длинная таблица: одна строка на фолд
rows_fold = []
for scheme_name in comparison.index:
    fold_df = results[scheme_name].get("folds")
    if fold_df is None or len(fold_df) == 0 or "rmse" not in fold_df.columns:
        continue
    fold_values = fold_df["rmse"].dropna().to_numpy(float)
    for fold_idx, fold_rmse in enumerate(fold_values, start=1):
        rows_fold.append({
            "cv_scheme": scheme_name,
            "fold_rmse": fold_rmse,
            "well_id": f"{scheme_name} · фолд {fold_idx}",   # используется в hover
        })
df_fold = pd.DataFrame(rows_fold)

if not df_fold.empty:
    # Цвета схем — из той же палитры, что и bar chart выше
    base_colors = PALETTE_DICT.get(style_fold["palette"], px.colors.qualitative.Plotly)
    cv_colors = {
        name: base_colors[i % len(base_colors)]
        for i, name in enumerate(comparison.index)
    }

    fig_fold_box = plot_zone_boxplot(
        df=df_fold,
        value_col="fold_rmse",
        well_label="fold",
        zone_col="cv_scheme",
        zone_label="CV scheme",
        box_opacity=box_opacity,
        box_outline_width=box_outline_width,
        box_orientation=box_orientation,
        title=f"{step2_fold_rmse_boxplot_title}",
        yaxis_title=f"RMSE fold, {PARAM_1_UNIT}",
        zone_colors=cv_colors,
        value_label=f"RMSE fold, {PARAM_1_UNIT}",
        zone_order=list(comparison.index),
    )

    st.plotly_chart(fig_fold_box, use_container_width=True)
else:
    st.info(step2_fold_boxplot_empty)

# ---- Таблица фолдов: отдельная таблица для каждой схемы CV (Шаг 2) ----
FOLD_TABLE_COLUMNS = [
    "Fold", "n_train", "n_test", 
    f"RMSE_fold, {PARAM_1_UNIT}",
    f"MAE_fold, {PARAM_1_UNIT}",
    f"Bias_fold, {PARAM_1_UNIT}",
]

def build_fold_table(fold_df):
    d = fold_df.copy()
    if "n_test" in d.columns and "n" in d.columns:
        d = d.drop(columns="n")
    d = d.rename(columns={
        "fold": "Fold",
        "n": "n_test",
        "r2": "R2_fold",
        "rmse": f"RMSE_fold, {PARAM_1_UNIT}",
        "mae": f"MAE_fold, {PARAM_1_UNIT}",
        "bias": f"Bias_fold, {PARAM_1_UNIT}",
    })
    d = d.reindex(columns=FOLD_TABLE_COLUMNS)
    d["Fold"] = d["Fold"].astype(str)
    return d

st.markdown(f"#### {step2_fold_table_title}")

any_fold_table = False
for i, scheme_name in enumerate(comparison.index):
    fold_df = results[scheme_name].get("folds")
    if fold_df is None or len(fold_df) == 0:
        continue

    any_fold_table = True
    fold_table_scheme = build_fold_table(fold_df)

    st.markdown(f"##### {scheme_name}")
    st.dataframe(fold_table_scheme, use_container_width=True)
    excel_download_button(
        fold_table_scheme,
        f"attribute_qc_step2_fold_table_{i + 1}",
        key=f"download_excel_step2_fold_table_{i}",
    )

if not any_fold_table:
    st.info(step2_fold_table_empty)

# RMSE_fold bar chart в шаге 2 : среднее RMSE по фолдам для каждой схемы CV
fig_fold = plot_bar_chart(
    x=comparison.index,
    y=comparison["rmse_fold_mean"],
    **style_fold,
    title=f"{step2_fold_rmse_title}",
    yaxis_title=f"RMSE fold, {PARAM_1_UNIT}",
    show_percent=False,
    legend_title="CV scheme",
)

st.plotly_chart(fig_fold, use_container_width=True)

# RMSE bar chart в шаге 2

fig_rmse = plot_bar_chart(
    x=comparison.index,
    y=comparison["rmse"],
    **style_rmse,
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
st.markdown(f"### {contents_step2_cp}")

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

if residual_map_category is not None and residual_map_category in df_cv.columns:
    plot_df[residual_map_category] = df_cv.loc[valid, residual_map_category].values

# --- Вычисляем невязки один раз ---
# CV residual = Actual - Predicted
resid_cv = plot_df["Zprm_well"] - plot_df["Predicted"]
# --- Разметка: слева Predicted vs Actual, справа кривые аппроксимации
col_pred, col_fit = st.columns(2)

# -- CV PLOT 1 — Predicted vs Actual + регрессия
with col_pred:
    st.markdown(
        f"#### {step2_cv_crossplot_title} "
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

    band_ok = add_cv_residual_band(fig_cv1, z_w_cv, resid_cv.to_numpy(float))
    if not band_ok:
        st.caption("Интервал P10–P90 не показан: недостаточно скважин для квантилей.")

    st.plotly_chart(fig_cv1, use_container_width=True)

# ---- CV PLOT 2 — кривые аппроксимации
with col_fit:
    st.markdown(
        f"#### {step2_approximations_title} "
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
cv_fit_stats.bias = -cv_fit_stats.bias
cv_metrics_df = build_qc_metrics_table(
    basic=cv_fit_stats,
    coeffs_qc=coeffs_cv,
    left_name="Predicted",
    right_name="Actual",
    r2_label=cv_pearson_r2_col_label,
    r_label=cv_r_col_label,
)

cv_metrics_df = cv_metrics_df.rename(columns={"R²_lin": cv_pearson_r2_col_label})

# Метрики Cross-Plot (Predicted vs Actual) для Шага 2
st.markdown(
    f"#### {step2_cv_metrics_title} "
    f"— {scheme_for_plot}"
)
st.dataframe(cv_metrics_df, use_container_width=True,)
st.caption(T["delta_help_cv"])
excel_download_button(cv_metrics_df, "attribute_qc_step2_cv_metrics_best",
                      key="download_excel_step2_cv_metrics_best", index=True)

st.markdown(
    f"#### {step2_approx_comparison_title} "
    f"— {scheme_for_plot}"
)


approx_cv_df = pd.DataFrame(approx_rows)
st.dataframe(approx_cv_df, use_container_width=True)
excel_download_button(approx_cv_df, "attribute_qc_step2_cv_approx",
                      key="download_excel_step2_cv_approx")

# -------------------------------------------------------------------- #
# STEP 2.3.: CV-невязки (Actual − Predicted) vs PARAM_1 + статистика
# -------------------------------------------------------------------- #
st.markdown('<div id="step-2-res"></div>', unsafe_allow_html=True)

CV_RESIDUAL_LABEL = f"Δ({PARAM_1} − Predicted)"
st.markdown(f"### {contents_step2_res} "
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
    st.caption(T["delta_help_residual"])
    # Статистика невязок в шаге 2
    cv_stats = compute_stats(resid_cv)
    st.dataframe(cv_stats.style.format({"Value": "{:.4f}"}), use_container_width=True)
    excel_download_button(cv_stats, "attribute_qc_step2_cv_residual_stats",
                      key="download_excel_step2_cv_resid_stats", index=True)

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
    st.markdown(f"#### {hist_title} CV ({CV_RESIDUAL_LABEL})")
    fig_hcv = plot_hist_kde(
        data=resid_cv,
        bins=cv_bins,
        histnorm=cv_histnorm_value,
        color=hist_color,
        title=f"{hist_title} CV ({CV_RESIDUAL_LABEL})",
        xlabel=f"{CV_RESIDUAL_LABEL}, {PARAM_1_UNIT}",
        show_sigma=True,
        sigma_k=sigma_k,
        opacity=hist_opacity,
        hist_outline_color=hist_outline_color,
        hist_outline_width=hist_outline_width_eff,
        kde_color=kde_color,       kde_dash=kde_dash,       kde_width=kde_width,
        mean_color=mean_color,     mean_dash=mean_dash,     mean_width=mean_width,
        median_color=median_color, median_dash=median_dash, median_width=median_width,
        sigma_color=sigma_color,   sigma_dash=sigma_dash,   sigma_width=sigma_width,
    )
    st.plotly_chart(fig_hcv, use_container_width=True)

# Boxplot в шаге 2
with col_bcv:
    st.markdown(f"#### {boxplot_title} CV ({CV_RESIDUAL_LABEL})")
    fig_bcv = plot_box(
        data=resid_cv,
        color=box_color,
        well_label=well_label,
        title=f"{boxplot_title} CV ({CV_RESIDUAL_LABEL})",
        ylabel=f"{CV_RESIDUAL_LABEL}, {PARAM_1_UNIT}",
        well_ids=plot_df["well_id"],
        value_label=f"{CV_RESIDUAL_LABEL}, {PARAM_1_UNIT}",
        mean_color=box_mean_color,     mean_dash=box_mean_dash,     mean_width=box_mean_width,
        median_color=box_median_color, median_dash=box_median_dash, median_width=box_median_width,
        **BOX_STYLE,
    )
    st.plotly_chart(fig_bcv, use_container_width=True)

# Карта CV-невязок
st.markdown(
    f"#### {spatial_residuals_title} "
    f"CV {CV_RESIDUAL_LABEL}"
)


fig_map = plot_residual_map(
    df=plot_df,
    residuals=resid_cv,
    palette=map_palette,
    wells_label=wells_legend,
    well_label=well_label,
    residual_label=CV_RESIDUAL_LABEL,
    point_size=residual_map_point_size,
    point_opacity=residual_map_point_opacity,
    title=f"{residual_map_title} CV {CV_RESIDUAL_LABEL}",
    category_col=residual_map_category,
    highlight_value=residual_map_highlight,
    highlight_opacity=residual_map_highlight_opacity,
    dim_opacity=residual_map_dim_opacity,
)
st.plotly_chart(fig_map, use_container_width=True)
st.caption(T["delta_help_residual"])
# --- Анализ выбросов по результатам CV-невязок
st.markdown('<div id="step-2-otl"></div>', unsafe_allow_html=True)
st.markdown(f"### {contents_step2_otl}")
show_outlier_tables(
    df=plot_df,
    residuals=resid_cv,
    sigma_thr=sigma_thr,
    iqr_k=iqr_k,
    pct_low=pct_low,
    pct_high=pct_high,
    param_1=PARAM_1,
    param_2=PARAM_2,
    residual_column_name=cv_residual_col_label.format(param=PARAM_1),
    no_outliers_label=no_outliers_label,
    key_prefix="step2",
    category_col="category_plot",
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
zone_hist_settings_expander = T["zone_hist_settings_expander"]
zone_hist_mode_label = T["zone_hist_mode_label"]
zone_histnorm_label = T["zone_histnorm_label"]
step3_cat_stats_title = T["step3_cat_stats_title"]

back_to_data_load()
st.markdown('<div id="step-3"></div>', unsafe_allow_html=True)
st.markdown("---")
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
    fig_zone_pie = plot_zone_pie(
        df, category_col=category_col, palette=pie_palette,
        title=f"{step3_well_count_title} : {category_col}",
        opacity=pie_opacity, outline_color=pie_outline_color, outline_width=pie_outline_width,
    )
    st.plotly_chart(fig_zone_pie, use_container_width=True)
    
    # Bar chart в Шаге 3
    zones = sorted(df[category_col].dropna().unique())
    if step3_bar_agg == "count" or step3_bar_value_col is None:
        zone_series = df.groupby(category_col)["well_id"].nunique().sort_index()
        zone_y_label = well_count_label
        zone_title = f"{step3_well_count_title} : {category_col}"
        zone_show_percent = True
    else:
        vals_s3 = pd.to_numeric(df[step3_bar_value_col], errors="coerce")
        zone_series = (
            getattr(vals_s3.groupby(df[category_col]), step3_bar_agg)()
            .dropna().sort_index()
        )
        zone_y_label = f"{T['crosstab_agg_' + step3_bar_agg]}, {display_name(step3_bar_value_col)}"
        zone_title = f"{T['crosstab_agg_' + step3_bar_agg]} {display_name(step3_bar_value_col)} : {category_col}"
        zone_show_percent = False

    fig_zones = plot_bar_chart(
        x=zone_series.index,
        y=zone_series.values,
        **style_step3,
        title=zone_title,
        yaxis_title=zone_y_label,
        show_percent=zone_show_percent,
    )

    st.plotly_chart(fig_zones, use_container_width=True)


    # ---- Общий заголовок для обеих таблиц ----
    st.markdown(f"#### {step3_cat_stats_title}")

    # ---- 1. Таблица по категориям: скважины / записи / доля ----
    st.markdown(f"**{well_count_label}: {display_name(category_col)}**")
    cat_stats_step3 = category_stats_table(
        df, category_col,
        T["cat_stats_wells"], T["cat_stats_records"], T["cat_stats_share"],
        display_name(category_col),
    )
    st.dataframe(cat_stats_step3, use_container_width=True, hide_index=True)
    excel_download_button(cat_stats_step3, f"attribute_qc_step3_category_stats_{category_col}",
                          key=f"download_excel_step3_cat_stats_{category_col}")

    # ---- 2. Таблица агрегаций по категориям (всегда полная) ----
    agg_value_col_s3 = step3_bar_value_col     

    well_counts_s3 = df.groupby(category_col)["well_id"].nunique().sort_index()
    agg_table_s3 = well_counts_s3.to_frame(name=well_count_label)

    if agg_value_col_s3 is not None:
        vals_s3_all = pd.to_numeric(df[agg_value_col_s3], errors="coerce")
        agg_extra_s3 = (
            vals_s3_all.groupby(df[category_col])
                        .agg(["min", "max", "median", "mean", "std"])
                        .sort_index()
        )
        agg_table_s3 = agg_table_s3.join(agg_extra_s3)
        table_title_s3 = f"{crosstab_table_title}: {display_name(agg_value_col_s3)}"
    else:
        table_title_s3 = f"{crosstab_table_title}: {display_name(category_col)}"

    agg_table_s3 = agg_table_s3.rename_axis(display_name(category_col)).reset_index()

    st.markdown(f"**{table_title_s3}**")
    st.dataframe(agg_table_s3, use_container_width=True, hide_index=True)
    excel_download_button(
        agg_table_s3,
        f"attribute_qc_step3_bar_aggs_{category_col}",
        key=f"download_excel_step3_bar_aggs_{category_col}",
    )


    # Цвета категорий из Sidebar
    zone_marker_colors = {zone: zone_point_colors.get(zone, "gray") for zone in zones}
    zone_curve_colors = {zone: zone_line_colors.get(zone, "gray") for zone in zones}

    # Статистика параметров по категориям
    st.markdown(
        f"#### {step3_stats_title}: "
        f"{category_col} "
        f"({PARAM_1} vs {PARAM_2})"
    )    
    for zone in zones:
        zone_part = df[df[category_col] == zone]
        with st.expander(f"{category_col}: {zone}", expanded=False):
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                stats_well_zone = show_depth_stats(zone_part["Zprm_well"], f"{PARAM_1_LABEL} ({zone})")
                if stats_well_zone is not None:
                    excel_download_button(stats_well_zone, f"attribute_qc_step3_stats_well_{zone}",
                                          key=f"download_excel_step3_stats_well_{zone}", index=True)
            with col_stat2:
                stats_map_zone = show_depth_stats(zone_part["Zprm_map"], f"{PARAM_2_LABEL} ({zone})")
                if stats_map_zone is not None:
                    excel_download_button(stats_map_zone, f"attribute_qc_step3_stats_map_{zone}",
                                          key=f"download_excel_step3_stats_map_{zone}", index=True)
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
            box_opacity=box_opacity,
            box_outline_width=box_outline_width,
            box_orientation=box_orientation,
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
            box_opacity=box_opacity,
            box_outline_width=box_outline_width,
            box_orientation=box_orientation,
            title=f"Boxplot {PARAM_2}",
            yaxis_title=PARAM_2_LABEL,
            zone_colors=zone_marker_colors,
            value_label=PARAM_2_LABEL,
        )
        st.plotly_chart(fig_box_p2, use_container_width=True)

    zone_fit_rows = []

    # Hist для категорий в шаге 3
    st.markdown(f"#### {hist_title} по {category_col}")
    # ---- Настройки наложенных гистограмм по зонам (отдельно от Шага 1) ----
    with st.expander(zone_hist_settings_expander, expanded=False):
        col_zs1, col_zs2, col_zs3, col_zs4 = st.columns(4)
        with col_zs1:
            bins_zone = st.slider(hist_bins_label, 10, 120, 40, key="bins_zone")
            zone_opacity = st.slider(zone_hist_opacity_label, 0.1, 1.0, 0.5, 0.05, key="zone_hist_opacity")
        with col_zs2:
            zone_hist_mode = st.radio(
                zone_hist_mode_label,
                ["hist", "kde", "both"],
                format_func=lambda x: T[f"zone_hist_mode_{x}"],
                index=2,
                horizontal=True,
                key="zone_hist_mode",
            )
        with col_zs3:
            zone_histnorm = st.selectbox(
                zone_histnorm_label,
                ["none", "density", "probability", "percent"],
                format_func=lambda x: hist_norm_map[x],
                index=0,
                key="zone_histnorm",
            )
        with col_zs4:
            zone_show_mean = st.checkbox(mean_line_label, value=True, key="zone_show_mean")
            zone_show_median = st.checkbox(median_line_label, value=True, key="zone_show_median")

    show_hist_zone = zone_hist_mode in ("hist", "both")
    show_kde_zone = zone_hist_mode in ("kde", "both")
    zone_histnorm_value = "" if zone_histnorm == "none" else zone_histnorm

    col_hz1, col_hz2 = st.columns(2)

    with col_hz1:
        fig_hz1 = plot_zone_overlay_hist(
            df=df,
            value_col="Zprm_well",
            zone_col=category_col,
            value_label=PARAM_1_LABEL,
            title=f"{hist_title} {PARAM_1}",
            bins=bins_zone,
            histnorm=zone_histnorm_value,
            opacity=zone_opacity,
            hist_outline_color=hist_outline_color,
            hist_outline_width=hist_outline_width_eff,
            zone_colors=zone_marker_colors,
            zone_order=list(zone_marker_colors.keys()) if zone_marker_colors else None,
            show_hist=show_hist_zone,
            show_kde=show_kde_zone,
            show_mean=zone_show_mean,
            show_median=zone_show_median,
            kde_width=kde_width,
            kde_dash=kde_dash,
            mean_width=mean_width,
            mean_dash=mean_dash,
            median_width=median_width,
            median_dash=median_dash,
        )
        st.plotly_chart(fig_hz1, use_container_width=True)

    with col_hz2:
        fig_hz2 = plot_zone_overlay_hist(
            df=df,
            value_col="Zprm_map",
            zone_col=category_col,
            value_label=PARAM_2_LABEL,
            title=f"{hist_title} {PARAM_2}",
            bins=bins_zone,
            histnorm=zone_histnorm_value,
            opacity=zone_opacity,
            hist_outline_color=hist_outline_color,
            hist_outline_width=hist_outline_width_eff,
            zone_colors=zone_marker_colors,
            zone_order=list(zone_marker_colors.keys()) if zone_marker_colors else None,
            show_hist=show_hist_zone,
            show_kde=show_kde_zone,
            show_mean=zone_show_mean,
            show_median=zone_show_median,
            kde_width=kde_width,
            kde_dash=kde_dash,
            mean_width=mean_width,
            mean_dash=mean_dash,
            median_width=median_width,
            median_dash=median_dash,
        )
        st.plotly_chart(fig_hz2, use_container_width=True)

    # ------------------------------------------------- #
    # # STEP 3.2.: ОБЩИЙ ГРАФИК ПО ВСЕМ ЗОНАМ
    # ------------------------------------------------- #
    st.markdown('<div id="step-3-cp"></div>', unsafe_allow_html=True)
    st.markdown(f"### {contents_step3_cp}")
    st.markdown(
        f"#### {step3_relationship_title}: "
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
    st.markdown(f"#### {step3_individual_categories_title} ({category_col})")

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
                z_fit, eq = cached_curve_numeric(approx, xz, yz, x_line_z)
                if approx == "Quantile Regression (τ=0.1/0.5/0.9)":
                    dash_by_tau = {0.1: "dot", 0.5: "solid", 0.9: "dot"}
                    for tau, y_pred, _ in z_fit:
                        fig_zone.add_trace(go.Scatter(
                            x=x_line_z, y=y_pred, mode="lines",
                            name=f"P{round(tau * 100)}: {eq_tau}",
                            line=dict(width=2, color=zone_line_color, dash=dash_by_tau[tau]),
                        ))
                    for tau in [0.1, 0.5, 0.9]:
                        try:
                            stats_q = cached_quantile_stats(xz, yz, tau)
                            zone_fit_rows.append({category_col: zone,
                                                  "Approximation": f"Quantile Regression (τ={tau})",
                                                  **fitstats_to_dict(stats_q)})
                        except Exception as e:
                            zone_fit_rows.append({category_col: zone,
                                                  "Approximation": f"Quantile Regression (τ={tau})",
                                                  error_column_label: str(e)})
                else:
                    approx_dash = APPROX_DASHES.get(approx, "solid")
                    approx_width = APPROX_WIDTHS.get(approx, 2)
                    if z_fit is not None:
                        fig_zone.add_trace(go.Scatter(
                            x=x_line_z, y=z_fit, mode="lines", name=f"{approx}: {eq}",
                            line=dict(width=approx_width, color=zone_line_color, dash=approx_dash),
                        ))
                    stats_zone = cached_fit_stats(xz, yz, approx)
                    zone_fit_rows.append({category_col: zone, "Approximation": approx,
                                  **fitstats_to_dict(stats_zone)})
            except Exception as e:
                zone_fit_rows.append({category_col: zone, "Approximation": approx,
                              error_column_label: str(e)})

        # Линия Y = X (идеальная карта)
        lo = min(xz.min(), yz.min())
        hi = max(xz.max(), yz.max())
        add_identity_line(fig_zone, lo, hi, color=identity_line_color, width=identity_line_width, dash=identity_line_dash,)
        
        # Настройка оформления cross-plot
        apply_crossplot_layout(fig_zone, PARAM_2_LABEL, PARAM_1_LABEL, height=360, title=f"{category_col}: {zone}")
        st.plotly_chart(fig_zone, use_container_width=True)

    st.markdown(
        f"#### {step3_metrics_comparison_title}: "
        f"{category_col}"
    )
    zone_fit_df = pd.DataFrame(zone_fit_rows)
    st.dataframe(zone_fit_df, use_container_width=True)
    excel_download_button(zone_fit_df, "attribute_qc_step3_zone_fit_metrics",
                          key="download_excel_step3_zone_fit")

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
step4_rand_series_title = T["step4_rand_series_title"]
step4_rand_series_title_in = T["step4_rand_series_title_in"]
step4_rand_series_title_out = T["step4_rand_series_title_out"]
step4_rand_series_caption = T["step4_rand_series_caption"]
step4_rand_series_seed_label = T["step4_rand_series_seed_label"]
step4_rand_series_run = T["step4_rand_series_run"]
step4_rand_series_spinner = T["step4_rand_series_spinner"]
step4_rand_series_not_run = T["step4_rand_series_not_run"]
step4_rand_series_seed_changed = T["step4_rand_series_seed_changed"]
step4_rand_series_status_ok = T["step4_rand_series_status_ok"]
step4_rand_series_status_skipped = T["step4_rand_series_status_skipped"]
step4_rand_series_status_no_valid = T["step4_rand_series_status_no_valid"]
step4_rand_series_box_step = T["step4_rand_series_box_step"]
step4_rand_series_box_title_in = T["step4_rand_series_box_title_in"]
step4_rand_series_box_title_out = T["step4_rand_series_box_title_out"]
step4_rand_series_zone_label = T["step4_rand_series_zone_label"]
step4_rand_series_no_data = T["step4_rand_series_no_data"]
step4_n_sets_shown = T["step4_n_sets_shown"]

RAND_SERIES_MIN_REMAINING = 5
back_to_data_load()
st.markdown('<div id="step-4"></div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"## {step4_title}") 

# --------------------------------------------------
# STEP 4.1.: ФИЛЬТРАЦИЯ СКВАЖИН
# ------------------------------------------------- #

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
@st.fragment
def render_step4_click_selector():
    st.markdown(f"### {step4_click_title}")
    st.caption(step4_click_hint)

    if "step4_click_excluded" not in st.session_state:
        st.session_state.step4_click_excluded = []

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

    # Линия регрессии QC — ориентир для выделения
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

    # Линия Y = X
    lo = min(df_click_source["Zprm_map"].min(), df_click_source["Zprm_well"].min())
    hi = max(df_click_source["Zprm_map"].max(), df_click_source["Zprm_well"].max())
    add_identity_line(
        fig_click, lo, hi,
        color=identity_line_color,
        width=identity_line_width,
        dash=identity_line_dash,
    )

    apply_crossplot_layout(fig_click, PARAM_2_LABEL, PARAM_1_LABEL, title=step4_click_title,)
    fig_click.update_layout(clickmode="event+select")

    # Выделение точек: этот вызов перезапускает только фрагмент
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

    # st.rerun() здесь перезапускает всё приложение,
    # поэтому сайдбар и шаги ниже обновятся после Apply / Reset
    if reset_clicked:
        st.session_state.step4_click_excluded = []
        st.rerun()

    if apply_clicked and selected_wells:
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


if filter_mode == FILTER_CLICK:
    render_step4_click_selector()
    excluded_wells_click = st.session_state.step4_click_excluded
else:
    excluded_wells_click = []

if excluded_wells_click:
    df_step4 = df_step4[~df_step4["well_id"].astype(str).isin(excluded_wells_click)]
# ------ Контроль ------
step4_ok = df_step4["well_id"].nunique() >= 3
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

    # STEP 4.3.1: Таблица метрик QC Cross-Plot — до и после фильтрации
    qc_metrics_before = build_qc_metrics_table(
        basic=basic, coeffs_qc=coeffs_qc_before,
        left_name=PARAM_1, right_name=PARAM_2,
    )
    qc_metrics_before.insert(0, T["step4_stage_label"], T["step4_stage_before"])

    qc_metrics_after = build_qc_metrics_table(
        basic=basic_step4, coeffs_qc=coeffs_qc_step4,
        left_name=PARAM_1, right_name=PARAM_2,
    )
    qc_metrics_after.insert(0, T["step4_stage_label"], T["step4_stage_after"])

    qc_metrics_step4 = pd.concat(
        [qc_metrics_before, qc_metrics_after], ignore_index=True
    )

    st.markdown(f"#### {step4_qc_metrics_title}")
    st.dataframe(qc_metrics_step4, use_container_width=True)
    st.caption(T["delta_help_step1"])
    excel_download_button(qc_metrics_step4, "attribute_qc_step4_qc_metrics",
                          key="download_excel_step4_qc_metrics", index=True)

    # STEP 4.3.2: Таблица метрик кривых аппроксимации — до и после фильтрации
    fit_table_before = pd.DataFrame(fit_rows_before)
    fit_table_before.insert(0, T["step4_stage_label"], T["step4_stage_before"])

    fit_table_after = pd.DataFrame(fit_rows_step4)
    fit_table_after.insert(0, T["step4_stage_label"], T["step4_stage_after"])

    fit_table_step4 = pd.concat(
        [fit_table_before, fit_table_after], ignore_index=True, sort=False,
    )

    st.markdown(f"#### {step4_approx_comparison_title}")
    st.dataframe(fit_table_step4, use_container_width=True)
    excel_download_button(fit_table_step4, "attribute_qc_step4_approx_comparison",
                          key="download_excel_step4_approx")

    # --------------------------------------------------
    # STEP 4.4.: Таблица метрик до и после фильтрации
    # --------------------------------------------------
    comparison_df = pd.DataFrame({
        "Metrics": ["Well", "R²_lin", "r", "RMSE", "MSE", "MAE", "Bias", "Std", "Variance", "WAPE %", "P-value"],
        "before filt": [
            f"{len(df)}",
            basic.r2, basic.r, basic.rmse, basic.mse, basic.mae,
            basic.bias, basic.std, basic.variance, basic.wape_pct, basic.p_value,
        ],
        "after filt": [
            f"{len(df_step4)}",
            basic_step4.r2, basic_step4.r, basic_step4.rmse, basic_step4.mse, basic_step4.mae,
            basic_step4.bias, basic_step4.std, basic_step4.variance, basic_step4.wape_pct, basic_step4.p_value,
        ]
    })

    # формат {:.4f} — только для строк с метриками (все, кроме первой «Well»)
    numeric_rows = pd.IndexSlice[1:, :]

    styled = (
        comparison_df.style
        .format({"before filt": "{:.4f}", "after filt": "{:.4f}"}, subset=numeric_rows)
        .format({"before filt": "{:.4e}", "after filt": "{:.4e}"}, subset=pd.IndexSlice[10:10, :])
        .format({"before filt": "{}",     "after filt": "{}"},     subset=pd.IndexSlice[0:0, :])
    )

    st.markdown(f"### {step4_before_after_metrics_title}")
    st.dataframe(styled, use_container_width=True)
    excel_download_button(comparison_df, "attribute_qc_step4_before_after",
                          key="download_excel_step4_before_after")

# --------------------------------------------------
# STEP 4.5.: Распределение R² (Pearson) при случайном исключении скважин
# --------------------------------------------------

# ------ Серия случайных исключений: распределение r² по уровням ------
st.markdown(f"### {step4_rand_series_title}")
st.caption(step4_rand_series_caption)
st.markdown(f"**{step4_rand_series_seed_label}:** {st.session_state.step4_random_seed}")
st.markdown(f"**{step4_n_sets_shown}:** {st.session_state.step4_n_sets}")

if st.button(step4_rand_series_run, key="step4_run_rand_series", use_container_width=True):
    with st.spinner(step4_rand_series_spinner):
        st.session_state["step4_rand_series"] = {
            "seed": st.session_state.step4_random_seed,
            "n_sets": st.session_state.step4_n_sets,
            "data": cached_random_exclusion_series(
                df,
                sq.RAND_SERIES_LEVELS,
                st.session_state.step4_n_sets,
                st.session_state.step4_random_seed,
                RAND_SERIES_MIN_REMAINING,
            ),
        }

res_pack = st.session_state.get("step4_rand_series")

if res_pack is None:
    st.info(step4_rand_series_not_run)
else:
    long_df, summary_df = res_pack["data"]

    if (res_pack["seed"] != st.session_state.step4_random_seed
            or res_pack["n_sets"] != st.session_state.step4_n_sets):
        st.warning(step4_rand_series_seed_changed)

    status_labels = {
        "ok": step4_rand_series_status_ok,
        "skipped": step4_rand_series_status_skipped,
        "no_valid": step4_rand_series_status_no_valid,
    }
    rand_titles = {"in": step4_rand_series_title_in, "out": step4_rand_series_title_out}
    rand_box_titles = {"in": step4_rand_series_box_title_in, "out": step4_rand_series_box_title_out}

    for mode in ("in", "out"):
        st.markdown(f"#### {rand_titles[mode]}")

        s_mode = summary_df[summary_df["mode"] == mode].drop(columns="mode").copy()
        s_mode["status"] = s_mode["status"].map(status_labels)

        if mode == "in":
            cols = ["level_pct", "n_excluded", "n_remaining", "n_train",
                    "n_sets", "status", "min", "max", "mean", "median"]
        else:
            cols = ["level_pct", "n_excluded", "n_remaining", "n_train", "n_test",
                    "n_sets", "status", "min", "max", "mean", "median"]
        s_mode = s_mode[cols]

        st.dataframe(
            s_mode.style.format({
                "level_pct": "{:.0f} %",
                "n_excluded": "{:.0f}",
                "n_remaining": "{:.0f}",
                "n_sets": "{:.0f}",
                "min": "{:.4f}", "max": "{:.4f}",
                "mean": "{:.4f}", "median": "{:.4f}",
            }, na_rep="—"),
            use_container_width=True,
        )
        excel_download_button(s_mode, f"attribute_qc_step4_rand_{mode}_summary",
                              key=f"download_excel_step4_rand_{mode}_summary", index=True)

        box_step = st.radio(
            step4_rand_series_box_step,
            [10, 5],
            format_func=lambda v: f"{v} %",
            horizontal=True,
            key=f"step4_rand_series_box_step_{mode}",
        )
        box_df = long_df[
            (long_df["level_pct"] % box_step == 0) & long_df[f"r2_{mode}"].notna()
        ].copy()

        if box_df.empty:
            st.info(step4_rand_series_no_data)
            continue

        box_df["level_label"] = box_df["level_pct"].astype(int).astype(str) + " %"
        box_df["well_id"] = box_df["level_label"] + " · " + box_df["set_no"].astype(str)
        box_df["r2_val"] = box_df[f"r2_{mode}"]
        order = [f"{int(p)} %" for p in sorted(box_df["level_pct"].unique())]

        base_colors = PALETTE_DICT.get(style_fold["palette"], px.colors.qualitative.Plotly)
        level_colors = {lab: base_colors[i % len(base_colors)] for i, lab in enumerate(order)}

        fig_rand = plot_zone_boxplot(
            df=box_df,
            value_col="r2_val",
            well_label="set",
            zone_col="level_label",
            zone_label=step4_rand_series_zone_label,
            box_opacity=box_opacity,
            box_outline_width=box_outline_width,
            box_orientation=box_orientation,
            title=rand_box_titles[mode],
            yaxis_title="R² (Pearson)",
            zone_colors=level_colors,
            value_label="R² (Pearson)",
            zone_order=order,
        )
        st.plotly_chart(fig_rand, use_container_width=True)

# ================================================================================================================================ #
# --------------------------------------------------   Шаг 5: МНОГОАТРИБУТНАЯ РЕГРЕССИЯ   ---------------------------------------
# ================================================================================================================================ #
step5_title = T["step5_title"]

back_to_data_load()
@st.cache_data(show_spinner=False, max_entries=8)
def cached_step5(df_in, cands, phi_upper):
    return sq.step5_run(df_in, list(cands), phi_upper=phi_upper, seed=sq.S5_SEED)

st.markdown('<div id="step-5"></div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"## {step5_title}")
st.caption(T["step5_intro"])

step5_auto_cands = [c for c in ATTR_CANDIDATES if c not in ("horizon_id", "tectonic_zone")]
if st.session_state.get("step5_attr_mode", "auto") == "manual":
    step5_cands = [c for c in st.session_state.get("step5_manual_attrs", []) if c in ATTR_CANDIDATES]
else:
    step5_cands = step5_auto_cands
step5_phi_unit = st.session_state.get("UNITS_BY_PARAM", {}).get("Phi", "fractions")
step5_phi_upper = 100.0 if step5_phi_unit == "%" else 1.0
step5_src_df = st.session_state["df_filtered"] if st.session_state.get("df_filtered") is not None else st.session_state["df"]
# сигнатура: данные (только столбцы шага 5) + кандидаты + phi_upper
step5_sig_cols = [c for c in ["well_id", "H_eff", "Phi", *step5_cands] if c in step5_src_df.columns]
step5_df_sig = int(pd.util.hash_pandas_object(step5_src_df[step5_sig_cols], index=False).sum())
step5_sig = (step5_df_sig, tuple(step5_cands), step5_phi_upper)
if st.session_state.get("step5_sig") != step5_sig:
    st.session_state["step5_result"] = cached_step5(step5_src_df, tuple(step5_cands), step5_phi_upper)
    st.session_state["step5_sig"] = step5_sig
step5 = st.session_state["step5_result"]

if not step5["ok"] or len(step5_cands) == 0:
    st.info(T["step5_too_few_wells"])
else:
    st.caption(T["step5_kinfo"].format(
        kmax=step5["kmax"], n_wells=step5["n_wells"], n_cand=len(step5_cands),
        excluded=step5["excluded_pairs"], K=step5["kfold_K"],
    ))
    step5_col_labels = {
        "Parameter": T["step5_col_parameter"], "Rank": T["step5_col_rank"],
        "Attributes": T["step5_col_attributes"], "k": T["step5_col_k"],
        "RMSE_cv": T["step5_col_rmse"], "R2_cv": T["step5_col_r2"], "r_cv": T["step5_col_r"],
        "WAPE_cv": T["step5_col_wape"], "Bias_cv": T["step5_col_bias"],
        "DeltaRMSE": T["step5_col_delta"], "Accepted": T["step5_col_accepted"],
        "Equation": T["step5_col_equation"],
        "RMSE_in": T["step5_col_rmse_in"], "R2_in": T["step5_col_r2_in"], "r_in": T["step5_col_r_in"],
        "Bias_in": T["step5_col_bias_in"], "CV_a": T["step5_col_cv_a"], "CV_b": T["step5_col_cv_b"],
    }
    step5_acc_labels = {"yes": T["step5_acc_yes"], "no": T["step5_acc_no"], "single": T["step5_acc_single"]}

    def _step5_display(frame, drop=()):
        out = frame.drop(columns=[c for c in drop if c in frame.columns]).copy()
        if "Accepted" in out.columns:
            out["Accepted"] = out["Accepted"].map(step5_acc_labels)
        return out.rename(columns=step5_col_labels)

    def _step5_scatter_fig(x, y, title, x_title, y_title, name, x_name, y_name,
                           line="reg", model_eq=None, legend_name=None):
        """
        Возвращает (fig, eq_text). Уравнение НЕ выводится в легенду — его показывает st.caption под графиком.
        line="reg":      линия OLS y по x на диапазоне x; eq_text = model_eq или уравнение регрессии.
        line="identity": линия y = x на общем диапазоне; eq_text = model_eq.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name=name,
            marker=dict(color=crossplot_point_color, size=crossplot_point_size, opacity=crossplot_point_opacity),
        ))
        eq_text = model_eq
        if line == "identity":
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines", name=legend_name or "Y = X",
                line=dict(color=qc_regression_color, width=qc_regression_width, dash=qc_regression_dash),
            ))
            if eq_text is None:
                eq_text = "Y = X"
        elif len(x) >= 2 and np.std(x) > 0:
            lo = float(np.min(x))
            hi = float(np.max(x))
            slope, icpt = np.polyfit(x, y, 1)
            if eq_text is None:
                eq_text = f"{y_name} = {icpt:.4g} + {slope:.4g}·{x_name}"
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[icpt + slope * lo, icpt + slope * hi], mode="lines",
                name=legend_name or T["step5_regression_label"],
                line=dict(color=qc_regression_color, width=qc_regression_width, dash=qc_regression_dash),
            ))
        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, height=420)
        return fig, eq_text

    def _step5_cross_pair(sch, param, scheme):
        """
        Порядок графиков Шага 5 (для каждого параметра и схемы CV):
          1) вклад атрибутов — лучший многоатрибутный набор из топ-5 Out-of-sample;
          2) строка из двух графиков — лучший одноатрибутный (по Out-of-sample): In-sample | Out-of-sample;
          3) строка из двух графиков — лучший многоатрибутный набор (по Out-of-sample): In-sample | Out-of-sample.
        """
        tp = sch["table"][sch["table"]["Parameter"] == param]
        multi_rows = tp[tp["k"] >= 2]
        m_rank = int(multi_rows.iloc[0]["Rank"]) if len(multi_rows) else None
        det_m = sch["details"].get((param, m_rank)) if m_rank is not None else None

        # 1. вклад атрибутов
        if det_m is not None:
            fig_bar = plot_bar_chart(
                x=det_m["attrs"], y=det_m["std_coef"], **style_step5,
                title=T["step5_contrib_title"].format(param=param, scheme=scheme),
                yaxis_title=T["step5_axis_contrib"],
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # 2. одноатрибутный: In-sample | Out-of-sample
        sb = sch["single_best"].get(param)
        if sb is not None:
            c1, c2 = st.columns(2)
            with c1:
                fig, eq = _step5_scatter_fig(
                    sb["x"], sb["y"],
                    T["step5_cp_single_in_title"].format(param=param, attr=sb["attr"], scheme=scheme),
                    sb["attr"], param, param, sb["attr"], param,
                    line="reg", model_eq=sb["equation"], legend_name=T["step5_regression_label"],
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(T["step5_equation_caption"].format(eq=eq))
            with c2:
                fig, eq = _step5_scatter_fig(
                    sb["y_cv"], sb["oof"],
                    T["step5_cp_single_oos_title"].format(param=param, attr=sb["attr"], scheme=scheme),
                    T["step5_axis_actual"], T["step5_axis_predicted"], param, "Actual", "Predicted",
                    line="reg", legend_name=T["step5_regression_label"],
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(T["step5_equation_caption"].format(eq=eq))

        # 3. многоатрибутный: In-sample | Out-of-sample
        if det_m is not None:
            c1, c2 = st.columns(2)
            with c1:
                fig, eq = _step5_scatter_fig(
                    det_m["fit_pred"], det_m["fit_actual"],
                    T["step5_cp_multi_in_title"].format(param=param, scheme=scheme),
                    T["step5_axis_fitted"], param, param, "Fitted", param,
                    line="identity", model_eq=det_m["equation"], legend_name=T["step5_model_label"],
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(T["step5_equation_caption"].format(eq=eq))
            with c2:
                fig, eq = _step5_scatter_fig(
                    det_m["actual"], det_m["pred"],
                    T["step5_cp_multi_oos_title"].format(param=param, scheme=scheme),
                    T["step5_axis_actual"], T["step5_axis_predicted"], param, "Actual", "Predicted",
                    line="reg", legend_name=T["step5_regression_label"],
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(T["step5_equation_caption"].format(eq=eq))

    _step5_anchor = {"H_eff": "heff", "Phi": "phi"}
    for step5_param in ("H_eff", "Phi"):
        ap = _step5_anchor[step5_param]
        st.markdown(f'<div id="step-5-{ap}"></div>', unsafe_allow_html=True)
        st.markdown(f"### {step5_param}")

        ref = step5["schemes"]["LOWO"]
        sg_in = ref["singles_in"][ref["singles_in"]["Parameter"] == step5_param]
        st.markdown(f"**{T['step5_tbl_singles_in_title']}**")
        st.dataframe(_step5_display(sg_in), use_container_width=True)
        excel_download_button(_step5_display(sg_in), f"attribute_qc_step5_singles_in_{ap}",
                              key=f"download_excel_step5_singles_in_{ap}")

        for step5_scheme, step5_scheme_title in [("LOWO", T["step5_lowo_title"]), ("K-fold", T["step5_kfold_title"])]:
            sch = step5["schemes"][step5_scheme]
            sa = "lowo" if step5_scheme == "LOWO" else "kfold"
            st.markdown(f'<div id="step-5-{ap}-{sa}"></div>', unsafe_allow_html=True)
            st.markdown(f"#### {step5_scheme_title}")

            sg_cv = sch["singles"][sch["singles"]["Parameter"] == step5_param]
            st.markdown(f"**{T['step5_tbl_singles_cv_title']}**")
            sg_cv_show = _step5_display(sg_cv, drop=("WAPE_cv", "Accepted", "CV_a", "CV_b"))
            st.dataframe(sg_cv_show, use_container_width=True)
            excel_download_button(sg_cv_show, f"attribute_qc_step5_singles_cv_{sa}_{ap}",
                                  key=f"download_excel_step5_singles_cv_{sa}_{ap}")

            tp_in = sch["top_in"][sch["top_in"]["Parameter"] == step5_param]
            st.markdown(f"**{T['step5_tbl_top_in_title']}**")
            tp_in_show = _step5_display(tp_in)
            st.dataframe(tp_in_show, use_container_width=True)
            excel_download_button(tp_in_show, f"attribute_qc_step5_top5_in_{sa}_{ap}",
                                  key=f"download_excel_step5_top_in_{sa}_{ap}")

            tp_cv = sch["table"][sch["table"]["Parameter"] == step5_param]
            st.markdown(f"**{T['step5_tbl_top_cv_title']}**")
            tp_cv_show = _step5_display(tp_cv)
            st.dataframe(tp_cv_show, use_container_width=True)
            excel_download_button(tp_cv_show, f"attribute_qc_step5_top5_cv_{sa}_{ap}",
                                  key=f"download_excel_step5_top_cv_{sa}_{ap}")

            for (p_key, rank_key), step5_det in sch["details"].items():
                if p_key != step5_param:
                    continue
                with st.expander(T["step5_coef_expander"].format(param=p_key, rank=rank_key), expanded=False):
                    st.dataframe(step5_det["coef_df"], use_container_width=True)

            coef_all = sch["coef_table"]
            if len(coef_all) and "Parameter" in coef_all.columns:
                coef_p = coef_all[coef_all["Parameter"] == step5_param]
                excel_download_button(coef_p, f"attribute_qc_step5_coefficients_{sa}_{ap}",
                                      key=f"download_excel_step5_coef_{sa}_{ap}")

            _step5_cross_pair(sch, step5_param, step5_scheme)

st.markdown("---")
st.caption("CROSSPL✳T↔CROSS⊗VALIDATION")
st.caption(T["footer_data_note"])
st.caption(T["footer_copyright"])