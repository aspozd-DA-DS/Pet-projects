import pandas as pd
import streamlit as st

def show_project_introduction():

    with st.expander("📖 About the project and how it works", expanded=False):

        st.markdown(r"""
In geology and seismic exploration, there is a recurring task of assessing the consistency of horizon elevation picks from wells with a structural map built from 2D/3D CDP seismic data — such a check is an integral part of quality control for structural interpretations. This task formed the basis of the project. However, the methodological apparatus used — the cross-plot, a set of approximation curves, and cross-validation schemes — is not specific to the "well — map" pair and is equally applicable to assessing the relationship between any two quantitative parameters.

---

▶️ **Where to start:** upload your CSV/XLSX in the left sidebar — or pick a ready-made example from the "Select a demo file" dropdown (nothing needs to be prepared in advance to see how it works). From there, everything is organized step by step, as described below.

---

**Structural QC — cross-plot & cross-validation** technically compares two numeric series at the same points (wells) — `PARAM_1` (reference value) and `PARAM_2` (value being compared) — and honestly tests how well one can be predicted from the other.

A geological example: **Z_well** (horizon depth from a well) and **Z_map** (depth of the same horizon from a structural map). Below, the notation Z_well / Z_map is used throughout for clarity — but all steps and formulas apply equally to any PARAM_1/PARAM_2 pair:

| Scenario | Example PARAM_1 / PARAM_2 |
|---|---|
| well ↔ map | Z_well / Z_map |
| well ↔ attribute | Z_well / Amplitude |
| map ↔ map | Z_map_v1 / Z_map_v2 |
| attribute ↔ attribute | Amplitude / Sweetness |

Below, the notation Z_well / Z_map is used throughout as an example — just keep in mind that this is a special case of PARAM_1 / PARAM_2.

---

## How it works, step by step

**Step 0 · Data loading.** CSV/XLSX import, column mapping, automatic cleaning (missing values, duplicates, garbage values, data types), correlation analysis of the selected parameters.

**Step 1 · Basic QC cross-plot.** Z_well (reference) vs Z_map (compared): the Y=X diagonal, linear regression + a set of approximation curves to choose from, a full set of quality metrics, distributions, residual analysis and a residual map, outlier detection.

**Step 2 · Cross-validation.** The same model, but honestly tested: train on a subset of wells, predict on the remaining ones, repeat across different splitting schemes — to see the real, not "fitted", accuracy. The scheme with the best combination of accuracy and stability is determined automatically.

**Step 3 · Analysis by categories.** The same parameter pair — but separately by horizons, tectonic zones, reservoir type, fluid type, or any other categorical column. This helps reveal whether accuracy "drifts" in a particular zone or horizon.

**Step 4 · Well filtering.** Manual exclusion of specific wells or random exclusion of a given percentage (0–95%, with resampling) — to check how stable the result is to reducing the number of wells and to individual observations.

*(Step 4 supports three filtering modes: manual well exclusion by list, random exclusion of a given percentage of wells with resampling, and interactive well selection by clicking points on the cross-plot — one point per click, multiple via Shift+click.)*

---

📎 **Below is a detailed reference** (why cross-validation is needed, all CV schemes, all approximation curves, all metrics with formulas). For a first introduction, it is enough to read up to this point — the rest will be useful as you work through specific steps.

## Why cross-validation at all, and not just regression

If you simply build a regression on all wells and compute R²/RMSE on those same wells — the result will be deceptively good: the model could have "peeked" at all the points during training. Cross-validation honestly excludes some wells from training and tests the prediction on them. The schemes below differ in **exactly how** the excluded wells are chosen — this is especially important for spatial data (neighboring wells are "similar" to each other, and random splitting ignores this).

### Types of cross-validation (from most optimistic to most honest)

| Method | Essence | When to apply |
|---|---|---|
| Random K-Fold | `sklearn.KFold(shuffle=True)`: wells are randomly shuffled and divided into k equal parts (if the given k exceeds the number of points n, it is automatically reduced to n); each part in turn becomes the test set, training is on the remaining parts | ⚠️ Not for spatial data — gives inflated accuracy |
| Leave-One-Well-Out (LOWO) | `LeaveOneGroupOut` by `well_id`: each entire well in turn (all its rows/horizons) is excluded from training and becomes the sole test — so every well is tested once | Standard for a small number of wells (<20–30) |
| Block CV | The area is divided by a regular grid_n×grid_n grid over X_coord/Y_coord; each entire block in turn becomes the test set, training is on wells outside the block (a block is skipped if it contains <3 training wells) | Uniform well grid |
| Buffered CV | Like Block CV, but wells whose minimum distance to the nearest test point of the block ≤ buffer are additionally removed from training | Reduces leakage across the block boundary |
| Leave-P-Out | On each of the repeats, p_percent% of wells are randomly selected (without replacement) — test, training is on the rest; the final prediction for a well is the average over all repeats where it fell into the test set | Estimating the spread of stability under different random test compositions |
| Spatial Leave-P-Out | On each of the repeats, a center well is randomly selected; all wells within radius ≤ radius from it — test, the rest — train; the prediction is averaged over repeats in the same way as in Leave-P-Out | Closest to the real prediction of a new undrilled area |
| Stratified by Zone | `LeaveOneGroupOut` by `tectonic_zone`: each entire tectonic zone in turn becomes the test set, training is on wells from the remaining zones (requires ≥2 unique zones) | Checking whether the model transfers to a new zone |
| Distance-based exclusion CV | For each well i in turn: all wells at a distance ≤ range_corr from it are excluded from training (range_corr = median of pairwise distances between wells × variogram_factor), the test set is only well i | The strictest check for spatial leakage — essentially LOWO with a buffer |

*The parameters k, grid_n, buffer, p_percent, radius, variogram_factor, and repeats are not constants but sliders in the sidebar; each also shows a recommended value there.*

---

## Approximation curves

Besides a straight line, one of 17 curves can be overlaid on the cross-plot — from simple polynomials to methods robust to outliers:

| Method | General equation form | When it is useful |
|---|---|---|
| Linear | $y = a + bx$ | Basic linear relationship |
| Quadratic | $y = a + bx + cx^2$ | Relationship with a single bend |
| Cubic | $y = a + bx + cx^2 + dx^3$ | Relationship with an inflection point |
| Polynomial Degree 4–10 | $y = a_0 + a_1x + \dots + a_nx^n$ | More complex, but less often justified, relationship shape |
| Power Law | $y = a x^b$ &nbsp;($x,y>0$) | Power-law dependence |
| Exponential | $y = a e^{bx}$ &nbsp;($y>0$) | Exponential growth/decay |
| Logarithmic | $y = a + b\ln x$ &nbsp;($x>0$) | Saturation/deceleration of growth |
| Robust (Huber) | $y = a + bx$; coefficients minimize the Huber loss function | Linear relationship robust to outliers |
| Theil-Sen | $y = a + bx$; $b = \mathrm{median}\left(\dfrac{y_j-y_i}{x_j-x_i}\right)$ | Linear relationship robust to outliers |
| RANSAC | $y = a + bx$; fitted by maximizing the number of inlier points | Linear relationship under strong outliers/anomalies |
| LOWESS | no single formula — local regression in a sliding window | Follows the shape of the point cloud without assuming a relationship form |
| Quantile Regression (τ=0.1/0.5/0.9) | $y = a_\tau + b_\tau x$ | Three lines for the lower/median/upper bounds of the spread |

*For the τ=0.1 and τ=0.9 curves, R² can be negative — this is normal. Quantile regression optimizes the pinball loss, not squared error (like linear regression), so its curves describe the bounds of the spread, not the center of the point cloud. For such curves, use Equation and Bias to judge quality, not R².*

---

## Cross-plot metrics and notation

Let Δ denote the residual (difference between the two values). On the basic QC cross-plot (Step 1), Δ = Z_well − Z_map; on approximation curves and in cross-validation (Steps 1–3), Δ = Predicted − Actual (the opposite sign — this is normal, they are different comparisons in meaning).

| Metric | Formula |
|---|---|
| R² (coefficient of determination / squared correlation) | $R^2_{lin} = r^2$ (Pearson, linear relationship) &nbsp;·&nbsp; $R^2_{fit} = 1 - \dfrac{\sum(\Delta)^2}{\sum(y-\bar y)^2}$ (any curve shape) |
| RMSE (root mean squared error) | $\sqrt{\dfrac{1}{N}\sum (\Delta)^2}$ |
| MSE (mean squared error) | $\dfrac{1}{N}\sum (\Delta)^2$ |
| MAE (mean absolute error) | $\dfrac{1}{N}\sum \lvert \Delta \rvert$ |
| Bias (systematic offset) | $\dfrac{1}{N}\sum (\Delta)$ |
| Std (standard deviation of residuals) | $\sqrt{\dfrac{1}{N-1}\sum(\Delta_i-\bar\Delta)^2}$ |
| Variance (variance of residuals) | $\dfrac{1}{N-1}\sum(\Delta_i-\bar\Delta)^2$ |
| MAPE, % (percentage error) | $\dfrac{100}{N}\sum \dfrac{\lvert \Delta \rvert}{\max(\lvert y \rvert,\,10^{-9})}$ |
| P-value (significance of linear correlation) | $t = r\sqrt{\dfrac{N-2}{1-r^2}}$, &nbsp; $p = 2\left(1-F_{t,\,N-2}(\lvert t\rvert)\right)$ |
| Linear regression equation | $y = a + bx$; slope b and intercept a — from `linregress` |

*The p-value answers the question: how likely is it to obtain an equally strong (or stronger) linear relationship between PARAM_1 and PARAM_2 by chance, if in reality there is no relationship (null hypothesis H₀: the true regression slope b=0). The smaller the p-value — the less plausible that the found dependence is a coincidence; the significance threshold is usually taken as p<0.05. It is computed only for the basic QC cross-plot (Step 1), via `scipy.stats.linregress`.*

---

## Parameter statistics 

For PARAM_1 and PARAM_2, extended descriptive statistics are computed separately — here is what each quantity means:

| Statistic | Formula / definition |
|---|---|
| Count | N — number of non-empty values |
| μ (mean) | $\mu = \dfrac{1}{N}\sum x_i$ — arithmetic mean |
| median | median — the value dividing the sample in half (50th percentile) |
| mode | mode — the most frequently occurring value |
| σ (std) | $\sigma = \sqrt{\dfrac{1}{N-1}\sum(x_i-\mu)^2}$ |
| variance | $\sigma^2 = \dfrac{1}{N-1}\sum(x_i-\mu)^2$ |
| CV, % | $\dfrac{\sigma}{\mu}\times 100\%$ — coefficient of variation (spread relative to the mean) |
| min | minimum value |
| Q05 | 5th percentile |
| Q25 | 25th percentile (first quartile) |
| Q75 | 75th percentile (third quartile) |
| Q95 | 95th percentile |
| max | maximum value |
| range | $\max - \min$ |
| IQR | $Q75 - Q25$ — interquartile range |
| IQR lower bound | $Q25 - 1.5 \cdot IQR$ — lower bound for the IQR rule |
| IQR upper bound | $Q75 + 1.5 \cdot IQR$ — upper bound for the IQR rule |

---

*The detailed methodology, formulas, and rationale for each CV scheme — in the project documentation.*
""")

def show_input_file_example():

    with st.expander("📂 Input file format", expanded=False):

        st.markdown("### Required columns")

        st.markdown("""
        - **well_id** — well identifier
        - **X_coord** — X coordinate
        - **Y_coord** — Y coordinate
        - **Parameter 1** — reference parameter (well, map, attribute, etc.)
        - **Parameter 2** — parameter being compared (well, map, attribute, etc.)

        - Possible parameter examples:
          - Z_well
          - Z_map
          - T_well
          - T_map
          - Vint_well
          - Vint_map
          - Vav_well
          - Vav_map
          - ΔT_well
          - ΔT_map
          - ΔZ_well
          - ΔZ_map
          - any other numeric attribute
        """)

        st.markdown("### Optional columns")

        st.markdown("""
        - **horizon_id** — horizon identifier
        - **tectonic_zone** — tectonic zone
        - **reservoir_flag** — reservoir type
        - **fluid_type** — fluid type
        - any additional categorical or numeric parameters
        """)

        example_df = pd.DataFrame({
            "well_id": ["Well_001", "Well_002", "Well_003"],
            "X_coord": [125000.0, 126500.0, 128200.0],
            "Y_coord": [745000.0, 746200.0, 748100.0],
            "Z_well": [3120.5, 3188.2, 3256.4],
            "Z_map": [3108.7, 3201.5, 3242.1],
            "horizon_id": ["H1", "H1", "H1"],
            "tectonic_zone": ["Zone_A", "Zone_A", "Zone_B"],
        })

        st.markdown("### Example file contents")

        st.dataframe(example_df, use_container_width=True)
