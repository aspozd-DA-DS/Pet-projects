import pandas as pd
import streamlit as st

def show_project_introduction():

    with st.expander("📖 About the project and how it works", expanded=False):

        st.markdown(r"""
In geology and seismic exploration there is a recurring task: assessing how well a reservoir property measured in wells is related to a seismic attribute sampled at the same points. Whether an attribute reliably predicts the property decides whether it can be used as a proxy in areas without wells. The project is built on a cross-plot, a set of fitted curves and cross-validation. This toolkit is not tied to one particular pair of parameters and applies equally to any two numeric quantities.

---

▶️ **Where to start:** upload your CSV/XLSX in the sidebar on the left, or pick a ready-made example from the "Select Demo Dataset" list. Nothing needs to be prepared in advance. The steps are described below.

---

**Attribute QC_CV — cross-plot & cross-validation** compares two numeric series at the same points (wells): the reference parameter `PARAM_1` (H_eff or Phi) and the compared parameter `PARAM_2` (a seismic attribute you pick from the columns of your file), and checks how well one can be predicted from the other.

Examples of pairs that can be analysed:

| Scenario | Example PARAM_1 / PARAM_2 |
|---|---|
| well ↔ attribute | H_eff / rms_amplitude, Phi / acoustic_impedance |
| attribute ↔ attribute | rms_amplitude / sweetness (multicollinearity check) |

All steps and formulas apply equally to any PARAM_1 / PARAM_2 pair. Below, H_eff and rms_amplitude are used for illustration.

---

## How it works, step by step

**Step 0 · Data loading.** Import of CSV/XLSX, column mapping (coordinates X and Y, H_eff, Phi, horizon, zone and other optional columns), units (by default H_eff in metres, Phi in fractions), automatic cleaning (gaps, duplicates, junk values, data types). Then the statistics of H_eff and Phi and the correlation analysis of three groups: all seismic attributes; attributes and H_eff; attributes and Phi. Each group has Pearson and Spearman matrices, a table of pairs with a correlation coefficient at or above the threshold (0.7 by default) and a pairplot. The attributes in each block are chosen with a multiselect. Pairplot is recommended for no more than 10 parameters. Below that come descriptive statistics, histograms and boxplots for every numeric column. Most tables can be downloaded to Excel.

**Step 1 · Parameter selection and the basic QC cross-plot.** At the start of the step you choose PARAM_1 (H_eff or Phi), PARAM_2 (seismic attribute) and the unit of PARAM_2. The chosen pair is used in steps 1–4. Then the cross-plot of PARAM_1 (Y axis) against PARAM_2 (X axis): the Y = X diagonal, linear regression, a set of fitted curves of your choice, the full set of quality metrics and the distributions of both parameters. In all steps the model is linear regression (polynomial of degree 1). The basic errors of this step (Bias, RMSE, MAE and others) depend on the scale and units of the parameters, so they can only be compared within one PARAM_1–PARAM_2 pair. Judge the quality of the relationship by R² and correlation coefficients. Metrics of this step are in-sample: the model is fitted and checked on all wells of the chosen pair.

**Step 2 · Cross-validation.** The same linear model (degree 1), checked honestly: train on part of the wells, predict the rest, and repeat over different splitting schemes, so you see the real accuracy rather than an overfitted one. There are nine schemes. The ninth, Stratified by {колонка}, uses any categorical column with at least two values (by default tectonic_zone, or horizon_id if it is absent); the column is chosen in the sidebar. If no suitable column exists, the scheme is skipped and eight remain. The best scheme is chosen automatically by the criterion set in the sidebar: RMSE + RMSE_fold_std (default), RMSE, MAE, maximum R² (Actual = Predicted), RMSE + |Bias|, or manual choice. All schemes take part in the choice, including Random K-Fold and Random K-Fold (buffered). The spread of RMSE across folds is shown as bar charts and a boxplot per scheme, and the full fold data as separate tables: fold number, n_train, n_test, RMSE, MAE and Bias. This step also analyses outliers of the CV residuals (Actual − Predicted) with four rules. The CV residual map shows the residuals by well coordinates: the colour of a point is Actual − Predicted, categories are shown with markers, and the selected category is highlighted. The map shows where the model systematically under- or overestimates the value. Results are reproducible: the random seed is 42 by default and is changed with the sidebar buttons; it affects Random K-Fold, Random K-Fold (buffered), Leave-P-Out and Spatial Leave-P-Out. If fewer than degree + 2 points remain in training for a fold, the fold is marked as skipped in the fold table: the Fold column shows the reason and the number of training points, n = 0, the metrics are empty, and such a fold is not drawn on the boxplot. The scheme comparison table shows the number of points with predictions (n_pred); if the schemes cover different numbers of points, the app shows a warning. With one horizon, the folds of Leave-One-Well-Out and Distance-based exclusion CV contain one point each, so R² (Pearson) per fold is undefined for them and is not shown on the boxplot. Metrics of this step are out-of-sample: every well is predicted by a model that did not see it during fitting.

**Step 3 · Analysis by category.** The same parameter pair, but split by horizon, tectonic zone, reservoir type, fluid type, well category or any other categorical column. It shows whether accuracy drifts in a particular zone or horizon. Each category has a statistics table. The overall boxplot and the overlaid histograms show all categories together, on one set of bins and with common normalisation and transparency settings. Metrics by category are in-sample within each category.

**Step 4 · Well filtering.** Manual exclusion of chosen wells, random exclusion of a given percentage of wells (0–95%, with resampling), or selection of wells by clicking on the cross-plot. The aim is to check how stable the result is to removing wells and to individual observations. With a given seed the random set is reproducible, and the button produces the next one. Block 4.5 repeats random exclusion at levels from 5% to 95% in steps of 5%, with 50–2000 sets per level (500 by default, set in the sidebar). For every set the R² (Pearson) is computed in-sample, on the remaining wells, and out-of-sample, where the line is fitted on the remaining wells and R² is computed on the excluded ones; the distributions of both values are shown. Metrics before and after filtering are in-sample on the filtered sample; block 4.5 shows both in-sample and out-of-sample values.

*Click selection: one point per click, several with Shift+click, or a rectangle (Box) or lasso (Lasso). Switch modes with the icons in the top-right corner of the chart.*

**Step 5 · Multi-attribute regression.** The final stage: separately for H_eff and for Phi, on several seismic attributes. The H_eff section comes first, then the Phi section. Inside each section:

- **Single-attribute models — baseline for comparison In-sample (RMSE_in, R²_in).** Regression of the parameter on one attribute over all wells where both values exist. The equation and the metrics refer to this regression; r equals the Pearson correlation from the matrix (Step 0). One table for both CV schemes.
- **Single-attribute models — baseline for comparison Out-of-sample (RMSE_cv, R²_cv).** Separately for LOWO and for K-fold by wells. The metrics and the equation Predicted = a + b·Actual are obtained from the same out-of-fold predictions.
- **Top-5 multi-attribute models In-sample (RMSE_in, R²_in).** Regression on the attribute set over all wells; the coefficients are in the equation.
- **Top-5 multi-attribute models Out-of-sample (RMSE_cv, R²_cv).** The same models; the metrics and the equation Predicted = a + b·Actual from out-of-fold predictions.

The selection criterion is RMSE_cv; a multi-attribute model is accepted only if it beats the best single-attribute model of the same scheme. Coefficients (intercept, coefficients, standardized coefficients) are in an expandable block under the table. Charts: the attribute contribution of the best multi-attribute model; then a row of two charts for the best single attribute (In-sample | Out-of-sample) and a row for the best multi-attribute set (In-sample | Out-of-sample). All tables are exported to Excel with a button under each table.

---

📎 **Further on — the detailed reference** (why cross-validation is needed, all CV schemes, all fitted curves, all metrics with formulas). For a first acquaintance it is enough to read up to this point.

## Why cross-validation at all, rather than plain regression

If you fit a regression on all wells and compute R² and RMSE on the same wells, the result looks deceptively good: the model has "seen" every point during training. Cross-validation removes part of the wells from training and checks the prediction on exactly those wells. The schemes below differ in **how** the excluded wells are chosen. This matters a lot for spatial data: neighbouring wells resemble each other, and random splitting ignores that.

### Types of cross-validation (from the most optimistic to the most honest)

| Method | Essence | When to use | Example (100 wells) |
|---|---|---|---|
| Random K-Fold | `sklearn.KFold(shuffle=True)`: wells are shuffled at random and split into k equal parts (if the requested k exceeds the number of wells n, it is reduced to n automatically); each part in turn is the test set, training is on the rest | ⚠️ Not for spatial data: gives inflated accuracy | 5 folds of about 20 wells: in each, training on about 80 wells, n_test ≈ 20, n_train ≈ 80. Neighbouring wells often end up in training and test at the same time |
| Random K-Fold (buffered) | The same random KFold with the same seed, so the same test folds. Before training, wells whose distance to the nearest test well is ≤ buffer are removed from the training set | Random folds without leakage from neighbouring wells; comparing with plain Random K-Fold shows how inflated the latter was | The same 20/80 split: n_test matches Random K-Fold, n_train ≈ 80 if the neighbours lie beyond buffer. On a dense network n_train can drop to 40–60, visible in the n_train column |
| Leave-One-Well-Out (LOWO) | `LeaveOneGroupOut` by `well_id`: each entire well (all its rows across horizons) in turn is excluded from training and becomes the only test set | The standard for a small number of wells (<20–30) | 100 folds with one well each: n_test = 1, n_train = 99 |
| Block CV | The area is divided by a regular grid_n×grid_n grid over the X and Y coordinates; each block in turn becomes the test set, training is on wells outside the block (a block is skipped if fewer than degree + 2 training points remain, i.e. fewer than 3 for the linear model) | A uniform well grid | A 3×3 grid gives up to 9 blocks of about 11 wells each: n_test ≈ 11, n_train ≈ 89 |
| Buffered CV | As Block CV, but wells whose minimum distance to the nearest test point of the block is ≤ buffer are also removed from training | Reduces leakage across the block boundary | The same 3×3 grid: n_test ≈ 11, n_train is below 89 by the number of wells inside the buffer around the block |
| Leave-P-Out | In each of the repeats, p_percent% of the wells are drawn at random (without replacement) as the test set, training is on the rest. The final prediction for a well is the average over all repeats in which it was in the test set; wells that never fell into a test set stay without a prediction (NaN) | Estimating the spread of stability across different random test compositions | 50 repeats of 10 random wells in the test: n_test = 10, n_train = 90. Wells that never fell into a test set have no prediction, so n_pred < 100 |
| Spatial Leave-P-Out | In each of the repeats a centre well is drawn at random; all wells within radius ≤ radius of it form the test set, the rest the training set. The prediction is averaged over repeats as in Leave-P-Out | Closest to the real task of predicting a new, undrilled area | Centre is a random well, radius 1000 m: all wells inside the circle go into the test set, and their number varies from repeat to repeat |
| Stratified by {колонка} | `LeaveOneGroupOut` by the chosen categorical column (any column with at least two values; by default tectonic_zone, or horizon_id if absent). Each category in turn is the whole test set, training is on the others | Checks whether the model transfers to a new zone or horizon | 4 tectonic zones give 4 folds: n_test is the number of wells in the zone, n_train is all the others |
| Distance-based exclusion CV | For each well i in turn: all wells within excl_radius of it are removed from training (excl_radius = median pairwise distance between wells × variogram_factor), the test set is well i alone. Folds are single-well, so the std of RMSE across folds is the spread of absolute residuals per point. If fewer than degree + 2 points remain in training, the fold is skipped and marked in the table, the prediction is NaN | The strictest check for spatial leakage, essentially LOWO with a buffer | 100 folds with one well each: n_train is the wells beyond excl_radius. Wells in the dense centre of the cloud have a small n_train, and such folds are marked as skipped |

*Parameters k, grid_n, buffer, p_percent, radius, variogram_factor and repeats are sliders in the sidebar; the recommended value is shown next to each. The recommended buffer equals 0.5 × the median distance between wells and is shared by Buffered CV and Random K-Fold (buffered). excl_radius for Distance-based is set either as median × variogram_factor or as an absolute value in metres (absolute mode). X and Y coordinates are assumed to be in metres; buffer, radius and excl_radius are in the same units. When several horizons are selected, all schemes split whole wells: the same well cannot be in training and test at once, and its records across horizons always stay in one group.*

---

## Fitted curves

Besides the straight line, one of 17 curves can be overlaid on the cross-plot — from simple polynomials to methods robust to outliers, plus quantile regression (τ = 0.1 / 0.5 / 0.9):

| Method | General form | When it is useful |
|---|---|---|
| Linear | $y = a + bx$ | Basic linear relationship |
| Quadratic | $y = a + bx + cx^2$ | A relationship with one bend |
| Cubic | $y = a + bx + cx^2 + dx^3$ | A relationship with an inflection point |
| Polynomial Degree 4–10 | $y = a_0 + a_1x + \dots + a_nx^n$ | A more complex form, rarely justified |
| Power Law | $y = a x^b$ &nbsp;($x,y>0$) | Power-law dependence |
| Exponential | $y = a e^{bx}$ &nbsp;($y>0$) | Exponential growth or decay |
| Logarithmic | $y = a + b\ln x$ &nbsp;($x>0$) | Saturation or slowing growth |
| Robust (Huber) | $y = a + bx$; coefficients minimise the Huber loss | Linear relationship robust to outliers |
| Theil-Sen | $y = a + bx$; $b = \mathrm{median}\left(\dfrac{y_j-y_i}{x_j-x_i}\right)$ | Linear relationship robust to outliers |
| RANSAC | $y = a + bx$; fit by the maximum number of inlier points | Linear relationship with strong outliers or anomalies |
| LOWESS | no single formula — local regression in a sliding window | Follows the shape of the point cloud without assuming its form |
| Quantile Regression (τ=0.1/0.5/0.9) | $y = a_\tau + b_\tau x$ | Three lines for the lower, middle and upper spread boundaries |

*Domain: the power and exponential fits do not use points with y ≤ 0 (for example H_eff = 0 at intervals without reservoir); the power and logarithmic fits do not use points with x ≤ 0 (for example negative avo_intercept values). Such points are excluded from the fit of these curves only; the other curves use all points, including these.*

*For the curves τ = 0.1 and τ = 0.9, R² can be negative — this is normal. Quantile regression minimises the "pinball loss" rather than the squared error of linear regression, so its curves describe the boundaries of the spread, not the centre of the cloud. Judge such curves by Equation and Bias rather than by R².*

---

## In-sample and out-of-sample in this project

**In-sample** — the model is fitted and checked on the same wells. The metrics show how well the model describes the data it has already seen, so they are usually optimistic. The in-sample equation is an ordinary linear regression over all wells of the chosen sample.

**Out-of-sample** — the model is fitted without some wells and checked on the wells it did not see during fitting (cross-validation by the LOWO, K-fold and other schemes). The metrics show the quality on new wells. A gap between in-sample and out-of-sample is a sign of overfitting. In the Out-of-sample tables the metrics and the equation Predicted = a + b·Actual come from the same out-of-fold predictions.

| Step | What is computed | Mode |
|---|---|---|
| 0 | Pearson and Spearman correlations, descriptive statistics | no model |
| 1 | basic QC cross-plot, fitted curves, R²_lin, R²_fit, P-value, regression equation | in-sample |
| 2 | nine CV schemes, folds, choice of the best scheme, CV residual outliers, R² (Pearson), R² (Actual = Predicted), equation Predicted = a + b·Actual, fitted curves of the best scheme | out-of-sample |
| 3 | fitted curves and metrics inside each category | in-sample (within a category) |
| 4 | metrics before and after filtering, fitted curves after filtering | in-sample (on the filtered sample) |
| 4.5 | R² (Pearson) for series of random exclusion | in-sample (remaining wells) and out-of-sample (excluded wells) |
| 5 | single-attribute models and top-5 multi-attribute models | In-sample: RMSE_in, R²_in, r_in, equation over all wells; Out-of-sample: RMSE_cv, R²_cv, r_cv, equation Predicted = a + b·Actual |

## Cross-plot metrics and notation

Let Δ denote the residual. On the basic QC cross-plot (Step 1, BasicStats) Δ = y − x = PARAM_1 − PARAM_2. In fits and cross-validation metrics (Steps 1–4, FitStats and fold_metrics) Δ = Predicted − Actual. The Step 2 CV residual plots use Actual − Predicted. Bias on the Step 2 cross-plot (Predicted vs Actual) has the same sign as in the CV tables: Δ = Predicted − Actual. Squared and absolute metrics (RMSE, MSE, MAE, R²) do not depend on the sign, but the signed Bias changes sign if the convention changes.

**Limitation of the basic errors in Step 1.** Bias, RMSE, MSE, MAE, Std, Variance, WAPE % and the residual Δ = PARAM_1 − PARAM_2 depend on the scale and units of the parameters: H_eff is measured in metres, the attribute in its own units. So the values of these metrics can only be compared within one PARAM_1–PARAM_2 pair. The quality of the relationship is judged by R² and by the Pearson and Spearman correlation coefficients. Absolute error values must not be compared between different attributes or different reference parameters.

| Metric | Formula |
|---|---|
| Pearson r (correlation coefficient, Step 0) | $r = \dfrac{\mathrm{cov}(a,b)}{\sigma_a\,\sigma_b}$; not R², cannot be expressed through Δ |
| Spearman ρ (correlation coefficient, Step 0) | Pearson coefficient on ranks; not R², cannot be expressed through Δ |
| R²_lin (coefficient of determination, Steps 1 and 4) — **in-sample** | $R^2_{lin} = 1 - \dfrac{\sum(\Delta_{OLS})^2}{\sum(y-\bar y)^2} = r^2$, where $\Delta_{OLS} = y - (a + bx)$ is the residual of the linear regression (`linregress`); linear relationship only |
| R²_fit (fitted curves, Steps 1, 3, 4) — **in-sample** | $R^2_{fit} = 1 - \dfrac{\sum(\Delta)^2}{\sum(y-\bar y)^2}$, $\Delta = Predicted - Actual$; any curve shape |
| R²_fit (CV fits, Step 2) — **out-of-sample** | $R^2_{fit} = 1 - \dfrac{\sum(\Delta)^2}{\sum(y-\bar y)^2}$, $\Delta = model(Actual) - Predicted$; here $x = Actual$ (PARAM_1), $y = Predicted$, the roles deliberately differ from Step 1 |
| R² (Actual = Predicted), Step 2 — **out-of-sample** | $1 - \dfrac{\sum(\Delta)^2}{\sum(Actual-\overline{Actual})^2}$, $\Delta = Predicted - Actual$, on out-of-fold predictions (`r2_score`) |
| R² (Pearson, Predicted vs Actual), Step 2, pooled — **out-of-sample** | $1 - \dfrac{\sum(\Delta_{OLS})^2}{\sum(Actual-\overline{Actual})^2} = r^2$, $\Delta_{OLS} = Actual - (a + b\cdot Predicted)$ on out-of-fold predictions (`linregress`) |
| R² (Pearson) per fold, Step 2 — **out-of-sample** | the same as above, inside each fold; at least 3 points and non-zero variance are needed |
| R² (Pearson), in-sample, Step 4.5 | $1 - \dfrac{\sum(\Delta_{OLS})^2}{\sum(PARAM_1-\overline{PARAM_1})^2} = r^2$, $\Delta_{OLS} = PARAM_1 - (a + b\cdot PARAM_2)$ on the remaining wells |
| R² (Pearson), out-of-sample, Step 4.5 | $r^2 = corrcoef(PARAM_1, pred)^2$ on the excluded wells, $pred = a + b\cdot PARAM_2$ fitted on the remaining wells; cannot be expressed through Δ, because the line was fitted on another sample |
| RMSE (root mean squared error) | $\sqrt{\dfrac{1}{N}\sum (\Delta)^2}$ |
| MSE (mean squared error) | $\dfrac{1}{N}\sum (\Delta)^2$ |
| MAE (mean absolute error) | $\dfrac{1}{N}\sum \lvert \Delta \rvert$ |
| Bias (systematic offset) | $\dfrac{1}{N}\sum (\Delta)$ |
| Std (standard deviation of residuals) | $\sqrt{\dfrac{1}{N-1}\sum(\Delta_i-\bar\Delta)^2}$ |
| Variance (variance of residuals) | $\dfrac{1}{N-1}\sum(\Delta_i-\bar\Delta)^2$ |
| WAPE, % (weighted absolute percentage error) | $100 \cdot \dfrac{\sum \lvert \Delta \rvert}{\sum \lvert y \rvert}$ |
| P-value (significance of the linear correlation) — **in-sample** (Steps 0, 1, 4); **out-of-sample** (Step 2) | $t = r\sqrt{\dfrac{N-2}{1-r^2}}$, &nbsp; $p = 2\left(1-F_{t,\,N-2}(\lvert t\rvert)\right)$ |
| Linear regression equation | $y = a + bx$; slope b and intercept a from `linregress` |

*The P-value answers the question: how likely is it to obtain the observed linear relationship between PARAM_1 and PARAM_2 (or a stronger one) by chance, if in fact there is no linear dependence (null hypothesis H₀: the true slope b = 0). The smaller the p-value, the less reason to think the dependence is accidental. A relationship is usually called statistically significant at p < 0.05. This indicator is computed for the basic QC cross-plot (Step 1) and for the Step 2 cross-plot on out-of-fold predictions, with `scipy.stats.linregress`.*

---

## Outliers of the CV residuals

Outliers are sought in Step 2 among the CV residuals (Actual − Predicted) with four independent rules. Each point is checked against each rule separately, so the results can be compared:

| Rule | Outlier condition | Sidebar parameter |
|---|---|---|
| Sigma | $\lvert \Delta \rvert > k\sigma$, where σ is the sample standard deviation of the residuals (ddof = 1) | k = 1.0–6.0, default 2.0 |
| IQR | $\Delta < Q25 - k\cdot IQR$ or $\Delta > Q75 + k\cdot IQR$ | k = 1.0–3.0, default 1.5 |
| Modified z-score | $\lvert 0.6745\,(\Delta - \tilde{\Delta})/MAD \rvert > 3.5$, where $\tilde{\Delta}$ is the median of the residuals and MAD is the median absolute deviation; robust to extreme values | fixed threshold 3.5 |
| Quantile | $\Delta < P_{low}$ or $\Delta > P_{high}$ — the tails beyond the chosen percentiles | P_low 0–20%, default 5%; P_high 80–100%, default 95% |

*If σ or MAD is zero, the corresponding rule flags no points.*

---

## Parameter statistics

For H_eff and Phi (Step 0) and for every numeric parameter, extended descriptive statistics are shown. Here is what each quantity means:

| Statistic | Formula / definition |
|---|---|
| Count | N — number of non-empty values |
| μ (mean) | $\mu = \dfrac{1}{N}\sum x_i$ — arithmetic mean |
| median | the value that splits the sample in half (50th percentile) |
| mode | the most frequent value |
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

*The IQR bounds, RMSE and μ±3σ are shown in the Step 2 CV residual statistics, not in the descriptive statistics of the parameters. The IQR coefficient k is set with a slider in the sidebar.*

---

"""
        )

def show_input_file_example():

    with st.expander("📂 Input file format", expanded=False):

        st.markdown(r"""
Column names in the file can be anything: the correspondence is set during column mapping in the sidebar. The roles of the columns are described below.

- **well_id** — well identifier
- **X, Y** — well coordinates in metres
- **H_eff** — effective reservoir thickness (default unit — metres)
- **Phi** — porosity (default unit — fractions)
- **Seismic attributes** — any numeric columns: rms_amplitude, envelope, sweetness, coherence and others. PARAM_2 is chosen from them at the start of Step 1.

Additional columns:

- **horizon_id** — horizon; when several horizons are selected, each well is kept whole
- **tectonic_zone** — tectonic zone
- **well_category** — well category (used in Step 3)
- **reservoir_flag** — reservoir type
- **fluid_type** — fluid type; the value `non_reservoir` means there is no fluid where there is no reservoir
- any other numeric and categorical columns

Example: three wells from the demo file.
""")

        example_df = pd.DataFrame({
            "well_id": ["W-001", "W-002", "W-004"],
            "X": [7501.1, 10766.6, 2702.5],
            "Y": [9671.5, 6577.6, 5237.4],
            "horizon_id": ["Top_Reservoir", "Top_Reservoir", "Top_Seal"],
            "tectonic_zone": ["Central_Graben", "South_Block", "North_Block"],
            "H_eff": [5.47, 4.19, 0.0],
            "Phi": [0.223, 0.193, 0.062],
            "rms_amplitude": [0.1047, 0.0734, 0.0460],
            "envelope": [0.1312, 0.0901, 0.0575],
            "sweetness": [0.0342, 0.0235, 0.0150],
        })

        st.markdown("### Example file contents")

        st.dataframe(example_df, use_container_width=True)
