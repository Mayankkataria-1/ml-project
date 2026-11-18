# India Temperature Analysis (1901–2021) — upgraded workflow

This folder adds a reproducible, time-series-aware analysis workflow to the original project. The original root-level files are preserved.

## Research scope
- Dataset: All-India annual and seasonal mean temperature, 1901–2021.
- Source data: `../TEMP_ANNUAL_SEASONAL_MEAN.csv`.
- Targets: ANNUAL, JAN-FEB, MAR-MAY, JUN-SEP, OCT-DEC.
- Predictor: calendar year only.

## Methods
1. Data validation and cleaning (the source CSV is never modified).
2. Exploratory analysis: descriptive statistics, correlation matrix and figures.
3. Full-period linear trend estimates with HAC/Newey–West standard errors.
4. Chronological holdout: train 1901–2001; test 2002–2021.
5. Expanding-window model selection on three recent five-year validation windows inside training: 1987–1991, 1992–1996 and 1997–2001.
6. Candidate models: linear regression, polynomial Ridge (degrees 2 and 3), continuous piecewise Ridge with a prespecified 1981 breakpoint, and a training-mean baseline.
7. Residual diagnostics and exploratory extrapolation through 2035.

The validation windows are intentionally recent enough for the 1981 piecewise model to learn a post-break slope. Validation and test sets are small, so selection uncertainty remains.

## Run on Ubuntu/Linux/macOS
From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python temperature_analysis/run_analysis.py
```

Outputs are written to `temperature_analysis/outputs/` (tables, figures and a Markdown report). All generated values are calculated by the script; no result tables are hand-entered.

## Important limitations
- Only 121 annual observations are available.
- The model uses year as its sole predictor; it does not include greenhouse gases, aerosols, ocean conditions or other climate drivers.
- The 1981 breakpoint is prespecified, not automatically discovered.
- A national mean hides regional and state-level variation.
- Extrapolations are demonstrations of model behaviour, not official forecasts or physical climate projections.
- Negative test R-squared values must be reported honestly; they indicate that the model did not beat the test-period mean benchmark under that metric.
