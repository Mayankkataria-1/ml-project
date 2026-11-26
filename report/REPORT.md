# Temperature Analysis and Prediction for India (1901–2021)
## Abstract
This study compares seven year-only regression approaches for five annual and seasonal temperature series. All selected models beat the historical training mean, but all have negative holdout R² in the locally executed extension.
## Dataset and methodology
121 annual observations; YEAR is the only predictor. Five targets are ANNUAL, JAN-FEB, MAR-MAY, JUN-SEP and OCT-DEC (°C). Numeric conversion, missing-value checks, sorting, duplicate detection and consecutive-year validation are applied. Training: 1901–2001; testing: 2002–2021. Expanding validation windows: 1987–1991, 1992–1996 and 1997–2001. Mean validation RMSE selects parameters and model family.
## Algorithms
Linear regression, polynomial Ridge, piecewise Ridge (1981 hinge), KNN, RBF SVR, Random Forest and Gradient Boosting; historical mean baseline. Scaling is fitted within training folds.
## Local extension results
| Series | Selected model | MAE | RMSE | R² | Baseline RMSE |
|---|---|---:|---:|---:|---:|
| ANNUAL | Piecewise Ridge | 0.144 | 0.198 | -0.181 | 0.613 |
| JAN-FEB | SVR | 0.377 | 0.491 | -0.473 | 0.773 |
| MAR-MAY | SVR | 0.341 | 0.404 | -0.488 | 0.667 |
| JUN-SEP | KNN | 0.247 | 0.307 | -1.234 | 0.483 |
| OCT-DEC | Polynomial Ridge 3 | 0.216 | 0.274 | -0.457 | 0.776 |

Run run_experiments.py to reproduce the tables and graphs. GitHub CI verification is separate.
## Historical trends
Full-period fitted warming per decade: annual 0.0628°C; Jan-Feb 0.0705°C; Mar-May 0.0572°C; Jun-Sep 0.0431°C; Oct-Dec 0.0896°C. These are associations, not causal attribution.
## Limitations and conclusion
Only 121 observations and one predictor; previously examined holdout, small validation windows and no physical climate covariates. Negative R² is not percentage accuracy. Tree models, KNN and RBF SVR have limited extrapolation. Results demonstrate time-aware evaluation rather than operational climate forecasting. Original documentation and AI disclosure are archived; ChatGPT assisted with this extension.
