# Temperature Analysis and Prediction for India (1901–2021)

Machine learning project by Mayank Kataria, B.Sc. (H) Computer Science, Ramanujan College, University of Delhi.

## Run
```bash
pip install -r requirements.txt
python run_experiments.py
python -m unittest discover -s tests
```

The extended experiment compares Linear Regression, Polynomial Ridge, Piecewise Ridge, KNN, SVR, Random Forest and Gradient Boosting against a training-mean baseline. Training uses 1901–2001; testing uses 2002–2021. Hyperparameters are selected by expanding chronological validation on training years only.

Generated tables and figures are in temperature_analysis/outputs; extended metrics are written to results.txt. See report/REPORT.md and DATA_DICTIONARY.md. Original documentation and AI disclosure are preserved in report/ORIGINAL_README.md. ChatGPT assisted with the extension.

The dataset has 121 observations and one predictor (YEAR). Results are retrospective, not independent climate forecasts. The holdout was previously examined. Negative R² values are retained.
