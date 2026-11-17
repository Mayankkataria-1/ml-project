# Machine Learning Application Project: Temperature Analysis and Prediction

## Project Overview
This project explores historical temperature data and applies machine-learning regression models to study temperature trends and generate predictions.

The notebook contains data preparation, exploratory analysis, and experiments with:
- Linear Regression
- Polynomial Regression
- Ridge Regression
- Time-series cross-validation

## Objectives
1. Load and inspect the temperature dataset.
2. Clean and prepare the data for analysis.
3. Explore annual and seasonal temperature patterns.
4. Train and compare regression models.
5. Evaluate model performance using appropriate validation methods.

## Repository Contents
- `Temperature ML Project.ipynb` — main Jupyter Notebook.
- `TEMP_ANNUAL_SEASONAL_MEAN.csv` — dataset (if this is the exact filename in the repository).
- `requirements.txt` — Python dependencies.

## Requirements
Python 3.9 or later is recommended. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## How to Run
1. Clone or download this repository.
2. Install the dependencies.
3. Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab.
4. Make sure the CSV dataset is in the path expected by the notebook.
5. Run the notebook from top to bottom.

## Model Evaluation
Do not report training-set scores as evidence of future prediction performance. For a time-ordered dataset, keep observations in chronological order:
- Train on earlier years.
- Validate/tune using time-series cross-validation on the training period.
- Evaluate the selected model once on a later, untouched test period.

Report MAE, RMSE, and R² for the held-out period. If R² is negative, report it honestly and discuss possible reasons, such as limited data, weak predictors, or a model that does not generalize.

## Reproducibility
- Keep the original dataset and notebook together, or update the notebook's data path.
- Record the Python and package versions used.
- Set random seeds wherever randomized procedures are used.
- Do not manually edit reported metrics; regenerate them from the notebook.

## Dataset and References
**Important:** Add the original dataset source, publisher/organisation, download URL, access date, and any licence/attribution requirements here before final submission. The repository materials available when this README was prepared did not establish the dataset's original source, so no source has been guessed.

## Limitations
A model's ability to fit historical observations does not guarantee accurate future forecasts. The conclusions should be limited to what the dataset and the held-out evaluation support.

## Author
Add the project team's names and contributions here.
