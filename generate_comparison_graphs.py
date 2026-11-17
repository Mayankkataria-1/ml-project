"""
Generate comparison graphs from model_comparison_TEMPLATE.csv.

IMPORTANT:
- This is a template, not measured project output.
- Enter the real MAE, RMSE and R2 values in the CSV first.
- Do not submit empty/template graphs as experimental results.
Run: python generate_comparison_graphs.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_file = Path("model_comparison_TEMPLATE.csv")
df = pd.read_csv(csv_file)

metrics = ["MAE", "RMSE", "R2"]
for metric in metrics:
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

valid = df.dropna(subset=metrics, how="all")
if valid.empty:
    raise SystemExit(
        "No real metrics found. Fill MAE, RMSE and R2 in "
        "model_comparison_TEMPLATE.csv, then run this script again."
    )

for metric in metrics:
    plot_data = valid.dropna(subset=[metric])
    if plot_data.empty:
        continue
    plt.figure(figsize=(9, 5))
    plt.bar(plot_data["Model"], plot_data[metric])
    plt.title(f"Model Comparison - {metric}")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{metric.lower()}_comparison.png", dpi=200)
    plt.close()

print("Graphs saved for metrics with available values.")
