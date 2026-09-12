"""
Model evaluation module for regression models.
Computes MAE, MSE, RMSE, R2 metrics and generates evaluation visualizations.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(model_name: str, y_true, y_pred) -> dict:
    """Compute regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2_Score": round(r2, 4),
    }


def plot_model_comparison(results_df: pd.DataFrame, save_dir: str = "images"):
    """Generate and save comparison bar plots across candidate models."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Plot 1: RMSE Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("viridis", len(results_df))
    bars = sns.barplot(x="Model", y="RMSE", data=results_df, ax=ax, palette=palette)
    ax.set_title("Model Comparison - Root Mean Squared Error (Lower is Better)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("RMSE ($)", fontsize=11)
    ax.set_xlabel("Model Name", fontsize=11)
    for p in bars.patches:
        ax.annotate(f"${p.get_height():.2f}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "comparison_RMSE.png", dpi=300)
    plt.close()

    # Plot 2: R2 Score Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("mako", len(results_df))
    bars = sns.barplot(x="Model", y="R2_Score", data=results_df, ax=ax, palette=palette)
    ax.set_title("Model Comparison - R² Goodness of Fit (Higher is Better)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("R² Score", fontsize=11)
    ax.set_xlabel("Model Name", fontsize=11)
    ax.set_ylim(0, 1.05)
    for p in bars.patches:
        ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "comparison_R2.png", dpi=300)
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, save_dir: str = "images"):
    """Scatter plot of actual vs predicted values with reference diagonal."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, color="#1f77b4", edgecolors="none", s=25)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Fit (y = x)")
    ax.set_title(f"Actual vs Predicted Freight ({model_name})", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Actual Freight ($)", fontsize=11)
    ax.set_ylabel("Predicted Freight ($)", fontsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "actual_vs_predicted.png", dpi=300)
    plt.close()
