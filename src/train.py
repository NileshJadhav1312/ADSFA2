"""
Training pipeline for Freight Cost Prediction.
Trains 4 model candidates (Linear Regression, Decision Tree, Random Forest, Gradient Boosting),
performs GridSearchCV hyperparameter tuning, logs comparative performance benchmarks,
generates evaluation visualizations, and serializes model artifacts for Streamlit deployment.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

import data_preprocessing as dp
import model_evaluation as me


def run_training():
    print("=" * 65)
    print("Running Freight Cost Prediction Model Training Pipeline")
    print("=" * 65)

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    # Artifact storage directories
    models_dir = project_root / "models"
    images_dir = project_root / "images"

    models_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Preprocessing & Splitting
    print("\n[1/5] Loading data and executing preprocessing pipeline...")
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, full_df = dp.preprocess_pipeline()
    print(f"  [OK] Training records: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"  [OK] Test records:     {X_test.shape[0]}")

    # Persist scaler artifact
    joblib.dump(scaler, models_dir / "scaler.joblib")
    print("  [OK] Feature scaler persisted.")

    # 2. Define Candidate Models (4 Models)
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=6),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=60, max_depth=8),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=60, max_depth=4, learning_rate=0.08),
    }

    # 3. Train & Evaluate Candidates
    print("\n[2/5] Training 4 candidate regression models...")
    results = []
    fitted_models = {}

    for name, model in models.items():
        print(f"  Fitting {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = me.evaluate_regression(name, y_test, y_pred)
        results.append(metrics)
        fitted_models[name] = model

        # Save individual candidate model
        file_slug = name.replace(" ", "_").lower()
        joblib.dump(model, models_dir / f"{file_slug}.joblib")

    # 4. Hyperparameter Tuning via GridSearchCV
    print("\n[3/5] Executing GridSearchCV hyperparameter tuning on Random Forest...")
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [6, 10, None],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train_scaled, y_train)
    best_rf = grid.best_estimator_
    print(f"  [OK] Optimal parameters identified: {grid.best_params_}")

    y_pred_tuned = best_rf.predict(X_test_scaled)
    tuned_metrics = me.evaluate_regression("Random Forest (Tuned)", y_test, y_pred_tuned)
    results.append(tuned_metrics)
    fitted_models["Random Forest (Tuned)"] = best_rf

    joblib.dump(best_rf, models_dir / "random_forest_tuned.joblib")
    # Save as default best model
    joblib.dump(best_rf, models_dir / "best_model.joblib")

    # 5. Benchmark Comparison & Artifact Export
    print("\n[4/5] Compiling performance benchmark results...")
    results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    print("\n" + results_df.to_string(index=False))

    # Persist benchmark tables and plots
    results_df.to_csv(models_dir / "results_regression.csv", index=False)

    print("\n[5/5] Generating evaluation comparison plots...")
    me.plot_model_comparison(results_df, save_dir=str(images_dir))

    # Generate actual vs predicted plot using the tuned model
    me.plot_actual_vs_predicted(y_test, y_pred_tuned, "Random Forest (Tuned)", save_dir=str(images_dir))
    print(f"  [OK] Charts saved to {images_dir}")

    print("\n" + "=" * 65)
    print(f"Best Performing Model: {results_df.iloc[0]['Model']} with RMSE = ${results_df.iloc[0]['RMSE']:.2f}")
    print("=" * 65)

    return results_df


if __name__ == "__main__":
    run_training()
