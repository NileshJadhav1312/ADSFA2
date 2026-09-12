"""
Inference module for Freight Cost Prediction.
Loads saved regression model artifacts and scaler to generate real-time predictions.
Supports single-model and multi-model comparison predictions.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd


MODEL_SLUGS = {
    "Random Forest (Tuned)": "random_forest_tuned.joblib",
    "Linear Regression": "linear_regression.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
    "Random Forest": "random_forest.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Best Model": "best_model.joblib",
}


def get_model_directory() -> Path:
    """Locate model directory across possible runtime locations."""
    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir.parent / "models",
        Path("models").resolve(),
        Path("ADSFA2/models").resolve(),
    ]
    for c in candidates:
        if c.exists() and (c / "scaler.joblib").exists():
            return c
    raise FileNotFoundError(f"Models directory not found in candidates: {candidates}")


def load_scaler(model_dir: Path = None):
    """Load the trained StandardScaler."""
    if model_dir is None:
        model_dir = get_model_directory()
    return joblib.load(model_dir / "scaler.joblib")


def load_model(model_name: str, model_dir: Path = None):
    """Load a specific candidate model from disk."""
    if model_dir is None:
        model_dir = get_model_directory()
    filename = MODEL_SLUGS.get(model_name, "best_model.joblib")
    return joblib.load(model_dir / filename)


def prepare_input(quantity: float, dollars: float, days_po_to_invoice: float) -> pd.DataFrame:
    """Construct structured DataFrame with engineered features matching model training."""
    price_per_unit = dollars / max(quantity, 1e-6)
    return pd.DataFrame({
        "Quantity": [float(quantity)],
        "Dollars": [float(dollars)],
        "days_po_to_invoice": [float(days_po_to_invoice)],
        "Price_per_Unit": [float(price_per_unit)],
    })


def predict_freight(quantity: float, dollars: float, days_po_to_invoice: float = 15,
                    model_name: str = "Random Forest (Tuned)") -> float:
    """Predict freight cost using the specified model."""
    model_dir = get_model_directory()
    scaler = load_scaler(model_dir)
    model = load_model(model_name, model_dir)

    raw_input = prepare_input(quantity, dollars, days_po_to_invoice)
    scaled_input = pd.DataFrame(scaler.transform(raw_input), columns=raw_input.columns)
    prediction = float(model.predict(scaled_input)[0])
    return max(0.0, round(prediction, 2))


def predict_all_models(quantity: float, dollars: float, days_po_to_invoice: float = 15) -> dict:
    """Run inference across all 4 candidate models plus the tuned model."""
    model_dir = get_model_directory()
    scaler = load_scaler(model_dir)
    raw_input = prepare_input(quantity, dollars, days_po_to_invoice)
    scaled_input = pd.DataFrame(scaler.transform(raw_input), columns=raw_input.columns)

    predictions = {}
    for name, slug in MODEL_SLUGS.items():
        if name == "Best Model":
            continue
        model_path = model_dir / slug
        if model_path.exists():
            model = joblib.load(model_path)
            pred = float(model.predict(scaled_input)[0])
            predictions[name] = max(0.0, round(pred, 2))

    return predictions


if __name__ == "__main__":
    test_qty = 1500
    test_dollars = 25000.0
    test_days = 14
    print("Inference Test:")
    print(f"Input: Quantity={test_qty}, Dollars=${test_dollars:,.2f}, Days={test_days}")
    results = predict_all_models(test_qty, test_dollars, test_days)
    for model, pred in results.items():
        print(f"  {model:25s}: ${pred:,.2f}")
