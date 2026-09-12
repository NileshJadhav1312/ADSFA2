"""
Data preprocessing and feature engineering pipeline for Freight Cost Prediction.
Handles data ingestion from SQLite, date parsing, feature engineering,
IQR outlier capping, and feature normalization.
"""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def resolve_db_path() -> Path:
    """Resolve the location of inventory.db across potential directory levels."""
    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir.parent / "data" / "inventory.db",
        Path("data/inventory.db").resolve(),
        current_dir.parent.parent / "data" / "inventory.db",
        current_dir.parent.parent / "notebooks" / "inventory.db",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"inventory.db not found in candidates: {candidates}")


def load_data(db_path: str = None) -> pd.DataFrame:
    """Load vendor invoice records from SQLite database."""
    if db_path is None:
        db_path = str(resolve_db_path())
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM vendor_invoice", conn)
    conn.close()
    return df


def cap_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Cap extreme outliers to 1.5 * IQR bounds to preserve distribution integrity."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering:
    - days_po_to_invoice: Turnaround duration between PO issuance and invoice generation
    - Price_per_Unit: Monetary density (Dollars / Quantity)
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["PODate"] = pd.to_datetime(df["PODate"], errors="coerce")
    df["days_po_to_invoice"] = (df["InvoiceDate"] - df["PODate"]).dt.days

    # Drop records with invalid or missing date calculations
    df = df.dropna(subset=["Quantity", "Dollars", "Freight", "days_po_to_invoice"])

    # Monetary value density per shipped unit
    df["Price_per_Unit"] = df["Dollars"] / df["Quantity"].replace(0, np.nan)
    median_unit_price = df["Price_per_Unit"].median()
    df["Price_per_Unit"] = df["Price_per_Unit"].fillna(median_unit_price)

    return df


def preprocess_pipeline(db_path: str = None, test_size: float = 0.2, random_state: int = 42):
    """
    Complete end-to-end preprocessing pipeline:
    Loads raw records, engineers features, applies IQR capping, splits data,
    fits and applies StandardScaler.
    """
    raw_df = load_data(db_path)
    df = engineer_features(raw_df)

    # Apply IQR capping on continuous numeric features
    numeric_cols = ["Quantity", "Dollars", "days_po_to_invoice", "Price_per_Unit", "Freight"]
    for col in numeric_cols:
        df = cap_outliers_iqr(df, col)

    feature_cols = ["Quantity", "Dollars", "days_po_to_invoice", "Price_per_Unit"]
    target_col = "Freight"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, df
