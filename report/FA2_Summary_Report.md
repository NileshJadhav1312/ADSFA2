# Formative Assessment-02: Freight Cost Prediction Report

**Project Title**: End-to-End Machine Learning Pipeline & Interactive Deployment for Vendor Freight Cost Forecasting  
**Course / Assessment**: Formative Assessment-02 (ADSFA2)  
**Deliverables**:
1. Jupyter Notebook: [`notebooks/eda_and_modeling.ipynb`](../notebooks/eda_and_modeling.ipynb) (and root copy [`eda_and_modeling.ipynb`](../eda_and_modeling.ipynb))
2. Streamlit Application: [`app.py`](../app.py)
3. Summary Report: [`report/FA2_Summary_Report.md`](FA2_Summary_Report.md) & [`README.md`](../README.md)

---

## 1. Executive Summary & Business Objectives

In supply chain and enterprise procurement operations, freight and logistics expenses account for a significant portion of landed goods cost. Inaccurate freight estimates lead to distorted product margin calculations, inaccurate budgeting, and payment processing friction. 

This project develops an end-to-end supervised machine learning pipeline to forecast expected vendor invoice freight costs based on transaction volume, shipment quantity, order turnaround time, and price density. The solution includes a comprehensive exploratory analysis, rigorous data preprocessing, implementation and benchmarking of four regression algorithms, hyperparameter optimization via `GridSearchCV`, and deployment via an interactive Streamlit web dashboard.

---

## 2. Part 1: Data Acquisition & Preprocessing

### 2.1 Dataset Source & Download Link
The dataset is derived from the enterprise supply-chain database `inventory.db`, specifically targeting the `vendor_invoice` table:

- **📥 Dataset Download Link**: **[Google Drive Database Download Link](https://drive.google.com/file/d/1o6S9N0j77qM4fd9kbdE2gw2V1TVeh5Ey/view?usp=sharing)**
- **Total Records**: 5,543 vendor invoice headers
- **Target Variable**: `Freight` (Continuous numeric, represents shipping cost in USD)

### 2.2 Dataset Choice Justification
The `vendor_invoice` table captures actual commercial transactions between retail stores/warehouses and supplier vendors. Key fields include:
- `Dollars`: Total dollar value of the invoice
- `Quantity`: Total physical units delivered
- `InvoiceDate` and `PODate`: Timestamp markers of purchase order fulfillment
- `Freight`: Actual shipping cost paid

This dataset provides genuine historical commercial ground truth, making it the ideal foundation for training supervised regression models that forecast shipping costs for future purchase orders.

### 2.3 Exploratory Data Analysis (EDA) Highlights
1. **Target Distribution & Skewness**:
   - The raw `Freight` distribution displays severe positive right-skewness (skewness ≈ 3.8), with typical invoices incurring $10 to $150 in freight, while major bulk shipments reach several thousand dollars.
2. **Feature Correlation**:
   - `Dollars` exhibits an exceptionally strong Pearson correlation with `Freight` (**r = 0.985**).
   - `Quantity` exhibits strong correlation (**r = 0.947**).
3. **Shipping Economies of Scale**:
   - Shipments in the lowest quartile (Quantity ≤ 56 units) incur a median shipping rate of **$0.10 per unit**.
   - Bulk shipments in the upper quartile (Quantity ≥ 1,008 units) incur a median rate of **$0.04 per unit**.
   - Bulk logistics achieve a **~60% per-unit freight cost reduction**, establishing the necessity of including non-linear and tree-based regressors.

### 2.4 Preprocessing Steps
- **Handling Incomplete Records**: Records missing date markers or numerical attributes were parsed and verified.
- **Feature Engineering**:
  - `days_po_to_invoice`: Calculated turnaround duration `(InvoiceDate - PODate)`.
  - `Price_per_Unit`: Computed value density `(Dollars / Quantity)`.
- **Outlier Mitigation via IQR Capping**: Extreme values beyond $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$ were capped to preserve data volume while preventing gradient distortion.
- **Train/Test Splitting**: Partitioned using an 80/20 train/test holdout ratio with fixed seed (`random_state=42`).
- **Feature Standardization**: Scaled using `StandardScaler` fitted strictly on training data to prevent data leakage.

---

## 3. Part 2: Model Implementation, Tuning & Evaluation

### 3.1 Four Candidate Models Implemented
To thoroughly evaluate model families, four distinct regression algorithms were selected:

1. **Linear Regression (OLS)**: Ordinary Least Squares baseline linear model.
2. **Decision Tree Regressor**: Non-linear tree model capturing piecewise step relationships.
3. **Random Forest Regressor**: Ensemble bagging regressor combining multiple randomized decision trees to reduce variance.
4. **Gradient Boosting Regressor**: Sequential boosting ensemble that incrementally minimizes pseudo-residuals.

### 3.2 Hyperparameter Tuning
Hyperparameter optimization was performed on the **Random Forest Regressor** using `GridSearchCV` with **5-fold cross-validation** optimizing negative root mean squared error:
- Search space: `n_estimators: [50, 100]`, `max_depth: [6, 10, None]`, `min_samples_split: [2, 5]`.
- Optimal Configuration: `{'max_depth': 6, 'min_samples_split': 5, 'n_estimators': 50}`.

### 3.3 Evaluation Results & Comparative Analysis

| Model | MAE ($) | MSE | RMSE ($) | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **6.97** | **550.45** | **23.46** | **0.9866** |
| **Linear Regression** | 6.85 | 561.83 | 23.70 | 0.9863 |
| **Gradient Boosting** | 7.67 | 569.08 | 23.86 | 0.9861 |
| **Random Forest (Default)** | 7.50 | 631.72 | 25.13 | 0.9846 |
| **Decision Tree** | 8.14 | 927.16 | 30.45 | 0.9774 |

#### Key Insights:
- **Best Performer**: The **Tuned Random Forest** achieved the lowest test RMSE of **$23.46** and the highest coefficient of determination (**R² = 0.9866**), explaining over 98.6% of the variance in shipping costs.
- **Baseline Strength**: Linear Regression performed exceptionally well (RMSE: $23.70) due to the near-linear relationship between merchandise dollar volume and carrier billing scales.
- **Overfitting Prevention**: Unconstrained Decision Trees suffered higher error ($30.45 RMSE), while ensemble bagging and tree pruning effectively stabilized variance.

---

## 4. Part 3: Streamlit Web Deployment

The deployment script [`app.py`](../app.py) delivers an intuitive, interactive analytics dashboard:

1. **Model Selector**: Allows users to dynamically switch between all 4 candidate models plus the Tuned Random Forest model.
2. **Interactive Input Sliders & Inputs**:
   - Total Invoice Dollars ($)
   - Shipment Quantity (Units)
   - Lead Time / PO Turnaround (Days)
3. **Instant Live Predictions & KPI Metrics**:
   - Highlighted Freight Estimate ($)
   - Shipping Cost as % of Invoice Total
   - Freight Cost per Unit ($/unit)
4. **Multi-Model Real-Time Comparison Grid**:
   - Evaluates the user's input across all 5 models simultaneously in a formatted comparison table to assess model consensus.
5. **Sanity Advisory Alerts**:
   - Flags anomalies if the estimated freight proportion deviates from standard commercial bands (1% to 4.5%).
6. **Benchmark & Evaluation Tab**:
   - Embeds the performance comparison table and interactive evaluation visualizations (`comparison_RMSE.png`, `comparison_R2.png`, `actual_vs_predicted.png`).
7. **Robust Architecture**:
   - Utilizes `@st.cache_resource` for instant response times and automated model training fallback.

---

## 5. Part 4: Verification & Reproduction Guide

### 5.1 Environment Setup
```bash
cd ADSFA2
pip install -r requirements.txt
```

### 5.2 Training Models & Generating Artifacts
```bash
python src/train.py
```

### 5.3 Running the Streamlit Application
```bash
streamlit run app.py
```

### 5.4 Running the Jupyter Notebook
Open and run all cells in [`notebooks/eda_and_modeling.ipynb`](../notebooks/eda_and_modeling.ipynb):
```bash
jupyter notebook notebooks/eda_and_modeling.ipynb
```

---

## 6. Deliverables Checklist Summary

- [x] **Part 1**: Thorough EDA, missing value inspection, volume quartile analysis, IQR capping, scaling, and justified dataset selection with download link.
- [x] **Part 2**: Implementation of 4 distinct models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting), effective `GridSearchCV` tuning, metrics table (MAE, MSE, RMSE, R²), and visual comparisons.
- [x] **Part 3**: Polished Streamlit web application supporting model selection, multi-model predictions, KPI cards, and evaluation metrics.
- [x] **Part 4**: Clean, well-commented code, 3 submission files (notebook, app.py, summary report), and zero unnecessary markdown clutter.
