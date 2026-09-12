"""
Part 3: Model Deployment using Streamlit [05 Marks]
- Build and deploy a simple web app that uses all the machine learning models implemented in Part 2 to make predictions.
- Uses the Python streamlit library:
  1. Model Preparation: Loads previously trained models saved using joblib.
  2. Streamlit App: app.py loads the saved models.
  3. User Interface: Uses Streamlit widgets (st.slider, st.button, st.selectbox, etc.) to create an input form.
  4. Prediction: When user clicks 'Predict', generates and displays prediction in a clear, user-friendly format.
"""

from pathlib import Path
import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom UI Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Freight Cost Prediction Web App",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 2px solid #3B82F6;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 15px 0;
    }
    .result-val {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1D4ED8;
    }
    .result-lbl {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1E40AF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge {
        display: inline-block;
        background-color: #F3F4F6;
        color: #374151;
        border: 1px solid #D1D5DB;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Model Preparation: Load Previously Trained Models & Scaler
# -----------------------------------------------------------------------------
MODELS_CONFIG = {
    "Linear Regression": "linear_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
    "Random Forest (Tuned)": "random_forest_tuned.joblib",
}

def get_models_directory() -> Path:
    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir / "models",
        Path("models").resolve(),
        Path("ADSFA2/models").resolve(),
    ]
    for d in candidates:
        if d.exists() and (d / "scaler.joblib").exists():
            return d
    return current_dir / "models"

@st.cache_resource
def load_models_and_scaler():
    models_dir = get_models_directory()
    scaler_path = models_dir / "scaler.joblib"

    # Self-training fallback if models are ever missing
    if not scaler_path.exists():
        import subprocess
        train_script = Path(__file__).resolve().parent / "src" / "train.py"
        if train_script.exists():
            subprocess.run([sys.executable, str(train_script)], check=True)
            models_dir = get_models_directory()

    scaler = joblib.load(models_dir / "scaler.joblib")
    loaded_models = {}
    for display_name, file_name in MODELS_CONFIG.items():
        model_file = models_dir / file_name
        if model_file.exists():
            loaded_models[display_name] = joblib.load(model_file)
        elif (models_dir / "best_model.joblib").exists() and "Tuned" in display_name:
            loaded_models[display_name] = joblib.load(models_dir / "best_model.joblib")

    return scaler, loaded_models

scaler, models = load_models_and_scaler()

# -----------------------------------------------------------------------------
# 3. User Interface: Header
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>📦 Vendor Freight Cost Prediction Web App</div>", unsafe_allow_html=True)
st.write("")

# -----------------------------------------------------------------------------
# Quick Scenario Presets (Main Screen)
# -----------------------------------------------------------------------------
st.markdown("### ⚡ Quick Scenario Presets")
c_p1, c_p2, c_p3 = st.columns(3)
preset = None
with c_p1:
    if st.button("📦 Small Order ($1,200 | 80 Units)", use_container_width=True):
        preset = {"dollars": 1200.0, "quantity": 80, "days": 5}
with c_p2:
    if st.button("🏢 Standard Order ($15,000 | 1,200 Units)", use_container_width=True):
        preset = {"dollars": 15000.0, "quantity": 1200, "days": 14}
with c_p3:
    if st.button("🚢 Bulk Enterprise Order ($50,000 | 4,000 Units)", use_container_width=True):
        preset = {"dollars": 50000.0, "quantity": 4000, "days": 21}

if preset:
    st.session_state["dollars"] = preset["dollars"]
    st.session_state["quantity"] = preset["quantity"]
    st.session_state["days"] = preset["days"]
    st.rerun()

st.write("")

# -----------------------------------------------------------------------------
# 3. User Interface: Widgets (st.selectbox, st.slider, st.button)
# -----------------------------------------------------------------------------
st.subheader("1. Select Machine Learning Model & Input Features")

col_model, col_info = st.columns([2, 2])

with col_model:
    model_options = [
        "Random Forest (Tuned)",
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "All Models (Compare All)",
    ]
    selected_model_option = st.selectbox(
        "Choose Model for Prediction (st.selectbox):",
        options=model_options,
        index=0,
        help="Choose any of the models implemented in Part 2, or compare all of them at once."
    )

with col_info:
    if selected_model_option == "All Models (Compare All)":
        st.info("ℹ️ **Mode**: Predicting using **all models** simultaneously for comparison.")
    else:
        st.info(f"ℹ️ **Selected Model**: **{selected_model_option}**")

st.markdown("### Enter Feature Values (st.slider):")

col1, col2, col3 = st.columns(3)

with col1:
    dollars_val = st.slider(
        "💰 Total Invoice Dollars ($)",
        min_value=100.0,
        max_value=100000.0,
        value=float(st.session_state.get("dollars", 15000.0)),
        step=250.0,
        help="Total monetary value of the invoice."
    )

with col2:
    quantity_val = st.slider(
        "📦 Shipment Quantity (Units)",
        min_value=1,
        max_value=10000,
        value=int(st.session_state.get("quantity", 1200)),
        step=25,
        help="Number of physical units shipped."
    )

with col3:
    days_val = st.slider(
        "⏱️ Lead Time: PO to Invoice (Days)",
        min_value=0,
        max_value=60,
        value=int(st.session_state.get("days", 14)),
        step=1,
        help="Turnaround days between PO creation and Invoice date."
    )

unit_price = dollars_val / max(quantity_val, 1)
st.caption(f"Calculated Unit Price: **${unit_price:.2f} / unit**")

st.write("")

# -----------------------------------------------------------------------------
# 4. Prediction: When user clicks 'Predict' (st.button)
# -----------------------------------------------------------------------------
predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    # Construct feature DataFrame matching model training pipeline
    input_df = pd.DataFrame({
        "Quantity": [float(quantity_val)],
        "Dollars": [float(dollars_val)],
        "days_po_to_invoice": [float(days_val)],
        "Price_per_Unit": [float(unit_price)],
    })

    # Scale inputs using fitted StandardScaler
    scaled_input = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    st.subheader("2. Prediction Result")

    # Generate predictions across models
    preds_dict = {}
    for m_name, model_obj in models.items():
        pred_val = float(model_obj.predict(scaled_input)[0])
        preds_dict[m_name] = max(0.0, round(pred_val, 2))

    if selected_model_option == "All Models (Compare All)":
        st.markdown(f"""
        <div class='result-box'>
            <div class='result-lbl'>Best Model Forecast: Random Forest (Tuned)</div>
            <div class='result-val'>${preds_dict.get('Random Forest (Tuned)', 0.0):,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        chosen_pred = preds_dict.get(selected_model_option, 0.0)
        freight_ratio = (chosen_pred / dollars_val) * 100.0
        freight_per_unit = chosen_pred / max(quantity_val, 1)

        st.markdown(f"""
        <div class='result-box'>
            <div class='result-lbl'>Predicted Freight Cost ({selected_model_option})</div>
            <div class='result-val'>${chosen_pred:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Estimated Freight Cost", f"${chosen_pred:,.2f}")
        with kpi2:
            st.metric("Freight % of Total Dollars", f"{freight_ratio:.2f}%")
        with kpi3:
            st.metric("Freight per Unit", f"${freight_per_unit:.3f} / unit")

    # Display comparison across all models
    st.write("")
    st.markdown("### 📊 Predictions Across All Models:")
    
    comp_rows = []
    for m_name, pred_val in preds_dict.items():
        comp_rows.append({
            "Model Name": m_name,
            "Predicted Freight ($)": f"${pred_val:,.2f}",
            "Freight % of Value": f"{(pred_val / dollars_val) * 100.0:.2f}%",
            "Cost per Unit": f"${pred_val / max(quantity_val, 1):.3f}",
            "Selected": "👉 Active" if m_name == selected_model_option else ""
        })

    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)


