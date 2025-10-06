import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go

# --- Page config ---
st.set_page_config(
    page_title="Medical Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# --- Custom Banner ---
st.markdown("""
    <style>
    .banner {
        background: linear-gradient(to right, #4B6EFF, #61FFDA);
        color: white;
        text-align: center;
        padding: 15px;
        font-size: 22px;
        font-weight: bold;
        border-radius: 8px;
        margin-bottom: 20px;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    </style>
    <div class="banner">🩺 Medical Risk Prediction Dashboard</div>
""", unsafe_allow_html=True)

# --- Intro Text ---
st.markdown("""
This tool predicts whether a patient is **multi-morbid** (has >2 medical conditions)
based on key **health** and **demographic** features.

You can:
- 🔹 Enter details manually for a **single patient**
- 🔹 Upload a **CSV file** for **batch predictions**

""")

# --- Load model and scaler ---
model = load_model('best_nn_model.h5')
scaler = joblib.load('scaler.pkl')

# --- Features ---
numeric_features = ['age','total_claims','num_conditions','num_meds','num_encounters']
categorical_features = ['gender','race']

# --- Sidebar Settings (no disclaimer now) ---
st.sidebar.title("⚙️ Settings")
smoothing_factor = st.sidebar.slider("Smoothing Factor", 0.5, 1.0, 0.9)

# --- Sigmoid smoothing ---
def smooth_prob(p, factor=0.95):
    return factor * p + (1 - factor) * 0.5

# --- Single Patient Input ---
st.subheader("📍 Single Patient Prediction")
with st.form("patient_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    total_claims = st.number_input("Total Claims", min_value=0, value=5)
    num_conditions = st.number_input("Number of Conditions", min_value=0, value=1)
    num_meds = st.number_input("Number of Medications", min_value=0, value=1)
    num_encounters = st.number_input("Number of Encounters", min_value=0, value=5)

    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])

    submitted = st.form_submit_button("🔍 Predict Risk")

if submitted:
    # Prepare dataframe
    data = pd.DataFrame([[age, total_claims, num_conditions, num_meds, num_encounters, gender, race]],
                        columns=numeric_features + categorical_features)

    # Encode categorical features
    data['gender'] = data['gender'].map({"Male": 0, "Female": 1})
    data['race'] = data['race'].map({"White": 0, "Black": 1, "Asian": 2, "Other": 3})

    # Scale numeric features
    data[numeric_features] = scaler.transform(data[numeric_features])

    # Predict
    raw_prob = model.predict(data)[0][0]
    pred_prob = smooth_prob(raw_prob, factor=smoothing_factor)
    pred_class = int(pred_prob > 0.5)

    # --- Styled Result ---
    if pred_class == 1:
        st.markdown(f"""
        <div style="background-color:#ffe6e6; padding:20px; border-radius:10px; border:1px solid red;">
            <h3 style="color:#cc0000;">⚠️ High Risk of Multi-Morbidity</h3>
            <p>Probability: <b>{pred_prob*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color:#e6ffe6; padding:20px; border-radius:10px; border:1px solid green;">
            <h3 style="color:#008000;">✅ Low Risk of Multi-Morbidity</h3>
            <p>Probability: <b>{pred_prob*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

    # --- Probability Gauge ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_prob*100,
        title={'text': "Risk Probability"},
        gauge={'axis': {'range': [0,100]},
               'bar': {'color': "red" if pred_class==1 else "green"}}
    ))
    st.plotly_chart(fig, use_container_width=True)

# --- Batch Prediction (Clean Version) ---
st.subheader("📂 Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Define features exactly as trained
    numeric_features = ['age','total_claims','num_conditions','num_meds','num_encounters']
    categorical_features = ['gender','race']
    required_cols = numeric_features + categorical_features

    # Check if all required columns exist
    if all(col in df.columns for col in required_cols):
        with st.spinner("Running predictions..."):

            # --- Encode categorical features ---
            df['gender'] = df['gender'].map({"Male": 0, "Female": 1})
            df['race'] = df['race'].map({"White": 0, "Black": 1, "Asian": 2, "Other": 3})

            # --- Ensure numeric column order matches training ---
            df_numeric = df[numeric_features].copy()

            # --- Scale numeric features ---
            df[numeric_features] = scaler.transform(df_numeric)

            # --- Generate probabilities ---
            if hasattr(model, "predict_proba"):  # scikit-learn / XGBoost
                raw_probs = model.predict_proba(df)[:, 1]  # probability of 'High'
            else:  # Keras / TensorFlow
                raw_probs = model.predict(df).flatten()
                # Apply sigmoid if NN output is not [0,1]
                if raw_probs.max() > 1 or raw_probs.min() < 0:
                    raw_probs = 1 / (1 + np.exp(-raw_probs))

            # --- Apply smoothing (optional) ---
            df['MultiMorbid_Probability'] = np.array(
                [smooth_prob(p, factor=smoothing_factor) for p in raw_probs]
            )

            # --- Label risk ---
            df['Risk'] = np.where(df['MultiMorbid_Probability'] > 0.5, 'High', 'Low')

        st.success("✅ Predictions Completed")
        st.dataframe(df)

        # --- Download option ---
        st.download_button(
            "📥 Download Predictions as CSV",
            df.to_csv(index=False),
            "predictions_corrected.csv",
            "text/csv"
        )

    else:
        st.error(f"❌ Uploaded CSV must contain these columns: {required_cols}")

