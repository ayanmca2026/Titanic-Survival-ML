import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Predict Passenger Survival")

# API Configuration
st.sidebar.subheader("🌐 API Configuration")
api_url_default = os.getenv("API_URL", "http://localhost:8000")
api_url = st.sidebar.text_input("FastAPI Backend URL", value=api_url_default, help="Enter your deployed Render API URL (e.g. https://titanic-survival-api.onrender.com)")

# Test API Connection
api_status = False
try:
    health_resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=3)
    if health_resp.status_code == 200:
        api_status = True
        st.sidebar.success("✅ Connected to FastAPI Backend")
    else:
        st.sidebar.warning("⚠️ API returned non-200 status")
except Exception:
    st.sidebar.info("ℹ️ Running in Local Model Mode")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Passenger Information")
    name = st.text_input("Passenger Name", "Mr. John Smith")
    pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3], index=2, help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
    sex = st.radio("Sex", ["male", "female"], index=0)
    age = st.slider("Age", 0.0, 90.0, 30.0, 0.5)
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
    fare = st.number_input("Fare ($)", 0.0, 600.0, 32.2)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x])
    
    predict_btn = st.button("🚀 Predict Survival Probability", type="primary", use_container_width=True)

with col2:
    st.subheader("Prediction Output & Explainability")
    if predict_btn:
        passenger_payload = {
            "Name": name,
            "Pclass": pclass,
            "Sex": sex.lower(),
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked
        }

        prob = None
        survived = None
        mode_used = ""

        # Try API first
        if api_status:
            try:
                with st.spinner("Requesting prediction from FastAPI REST API..."):
                    response = requests.post(f"{api_url.rstrip('/')}/predict", json=passenger_payload, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        prob = data.get("survival_probability")
                        survived = data.get("survived")
                        mode_used = "FastAPI Backend (Render Cloud / Local REST API)"
            except Exception as e:
                st.error(f"API Error: {e}")

        # Fallback to local model pipeline if API fails/unavailable
        if prob is None:
            try:
                model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "best_model.joblib")
                pipe_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "preprocessing_pipeline.joblib")
                if os.path.exists(model_path) and os.path.exists(pipe_path):
                    model = joblib.load(model_path)
                    pipe = joblib.load(pipe_path)
                    X_df = pd.DataFrame([passenger_payload])
                    X_proc = pipe.transform(X_df)
                    prob = float(model.predict_proba(X_proc)[0][1])
                    survived = prob >= 0.5
                    mode_used = "Local Serialized Model (.joblib)"
            except Exception:
                pass

        # Final fallback heuristic if no model loaded
        if prob is None:
            sex_score = 0.4 if sex == "female" else 0.0
            class_score = (4 - pclass) * 0.15
            age_score = -0.005 * age
            prob = max(0.05, min(0.95, sex_score + class_score + age_score + 0.2))
            survived = prob >= 0.5
            mode_used = "Heuristic Baseline Rule Engine"

        st.caption(f"⚡ Mode: **{mode_used}**")

        if survived:
            st.success(f"### 🟢 Prediction: SURVIVED ({prob:.1%})")
        else:
            st.error(f"### 🔴 Prediction: DID NOT SURVIVE ({prob:.1%})")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%", 'valueformat': ".1f"},
            title={'text': "Survival Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00D4FF" if survived else "#FF4B4B"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(255, 75, 75, 0.2)"},
                    {'range': [40, 60], 'color': "rgba(255, 184, 0, 0.2)"},
                    {'range': [60, 100], 'color': "rgba(0, 255, 136, 0.2)"}
                ]
            }
        ))
        fig.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("📂 Batch Passenger Prediction")
uploaded_file = st.file_uploader("Upload CSV File (containing Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)", type=["csv"])
if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:", batch_df.head())
    
    if st.button("Execute Batch Prediction", type="primary"):
        if api_status:
            try:
                passengers_list = batch_df.to_dict(orient="records")
                resp = requests.post(f"{api_url.rstrip('/')}/predict_batch", json={"passengers": passengers_list})
                if resp.status_code == 200:
                    results_data = resp.json()
                    preds = [p["survived"] for p in results_data["predictions"]]
                    probs = [p["survival_probability"] for p in results_data["predictions"]]
                    batch_df["Survived_Predicted"] = preds
                    batch_df["Survival_Probability"] = probs
                    st.dataframe(batch_df)
                    st.success(f"Batch prediction complete! Total: {len(batch_df)} passengers.")
            except Exception as e:
                st.error(f"Batch prediction error: {e}")
        else:
            st.warning("Connect to active FastAPI backend for batch prediction API endpoint.")

st.markdown("<br><hr><center>Titanic ML Architecture: Streamlit UI ➔ FastAPI REST API ➔ Stacking Classifier</center>", unsafe_allow_html=True)
