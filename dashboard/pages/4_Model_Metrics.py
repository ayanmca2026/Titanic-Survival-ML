import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px

st.set_page_config(page_title="Model Metrics", page_icon="📈", layout="wide")

st.title("📈 Model Metrics & Performance")

report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports", "model_comparison.json")
metrics_data = None

if os.path.exists(report_path):
    try:
        with open(report_path, 'r') as f:
            metrics_data = json.load(f)
    except:
        pass

if metrics_data:
    st.subheader("Model Comparison")
    df_metrics = pd.DataFrame(metrics_data).T
    st.dataframe(df_metrics, use_container_width=True)
    
    fig = px.bar(df_metrics, x=df_metrics.index, y="accuracy", template="plotly_dark", title="Model Accuracy Comparison", color_discrete_sequence=["#00D4FF"])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No model evaluation reports found. Run training script to generate metrics.")
    
    # Placeholder for UI presentation
    st.subheader("Model Comparison (Placeholder)")
    data = {
        "Random Forest": {"Accuracy": 0.82, "Precision": 0.80, "Recall": 0.75, "F1": 0.77},
        "Logistic Regression": {"Accuracy": 0.79, "Precision": 0.76, "Recall": 0.71, "F1": 0.73},
        "Gradient Boosting": {"Accuracy": 0.84, "Precision": 0.83, "Recall": 0.77, "F1": 0.80}
    }
    df = pd.DataFrame(data).T
    st.dataframe(df, use_container_width=True)
    
    fig = px.bar(df.reset_index(), x="index", y="Accuracy", template="plotly_dark", title="Model Accuracy Comparison", color_discrete_sequence=["#00D4FF"], labels={'index': 'Model'})
    st.plotly_chart(fig, use_container_width=True)
    
st.markdown("<br><hr><center>Built with ❤️ using Streamlit</center>", unsafe_allow_html=True)
