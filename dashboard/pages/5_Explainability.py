import streamlit as st
import os

st.set_page_config(page_title="Explainability", page_icon="🧠", layout="wide")

st.title("🧠 Model Explainability")
st.write("Understand how the model makes decisions using SHAP and other techniques.")

img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")

tab1, tab2 = st.tabs(["SHAP Values", "Feature Importance"])

with tab1:
    st.subheader("SHAP Summary Plot")
    st.write("SHAP (SHapley Additive exPlanations) values show how much each feature contributes to the prediction.")
    shap_path = os.path.join(img_dir, "shap_summary.png")
    if os.path.exists(shap_path):
        st.image(shap_path)
    else:
        st.info("SHAP plot not found. Run explainability script to generate it.")
        
with tab2:
    st.subheader("Global Feature Importance")
    st.write("Feature importance from the tree-based model.")
    feat_path = os.path.join(img_dir, "feature_importance.png")
    if os.path.exists(feat_path):
        st.image(feat_path)
    else:
        st.info("Feature importance plot not found. Run model training to generate it.")

st.markdown("<br><hr><center>Built with ❤️ using Streamlit</center>", unsafe_allow_html=True)
