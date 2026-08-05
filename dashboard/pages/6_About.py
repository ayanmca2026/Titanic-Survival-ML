import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About the Project")

st.markdown("""
### Titanic Survival Prediction
This is an end-to-end Machine Learning project demonstrating data analysis, model training, and a deployed Streamlit application.

### Tech Stack
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost
- **Visualization**: Plotly, Streamlit
- **Explainability**: SHAP

### Architecture
The project follows a standard ML pipeline structure:
1. Data Ingestion
2. Data Preprocessing & Feature Engineering
3. Model Training & Evaluation
4. Model Deployment (Streamlit)

### Contact
Built by an AI Assistant.
""")

st.markdown("<br><hr><center>Built with ❤️ using Streamlit</center>", unsafe_allow_html=True)
