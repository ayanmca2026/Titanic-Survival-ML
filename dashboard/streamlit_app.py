import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('''
<style>
    /* Glassmorphism cards */
    .stMetric {
        background: rgba(26, 31, 46, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 16px;
    }
    /* Gradient header */
    .main-header {
        background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 50%, #0E1117 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        margin-bottom: 2rem;
    }
    /* Custom metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 212, 255, 0.05));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    /* Plotly chart containers */
    .chart-container {
        background: rgba(26, 31, 46, 0.6);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1F2E 100%);
    }
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Smooth animations */
    .stApp { transition: all 0.3s ease; }
</style>
''', unsafe_allow_html=True)

st.sidebar.title("🚢 Titanic Dashboard")
st.sidebar.info("A premium interactive dashboard for Titanic Survival Prediction.")

st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🚢 Titanic Survival Prediction")
st.markdown("Explore the dataset, analyze features, and predict survival probabilities.")
st.markdown('</div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "titanic.csv")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

df = load_data()

if df is not None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Passengers", len(df))
    with col2:
        survival_rate = df['Survived'].mean() * 100
        st.metric("Overall Survival Rate", f"{survival_rate:.1f}%")
    with col3:
        st.metric("Features", len(df.columns))
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.markdown("---")
    
    st.subheader("Quick Overview")
    st.write("This dashboard provides an end-to-end view of the Titanic machine learning project.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><h4>Dataset Explorer</h4><p>Browse raw data and statistics</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h4>EDA Dashboard</h4><p>Interactive data visualizations</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h4>Prediction</h4><p>Predict survival probabilities</p></div>', unsafe_allow_html=True)

st.markdown("<br><hr><center>Built with ❤️ using Streamlit</center>", unsafe_allow_html=True)
