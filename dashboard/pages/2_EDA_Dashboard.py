import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "titanic.csv")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    try:
        return pd.read_csv(url)
    except:
        return None

df = load_data()

st.title("📊 EDA Dashboard")

if df is not None:
    tab1, tab2, tab3 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(df, x="Survived", color="Survived", template="plotly_dark", title="Survival Count", color_discrete_sequence=["#FF4B4B", "#00FF88"])
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.histogram(df, x="Age", template="plotly_dark", title="Age Distribution", marginal="box", color_discrete_sequence=["#00D4FF"])
            st.plotly_chart(fig2, use_container_width=True)
            
        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.histogram(df, x="Fare", template="plotly_dark", title="Fare Distribution", marginal="box", color_discrete_sequence=["#FFB800"])
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            fig4 = px.bar(df['FamilySize'].value_counts().reset_index(), x='FamilySize', y='count', template='plotly_dark', title="Family Size Distribution", color_discrete_sequence=["#636EFA"])
            st.plotly_chart(fig4, use_container_width=True)
            
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig5 = px.histogram(df, x="Sex", color="Survived", barmode="group", template="plotly_dark", title="Gender vs Survival", color_discrete_sequence=["#FF4B4B", "#00FF88"])
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            fig6 = px.histogram(df, x="Pclass", color="Survived", barmode="group", template="plotly_dark", title="Pclass vs Survival", color_discrete_sequence=["#FF4B4B", "#00FF88"])
            st.plotly_chart(fig6, use_container_width=True)
            
        col3, col4 = st.columns(2)
        with col3:
            fig7 = px.violin(df, y="Age", x="Pclass", color="Survived", box=True, template="plotly_dark", title="Age by Pclass & Survival", color_discrete_sequence=["#FF4B4B", "#00FF88"])
            st.plotly_chart(fig7, use_container_width=True)
        with col4:
            fig8 = px.box(df, x="Pclass", y="Fare", template="plotly_dark", title="Fare by Pclass", color_discrete_sequence=["#00D4FF"])
            st.plotly_chart(fig8, use_container_width=True)
            
        col5, col6 = st.columns(2)
        with col5:
            fig9 = px.histogram(df, x="Embarked", color="Survived", barmode="stack", template="plotly_dark", title="Embarked vs Survival", color_discrete_sequence=["#FF4B4B", "#00FF88"])
            st.plotly_chart(fig9, use_container_width=True)
            
    with tab3:
        corr = df.select_dtypes(include='number').corr()
        fig10 = px.imshow(corr, text_auto=True, template="plotly_dark", title="Correlation Heatmap", color_continuous_scale="Viridis")
        st.plotly_chart(fig10, use_container_width=True)
else:
    st.error("Failed to load data.")
    
st.markdown("<br><hr><center>Built with ❤️ using Streamlit</center>", unsafe_allow_html=True)
