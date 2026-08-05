import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Dataset Explorer", page_icon="🔍", layout="wide")

st.title("🔍 Dataset Explorer")

@st.cache_data
def load_data():
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "titanic.csv")
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
    st.subheader("Data Browser")
    st.dataframe(df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Info")
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index().rename(columns={'index': 'Column'})
        st.dataframe(dtypes_df, use_container_width=True)
    with col2:
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0].reset_index()
        missing.columns = ['Feature', 'Missing Count']
        if not missing.empty:
            fig = px.bar(missing, x='Feature', y='Missing Count', template='plotly_dark', color_discrete_sequence=['#00D4FF'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found!")
            
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="titanic.csv", mime="text/csv")
    
st.markdown("<br><hr><center>Built with ❤️ using Streamlit</center>", unsafe_allow_html=True)
