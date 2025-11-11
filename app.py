# ============================================
# Streamlit Panel Data Analysis App using MMQR
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.formula.api import quantreg
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ============================================
# App Configuration
# ============================================

st.set_page_config(page_title="Panel Data Analysis Dashboard", layout="wide")

# ============================================
# Section A: Data Upload
# ============================================

st.title("📊 Panel Data Analysis Dashboard (MMQR Framework)")

st.sidebar.header("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom data loaded successfully!")
else:
    st.info("No file uploaded. Using sample dataset (sample_data.csv).")
    np.random.seed(42)
    countries = ['Country_' + str(i) for i in range(1, 11)]
    years = list(range(2000, 2020))
    sample_data = []
    for country in countries:
        for year in years:
            gdp = np.random.normal(100, 20)
            tourism = gdp * 0.3 + np.random.normal(0, 5)
            sample_data.append({
                'Country': country,
                'Year': year,
                'GDP': gdp,
                'Tourism': tourism,
                'Investment': np.random.normal(50, 10),
                'Trade': np.random.normal(60, 15)
            })
    df = pd.DataFrame(sample_data)
    st.session_state["uploaded_data"] = df

# ============================================
# Section A1: Variable Distribution Explorer
# ============================================

st.header("A1. Variable Distribution Explorer")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if numeric_cols:
    selected_var = st.selectbox("Select a variable to visualize its distribution:", options=numeric_cols)
    color_choice = st.color_picker("Choose plot color", "#2E86AB")

    if selected_var:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Bar Plot
        sns.histplot(df[selected_var], bins=20, kde=False, color=color_choice, ax=axes[0,0])
        axes[0,0].set_title("Bar Plot")
        axes[0,0].set_xlabel(selected_var)
        
        # Box Plot
        sns.boxplot(x=df[selected_var], color=color_choice, ax=axes[0,1])
        axes[0,1].set_title("Box Plot")
        
        # Violin Plot
        sns.violinplot(x=df[selected_var], color=color_choice, ax=axes[1,0])
        axes[1,0].set_title("Violin Plot")
        
        # Strip Plot
        sns.stripplot(x=df[selected_var], color=color_choice, ax=axes[1,1], alpha=0.6)
        axes[1,1].set_title("Strip Plot")
        
        st.pyplot(fig)

        # Formal interpretation note
        mean_val = df[selected_var].mean()
        std_val = df[selected_var].std()
        skew_val = df[selected_var].skew()
        kurt_val = df[selected_var].kurtosis()

        interpretation = f"""
        **Distributional Summary of {selected_var}:**
        - Mean: {mean_val:.2f}, Standard Deviation: {std_val:.2f}
        - Skewness: {skew_val:.2f}, Kurtosis: {kurt_val:.2f}

        The variable **{selected_var}** displays a {'right' if skew_val>0 else 'left' if skew_val<0 else 'symmetrical'}-skewed distribution, 
        indicating {'a longer right tail and concentration of values below the mean' if skew_val>0 else 'a longer left tail and concentration of values above the mean' if skew_val<0 else 'approximate symmetry around the mean'}. 
        The kurtosis value of {kurt_val:.2f} suggests that the distribution_
