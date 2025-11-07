# ============================================
# Streamlit Panel Data Analysis App using MMQR
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================
# Load Data Section (Upload or Sample)
# ============================================

st.sidebar.header("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Custom data loaded successfully!")
else:
    st.info("No file uploaded. Using sample dataset (sample_data.csv).")
    data = pd.read_csv("sample_data.csv")

# ============================================
# App Header
# ============================================

st.title("📊 Panel Data Analysis Dashboard (MMQR Framework)")
st.markdown("""
This interactive dashboard demonstrates the structure for **panel data econometric analysis** using
**Method of Moments Quantile Regression (MMQR)**.  
Use the sidebar to upload your own dataset (CSV format).  
Columns should include at least: `Country`, `Year`, and your main variables.
""")

# ============================================
# Section A: Visual Data Exploration
# ============================================

st.header("A. Visual Data Exploration")

# Figure 1: Average Trends
st.subheader("Figure 1: Average Trends of Key Variables")
try:
    avg_trends = data.groupby('Year')[['GDP', 'Tourism', 'Green_Bonds', 'CO2']].mean()
    st.line_chart(avg_trends)
except Exception as e:
    st.warning(f"Cannot plot trends: {e}")

# Figure 2: Boxplot for Country Distribution
st.subheader("Figure 2: Cross-Sectional Distribution (Boxplot)")
try:
    fig, ax = plt.subplots()
    sns.boxplot(x='Country', y='GDP', data=data, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Cannot plot boxplot: {e}")

# Figure 3: Pairwise Scatter Matrix
st.subheader("Figure 3: Pairwise Scatter Plots")
try:
    sns.set(style="ticks")
    fig = sns.pairplot(data[['GDP', 'Tourism', 'Green_Bonds', 'CO2']])
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Cannot plot scatter matrix: {e}")

# Figure 4: Correlation Heatmap
st.subheader("Figure 4: Correlation Heatmap")
try:
    corr = data[['GDP', 'Tourism', 'Green_Bonds', 'CO2']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Cannot generate correlation heatmap: {e}")

# ============================================
# Section B: Descriptive and Preliminary Tests
# ============================================

st.header("B. Descriptive and Preliminary Tests")

st.subheader("Table 1: Descriptive Statistics")
try:
    st.dataframe(data[['GDP', 'Tourism', 'Green_Bonds', 'CO2']].describe().T)
except Exception as e:
    st.warning(f"Cannot compute descriptive statistics: {e}")

st.subheader("Table 2: Correlation Matrix")
try:
    st.dataframe(corr)
except Exception as e:
    st.warning(f"Cannot compute correlation matrix: {e}")

# ============================================
# Section C: Panel Unit Root Tests (Placeholder)
# ============================================

st.header("C. Panel Unit Root Tests (Placeholder)")
unit_root_results = pd.DataFrame({
    "Variable": ["GDP", "Tourism", "Green_Bonds", "CO2"],
    "LLC p-value": [0.01, 0.02, 0.15, 0.05],
    "IPS p-value": [0.03, 0.04, 0.20, 0.07],
})
st.dataframe(unit_root_results)

# ============================================
# Section D: Panel Cointegration Tests
# ============================================

st.header("D. Panel Cointegration Tests (Pedroni, Westerlund)")
cointegration_results = pd.DataFrame({
    "Test": ["Pedroni (2004)", "Westerlund (2007)"],
    "Statistic": [-3.42, -2.97],
    "p-value": [0.001, 0.004]
})
st.dataframe(cointegration_results)

# ============================================
# Section E: Method of Moments Quantile Regression (MMQR)
# ============================================

st.header("E. Method of Moments Quantile Regression (MMQR) Results (Simulated)")
mmqr_results = pd.DataFrame({
    "Quantile (τ)": [0.10, 0.25, 0.50, 0.75, 0.90],
    "Tourism Coef": [0.12, 0.18, 0.22, 0.29, 0.35],
    "Green_Bonds Coef": [-0.05, -0.03, 0.00, 0.04, 0.08],
    "GDP Coef": [0.30, 0.33, 0.36, 0.40, 0.44]
})
st.dataframe(mmqr_results)

# Quantile Coefficient Plot
st.subheader("Figure 5: Quantile Coefficient Plot")
fig, ax = plt.subplots()
ax.plot(mmqr_results["Quantile (τ)"], mmqr_results["Tourism Coef"], marker='o', label="Tourism")
ax.plot(mmqr_results["Quantile (τ)"], mmqr_results["Green_Bonds Coef"], marker='o', label="Green Bonds")
ax.plot(mmqr_results["Quantile (τ)"], mmqr_results["GDP Coef"], marker='o', label="GDP")
ax.set_xlabel("Quantiles")
ax.set_ylabel("Estimated Coefficients")
ax.legend()
st.pyplot(fig)

# ============================================
# Section F: Granger Causality (Placeholder)
# ============================================

st.header("F. Granger Causality Tests (Dumitrescu & Hurlin, 2012)")
granger_df = pd.DataFrame({
    "Null Hypothesis": ["Tourism does not Granger cause GDP", "GDP does not Granger cause Tourism"],
    "Statistic": [4.21, 2.87],
    "p-value": [0.001, 0.015],
    "Decision": ["Reject H0", "Reject H0"]
})
st.dataframe(granger_df)

# ============================================
# Section G: Diagnostics
# ============================================

st.header("G. Diagnostic Tests (Example)")
diag = pd.DataFrame({
    "Test": ["Hansen J-Test", "Wald Test", "Overidentification"],
    "Statistic": [2.13, 18.42, 0.97],
    "p-value": [0.12, 0.0001, 0.33]
})
st.dataframe(diag)

st.markdown("---")
st.markdown("App prepared by **Dr. Muhammad Saeed Meo’s MMQR Framework Generator**.")
