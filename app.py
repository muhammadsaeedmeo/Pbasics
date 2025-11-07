# ============================================
# Streamlit Panel Data Analysis App using MMQR
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    df = pd.read_csv("sample_data.csv")
    return df

data = load_data()

st.title("📊 Panel Data Analysis Dashboard (MMQR Framework)")
st.markdown("This demo app shows a complete structure for panel data econometric analysis using Method of Moments Quantile Regression (MMQR).")

st.header("A. Visual Data Exploration")
st.subheader("Figure 1: Average Trends of Key Variables")
avg_trends = data.groupby('Year')[['GDP', 'Tourism', 'Green_Bonds', 'CO2']].mean()
st.line_chart(avg_trends)

st.subheader("Figure 2: Cross-Sectional Distribution (Boxplot)")
fig, ax = plt.subplots()
sns.boxplot(x='Country', y='GDP', data=data, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Figure 3: Pairwise Scatter Plots")
sns.set(style="ticks")
fig = sns.pairplot(data[['GDP', 'Tourism', 'Green_Bonds', 'CO2']])
st.pyplot(fig)

st.subheader("Figure 4: Correlation Heatmap")
corr = data[['GDP', 'Tourism', 'Green_Bonds', 'CO2']].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
st.pyplot(fig)

st.header("B. Descriptive and Preliminary Tests")
st.subheader("Table 1: Descriptive Statistics")
st.dataframe(data[['GDP', 'Tourism', 'Green_Bonds', 'CO2']].describe().T)

st.subheader("Table 2: Correlation Matrix")
st.dataframe(corr)

st.header("C. Panel Unit Root Tests (Placeholder)")
unit_root_results = pd.DataFrame({
    "Variable": ["GDP", "Tourism", "Green_Bonds", "CO2"],
    "LLC p-value": [0.01, 0.02, 0.15, 0.05],
    "IPS p-value": [0.03, 0.04, 0.20, 0.07],
})
st.dataframe(unit_root_results)

st.header("D. Panel Cointegration Tests (Pedroni, Westerlund)")
cointegration_results = pd.DataFrame({
    "Test": ["Pedroni (2004)", "Westerlund (2007)"],
    "Statistic": [-3.42, -2.97],
    "p-value": [0.001, 0.004]
})
st.dataframe(cointegration_results)

st.header("E. Method of Moments Quantile Regression (MMQR) Results (Simulated)")
mmqr_results = pd.DataFrame({
    "Quantile (τ)": [0.10, 0.25, 0.50, 0.75, 0.90],
    "Tourism Coef": [0.12, 0.18, 0.22, 0.29, 0.35],
    "Green_Bonds Coef": [-0.05, -0.03, 0.00, 0.04, 0.08],
    "GDP Coef": [0.30, 0.33, 0.36, 0.40, 0.44]
})
st.dataframe(mmqr_results)

st.subheader("Figure 5: Quantile Coefficient Plot")
fig, ax = plt.subplots()
ax.plot(mmqr_results["Quantile (τ)"], mmqr_results["Tourism Coef"], marker='o', label="Tourism")
ax.plot(mmqr_results["Quantile (τ)"], mmqr_results["Green_Bonds Coef"], marker='o', label="Green Bonds")
ax.plot(mmqr_results["Quantile (τ)"], mmqr_results["GDP Coef"], marker='o', label="GDP")
ax.set_xlabel("Quantiles")
ax.set_ylabel("Estimated Coefficients")
ax.legend()
st.pyplot(fig)

st.header("F. Granger Causality Tests (Placeholder)")
granger_df = pd.DataFrame({
    "Null Hypothesis": ["Tourism does not Granger cause GDP", "GDP does not Granger cause Tourism"],
    "Statistic": [4.21, 2.87],
    "p-value": [0.001, 0.015],
    "Decision": ["Reject H0", "Reject H0"]
})
st.dataframe(granger_df)

st.header("G. Diagnostic Tests (Example)")
diag = pd.DataFrame({
    "Test": ["Hansen J-Test", "Wald Test", "Overidentification"],
    "Statistic": [2.13, 18.42, 0.97],
    "p-value": [0.12, 0.0001, 0.33]
})
st.dataframe(diag)

st.markdown("App prepared by **Dr. Muhammad Saeed Meo’s MMQR Framework Generator**.")
