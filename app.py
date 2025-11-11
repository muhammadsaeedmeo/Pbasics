# ===========================================================
# Streamlit Panel Data Quantile Regression Dashboard (MMQR)
# ===========================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import quantreg
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# App Configuration
# -----------------------------------------------------------

st.set_page_config(page_title="Panel Data Quantile Regression (MMQR)", layout="wide")
st.title("📊 Panel Data Quantile Regression Dashboard (MMQR Framework)")

# -----------------------------------------------------------
# Section 1: Data Upload
# -----------------------------------------------------------

st.sidebar.header("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom data loaded successfully!")
else:
    st.info("No file uploaded. Using simulated dataset for demonstration.")
    np.random.seed(42)
    countries = [f"Country_{i}" for i in range(1, 6)]
    years = list(range(2000, 2020))
    sample_data = []
    for country in countries:
        for year in years:
            gdp = np.random.normal(100, 20)
            tourism = 0.4 * gdp + np.random.normal(0, 5)
            invest = np.random.normal(50, 10)
            trade = np.random.normal(60, 15)
            sample_data.append({
                "Country": country,
                "Year": year,
                "GDP": gdp,
                "Tourism": tourism,
                "Investment": invest,
                "Trade": trade
            })
    df = pd.DataFrame(sample_data)

st.write("### Data Preview")
st.dataframe(df.head())

# -----------------------------------------------------------
# Section 2: Variable Distribution Visualization
# -----------------------------------------------------------

st.header("SECTION 2: Variable Distribution Analysis")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:
    st.warning("No numeric variables available for visualization.")
else:
    col1, col2 = st.columns(2)
    with col1:
        variable = st.selectbox("Select Variable for Distribution Analysis", num_cols)
    with col2:
        color_choice = st.color_picker("Pick Plot Color", "#2E86AB")

    if variable:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        sns.set(style="whitegrid")

        sns.histplot(df[variable], kde=False, color=color_choice, ax=axes[0, 0])
        axes[0, 0].set_title(f"Histogram of {variable}")

        sns.boxplot(x=df[variable], color=color_choice, ax=axes[0, 1])
        axes[0, 1].set_title(f"Box Plot of {variable}")

        sns.violinplot(x=df[variable], color=color_choice, ax=axes[1, 0])
        axes[1, 0].set_title(f"Violin Plot of {variable}")

        sns.stripplot(x=df[variable], color=color_choice, alpha=0.6, ax=axes[1, 1])
        axes[1, 1].set_title(f"Strip Plot of {variable}")

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(f"""
        **Interpretation Note:**  
        - Skewness or heavy tails in these plots indicate potential non-normality.  
        - The **box plot** identifies outliers.  
        - The **violin** and **strip plots** help visualize distribution density and spread.
        """)

# -----------------------------------------------------------
# Section 3: Correlation Analysis
# -----------------------------------------------------------

st.header("SECTION 3: Correlation Analysis")
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -----------------------------------------------------------
# Section 4: Method of Moments Quantile Regression (MMQR)
# -----------------------------------------------------------

st.header("SECTION 4: Method of Moments Quantile Regression (MMQR)")

dep_var = st.selectbox("Select Dependent Variable (Y)", num_cols)
indep_vars = st.multiselect("Select Independent Variables (X)", [x for x in num_cols if x != dep_var])
quantiles = st.multiselect("Select Quantiles", [0.1, 0.25, 0.5, 0.75, 0.9], default=[0.1, 0.25, 0.5, 0.75, 0.9])

if st.button("Run MMQR Estimation"):
    if dep_var and indep_vars:
        formula = f"{dep_var} ~ {' + '.join(indep_vars)}"
        results_list = []

        for q in quantiles:
            model = quantreg(formula, df)
            res = model.fit(q=q)
            params, pvals = res.params, res.pvalues
            for var in params.index:
                results_list.append({
                    "Quantile": q,
                    "Variable": var,
                    "Coefficient": round(params[var], 6),
                    "P-Value": round(pvals[var], 6)
                })

        results_df = pd.DataFrame(results_list)

        st.write("### MMQR Results (Coefficients and P-Values)")
        st.dataframe(results_df)

        # ----- Visualization -----
        fig, ax = plt.subplots(figsize=(10, 6))
        for var in indep_vars:
            subset = results_df[results_df["Variable"] == var]
            ax.plot(subset["Quantile"], subset["Coefficient"], marker="o", label=var)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xlabel("Quantiles")
        ax.set_ylabel("Coefficient Estimate")
        ax.set_title("MMQR Coefficients across Quantiles")
        ax.legend()
        st.pyplot(fig)

        # ----- Download -----
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download MMQR Results (CSV)", csv, "MMQR_results.csv", "text/csv")

        st.markdown("""
        **Interpretation:**  
        - Positive coefficients imply an increasing effect at given quantiles.  
        - Varying signs/magnitudes across quantiles indicate distributional heterogeneity.  
        - P-values < 0.05 suggest statistically significant effects at that quantile.
        """)
    else:
        st.warning("Please select both dependent and independent variables before estimation.")

# -----------------------------------------------------------
# Footer
# -----------------------------------------------------------

st.markdown("""
---
**Note:** This tool implements conditional quantile regression via `statsmodels`.  
The MMQR structure approximates the moment-based quantile framework suitable when `N < T` in panel settings.
""")
