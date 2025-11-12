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
from scipy.stats import shapiro
from scipy import stats
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

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
    st.info("No file uploaded. Using sample dataset.")
    np.random.seed(42)
    countries = [f"Country_{i}" for i in range(1, 11)]
    years = list(range(2000, 2020))
    sample = []
    for c in countries:
        for y in years:
            gdp = np.random.normal(100, 20)
            tourism = gdp * 0.3 + np.random.normal(0, 5)
            sample.append({
                "Country": c,
                "Year": y,
                "GDP": gdp,
                "Tourism": tourism,
                "Investment": np.random.normal(50, 10),
                "Trade": np.random.normal(60, 15)
            })
    df = pd.DataFrame(sample)

st.header("A. Data Overview")
st.dataframe(df.head())
st.write(f"Dataset shape: {df.shape}")

if "Country" not in df.columns or "Year" not in df.columns:
    st.error("❌ Required columns 'Country' and/or 'Year' missing.")
    st.stop()

# ======================================================================
# 📊 SECTION: DESCRIPTIVE STATISTICS AND DISTRIBUTION ANALYSIS (Enhanced)
# ======================================================================

st.subheader("Descriptive Statistics and Distribution Analysis")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected_col = st.selectbox(
    "Select a variable (or choose 'All Variables - Combined Summary Plot')",
    options=["All Variables - Combined Summary Plot"] + numeric_cols
)

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(col):
    data = df[col].dropna()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    
    sns.histplot(data, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Histogram + KDE")
    
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title("QQ Plot")
    
    sns.boxplot(y=data, ax=axes[2], color="mediumseagreen")
    axes[2].set_title("Box Plot")
    
    sns.violinplot(y=data, ax=axes[3], color="salmon")
    axes[3].set_title("Violin Plot")
    
    plt.tight_layout()
    st.pyplot(fig)

    if len(data) > 3:
        stat, p = stats.shapiro(data)
        if p > 0.05:
            st.info(f"**{col}** appears normally distributed (p = {p:.3f}).")
        else:
            st.warning(f"**{col}** deviates from normality (p = {p:.3f}).")
    else:
        st.write("Sample too small for normality test.")
    st.markdown("---")


# ---- Combined Plot for All Variables ----
def combined_distribution_plot(df, numeric_cols):
    n = len(numeric_cols)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        data = df[col].dropna()
        sns.kdeplot(data, fill=True, ax=axes[i], color=sns.color_palette("husl", n)[i])
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Density")
    
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle("Combined Distribution of All Variables", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    st.pyplot(fig)
    st.markdown("**Note:** The density plots show each variable’s overall distribution pattern for quick comparison.")


# ---- Logic ----
if selected_col == "All Variables - Combined Summary Plot":
    combined_distribution_plot(df, numeric_cols)
else:
    st.subheader(f"Descriptive Analysis for {selected_col}")
    plot_distribution(selected_col)

# ============================================
# Section C: Correlation Analysis
# ============================================

st.header("C. Correlation Analysis")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    col1, col2 = st.columns(2)
    with col1:
        dep_var = st.selectbox("Select Dependent Variable", options=numeric_cols)
    with col2:
        indep_vars = st.multiselect(
            "Select Independent Variable(s)",
            options=[c for c in numeric_cols if c != dep_var],
            default=[c for c in numeric_cols if c != dep_var][:3]
        )

    color_option = st.selectbox(
        "Heatmap Color Palette",
        options=["coolwarm","viridis","plasma","magma","cividis","Blues","Greens","Reds"],
        index=0
    )

    if indep_vars:
        selected_vars = [dep_var] + indep_vars
        corr = df[selected_vars].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap=color_option, center=0, linewidths=0.5, fmt=".2f", ax=ax)
        plt.title("Correlation Heatmap")
        st.pyplot(fig)

        def interpret_corr(v):
            v = abs(v)
            if v < 0.20: return "very weak"
            if v < 0.40: return "weak"
            if v < 0.60: return "moderate"
            if v < 0.80: return "strong"
            return "very strong"

        st.subheader("Correlation Interpretation")
        for var in indep_vars:
            val = corr.loc[dep_var, var]
            st.write(f"- {dep_var} and {var}: {val:.2f} ({interpret_corr(val)} {'positive' if val>0 else 'negative'})")
else:
    st.warning("No numeric variables for correlation.")

# ============================================
# Section D: MMQR Implementation with Location, Scale, and SEs
# ============================================

st.markdown("---")
st.header("D. Method of Moments Quantile Regression (MMQR) — Machado & Santos Silva (2019)")

if 'dep_var' not in locals() or 'indep_vars' not in locals() or not indep_vars:
    st.warning("Please complete the correlation analysis first to select dependent and independent variables.")
else:
    st.subheader("MMQR Configuration")

    col1, col2 = st.columns(2)
    with col1:
        quantiles = st.text_input("Quantiles (comma-separated)", "0.05,0.25,0.50,0.75,0.95")
        quantiles = [float(q.strip()) for q in quantiles.split(",")]
    with col2:
        reference_quantile = st.selectbox("Reference Quantile for Location", [0.25, 0.50, 0.75], index=1)

    # MMQR Function
    def run_mmqr(data, dep_var, indep_vars, quantiles, reference_quantile=0.5):
        results = {}
        X = data[indep_vars]
        y = data[dep_var]
        formula = f"{dep_var} ~ {' + '.join(indep_vars)}"

        # Step 1: Location (reference quantile)
        loc_model = quantreg(formula, data).fit(q=reference_quantile, vcov='robust')
        location_params = loc_model.params
        location_se = loc_model.bse
        location_p = loc_model.pvalues

        # Step 2: Scale parameters (difference between τ=0.75 and τ=0.25)
        q_high, q_low = 0.75, 0.25
        model_high = quantreg(formula, data).fit(q=q_high, vcov='robust')
        model_low = quantreg(formula, data).fit(q=q_low, vcov='robust')

        scale_params = (model_high.params - model_low.params) / (q_high - q_low)
        scale_se = np.sqrt((model_high.bse**2 + model_low.bse**2) / ((q_high - q_low)**2))
        t_stats = scale_params / scale_se
        dfree = len(data) - len(indep_vars) - 1
        scale_p = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=dfree))

        # Step 3: MMQR across quantiles
        for tau in quantiles:
            model = quantreg(formula, data).fit(q=tau, vcov='robust')
            results[tau] = {
                "coefficients": model.params,
                "stderr": model.bse,
                "pvalues": model.pvalues
            }

        return location_params, location_se, location_p, scale_params, scale_se, scale_p, results

    try:
        loc_b, loc_se, loc_p, sc_b, sc_se, sc_p, mmqr_results = run_mmqr(df, dep_var, indep_vars, quantiles, reference_quantile)

        # Table 1: Location Parameters
        st.subheader(f"Table 1. Location Parameters (τ = {reference_quantile})")
        loc_table = pd.DataFrame({
            "Variable": loc_b.index,
            "Coefficient": loc_b.values,
            "Std. Error": loc_se.values,
            "P-Value": loc_p.values,
            "Significance": ["***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "" for p in loc_p.values]
        })
        st.dataframe(loc_table, use_container_width=True)

        # Table 2: Scale Parameters
        st.subheader("Table 2. Scale Parameters (τ = 0.75 − τ = 0.25)")
        scale_table = pd.DataFrame({
            "Variable": sc_b.index,
            "Scale Coefficient": sc_b.values,
            "Std. Error": sc_se.values,
            "P-Value": sc_p,
            "Significance": ["***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "" for p in sc_p]
        })
        st.dataframe(scale_table, use_container_width=True)

        # Table 3: MMQR Results for Each Quantile
        st.subheader("Table 3. MMQR Estimation Results")
        combined_rows = []
        for var in loc_b.index:
            row = {"Variable": var}
            for tau in quantiles:
                coef = mmqr_results[tau]["coefficients"][var]
                pval = mmqr_results[tau]["pvalues"][var]
                se = mmqr_results[tau]["stderr"][var]
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                row[f"τ={tau}"] = f"{coef:.3f}{stars}\n({se:.3f})"
            combined_rows.append(row)
        mmqr_table = pd.DataFrame(combined_rows)
        st.dataframe(mmqr_table, use_container_width=True)

        # Plot Coefficient Dynamics
        st.subheader("Figure 1. MMQR Coefficient Dynamics")
        for var in indep_vars:
            fig, ax = plt.subplots(figsize=(8, 5))
            coefs = [mmqr_results[t]["coefficients"][var] for t in quantiles]
            ax.plot(quantiles, coefs, 'o-', color='blue', label=var)
            ax.axhline(y=loc_b[var], color='red', linestyle='--', label='Location (ref quantile)')
            ax.set_xlabel("Quantile (τ)")
            ax.set_ylabel("Coefficient")
            ax.set_title(f"MMQR Coefficient Dynamics: {var}")
            ax.legend()
            st.pyplot(fig)

        # Download results
        download_df = pd.concat([
            loc_table.assign(Type="Location"),
            scale_table.assign(Type="Scale"),
            mmqr_table.melt(id_vars=["Variable"], var_name="Quantile", value_name="Coef(SE)").assign(Type="MMQR")
        ])
        csv_data = download_df.to_csv(index=False)
        st.download_button("📥 Download MMQR Results (CSV)", data=csv_data, file_name="MMQR_Results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"MMQR estimation failed: {str(e)}")


# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("**Panel Data Analysis Dashboard** | Built with Streamlit")
