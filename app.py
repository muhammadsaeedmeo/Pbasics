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

* ==============================================================
* 🟩 MMQR (Machado & Santos Silva, 2019) – FULL IMPLEMENTATION
* ==============================================================

clear all
set more off

*--- Load data
use "your_data.dta", clear

*--- Install dependencies once
ssc install moremata, replace

*--- Define variables
local depvar tourism
local indepvars reer gdp inflation exchange

*--- Create results file
capture postclose mmqr_results
postfile mmqr_results str15 component str20 varname quantile coef se pval using mmqr_results.dta, replace

*--- Loop through quantiles
foreach q of numlist 0.05(0.05)0.95 {

    * Run MMQR with both location and scale components
    quietly mmqreg `depvar' `indepvars', quantile(`q') location scale

    * Extract location parameters
    matrix bl = e(b_location)
    matrix Vl = e(V_location)
    local nloc = colsof(bl)

    forvalues i = 1/`nloc' {
        local vname : colname bl[`i']
        local coef = bl[1,`i']
        local se = sqrt(Vl[`i',`i'])
        local pval = 2*ttail(e(df_r), abs(`coef'/`se'))
        post mmqr_results ("location") ("`vname'") ("`q'") (`coef') (`se') (`pval')
    }

    * Extract scale parameters
    matrix bs = e(b_scale)
    matrix Vs = e(V_scale)
    local nsc = colsof(bs)

    forvalues i = 1/`nsc' {
        local vname : colname bs[`i']
        local coef = bs[1,`i']
        local se = sqrt(Vs[`i',`i'])
        local pval = 2*ttail(e(df_r), abs(`coef'/`se'))
        post mmqr_results ("scale") ("`vname'") ("`q'") (`coef') (`se') (`pval')
    }
}
postclose mmqr_results

*--- Use and organize results
use mmqr_results.dta, clear
destring quantile, replace force

*--- Plot example: REER location coefficient profile
preserve
keep if varname == "reer" & component == "location"
gen ub = coef + 1.96*se
gen lb = coef - 1.96*se

twoway (rarea ub lb quantile, color(gs14)) ///
       (line coef quantile, lcolor(blue) lwidth(medthick)) ///
       , title("MMQR Location Coefficient Profile") ///
         subtitle("Variable: REER") ///
         ytitle("Coefficient Estimate") ///
         xtitle("Quantile (τ)") ///
         legend(off)
restore

*--- Export full results
export excel using "MMQR_full_results.xlsx", firstrow(variables) replace

*--- Display in Stata
list component varname quantile coef se pval, clean


# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("**Panel Data Analysis Dashboard** | Built with Streamlit")
