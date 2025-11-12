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
# Section D: MMQR Implementation (unchanged)
# ============================================

st.markdown("---")
st.markdown("**Panel Data Analysis Dashboard** | Built with Streamlit")

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.formula.api import quantreg # Assuming this is available
from scipy import stats
import matplotlib.pyplot as plt

# ASSUMPTION: df, dep_var, and indep_vars are defined in the Streamlit scope.
# For demonstration purposes, we'll assume a dummy setup for the function to run:
# def correct_mmqr_estimation(data, y_var, x_vars, quantiles, reference_quantile=0.5, bootstrap=True, n_boot=200):
#   ...

# ============================================
# Section D: Updated MMQR Implementation
# ============================================

st.header("D. Method of Moments Quantile Regression (MMQR) - Updated")

if 'dep_var' not in locals() or 'indep_vars' not in locals() or not indep_vars:
    st.warning("Please complete the correlation analysis first to select variables.")
else:
    st.subheader("MMQR Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        quantiles = st.text_input("Quantiles (comma-separated)", "0.05,0.25,0.50,0.75,0.95")
        quantiles = [float(q.strip()) for q in quantiles.split(",")]
    
    with col2:
        bootstrap_ci = st.checkbox("Bootstrap Scale Inference", True)
        n_bootstrap = st.slider("Bootstrap Samples for Scale", 200, 2000, 500) if bootstrap_ci else 200
        reference_quantile = st.selectbox("Reference Quantile for Location (α)", [0.25, 0.50, 0.75], index=1)

    def correct_mmqr_estimation(data, y_var, x_vars, quantiles, reference_quantile=0.5, bootstrap=True, n_boot=500):
        """
        MMQR implementation following Machado & Santos Silva (2019) structure,
        using coefficient combination and robust bootstrap inference for scale (δ).
        """
        results = {}
        
        # Prepare data
        # Note: Add 'Intercept' if not automatically included by quantreg/formula
        # Assuming statsmodels.formula.api.quantreg handles the Intercept automatically
        
        # Step 1: Estimate location parameters (α) using reference quantile
        formula_ref = f"{y_var} ~ {' + '.join(x_vars)}"
        
        try:
            location_model = quantreg(formula_ref, data).fit(q=reference_quantile, vcov='robust')
            location_params = location_model.params
            location_pvalues = location_model.pvalues
        except Exception as e:
            raise ValueError(f"Location estimation failed at τ={reference_quantile}: {str(e)}")
        
        # Step 2: Estimate scale parameters (δ) using symmetric quantiles
        # M&SS suggest a value related to the inter-quartile range for robust delta estimation.
        tau_high = 0.75
        tau_low = 0.25
        
        try:
            model_high = quantreg(formula_ref, data).fit(q=tau_high, vcov='robust')
            model_low = quantreg(formula_ref, data).fit(q=tau_low, vcov='robust')
        except Exception as e:
            raise ValueError(f"Scale quantile estimation failed: {str(e)}")

        # Scale parameters are proportional to the difference between high and low quantiles
        scale_params = (model_high.params - model_low.params) / (tau_high - tau_low)
        scale_pvalues = {var: 1.0 for var in scale_params.index} # Initialize p-values

        # Alternative method: Bootstrap for scale parameters (Robust Inference)
        if bootstrap and n_boot > 1:
            st.info(f"Bootstrapping scale parameters with N={n_boot} samples...")
            bootstrap_scale = {var: [] for var in scale_params.index}
            progress_bar = st.progress(0)
            
            for i in range(n_boot):
                boot_sample = data.sample(n=len(data), replace=True, random_state=i) # Using i as seed for reproducibility
                try:
                    boot_high = quantreg(formula_ref, boot_sample).fit(q=tau_high)
                    boot_low = quantreg(formula_ref, boot_sample).fit(q=tau_low)
                    boot_scale = (boot_high.params - boot_low.params) / (tau_high - tau_low)
                    
                    for var in scale_params.index:
                        # Only append if the boot_scale estimate exists (i.e., convergence)
                        if var in boot_scale:
                            bootstrap_scale[var].append(boot_scale[var])
                except:
                    # Ignore non-convergent bootstrap samples
                    continue
                
                progress_bar.progress((i + 1) / n_boot)
            
            progress_bar.empty() # Clear the progress bar after completion

            # Calculate bootstrap p-values for scale parameters
            for var in scale_params.index:
                boot_vals = np.array(bootstrap_scale[var])
                if len(boot_vals) > 0 and len(boot_vals) >= (n_boot * 0.9): # Check for sufficient convergence
                    # Centered at zero, two-sided p-value: proportion of bootstrap estimates with opposite sign
                    # P-value = 2 * min(P(b* > 0), P(b* <= 0))
                    
                    # Number of successful bootstrap runs
                    N_success = len(boot_vals) 
                    
                    # Count how many bootstrap estimates are on the opposite side of the initial estimate
                    if scale_params[var] >= 0:
                        opposite_count = np.sum(boot_vals <= 0)
                    else: # scale_params[var] < 0
                        opposite_count = np.sum(boot_vals > 0)
                        
                    p_value_boot = 2 * (opposite_count / N_success)
                    
                    # Ensure p-value is between 0 and 1
                    scale_pvalues[var] = min(p_value_boot, 1.0)
                else:
                    st.warning(f"Bootstrap failed to converge enough times for variable '{var}'. Using default p=1.0.")
                    scale_pvalues[var] = 1.0

        # Step 3: MMQR coefficient combination: β(τ) = α + δ·h(τ)
        # Use h(τ) = τ, the simplest form of the quantile index function for M&SS
        for q in quantiles:
            # h(τ) is the quantile index function, often chosen as τ or a function of τ.
            # For simplicity and direct interpretation as the coefficient trend, we use:
            h_tau = q
            
            # MMQR coefficients: β(τ) = α + δ·h(τ)
            mmqr_coefficients = location_params + scale_params * h_tau
            
            # Store results, including the standard QR for comparison/conf_int access
            results[q] = {
                'coefficients': mmqr_coefficients, # This is the MMQR result
                'pvalues': location_pvalues.apply(lambda x: 1.0), # MMQR coefficients don't have standard p-values from this method, use location/scale for inference.
                'conf_int': None, # Cannot easily compute CI with this method without boostrap
                'location_params': location_params,
                'scale_params': scale_params,
                'scale_pvalues': scale_pvalues,
                'mmqr_coefficients': mmqr_coefficients,
                'quantile': q,
                'model': None # No final QR model in this approach
            }
            # Note: The 'pvalues' and 'conf_int' fields for the final MMQR results are set to placeholder/None
            # because the coefficient combination method does not directly provide them.
            # Inference is primarily done on the location (α) and scale (δ) parameters.

        return results, location_params, scale_params, scale_pvalues

    # Run corrected MMQR with scale p-values
    try:
        mmqr_results, location_params, scale_params, scale_pvalues = correct_mmqr_estimation(
            df, dep_var, indep_vars, quantiles, reference_quantile, bootstrap_ci, n_bootstrap
        )
        
        # ---
        # NOTE: The rest of your display code (Table 2, Table 3, Table 4, and the Plotting section) 
        # is EXCELLENT for visualizing the MMQR results and does not need major revision 
        # now that the underlying function provides the robust bootstrap p-values.
        # ---
        
        # ========================
        # Table 1: Location Parameters (Reference Quantile)
        # ========================
        st.subheader(f"Table 2: Location Parameters, $\\hat{{\\alpha}}$ (Ref. $\\tau$ = {reference_quantile})")
        # Since the location estimate comes from a direct QR, we'll try to get its actual p-values
        
        # Re-run location model to get p-values reliably (if needed, otherwise rely on the one inside the function)
        formula_ref = f"{dep_var} ~ {' + '.join(indep_vars)}"
        location_model = quantreg(formula_ref, df).fit(q=reference_quantile, vcov='robust')
        location_pvalues = location_model.pvalues
        
        location_data = []
        for var in location_params.index:
            var_name = 'Intercept' if var == 'Intercept' else var
            coef = location_params[var]
            pval = location_pvalues[var]
            
            location_data.append({
                'Variable': var_name,
                'Coefficient': f"{coef:.3f}",
                'P-Value': f"{pval:.3f}",
                'Significance': '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            })
        
        location_df = pd.DataFrame(location_data)
        st.dataframe(location_df, use_container_width=True)
        
        # ========================
        # Table 2: Scale Parameters (δ) with P-Values
        # ========================
        st.subheader("Table 3: Scale Parameters, $\\hat{{\\delta}}$ (Heterogeneity Effect)")
        
        scale_data = []
        for var in scale_params.index:
            var_name = 'Intercept' if var == 'Intercept' else var
            scale_val = scale_params[var]
            pval = scale_pvalues[var]
            
            if var in location_params:
                relative_effect = abs(scale_val) / abs(location_params[var]) if location_params[var] != 0 else float('inf')
            else:
                relative_effect = 0
            
            scale_data.append({
                'Variable': var_name,
                'Scale Coeff. ($\hat{\delta}$ )': f"{scale_val:.3f}",
                'P-Value (Bootstrap)': f"{pval:.3f}",
                'Significance': '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else '',
                'Relative to Location': f"{relative_effect:.3f}" if relative_effect != float('inf') else "N/A",
                'Interpretation': 'Strong heterogeneity' if pval < 0.05 and relative_effect > 0.5 else 
                                'Moderate heterogeneity' if pval < 0.05 else 
                                'No significant heterogeneity' if pval >= 0.05 else 'Inconclusive'
            })
        
        scale_df = pd.DataFrame(scale_data)
        st.dataframe(scale_df, use_container_width=True)
        
        # ========================
        # Table 3: MMQR Results: β(τ) = α + δ·τ
        # ========================
        st.subheader("Table 4: MMQR Coefficient Estimates, $\\hat{{\\beta}}(\\tau)$")
        
        mmqr_data = []
        coef_names = scale_params.index.tolist()
        
        for var in coef_names:
            var_name = 'Intercept' if var == 'Intercept' else var
            row = {'Variable': var_name}
            
            for q in quantiles:
                if q in mmqr_results:
                    # Use MMQR coefficients
                    coef = mmqr_results[q]['mmqr_coefficients'][var]
                    # Use scale p-value for dynamic coefficient significance interpretation
                    scale_pval = scale_pvalues[var] 
                    
                    # The significance for the final MMQR coefficient is based on
                    # the significance of the scale parameter (for heterogeneity)
                    # AND the location parameter (for level). This is a simplification.
                    # We'll stick to displaying the coefficient only for clarity here.
                    
                    # For a simple table, just show the coefficient value
                    row[f'τ = {q}'] = f"{coef:.3f}" 
            
            mmqr_data.append(row)
        
        mmqr_df = pd.DataFrame(mmqr_data)
        st.dataframe(mmqr_df, use_container_width=True)
        
        # ========================
        # Scale Parameters Interpretation
        # ========================
        st.subheader("Scale Parameters Interpretation")
        
        st.markdown(f"The coefficients $\\hat{{\\beta}}(\\tau)$ are computed as $\\hat{{\\alpha}} + \\hat{{\\delta}} \\cdot \\tau$, where $\\hat{{\\alpha}}$ is from $\\tau={reference_quantile}$ and $\\hat{{\\delta}}$ is estimated from $\\tau=0.25$ and $\\tau=0.75$. The significance of the **scale effect** is determined by the **Bootstrap P-Value** of $\\hat{{\\delta}}$ in Table 3.")
        
        significant_scale_vars = [var for var in scale_params.index 
                                 if var != 'Intercept' and scale_pvalues[var] < 0.1]
        insignificant_scale_vars = [var for var in scale_params.index 
                                    if var != 'Intercept' and scale_pvalues[var] >= 0.1]
        
        if significant_scale_vars:
            st.success(f"**Significant heterogeneity detected in:** {', '.join(significant_scale_vars)}")
            st.write("These variables show a statistically significant variation in their effects across different quantiles (i.e., their impact is not constant).")
        
        if insignificant_scale_vars:
            st.info(f"**No significant heterogeneity in:** {', '.join(insignificant_scale_vars)}")
            st.write("These variables show relatively stable effects across quantiles, suggesting their impact is mainly on the location (mean/median) of the dependent variable.")
        
        # Detailed interpretation for each variable
        st.write("**Variable-specific Scale Effects:**")
        for var in scale_params.index:
            if var != 'Intercept':
                scale_val = scale_params[var]
                pval = scale_pvalues[var]
                loc_val = location_params[var] if var in location_params else 0
                
                if pval < 0.05:
                    direction = "increasing" if scale_val > 0 else "decreasing"
                    st.write(f"- **{var}**: Significant heterogeneity (p={pval:.3f})")
                    st.write(f"  - The marginal effect of this variable is **{direction}** across the conditional distribution quantiles.")
                    if loc_val != 0:
                        relative_mag = abs(scale_val / loc_val)
                        st.write(f"  - The magnitude of the scale effect is approximately {relative_mag:.1%} of the location effect, indicating a strong spread.")
                else:
                    st.write(f"- **{var}**: No significant heterogeneity (p={pval:.3f})")
                    st.write(f"  - The effect remains relatively constant across quantiles; the model is primarily a **Location-Shift** model for this variable.")

        # ========================
        # MMQR Coefficient Dynamics
        # ========================
        st.subheader("Figure 1: MMQR Coefficient Dynamics ($\\hat{{\\beta}}(\\tau)$ vs. $\\tau$)")
        

        plot_vars = [var for var in coef_names if var != 'Intercept']
        
        for var in plot_vars:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot MMQR coefficients
            mmqr_coefs = [mmqr_results[q]['mmqr_coefficients'][var] for q in quantiles if q in mmqr_results]
            quantiles_avail = [q for q in quantiles if q in mmqr_results]
            
            if len(mmqr_coefs) > 0:
                # Plot the MMQR trajectory
                ax.plot(quantiles_avail, mmqr_coefs, 'o-', linewidth=2.5, 
                        label=f'MMQR Coefficients $\\hat{{\\alpha}} + \\hat{{\\delta}} \\cdot \\tau$', color='#2E86AB', markersize=8)
                
                # Add location parameter (horizontal line)
                loc_coef = location_params[var]
                ax.axhline(y=loc_coef, color='red', linestyle='--', alpha=0.7, 
                            label=f'Location $\\hat{{\\alpha}}$ (Ref. $\\tau={reference_quantile}$)')
                
                # Add scale effect indication
                scale_pval = scale_pvalues[var]
                
                # Since the MMQR coefficient is explicitly linear in tau, we can draw the line
                # The line *is* the trajectory, but we can highlight it.
                if scale_pval < 0.1:
                    ax.plot(quantiles_avail, mmqr_coefs, '-', color='#2E86AB', linewidth=1, alpha=0.7)
                
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_xlabel("Quantiles ($\\tau$)", fontsize=12)
                ax.set_ylabel("Coefficient Estimate", fontsize=12)
                
                # Add scale significance to title
                scale_sig = '***' if scale_pval < 0.01 else '**' if scale_pval < 0.05 else '*' if scale_pval < 0.1 else 'ns'
                ax.set_title(f"MMQR Coefficient Dynamics: {var} (Scale $\\hat{{\\delta}}$: {scale_sig})", 
                            fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)

        # ========================
        # Download Section with Scale P-values
        # ========================
        st.subheader("Download MMQR Results")
        
        # Prepare comprehensive download data
        download_data = []
        
        # Location parameters
        for var in location_params.index:
            var_name = 'Intercept' if var == 'Intercept' else var
            pval = location_pvalues[var]
            download_data.append({
                'Variable': var_name,
                'Type': f'Location (α) at τ={reference_quantile}',
                'Coefficient': round(location_params[var], 3),
                'P-Value': round(pval, 3),
                'Quantile': reference_quantile,
                'Method': 'MMQR'
            })
        
        # Scale parameters with p-values
        for var in scale_params.index:
            var_name = 'Intercept' if var == 'Intercept' else var
            download_data.append({
                'Variable': var_name,
                'Type': 'Scale (δ)', 
                'Coefficient': round(scale_params[var], 3),
                'P-Value (Bootstrap)': round(scale_pvalues[var], 3),
                'Quantile': 'N/A',
                'Method': 'MMQR'
            })
        
        # MMQR coefficients
        for var in coef_names:
            var_name = 'Intercept' if var == 'Intercept' else var
            for q in quantiles:
                if q in mmqr_results:
                    download_data.append({
                        'Variable': var_name,
                        'Type': f'MMQR_Coefficient (β) at τ={q}',
                        'Coefficient': round(mmqr_results[q]['mmqr_coefficients'][var], 3),
                        'P-Value (Inferred)': round(scale_pvalues[var], 3), # Use scale p-value as primary inference
                        'Quantile': q,
                        'Method': 'MMQR'
                    })
        
        download_df = pd.DataFrame(download_data)
        csv_data = download_df.to_csv(index=False)
        
        st.download_button(
            "📥 Download Complete MMQR Results (CSV)",
            data=csv_data,
            file_name="MMQR_Complete_Results_Updated.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"MMQR estimation failed: {str(e)}")

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("**Panel Data Analysis Dashboard** | Built with Streamlit")
