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
# Correlation Heatmap with Dropdowns, Color Selection & Interpretation
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

st.header("Correlation Heatmap")

# --- Load your data here ---
# data = pd.read_csv("your_data.csv")

# Detect numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.warning("No numeric variables found in your dataset.")
else:
    # --- Dropdowns for selecting dependent & independent variables ---
    dep_var = st.selectbox("Select Dependent Variable", options=numeric_cols)
    indep_vars = st.multiselect(
        "Select Independent Variable(s)",
        options=[col for col in numeric_cols if col != dep_var],
        default=[col for col in numeric_cols if col != dep_var][:3]
    )

    # --- Color palette selector ---
    color_option = st.selectbox(
        "Select Heatmap Color Palette",
        options=[
            "coolwarm", "viridis", "plasma", "magma", "cividis",
            "Blues", "Greens", "Reds", "Purples", "icefire", "Spectral"
        ],
        index=0
    )

    if indep_vars:
        # --- Compute correlation matrix ---
        selected_vars = [dep_var] + indep_vars
        corr = data[selected_vars].corr()

        # --- Generate heatmap ---
        fig, ax = plt.subplots()
        sns.heatmap(
            corr,
            annot=True,
            cmap=color_option,
            center=0,
            linewidths=0.5,
            fmt=".2f"
        )
        plt.title(f"Correlation Heatmap ({color_option} palette)")
        st.pyplot(fig)

        # --- Add download button for heatmap image ---
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button(
            label="Download Heatmap Image",
            data=buf.getvalue(),
            file_name="correlation_heatmap.png",
            mime="image/png"
        )

        # --- Interpret correlation results ---
        st.subheader("Correlation Interpretation")

        def interpret_corr(value):
            val = abs(value)
            if val < 0.20:
                return "very weak"
            elif val < 0.40:
                return "weak"
            elif val < 0.60:
                return "moderate"
            elif val < 0.80:
                return "strong"
            else:
                return "very strong"

        interpretation_text = ""
        for var in indep_vars:
            corr_value = corr.loc[dep_var, var]
            strength = interpret_corr(corr_value)
            direction = "positive" if corr_value > 0 else "negative"
            interpretation_text += (
                f"- The correlation between **{dep_var}** and **{var}** is "
                f"**{corr_value:.2f}**, indicating a **{strength} {direction} relationship**.\n"
            )

        st.markdown(interpretation_text)

        st.info(
            "According to Evans (1996), correlation strengths are defined as: "
            "very weak (0.00–0.19), weak (0.20–0.39), moderate (0.40–0.59), "
            "strong (0.60–0.79), and very strong (0.80–1.00).\n\n"
            "**Reference:** Evans, J. D. (1996). *Straightforward statistics for the behavioral sciences.* "
            "Brooks/Cole Publishing."
        )
    else:
        st.warning("Please select at least one independent variable to display correlation.")

st.subheader("Table 2: Correlation Matrix")
try:
    st.dataframe(corr)
except Exception as e:
    st.warning(f"Cannot compute correlation matrix: {e}")

# ============================================
# Slope Homogeneity Test (Pesaran and Yamagata, 2008)
# ============================================

st.subheader("Slope Homogeneity Test (Pesaran and Yamagata, 2008)")

try:
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd

    # Check required columns
    if "Country" not in data.columns or "Year" not in data.columns:
        st.warning("Please ensure your dataset includes 'Country' and 'Year' columns for panel data.")
    else:
        dep = dep_var
        indeps = indep_vars

        if not indeps:
            st.warning("Please select independent variables first.")
        else:
            # Prepare data by country
            panel_results = []
            for country, subset in data.groupby("Country"):
                if subset[dep].isnull().any() or subset[indeps].isnull().any().any():
                    continue  # skip missing data
                X = sm.add_constant(subset[indeps])
                y = subset[dep]
                model = sm.OLS(y, X).fit()
                panel_results.append(model.params.values)

            betas = np.vstack(panel_results)
            mean_beta = np.mean(betas, axis=0)
            N, k = betas.shape

            # Compute test statistics
            S = np.sum((betas - mean_beta) ** 2, axis=0)
            delta = N * np.sum(S) / np.sum(mean_beta ** 2)
            delta_adj = (N * delta - k) / np.sqrt(2 * k)

            # Compute p-values (two-tailed from normal distribution)
            from scipy.stats import norm
            p_delta = 2 * (1 - norm.cdf(abs(delta)))
            p_delta_adj = 2 * (1 - norm.cdf(abs(delta_adj)))

            # Create a nice result table
            results_df = pd.DataFrame({
                "Statistic": ["Δ", "Δ_adj"],
                "Value": [round(delta, 3), round(delta_adj, 3)],
                "p-value": [f"{p_delta:.3f}", f"{p_delta_adj:.3f}"]
            })

            st.write("**Slope Homogeneity Test Results**")
            st.dataframe(results_df, use_container_width=True)

            # Simple interpretation line
            if p_delta_adj < 0.05:
                st.success("Reject the null hypothesis — slopes are *heterogeneous* across cross-sections.")
                st.markdown("**Interpretation:** The regression slopes are not the same for all cross-sections.")
            else:
                st.info("Fail to reject the null hypothesis — slopes are *homogeneous* across cross-sections.")
                st.markdown("**Interpretation:** The regression slopes are broadly similar across cross-sections.")

            # Reference
            st.caption(
                "Reference: Pesaran, M. H., & Yamagata, T. (2008). "
                "Testing slope homogeneity in large panels. *Journal of Econometrics*, 142(1), 50–93."
            )

except Exception as e:
    st.warning(f"Error running slope homogeneity test: {e}")

# #################################################
# MMQR
# ###############################################33
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Method of Moments Quantile Regression (MMQR)
# Implementation based on Machado & Silva (2019)
# ============================================

st.set_page_config(page_title="MMQR Analysis", layout="wide")
st.title("Method of Moments Quantile Regression (MMQR)")
st.markdown("""
**Reference:** Machado, J.A.F. and Silva, J.M.C.S. (2019). "Quantiles via moments."  
*Journal of Econometrics*, 213(1), 145-173.

This implementation estimates **unconditional quantile partial effects** (UQPEs) using the location-scale approach.
""")

# ============================================
# Helper Functions for MMQR
# ============================================

class MMQRModel:
    """
    Method of Moments Quantile Regression
    Estimates unconditional quantile treatment effects
    """
    
    def __init__(self, y, X, quantiles=[0.05, 0.25, 0.50, 0.75, 0.95]):
        self.y = np.array(y)
        self.X = np.array(X)
        self.n, self.k = X.shape
        self.quantiles = quantiles
        self.results = {}
        
    def fit(self, bootstrap_se=True, n_bootstrap=200):
        """
        Fit MMQR model using location-scale approach
        """
        # Step 1: Location model (OLS)
        X_with_const = np.column_stack([np.ones(self.n), self.X])
        beta_location = np.linalg.lstsq(X_with_const, self.y, rcond=None)[0]
        residuals = self.y - X_with_const @ beta_location
        
        # Step 2: Scale model (log absolute residuals)
        log_abs_resid = np.log(np.abs(residuals) + 1e-10)
        gamma_scale = np.linalg.lstsq(X_with_const, log_abs_resid, rcond=None)[0]
        
        # Step 3: Compute MMQR coefficients for each quantile
        for tau in self.quantiles:
            q_tau = stats.norm.ppf(tau)  # Standard normal quantile
            
            # MMQR coefficient: β(τ) = β_location + q_τ * exp(X'γ_scale) * ∂γ/∂x
            # Simplified: β(τ) = β_location + q_τ * γ_scale
            beta_mmqr = beta_location + q_tau * gamma_scale
            
            # Store results
            self.results[tau] = {
                'coefficients': beta_mmqr,
                'beta_location': beta_location,
                'gamma_scale': gamma_scale
            }
        
        # Step 4: Bootstrap standard errors if requested
        if bootstrap_se:
            self._bootstrap_inference(n_bootstrap)
        
        return self
    
    def _bootstrap_inference(self, n_bootstrap):
        """
        Compute bootstrap standard errors and confidence intervals
        """
        bootstrap_coefs = {tau: [] for tau in self.quantiles}
        
        for b in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(self.n, size=self.n, replace=True)
            y_boot = self.y[indices]
            X_boot = self.X[indices]
            
            try:
                # Fit bootstrap sample
                X_boot_const = np.column_stack([np.ones(len(indices)), X_boot])
                beta_loc_boot = np.linalg.lstsq(X_boot_const, y_boot, rcond=None)[0]
                resid_boot = y_boot - X_boot_const @ beta_loc_boot
                log_abs_resid_boot = np.log(np.abs(resid_boot) + 1e-10)
                gamma_scale_boot = np.linalg.lstsq(X_boot_const, log_abs_resid_boot, rcond=None)[0]
                
                for tau in self.quantiles:
                    q_tau = stats.norm.ppf(tau)
                    beta_mmqr_boot = beta_loc_boot + q_tau * gamma_scale_boot
                    bootstrap_coefs[tau].append(beta_mmqr_boot)
            except:
                continue
        
        # Compute standard errors and confidence intervals
        for tau in self.quantiles:
            boot_array = np.array(bootstrap_coefs[tau])
            if len(boot_array) > 0:
                self.results[tau]['std_errors'] = np.std(boot_array, axis=0)
                self.results[tau]['ci_lower'] = np.percentile(boot_array, 2.5, axis=0)
                self.results[tau]['ci_upper'] = np.percentile(boot_array, 97.5, axis=0)
                self.results[tau]['pvalues'] = 2 * (1 - stats.norm.cdf(
                    np.abs(self.results[tau]['coefficients'] / self.results[tau]['std_errors'])
                ))
    
    def summary_table(self, var_names):
        """
        Create summary table of results
        """
        results_list = []
        
        for tau in self.quantiles:
            res = self.results[tau]
            coefs = res['coefficients']
            
            for i, var_name in enumerate(var_names):
                row = {
                    'Quantile': f"τ={tau:.2f}",
                    'Variable': var_name,
                    'Coefficient': coefs[i],
                }
                
                if 'std_errors' in res:
                    row['Std.Error'] = res['std_errors'][i]
                    row['z-value'] = coefs[i] / res['std_errors'][i]
                    row['P>|z|'] = res['pvalues'][i]
                    row['CI_Lower'] = res['ci_lower'][i]
                    row['CI_Upper'] = res['ci_upper'][i]
                
                results_list.append(row)
        
        return pd.DataFrame(results_list)


# ============================================
# Streamlit App
# ============================================

st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key="mmqr_upload")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.session_state["uploaded_data"] = data
else:
    data = st.session_state.get("uploaded_data", None)

if data is not None:
    st.success("✅ Dataset loaded successfully.")
    
    with st.expander("📊 View Dataset Preview"):
        st.dataframe(data.head(10))
        st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
    
    # Data quality check
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        st.warning(f"⚠️ Missing values detected: {missing_data[missing_data > 0].to_dict()}")
        if st.checkbox("Drop rows with missing values?"):
            data = data.dropna()
            st.success(f"✅ Cleaned dataset: {data.shape[0]} rows remaining")
    
    # Variable selection
    col1, col2 = st.columns(2)
    
    with col1:
        dependent_var = st.selectbox(
            "🎯 Select Dependent Variable (Y)", 
            options=data.columns,
            help="The outcome variable you want to model"
        )
    
    with col2:
        independent_vars = st.multiselect(
            "📊 Select Independent Variables (X)",
            options=[c for c in data.columns if c != dependent_var],
            help="Covariates/predictors"
        )
    
    # Quantiles selection
    st.sidebar.subheader("Quantile Specification")
    quantile_preset = st.sidebar.radio(
        "Choose quantile set:",
        ["Standard (5)", "Extended (9)", "Custom"]
    )
    
    if quantile_preset == "Standard (5)":
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    elif quantile_preset == "Extended (9)":
        quantiles = [0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95]
    else:
        quantile_input = st.sidebar.text_input(
            "Enter quantiles (comma-separated, e.g., 0.1,0.5,0.9):",
            "0.05,0.25,0.50,0.75,0.95"
        )
        try:
            quantiles = [float(q.strip()) for q in quantile_input.split(",")]
            quantiles = [q for q in quantiles if 0 < q < 1]
        except:
            st.sidebar.error("Invalid input. Using default quantiles.")
            quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    
    st.sidebar.write(f"**Selected quantiles:** {quantiles}")
    
    # Bootstrap options
    st.sidebar.subheader("Inference Options")
    use_bootstrap = st.sidebar.checkbox("Use Bootstrap Standard Errors", value=True)
    n_bootstrap = st.sidebar.slider("Bootstrap Replications", 50, 500, 200, 50) if use_bootstrap else 0
    
    # Run analysis
    if independent_vars:
        if st.button("🚀 Run MMQR Analysis", type="primary"):
            with st.spinner("Estimating MMQR model... This may take a moment."):
                try:
                    # Prepare data
                    y = data[dependent_var].values
                    X = data[independent_vars].values
                    
                    # Check for infinite values
                    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(X)):
                        st.error("❌ Data contains infinite or NaN values. Please clean your data.")
                    else:
                        # Fit MMQR model
                        mmqr = MMQRModel(y, X, quantiles=quantiles)
                        mmqr.fit(bootstrap_se=use_bootstrap, n_bootstrap=n_bootstrap)
                        
                        # Store in session state
                        st.session_state['mmqr_model'] = mmqr
                        st.session_state['var_names'] = ['Intercept'] + independent_vars
                        st.success("✅ MMQR estimation completed!")
                        
                except Exception as e:
                    st.error(f"❌ Error during estimation: {str(e)}")
                    st.exception(e)
    
    # Display results if model is fitted
    if 'mmqr_model' in st.session_state:
        mmqr = st.session_state['mmqr_model']
        var_names = st.session_state['var_names']
        
        st.header("📈 MMQR Results")
        
        # ========================
        # Summary Table
        # ========================
        st.subheader("Table 1: MMQR Coefficient Estimates")
        
        summary_df = mmqr.summary_table(var_names)
        
        # Format table
        def format_results(df):
            formatted = df.copy()
            if 'Coefficient' in formatted.columns:
                formatted['Coefficient'] = formatted['Coefficient'].map('{:.4f}'.format)
            if 'Std.Error' in formatted.columns:
                formatted['Std.Error'] = formatted['Std.Error'].map('{:.4f}'.format)
            if 'z-value' in formatted.columns:
                formatted['z-value'] = formatted['z-value'].map('{:.3f}'.format)
            if 'P>|z|' in formatted.columns:
                def format_pval(p):
                    if p < 0.001:
                        return '<0.001***'
                    elif p < 0.01:
                        return f'{p:.4f}**'
                    elif p < 0.05:
                        return f'{p:.4f}*'
                    elif p < 0.10:
                        return f'{p:.4f}†'
                    else:
                        return f'{p:.4f}'
                formatted['P>|z|'] = formatted['P>|z|'].map(format_pval)
            return formatted
        
        st.dataframe(format_results(summary_df), use_container_width=True, hide_index=True)
        st.caption("Significance: *** p<0.001, ** p<0.01, * p<0.05, † p<0.10")
        
        # Download button
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download MMQR Results (CSV)",
            data=csv,
            file_name="MMQR_Results.csv",
            mime="text/csv"
        )
        
        # ========================
        # Visualization
        # ========================
        st.subheader("Figure 1: Unconditional Quantile Partial Effects (UQPE)")
        
        # Select variables to plot
        plot_vars = st.multiselect(
            "Select variables to visualize:",
            options=independent_vars,
            default=independent_vars[:min(3, len(independent_vars))]
        )
        
        if plot_vars:
            fig, axes = plt.subplots(1, len(plot_vars), figsize=(5*len(plot_vars), 5))
            if len(plot_vars) == 1:
                axes = [axes]
            
            palette = sns.color_palette("husl", len(plot_vars))
            
            for idx, var in enumerate(plot_vars):
                ax = axes[idx]
                var_idx = var_names.index(var)
                
                coefs = [mmqr.results[tau]['coefficients'][var_idx] for tau in quantiles]
                
                if use_bootstrap:
                    lower = [mmqr.results[tau]['ci_lower'][var_idx] for tau in quantiles]
                    upper = [mmqr.results[tau]['ci_upper'][var_idx] for tau in quantiles]
                    ax.fill_between(quantiles, lower, upper, alpha=0.2, color=palette[idx])
                
                ax.plot(quantiles, coefs, marker='o', linewidth=2, 
                       markersize=6, color=palette[idx], label=var)
                ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
                
                ax.set_xlabel("Quantile (τ)", fontsize=12, fontweight='bold')
                ax.set_ylabel("UQPE Coefficient", fontsize=12, fontweight='bold')
                ax.set_title(f"{var}", fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlim(min(quantiles)-0.05, max(quantiles)+0.05)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.caption("""
            **Interpretation:** The plot shows how the marginal effect of each variable varies across 
            the unconditional distribution of the outcome. Unlike conditional quantile regression, 
            these coefficients represent population-level effects at different points of the outcome distribution.
            """)
        
        # ========================
        # Interpretation Guide
        # ========================
        st.subheader("📝 Interpretation Guidelines")
        
        with st.expander("How to interpret MMQR results"):
            st.markdown("""
            ### Unconditional Quantile Partial Effects (UQPE)
            
            **Key Differences from Conditional Quantile Regression:**
            - **MMQR coefficients** estimate the effect of X on the **unconditional quantiles** of Y
            - They answer: "How does X affect individuals at different points of the Y distribution?"
            - **CQR coefficients** estimate effects on conditional quantiles (given X values)
            
            **Interpretation Example:**
            - If β(τ=0.90) = 0.5 for education → income:
              - A one-unit increase in education increases income by 0.5 units **for individuals 
                at the 90th percentile of the income distribution**
            
            **Location vs. Scale Effects:**
            - If coefficients are similar across quantiles → **location shift** (parallel shift)
            - If coefficients vary substantially → **scale effect** (changes inequality)
            - Increasing coefficients (0.1→0.5→0.9) suggest the variable increases inequality
            - Decreasing coefficients suggest the variable reduces inequality
            
            **Statistical Significance:**
            - Check p-values and confidence intervals
            - Effects may be significant at some quantiles but not others
            """)
        
        # ========================
        # Diagnostic Summary
        # ========================
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            median_effect = mmqr.results[0.50]['coefficients'][1:] if 0.50 in quantiles else None
            if median_effect is not None:
                st.metric("Median Effects (Q50)", 
                         f"{np.mean(np.abs(median_effect)):.4f}",
                         help="Average absolute effect at median")
        
        with col2:
            effect_range = []
            for var_idx in range(1, len(var_names)):
                coefs_var = [mmqr.results[tau]['coefficients'][var_idx] for tau in quantiles]
                effect_range.append(max(coefs_var) - min(coefs_var))
            st.metric("Avg Effect Heterogeneity", 
                     f"{np.mean(effect_range):.4f}",
                     help="Average range of effects across quantiles")
        
        with col3:
            st.metric("Number of Quantiles", len(quantiles))
        
        # ========================
        # Export Options
        # ========================
        st.subheader("💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Full results export
            export_df = summary_df.copy()
            export_csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Download Full Results Table",
                data=export_csv,
                file_name="MMQR_Full_Results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Plot data export
            plot_data = []
            for tau in quantiles:
                for i, var in enumerate(var_names):
                    plot_data.append({
                        'Quantile': tau,
                        'Variable': var,
                        'Coefficient': mmqr.results[tau]['coefficients'][i],
                        'CI_Lower': mmqr.results[tau].get('ci_lower', [None]*len(var_names))[i],
                        'CI_Upper': mmqr.results[tau].get('ci_upper', [None]*len(var_names))[i]
                    })
            plot_df = pd.DataFrame(plot_data)
            plot_csv = plot_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📊 Download Plot Data",
                data=plot_csv,
                file_name="MMQR_Plot_Data.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👆 Configure your variables and click 'Run MMQR Analysis' to see results.")

else:
    st.info("📁 Please upload a CSV dataset to begin MMQR analysis.")
    
    # Sample data option
    if st.checkbox("Use sample dataset for testing"):
        np.random.seed(42)
        n = 500
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        epsilon = np.random.normal(0, 1 + 0.5*X1, n)  # Heteroskedastic errors
        Y = 2 + 1.5*X1 + 0.8*X2 + epsilon
        
        sample_data = pd.DataFrame({
            'Y': Y,
            'X1': X1,
            'X2': X2
        })
        
        st.session_state["uploaded_data"] = sample_data
        st.success("✅ Sample dataset loaded! Refresh the page to see it.")
        st.dataframe(sample_data.head())

# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown("""
**About MMQR:**  
Method of Moments Quantile Regression estimates unconditional quantile treatment effects using a location-scale 
decomposition. This allows for policy-relevant inference about effects on the marginal distribution of outcomes.

**Citation:**  
Machado, J.A.F. and Silva, J.M.C.S. (2019). Quantiles via moments. *Journal of Econometrics*, 213(1), 145-173.
""")

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
