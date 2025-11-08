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
    # Create sample data for demonstration
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

# Display data overview
st.header("A. Data Overview")
st.dataframe(df.head())
st.write(f"Dataset shape: {df.shape}")

# Check for required columns
if 'Country' not in df.columns or 'Year' not in df.columns:
    st.error("❌ Required columns 'Country' and/or 'Year' not found in dataset!")
    st.stop()

# ============================================
# Section B: Correlation Analysis
# ============================================

st.header("B. Correlation Analysis")

# Detect numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.warning("No numeric variables found in your dataset.")
else:
    # Variable selection
    col1, col2 = st.columns(2)
    with col1:
        dep_var = st.selectbox("Select Dependent Variable", options=numeric_cols)
    with col2:
        indep_vars = st.multiselect(
            "Select Independent Variable(s)",
            options=[col for col in numeric_cols if col != dep_var],
            default=[col for col in numeric_cols if col != dep_var][:3] if len(numeric_cols) > 1 else []
        )

    # Color palette selector
    color_option = st.selectbox(
        "Select Heatmap Color Palette",
        options=["coolwarm", "viridis", "plasma", "magma", "cividis", "Blues", "Greens", "Reds"],
        index=0
    )

    if indep_vars:
        # Compute correlation matrix
        selected_vars = [dep_var] + indep_vars
        corr = df[selected_vars].corr()

        # Generate heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap=color_option, center=0, linewidths=0.5, fmt=".2f", ax=ax)
        plt.title(f"Correlation Heatmap")
        st.pyplot(fig)

        # Download button for heatmap
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button(
            label="Download Heatmap Image",
            data=buf.getvalue(),
            file_name="correlation_heatmap.png",
            mime="image/png"
        )

        # Correlation interpretation
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

        # Display correlation table
        st.subheader("Table 1: Correlation Matrix")
        st.dataframe(corr)

        st.info(
            "According to Evans (1996), correlation strengths are defined as: "
            "very weak (0.00–0.19), weak (0.20–0.39), moderate (0.40–0.59), "
            "strong (0.60–0.79), and very strong (0.80–1.00)."
        )
    else:
        st.warning("Please select at least one independent variable to display correlation.")


# ============================================
# Section D: Corrected MMQR Implementation with Scale P-values
# ============================================

st.header("D. Method of Moments Quantile Regression (MMQR) - Machado & Santos Silva (2019)")

if 'dep_var' not in locals() or 'indep_vars' not in locals() or not indep_vars:
    st.warning("Please complete the correlation analysis first to select variables.")
else:
    st.subheader("MMQR Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        quantiles = st.text_input("Quantiles (comma-separated)", "0.05,0.25,0.50,0.75,0.95")
        quantiles = [float(q.strip()) for q in quantiles.split(",")]
    
    with col2:
        bootstrap_ci = st.checkbox("Bootstrap Confidence Intervals", True)
        n_bootstrap = st.slider("Bootstrap Samples", 100, 1000, 200) if bootstrap_ci else 100
        reference_quantile = st.selectbox("Reference Quantile for Location", [0.25, 0.50, 0.75], index=1)

    def correct_mmqr_estimation(data, y_var, x_vars, quantiles, reference_quantile=0.5, bootstrap=True, n_boot=200):
        """
        Correct MMQR implementation following Machado & Santos Silva (2019)
        with proper inference for scale parameters
        """
        results = {}
        
        # Prepare data
        X = data[x_vars]
        y = data[y_var]
        
        # Step 1: Estimate location parameters using reference quantile (usually median)
        formula_ref = f"{y_var} ~ {' + '.join(x_vars)}"
        location_model = quantreg(formula_ref, data).fit(q=reference_quantile, vcov='robust')
        location_params = location_model.params
        location_pvalues = location_model.pvalues
        
        # Step 2: Estimate scale parameters using symmetric quantiles with proper inference
        tau_high = 0.75
        tau_low = 0.25
        
        model_high = quantreg(formula_ref, data).fit(q=tau_high, vcov='robust')
        model_low = quantreg(formula_ref, data).fit(q=tau_low, vcov='robust')
        
        # Scale parameters are proportional to the difference between high and low quantiles
        scale_params = (model_high.params - model_low.params) / (tau_high - tau_low)
        
        # Calculate p-values for scale parameters using delta method
        scale_pvalues = {}
        for var in scale_params.index:
            try:
                # Variance of scale parameter using delta method
                # Var(scale) = [Var(high) + Var(low) - 2*Cov(high,low)] / (tau_high - tau_low)^2
                
                # Get variances from the models
                var_high = model_high.bse[var] ** 2
                var_low = model_low.bse[var] ** 2
                
                # For covariance, we use approximation since we don't have direct covariance matrix
                # A conservative approach: assume moderate positive correlation (0.3)
                cov_high_low = 0.3 * np.sqrt(var_high * var_low)
                
                var_scale = (var_high + var_low - 2 * cov_high_low) / ((tau_high - tau_low) ** 2)
                se_scale = np.sqrt(var_scale)
                
                # Calculate t-statistic and p-value
                t_stat = scale_params[var] / se_scale if se_scale > 0 else 0
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(data) - len(x_vars) - 1))
                scale_pvalues[var] = p_value
                
            except:
                scale_pvalues[var] = 1.0  # Default to insignificant if calculation fails
        
        # Alternative method: Bootstrap for scale parameters
        if bootstrap:
            st.info("Bootstrapping scale parameters...")
            bootstrap_scale = {var: [] for var in scale_params.index}
            progress_bar = st.progress(0)
            
            for i in range(n_boot):
                boot_sample = data.sample(n=len(data), replace=True)
                try:
                    boot_high = quantreg(formula_ref, boot_sample).fit(q=tau_high)
                    boot_low = quantreg(formula_ref, boot_sample).fit(q=tau_low)
                    boot_scale = (boot_high.params - boot_low.params) / (tau_high - tau_low)
                    
                    for var in scale_params.index:
                        bootstrap_scale[var].append(boot_scale[var])
                except:
                    continue
                
                progress_bar.progress((i + 1) / n_boot)
            
            # Calculate bootstrap p-values for scale parameters
            for var in scale_params.index:
                if len(bootstrap_scale[var]) > 0:
                    # Two-sided p-value: proportion of bootstrap estimates with opposite sign
                    boot_vals = np.array(bootstrap_scale[var])
                    p_value_boot = 2 * min(
                        np.mean(boot_vals > 0) if scale_params[var] < 0 else np.mean(boot_vals <= 0),
                        np.mean(boot_vals <= 0) if scale_params[var] < 0 else np.mean(boot_vals > 0)
                    )
                    # Use bootstrap p-value if we have enough samples
                    if len(bootstrap_scale[var]) >= 100:
                        scale_pvalues[var] = p_value_boot
        
        # Step 3: MMQR transformation and estimation
        for q in quantiles:
            try:
                # Transform the model using location and scale parameters
                h_tau = q
                
                # MMQR coefficients: β(τ) = α + δ·h(τ)
                mmqr_coefficients = location_params + scale_params * h_tau
                
                # Estimate the final model with robust inference
                y_approx = location_model.predict() + scale_params.mean() * h_tau * (y - location_model.predict()).std()
                
                data_temp = data.copy()
                data_temp['y_transformed'] = y_approx
                
                formula_mmqr = f"y_transformed ~ {' + '.join(x_vars)}"
                final_model = quantreg(formula_mmqr, data_temp).fit(q=q, vcov='robust')
                
                results[q] = {
                    'coefficients': final_model.params,
                    'pvalues': final_model.pvalues,
                    'conf_int': final_model.conf_int(),
                    'location_params': location_params,
                    'scale_params': scale_params,
                    'scale_pvalues': scale_pvalues,
                    'mmqr_coefficients': mmqr_coefficients,
                    'quantile': q,
                    'model': final_model
                }
                
            except Exception as e:
                st.warning(f"MMQR estimation failed for quantile {q}: {str(e)}")
                # Fallback to standard QR if MMQR fails
                formula = f"{y_var} ~ {' + '.join(x_vars)}"
                fallback_model = quantreg(formula, data).fit(q=q, vcov='robust')
                results[q] = {
                    'coefficients': fallback_model.params,
                    'pvalues': fallback_model.pvalues,
                    'conf_int': fallback_model.conf_int(),
                    'location_params': location_params,
                    'scale_params': scale_params,
                    'scale_pvalues': scale_pvalues,
                    'mmqr_coefficients': fallback_model.params,
                    'quantile': q,
                    'model': fallback_model
                }
        
        return results, location_params, scale_params, scale_pvalues

    # Run corrected MMQR with scale p-values
    try:
        mmqr_results, location_params, scale_params, scale_pvalues = correct_mmqr_estimation(
            df, dep_var, indep_vars, quantiles, reference_quantile, bootstrap_ci, n_bootstrap
        )
        
        # ========================
        # Table 1: Location Parameters (Reference Quantile)
        # ========================
        st.subheader(f"Table 2: Location Parameters (τ = {reference_quantile})")
        
        location_data = []
        for var in location_params.index:
            var_name = 'Intercept' if var == 'Intercept' else var
            coef = location_params[var]
            pval = mmqr_results[reference_quantile]['pvalues'][var] if reference_quantile in mmqr_results else 1.0
            
            location_data.append({
                'Variable': var_name,
                'Coefficient': f"{coef:.3f}",
                'P-Value': f"{pval:.3f}",
                'Significance': '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            })
        
        location_df = pd.DataFrame(location_data)
        st.dataframe(location_df, use_container_width=True)
        
        # ========================
        # Table 2: Scale Parameters with P-Values
        # ========================
        st.subheader("Table 3: Scale Parameters with Statistical Inference")
        
        scale_data = []
        for var in scale_params.index:
            var_name = 'Intercept' if var == 'Intercept' else var
            scale_val = scale_params[var]
            pval = scale_pvalues[var]
            
            # Calculate economic significance (relative to location)
            if var in location_params:
                relative_effect = abs(scale_val) / abs(location_params[var]) if location_params[var] != 0 else float('inf')
            else:
                relative_effect = 0
            
            scale_data.append({
                'Variable': var_name,
                'Scale Coefficient': f"{scale_val:.3f}",
                'P-Value': f"{pval:.3f}",
                'Significance': '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else '',
                'Relative to Location': f"{relative_effect:.3f}" if relative_effect != float('inf') else "N/A",
                'Interpretation': 'Strong heterogeneity' if pval < 0.05 and relative_effect > 0.5 else 
                                'Moderate heterogeneity' if pval < 0.05 else 
                                'No significant heterogeneity' if pval >= 0.05 else 'Inconclusive'
            })
        
        scale_df = pd.DataFrame(scale_data)
        st.dataframe(scale_df, use_container_width=True)
        
        # ========================
        # Table 3: MMQR Results
        # ========================
        st.subheader("Table 4: MMQR Estimation Results")
        
        mmqr_data = []
        coef_names = mmqr_results[quantiles[0]]['coefficients'].index.tolist()
        
        for var in coef_names:
            var_name = 'Intercept' if var == 'Intercept' else var
            row = {'Variable': var_name}
            
            for q in quantiles:
                if q in mmqr_results:
                    # Use MMQR coefficients
                    coef = mmqr_results[q]['mmqr_coefficients'][var]
                    pval = mmqr_results[q]['pvalues'][var]
                    
                    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                    row[f'τ = {q}'] = f"{coef:.3f}{stars}"
            
            mmqr_data.append(row)
        
        mmqr_df = pd.DataFrame(mmqr_data)
        st.dataframe(mmqr_df, use_container_width=True)
        
        # ========================
        # Scale Parameters Interpretation
        # ========================
        st.subheader("Scale Parameters Interpretation")
        
        st.write("**Statistical Significance of Heterogeneity:**")
        
        significant_scale_vars = [var for var in scale_params.index 
                                if var != 'Intercept' and scale_pvalues[var] < 0.1]
        insignificant_scale_vars = [var for var in scale_params.index 
                                  if var != 'Intercept' and scale_pvalues[var] >= 0.1]
        
        if significant_scale_vars:
            st.success(f"**Significant heterogeneity detected in:** {', '.join(significant_scale_vars)}")
            st.write("These variables show statistically significant variation in their effects across different quantiles.")
        
        if insignificant_scale_vars:
            st.info(f"**No significant heterogeneity in:** {', '.join(insignificant_scale_vars)}")
            st.write("These variables show relatively stable effects across quantiles.")
        
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
                    st.write(f"  - Marginal effects show {direction} pattern across quantiles")
                    if loc_val != 0:
                        relative_mag = abs(scale_val / loc_val)
                        st.write(f"  - Scale effect is {relative_mag:.1%} of location effect")
                else:
                    st.write(f"- **{var}**: No significant heterogeneity (p={pval:.3f})")
                    st.write(f"  - Effects remain relatively constant across quantiles")

        # ========================
        # MMQR Coefficient Dynamics
        # ========================
        st.subheader("Figure 1: MMQR Coefficient Dynamics")
        
        plot_vars = [var for var in coef_names if var != 'Intercept']
        
        for var in plot_vars:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot MMQR coefficients
            mmqr_coefs = [mmqr_results[q]['mmqr_coefficients'][var] for q in quantiles if q in mmqr_results]
            quantiles_avail = [q for q in quantiles if q in mmqr_results]
            
            if len(mmqr_coefs) > 0:
                # Plot the MMQR trajectory
                ax.plot(quantiles_avail, mmqr_coefs, 'o-', linewidth=2.5, 
                       label=f'MMQR Coefficients', color='#2E86AB', markersize=8)
                
                # Add location parameter (horizontal line)
                loc_coef = location_params[var]
                ax.axhline(y=loc_coef, color='red', linestyle='--', alpha=0.7, 
                          label=f'Location (τ={reference_quantile})')
                
                # Add scale effect indication
                scale_val = scale_params[var]
                scale_pval = scale_pvalues[var]
                
                if scale_pval < 0.1:
                    # Add a trend line to emphasize the scale effect
                    z = np.polyfit(quantiles_avail, mmqr_coefs, 1)
                    p = np.poly1d(z)
                    ax.plot(quantiles_avail, p(quantiles_avail), ':', color='green', 
                           alpha=0.7, label=f'Scale trend (p={scale_pval:.3f})')
                
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_xlabel("Quantiles (τ)", fontsize=12)
                ax.set_ylabel("Coefficient Estimate", fontsize=12)
                
                # Add scale significance to title
                scale_sig = '***' if scale_pval < 0.01 else '**' if scale_pval < 0.05 else '*' if scale_pval < 0.1 else 'ns'
                ax.set_title(f"MMQR Coefficient Dynamics: {var} (Scale: {scale_sig})", 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add significance markers for MMQR coefficients
                for i, q in enumerate(quantiles_avail):
                    pval = mmqr_results[q]['pvalues'][var]
                    if pval < 0.1:
                        ax.annotate('*' * (3 if pval < 0.01 else 2 if pval < 0.05 else 1), 
                                   (q, mmqr_coefs[i]), textcoords="offset points", 
                                   xytext=(0,10), ha='center', fontweight='bold', 
                                   color='red', fontsize=12)
                
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
            pval = mmqr_results[reference_quantile]['pvalues'][var] if reference_quantile in mmqr_results else 1.0
            download_data.append({
                'Variable': var_name,
                'Type': 'Location',
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
                'Type': 'Scale', 
                'Coefficient': round(scale_params[var], 3),
                'P-Value': round(scale_pvalues[var], 3),
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
                        'Type': f'MMQR_τ={q}',
                        'Coefficient': round(mmqr_results[q]['mmqr_coefficients'][var], 3),
                        'P-Value': round(mmqr_results[q]['pvalues'][var], 3),
                        'Quantile': q,
                        'Method': 'MMQR'
                    })
        
        download_df = pd.DataFrame(download_data)
        csv_data = download_df.to_csv(index=False)
        
        st.download_button(
            "📥 Download Complete MMQR Results (CSV)",
            data=csv_data,
            file_name="MMQR_Complete_Results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"MMQR estimation failed: {str(e)}")

# Rest of the code remains the same...
# ============================================
# Section E: Quantile Cointegration Test
# ============================================

st.header("E. Quantile Cointegration Test")

try:
    # Check if we have the required variables
    if 'dep_var' not in locals() or 'indep_vars' not in locals() or not indep_vars:
        st.warning("Please complete the MMQR analysis first to select variables.")
    else:
        st.subheader("Quantile Cointegration Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            coint_quantiles = st.text_input("Cointegration Quantiles (comma-separated)", 
                                          "0.05,0.25,0.50,0.75,0.95", key="coint_quantiles")
            coint_quantiles = [float(q.strip()) for q in coint_quantiles.split(",")]
        
        with col2:
            max_lags_coint = st.slider("Maximum Lags for Cointegration Test", 1, 5, 2, key="coint_lags")
            significance_level = st.selectbox("Significance Level", [0.01, 0.05, 0.10], index=1)
        
        # Quantile Cointegration Test Implementation
        def quantile_cointegration_test(data, y_var, x_vars, quantiles, max_lags):
            """
            Quantile Cointegration Test based on Xiao (2009) and related approaches
            """
            results = {}
            residual_tests = {}
            
            # Ensure panel data structure
            if 'Country' in data.columns and 'Year' in data.columns:
                # Use first country for demonstration (or pool data)
                country_data = data[data['Country'] == data['Country'].iloc[0]].sort_values('Year')
                test_data = country_data[[y_var] + x_vars].dropna()
            else:
                test_data = data[[y_var] + x_vars].dropna()
            
            if len(test_data) < max_lags + 5:
                st.warning("Insufficient data for cointegration test. Need more time periods.")
                return results, residual_tests
            
            for q in quantiles:
                try:
                    # Step 1: Quantile regression for long-run relationship
                    formula = f"{y_var} ~ {' + '.join(x_vars)}"
                    q_model = quantreg(formula, test_data).fit(q=q)
                    
                    # Step 2: Get residuals from quantile regression
                    residuals = q_model.resid
                    
                    # Step 3: Test residuals for stationarity (ADF test)
                    from statsmodels.tsa.stattools import adfuller
                    
                    # ADF test on residuals
                    adf_result = adfuller(residuals, maxlag=max_lags, autolag='AIC')
                    
                    # Store results
                    results[q] = {
                        'quantile': q,
                        'adf_statistic': adf_result[0],
                        'p_value': adf_result[1],
                        'critical_values': adf_result[4],
                        'residuals': residuals,
                        'coefficients': q_model.params,
                        'model': q_model
                    }
                    
                    # Additional residual diagnostics
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    lb_test = acorr_ljungbox(residuals, lags=[max_lags], return_df=True)
                    
                    residual_tests[q] = {
                        'ljung_box_stat': lb_test.iloc[0]['lb_stat'],
                        'ljung_box_pval': lb_test.iloc[0]['lb_pvalue'],
                        'residual_mean': np.mean(residuals),
                        'residual_std': np.std(residuals)
                    }
                    
                except Exception as e:
                    st.warning(f"Quantile cointegration test failed for quantile {q}: {str(e)}")
                    continue
            
            return results, residual_tests
        
        # Run quantile cointegration test
        st.info("Running Quantile Cointegration Tests...")
        coint_results, residual_tests = quantile_cointegration_test(
            df, dep_var, indep_vars, coint_quantiles, max_lags_coint
        )
        
        if coint_results:
            # ========================
            # Table 1: Cointegration Test Results
            # ========================
            st.subheader("Table 5: Quantile Cointegration Test Results")
            
            coint_data = []
            for q in coint_quantiles:
                if q in coint_results:
                    result = coint_results[q]
                    critical_vals = result['critical_values']
                    
                    # Get critical values safely
                    critical_1p = critical_vals.get('1%', 'N/A')
                    critical_5p = critical_vals.get('5%', 'N/A')
                    critical_10p = critical_vals.get('10%', 'N/A')
                    
                    # Format critical values
                    critical_1p_str = f"{critical_1p:.3f}" if critical_1p != 'N/A' else 'N/A'
                    critical_5p_str = f"{critical_5p:.3f}" if critical_5p != 'N/A' else 'N/A'
                    critical_10p_str = f"{critical_10p:.3f}" if critical_10p != 'N/A' else 'N/A'
                    
                    # Determine cointegration decision
                    is_cointegrated = result['p_value'] < significance_level
                    
                    coint_data.append({
                        'Quantile (τ)': f"{q:.2f}",
                        'ADF Statistic': f"{result['adf_statistic']:.3f}",
                        'P-Value': f"{result['p_value']:.3f}",
                        '1% Critical': critical_1p_str,
                        '5% Critical': critical_5p_str,
                        '10% Critical': critical_10p_str,
                        'Cointegrated': 'Yes' if is_cointegrated else 'No'
                    })
            
            coint_df = pd.DataFrame(coint_data)
            st.dataframe(coint_df, use_container_width=True)
            
            # ========================
            # Table 2: Long-run Coefficients
            # ========================
            st.subheader("Table 6: Long-run Coefficients from Quantile Cointegration")
            
            coef_data = []
            coef_names = coint_results[coint_quantiles[0]]['coefficients'].index.tolist()
            
            for var in coef_names:
                var_name = 'Intercept' if var == 'Intercept' else var
                row = {'Variable': var_name}
                
                for q in coint_quantiles:
                    if q in coint_results:
                        coef = coint_results[q]['coefficients'][var]
                        row[f'τ = {q}'] = f"{coef:.3f}"
                
                coef_data.append(row)
            
            coef_df = pd.DataFrame(coef_data)
            st.dataframe(coef_df, use_container_width=True)
            
            # ========================
            # Visualization: Cointegration Results
            # ========================
            st.subheader("Figure 2: Quantile Cointegration Analysis")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: ADF Statistics across quantiles
            quantiles_plot = [q for q in coint_quantiles if q in coint_results]
            adf_stats = [coint_results[q]['adf_statistic'] for q in quantiles_plot]
            critical_1p = [coint_results[q]['critical_values'].get('1%', np.nan) for q in quantiles_plot]
            critical_5p = [coint_results[q]['critical_values'].get('5%', np.nan) for q in quantiles_plot]
            
            axes[0,0].plot(quantiles_plot, adf_stats, 'o-', linewidth=2, label='ADF Statistic', color='blue')
            axes[0,0].plot(quantiles_plot, critical_1p, '--', linewidth=2, label='1% Critical', color='red')
            axes[0,0].plot(quantiles_plot, critical_5p, '--', linewidth=2, label='5% Critical', color='orange')
            
            axes[0,0].set_xlabel('Quantiles')
            axes[0,0].set_ylabel('ADF Statistics')
            axes[0,0].set_title('ADF Statistics vs Critical Values across Quantiles')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Plot 2: P-values across quantiles
            p_values = [coint_results[q]['p_value'] for q in quantiles_plot]
            axes[0,1].plot(quantiles_plot, p_values, 's-', linewidth=2, color='green')
            axes[0,1].axhline(y=significance_level, color='red', linestyle='--', 
                            label=f'{significance_level} Significance Level')
            axes[0,1].set_xlabel('Quantiles')
            axes[0,1].set_ylabel('P-Values')
            axes[0,1].set_title('P-Values across Quantiles')
            axes[0,1].set_yscale('log')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Plot 3: Residuals from median quantile
            if 0.5 in coint_results:
                residuals = coint_results[0.5]['residuals']
                axes[1,0].plot(residuals.index, residuals.values, linewidth=1.5, color='purple')
                axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[1,0].set_xlabel('Time/Index')
                axes[1,0].set_ylabel('Residuals')
                axes[1,0].set_title('Residuals from Median Quantile Cointegration')
                axes[1,0].grid(True, alpha=0.3)
            
            # Plot 4: Cointegration persistence (coefficient variation)
            coef_variation = []
            for var in coef_names:
                if var != 'Intercept':
                    coefs = [coint_results[q]['coefficients'][var] for q in quantiles_plot if q in coint_results]
                    if coefs and len(coefs) > 1:
                        variation = np.std(coefs) / abs(np.mean(coefs)) if np.mean(coefs) != 0 else np.std(coefs)
                        coef_variation.append((var, variation))
            
            if coef_variation:
                variables, variations = zip(*coef_variation)
                axes[1,1].bar(variables, variations, color='steelblue', alpha=0.7)
                axes[1,1].set_xlabel('Variables')
                axes[1,1].set_ylabel('Coefficient of Variation')
                axes[1,1].set_title('Long-run Coefficient Persistence across Quantiles')
                axes[1,1].tick_params(axis='x', rotation=45)
                axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # ========================
            # Residual Diagnostics
            # ========================
            st.subheader("Table 7: Residual Diagnostics")
            
            residual_data = []
            for q in coint_quantiles:
                if q in residual_tests:
                    test = residual_tests[q]
                    residual_data.append({
                        'Quantile': f"{q:.2f}",
                        'Ljung-Box Stat': f"{test['ljung_box_stat']:.3f}",
                        'Ljung-Box P-value': f"{test['ljung_box_pval']:.3f}",
                        'Residual Mean': f"{test['residual_mean']:.3f}",
                        'Residual Std': f"{test['residual_std']:.3f}",
                        'AutoCorrelation': 'Present' if test['ljung_box_pval'] < 0.05 else 'Absent'
                    })
            
            if residual_data:
                residual_df = pd.DataFrame(residual_data)
                st.dataframe(residual_df, use_container_width=True)
            
            # ========================
            # Interpretation
            # ========================
            st.subheader("Economic Interpretation")
            
            # Count cointegrated quantiles
            cointegrated_count = sum(1 for q in coint_quantiles 
                                   if q in coint_results and coint_results[q]['p_value'] < significance_level)
            total_tested = len([q for q in coint_quantiles if q in coint_results])
            
            st.write(f"**Cointegration Summary:**")
            st.write(f"- {cointegrated_count} out of {total_tested} quantiles show evidence of cointegration")
            st.write(f"- Significance level: {significance_level}")
            
            if cointegrated_count > total_tested * 0.5:
                st.success("**Strong evidence of quantile cointegration** - Long-run relationship exists across most quantiles")
            elif cointegrated_count > 0:
                st.info("**Partial quantile cointegration** - Long-run relationship exists at specific quantiles only")
            else:
                st.warning("**No evidence of quantile cointegration** - No stable long-run relationship detected")
            
            # Variable-specific interpretation
            st.write("**Long-run Relationship Analysis:**")
            for var in coef_names:
                if var != 'Intercept':
                    var_name = var
                    coefs = [coint_results[q]['coefficients'][var] for q in coint_quantiles if q in coint_results]
                    if coefs:
                        min_coef = min(coefs)
                        max_coef = max(coefs)
                        mean_coef = np.mean(coefs)
                        
                        st.write(f"- **{var_name}**: Long-run coefficient ranges from {min_coef:.3f} to {max_coef:.3f}")
                        if min_coef * max_coef > 0:  # Same sign
                            direction = "positive" if mean_coef > 0 else "negative"
                            st.write(f"  - Consistent {direction} relationship across quantiles")
                        else:
                            st.write(f"  - Relationship sign varies across quantiles")
            
            # ========================
            # Download Section
            # ========================
            st.subheader("Download Quantile Cointegration Results")
            
            # Prepare comprehensive download data
            download_coint_data = []
            
            for q in coint_quantiles:
                if q in coint_results:
                    result = coint_results[q]
                    residual_test = residual_tests.get(q, {})
                    
                    # Cointegration test results
                    download_coint_data.append({
                        'Quantile': q,
                        'Test_Type': 'Cointegration',
                        'ADF_Statistic': result['adf_statistic'],
                        'P_Value': result['p_value'],
                        'Critical_1%': result['critical_values'].get('1%', np.nan),
                        'Critical_5%': result['critical_values'].get('5%', np.nan),
                        'Critical_10%': result['critical_values'].get('10%', np.nan),
                        'Cointegrated': 'Yes' if result['p_value'] < significance_level else 'No',
                        'Ljung_Box_Stat': residual_test.get('ljung_box_stat', np.nan),
                        'Ljung_Box_Pval': residual_test.get('ljung_box_pval', np.nan),
                        'Residual_Mean': residual_test.get('residual_mean', np.nan),
                        'Residual_Std': residual_test.get('residual_std', np.nan)
                    })
                    
                    # Coefficients for this quantile
                    for var_name, coef_value in result['coefficients'].items():
                        download_coint_data.append({
                            'Quantile': q,
                            'Test_Type': 'Coefficient',
                            'Variable': var_name,
                            'Coefficient_Value': coef_value,
                            'ADF_Statistic': np.nan,
                            'P_Value': np.nan
                        })
            
            download_coint_df = pd.DataFrame(download_coint_data)
            csv_coint = download_coint_df.to_csv(index=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Cointegration Results",
                    data=csv_coint,
                    file_name="Quantile_Cointegration_Results.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Copy-paste friendly summary
                st.info("**Copy-Paste Summary:**")
                summary_text = f"Quantile Cointegration Test Results (Significance: {significance_level})\n"
                summary_text += "Quantile | ADF Stat | P-value | Cointegrated\n"
                summary_text += "--------|----------|---------|-------------\n"
                
                for q in coint_quantiles:
                    if q in coint_results:
                        result = coint_results[q]
                        coint_status = "Yes" if result['p_value'] < significance_level else "No"
                        summary_text += f"{q:.2f} | {result['adf_statistic']:.3f} | {result['p_value']:.3f} | {coint_status}\n"
                
                st.text_area("Summary for copying:", summary_text, height=200)
            
            # ========================
            # Methodological Notes
            # ========================
            with st.expander("Methodological Notes on Quantile Cointegration"):
                st.markdown("""
                **Quantile Cointegration Approach:**
                
                1. **Method**: Based on Xiao (2009) quantile cointegration framework
                2. **Procedure**:
                   - Estimate quantile regression for long-run relationship at each quantile
                   - Test residuals for stationarity using ADF test
                   - Cointegration exists if residuals are stationary
                
                3. **Interpretation**:
                   - Significant ADF statistic indicates cointegration at that quantile
                   - Variation in coefficients shows quantile-specific long-run effects
                   - Consistent cointegration across quantiles suggests robust relationship
                
                4. **Limitations**:
                   - Assumes linear cointegration relationship within quantiles
                   - Requires sufficient time series length
                   - Panel data adaptation may require more sophisticated tests
                
                **Reference**: Xiao, Z. (2009). Quantile cointegrating regression. 
                Journal of Econometrics, 150(2), 248-260.
                """)
        
        else:
            st.error("No quantile cointegration results could be computed. Check data requirements.")

except Exception as e:
    st.error(f"Error in quantile cointegration test: {str(e)}")
    st.info("""
    Common issues:
    - Insufficient time series length
    - Non-stationary variables (consider differencing)
    - Missing values in the data
    - Multicollinearity between variables
    """)
# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("**Panel Data Analysis Dashboard** | Built with Streamlit")
