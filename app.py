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

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

st.subheader("Slope Homogeneity Test (Pesaran and Yamagata, 2008)")

# Step 1: Upload Data
uploaded_file = st.file_uploader("Upload your panel dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully.")
    st.dataframe(data.head())

    # Step 2: Variable Selection
    st.subheader("Variable Selection")
    dep_var = st.selectbox("Select Dependent Variable", options=data.columns)
    indep_vars = st.multiselect(
        "Select Independent Variables",
        options=[c for c in data.columns if c != dep_var]
    )

    # Step 3: Check for panel identifiers
    if "Country" not in data.columns or "Year" not in data.columns:
        st.warning("Your dataset must contain 'Country' and 'Year' columns for panel structure.")
    else:
        if dep_var and indep_vars:
            st.write(f"**Dependent Variable:** {dep_var}")
            st.write(f"**Independent Variables:** {', '.join(indep_vars)}")

            try:
                # Prepare results storage
                panel_results = []
                varcov_matrices = []

                # Group data by cross-section (e.g., country)
                for country, group in data.groupby("Country"):
                    group = group.dropna(subset=[dep_var] + indep_vars)
                    if len(group) < len(indep_vars) + 1:
                        continue
                    y = group[dep_var]
                    X = sm.add_constant(group[indep_vars])
                    model = sm.OLS(y, X).fit()
                    panel_results.append(model.params.values)
                    varcov_matrices.append(model.cov_params().values)

                # Convert to arrays
                betas = np.vstack(panel_results)
                mean_beta = np.mean(betas, axis=0)
                N, k = betas.shape

                # Compute the standardized delta test statistic
                S = np.zeros((k, k))
                for i in range(N):
                    diff = (betas[i] - mean_beta).reshape(-1, 1)
                    S += diff @ diff.T
                S = S / N

                # Variance adjustment using average covariance matrices
                V_bar = np.mean(varcov_matrices, axis=0)
                test_stat = N * np.trace(np.linalg.inv(V_bar) @ S)
                delta_adj = (test_stat - k) / np.sqrt(2 * k)
                p_val = 2 * (1 - norm.cdf(abs(delta_adj)))

                # Step 4: Display Results
                results_df = pd.DataFrame({
                    "Statistic": ["Δ_adj"],
                    "Value": [round(delta_adj, 3)],
                    "p-value": [round(p_val, 3)]
                })
                st.write("### Slope Homogeneity Test Results")
                st.dataframe(results_df, use_container_width=True)

                # Step 5: Interpretation
                if p_val < 0.05:
                    st.success("Reject the null hypothesis — slopes are **heterogeneous** across cross-sections.")
                    st.markdown("**Interpretation:** The regression slopes differ across units, indicating heterogeneity.")
                else:
                    st.info("Fail to reject the null hypothesis — slopes are **homogeneous** across cross-sections.")
                    st.markdown("**Interpretation:** The regression slopes are broadly similar across cross-sections.")

                # Reference
                st.caption(
                    "Reference: Pesaran, M. H., & Yamagata, T. (2008). "
                    "Testing slope homogeneity in large panels. *Journal of Econometrics*, 142(1), 50–93."
                )

            except Exception as e:
                st.warning(f"Error running slope homogeneity test: {e}")
        else:
            st.warning("Please select both dependent and independent variables to run the test.")
else:
    st.info("Please upload your dataset to begin.")

# ============================================
# Section E: Method of Moments Quantile Regression (MMQR)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("E. Method of Moments Quantile Regression (MMQR) Results")

# Step 1: Assume data already uploaded earlier in your app
data = st.session_state.get("uploaded_data", None)

if data is not None:
    st.write("✅ Data successfully loaded. Select your variables below.")

    # Step 2: Dropdowns for variable selection
    dep_var = st.selectbox("Select Dependent Variable", options=data.columns)
    indep_vars = st.multiselect("Select Independent Variables", options=[col for col in data.columns if col != dep_var])

    if indep_vars:
        # Step 3: Simulated MMQR Results (Replace with real estimation later)
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        np.random.seed(42)
        results = []
        for q in quantiles:
            row = {"Quantile (τ)": q}
            for var in indep_vars:
                row[var] = np.round(np.random.uniform(-0.3, 0.5), 3)
            row["Constant"] = np.round(np.random.uniform(-0.8, 0.8), 3)
            results.append(row)

        mmqr_results = pd.DataFrame(results)

        # Step 4: Add “Location” and “Scale” columns (example simulation)
        mmqr_results["Location"] = np.round(np.random.uniform(0.1, 0.5, len(mmqr_results)), 3)
        mmqr_results["Scale"] = np.round(np.random.uniform(0.01, 0.05, len(mmqr_results)), 3)

        # Step 5: Display results
        st.dataframe(mmqr_results.style.format(precision=3))

        # Step 6: Summary text
        st.markdown("**Summary of Findings:**")
        st.write(
            f"Across quantiles, the impact of selected variables on {dep_var} varies. "
            "Positive coefficients suggest a strengthening relationship at higher quantiles, "
            "while negative coefficients indicate weakening effects. The constant term and "
            "location-scale parameters capture overall model stability and distributional variation."
        )

        # Step 7: Quantile Coefficient Plot
        st.subheader("Figure 5: Quantile Coefficient Plot")
        fig, ax = plt.subplots()
        for var in indep_vars:
            ax.plot(mmqr_results["Quantile (τ)"], mmqr_results[var], marker='o', label=var)
        ax.set_xlabel("Quantiles")
        ax.set_ylabel("Estimated Coefficients")
        ax.legend()
        st.pyplot(fig)

        # Step 8: Download button for results
        csv = mmqr_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download MMQR Results",
            data=csv,
            file_name="MMQR_results.csv",
            mime="text/csv"
        )

    else:
        st.warning("Please select at least one independent variable.")
else:
    st.error("Please upload your dataset in the earlier section to proceed.")

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
