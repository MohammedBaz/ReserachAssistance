import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Hypothesis Tester")

# --- Hardcoded Data Generation Parameters ---
N_POINTS = 150  # Fixed number of data points
# Series 1: AR(1) Process Parameters
AR_PHI = 0.75
AR_C = 0.5
AR_NOISE_STD = 1.0
# Series 2: White Noise Parameters
WN_MEAN = 0.0
WN_STD = 1.2

# --- Helper Functions for Data Generation (using predefined parameters) ---
def generate_predefined_ar_process():
    """Generates Series 1: An AR(1) process with fixed parameters."""
    y = np.zeros(N_POINTS)
    # Start near unconditional mean if stable, otherwise use constant
    if abs(AR_PHI) < 1:
        y[0] = AR_C / (1 - AR_PHI)
    else:
        y[0] = AR_C
    for t in range(1, N_POINTS):
        y[t] = AR_C + AR_PHI * y[t-1] + np.random.normal(scale=AR_NOISE_STD)
    return pd.Series(y, name=f"Series 1: AR(1) [φ={AR_PHI}, c={AR_C}, σ_ε={AR_NOISE_STD}]")

def generate_predefined_white_noise():
    """Generates Series 2: A White Noise process with fixed parameters."""
    series = np.random.normal(WN_MEAN, WN_STD, N_POINTS)
    return pd.Series(series, name=f"Series 2: White Noise [μ={WN_MEAN}, σ={WN_STD}]")

# --- Plotting Function ---
def plot_series_data(series_list, title="Time Series Data"):
    """Plots one or two time series."""
    fig, ax = plt.subplots()
    for s_obj in series_list:
        if s_obj is not None: # Check if the series object exists
            s_obj.plot(ax=ax, label=s_obj.name, grid=True)
    ax.legend()
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Value")
    st.pyplot(fig)

# --- Initialize Session State ---
def init_session_state():
    """Initializes or re-initializes data in session state if not present."""
    if 'ts_data1' not in st.session_state or st.session_state.ts_data1 is None:
        st.session_state.ts_data1 = generate_predefined_ar_process()
    if 'ts_data2' not in st.session_state or st.session_state.ts_data2 is None:
        st.session_state.ts_data2 = generate_predefined_white_noise()
    if 'selected_test' not in st.session_state:
        st.session_state.selected_test = None
    if 'hypothesized_mean' not in st.session_state: # Default for t-test input
        st.session_state.hypothesized_mean = 0.0
    if 'ljung_box_lags' not in st.session_state: # Default for Ljung-Box lags
        st.session_state.ljung_box_lags = min(10, N_POINTS // 2 -1 if N_POINTS > 4 else 1)

init_session_state() # Ensure data is loaded initially and session variables are set

# --- Main Application ---
st.title("🔬 Hypothesis Tester")
st.markdown("""
This interface allows you to apply common hypothesis tests to datasets by selecting a test using the buttons below the plot.
""")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🔄 Regenerate All Data", use_container_width=True, help="Generate new random samples for Series 1 and Series 2"):
        st.session_state.ts_data1 = generate_predefined_ar_process()
        st.session_state.ts_data2 = generate_predefined_white_noise()
        st.session_state.selected_test = None # Reset selected test to avoid showing old results
        st.rerun() # Rerun the script to update plots and clear old test states

    st.header("⚙️ Test Settings")
    alpha = st.slider("Significance Level (α):", 0.01, 0.10, 0.05, 0.01,
                      help="The probability of rejecting the null hypothesis when it is true.")

# --- Display Data ---
st.header("📊 Generated Time Series Data")
if st.session_state.ts_data1 is not None and st.session_state.ts_data2 is not None:
    plot_series_data([st.session_state.ts_data1, st.session_state.ts_data2], title="Predefined Time Series Samples")
    with st.expander("View Raw Data Values"):
        # Ensure names are set for concat if somehow lost (shouldn't be)
        s1_named = st.session_state.ts_data1.rename(st.session_state.ts_data1.name if hasattr(st.session_state.ts_data1, 'name') and st.session_state.ts_data1.name else "Series 1")
        s2_named = st.session_state.ts_data2.rename(st.session_state.ts_data2.name if hasattr(st.session_state.ts_data2, 'name') and st.session_state.ts_data2.name else "Series 2")
        df_both = pd.concat([s1_named, s2_named], axis=1)
        st.dataframe(df_both)
else:
    st.warning("Data is not available. Please try regenerating or reloading the app.")


# --- Test Selection Buttons ---
st.header("🚀 Select a Test to Perform:")
st.markdown("_Single-series tests (those with 'S1') operate on Series 1. Correlation test uses both Series 1 & 2._")

col1, col2, col3 = st.columns(3) # Arrange buttons in columns for better layout
with col1:
    if st.button("Shapiro-Wilk (Normality - S1)", use_container_width=True, key="shapiro_wilk_btn"):
        st.session_state.selected_test = "Shapiro-Wilk Test (Normality)"
    if st.button("Ljung-Box (Autocorrelation - S1)", use_container_width=True, key="ljung_box_btn"):
        st.session_state.selected_test = "Ljung-Box Test (Autocorrelation)"
with col2:
    if st.button("ADF Test (Stationarity - S1)", use_container_width=True, key="adf_test_btn"):
        st.session_state.selected_test = "Augmented Dickey-Fuller Test (Stationarity)"
    if st.button("Pearson Correlation (S1 & S2)", use_container_width=True, key="pearson_corr_btn"):
        st.session_state.selected_test = "Pearson Correlation Test"
with col3:
    if st.button("One-Sample t-test (Mean - S1)", use_container_width=True, key="one_sample_ttest_btn"):
        st.session_state.selected_test = "One-Sample t-test (Mean)"
    # You can add more buttons here if expanding test suite

# --- Display Test Inputs (if any) and Results ---
if st.session_state.selected_test:
    st.subheader(f"🧪 Results for: {st.session_state.selected_test}")
    current_test_name = st.session_state.selected_test
    # Default target series for single-series tests is Series 1
    series1_data = st.session_state.ts_data1
    series2_data = st.session_state.ts_data2 # For two-series tests

    # --- Test Specific Inputs (appear only when test is selected) ---
    if current_test_name == "One-Sample t-test (Mean)":
        st.session_state.hypothesized_mean = st.number_input(
            "Hypothesized Mean (μ₀) for Series 1:",
            value=float(st.session_state.hypothesized_mean), # Ensure type for widget
            step=0.1, key="ttest_mean_input_field",
            help="The mean value to test against for Series 1."
        )
    elif current_test_name == "Ljung-Box Test (Autocorrelation)":
        # Determine a sensible max number of lags
        max_allowable_lags = len(series1_data.dropna()) // 2 - 1 if len(series1_data.dropna()) > 4 else 1
        max_lags_slider = min(20, max_allowable_lags) # Cap at 20 or data-dependent max
        if max_lags_slider < 1: max_lags_slider = 1 # Ensure at least 1 possible lag

        st.session_state.ljung_box_lags = st.slider(
            "Lags to test for Ljung-Box (Series 1):",
            min_value=1,
            max_value=max_lags_slider,
            value=min(st.session_state.ljung_box_lags, max_lags_slider), # Ensure current value is valid
            step=1, key="lb_lags_input_field",
            help="Number of lags to include in the Ljung-Box test for autocorrelation in Series 1."
        )

    st.markdown("---") # Visual separator before results

    # Check if data is available before proceeding with tests
    if series1_data is None or (current_test_name == "Pearson Correlation Test" and series2_data is None):
        st.warning("Required series data is not available. Please try regenerating data.")
    elif len(series1_data.dropna()) == 0 and current_test_name != "Pearson Correlation Test":
        st.warning("Series 1 contains no valid (non-NaN) data points.")
    elif current_test_name == "Pearson Correlation Test" and (len(series1_data.dropna()) == 0 or len(series2_data.dropna()) == 0):
        st.warning("One or both series for Pearson Correlation contain no valid (non-NaN) data points.")
    else:
        # Perform and Display Shapiro-Wilk Test
        if current_test_name == "Shapiro-Wilk Test (Normality)":
            st.markdown(f"**Testing for Normality on: `{series1_data.name}`**")
            st.markdown("""- $H_0$: The data is normally distributed.
            - $H_1$: The data is not normally distributed.""")
            if len(series1_data.dropna()) < 3:
                st.warning("Shapiro-Wilk test requires at least 3 data points.")
            else:
                stat, p_value = stats.shapiro(series1_data.dropna())
                st.write(f"Test Statistic: `{stat:.4f}` | P-value: `{p_value:.4f}`")
                if p_value > alpha: st.success(f"Conclusion (α={alpha}): Fail to reject $H_0$. The data appears to be normally distributed.")
                else: st.error(f"Conclusion (α={alpha}): Reject $H_0$. The data does not appear to be normally distributed.")

        # Perform and Display Augmented Dickey-Fuller Test
        elif current_test_name == "Augmented Dickey-Fuller Test (Stationarity)":
            st.markdown(f"**Testing for Stationarity (ADF Test) on: `{series1_data.name}`**")
            st.markdown("""- $H_0$: The series has a unit root (it is non-stationary).
            - $H_1$: The series does not have a unit root (it is stationary).""")
            try:
                adf_result = adfuller(series1_data.dropna())
                st.write(f"ADF Statistic: `{adf_result[0]:.4f}` | P-value: `{adf_result[1]:.4f}`")
                # Optional: Display critical values
                # st.write("Critical Values:")
                # for key_cv, value_cv in adf_result[4].items(): st.write(f"\t{key_cv}: {value_cv:.4f}")
                if adf_result[1] <= alpha: st.success(f"Conclusion (α={alpha}): Reject $H_0$. The series appears to be stationary.")
                else: st.error(f"Conclusion (α={alpha}): Fail to reject $H_0$. The series appears to be non-stationary.")
            except Exception as e: st.error(f"Error running ADF test: {e}. (This can occur with very short series or series with zero variance).")

        # Perform and Display Ljung-Box Test
        elif current_test_name == "Ljung-Box Test (Autocorrelation)":
            st.markdown(f"**Testing for Autocorrelation (Ljung-Box Test) on: `{series1_data.name}`**")
            st.markdown("""- $H_0$: The data are independently distributed (i.e., no serial autocorrelation up to the specified lag).
            - $H_1$: The data are not independently distributed; they exhibit serial autocorrelation.""")
            lags_to_use = st.session_state.ljung_box_lags
            if len(series1_data.dropna()) <= lags_to_use:
                 st.warning(f"Not enough data points ({len(series1_data.dropna())}) for Ljung-Box test with {lags_to_use} lags.")
            else:
                try:
                    # acorr_ljungbox returns a DataFrame
                    lb_df_results = acorr_ljungbox(series1_data.dropna(), lags=[lags_to_use], return_df=True)
                    lb_statistic = lb_df_results['lb_stat'].iloc[0]
                    lb_p_value = lb_df_results['lb_pvalue'].iloc[0]
                    st.write(f"Ljung-Box Statistic (at lag {lags_to_use}): `{lb_statistic:.4f}` | P-value: `{lb_p_value:.4f}`")
                    if lb_p_value > alpha: st.success(f"Conclusion (α={alpha}): Fail to reject $H_0$. No significant autocorrelation detected up to lag {lags_to_use}.")
                    else: st.error(f"Conclusion (α={alpha}): Reject $H_0$. Significant autocorrelation detected up to lag {lags_to_use}.")
                except Exception as e: st.error(f"Error running Ljung-Box test: {e}")

        # Perform and Display One-Sample t-test
        elif current_test_name == "One-Sample t-test (Mean)":
            st.markdown(f"**Testing the Mean (One-Sample t-test) of: `{series1_data.name}`**")
            st.markdown("""- $H_0$: The true mean of the sample is equal to the hypothesized mean ($\mu = \mu_0$).
            - $H_1$: The true mean of the sample is not equal to the hypothesized mean ($\mu \neq \mu_0$).
            *(This test assumes the data is approximately normally distributed. Consider checking with Shapiro-Wilk first.)*""")
            hypothesized_mean_value = st.session_state.hypothesized_mean
            if len(series1_data.dropna()) < 2: # t-test needs at least 2 points for variance calculation
                st.warning("One-sample t-test requires at least 2 data points.")
            else:
                t_stat, p_value_t = stats.ttest_1samp(series1_data.dropna(), hypothesized_mean_value)
                st.write(f"T-statistic: `{t_stat:.4f}` | P-value: `{p_value_t:.4f}`")
                if p_value_t > alpha: st.success(f"Conclusion (α={alpha}): Fail to reject $H_0$. The sample mean is not significantly different from {hypothesized_mean_value}.")
                else: st.error(f"Conclusion (α={alpha}): Reject $H_0$. The sample mean is significantly different from {hypothesized_mean_value}.")

        # Perform and Display Pearson Correlation Test
        elif current_test_name == "Pearson Correlation Test":
            s1_clean = series1_data.dropna()
            s2_clean = series2_data.dropna()
            st.markdown(f"**Testing for Pearson Correlation between `{s1_clean.name}` and `{s2_clean.name}`**")
            st.markdown("""- $H_0$: The true correlation coefficient between the two series is 0 (no linear correlation).
            - $H_1$: The true correlation coefficient is not 0 (there is a linear correlation).
            *(Assumptions: Both variables should be approximately normally distributed, and the relationship should be linear.)*""")
            if len(s1_clean) < 3 or len(s2_clean) < 3: # Pearson r needs at least 3 points for meaningful p-value
                st.warning("Pearson correlation test requires at least 3 non-NaN data points for each series.")
            else:
                # Align series to the minimum length in case they differ after dropping NaNs individually
                min_aligned_len = min(len(s1_clean), len(s2_clean))
                s1_aligned = s1_clean[:min_aligned_len]
                s2_aligned = s2_clean[:min_aligned_len]

                corr_coeff, p_value_corr = stats.pearsonr(s1_aligned, s2_aligned)
                st.write(f"Pearson Correlation Coefficient: `{corr_coeff:.4f}` | P-value: `{p_value_corr:.4f}`")
                if p_value_corr > alpha: st.success(f"Conclusion (α={alpha}): Fail to reject $H_0$. No significant linear correlation detected.")
                else: st.error(f"Conclusion (α={alpha}): Reject $H_0$. Significant linear correlation detected.")
else:
    st.info("Select a test from the buttons above to view its results.")

st.sidebar.markdown("---")
#st.sidebar.info("This app uses predefined data generation. Click 'Regenerate' for new samples.")
