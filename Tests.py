import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Time Series Hypothesis Tester")

# --- Helper Functions for Data Generation ---
def generate_random_walk(n_points=100, drift=0.01):
    noise = np.random.randn(n_points)
    series = np.cumsum(noise) + np.arange(n_points) * drift
    return pd.Series(series, name="Random Walk")

def generate_ar_process(n_points=100, phi=0.8, c=0.1, noise_std=1.0):
    y = np.zeros(n_points)
    if abs(phi) < 1:
        y[0] = c / (1 - phi)
    else: # For non-stationary phi=1 or phi=-1
        y[0] = c
    for t in range(1, n_points):
        y[t] = c + phi * y[t-1] + np.random.normal(scale=noise_std)
    return pd.Series(y, name=f"AR(1) Process (phi={phi})")

def generate_white_noise(n_points=100, mean=0, std=1):
    series = np.random.normal(mean, std, n_points)
    return pd.Series(series, name="White Noise")

# --- Plotting Function ---
def plot_series(series_data, title="Time Series Data"):
    fig, ax = plt.subplots()
    if isinstance(series_data, list): # For plotting two series
        for i, s in enumerate(series_data):
            if s is not None:
                s.plot(ax=ax, label=s.name if hasattr(s, 'name') else f"Series {i+1}", grid=True)
        ax.legend()
    elif series_data is not None: # For plotting one series
        series_data.plot(ax=ax, title=title, grid=True)
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Value")
    st.pyplot(fig)

# --- Initialize Session State ---
if 'ts_data1' not in st.session_state:
    st.session_state.ts_data1 = generate_white_noise(100) # Default data on load
if 'ts_data2' not in st.session_state:
    st.session_state.ts_data2 = None # No second series by default
if 'selected_test' not in st.session_state:
    st.session_state.selected_test = None
if 'hypothesized_mean' not in st.session_state:
    st.session_state.hypothesized_mean = 0.0
if 'ljung_box_lags' not in st.session_state:
    st.session_state.ljung_box_lags = 5

# --- Main Application ---
st.title("🔬 Time Series Hypothesis Tester")
st.markdown("""
This interface helps you test hypotheses on sample time series data.
Configure data generation in the sidebar, then select a test using the buttons below the plot.
""")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Data Controls")

    n_points = st.slider("Number of Data Points:", 50, 500,
                         len(st.session_state.ts_data1) if st.session_state.ts_data1 is not None else 100)

    data_type_options = ["White Noise", "Random Walk", "AR(1) Process", "Two White Noise Series (for correlation)"]
    # Try to guess current data type for default selection
    current_data_name = st.session_state.ts_data1.name if st.session_state.ts_data1 is not None else "White Noise"
    default_data_type_index = 0
    for i, opt in enumerate(data_type_options):
        if opt.startswith(current_data_name.split(' (')[0]): # Match "AR(1) Process" from "AR(1) Process (phi=0.8)"
            default_data_type_index = i
            break
    if st.session_state.ts_data2 is not None and "White Noise" in current_data_name:
        default_data_type_index = data_type_options.index("Two White Noise Series (for correlation)")


    data_type = st.selectbox(
        "Select Time Series Type:",
        options=data_type_options,
        index=default_data_type_index
    )

    # Parameters based on data type
    rw_drift = 0.01
    ar_phi = 0.8
    ar_c = 0.1
    ar_noise_std = 1.0
    wn_mean = 0.0
    wn_std = 1.0

    if data_type == "Random Walk":
        rw_drift = st.slider("Drift for Random Walk:", -0.5, 0.5, 0.01, 0.01)
    elif data_type == "AR(1) Process":
        ar_phi = st.slider("AR(1) Coefficient (phi):", -1.5, 1.5, 0.8, 0.01) # Allow unstable
        ar_c = st.slider("AR(1) Constant (c):", -2.0, 2.0, 0.1, 0.05)
        ar_noise_std = st.slider("AR(1) Noise Std Dev:", 0.1, 5.0, 1.0, 0.1)
    elif "White Noise" in data_type: # Covers single and two series
        wn_mean = st.slider("White Noise Mean:", -2.0, 2.0, 0.0, 0.1)
        wn_std = st.slider("White Noise Std Dev:", 0.1, 5.0, 1.0, 0.1)

    if st.button("🔄 Regenerate Data", use_container_width=True):
        if data_type == "Random Walk":
            st.session_state.ts_data1 = generate_random_walk(n_points, rw_drift)
            st.session_state.ts_data2 = None
        elif data_type == "AR(1) Process":
            st.session_state.ts_data1 = generate_ar_process(n_points, ar_phi, ar_c, ar_noise_std)
            st.session_state.ts_data2 = None
        elif data_type == "White Noise":
            st.session_state.ts_data1 = generate_white_noise(n_points, wn_mean, wn_std)
            st.session_state.ts_data2 = None
        elif data_type == "Two White Noise Series (for correlation)":
            st.session_state.ts_data1 = generate_white_noise(n_points, wn_mean, wn_std)
            st.session_state.ts_data1.name = "White Noise Series 1"
            # Slightly different parameters for the second series
            mean2 = wn_mean + np.random.uniform(-0.5, 0.5) if wn_mean !=0 else np.random.uniform(-0.5,0.5)
            std2 = max(0.1, wn_std + np.random.uniform(-0.2, 0.2)) # Ensure std is positive
            st.session_state.ts_data2 = generate_white_noise(n_points, mean2, std2)
            st.session_state.ts_data2.name = "White Noise Series 2"
        st.session_state.selected_test = None # Reset selected test on data regeneration
        st.rerun()

    st.header("⚙️ Test Controls")
    alpha = st.slider("Significance Level (alpha):", 0.01, 0.10, 0.05, 0.01)

# --- Display Data ---
st.header("📊 Generated Data")
if st.session_state.ts_data1 is not None:
    if st.session_state.ts_data2 is not None:
        plot_series([st.session_state.ts_data1, st.session_state.ts_data2], title="Generated Time Series")
        with st.expander("View Data Values"):
            df_both = pd.concat([st.session_state.ts_data1, st.session_state.ts_data2], axis=1)
            st.dataframe(df_both)
    else:
        plot_series(st.session_state.ts_data1, title=st.session_state.ts_data1.name)
        with st.expander("View Data Values"):
            st.dataframe(st.session_state.ts_data1)

# --- Test Selection Buttons ---
st.header("🚀 Select a Test to Perform:")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Shapiro-Wilk (Normality)", use_container_width=True):
        st.session_state.selected_test = "Shapiro-Wilk Test (Normality)"
    if st.button("Ljung-Box (Autocorrelation)", use_container_width=True):
        st.session_state.selected_test = "Ljung-Box Test (Autocorrelation)"
with col2:
    if st.button("ADF Test (Stationarity)", use_container_width=True):
        st.session_state.selected_test = "Augmented Dickey-Fuller Test (Stationarity)"
    if st.session_state.ts_data2 is not None: # Only show if two series exist
        if st.button("Pearson Correlation", use_container_width=True):
            st.session_state.selected_test = "Pearson Correlation Test"
    else: # Placeholder if no second series to keep layout consistent or provide info
        st.markdown("<div style='height: 38px; display: flex; align-items: center; justify-content: center; border: 1px dashed gray; border-radius: 5px; opacity: 0.5;'>Pearson Correlation N/A</div>", unsafe_allow_html=True)


with col3:
    if st.button("One-Sample t-test (Mean)", use_container_width=True):
        st.session_state.selected_test = "One-Sample t-test (Mean)"
    # Add more buttons here if needed, or adjust columns

# --- Display Test Inputs and Results ---
if st.session_state.selected_test:
    st.subheader(f"🧪 Results for: {st.session_state.selected_test}")
    current_test = st.session_state.selected_test
    target_series = st.session_state.ts_data1 # Default target series

    # --- Test Specific Inputs ---
    if current_test == "One-Sample t-test (Mean)":
        st.session_state.hypothesized_mean = st.number_input(
            "Hypothesized Mean ($\mu_0$):",
            value=st.session_state.hypothesized_mean,
            step=0.1, key="ttest_mean_input"
        )
    elif current_test == "Ljung-Box Test (Autocorrelation)":
        max_lags = min(20, len(target_series) // 2 - 1) if len(target_series) > 4 else 1
        if max_lags < 1: max_lags = 1 # Ensure at least 1 lag can be chosen
        st.session_state.ljung_box_lags = st.slider(
            "Lags to test for Ljung-Box:", 1, max_lags,
            min(st.session_state.ljung_box_lags, max_lags), 1, key="lb_lags_input"
        )

    # --- Perform and Display Test ---
    st.markdown("---") # Visual separator

    if target_series is None:
        st.warning("Please generate data first.")
    elif len(target_series.dropna()) == 0:
        st.warning("The selected series contains no valid data points.")
    else:
        # Shapiro-Wilk Test
        if current_test == "Shapiro-Wilk Test (Normality)":
            st.markdown(f"**Testing for Normality on: `{target_series.name}`**")
            st.markdown("""
            - $H_0$: The data is normally distributed.
            - $H_1$: The data is not normally distributed.
            """)
            if len(target_series.dropna()) < 3:
                st.warning("Shapiro-Wilk test requires at least 3 data points.")
            else:
                stat, p_value = stats.shapiro(target_series.dropna())
                st.write(f"Test Statistic: `{stat:.4f}`")
                st.write(f"P-value: `{p_value:.4f}`")
                if p_value > alpha:
                    st.success(f"Conclusion: Fail to reject $H_0$. The data appears to be normally distributed (p > {alpha}).")
                else:
                    st.error(f"Conclusion: Reject $H_0$. The data does not appear to be normally distributed (p <= {alpha}).")

        # Augmented Dickey-Fuller Test
        elif current_test == "Augmented Dickey-Fuller Test (Stationarity)":
            st.markdown(f"**Testing for Stationarity (ADF Test) on: `{target_series.name}`**")
            st.markdown("""
            - $H_0$: The series has a unit root (it is non-stationary).
            - $H_1$: The series does not have a unit root (it is stationary).
            """)
            try:
                result = adfuller(target_series.dropna())
                st.write(f"ADF Statistic: `{result[0]:.4f}`")
                st.write(f"P-value: `{result[1]:.4f}`")
                st.write("Critical Values:")
                for key, value in result[4].items():
                    st.write(f"\t{key}: {value:.4f}")
                if result[1] <= alpha:
                    st.success(f"Conclusion: Reject $H_0$. The series appears to be stationary (p <= {alpha}).")
                else:
                    st.error(f"Conclusion: Fail to reject $H_0$. The series appears to be non-stationary (p > {alpha}).")
            except Exception as e:
                st.error(f"Error running ADF test: {e}. (Often occurs with very short series or series with no variance).")

        # Ljung-Box Test
        elif current_test == "Ljung-Box Test (Autocorrelation)":
            st.markdown(f"**Testing for Autocorrelation (Ljung-Box Test) on: `{target_series.name}`**")
            st.markdown("""
            - $H_0$: The data are independently distributed (no serial autocorrelation).
            - $H_1$: The data are not independently distributed (exhibit serial autocorrelation).
            """)
            lags_to_test = st.session_state.ljung_box_lags
            if len(target_series.dropna()) <= lags_to_test:
                 st.warning(f"Not enough data points ({len(target_series.dropna())}) for Ljung-Box test with {lags_to_test} lags.")
            else:
                try:
                    lb_results = acorr_ljungbox(target_series.dropna(), lags=[lags_to_test], return_df=True)
                    lb_stat = lb_results['lb_stat'].iloc[0]
                    p_value = lb_results['lb_pvalue'].iloc[0]
                    st.write(f"Ljung-Box Statistic (at lag {lags_to_test}): `{lb_stat:.4f}`")
                    st.write(f"P-value: `{p_value:.4f}`")
                    if p_value > alpha:
                        st.success(f"Conclusion: Fail to reject $H_0$. No significant autocorrelation detected up to lag {lags_to_test} (p > {alpha}).")
                    else:
                        st.error(f"Conclusion: Reject $H_0$. Significant autocorrelation detected up to lag {lags_to_test} (p <= {alpha}).")
                except Exception as e:
                    st.error(f"Error running Ljung-Box test: {e}")

        # One-Sample t-test
        elif current_test == "One-Sample t-test (Mean)":
            st.markdown(f"**Testing the Mean (One-Sample t-test) of: `{target_series.name}`**")
            st.markdown("""
            - $H_0$: The true mean of the sample is equal to $\mu_0$.
            - $H_1$: The true mean of the sample is not equal to $\mu_0$.
            *(Assumption: Data should be approximately normally distributed)*
            """)
            hyp_mean = st.session_state.hypothesized_mean
            if len(target_series.dropna()) < 2:
                st.warning("One-sample t-test requires at least 2 data points.")
            else:
                stat, p_value = stats.ttest_1samp(target_series.dropna(), hyp_mean)
                st.write(f"T-statistic: `{stat:.4f}`")
                st.write(f"P-value: `{p_value:.4f}`")
                if p_value > alpha:
                    st.success(f"Conclusion: Fail to reject $H_0$. The sample mean is not significantly different from {hyp_mean} (p > {alpha}).")
                else:
                    st.error(f"Conclusion: Reject $H_0$. The sample mean is significantly different from {hyp_mean} (p <= {alpha}).")

        # Pearson Correlation Test
        elif current_test == "Pearson Correlation Test":
            if st.session_state.ts_data1 is not None and st.session_state.ts_data2 is not None:
                s1 = st.session_state.ts_data1.dropna()
                s2 = st.session_state.ts_data2.dropna()
                st.markdown(f"**Testing for Pearson Correlation between `{s1.name}` and `{s2.name}`**")
                st.markdown("""
                - $H_0$: The true correlation coefficient is 0 (no linear correlation).
                - $H_1$: The true correlation coefficient is not 0 (there is a linear correlation).
                *(Assumptions: Variables approx. normal, linear relationship)*
                """)
                if len(s1) < 3 or len(s2) < 3:
                    st.warning("Pearson correlation test requires at least 3 data points for each series after dropping NaNs.")
                else:
                    min_len = min(len(s1), len(s2)) # Ensure equal length for comparison
                    s1_aligned = s1[:min_len]
                    s2_aligned = s2[:min_len]
                    corr_coeff, p_value = stats.pearsonr(s1_aligned, s2_aligned)
                    st.write(f"Pearson Correlation Coefficient: `{corr_coeff:.4f}`")
                    st.write(f"P-value: `{p_value:.4f}`")
                    if p_value > alpha:
                        st.success(f"Conclusion: Fail to reject $H_0$. No significant linear correlation detected (p > {alpha}).")
                    else:
                        st.error(f"Conclusion: Reject $H_0$. Significant linear correlation detected (p <= {alpha}).")
            else:
                st.warning("Two series are required for the Pearson Correlation Test. Please select 'Two White Noise Series' and regenerate data.")
else:
    st.info("Select a test from the buttons above to see results.")


st.sidebar.markdown("---")
st.sidebar.info("App updated based on user feedback.")
