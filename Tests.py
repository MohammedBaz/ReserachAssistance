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
    """Generates a random walk time series."""
    noise = np.random.randn(n_points)
    series = np.cumsum(noise) + np.arange(n_points) * drift
    return pd.Series(series, name="Random Walk")

def generate_ar_process(n_points=100, phi=0.8, c=0.1, noise_std=1.0):
    """Generates an AR(1) time series: y_t = c + phi*y_{t-1} + epsilon_t."""
    y = np.zeros(n_points)
    y[0] = c / (1 - phi) if phi != 1 else c # Start near unconditional mean if stable
    for t in range(1, n_points):
        y[t] = c + phi * y[t-1] + np.random.normal(scale=noise_std)
    return pd.Series(y, name=f"AR(1) Process (phi={phi})")

def generate_white_noise(n_points=100, mean=0, std=1):
    """Generates a white noise time series."""
    series = np.random.normal(mean, std, n_points)
    return pd.Series(series, name="White Noise")

# --- Plotting Function ---
def plot_series(series, title="Time Series Data"):
    """Plots a time series."""
    fig, ax = plt.subplots()
    series.plot(ax=ax, title=title, grid=True)
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Value")
    st.pyplot(fig)

# --- Main Application ---
st.title("🔬 Time Series Hypothesis Tester")
st.markdown("""
This interface allows you to generate sample time series data and perform common hypothesis tests.
Select data generation parameters and the test you wish to perform from the sidebar.
""")

# --- Sidebar for Controls ---
st.sidebar.header("⚙️ Controls")

st.sidebar.subheader("1. Data Generation")
n_points = st.sidebar.slider("Number of Data Points:", 50, 500, 100)

data_type = st.sidebar.selectbox(
    "Select Time Series Type:",
    ("Random Walk", "AR(1) Process", "White Noise", "Two White Noise Series (for correlation)")
)

# Parameters for Random Walk
rw_drift = 0.0
if data_type == "Random Walk":
    rw_drift = st.sidebar.slider("Drift for Random Walk:", -0.5, 0.5, 0.01, 0.01)

# Parameters for AR(1)
ar_phi = 0.8
ar_c = 0.1
ar_noise_std = 1.0
if data_type == "AR(1) Process":
    ar_phi = st.sidebar.slider("AR(1) Coefficient (phi):", -0.99, 0.99, 0.8, 0.01)
    ar_c = st.sidebar.slider("AR(1) Constant (c):", -2.0, 2.0, 0.1, 0.05)
    ar_noise_std = st.sidebar.slider("AR(1) Noise Std Dev:", 0.1, 5.0, 1.0, 0.1)

# Parameters for White Noise
wn_mean = 0.0
wn_std = 1.0
if data_type == "White Noise" or data_type == "Two White Noise Series (for correlation)":
    wn_mean = st.sidebar.slider("White Noise Mean:", -2.0, 2.0, 0.0, 0.1)
    wn_std = st.sidebar.slider("White Noise Std Dev:", 0.1, 5.0, 1.0, 0.1)


# Generate Data
ts_data1 = None
ts_data2 = None # For correlation

if data_type == "Random Walk":
    ts_data1 = generate_random_walk(n_points, rw_drift)
elif data_type == "AR(1) Process":
    ts_data1 = generate_ar_process(n_points, ar_phi, ar_c, ar_noise_std)
elif data_type == "White Noise":
    ts_data1 = generate_white_noise(n_points, wn_mean, wn_std)
elif data_type == "Two White Noise Series (for correlation)":
    ts_data1 = generate_white_noise(n_points, wn_mean, wn_std)
    ts_data1.name = "White Noise Series 1"
    # Slightly different parameters for the second series or just another realization
    ts_data2 = generate_white_noise(n_points, wn_mean + np.random.uniform(-0.5, 0.5), wn_std + np.random.uniform(-0.2, 0.2))
    # Ensure std is positive for ts_data2
    if ts_data2.std() <= 0: # Should not happen with positive wn_std but as a safeguard
        ts_data2 = generate_white_noise(n_points, wn_mean + 0.1, wn_std + 0.1)
    ts_data2.name = "White Noise Series 2"


st.sidebar.subheader("2. Hypothesis Test Selection")
available_tests = ["-- Select a Test --"]
if ts_data1 is not None and ts_data2 is None:
    available_tests.extend([
        "Shapiro-Wilk Test (Normality)",
        "Augmented Dickey-Fuller Test (Stationarity)",
        "Ljung-Box Test (Autocorrelation)",
        "One-Sample t-test (Mean)"
    ])
elif ts_data1 is not None and ts_data2 is not None:
    available_tests.extend([
        "Pearson Correlation Test (between Series 1 & 2)",
        "Shapiro-Wilk Test (Normality - Series 1)",
        "Augmented Dickey-Fuller Test (Stationarity - Series 1)",
        "Ljung-Box Test (Autocorrelation - Series 1)",
        "One-Sample t-test (Mean - Series 1)"
    ])


selected_test = st.sidebar.selectbox(
    "Choose a Test:",
    available_tests
)

# Alpha level
alpha = st.sidebar.slider("Significance Level (alpha):", 0.01, 0.10, 0.05, 0.01)

# --- Display Data and Test Results ---
st.header("📊 Generated Data")
if ts_data1 is not None:
    plot_series(ts_data1, title=ts_data1.name)
    if data_type != "Two White Noise Series (for correlation)": # Show dataframe for single series
        with st.expander("View Data Values (Series 1)"):
            st.dataframe(ts_data1.head())
if ts_data2 is not None:
    plot_series(ts_data2, title=ts_data2.name)
    with st.expander("View Data Values (Series 1 & 2)"):
        df_both = pd.concat([ts_data1, ts_data2], axis=1)
        st.dataframe(df_both.head())


st.header("🧪 Test Results")

if selected_test != "-- Select a Test --" and ts_data1 is not None:
    st.subheader(f"Results for: {selected_test}")

    # --- Shapiro-Wilk Test ---
    if selected_test == "Shapiro-Wilk Test (Normality)" or selected_test == "Shapiro-Wilk Test (Normality - Series 1)":
        target_series = ts_data1
        st.markdown(f"**Testing for Normality on: `{target_series.name}`**")
        st.markdown("""
        The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.
        - $H_0$: The data is normally distributed.
        - $H_1$: The data is not normally distributed.
        """)
        if len(target_series) < 3:
            st.warning("Shapiro-Wilk test requires at least 3 data points.")
        else:
            stat, p_value = stats.shapiro(target_series.dropna())
            st.write(f"Test Statistic: `{stat:.4f}`")
            st.write(f"P-value: `{p_value:.4f}`")
            if p_value > alpha:
                st.success(f"Conclusion: Fail to reject $H_0$. The data appears to be normally distributed (p > {alpha}).")
            else:
                st.error(f"Conclusion: Reject $H_0$. The data does not appear to be normally distributed (p <= {alpha}).")

    # --- Augmented Dickey-Fuller Test ---
    elif selected_test == "Augmented Dickey-Fuller Test (Stationarity)" or selected_test == "Augmented Dickey-Fuller Test (Stationarity - Series 1)":
        target_series = ts_data1
        st.markdown(f"**Testing for Stationarity (ADF Test) on: `{target_series.name}`**")
        st.markdown("""
        The Augmented Dickey-Fuller (ADF) test tests the null hypothesis that a unit root is present in a time series sample, implying non-stationarity.
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
            st.error(f"Error running ADF test: {e}")


    # --- Ljung-Box Test ---
    elif selected_test == "Ljung-Box Test (Autocorrelation)" or selected_test == "Ljung-Box Test (Autocorrelation - Series 1)":
        target_series = ts_data1
        st.markdown(f"**Testing for Autocorrelation (Ljung-Box Test) on: `{target_series.name}`**")
        st.markdown("""
        The Ljung-Box test examines whether there is significant autocorrelation in a time series for a fixed number of lags.
        - $H_0$: The data are independently distributed (i.e., the correlations in the population from which the sample is taken are 0).
        - $H_1$: The data are not independently distributed; they exhibit serial correlation.
        """)
        max_lags = min(10, len(target_series) // 2 -1) # Default sensible lags
        if max_lags < 1:
             st.warning("Not enough data points for Ljung-Box test after considering lags.")
        else:
            lags_to_test = st.slider("Lags to test for Ljung-Box:", 1, max_lags, min(5, max_lags), 1)
            try:
                # acorr_ljungbox returns a DataFrame with lb_stat and lb_pvalue
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

    # --- One-Sample t-test ---
    elif selected_test == "One-Sample t-test (Mean)" or selected_test == "One-Sample t-test (Mean - Series 1)":
        target_series = ts_data1
        st.markdown(f"**Testing the Mean (One-Sample t-test) of: `{target_series.name}`**")
        st.markdown("""
        The one-sample t-test determines whether the sample mean is statistically different from a known or hypothesized population mean.
        **Assumption:** The data should be approximately normally distributed. Consider checking with Shapiro-Wilk first.
        - $H_0$: The true mean of the sample is equal to the hypothesized mean ($\mu = \mu_0$).
        - $H_1$: The true mean of the sample is not equal to the hypothesized mean ($\mu \neq \mu_0$).
        """)
        hypothesized_mean = st.number_input("Hypothesized Mean ($\mu_0$):", value=0.0, step=0.1)

        if len(target_series) < 2:
            st.warning("One-sample t-test requires at least 2 data points.")
        else:
            stat, p_value = stats.ttest_1samp(target_series.dropna(), hypothesized_mean)
            st.write(f"T-statistic: `{stat:.4f}`")
            st.write(f"P-value: `{p_value:.4f}`")

            if p_value > alpha:
                st.success(f"Conclusion: Fail to reject $H_0$. The sample mean is not significantly different from {hypothesized_mean} (p > {alpha}).")
            else:
                st.error(f"Conclusion: Reject $H_0$. The sample mean is significantly different from {hypothesized_mean} (p <= {alpha}).")

    # --- Pearson Correlation Test ---
    elif selected_test == "Pearson Correlation Test (between Series 1 & 2)":
        if ts_data1 is not None and ts_data2 is not None:
            st.markdown(f"**Testing for Pearson Correlation between `{ts_data1.name}` and `{ts_data2.name}`**")
            st.markdown("""
            The Pearson correlation coefficient measures the linear correlation between two continuous variables.
            The test assesses whether this correlation is statistically significant.
            - $H_0$: The true correlation coefficient is 0 (no linear correlation).
            - $H_1$: The true correlation coefficient is not 0 (there is a linear correlation).
            **Assumptions:** Variables should be approximately normally distributed, and the relationship should be linear.
            """)
            if len(ts_data1) < 3 or len(ts_data2) < 3:
                st.warning("Pearson correlation test requires at least 3 data points for each series.")
            else:
                # Ensure equal length if somehow they differ (shouldn't with current generation)
                min_len = min(len(ts_data1), len(ts_data2))
                data1_clean = ts_data1.dropna()[:min_len]
                data2_clean = ts_data2.dropna()[:min_len]

                if len(data1_clean) <3 or len(data2_clean) < 3:
                     st.warning("Not enough non-NaN data points after cleaning for Pearson correlation.")
                else:
                    corr_coeff, p_value = stats.pearsonr(data1_clean, data2_clean)
                    st.write(f"Pearson Correlation Coefficient: `{corr_coeff:.4f}`")
                    st.write(f"P-value: `{p_value:.4f}`")

                    if p_value > alpha:
                        st.success(f"Conclusion: Fail to reject $H_0$. No significant linear correlation detected (p > {alpha}).")
                    else:
                        st.error(f"Conclusion: Reject $H_0$. Significant linear correlation detected (p <= {alpha}).")
        else:
            st.warning("Please generate two series to perform the correlation test.")

elif selected_test == "-- Select a Test --":
    st.info("Please select a hypothesis test from the sidebar.")
else:
    st.info("Please generate data first using the controls in the sidebar.")

st.sidebar.markdown("---")
st.sidebar.info("Developed by AI for educational purposes.")
