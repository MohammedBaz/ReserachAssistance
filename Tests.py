import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Time Series Hypothesis Tester")

# --- Hardcoded Data Generation Parameters ---
N_POINTS = 150
# Series 1: AR(1) Process
AR_PHI = 0.75
AR_C = 0.5
AR_NOISE_STD = 1.0
# Series 2: White Noise
WN_MEAN = 0.0
WN_STD = 1.2

# --- Helper Functions for Data Generation (using predefined parameters) ---
def generate_predefined_ar_process():
    y = np.zeros(N_POINTS)
    if abs(AR_PHI) < 1:
        y[0] = AR_C / (1 - AR_PHI)
    else:
        y[0] = AR_C
    for t in range(1, N_POINTS):
        y[t] = AR_C + AR_PHI * y[t-1] + np.random.normal(scale=AR_NOISE_STD)
    return pd.Series(y, name=f"Series 1: AR(1) [φ={AR_PHI}]")

def generate_predefined_white_noise():
    series = np.random.normal(WN_MEAN, WN_STD, N_POINTS)
    return pd.Series(series, name=f"Series 2: White Noise [μ={WN_MEAN}, σ={WN_STD}]")

# --- Plotting Function ---
def plot_series_data(series_list, title="Time Series Data"):
    fig, ax = plt.subplots()
    for s in series_list:
        if s is not None:
            s.plot(ax=ax, label=s.name, grid=True)
    ax.legend()
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Value")
    st.pyplot(fig)

# --- Initialize Session State ---
def init_session_state():
    if 'ts_data1' not in st.session_state or st.session_state.ts_data1 is None:
        st.session_state.ts_data1 = generate_predefined_ar_process()
    if 'ts_data2' not in st.session_state or st.session_state.ts_data2 is None:
        st.session_state.ts_data2 = generate_predefined_white_noise()
    if 'selected_test' not in st.session_state:
        st.session_state.selected_test = None
    if 'hypothesized_mean' not in st.session_state:
        st.session_state.hypothesized_mean = 0.0
    if 'ljung_box_lags' not in st.session_state:
        st.session_state.ljung_box_lags = min(10, N_POINTS // 2 -1) # Sensible default

init_session_state() # Ensure data is loaded initially

# --- Main Application ---
st.title("🔬 Time Series Hypothesis Tester")
st.markdown("""
This interface helps you test hypotheses on pre-generated sample time series data.
The data is generated automatically. Use the sidebar to regenerate new samples or adjust the significance level.
Select a test using the buttons below the plot.
""")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🔄 Regenerate All Data", use_container_width=True):
        st.session_state.ts_data1 = generate_predefined_ar_process()
        st.session_state.ts_data2 = generate_predefined_white_noise()
        st.session_state.selected_test = None # Reset selected test
        st.rerun()

    st.header("⚙️ Test Controls")
    alpha = st.slider("Significance Level (alpha):", 0.01, 0.10, 0.05, 0.01)

# --- Display Data ---
st.header("📊 Generated Time Series Data")
if st.session_state.ts_data1 is not None and st.session_state.ts_data2 is not None:
    plot_series_data([st.session_state.ts_data1, st.session_state.ts_data2], title="Predefined Time Series")
    with st.expander("View Data Values"):
        df_both = pd.concat([st.session_state.ts_data1, st.session_state.ts_data2], axis=1)
        st.dataframe(df_both)
else:
    st.warning("Data could not be generated. Please try regenerating.")


# --- Test Selection Buttons ---
st.header("🚀 Select a Test to Perform:")
st.markdown("_Single-series tests operate on Series 1._")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Shapiro-Wilk (Normality - S1)", use_container_width=True, key="shapiro"):
        st.session_state.selected_test = "Shapiro-Wilk Test (Normality)"
    if st.button("Ljung-Box (Autocorrelation - S1)", use_container_width=True, key="ljungbox"):
        st.session_state.selected_test = "Ljung-Box Test (Autocorrelation)"
with col2:
    if st.button("ADF Test (Stationarity - S1)", use_container_width=True, key="adf"):
        st.session_state.selected_test = "Augmented Dickey-Fuller Test (Stationarity)"
    if st.button("Pearson Correlation (S1 & S2)", use_container_width=True, key="pearson"):
        st.session_state.selected_test = "Pearson Correlation Test"
with col3:
    if st.button("One-Sample t-test (Mean - S1)", use_container_width=True, key="ttest"):
        st.session_state.selected_test = "One-Sample t-test (Mean)"

# --- Display Test Inputs and Results ---
if st.session_state.selected_test:
    st.subheader(f"🧪 Results for: {st.session_state.selected_test}")
    current_test = st.session_state.selected_test
    # Default target series for single-series tests
    target_series_s1 = st.session_state.ts_data1

    # --- Test Specific Inputs ---
    if current_test == "One-Sample t-test (Mean)":
        st.session_state.hypothesized_mean = st.number_input(
            "Hypothesized Mean ($\mu_0$) for Series 1:",
            value=float(st.session_state.hypothesized_mean), # Ensure it's float for number_input
            step=0.1, key="ttest_mean_input"
        )
    elif current_test == "Ljung-Box Test (Autocorrelation)":
        max_lags = min(20, len(target_series_s1) // 2 - 1) if len(target_series_s1) > 4 else 1
        if max_lags < 1: max_lags = 1
        st.session_state.ljung_box_lags = st.slider(
            "Lags to test for Ljung-Box (Series 1):", 1, max_lags,
            min(st.session_state.ljung_box_lags, max_lags), 1, key="lb_lags_input"
        )

    st.markdown("---")

    if target_series_s1 is None: # Should not happen with current setup but good check
        st.warning("Series 1 data is not available. Please try regenerating data.")
    elif len(target_series_s1.dropna()) == 0 and current_test != "Pearson Correlation Test":
        st.warning("Series 1 contains no valid data points.")
    else:
        # Shapiro-Wilk Test
        if current_test == "Shapiro-Wilk Test (Normality)":
            st.markdown(f"**Testing for Normality on: `{target_series_s1.name}`**")
            st.markdown("""- $H_0$: The data is normally distributed.
            - $H_1$: The data is not normally distributed.""")
            if len(target_series_s1.dropna()) < 3:
                st.warning("Shapiro-Wilk test requires at least 3 data points.")
            else:
                stat, p_value = stats.shapiro(target_series_s1.dropna())
                st.write(f"Test Statistic: `{stat:.4f}` | P-value: `{p_value:.4f}`")
                if p_value > alpha: st.success(f"Conclusion: Fail to reject $H_0$ (p > {alpha}). Data appears normal.")
                else: st.error(f"Conclusion: Reject $H_0$ (p <= {alpha}). Data does not appear normal.")

        # Augmented Dickey-Fuller Test
        elif current_test == "Augmented Dickey-Fuller Test (Stationarity)":
            st.markdown(f"**Testing for Stationarity (ADF Test) on: `{target_series_s1.name}`**")
            st.markdown("""- $H_0$: The series has a unit root (non-stationary).
            - $H_1$: The series does not have a unit root (stationary).""")
            try:
                result = adfuller(target_series_s1.dropna())
                st.write(f"ADF Statistic: `{result[0]:.4f}` | P-value: `{result[1]:.4f}`")
                # st.write("Critical Values:")
                # for key_cv, value_cv in result[4].items(): st.write(f"\t{key_cv}: {value_cv:.4f}")
                if result[1] <= alpha: st.success(f"Conclusion: Reject $H_0$ (p <= {alpha}). Series appears stationary.")
                else: st.error(f"Conclusion: Fail to reject $H_0$ (p > {alpha}). Series appears non-stationary.")
            except Exception as e: st.error(f"Error running ADF test: {e}.")

        # Ljung-Box Test
        elif current_test == "Ljung-Box Test (Autocorrelation)":
            st.markdown(f"**Testing for Autocorrelation (Ljung-Box Test) on: `{target_series_s1.name}`**")
            st.markdown("""- $H_0$: Data are independently distributed (no serial autocorrelation).
            - $H_1$: Data exhibit serial autocorrelation.""")
            lags_to_test = st.session_state.ljung_box_lags
            if len(target_series_s1.dropna()) <= lags_to_test:
                 st.warning(f"Not enough data points for Ljung-Box test with {lags_to_test} lags.")
            else:
                try:
                    lb_results = acorr_ljungbox(target_series_s1.dropna(), lags=[lags_to_test], return_df=True)
                    lb_stat = lb_results['lb_stat'].iloc[0]
                    p_value = lb_results['lb_pvalue'].iloc[0]
                    st.write(f"Ljung-Box Statistic (lag {lags_to_test}): `{lb_stat:.4f}` | P-value: `{p_value:.4f}`")
                    if p_value > alpha: st.success(f"Conclusion: Fail to reject $H_0$ (p > {alpha}). No significant autocorrelation.")
                    else: st.error(f"Conclusion: Reject $H_0$ (p <= {alpha}). Significant autocorrelation detected.")
                except Exception as e: st.error(f"Error running Ljung-Box test: {e}")

        # One-Sample t-test
        elif current_test == "One-Sample t-test (Mean)":
            st.markdown(f"**Testing the Mean (One-Sample t-test) of: `{target_series_s1.name}`**")
            st.markdown("""- $H_0$: The true mean is equal to $\mu_0$.
            - $H_1$: The true mean is not equal to $\mu_0$.
            *(Assumption: Data approx. normally distributed)*""")
            hyp_mean = st.session_state.hypothesized_mean
            if len(target_series_s1.dropna()) < 2:
                st.warning("One-sample t-test requires at least 2 data points.")
            else:
                stat, p_value = stats.ttest_1samp(target_series_s1.dropna(), hyp_mean)
                st.write(f"T-statistic: `{stat:.4f}` | P-value: `{p_value:.4f}`")
                if p_value > alpha: st.success(f"Conclusion: Fail to reject $H_0$ (p > {alpha}). Mean not significantly different from {hyp_mean}.")
                else: st.error(f"Conclusion: Reject $H_0$ (p <= {alpha}). Mean significantly different from {hyp_mean}.")

        # Pearson Correlation Test
        elif current_test == "Pearson Correlation Test":
            s1 = st.session_state.ts_data1.dropna()
            s2 = st.session_state.ts_data2.dropna()
            if s1 is None or s2 is None:
                st.warning("Both Series 1 and Series 2 must be available for Pearson correlation.")
            else:
                st.markdown(f"**Testing for Pearson Correlation between `{s1.name}` and `{s2.name}`**")
                st.markdown("""- $H_0$: True correlation coefficient is 0.
                - $H_1$: True correlation coefficient is not 0.
                *(Assumptions: Variables approx. normal, linear relationship)*""")
                if len(s1) < 3 or len(s2) < 3:
                    st.warning("Pearson correlation requires at least 3 data points for each series after dropping NaNs.")
                else:
                    min_len = min(len(s1), len(s2))
                    s1_aligned, s2_aligned = s1[:min_len], s2[:min_len]
                    corr_coeff, p_value = stats.pearsonr(s1_aligned, s2_aligned)
                    st.write(f"Pearson Correlation Coefficient: `{corr_coeff:.4f}` | P-value: `{p_value:.4f}`")
                    if p_value > alpha: st.success(f"Conclusion: Fail to reject $H_0$ (p > {alpha}). No significant linear correlation.")
                    else: st.error(f"Conclusion: Reject $H_0$ (p <= {alpha}). Significant linear correlation detected.")
else:
    st.info("Select a test from the buttons above to see results.")

st.sidebar.markdown("---")
st.sidebar.info("Data generation is now predefined. Regenerate for new samples.")
