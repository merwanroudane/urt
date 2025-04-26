import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, coint
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.seasonal import seasonal_decompose
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats
import pmdarima as pm

# Set page config
st.set_page_config(
    page_title="Unit Root Tests Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 30px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2E7D32;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .section {
        font-size: 20px;
        font-weight: bold;
        color: #6A1B9A;
        margin-top: 25px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .formula {
        background-color: #F5F5F5;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        margin: 10px 0;
    }
    .highlight {
        color: #D32F2F;
        font-weight: bold;
    }
    .note {
        font-size: 14px;
        font-style: italic;
        color: #616161;
    }
    .card {
        background-color: #FAFAFA;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
def generate_ts(n=200, process_type="random_walk", ar_params=None, ma_params=None, d=0.4,
                deterministic=None, near_explosive=False, seasonal=False, s_period=4):
    """Generate different types of time series"""
    np.random.seed(42)

    if process_type == "random_walk":
        # y_t = y_{t-1} + ε_t
        errors = np.random.normal(0, 1, n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = y[t - 1] + errors[t]

    elif process_type == "trend_stationary":
        # y_t = α + βt + ε_t
        t = np.arange(n)
        errors = np.random.normal(0, 1, n)
        y = 0.5 + 0.02 * t + errors

    elif process_type == "arma":
        # ARMA process
        if ar_params is None:
            ar_params = [0.7]
        if ma_params is None:
            ma_params = [0.2]
        ar = np.r_[1, -np.array(ar_params)]
        ma = np.r_[1, np.array(ma_params)]
        y = arma_generate_sample(ar, ma, n)

    elif process_type == "fractional":
        # Fractionally integrated process
        errors = np.random.normal(0, 1, n + 100)
        y = np.zeros(n + 100)

        # Approximate fractional integration
        for t in range(1, n + 100):
            # Use finite approximation of fractional differencing
            weights = [1]
            for k in range(1, t):
                weight = weights[-1] * (d - k + 1) / k
                weights.append(weight)

            for j in range(min(t, 100)):
                y[t] += weights[j] * errors[t - j]

        y = y[-n:]  # Remove burn-in period

    elif process_type == "near_explosive":
        # Near explosive process y_t = 0.99 * y_{t-1} + ε_t
        errors = np.random.normal(0, 1, n)
        y = np.zeros(n)
        rho = 0.995 if near_explosive else 0.7
        for t in range(1, n):
            y[t] = rho * y[t - 1] + errors[t]

    elif process_type == "seasonal":
        # Seasonal process
        t = np.arange(n)
        period = s_period
        seasonal_comp = 2 * np.sin(2 * np.pi * t / period)

        # Add either a random walk or stationary component
        if seasonal:
            # Add random walk
            errors = np.random.normal(0, 1, n)
            rw = np.zeros(n)
            for t in range(1, n):
                rw[t] = rw[t - 1] + errors[t]
            y = seasonal_comp + rw
        else:
            # Add stationary noise
            errors = np.random.normal(0, 0.5, n)
            y = seasonal_comp + errors

    elif process_type == "multiple_roots":
        # Process with multiple unit roots: (1-L)(1-L)y_t = ε_t
        # This is I(2) process
        errors = np.random.normal(0, 1, n)
        y1 = np.zeros(n)  # I(1) process
        y = np.zeros(n)  # I(2) process

        for t in range(1, n):
            y1[t] = y1[t - 1] + errors[t]

        for t in range(1, n):
            y[t] = y[t - 1] + y1[t]

    # Add deterministic components if specified
    if deterministic:
        t = np.arange(n)
        if 'constant' in deterministic:
            y = y + 5
        if 'trend' in deterministic:
            y = y + 0.1 * t

    return y


def plot_ts(y, title="Time Series Plot"):
    """Plot time series data"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    return fig


def run_adfuller(y, max_lags=None, regression='c'):
    """Run Augmented Dickey-Fuller test and return results"""
    result = adfuller(y, maxlag=max_lags, regression=regression)

    output = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'Used Lags': result[2],
        'Number of Observations': result[3],
        'Reject H0': result[1] < 0.05
    }

    return output


def run_kpss(y, regression='c'):
    """Run KPSS test and return results"""
    result = kpss(y, regression=regression)

    output = {
        'KPSS Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[3],
        'Used Lags': result[2],
        'Reject H0': result[1] < 0.05
    }

    return output


def generate_power_size_data():
    """Generate data for power and size analysis"""
    # Parameters
    n_samples = 200
    n_replications = 100
    alphas = [0.01, 0.05, 0.10]

    # Different processes to test
    processes = [
        {"name": "Stationary AR(1)", "ar": 0.5, "has_unit_root": False},
        {"name": "Unit Root (RW)", "ar": 1.0, "has_unit_root": True},
        {"name": "Near Unit Root", "ar": 0.95, "has_unit_root": False},
        {"name": "Near Explosive", "ar": 0.99, "has_unit_root": False}
    ]

    results = []

    for process in processes:
        adf_rejects = {alpha: 0 for alpha in alphas}
        kpss_rejects = {alpha: 0 for alpha in alphas}

        ar_coef = process["ar"]

        for _ in range(n_replications):
            # Generate series
            errors = np.random.normal(0, 1, n_samples)
            y = np.zeros(n_samples)

            for t in range(1, n_samples):
                y[t] = ar_coef * y[t - 1] + errors[t]

            # Run tests
            adf_res = adfuller(y, regression='c')
            kpss_res = kpss(y, regression='c')

            # Check rejection at different alpha levels
            for alpha in alphas:
                if adf_res[1] < alpha:
                    adf_rejects[alpha] += 1

                # For KPSS, critical values need to be compared (smaller p-value means higher test statistic)
                if kpss_res[1] < alpha:
                    kpss_rejects[alpha] += 1

        # Calculate proportions
        for alpha in alphas:
            adf_power = adf_rejects[alpha] / n_replications
            kpss_power = kpss_rejects[alpha] / n_replications

            # For size and power
            if process["has_unit_root"]:
                # For unit root processes, ADF rejection = power, KPSS rejection = size
                adf_metric_name = "Power"
                kpss_metric_name = "Size"
            else:
                # For stationary processes, ADF rejection = size, KPSS rejection = power
                adf_metric_name = "Size"
                kpss_metric_name = "Power"

            results.append({
                "Process": process["name"],
                "AR Coefficient": ar_coef,
                "Alpha": alpha,
                "Test": "ADF",
                "Metric": adf_metric_name,
                "Value": adf_power
            })

            results.append({
                "Process": process["name"],
                "AR Coefficient": ar_coef,
                "Alpha": alpha,
                "Test": "KPSS",
                "Metric": kpss_metric_name,
                "Value": kpss_power
            })

    return pd.DataFrame(results)


def simulate_multiple_unit_roots():
    """Simulate data with multiple unit roots and test effectiveness of tests"""
    # Parameters
    n_samples = 200
    n_replications = 50

    # Processes with different integration orders
    processes = [
        {"name": "I(0)", "order": 0},
        {"name": "I(1)", "order": 1},
        {"name": "I(2)", "order": 2}
    ]

    results = []

    for process in processes:
        adf_level_rejects = 0
        adf_diff_rejects = 0
        kpss_level_rejects = 0
        kpss_diff_rejects = 0

        for _ in range(n_replications):
            # Generate series based on integration order
            order = process["order"]
            errors = np.random.normal(0, 1, n_samples)

            if order == 0:
                # I(0) - stationary
                y = errors
            elif order == 1:
                # I(1) - random walk
                y = np.zeros(n_samples)
                for t in range(1, n_samples):
                    y[t] = y[t - 1] + errors[t]
            elif order == 2:
                # I(2) - double integration
                y1 = np.zeros(n_samples)  # I(1)
                y = np.zeros(n_samples)  # I(2)

                for t in range(1, n_samples):
                    y1[t] = y1[t - 1] + errors[t]

                for t in range(1, n_samples):
                    y[t] = y[t - 1] + y1[t]

            # Tests on levels
            adf_level = adfuller(y, regression='c')
            kpss_level = kpss(y, regression='c')

            # Tests on first difference (if possible)
            if len(y) > 1:
                diff_y = np.diff(y)
                adf_diff = adfuller(diff_y, regression='c')
                kpss_diff = kpss(diff_y, regression='c')
            else:
                adf_diff = (0, 1, 0, 0)  # Dummy values
                kpss_diff = (0, 1, 0, {})

            # Count rejections at 5% level
            if adf_level[1] < 0.05:
                adf_level_rejects += 1
            if adf_diff[1] < 0.05:
                adf_diff_rejects += 1
            if kpss_level[1] < 0.05:
                kpss_level_rejects += 1
            if kpss_diff[1] < 0.05:
                kpss_diff_rejects += 1

        # Calculate rejection rates
        results.append({
            "Process": process["name"],
            "Integration Order": order,
            "ADF Level Rejection Rate": adf_level_rejects / n_replications,
            "ADF Differenced Rejection Rate": adf_diff_rejects / n_replications,
            "KPSS Level Rejection Rate": kpss_level_rejects / n_replications,
            "KPSS Differenced Rejection Rate": kpss_diff_rejects / n_replications
        })

    return pd.DataFrame(results)


def simulate_fractional_integration():
    """Simulate data with different fractional integration parameters"""
    # Parameters
    n_samples = 300
    d_values = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

    results = []

    for d in d_values:
        # Generate series with fractional integration parameter d
        y = generate_ts(n=n_samples, process_type="fractional", d=d)

        # Run tests
        adf_res = adfuller(y, regression='c')
        kpss_res = kpss(y, regression='c')

        # Store results
        results.append({
            "d Parameter": d,
            "ADF Statistic": adf_res[0],
            "ADF p-value": adf_res[1],
            "KPSS Statistic": kpss_res[0],
            "KPSS p-value": kpss_res[1]
        })

    return pd.DataFrame(results)


def simulate_seasonality():
    """Simulate seasonal data and analyze test performance"""
    # Parameters
    n_samples = 200
    periods = [4, 12]  # Quarterly and monthly

    results = []

    for period in periods:
        # Generate stationary seasonal data
        y_stationary = generate_ts(n=n_samples, process_type="seasonal", seasonal=False, s_period=period)

        # Generate non-stationary seasonal data
        y_nonstationary = generate_ts(n=n_samples, process_type="seasonal", seasonal=True, s_period=period)

        # Run standard tests
        adf_stat = adfuller(y_stationary, regression='ct')
        adf_nonstat = adfuller(y_nonstationary, regression='ct')

        kpss_stat = kpss(y_stationary, regression='ct')
        kpss_nonstat = kpss(y_nonstationary, regression='ct')

        # Run tests on seasonally differenced data
        diff_y_stat = y_stationary[period:] - y_stationary[:-period]
        diff_y_nonstat = y_nonstationary[period:] - y_nonstationary[:-period]

        adf_diff_stat = adfuller(diff_y_stat, regression='c')
        adf_diff_nonstat = adfuller(diff_y_nonstat, regression='c')

        # Store results
        results.append({
            "Period": period,
            "Data Type": "Stationary Seasonal",
            "ADF Level p-value": adf_stat[1],
            "ADF Seasonal Diff p-value": adf_diff_stat[1],
            "KPSS Level p-value": kpss_stat[1]
        })

        results.append({
            "Period": period,
            "Data Type": "Non-stationary Seasonal",
            "ADF Level p-value": adf_nonstat[1],
            "ADF Seasonal Diff p-value": adf_diff_nonstat[1],
            "KPSS Level p-value": kpss_nonstat[1]
        })

    return pd.DataFrame(results)


# Main app structure
def main():
    st.markdown('<div class="main-header">Comprehensive Unit Root Tests Explorer</div>', unsafe_allow_html=True)

    st.markdown("""
    This interactive application provides a comprehensive exploration of unit root tests in time series analysis. 
    It covers theoretical foundations, different types of unit root tests, their power and size properties, 
    and special cases like fractional integration, near-explosive roots, multiple unit roots, and seasonal unit roots.
    """)

    # Sidebar navigation
    st.sidebar.image("https://i.ibb.co/mzjBN8s/time-series.png", width=100)
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        [
            "1. Introduction to Unit Roots",
            "2. Types of Unit Root Tests",
            "3. Interactive Simulator",
            "4. Size and Power Analysis",
            "5. Special Cases",
            "6. References and Resources"
        ]
    )

    # Pages
    if page == "1. Introduction to Unit Roots":
        introduction_page()
    elif page == "2. Types of Unit Root Tests":
        unit_root_tests_page()
    elif page == "3. Interactive Simulator":
        interactive_simulator_page()
    elif page == "4. Size and Power Analysis":
        size_power_page()
    elif page == "5. Special Cases":
        special_cases_page()
    elif page == "6. References and Resources":
        references_page()


def introduction_page():
    st.markdown('<div class="sub-header">Introduction to Unit Roots</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Unit root tests are statistical methods used to determine whether a time series is stationary or non-stationary.
    This distinction is crucial in time series analysis as many statistical procedures assume stationarity.
    </div>
    """, unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section">What is a Unit Root?</div>', unsafe_allow_html=True)
        st.markdown("""
        A unit root is a feature of a time series that makes it non-stationary. In an autoregressive model:

        <div class="formula">
        y<sub>t</sub> = ρy<sub>t-1</sub> + ε<sub>t</sub>
        </div>

        If ρ = 1, the process has a unit root and is non-stationary. This is a random walk:

        <div class="formula">
        y<sub>t</sub> = y<sub>t-1</sub> + ε<sub>t</sub>
        </div>

        The name "unit root" comes from the fact that ρ = 1 is a root of the characteristic equation.
        """, unsafe_allow_html=True)

    with col2:
        # Generate and plot examples
        st.markdown('<div class="section">Visual Examples</div>', unsafe_allow_html=True)

        # Generate stationary AR process
        ar_stationary = generate_ts(process_type="arma", ar_params=[0.7])

        # Generate random walk (unit root)
        random_walk = generate_ts(process_type="random_walk")

        # Plot
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(ar_stationary)
        ax[0].set_title('Stationary AR(1) Process: ρ = 0.7')
        ax[0].set_ylabel('Value')

        ax[1].plot(random_walk)
        ax[1].set_title('Non-stationary Random Walk: ρ = 1')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Value')

        plt.tight_layout()
        st.pyplot(fig)

    st.markdown('<div class="section">Why Unit Roots Matter</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Importance in Time Series Analysis

    Unit roots have profound implications for time series analysis:

    1. **Statistical Properties**:
       - Non-stationary series have time-dependent variance and mean
       - Standard statistical tests become invalid
       - Conventional forecasting methods fail

    2. **Spurious Regression Problem**:
       - Regressing unrelated non-stationary series can produce misleadingly high R² and t-statistics
       - False impression of meaningful relationships between variables

    3. **Economic Significance**:
       - Unit root processes have "infinite memory" - shocks persist indefinitely
       - Contradicts mean-reversion assumptions in many economic models
       - Central to debates about economic persistence (e.g., GDP, unemployment)

    4. **Model Selection**:
       - Determines appropriate modeling approach (ARIMA vs. ARMA)
       - Guides differencing requirements
       - Informs cointegration analysis
    """)

    st.markdown('<div class="section">Detecting Unit Roots</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Methods for Unit Root Detection

    Several statistical tests have been developed to detect unit roots:

    1. **Dickey-Fuller (DF) & Augmented Dickey-Fuller (ADF) Tests**:
       - Test the null hypothesis that a unit root is present
       - Most widely used tests for unit root detection

    2. **Phillips-Perron (PP) Test**:
       - Non-parametric adjustment to the DF test
       - More robust to heteroskedasticity and autocorrelation

    3. **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test**:
       - Tests the null hypothesis of stationarity
       - Complementary to ADF (reversed hypotheses)

    4. **Elliott-Rothenberg-Stock (ERS) Test**:
       - Improved power through GLS detrending
       - More powerful than standard ADF

    5. **Ng-Perron Test**:
       - Further refinements of the ERS approach
       - Better performance in finite samples
    """)

    # Interactive component
    st.markdown('<div class="section">Interactive Examination</div>', unsafe_allow_html=True)

    process_type = st.radio(
        "Select process type to visualize:",
        ["Stationary AR(1)", "Random Walk (Unit Root)", "Trend Stationary"]
    )

    n_samples = st.slider("Number of observations", 50, 500, 200)

    if process_type == "Stationary AR(1)":
        ar_coef = st.slider("AR coefficient", 0.1, 0.9, 0.7, 0.1)
        series = generate_ts(n=n_samples, process_type="arma", ar_params=[ar_coef])
        title = f"Stationary AR(1) Process: ρ = {ar_coef}"
    elif process_type == "Random Walk (Unit Root)":
        series = generate_ts(n=n_samples, process_type="random_walk")
        title = "Random Walk Process (with Unit Root)"
    else:  # Trend Stationary
        series = generate_ts(n=n_samples, process_type="trend_stationary")
        title = "Trend Stationary Process"

    # Plot
    fig = plot_ts(series, title=title)
    st.pyplot(fig)

    # Run tests
    if st.button("Run Unit Root Tests"):
        adf_result = run_adfuller(series)
        kpss_result = run_kpss(series)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Augmented Dickey-Fuller Test")
            st.write(f"ADF Statistic: {adf_result['ADF Statistic']:.4f}")
            st.write(f"p-value: {adf_result['p-value']:.4f}")
            st.write("Critical Values:")
            for key, value in adf_result['Critical Values'].items():
                st.write(f"  {key}: {value:.4f}")

            if adf_result['Reject H0']:
                st.success("✅ Reject null hypothesis of unit root (series is stationary)")
            else:
                st.error("❌ Fail to reject null hypothesis (series has a unit root)")

        with col2:
            st.markdown("#### KPSS Test")
            st.write(f"KPSS Statistic: {kpss_result['KPSS Statistic']:.4f}")
            st.write(f"p-value: {kpss_result['p-value']:.4f}")
            st.write("Critical Values:")
            for key, value in kpss_result['Critical Values'].items():
                st.write(f"  {key}: {value:.4f}")

            if kpss_result['Reject H0']:
                st.error("❌ Reject null hypothesis of stationarity (series is non-stationary)")
            else:
                st.success("✅ Fail to reject null hypothesis (series is stationary)")


def unit_root_tests_page():
    st.markdown('<div class="sub-header">Types of Unit Root Tests</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Various unit root tests have been developed, each with different approaches, assumptions, and properties.
    Understanding these differences is crucial for proper test selection and interpretation.
    </div>
    """, unsafe_allow_html=True)

    test_type = st.selectbox(
        "Select test type to explore:",
        ["Dickey-Fuller (DF) Test", "Augmented Dickey-Fuller (ADF) Test",
         "Phillips-Perron (PP) Test", "KPSS Test",
         "Elliott-Rothenberg-Stock (ERS) Test", "Ng-Perron Test"]
    )

    if test_type == "Dickey-Fuller (DF) Test":
        st.markdown("""
        ### Dickey-Fuller (DF) Test

        <div class="card">
        The Dickey-Fuller test is the foundational unit root test developed by David Dickey and Wayne Fuller in 1979.
        </div>

        #### Mathematical Framework

        The test is based on estimating the following equation:

        <div class="formula">
        Δy<sub>t</sub> = (ρ-1)y<sub>t-1</sub> + ε<sub>t</sub> = δy<sub>t-1</sub> + ε<sub>t</sub>
        </div>

        where δ = (ρ-1).

        The null and alternative hypotheses are:
        - H<sub>0</sub>: δ = 0 (Unit root present, series is non-stationary)
        - H<sub>1</sub>: δ < 0 (No unit root, series is stationary)

        #### Test Variants

        The DF test comes in three main specifications:

        1. **Random Walk**: Δy<sub>t</sub> = δy<sub>t-1</sub> + ε<sub>t</sub>
        2. **Random Walk with Drift**: Δy<sub>t</sub> = α + δy<sub>t-1</sub> + ε<sub>t</sub>
        3. **Random Walk with Drift and Trend**: Δy<sub>t</sub> = α + βt + δy<sub>t-1</sub> + ε<sub>t</sub>

        #### Critical Values

        The DF test statistic does not follow the standard t-distribution. Instead, it follows a non-standard distribution derived by Dickey and Fuller through simulation.

        #### Limitations

        - Assumes errors are independently and identically distributed (i.i.d.)
        - Cannot handle serial correlation in errors
        - Requires the augmented version (ADF) for most real-world applications
        """, unsafe_allow_html=True)

    elif test_type == "Augmented Dickey-Fuller (ADF) Test":
        st.markdown("""
        ### Augmented Dickey-Fuller (ADF) Test

        <div class="card">
        The ADF test extends the basic Dickey-Fuller test to handle serial correlation in the error terms by adding lagged difference terms.
        </div>

        #### Mathematical Framework

        The ADF test estimates the following equation:

        <div class="formula">
        Δy<sub>t</sub> = α + βt + δy<sub>t-1</sub> + Σ<sub>i=1</sub><sup>p</sup> γ<sub>i</sub>Δy<sub>t-i</sub> + ε<sub>t</sub>
        </div>

        where:
        - p is the lag order of the autoregressive process
        - α is a constant term (drift)
        - βt is a deterministic trend
        - δ = (ρ-1)

        The null and alternative hypotheses remain:
        - H<sub>0</sub>: δ = 0 (Unit root present)
        - H<sub>1</sub>: δ < 0 (No unit root)

        #### Lag Selection

        Proper lag selection is crucial for the ADF test. Too few lags won't capture the autocorrelation structure, while too many reduce test power. Methods include:

        - Information criteria (AIC, BIC, HQIC)
        - Sequential testing procedures
        - Fixed rule based on sample size

        #### Deterministic Components

        The choice of deterministic components affects critical values and test power:

        1. **None**: No constant or trend (rarely used)
        2. **Constant**: Includes intercept only (for series with non-zero mean)
        3. **Constant and Trend**: Includes both (for trending series)

        #### Practical Considerations

        - Most widely used unit root test in applied research
        - Available in all major statistical software
        - Low power against near-unit-root processes
        - Different specifications can lead to different conclusions
        """, unsafe_allow_html=True)

    elif test_type == "Phillips-Perron (PP) Test":
        st.markdown("""
        ### Phillips-Perron (PP) Test

        <div class="card">
        The Phillips-Perron test, developed by Peter Phillips and Pierre Perron in 1988, uses non-parametric statistical methods to handle serial correlation without adding lagged difference terms.
        </div>

        #### Mathematical Framework

        The PP test starts with the standard DF regression:

        <div class="formula">
        Δy<sub>t</sub> = α + βt + δy<sub>t-1</sub> + ε<sub>t</sub>
        </div>

        But instead of adding lags, it makes a non-parametric correction to the t-statistic to account for serial correlation:

        <div class="formula">
        Z<sub>t</sub> = t<sub>δ</sub> * (s / s<sub>L</sub>) - λ
        </div>

        where:
        - t<sub>δ</sub> is the t-statistic for δ
        - s is the standard error of the regression
        - s<sub>L</sub> is a consistent estimator of the long-run variance
        - λ is a correction factor

        #### Advantages

        - Robust to heteroskedasticity and autocorrelation
        - No need to specify lag length
        - Works well with general forms of dependence in the error terms

        #### Limitations

        - Poor size properties in small samples
        - Less powerful than ADF in many scenarios
        - Sensitivity to structural breaks

        #### Practical Considerations

        - Uses Newey-West type correction for serial correlation
        - Same null hypothesis as ADF: H<sub>0</sub>: δ = 0 (Unit root present)
        - Same deterministic component options as ADF
        """, unsafe_allow_html=True)

    elif test_type == "KPSS Test":
        st.markdown("""
        ### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test

        <div class="card">
        The KPSS test, developed in 1992, takes a different approach by testing the null hypothesis of stationarity against the alternative of a unit root.
        </div>

        #### Mathematical Framework

        The KPSS test is based on the model:

        <div class="formula">
        y<sub>t</sub> = β<sub>t</sub> + r<sub>t</sub> + ε<sub>t</sub>
        </div>
        <div class="formula">
        r<sub>t</sub> = r<sub>t-1</sub> + u<sub>t</sub>
        </div>

        where:
        - β<sub>t</sub> is a deterministic trend
        - r<sub>t</sub> is a random walk
        - ε<sub>t</sub> is a stationary error
        - u<sub>t</sub> is i.i.d.(0, σ<sub>u</sub><sup>2</sup>)

        The null hypothesis is:
        - H<sub>0</sub>: σ<sub>u</sub><sup>2</sup> = 0 (Series is stationary)
        - H<sub>1</sub>: σ<sub>u</sub><sup>2</sup> > 0 (Series has a unit root)

        The test statistic is based on the partial sums of residuals:

        <div class="formula">
        KPSS = (1/T<sup>2</sup>) * Σ S<sub>t</sub><sup>2</sup> / s<sup>2</sup>
        </div>

        where S<sub>t</sub> is the partial sum of residuals and s<sup>2</sup> is a consistent estimator of the long-run variance.

        #### Significance and Usage

        - The KPSS test complements the ADF test by reversing the null hypothesis
        - Combined with ADF, provides stronger evidence about stationarity
        - Four possible outcomes when using both tests:
          1. Reject neither test: data may be fractionally integrated
          2. Reject ADF but not KPSS: evidence of stationarity
          3. Reject KPSS but not ADF: evidence of unit root
          4. Reject both: possible misspecification or structural break

        #### Limitations

        - Sensitive to structural breaks (falsely indicates non-stationarity)
        - Performance affected by lag selection in variance estimator
        - Lower power than ADF for near-unit-root processes
        """, unsafe_allow_html=True)

    elif test_type == "Elliott-Rothenberg-Stock (ERS) Test":
        st.markdown("""
        ### Elliott-Rothenberg-Stock (ERS) Test

        <div class="card">
        The ERS test, also known as the DF-GLS test, was developed in 1996 to address power problems in traditional unit root tests.
        </div>

        #### Mathematical Framework

        The ERS test improves the ADF test through GLS (Generalized Least Squares) detrending:

        1. First, the data is detrended using GLS:

        <div class="formula">
        y<sub>t</sub><sup>d</sup> = y<sub>t</sub> - β̂<sub>0</sub> - β̂<sub>1</sub>t
        </div>

        where β̂<sub>0</sub> and β̂<sub>1</sub> are GLS estimates.

        2. Then an ADF-style test is run on the detrended data:

        <div class="formula">
        Δy<sub>t</sub><sup>d</sup> = δy<sub>t-1</sub><sup>d</sup> + Σ<sub>i=1</sub><sup>p</sup> γ<sub>i</sub>Δy<sub>t-i</sub><sup>d</sup> + ε<sub>t</sub>
        </div>

        The GLS detrending uses a local-to-unity framework where:

        <div class="formula">
        ρ = 1 + c/T
        </div>

        where c is set to -7 for the constant-only case and -13.5 for the constant-and-trend case.

        #### Advantages

        - Significantly more powerful than standard ADF tests
        - Nearly optimal in terms of asymptotic power
        - Better size properties in finite samples

        #### Practical Considerations

        - Two variants: Point Optimal test (P-test) and DF-GLS test
        - Similar implementation to ADF but with transformed data
        - Still requires lag length selection
        - Same null hypothesis as ADF: H<sub>0</sub>: unit root present
        """, unsafe_allow_html=True)

    elif test_type == "Ng-Perron Test":
        st.markdown("""
        ### Ng-Perron Test

        <div class="card">
        The Ng-Perron tests, developed in 2001, are a set of modified unit root tests designed to address size distortions and power problems in traditional tests.
        </div>

        #### Mathematical Framework

        Ng and Perron combined the GLS detrending approach of Elliott-Rothenberg-Stock with modified test statistics and a new information criterion for lag selection. They introduced four test statistics:

        1. **MZ<sub>α</sub>**: A modified version of the Phillips-Perron Z<sub>α</sub> test
        2. **MZ<sub>t</sub>**: A modified version of the Phillips-Perron Z<sub>t</sub> test
        3. **MSB**: Modified Sargan-Bhargava test
        4. **MPT**: Modified Point Optimal test

        These statistics are based on GLS detrended data. The key innovation is the Modified Akaike Information Criterion (MAIC) for lag selection:

        <div class="formula">
        MAIC(k) = ln(σ̂<sub>k</sub><sup>2</sup>) + 2(τ<sub>T</sub>(k) + k) / (T - k<sub>max</sub>)
        </div>

        where τ<sub>T</sub>(k) is a penalty function that addresses the distortions in standard information criteria.

        #### Advantages

        - Superior size and power properties compared to earlier tests
        - Less sensitive to lag selection errors
        - Robust to various error distributions
        - Performs well in small samples

        #### Limitations

        - More computationally intensive
        - Less widely available in statistical software
        - Interpretation more complex due to multiple test statistics

        #### Practical Considerations

        - Uses GLS detrending like ERS
        - Same null hypothesis as ADF: H<sub>0</sub>: unit root present
        - Recommended for serious econometric analysis when sample size permits
        """, unsafe_allow_html=True)

    # Comparison table of tests
    st.markdown('<div class="section">Comparison of Unit Root Tests</div>', unsafe_allow_html=True)

    comparison_data = {
        'Test': ['Dickey-Fuller (DF)', 'Augmented Dickey-Fuller (ADF)', 'Phillips-Perron (PP)',
                 'KPSS', 'Elliott-Rothenberg-Stock (ERS)', 'Ng-Perron'],
        'Null Hypothesis': ['Unit Root', 'Unit Root', 'Unit Root',
                            'Stationarity', 'Unit Root', 'Unit Root'],
        'Handles Autocorrelation': ['No', 'Yes, through lags', 'Yes, non-parametrically',
                                    'Yes, non-parametrically', 'Yes, through lags', 'Yes, through lags'],
        'Power vs. Near Unit Root': ['Low', 'Low', 'Low',
                                     'Moderate', 'Higher', 'Highest'],
        'Robust to Heteroskedasticity': ['No', 'No', 'Yes',
                                         'Yes', 'No', 'Moderate'],
        'Computational Complexity': ['Low', 'Low', 'Moderate',
                                     'Moderate', 'Moderate', 'High']
    }

    df_comparison = pd.DataFrame(comparison_data)
    st.table(df_comparison)

    # Decision tree for test selection
    st.markdown('<div class="section">Decision Guide for Test Selection</div>', unsafe_allow_html=True)

    st.markdown("""
    ### How to Choose the Right Unit Root Test

    The following decision tree can help you select the appropriate unit root test:

    1. **Is serial correlation present in your data?**
       - YES → Avoid basic DF test, use ADF, PP, ERS, or Ng-Perron
       - NO → Any test is appropriate

    2. **Is your sample size small (< 50 observations)?**
       - YES → Consider ERS or Ng-Perron for better small-sample properties
       - NO → Any test can be used, but ADF is standard

    3. **Are you concerned about test power?**
       - YES → Use ERS or Ng-Perron tests
       - NO → ADF or PP are sufficient

    4. **Is heteroskedasticity a concern?**
       - YES → Consider PP or KPSS tests
       - NO → Any test is appropriate

    5. **Do you want to confirm results with complementary tests?**
       - YES → Use both ADF (null: unit root) and KPSS (null: stationary)
       - NO → Choose based on other criteria

    6. **Is your series potentially fractionally integrated?**
       - YES → Standard tests may not be appropriate, consider specialized tests
       - NO → Standard tests are appropriate
    """)

    # Interpretation guide
    st.markdown('<div class="section">Interpretation Guide</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Interpreting Test Results

    #### For Tests with Null Hypothesis of Unit Root (ADF, PP, ERS, Ng-Perron):

    - **p-value < 0.05**: Reject the null hypothesis. 
      - Conclusion: Series is stationary
      - Action: Can use the series in level form

    - **p-value ≥ 0.05**: Fail to reject the null hypothesis. 
      - Conclusion: Series has a unit root
      - Action: Need to difference the series

    #### For Tests with Null Hypothesis of Stationarity (KPSS):

    - **p-value < 0.05**: Reject the null hypothesis. 
      - Conclusion: Series has a unit root
      - Action: Need to difference the series

    - **p-value ≥ 0.05**: Fail to reject the null hypothesis. 
      - Conclusion: Series is stationary
      - Action: Can use the series in level form

    #### Combined Interpretation (ADF and KPSS):

    | ADF Result | KPSS Result | Interpretation |
    |------------|-------------|----------------|
    | Reject H₀ (p < 0.05) | Fail to Reject H₀ (p ≥ 0.05) | Strong evidence of stationarity |
    | Fail to Reject H₀ (p ≥ 0.05) | Reject H₀ (p < 0.05) | Strong evidence of unit root |
    | Reject H₀ (p < 0.05) | Reject H₀ (p < 0.05) | Conflicting results, possible structural breaks or misspecification |
    | Fail to Reject H₀ (p ≥ 0.05) | Fail to Reject H₀ (p ≥ 0.05) | Insufficient evidence, possible fractional integration or low test power |
    """)


def interactive_simulator_page():
    st.markdown('<div class="sub-header">Interactive Unit Root Test Simulator</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    This interactive simulator allows you to generate different types of time series processes and analyze their unit root properties. 
    You can customize parameters, run various unit root tests, and visualize the results.
    </div>
    """, unsafe_allow_html=True)

    # Create two columns for the parameter selection
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section">Process Parameters</div>', unsafe_allow_html=True)

        process_type = st.selectbox(
            "Select process type:",
            ["Random Walk", "AR(1)", "Trend Stationary", "ARMA", "Fractionally Integrated",
             "Near Explosive", "Seasonal", "Multiple Unit Roots"]
        )

        n_samples = st.slider("Number of observations", 100, 1000, 300)

        # Parameters specific to process types
        if process_type == "AR(1)":
            ar_coef = st.slider("AR coefficient", 0.1, 0.95, 0.7, 0.05)
            process_params = {"ar_params": [ar_coef]}
            process_code = "arma"

        elif process_type == "ARMA":
            ar_coef = st.slider("AR coefficient", 0.1, 0.95, 0.7, 0.05)
            ma_coef = st.slider("MA coefficient", 0.1, 0.95, 0.2, 0.05)
            process_params = {"ar_params": [ar_coef], "ma_params": [ma_coef]}
            process_code = "arma"

        elif process_type == "Fractionally Integrated":
            d_param = st.slider("Fractional integration parameter (d)", -0.4, 1.4, 0.4, 0.1)
            process_params = {"d": d_param}
            process_code = "fractional"

        elif process_type == "Near Explosive":
            process_params = {"near_explosive": True}
            process_code = "near_explosive"

        elif process_type == "Seasonal":
            seasonal_period = st.selectbox("Seasonal period", [4, 12], index=0)
            has_unit_root = st.checkbox("Include random walk component", value=True)
            process_params = {"seasonal": has_unit_root, "s_period": seasonal_period}
            process_code = "seasonal"

        elif process_type == "Multiple Unit Roots":
            process_params = {}
            process_code = "multiple_roots"

        else:  # Random Walk or Trend Stationary
            process_params = {}
            process_code = process_type.lower().replace(" ", "_")

    with col2:
        st.markdown('<div class="section">Deterministic Components</div>', unsafe_allow_html=True)

        include_const = st.checkbox("Include constant (drift)", value=False)
        include_trend = st.checkbox("Include deterministic trend", value=False)

        deterministic = []
        if include_const:
            deterministic.append("constant")
        if include_trend:
            deterministic.append("trend")

        process_params["deterministic"] = deterministic if deterministic else None

        st.markdown('<div class="section">Test Options</div>', unsafe_allow_html=True)

        test_types = st.multiselect(
            "Select tests to run:",
            ["ADF", "KPSS", "PP"],
            default=["ADF", "KPSS"]
        )

        if "ADF" in test_types:
            adf_regression = st.selectbox(
                "ADF regression type:",
                ["c", "ct", "n"],
                format_func=lambda x: {"c": "Constant", "ct": "Constant and Trend", "n": "None"}[x]
            )
        else:
            adf_regression = "c"

        if "KPSS" in test_types:
            kpss_regression = st.selectbox(
                "KPSS regression type:",
                ["c", "ct"],
                format_func=lambda x: {"c": "Constant", "ct": "Constant and Trend"}[x]
            )
        else:
            kpss_regression = "c"

    # Generate the time series
    if st.button("Generate Series and Run Tests"):
        # Generate the series
        series = generate_ts(n=n_samples, process_type=process_code, **process_params)

        # Plot the series
        fig = plot_ts(series, title=f"{process_type} Process")
        st.pyplot(fig)

        # Create columns for test results
        test_cols = st.columns(len(test_types))

        # Run selected tests
        for i, test in enumerate(test_types):
            with test_cols[i]:
                if test == "ADF":
                    result = run_adfuller(series, regression=adf_regression)
                    st.markdown(f"#### ADF Test Results")
                    st.write(f"ADF Statistic: {result['ADF Statistic']:.4f}")
                    st.write(f"p-value: {result['p-value']:.4f}")
                    st.write(f"Used Lags: {result['Used Lags']}")
                    st.write("Critical Values:")
                    for key, value in result['Critical Values'].items():
                        st.write(f"  {key}: {value:.4f}")

                    if result['Reject H0']:
                        st.success("✅ Reject null of unit root (stationary)")
                    else:
                        st.error("❌ Fail to reject null (non-stationary)")

                elif test == "KPSS":
                    result = run_kpss(series, regression=kpss_regression)
                    st.markdown(f"#### KPSS Test Results")
                    st.write(f"KPSS Statistic: {result['KPSS Statistic']:.4f}")
                    st.write(f"p-value: {result['p-value']:.4f}")
                    st.write(f"Used Lags: {result['Used Lags']}")
                    st.write("Critical Values:")
                    for key, value in result['Critical Values'].items():
                        st.write(f"  {key}: {value:.4f}")

                    if result['Reject H0']:
                        st.error("❌ Reject null of stationarity (non-stationary)")
                    else:
                        st.success("✅ Fail to reject null (stationary)")

                elif test == "PP":
                    # For PP test, use the interface directly
                    pp_result = sm.tsa.stattools.phillips_ouliaris(series, regression=adf_regression)
                    st.markdown(f"#### Phillips-Perron Test Results")
                    st.write(f"PP Statistic: {pp_result[0]:.4f}")
                    st.write(f"p-value: {pp_result[1]:.4f}")

                    if pp_result[1] < 0.05:
                        st.success("✅ Reject null of unit root (stationary)")
                    else:
                        st.error("❌ Fail to reject null (non-stationary)")

        # Additional diagnostics
        st.markdown('<div class="section">Series Diagnostics</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["ACF/PACF", "First Difference", "Statistics"])

        with tab1:
            # Plot ACF and PACF
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            sm.graphics.tsa.plot_acf(series, lags=30, ax=ax[0])
            ax[0].set_title('Autocorrelation Function (ACF)')

            sm.graphics.tsa.plot_pacf(series, lags=30, ax=ax[1])
            ax[1].set_title('Partial Autocorrelation Function (PACF)')

            plt.tight_layout()
            st.pyplot(fig)

        with tab2:
            # First difference of the series
            if len(series) > 1:
                diff_series = np.diff(series)

                fig, ax = plt.subplots(2, 1, figsize=(10, 8))
                ax[0].plot(series)
                ax[0].set_title('Original Series')

                ax[1].plot(diff_series)
                ax[1].set_title('First Differenced Series')

                plt.tight_layout()
                st.pyplot(fig)

                # Run tests on differenced series
                st.markdown("#### Unit Root Tests on Differenced Series")

                diff_cols = st.columns(len(test_types))

                for i, test in enumerate(test_types):
                    with diff_cols[i]:
                        if test == "ADF":
                            result = run_adfuller(diff_series, regression=adf_regression)
                            st.markdown(f"**ADF Test on Diff Series**")
                            st.write(f"ADF Stat: {result['ADF Statistic']:.4f}")
                            st.write(f"p-value: {result['p-value']:.4f}")

                            if result['Reject H0']:
                                st.success("✅ Stationary")
                            else:
                                st.error("❌ Non-stationary")

                        elif test == "KPSS":
                            result = run_kpss(diff_series, regression=kpss_regression)
                            st.markdown(f"**KPSS Test on Diff Series**")
                            st.write(f"KPSS Stat: {result['KPSS Statistic']:.4f}")
                            st.write(f"p-value: {result['p-value']:.4f}")

                            if result['Reject H0']:
                                st.error("❌ Non-stationary")
                            else:
                                st.success("✅ Stationary")

                        elif test == "PP":
                            pp_result = sm.tsa.stattools.phillips_ouliaris(diff_series, regression=adf_regression)
                            st.markdown(f"**PP Test on Diff Series**")
                            st.write(f"PP Stat: {pp_result[0]:.4f}")
                            st.write(f"p-value: {pp_result[1]:.4f}")

                            if pp_result[1] < 0.05:
                                st.success("✅ Stationary")
                            else:
                                st.error("❌ Non-stationary")
            else:
                st.write("Series too short to calculate differences.")

        with tab3:
            # Descriptive statistics
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                'Value': [
                    np.mean(series),
                    np.std(series),
                    np.min(series),
                    np.max(series),
                    stats.skew(series),
                    stats.kurtosis(series)
                ]
            })

            st.table(stats_df)

            # Normality test
            jb_stat, jb_pval = stats.jarque_bera(series)
            st.write(f"Jarque-Bera Test - Statistic: {jb_stat:.4f}, p-value: {jb_pval:.4f}")

            if jb_pval < 0.05:
                st.write("Residuals are not normally distributed (reject normality)")
            else:
                st.write("Residuals appear normally distributed (fail to reject normality)")

    # Explanation section
    st.markdown('<div class="section">Key Insights and Interpretation</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Interpretation Tips

    #### Testing Strategy

    1. **Confirm with Multiple Tests**: Use both ADF (null: unit root) and KPSS (null: stationarity) 
       for more robust conclusions.

    2. **Check Graphical Patterns**:
       - Random walks typically show sustained upward or downward movements
       - Stationary series tend to return to their mean
       - ACF of non-stationary series decays very slowly

    3. **Examine First Differences**:
       - If original series is I(1), first differences should be stationary
       - Multiple unit roots require multiple differencing

    #### Common Pitfalls

    1. **Structural Breaks**: Can be mistaken for unit roots. Check for regime changes.

    2. **Deterministic Components**: Including/excluding constants and trends changes interpretation.

    3. **Near Unit Roots**: Difficult to distinguish from true unit roots, especially in small samples.

    4. **Seasonal Integration**: Standard tests don't properly capture seasonal unit roots.

    5. **Multiple Testing**: Conducting many tests increases the chance of Type I errors.
    """)


def size_power_page():
    st.markdown('<div class="sub-header">Size and Power Analysis of Unit Root Tests</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    The performance of unit root tests can be evaluated through their size (Type I error rate) and 
    power (ability to correctly reject the null when it's false). This section provides interactive 
    visualizations of these properties across different scenarios.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Key Concepts in Test Evaluation

    - **Size**: The probability of rejecting the null hypothesis when it is true (Type I error rate)
    - **Power**: The probability of rejecting the null hypothesis when it is false (1 - Type II error rate)

    Ideally, a good test should have:
    1. Size close to the nominal significance level (e.g., 5%)
    2. High power across a wide range of alternatives

    Unit root tests often struggle with:
    1. Size distortions (especially with moving average components)
    2. Low power against near-unit-root alternatives
    """)

    # Generate power and size data
    with st.spinner("Generating simulation data (this may take a moment)..."):
        power_size_df = generate_power_size_data()

    # Plot options
    plot_type = st.radio(
        "Select analysis type:",
        ["Power Analysis", "Size Analysis", "Size-Power Tradeoff"]
    )

    alpha_level = st.selectbox("Significance level (α):", [0.01, 0.05, 0.10], index=1)

    if plot_type == "Power Analysis":
        # Filter for power metrics and selected alpha
        power_df = power_size_df[(power_size_df['Metric'] == 'Power') &
                                 (power_size_df['Alpha'] == alpha_level)]

        fig, ax = plt.subplots(figsize=(10, 6))
        for test in power_df['Test'].unique():
            test_data = power_df[power_df['Test'] == test]
            ax.plot(test_data['AR Coefficient'], test_data['Value'],
                    marker='o', label=f"{test} Test")

        ax.set_title(f'Power Analysis at α = {alpha_level}')
        ax.set_xlabel('AR Coefficient')
        ax.set_ylabel('Power (Probability of Rejecting False Null)')
        ax.axhline(y=alpha_level, color='r', linestyle='--',
                   label=f'Nominal Size (α = {alpha_level})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(fig)

        st.markdown("""
        #### Interpretation of Power Analysis

        Power analysis shows how well tests can detect stationarity when the true process is stationary.

        - **Higher values are better**: Indicates greater ability to correctly reject the false null hypothesis
        - **Impact of persistence**: As the AR coefficient approaches 1 (unit root), power decreases
        - **Test comparison**: Tests with higher curves have better power

        #### Key Observations

        1. **Near Unit Roots**: All tests struggle with processes where AR coefficient is close to 1 (0.95+)
        2. **KPSS Advantage**: For stationary processes, KPSS often has higher power as the null is stationarity
        3. **AR Coefficient Effect**: Power decreases as persistence increases
        """)

    elif plot_type == "Size Analysis":
        # Filter for size metrics and selected alpha
        size_df = power_size_df[(power_size_df['Metric'] == 'Size') &
                                (power_size_df['Alpha'] == alpha_level)]

        fig, ax = plt.subplots(figsize=(10, 6))
        for test in size_df['Test'].unique():
            test_data = size_df[size_df['Test'] == test]
            ax.plot(test_data['AR Coefficient'], test_data['Value'],
                    marker='o', label=f"{test} Test")

        ax.set_title(f'Size Analysis at α = {alpha_level}')
        ax.set_xlabel('AR Coefficient')
        ax.set_ylabel('Size (Type I Error Rate)')
        ax.axhline(y=alpha_level, color='r', linestyle='--',
                   label=f'Nominal Size (α = {alpha_level})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(fig)

        st.markdown("""
        #### Interpretation of Size Analysis

        Size analysis shows how often the test incorrectly rejects the null hypothesis when it is true.

        - **Values close to α are ideal**: The actual rejection rate should match the nominal significance level
        - **Values above α indicate size distortion**: The test rejects the true null too often
        - **Values below α are conservative**: The test may have reduced power

        #### Key Observations

        1. **Size Distortions**: Some tests show significant size distortions, especially for certain parameter values
        2. **ADF vs KPSS**: The tests have different size properties due to their different null hypotheses
        3. **Process Dependence**: Size properties can vary substantially depending on the underlying process
        """)

    else:  # Size-Power Tradeoff
        # Create ROC-like curves
        tests = power_size_df['Test'].unique()
        processes = power_size_df['Process'].unique()

        # Let user select process
        selected_process = st.selectbox(
            "Select process type for analysis:",
            processes)

        # Filter data for selected process and create size-power pairs
        process_data = power_size_df[power_size_df['Process'] == selected_process]

        fig, ax = plt.subplots(figsize=(10, 6))

        for test in tests:
            test_data = process_data[process_data['Test'] == test]

            # Get size and power for each alpha level
            size_values = test_data[test_data['Metric'] == 'Size']['Value'].values
            power_values = test_data[test_data['Metric'] == 'Power']['Value'].values
            alpha_values = test_data[test_data['Metric'] == 'Size']['Alpha'].values

            # Plot size vs power
            ax.plot(size_values, power_values, marker='o', label=f"{test} Test")

            # Annotate points with alpha values
            for i, alpha in enumerate(alpha_values):
                ax.annotate(f"α={alpha}",
                            (size_values[i], power_values[i]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center')

        # Add diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

        ax.set_title(f'Size-Power Tradeoff for {selected_process}')
        ax.set_xlabel('Size (Type I Error Rate)')
        ax.set_ylabel('Power')
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(fig)

        st.markdown("""
        #### Interpretation of Size-Power Tradeoff

        This analysis visualizes the tradeoff between size and power for different tests and significance levels.

        - **Ideal position**: Top-left corner (high power, low size)
        - **Points above diagonal**: Test performs better than random
        - **Comparing tests**: Tests with curves higher and to the left are superior

        #### Key Observations

        1. **Significance Level Effect**: Higher α increases both power and size
        2. **Test Performance**: Some tests maintain better power with less size distortion
        3. **Process Dependence**: The tradeoff varies significantly by process type
        """)

    st.markdown('<div class="section">Sample Size Effects</div>', unsafe_allow_html=True)

    # Create interactive visualization of sample size effects
    sample_sizes = [50, 100, 200, 500, 1000]
    ar_coefs = [0.5, 0.9, 0.95, 0.99, 1.0]

    # Let user select parameters to investigate
    col1, col2 = st.columns(2)

    with col1:
        selected_ar = st.selectbox("Select AR coefficient:", ar_coefs, index=2)

    with col2:
        test_type = st.selectbox("Select test type:", ["ADF", "KPSS"], index=0)

    # Plot visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    # Theoretical power curves (simplified approximation)
    for sample_size in sample_sizes:
        if selected_ar < 1:  # For stationary processes
            if test_type == "ADF":
                # For ADF, power increases with sample size and distance from unit root
                power_curve = np.minimum(1, 0.05 + (1 - selected_ar) * np.sqrt(sample_size) / 2)
                ax.plot(sample_size, power_curve, 'o', markersize=10,
                        label=f"n = {sample_size}")
            else:  # KPSS
                # For KPSS (null is stationarity), we're looking at size rather than power
                size_curve = 0.05 * (1 + (1 - selected_ar) * np.sqrt(sample_size) / 10)
                ax.plot(sample_size, size_curve, 'o', markersize=10,
                        label=f"n = {sample_size}")
        else:  # For unit root process
            if test_type == "ADF":
                # For ADF, we're looking at size
                size_curve = 0.05  # Should be close to nominal level
                ax.plot(sample_size, size_curve, 'o', markersize=10,
                        label=f"n = {sample_size}")
            else:  # KPSS
                # For KPSS, we're looking at power
                power_curve = np.minimum(1, 0.05 + np.sqrt(sample_size) / 10)
                ax.plot(sample_size, power_curve, 'o', markersize=10,
                        label=f"n = {sample_size}")

    ax.set_title(
        f'Sample Size Effect on {"Power" if (selected_ar < 1 and test_type == "ADF") or (selected_ar == 1 and test_type == "KPSS") else "Size"} for AR({selected_ar})')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Rejection Probability')
    ax.grid(True, alpha=0.3)
    ax.legend(title="Sample Size")

    # Add reference line for nominal size
    ax.axhline(y=0.05, color='r', linestyle='--', label='Nominal Size (α = 0.05)')

    st.pyplot(fig)

    st.markdown("""
    #### Impact of Sample Size on Test Performance

    Sample size significantly affects both the size and power of unit root tests:

    1. **Power Increases with Sample Size**:
       - Larger samples provide more information to detect departures from the null
       - For near unit roots, very large samples may be needed for adequate power

    2. **Size Distortions**:
       - Can be more severe in small samples
       - Some tests have better small-sample properties than others

    3. **Asymptotic vs. Finite Sample Properties**:
       - Tests are often derived based on asymptotic theory
       - In finite samples, actual performance may differ from theoretical predictions
       - Small-sample corrections are available for some tests
    """)

    # Multiple unit roots analysis
    st.markdown('<div class="section">Multiple Integration Orders</div>', unsafe_allow_html=True)

    with st.spinner("Generating data for multiple unit roots analysis..."):
        multiple_roots_df = simulate_multiple_unit_roots()

    # Create heatmap-like visualization
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # ADF test heatmap
    adf_data = multiple_roots_df.pivot(index='Process',
                                       columns='Integration Order',
                                       values='ADF Level Rejection Rate')
    sns.heatmap(adf_data, annot=True, cmap='YlGnBu', ax=ax[0])
    ax[0].set_title('ADF Test Rejection Rates (Level)')

    # KPSS test heatmap
    kpss_data = multiple_roots_df.pivot(index='Process',
                                        columns='Integration Order',
                                        values='KPSS Level Rejection Rate')
    sns.heatmap(kpss_data, annot=True, cmap='YlGnBu', ax=ax[1])
    ax[1].set_title('KPSS Test Rejection Rates (Level)')

    plt.tight_layout()
    st.pyplot(fig)

    # Create heatmap for differenced data
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # ADF test heatmap (differenced)
    adf_diff_data = multiple_roots_df.pivot(index='Process',
                                            columns='Integration Order',
                                            values='ADF Differenced Rejection Rate')
    sns.heatmap(adf_diff_data, annot=True, cmap='YlGnBu', ax=ax[0])
    ax[0].set_title('ADF Test Rejection Rates (First Difference)')

    # KPSS test heatmap (differenced)
    kpss_diff_data = multiple_roots_df.pivot(index='Process',
                                             columns='Integration Order',
                                             values='KPSS Differenced Rejection Rate')
    sns.heatmap(kpss_diff_data, annot=True, cmap='YlGnBu', ax=ax[1])
    ax[1].set_title('KPSS Test Rejection Rates (First Difference)')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    #### Interpreting Test Performance with Multiple Integration Orders

    These heatmaps show how unit root tests perform with processes of different integration orders:

    1. **For I(0) processes**:
       - ADF should reject the null (high rejection rate)
       - KPSS should not reject the null (low rejection rate)

    2. **For I(1) processes**:
       - ADF should not reject the null on level (low rejection rate)
       - ADF should reject the null on first difference (high rejection rate)
       - KPSS should reject the null on level (high rejection rate)

    3. **For I(2) processes**:
       - Both tests should not reject non-stationarity for level
       - First differencing is insufficient - series remains non-stationary
       - Second differencing would be required

    #### Practical Implications

    - Testing for unit roots should be sequential (test, difference, test again)
    - Multiple unit roots require multiple differencing
    - Different tests may lead to different conclusions about the order of integration
    """)


def special_cases_page():
    st.markdown('<div class="sub-header">Special Cases in Unit Root Testing</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Standard unit root tests may not perform well in various special scenarios. This section explores 
    challenging cases including fractional integration, near-explosive roots, seasonal unit roots, 
    and structural breaks.
    </div>
    """, unsafe_allow_html=True)

    special_case = st.selectbox(
        "Select special case to explore:",
        ["Fractional Integration", "Near-Explosive Processes",
         "Seasonal Unit Roots", "Structural Breaks"]
    )

    if special_case == "Fractional Integration":
        st.markdown("""
        ### Fractional Integration

        <div class="card">
        Fractionally integrated processes (or fractional differencing) extend the concept of integration 
        to non-integer orders. They are characterized by long memory and slower decay in autocorrelation 
        than standard ARMA models.
        </div>

        #### Mathematical Framework

        A fractionally integrated process is defined by:

        <div class="formula">
        (1-L)<sup>d</sup>y<sub>t</sub> = u<sub>t</sub>
        </div>

        Where:
        - L is the lag operator
        - d is the fractional differencing parameter
        - u<sub>t</sub> is a stationary and invertible ARMA process

        The parameter d determines the memory properties:
        - d = 0: Short memory (standard ARMA)
        - 0 < d < 0.5: Long memory, stationary
        - 0.5 ≤ d < 1: Long memory, non-stationary but mean-reverting
        - d ≥ 1: Non-stationary, non-mean-reverting
        """, unsafe_allow_html=True)

        # Simulate fractional integration
        with st.spinner("Generating fractionally integrated processes..."):
            fractional_df = simulate_fractional_integration()

        # Plot test results across d values
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        ax[0].plot(fractional_df['d Parameter'], fractional_df['ADF Statistic'], 'o-', label='ADF Statistic')
        ax[0].axhline(y=fractional_df['ADF Statistic'].iloc[1], color='r', linestyle='--', label='d=0 value')
        ax[0].set_title('ADF Test Statistic for Different Fractional d Values')
        ax[0].set_xlabel('Fractional Integration Parameter (d)')
        ax[0].set_ylabel('ADF Test Statistic')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(fractional_df['d Parameter'], fractional_df['KPSS Statistic'], 'o-', label='KPSS Statistic')
        ax[1].axhline(y=fractional_df['KPSS Statistic'].iloc[1], color='r', linestyle='--', label='d=0 value')
        ax[1].set_title('KPSS Test Statistic for Different Fractional d Values')
        ax[1].set_xlabel('Fractional Integration Parameter (d)')
        ax[1].set_ylabel('KPSS Test Statistic')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Plot example series for different d values
        st.markdown("#### Example Series with Different d Values")

        d_examples = [-0.2, 0.2, 0.4, 0.8]
        fig, axes = plt.subplots(len(d_examples), 1, figsize=(10, 12))

        for i, d in enumerate(d_examples):
            series = generate_ts(n=300, process_type="fractional", d=d)
            axes[i].plot(series)
            if d < 0:
                memory_type = "Antipersistent (negative memory)"
            elif d == 0:
                memory_type = "No memory (white noise)"
            elif 0 < d < 0.5:
                memory_type = "Long memory, stationary"
            elif 0.5 <= d < 1:
                memory_type = "Long memory, non-stationary but mean-reverting"
            else:
                memory_type = "Non-stationary, non-mean-reverting"

            axes[i].set_title(f'd = {d}: {memory_type}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Value')

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        #### Implications for Unit Root Testing

        Standard unit root tests face challenges with fractionally integrated processes:

        1. **Gray Area Between I(0) and I(1)**:
           - Processes with 0 < d < 1 are neither I(0) nor I(1)
           - Standard tests are designed for integer orders of integration

        2. **Test Behavior**:
           - ADF test: Low power against fractional alternatives
           - KPSS test: Better at detecting long memory but not designed for it

        3. **Practical Consequences**:
           - False conclusions about integration order
           - Inappropriate differencing (over or under-differencing)
           - Misspecified models

        #### Specialized Tests

        For fractionally integrated processes, specialized tests are available:

        1. **Geweke and Porter-Hudak (GPH) Test**:
           - Semiparametric estimator of d based on periodogram

        2. **Robinson's Test**:
           - Tests for specific values of d

        3. **ARFIMA Model Estimation**:
           - MLE-based estimation of ARFIMA(p,d,q) models

        4. **Wavelet-Based Tests**:
           - Multiresolution analysis for long memory detection
        """)

    elif special_case == "Near-Explosive Processes":
        st.markdown("""
        ### Near-Explosive Processes

        <div class="card">
        Near-explosive processes have an autoregressive parameter close to but slightly less than one, 
        making them difficult to distinguish from unit root processes. They can also refer to mildly 
        explosive processes where the parameter is slightly greater than one.
        </div>

        #### Mathematical Framework

        A near-explosive AR(1) process is defined as:

        <div class="formula">
        y<sub>t</sub> = ρy<sub>t-1</sub> + ε<sub>t</sub>
        </div>

        Where:
        - ρ is close to 1 (e.g., ρ = 0.95 or ρ = 0.99)
        - For mildly explosive processes: ρ > 1 (e.g., ρ = 1.01)

        These processes can be modeled in a local-to-unity framework:

        <div class="formula">
        ρ = 1 + c/T
        </div>

        Where:
        - T is the sample size
        - c is a constant (negative for near unit root, positive for mildly explosive)
        """, unsafe_allow_html=True)

        # Interactive demonstration
        st.markdown("#### Interactive Demonstration of Near-Explosive Processes")

        rho_value = st.slider("AR coefficient (ρ)", 0.9, 1.03, 0.99, 0.01)
        sample_size = st.slider("Sample size", 100, 500, 200)

        # Generate series
        np.random.seed(42)
        errors = np.random.normal(0, 1, sample_size)
        series = np.zeros(sample_size)

        for t in range(1, sample_size):
            series[t] = rho_value * series[t - 1] + errors[t]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(series)
        title = f"Near-{'Explosive' if rho_value > 1 else 'Unit Root'} Process: ρ = {rho_value}"
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        st.pyplot(fig)

        # Run tests
        adf_result = run_adfuller(series)
        kpss_result = run_kpss(series)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ADF Test Results")
            st.write(f"ADF Statistic: {adf_result['ADF Statistic']:.4f}")
            st.write(f"p-value: {adf_result['p-value']:.4f}")

            if adf_result['Reject H0']:
                st.success("✅ Reject null of unit root (stationary)")
            else:
                st.error("❌ Fail to reject null (non-stationary)")

        with col2:
            st.markdown("#### KPSS Test Results")
            st.write(f"KPSS Statistic: {kpss_result['KPSS Statistic']:.4f}")
            st.write(f"p-value: {kpss_result['p-value']:.4f}")

            if kpss_result['Reject H0']:
                st.error("❌ Reject null of stationarity (non-stationary)")
            else:
                st.success("✅ Fail to reject null (stationary)")

        st.markdown("""
        #### Challenges in Unit Root Testing

        Near-explosive processes pose several challenges:

        1. **Low Test Power**:
           - Standard tests have low power against near-explosive alternatives
           - Large samples needed to distinguish from unit roots

        2. **Importance in Finance**:
           - Asset price bubbles often follow mildly explosive processes
           - Early detection crucial for financial stability

        3. **Conventional Test Limitations**:
           - ADF, KPSS, and other standard tests not designed for explosive alternatives
           - May miss the transition from unit root to explosive behavior

        #### Specialized Tests for Explosiveness

        Several tests have been developed specifically for detecting explosive behavior:

        1. **Right-Tailed ADF Test**:
           - Modified ADF with alternative of ρ > 1
           - Changed critical values

        2. **SADF and GSADF Tests (Phillips, Wu, Yu)**:
           - Supreme ADF test using forward recursive samples
           - Generalized supreme ADF using flexible windows
           - Can date-stamp bubble periods

        3. **CUSUM Tests**:
           - Monitor for structural change in persistence

        4. **Sign-Based Tests**:
           - More robust to heavy-tailed distributions
        """)

    elif special_case == "Seasonal Unit Roots":
        st.markdown("""
        ### Seasonal Unit Roots

        <div class="card">
        Seasonal unit roots represent non-stationarity at seasonal frequencies. They are common in 
        economic and financial time series with quarterly or monthly data and require special testing procedures.
        </div>

        #### Mathematical Framework

        For a time series with seasonal period s, the seasonal difference operator is:

        <div class="formula">
        (1-L<sup>s</sup>)y<sub>t</sub> = y<sub>t</sub> - y<sub>t-s</sub>
        </div>

        This can be decomposed into:

        <div class="formula">
        (1-L<sup>s</sup>) = (1-L)(1+L+L<sup>2</sup>+...+L<sup>s-1</sup>)
        </div>

        Each factor corresponds to a different frequency, including:
        - Zero frequency (standard unit root): (1-L)
        - Seasonal frequencies: various cyclical components
        """, unsafe_allow_html=True)

        # Simulating seasonal data
        with st.spinner("Generating seasonal data for analysis..."):
            seasonal_df = simulate_seasonality()

        # Let user select seasonal period to visualize
        period = st.radio("Select seasonal period:", [4, 12], index=0)

        # Filter data for the selected period
        period_data = seasonal_df[seasonal_df['Period'] == period]

        # Plot examples
        st.markdown(f"#### Example of Seasonal Time Series (Period = {period})")

        # Generate seasonal examples
        t = np.arange(200)
        seasonal_stationary = generate_ts(n=200, process_type="seasonal",
                                          seasonal=False, s_period=period)
        seasonal_nonstationary = generate_ts(n=200, process_type="seasonal",
                                             seasonal=True, s_period=period)

        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].plot(seasonal_stationary)
        ax[0].set_title('Stationary Seasonal Process')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Value')

        ax[1].plot(seasonal_nonstationary)
        ax[1].set_title('Non-stationary Seasonal Process (with Random Walk)')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Value')

        plt.tight_layout()
        st.pyplot(fig)

        # Plot ACF for both processes
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        sm.graphics.tsa.plot_acf(seasonal_stationary, lags=40, ax=ax[0])
        ax[0].set_title('ACF: Stationary Seasonal Process')

        sm.graphics.tsa.plot_acf(seasonal_nonstationary, lags=40, ax=ax[1])
        ax[1].set_title('ACF: Non-stationary Seasonal Process')

        plt.tight_layout()
        st.pyplot(fig)

        # Test results comparison
        st.markdown("#### Test Performance on Seasonal Data")

        # Create bar chart of p-values
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        # For ADF test
        stationary_data = period_data[period_data['Data Type'] == 'Stationary Seasonal']
        nonstationary_data = period_data[period_data['Data Type'] == 'Non-stationary Seasonal']

        x = np.arange(2)
        width = 0.35

        ax[0].bar(x - width / 2, [stationary_data['ADF Level p-value'].values[0],
                                  nonstationary_data['ADF Level p-value'].values[0]],
                  width, label='ADF on Level')
        ax[0].bar(x + width / 2, [stationary_data['ADF Seasonal Diff p-value'].values[0],
                                  nonstationary_data['ADF Seasonal Diff p-value'].values[0]],
                  width, label='ADF on Seasonal Diff')

        ax[0].set_title('ADF Test p-values')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(['Stationary Seasonal', 'Non-stationary Seasonal'])
        ax[0].axhline(y=0.05, color='r', linestyle='--', label='5% Significance')
        ax[0].set_ylabel('p-value')
        ax[0].legend()

        # For KPSS test
        ax[1].bar(x, [stationary_data['KPSS Level p-value'].values[0],
                      nonstationary_data['KPSS Level p-value'].values[0]],
                  width, label='KPSS on Level')

        ax[1].set_title('KPSS Test p-values')
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(['Stationary Seasonal', 'Non-stationary Seasonal'])
        ax[1].axhline(y=0.05, color='r', linestyle='--', label='5% Significance')
        ax[1].set_ylabel('p-value')
        ax[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        #### Challenges with Seasonal Unit Roots

        Standard unit root tests face several issues with seasonal data:

        1. **Frequency Specificity**:
           - Standard tests only check for unit roots at zero frequency
           - Seasonal unit roots exist at seasonal frequencies (e.g., π/2, π for quarterly data)

        2. **Incorrect Differencing**:
           - Using first differences when seasonal differencing is needed leads to misspecification
           - Over-differencing if both regular and seasonal differences are applied unnecessarily

        3. **Test Complications**:
           - Standard tests may have distorted size and power with seasonal data
           - Deterministic seasonal patterns can be confused with stochastic seasonality

        #### Specialized Tests for Seasonal Unit Roots

        1. **HEGY Test (Hylleberg, Engle, Granger, Yoo)**:
           - Tests for unit roots at zero and seasonal frequencies separately
           - Allows identification of which specific frequencies contain unit roots

        2. **Canova-Hansen Test**:
           - Tests the null of deterministic seasonality against stochastic seasonality
           - Complements HEGY (which tests the opposite null)

        3. **OCSB Test (Osborn, Chui, Smith, Birchenhall)**:
           - Simpler procedure focusing on need for seasonal differencing

        4. **Seasonal KPSS Test**:
           - Extension of KPSS for seasonal frequencies

        #### Practical Approach

        1. Examine ACF/PACF for seasonal patterns
        2. Apply HEGY or other seasonal unit root tests
        3. Difference appropriately based on test results
        4. Consider seasonal ARIMA (SARIMA) models
        """)

    elif special_case == "Structural Breaks":
        st.markdown("""
        ### Structural Breaks

        <div class="card">
        Structural breaks are sudden changes in time series patterns that can severely affect unit root 
        test performance. They can be in the form of mean shifts, trend changes, or variance changes.
        </div>

        #### Types of Structural Breaks

        1. **Level Shift**: A sudden change in the mean of the series
        2. **Trend Change**: A change in the slope of the deterministic trend
        3. **Variance Break**: A change in the volatility of the series
        4. **Combination**: Multiple types of breaks occurring simultaneously

        These breaks can occur at a single point (one-time break) or at multiple points (multiple breaks).
        """, unsafe_allow_html=True)

        # Interactive demonstration
        st.markdown("#### Interactive Demonstration of Structural Breaks")

        break_type = st.selectbox(
            "Select type of structural break:",
            ["Level Shift", "Trend Change", "Variance Change", "Combined Break"]
        )

        sample_size = 200
        break_point = int(sample_size * 0.5)  # Break at midpoint

        np.random.seed(42)

        if break_type == "Level Shift":
            # Generate AR(1) with level shift
            shift_magnitude = st.slider("Shift magnitude", 1.0, 10.0, 5.0, 0.5)

            errors = np.random.normal(0, 1, sample_size)
            series = np.zeros(sample_size)

            # AR(1) process with level shift
            for t in range(1, sample_size):
                series[t] = 0.7 * series[t - 1] + errors[t]

                # Add level shift after break point
                if t >= break_point:
                    series[t] += shift_magnitude

        elif break_type == "Trend Change":
            # Generate trend stationary with trend change
            trend_change = st.slider("Change in trend slope", 0.05, 0.5, 0.2, 0.05)

            t = np.arange(sample_size)
            errors = np.random.normal(0, 1, sample_size)

            # Initial trend
            initial_trend = 0.01 * t

            # Changed trend after break
            additional_trend = np.zeros(sample_size)
            additional_trend[break_point:] = trend_change * (t[break_point:] - t[break_point])

            # Combine
            series = initial_trend + additional_trend + errors

        elif break_type == "Variance Change":
            # Generate series with variance change
            variance_ratio = st.slider("Variance ratio (after/before)", 1.5, 5.0, 3.0, 0.5)

            errors1 = np.random.normal(0, 1, break_point)
            errors2 = np.random.normal(0, np.sqrt(variance_ratio), sample_size - break_point)
            errors = np.concatenate([errors1, errors2])

            series = np.zeros(sample_size)
            for t in range(1, sample_size):
                series[t] = 0.5 * series[t - 1] + errors[t]

        else:  # Combined Break
            # Generate combined breaks
            level_shift = st.slider("Level shift", 0.0, 5.0, 3.0, 0.5)
            variance_ratio = st.slider("Variance ratio", 1.0, 3.0, 2.0, 0.2)

            t = np.arange(sample_size)
            errors1 = np.random.normal(0, 1, break_point)
            errors2 = np.random.normal(0, np.sqrt(variance_ratio), sample_size - break_point)
            errors = np.concatenate([errors1, errors2])

            series = np.zeros(sample_size)
            for t in range(1, sample_size):
                series[t] = 0.5 * series[t - 1] + errors[t]

                # Add level shift after break point
                if t >= break_point:
                    series[t] += level_shift

        # Plot the series
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(series)
        ax.axvline(x=break_point, color='r', linestyle='--', label='Break Point')
        ax.set_title(f'Time Series with {break_type}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        st.pyplot(fig)

        # Run standard and break-aware tests
        adf_result = run_adfuller(series)
        kpss_result = run_kpss(series)

        # Run tests separately on each segment
        before_break = series[:break_point]
        after_break = series[break_point:]

        adf_before = run_adfuller(before_break)
        adf_after = run_adfuller(after_break)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Standard Unit Root Tests (Full Series)")
            st.write(f"ADF Statistic: {adf_result['ADF Statistic']:.4f}, p-value: {adf_result['p-value']:.4f}")
            st.write(f"KPSS Statistic: {kpss_result['KPSS Statistic']:.4f}, p-value: {kpss_result['p-value']:.4f}")

            if adf_result['Reject H0']:
                st.success("ADF: ✅ Reject null of unit root (stationary)")
            else:
                st.error("ADF: ❌ Fail to reject null (non-stationary)")

            if kpss_result['Reject H0']:
                st.error("KPSS: ❌ Reject null of stationarity (non-stationary)")
            else:
                st.success("KPSS: ✅ Fail to reject null (stationary)")

        with col2:
            st.markdown("#### Segment-wise Testing")
            st.write("Before Break:")
            st.write(f"ADF Statistic: {adf_before['ADF Statistic']:.4f}, p-value: {adf_before['p-value']:.4f}")

            st.write("After Break:")
            st.write(f"ADF Statistic: {adf_after['ADF Statistic']:.4f}, p-value: {adf_after['p-value']:.4f}")

            if adf_before['Reject H0']:
                st.success("Before Break: ✅ Stationary")
            else:
                st.error("Before Break: ❌ Non-stationary")

            if adf_after['Reject H0']:
                st.success("After Break: ✅ Stationary")
            else:
                st.error("After Break: ❌ Non-stationary")

        st.markdown("""
        #### Impact on Unit Root Tests

        Structural breaks can severely affect standard unit root tests:

        1. **Reduced Power**:
           - Breaks can make a stationary series appear non-stationary
           - ADF test power drops dramatically with breaks

        2. **Size Distortions**:
           - False rejections of the null hypothesis
           - Particularly severe for KPSS test with level shifts

        3. **Misleading Inference**:
           - May lead to over-differencing
           - Incorrect model specification

        #### Break-Aware Unit Root Tests

        Several tests have been developed to handle structural breaks:

        1. **Perron Test (1989)**:
           - Earliest break-aware test
           - Requires pre-specified break point
           - Tests null of unit root with break against alternative of stationarity around broken trend

        2. **Zivot-Andrews Test (1992)**:
           - Endogenous break detection
           - More powerful than standard ADF when breaks present
           - Tests null of unit root against alternative of stationarity with single break

        3. **Lumsdaine-Papell Test**:
           - Extension of Zivot-Andrews to allow two breaks

        4. **Lee-Strazicich Test**:
           - Tests null of unit root with break against alternative of stationarity with break
           - Avoids spurious rejections common in other tests

        5. **Carrion-i-Silvestre Test**:
           - Allows for multiple breaks under both null and alternative

        #### Practical Approach

        1. Visually inspect data for potential breaks
        2. Use structural break tests (Chow, Quandt, Bai-Perron) to identify breaks
        3. Apply break-aware unit root tests
        4. Consider segmented analysis if appropriate
        5. Model breaks explicitly in subsequent time series models
        """)


def references_page():
    st.markdown('<div class="sub-header">References and Resources</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    This section provides key references, papers, books, and online resources for further study of unit root tests.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Key Academic Papers

    #### Foundational Papers

    1. Dickey, D.A., and Fuller, W.A. (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association*, 74, 427-431.

    2. Phillips, P.C.B., and Perron, P. (1988). "Testing for a Unit Root in Time Series Regression." *Biometrika*, 75, 335-346.

    3. Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., and Shin, Y. (1992). "Testing the Null Hypothesis of Stationarity against the Alternative of a Unit Root." *Journal of Econometrics*, 54, 159-178.

    4. Elliott, G., Rothenberg, T.J., and Stock, J.H. (1996). "Efficient Tests for an Autoregressive Unit Root." *Econometrica*, 64, 813-836.

    5. Ng, S., and Perron, P. (2001). "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica*, 69, 1519-1554.

    #### Special Cases

    6. Hylleberg, S., Engle, R.F., Granger, C.W.J., and Yoo, B.S. (1990). "Seasonal Integration and Cointegration." *Journal of Econometrics*, 44, 215-238.

    7. Perron, P. (1989). "The Great Crash, the Oil Price Shock, and the Unit Root Hypothesis." *Econometrica*, 57, 1361-1401.

    8. Zivot, E., and Andrews, D.W.K. (1992). "Further Evidence on the Great Crash, the Oil-Price Shock, and the Unit-Root Hypothesis." *Journal of Business & Economic Statistics*, 10, 251-270.

    9. Granger, C.W.J., and Joyeux, R. (1980). "An Introduction to Long-Memory Time Series Models and Fractional Differencing." *Journal of Time Series Analysis*, 1, 15-29.

    10. Phillips, P.C.B., Wu, Y., and Yu, J. (2011). "Explosive Behavior in the 1990s Nasdaq: When Did Exuberance Escalate Asset Values?" *International Economic Review*, 52, 201-226.

    ### Books and Textbooks

    1. Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press.

    2. Enders, W. (2014). *Applied Econometric Time Series*. Wiley.

    3. Patterson, K. (2011). *Unit Root Tests in Time Series Volume 1: Key Concepts and Problems*. Palgrave Macmillan.

    4. Patterson, K. (2012). *Unit Root Tests in Time Series Volume 2: Extensions and Developments*. Palgrave Macmillan.

    5. Tsay, R.S. (2010). *Analysis of Financial Time Series*. Wiley.

    ### Online Resources

    1. [Rob Hyndman's Forecasting: Principles and Practice](https://otexts.com/fpp3/) - Open-access textbook with sections on stationarity and unit roots

    2. [Kevin Sheppard's Python for Econometrics](https://www.kevinsheppard.com/teaching/python/notes/) - Python resources for time series analysis

    3. [StatsModels Documentation on Unit Root Tests](https://www.statsmodels.org/stable/tsa.html) - Python implementation details

    4. [Duke University's Unit Root Testing Guide](https://people.duke.edu/~rnau/411diff.htm) - Intuitive explanation of unit roots and differencing

    5. [R's tseries and urca Packages Documentation](https://cran.r-project.org/web/packages/urca/urca.pdf) - Comprehensive unit root testing in R

    ### Software Implementations

    1. **Python**:
       - statsmodels: ADF, KPSS, Phillips-Perron
       - arch: Zivot-Andrews test
       - pmdarima: Unit root testing for ARIMA modeling

    2. **R**:
       - tseries: ADF, KPSS
       - urca: Comprehensive unit root tests including seasonal and break tests
       - forecast: Testing as part of automatic forecasting

    3. **MATLAB**:
       - Econometrics Toolbox: Multiple unit root tests

    4. **EViews**:
       - Built-in unit root testing framework
       - Specialized tests for panel data

    5. **Stata**:
       - dfuller, pperron, kpss commands
       - Additional user-written packages for specialized tests
    """)

    # Embed a downloadable cheat sheet
    st.markdown('<div class="section">Unit Root Testing Cheat Sheet</div>', unsafe_allow_html=True)

    # Create a dataframe for the cheat sheet
    cheatsheet_data = {
        'Test': ['Augmented Dickey-Fuller (ADF)', 'KPSS', 'Phillips-Perron (PP)',
                 'Elliott-Rothenberg-Stock (DF-GLS)', 'Zivot-Andrews', 'HEGY'],
        'Null Hypothesis': ['Unit root', 'Stationarity', 'Unit root',
                            'Unit root', 'Unit root (without breaks)', 'Seasonal unit root'],
        'When to Use': ['General purpose', 'Complementary to ADF', 'Serial correlation issues',
                        'Need higher power', 'Structural breaks present', 'Seasonal data'],
        'Key Parameters': ['Lag selection, deterministic terms', 'Bandwidth, deterministic terms',
                           'Bandwidth, deterministic terms', 'Deterministic terms, lag selection',
                           'Break type, lag selection', 'Frequency selection, deterministics'],
        'Python Implementation': ['statsmodels.tsa.stattools.adfuller', 'statsmodels.tsa.stattools.kpss',
                                  'statsmodels.tsa.stattools.phillips_ouliaris',
                                  'Not in statsmodels core',
                                  'arch.unitroot.ZivotAndrews',
                                  'Not widely implemented']
    }

    cheatsheet_df = pd.DataFrame(cheatsheet_data)

    # Create CSV for download
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(cheatsheet_df)

    st.download_button(
        label="Download Unit Root Testing Cheat Sheet",
        data=csv,
        file_name='unit_root_testing_cheatsheet.csv',
        mime='text/csv',
    )

    # Display the cheat sheet
    st.table(cheatsheet_df)

    # Upcoming developments section
    st.markdown('<div class="section">Recent and Upcoming Developments</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Recent Developments in Unit Root Testing

    1. **Machine Learning Integration**:
       - Hybrid approaches combining traditional tests with ML for break detection
       - Neural network-based unit root tests with improved small-sample properties

    2. **Robust Methods**:
       - Tests robust to heavy-tailed distributions
       - Quantile-based unit root testing

    3. **High-Dimensional Methods**:
       - Factor models for panel unit root testing
       - Multiple testing procedures with controlled family-wise error rate

    4. **Improved Local Power**:
       - New tests with better power against local alternatives
       - GLS extensions with optimal power curves

    5. **Nonparametric and Semiparametric Approaches**:
       - Wavelet-based unit root tests
       - Tests without distributional assumptions
    """)


if __name__ == "__main__":
    main()