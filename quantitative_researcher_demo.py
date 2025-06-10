import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from fredapi import Fred
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

# -----------------------------
# ADF Functions
# -----------------------------
def run_adf_test(series):
    result = adfuller(series)
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
        "crit_values": result[4]
    }

def display_adf_results(adf_result):
    st.write(f"**ADF Statistic:** {adf_result['adf_stat']:.2f}")
    st.write(f"**p-value:** {adf_result['p_value']:.2f}")
    st.write(f"**Used lag:** {adf_result['used_lag']}")
    st.write(f"**Number of observations:** {adf_result['n_obs']}")
    st.write("**Critical Values:**")
    for key, value in adf_result["crit_values"].items():
        st.write(f"{key}:{value:.2f}")

# -----------------------------
# Data Loading Functions
# -----------------------------
@st.cache_data
def load_spot_data(hist_years, end_date):
    start_date = end_date - datetime.timedelta(days=365 * (hist_years+1))
    data = fred.get_series("DEXUSEU", observation_start=start_date, observation_end=end_date)
    df = pd.DataFrame(data, columns=["EURUSD"])
    df["EURUSD"] = df["EURUSD"].interpolate(method='linear')
    df = df.reindex(pd.date_range(start=start_date, end=end_date, freq='B'))
    df.index.name = 'Date'
    return df

def load_carry_data(spot_data):
    start_date = spot_data.index.min()
    end_date = spot_data.index.max()
    eur_3m = fred.get_series("IR3TIB01EZM156N", observation_start=start_date, observation_end=end_date)
    usd_3m = fred.get_series("DTB3", observation_start=start_date, observation_end=end_date)
    df = pd.DataFrame({
        "3-month EURIBOR": eur_3m,
        "3-month T-Bill": usd_3m
    })
    df["3-month T-Bill"] = df["3-month T-Bill"].ffill()
    df["3-month EURIBOR"] = df["3-month EURIBOR"].interpolate(method='linear')
    df["Carry"] = df["3-month T-Bill"] - df["3-month EURIBOR"]
    df = df.reindex(pd.date_range(start=start_date, end=end_date, freq='B'))
    df.index.name = 'Date'
    return df

# -----------------------------
# Feature Engineering
# -----------------------------
def calculate_first_difference(df):
    df["First-Order Difference"] = df["EURUSD"].diff()
    return df

def calculate_log_returns(df):
    df["Log Return"] = np.log(df["EURUSD"] / df["EURUSD"].shift(1))
    return df

def calculate_momentum(df, windows):
    for window in windows:
        df[f"{window}-day Momentum"] = df["Log Return"].rolling(window).sum()
    return df

def calculate_volatility(df, windows):
    for window in windows:
        df[f"{window}-day Realised Volatility"] = df["Log Return"].rolling(window).std()
    return df

def calculate_value(df):
    df["Value Factor"] = np.log(df['EURUSD'] / df['EURUSD'].rolling(window=252).mean())
    return df

def calculate_quality(df):
    df["Quality Factor"] = 1 / df['Carry'].rolling(window=21).std()
    return df

def prepare_feature_dataset(hist_years, end_date, momentum_windows, volatility_windows):
    df = load_spot_data(hist_years, end_date)
    df = calculate_first_difference(df)
    df = calculate_log_returns(df)
    df = df.join(load_carry_data(df), how='left')
    df = calculate_momentum(df, momentum_windows)
    df = calculate_volatility(df, volatility_windows)
    df = calculate_value(df)
    df = calculate_quality(df)
    start_date = end_date - datetime.timedelta(days=365 * (hist_years))
    return df[start_date:end_date]

# -----------------------------
# Model Training & Backtesting
# -----------------------------
def train_and_backtest_model(df):
    # Drop rows with NaNs
    df = df.dropna()

    # Label creation: Long (1), Neutral (0), Short (-1)
    df["Signal"] = df["Log Return"].shift(-1)
    df["Signal"] = df["Signal"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # Features and labels
    features = [
        'Carry',
        '5-day Momentum', '10-day Momentum', '20-day Momentum',
        '5-day Realised Volatility', '10-day Realised Volatility', '20-day Realised Volatility',
        'Value Factor', 'Quality Factor'
    ]
    df = df.dropna(subset=features + ["Signal"])

    X = df[features]
    y = df["Signal"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    df.loc[X_test.index, "Predicted Signal"] = clf.predict(X_test)

    # Backtest strategy
    df["Strategy Return"] = df["Predicted Signal"].shift(1) * df["Log Return"]
    df.dropna(inplace=True)
    
    return df
    
# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.markdown("""
    ## Stationarity and Dickey-Fuller
    
    A time series is said to be **stationary** if its statistical properties — like **mean**, **variance**, and **autocorrelation** — do not change over time.
    
    Many statistical and machine learning models **assume stationarity**. If a time series is not stationary model estimates may become **biased** or **inefficient** and forecasts may be **unreliable** or even **invalid**.
        
    The **Augmented Dickie-Fuller (ADF) test** checks for stationarity by testing the presence of a **unit root**.
    
    - **Null Hypothesis (H₀)**: Series has a unit root → it's non-stationary.
    - **Alternative Hypothesis (H₁)**: Series is stationary.
    
    The test returns:
    - A **test statistic** (more negative means more evidence against H₀),
    - A **p-value** (if less than 0.05, reject H₀),
    - **Critical values** at common confidence levels (1%, 5%, 10%).
    
    Let's run the ADF test on the EUR/USD spot rate:
    """)
    
    # Streamlit slider to choose the number of years
    hist_years = st.slider("Select number of years of historical data", min_value=1, max_value=20, value=10)
    
    # Load data
    end_date = datetime.date(2025,4,1)
    momentum_windows = volatility_windows = [5,10,20]
    df = prepare_feature_dataset(hist_years, end_date, momentum_windows, volatility_windows)
    
    # Plot EUR/USD spot rate
    st.subheader(f"EUR/USD Spot Rate (Last {hist_years} Years)")
    st.line_chart(df["EURUSD"].dropna())
    
    # Perform ADF test on EUR/USD spot rate
    st.subheader(f"ADF Test on EUR/USD Spot Rate (Last {hist_years} Years)")
    adf_res = run_adf_test(df["EURUSD"])
    display_adf_results(adf_res)
    
    # Interpretation
    if adf_res["p_value"] < 0.05:
        st.success("The p-value is less than 0.05, so we reject the null hypothesis. The series is likely **stationary**.")
    else:
        st.warning("The p-value is greater than 0.05, so we fail to reject the null hypothesis. The series is likely **non-stationary**.")
    
    st.markdown("""
    ---
    ## Stationarity in FX Markets
    
    Foreign exchange rates like EUR/USD are typically **non-stationary** because they are influenced by a wide range of macroeconomic, geopolitical, and market-driven factors that evolve over time.

    This causes the mean and variance to change over time, meaning they do not revert to a long-term average, and their statistical properties evolve.
    
    To address this, we typically **transform the data** using:
    - **First-order differencing** to capture absolute changes, or
    - **Log-returns** to measure relative changes.

    These transformations help stabilize the time series, making it more suitable for modeling and forecasting.
    
    ### 1. First-Order Difference
    """)
    
    st.latex(r"P_t - P_{t-1}")
    
    st.markdown("""
    - Measures the **absolute change** in price between periods.
    - Use when modeling **absolute price changes**.
    - Helps remove trends and stabilize the mean.
    
    Let's visualize the first-order difference and test its stationarity:
    """)
    
    st.subheader(f"First-Order Difference of EUR/USD Spot (Last {hist_years} Years)")
    st.line_chart(df["First-Order Difference"].dropna())
    
    st.subheader("ADF Test on First-Order Difference")
    adf_fd_res = run_adf_test(df["First-Order Difference"].dropna())
    display_adf_results(adf_fd_res)
    
    if adf_fd_res["p_value"] < 0.05:
        st.success("The p-value is less than 0.05, so we reject the null hypothesis. The first-order differenced series is likely **stationary**.")
    else:
        st.warning("The p-value is greater than 0.05, so we fail to reject the null hypothesis. The first-order differenced series may still be **non-stationary**.")
    
    st.markdown("""
    ### 2. Log-Return
    """)
    
    st.latex(r"\log\left(\frac{\text{P}_t}{\text{P}_{t-1}}\right)")
    
    st.markdown("""
    - Measures **proportional or percentage change** between periods.
    - Preferred in financial analysis for modeling returns and volatility.
    - Use when modeling **relative changes or returns**.
    
    Let's visualize the log returns and test their stationarity:
    """)
    
    # Plot log returns
    st.subheader(f"Log Returns of EUR/USD Spot (Last {hist_years} Years)")
    st.line_chart(df["Log Return"].dropna())

    # ADF test on log returns
    st.subheader("ADF Test on Log Returns")
    adf_lr_res = run_adf_test(df["Log Return"].dropna())
    display_adf_results(adf_lr_res)
    
    # Interpretation for log return test
    if adf_lr_res["p_value"] < 0.05:
        st.success("The p-value is less than 0.05, so we reject the null hypothesis. The **log return series is stationary**, and can be used for modeling and forecasting.")
    else:
        st.warning("The p-value is greater than 0.05, so we fail to reject the null hypothesis. Even the log return series is **non-stationary**, and further transformation may be required.")

    st.markdown("""
    Once stationarity is achieved, you can begin modeling.
    
    The transformed raw price data—often next-day returns—becomes the **target variable** your model aims to predict.
    
    **Signals** are created from the transformed data or related economic and market factors. These signals serve as input features or predictors your model uses to forecast future returns or price movements.
    
    You feed these signals into a machine learning model (e.g., regression, random forest, XGBoost, neural networks) or a statistical model (e.g., linear regression, ARIMA with exogenous variables).
    
    The model learns the relationships between the signals and future returns.
    
    Finally, the model outputs predictions—weighted combinations of the signals—that can be translated into trading decisions such as buy, sell, or hold.

    ---
    ## Signal Generation
    
    After preparing and transforming your FX data, the next step is to **generate signals** that capture meaningful patterns or factors driving currency returns. These signals act as features in your predictive models.

    Below are common FX signals with explanations and example visualizations:

    ### 1. Carry
    
    Carry measures the interest rate differential between two currencies, reflecting returns from holding a position.
    
    - Typically, carry is the difference between short-term interest rates (e.g., 3-month rates).
    - Positive carry means the base currency pays higher interest.
    
    Example: 3-month T-Bill minus 3-month EURIBOR
    """)
    
    st.line_chart(df["Carry"])
    
    st.markdown(f"""
    ### 2. Momentum
    
    Momentum measures the tendency of prices to continue moving in the same direction over a specified window.
    
    - Calculated as the rolling sum of log returns over `n` days.
    - Captures recent trend strength.
    
    Example: 5-day, 10-day and 20-day momentum of EUR/USD Spot Rate (Last {hist_years} Years)
    """)
    
    st.line_chart(df[[f"{w}-day Momentum" for w in momentum_windows]])
    
    st.markdown("""
    ### 3. Volatility
    
    Volatility measures the degree of variation in price or returns over a window and indicates risk.
    
    - Calculated as rolling standard deviation of log returns over `n` days.
    - Captures market uncertainty or turbulence.
    
    Example: 5-day, 10-day and 20-day realised volatility of log returns
    
    """)

    st.line_chart(df[[f"{w}-day Realised Volatility" for w in volatility_windows]])
    
    st.markdown("""
    ### 4. Value
    
    Value factors assess whether the currency is cheap or expensive relative to a longer-term average.
    
    - Calculated as the log of the spot rate divided by its rolling average (e.g. 252 days).
    - A high positive value indicates the currency is expensive; negative means cheap.
    
    Example: Value factor
    
    """)

    st.line_chart(df["Value Factor"])
    
    st.markdown("""
    ### 5. Quality
    
    Quality signals often measure stability or low volatility in carry or other factors.
    
    - Could be the inverse of the rolling standard deviation of the carry.
    - Indicates the reliability of the carry or other signals.
    
    Example: Quality factor
    
    """)

    st.line_chart(df["Quality Factor"])

    st.markdown("""
    ---

    ## Model Training and Backtesting 

    Now that we've generated a set of FX signals—including carry, momentum, volatility, value, and quality—we're moving into the **modeling and backtesting** stage.

    In this section, we:
    
    1. **Define the prediction target**: We aim to forecast the **sign of next-day log returns**—whether they will be **positive (go long)**, **negative (go short)**, or **neutral**.
    2. **Train a machine learning model**: We'll use a **scikit-learn classifier** (e.g., Random Forest or Logistic Regression) trained on our engineered features.
    3. **Generate trading signals**: The model outputs directional signals:
       - **1** for long,
       - **-1** for short,
       - **0** for neutral.
    4. **Backtest the strategy**:
       - Compute **daily strategy returns** by multiplying predicted positions with actual log returns.
       - Calculate **cumulative returns** to visualize performance.
       - Compare against a **buy-and-hold baseline** in EUR/USD.
    
    This helps evaluate whether our signal-based strategy outperforms passive market exposure.

    ---

    ## Strategy vs. Buy & Hold Cumulative Return
    
    To evaluate the performance of our machine learning-based trading strategy, we compare it against a simple **buy-and-hold** strategy on EUR/USD.
    
    We calculate **cumulative returns** starting from a user-defined date to understand relative performance over time.
    
    - **Strategy Return**: Based on model predictions (long/short/neutral).
    - **Buy & Hold**: Simply holding EUR/USD from the selected start date.
    - The chart below visualizes both series to reveal any outperformance.
    
    Use the **slider** below to select the start date for cumulative return comparison.
    """)

    df = train_and_backtest_model(df)

    # Date slider
    start_date_slider = st.slider(
        "Select start date for cumulative return comparison",
        min_value=df.index.min().date(),
        max_value=df.index.max().date(),
        value=df.index.min().date()
    )
    
    # Filter the DataFrame based on the slider
    df_filtered = df[df.index.date >= start_date_slider].copy()
    
    # Recalculate cumulative returns from the selected date
    df_filtered["Strategy Cumulative Return"] = (df_filtered["Strategy Return"] + 1).cumprod()
    df_filtered["Buy & Hold Cumulative Return"] = (df_filtered["Log Return"] + 1).cumprod()
    
    # Rename for clean plotting
    df_plot = df_filtered[["EURUSD", "Predicted Signal", "Strategy Cumulative Return", "Buy & Hold Cumulative Return"]].rename(
        columns={
            "Strategy Cumulative Return": "Strategy",
            "Buy & Hold Cumulative Return": "Buy & Hold"
        }
    )

    # Plot as a line chart in Streamlit
    st.line_chart(df_plot[["Strategy", "Buy & Hold"]])
    
    # Additional explanation
    st.markdown("""
    ### Interpretation
    
    - If the **blue line (Strategy)** is consistently above the **green line (Buy & Hold)**, it means our model-based trading decisions outperformed a passive EUR/USD investment.
    - Large divergences suggest periods of strong signal accuracy.
    - Flat strategy returns imply the model went neutral or had mixed prediction success.
    
    This comparison helps assess whether machine learning adds value beyond traditional currency exposure.
    """)
        
if __name__ == "__main__":
    main()
