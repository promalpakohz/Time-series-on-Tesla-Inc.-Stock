import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Title of the app
st.title("Tesla Stock Price Analysis with ARIMA and GARCH")

# Load data
try:
    df = pd.read_csv('TSLA.csv')
except FileNotFoundError:
    st.error("TSLA.csv not found. Please ensure the file is in the repository.")
    st.stop()

# Preprocessing the data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
close_prices = df['Close']

# Display missing values
st.write("Missing values:", close_prices.isna().sum())

# Plot closing prices
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(close_prices, label='TSLA Close Price')
ax.set_title('Tesla (TSLA) Closing Prices Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price (USD)')
ax.legend()
st.pyplot(fig)

# ADF Test function
def adf_test(series, title=''):
    result = adfuller(series.dropna())
    st.write(f'ADF Test for {title}:')
    st.write(f'ADF Statistic: {result[0]}')
    st.write(f'p-value: {result[1]}')
    st.write('Critical Values:')
    for key, value in result[4].items():
        st.write(f'\t{key}: {value}')
    if result[1] < 0.05:
        st.write("Result: Stationary (reject null hypothesis)\n")
    else:
        st.write("Result: Non-stationary (fail to reject null hypothesis)\n")

# Testing stationarity
st.subheader("Stationarity Test")
adf_test(close_prices, 'Original Close Prices')

# Differencing the series
diff_prices = close_prices.diff().dropna()
adf_test(diff_prices, 'Differenced Close Prices')

# Plot differenced series
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(diff_prices, label='Differenced TSLA Close Price')
ax.set_title('Differenced Tesla Closing Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Price Difference')
ax.legend()
st.pyplot(fig)

# ACF and PACF plots
st.subheader("ACF and PACF of Differenced Series")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plot_acf(diff_prices, lags=40, ax=ax1)
ax1.set_title('ACF of Differenced Series')
plot_pacf(diff_prices, lags=40, ax=ax2)
ax2.set_title('PACF of Differenced Series')
plt.tight_layout()
st.pyplot(fig)

# ARIMA Model
st.subheader("ARIMA Model")
arima_model = ARIMA(close_prices, order=(1, 1, 1))
arima_result = arima_model.fit()
st.text(str(arima_result.summary()))

# Forecast
forecast_steps = 30
forecast = arima_result.forecast(steps=forecast_steps)
last_date = close_prices.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

# Plot actual vs forecast
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(close_prices[-100:], label='Actual Close Price')
ax.plot(forecast_index, forecast, label='Forecasted Close Price', color='red')
ax.set_title('ARIMA Forecast of TSLA Close Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price (USD)')
ax.legend()
st.pyplot(fig)

# Daily returns
returns = close_prices.pct_change().dropna() * 100
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(returns, label='TSLA Daily Returns')
ax.set_title('Tesla Daily Returns')
ax.set_xlabel('Date')
ax.set_ylabel('Returns (%)')
ax.legend()
st.pyplot(fig)

# GARCH Model
st.subheader("GARCH Model")
garch_model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
garch_result = garch_model.fit(disp='off')
st.text(str(garch_result.summary()))

# Plot conditional volatility
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(garch_result.conditional_volatility, label='Conditional Volatility')
ax.set_title('GARCH(1,1) Conditional Volatility of TSLA Returns')
ax.set_xlabel('Date')
ax.set_ylabel('Volatility')
ax.legend()
st.pyplot(fig)

# ARIMA Residuals
st.subheader("ARIMA Residuals")
arima_residuals = arima_result.resid
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
ax1.plot(arima_residuals, label='ARIMA Residuals')
ax1.set_title('ARIMA Residuals')
ax1.legend()
plot_acf(arima_residuals, lags=40, ax=ax2)
ax2.set_title('ACF of ARIMA Residuals')
plt.tight_layout()
st.pyplot(fig)

# GARCH Residuals
st.subheader("GARCH Residuals")
garch_residuals = garch_result.resid / garch_result.conditional_volatility
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
ax1.plot(garch_residuals, label='GARCH Standardized Residuals')
ax1.set_title('Standardized GARCH Residuals')
ax1.legend()
plot_acf(garch_residuals, lags=40, ax=ax2)
ax2.set_title('ACF of Standardized GARCH Residuals')
plt.tight_layout()
st.pyplot(fig)
