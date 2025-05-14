import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
print(type(pd.read_csv))
df=pd.read_csv('TSLA.csv')
#preprocessing the data

#convert date column to datetime and set as the index
df['Date']=pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)
#selection of closing prices for analysis
close_prices=df['Close']
#Check for missing values
print("Missing values:\n",close_prices.isna().sum())
#plotting closing prices
plt.figure(figsize=(12, 6))
plt.plot(close_prices, label='TSLA Close Price')
plt.title('Tesla (TSLA) Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.show()
# performing ADF test
def adf_test(series, title=''):
    result = adfuller(series.dropna())
    print(f'ADF Test for {title}:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    if result[1] < 0.05:
        print("Result: Stationary (reject null hypothesis)\n")
    else:
        print("Result: Non-stationary (fail to reject null hypothesis)\n")

# Testing stationarity of original series
adf_test(close_prices, 'Original Close Prices')

# Differencing the series to make it stationary
diff_prices = close_prices.diff().dropna()

# Testing stationarity of differenced series
adf_test(diff_prices, 'Differenced Close Prices')

# Plotting the differenced series
plt.figure(figsize=(12, 6))
plt.plot(diff_prices, label='Differenced TSLA Close Price')
plt.title('Differenced Tesla Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price Difference')
plt.legend()
plt.show()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plotting the ACF and PACF for the differenced series
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(diff_prices, lags=40, ax=plt.gca())
plt.title('ACF of Differenced Series')
plt.subplot(122)
plot_pacf(diff_prices, lags=40, ax=plt.gca())
plt.title('PACF of Differenced Series')
plt.tight_layout()
plt.show()
#fitting the model(1,1,1)model
arima_model = ARIMA(close_prices, order=(1, 1, 1))
arima_result = arima_model.fit()

# Printing model summary
print(arima_result.summary())

# Forecasting the next 30 days
forecast_steps = 30
forecast = arima_result.forecast(steps=forecast_steps)

# Create forecast index
last_date = close_prices.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

# Plotting actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(close_prices[-100:], label='Actual Close Price')
plt.plot(forecast_index, forecast, label='Forecasted Close Price', color='red')
plt.title('ARIMA Forecast of TSLA Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.show()
# Calculation of the daily returns
returns = close_prices.pct_change().dropna() * 100  # In percentage

# Plotting of the returns
plt.figure(figsize=(12, 6))
plt.plot(returns, label='TSLA Daily Returns')
plt.title('Tesla Daily Returns')
plt.xlabel('Date')
plt.ylabel('Returns (%)')
plt.legend()
plt.show()

# Fitting GARCH(1,1) model
garch_model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
garch_result = garch_model.fit(disp='off')

# Printing GARCH model summary
print(garch_result.summary())

# Plotting conditional volatility
plt.figure(figsize=(12, 6))
plt.plot(garch_result.conditional_volatility, label='Conditional Volatility')
plt.title('GARCH(1,1) Conditional Volatility of TSLA Returns')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()
# The residuals of ARIMA diagnostics
arima_residuals = arima_result.resid
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(arima_residuals, label='ARIMA Residuals')
plt.title('ARIMA Residuals')
plt.legend()
plt.subplot(212)
plot_acf(arima_residuals, lags=40, ax=plt.gca())
plt.title('ACF of ARIMA Residuals')
plt.tight_layout()
plt.show()

# standardized GARCH residuals
garch_residuals = garch_result.resid / garch_result.conditional_volatility
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(garch_residuals, label='GARCH Standardized Residuals')
plt.title('Standardized GARCH Residuals')
plt.legend()
plt.subplot(212)
plot_acf(garch_residuals, lags=40, ax=plt.gca())
plt.title('ACF of Standardized GARCH Residuals')
plt.tight_layout()
plt.show()

