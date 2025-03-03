import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# NVDA stock data
nvda_data = yf.Ticker("NVDA")
nvda = nvda_data.history(start='2023-03-03', end='2025-03-04')

# Benchmark data
nasdaq_data = yf.Ticker("^NDX")
nasdaq = nasdaq_data.history(start='2023-03-01', end='2025-03-01')

# Technical Indicators
nvda['SMA_50'] = nvda['Close'].rolling(window=50).mean()
nvda['SMA_200'] = nvda['Close'].rolling(window=200).mean()
nvda['EMA_12'] = nvda['Close'].ewm(span=12, adjust=False).mean()
nvda['EMA_26'] = nvda['Close'].ewm(span=26, adjust=False).mean()
nvda['MACD'] = nvda['EMA_12'] - nvda['EMA_26']
nvda['Signal_Line'] = nvda['MACD'].ewm(span=9, adjust=False).mean()
nvda['RSI'] = 100 - (100 / (1 + nvda['Close'].diff(1).clip(lower=0).rolling(14).mean() / nvda['Close'].diff(1).clip(upper=0).abs().rolling(14).mean()))

# Daily returns and cumulative returns
nvda['Daily_Return'] = nvda['Close'].pct_change()
nvda['Cumulative_Return'] = (1 + nvda['Daily_Return']).cumprod() - 1

# Visualisation of stock price and moving averages
plt.plot(nvda.index, nvda['Close'], label='NVDA Close Price')
plt.plot(nvda.index, nvda['SMA_50'], label='50-day SMA')
plt.plot(nvda.index, nvda['SMA_200'] , label='200-day SMA')
plt.title('Nvidia Stock Price with Moving Averages')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.show()

# Visualisation of MACD
plt.plot(nvda.index, nvda['MACD'], label='MACD')
plt.plot(nvda.index, nvda['Signal_Line'], label='Signal Line')
plt.title('MACD for Nvidia Stock')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('MACD')
plt.grid(True)
plt.legend()
plt.show()

# Visualisation of RSI
plt.plot(nvda.index, nvda['RSI'], label='RSI')
plt.axhline(y=70, color='r', linestyle='--')
plt.axhline(y=30, color='g', linestyle='--')
plt.title('RSI for Nvidia Stock')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('RSI')
plt.grid(True)
plt.legend()
plt.show()

# Correlation matrix
correlation_matrix = nvda[['Close', 'Volume', 'Daily_Return', 'SMA_50', 'SMA_200', 'RSI']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True)
plt.gca().xaxis.set_ticks_position('top')
plt.title('Correlation Matrix of Nvidia Stock Metrics')
plt.show()

# Basis Statistical analysis
print("Summary Statistics of Daily Returns:")
print(nvda['Daily_Return'].describe())

# Hypothesis testing
t_stat, p_value = stats.ttest_1samp(nvda['Daily_Return'].dropna(), 0)
print(f"\nOne-sample t-test results:")
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# Risk metrics
annualized_return = nvda['Daily_Return'].mean() * 252
annualized_volatility = nvda['Daily_Return'].std() * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_volatility

print(f"\nRisk Metrics:")
print(f"Annualized Return: {annualized_return:.4f}")
print(f"Annualized Volatility: {annualized_volatility:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

# Drawdowns
nvda['Peak'] = nvda['Close'].cummax()
nvda['Drawdown'] = (nvda['Close'] - nvda['Peak']) / nvda['Peak']

plt.plot(nvda.index, nvda['Drawdown'])
plt.title('Drawdown for Nvidia Stock')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Drawdown')
plt.grid(True)
plt.show()

print(f"\nMaximum Drawdown: {nvda['Drawdown'].min():.4f}")

# Visualisation of cumulative returns
plt.plot(nvda.index, nvda['Cumulative_Return'])
plt.title('Cumulative Returns for Nvidia Stock')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()

# Comparison to benchmarck
# Normalized prices 
nvda['Normalized'] = nvda['Close'] / nvda['Close'].iloc[0]
nasdaq['Normalized'] = nasdaq['Close'] / nasdaq['Close'].iloc[0]

plt.plot(nvda.index, nvda['Normalized'], label="Nvidia (NVDA)")
plt.plot(nasdaq.index, nasdaq['Normalized'], label="NASDAQ-100 (^NDX)")
plt.title("Nvidia vs. NASDAQ Performance")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.ylabel("Normalized Price")
plt.grid(True)
plt.legend()
plt.show()

# Price prediction (Machine Learning)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#prepare data
nvda['Return'] = nvda['Close'].pct_change()
nvda['Lagged_Return'] = nvda['Return'].shift(1)

ml_data = nvda.dropna(subset=['Return', 'Lagged_Return'])

# Features X and target y
X = ml_data['Lagged_Return'].values.reshape(-1, 1)
y = ml_data['Return']

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and evaluation of the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Model Coefficient: {model.coef_[0]:.4f}")
print(f"Model Intercept: {model.intercept_:.4f}")

plt.plot(y_test.index, y_test, label="Actual Returns", color='blue')
plt.plot(y_test.index, y_pred, label="Predicted Returns", color='orange', linestyle='--')
plt.title("Actual vs. Predicted Returns for Nvidia Stocks")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.ylabel("Returns")
plt.grid(True)
plt.legend()
plt.show()

# Monte Carlo Simulation for risk assesment
# Parameters
num_simulations = 1000
num_days = 252
last_price = nvda['Close'].iloc[-1]
mean_return = nvda['Return'].mean()
std_return = nvda['Return'].std()

np.random.seed(42)      # for reproducibility

simulated_prices = np.zeros((num_days, num_simulations))

for sim in range(num_simulations):
    prices = [last_price]
    for day in range(1, num_days):
        random_return = np.random.normal(mean_return, std_return)
        prices.append(prices[-1] * (1 + random_return))
    simulated_prices[:, sim] = prices[:num_days]

for sim in range(num_simulations):
    plt.plot(simulated_prices[:, sim], alpha=0.1, color='blue')
plt.title("Monte Carlo Simulation of Nvidia Stock Price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# Expected price and risk metrics
expected_price = simulated_prices[-1].mean()
price_std = simulated_prices[-1].std()
print(f"Expected Price After 1 Year: ${expected_price:.2f}")
print(f"Risk (Standard deviation of final prices): ${price_std:.2f}")
