import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# -----------------------------
# Step 1: Define assets
# -----------------------------
tickers = ["LLY", "FCX", "AAPL", "PLD", "JPM", "GLD", "BNO", "SPY"]
asset_tickers = ["LLY", "FCX", "AAPL", "PLD", "JPM", "GLD", "BNO"]
benchmark_ticker = "SPY"

# -----------------------------
# Step 2: Define periods
# -----------------------------
train_start = "2019-01-01"
train_end = "2023-12-31"

test_start = "2024-01-01"
test_end = "2026-03-12"

full_start = train_start
full_end = test_end

# -----------------------------
# Step 3: Download data
# -----------------------------
data = yf.download(tickers, start=full_start, end=full_end)

prices = data["Close"].dropna()

asset_prices = prices[asset_tickers]
benchmark_prices = prices[benchmark_ticker]

print("\nAsset Price Data Preview:")
print(asset_prices.head())

print("\nBenchmark Price Data Preview:")
print(benchmark_prices.head())

print("\nAsset Price Data Tail:")
print(asset_prices.tail())
print("\nBenchmark Price Data Tail:")
print(benchmark_prices.tail())

# -----------------------------
# Step 4: Split into train and test
# -----------------------------
train_prices = asset_prices.loc[train_start:train_end]
test_prices = asset_prices.loc[test_start:test_end]

print("\nTraining Price Data Shape:")
print(train_prices.shape)

print("\nTest Price Data Shape:")
print(test_prices.shape)

# Compute returns
train_returns = train_prices.pct_change().dropna()
test_returns = test_prices.pct_change().dropna()

print("\nTraining Returns Preview:")
print(train_returns.head())

print("\nTest Returns Preview:")
print(test_returns.head())

# -----------------------------
# Step 5: Optimize on training data
# -----------------------------
risk_free_rate = 0.02
num_assets = len(asset_tickers)

# Annualized training inputs
train_annual_returns = train_returns.mean() * 252
train_annual_cov_matrix = train_returns.cov() * 252

print("\nTraining Annual Returns:")
print(train_annual_returns)

print("\nTraining Annual Covariance Matrix:")
print(train_annual_cov_matrix)

# Portfolio return function
def portfolio_return(weights, annual_returns):
    return np.dot(weights, annual_returns)

# Portfolio volatility function
def portfolio_volatility(weights, annual_cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))

# Negative Sharpe ratio
def negative_sharpe_ratio(weights, annual_returns, annual_cov_matrix, risk_free_rate):
    p_return = portfolio_return(weights, annual_returns)
    p_volatility = portfolio_volatility(weights, annual_cov_matrix)
    return -((p_return - risk_free_rate) / p_volatility)

# Initial equal weights
initial_weights = np.array(num_assets * [1 / num_assets])

# Constraints: fully invested
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

# Long-short bounds (same spirit as your earlier project)
bounds = tuple((-1, 1) for _ in range(num_assets))

# Run optimizer on TRAINING data only
optimized_result = minimize(
    negative_sharpe_ratio,
    initial_weights,
    args=(train_annual_returns, train_annual_cov_matrix, risk_free_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Extract optimized weights
optimal_weights = optimized_result.x

# Training-period optimized portfolio stats
optimal_train_return = portfolio_return(optimal_weights, train_annual_returns)
optimal_train_volatility = portfolio_volatility(optimal_weights, train_annual_cov_matrix)
optimal_train_sharpe = (optimal_train_return - risk_free_rate) / optimal_train_volatility

print("\nOptimized Training Portfolio:")
print(f"Return: {optimal_train_return:.2%}")
print(f"Volatility: {optimal_train_volatility:.2%}")
print(f"Sharpe Ratio: {optimal_train_sharpe:.4f}")

print("\nOptimized Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.2%}")

# -----------------------------
# Step 6: Backtest on test period
# -----------------------------

# Daily portfolio returns in the test period
test_portfolio_returns = test_returns.dot(optimal_weights)

print("\nTest Portfolio Returns Preview:")
print(test_portfolio_returns.head())

# Growth of $1 invested
portfolio_cumulative = (1 + test_portfolio_returns).cumprod()

print("\nPortfolio Cumulative Performance Preview:")
print(portfolio_cumulative.head())

# Basic backtest stats
total_return = portfolio_cumulative.iloc[-1] - 1
annualized_return = (portfolio_cumulative.iloc[-1]) ** (252 / len(test_portfolio_returns)) - 1
annualized_volatility = test_portfolio_returns.std() * np.sqrt(252)
backtest_sharpe = (annualized_return - risk_free_rate) / annualized_volatility

print("\nBacktest Results (2024 to date):")
print(f"Total Return: {total_return:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_volatility:.2%}")
print(f"Sharpe Ratio: {backtest_sharpe:.4f}")


# -----------------------------
# Step 7: Compare against equal-weight and SPY
# -----------------------------

# Equal-weight portfolio on test period
equal_weights = np.array([1 / len(asset_tickers)] * len(asset_tickers))
equal_weight_returns = test_returns.dot(equal_weights)
equal_weight_cumulative = (1 + equal_weight_returns).cumprod()

# SPY benchmark on test period
benchmark_test_prices = benchmark_prices.loc[test_start:test_end]
benchmark_returns = benchmark_test_prices.pct_change().dropna()
benchmark_cumulative = (1 + benchmark_returns).cumprod()

# Align dates so all series match
comparison_df = pd.concat(
    [
        portfolio_cumulative.rename("Optimized Portfolio"),
        equal_weight_cumulative.rename("Equal-Weight Portfolio"),
        benchmark_cumulative.rename("SPY")
    ],
    axis=1
).dropna()

print("\nComparison Preview:")
print(comparison_df.head())

# Plot
plt.figure(figsize=(12, 7))
plt.plot(comparison_df.index, comparison_df["Optimized Portfolio"], label="Optimized Portfolio", linewidth=2)
plt.plot(comparison_df.index, comparison_df["Equal-Weight Portfolio"], label="Equal-Weight Portfolio", linewidth=2)
plt.plot(comparison_df.index, comparison_df["SPY"], label="SPY", linewidth=2)

plt.title("Out-of-Sample Backtest: Optimized Portfolio vs Benchmarks")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Step 8: Performance summary table
# -----------------------------

# Optimized portfolio stats
opt_total_return = portfolio_cumulative.iloc[-1] - 1
opt_annual_return = portfolio_cumulative.iloc[-1] ** (252 / len(test_portfolio_returns)) - 1
opt_annual_vol = test_portfolio_returns.std() * np.sqrt(252)
opt_sharpe = (opt_annual_return - risk_free_rate) / opt_annual_vol

# Equal-weight portfolio stats
eq_total_return = equal_weight_cumulative.iloc[-1] - 1
eq_annual_return = equal_weight_cumulative.iloc[-1] ** (252 / len(equal_weight_returns)) - 1
eq_annual_vol = equal_weight_returns.std() * np.sqrt(252)
eq_sharpe = (eq_annual_return - risk_free_rate) / eq_annual_vol

# SPY stats
spy_total_return = benchmark_cumulative.iloc[-1] - 1
spy_annual_return = benchmark_cumulative.iloc[-1] ** (252 / len(benchmark_returns)) - 1
spy_annual_vol = benchmark_returns.std() * np.sqrt(252)
spy_sharpe = (spy_annual_return - risk_free_rate) / spy_annual_vol

# Build summary table
performance_table = pd.DataFrame({
    "Total Return": [opt_total_return, eq_total_return, spy_total_return],
    "Annualized Return": [opt_annual_return, eq_annual_return, spy_annual_return],
    "Annualized Volatility": [opt_annual_vol, eq_annual_vol, spy_annual_vol],
    "Sharpe Ratio": [opt_sharpe, eq_sharpe, spy_sharpe]
}, index=["Optimized Portfolio", "Equal-Weight Portfolio", "SPY"])

print("\nPerformance Summary Table:")
print(performance_table)

# Optional prettier percentage display
performance_table_display = performance_table.copy()
for col in ["Total Return", "Annualized Return", "Annualized Volatility"]:
    performance_table_display[col] = performance_table_display[col].map(lambda x: f"{x:.2%}")
performance_table_display["Sharpe Ratio"] = performance_table_display["Sharpe Ratio"].map(lambda x: f"{x:.4f}")

print("\nFormatted Performance Summary:")
print(performance_table_display)


# -----------------------------
# Step 9: Optimized weights bar chart
# -----------------------------
weights_df = pd.Series(optimal_weights, index=asset_tickers)

plt.figure(figsize=(10, 6))
weights_df.sort_values().plot(kind="bar")
plt.title("Optimized Portfolio Weights")
plt.xlabel("Assets")
plt.ylabel("Weight")
plt.grid(True, axis="y")
plt.show()