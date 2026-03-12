import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

print("All libraries imported successfully.")

#--------------------
#Stage - 2: Define Assets 
#--------------------

tickers = ["LLY", "FCX", "AAPL", "PLD", "JPM", "GLD", "BNO"]

#Time period
start_date = "2019-01-01"
end_date = "2024-12-31"

#--------------------
#Downlaod price data
#-------------------
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

#Keep adjusted close prices only 
prices = data["Close"]

# Drop rows with missing values (important when assets have slightly different histories)
prices = prices.dropna()

# -----------------------------
# Preview the dataset
# -----------------------------
print("Adjusted Close Prices:")
print(prices.head())
print("\nDataset shape:")
print(prices.shape)
print("\nColumns:")
print(prices.columns)

#----------------------------
#Stage - 3: Calculate daily returns
#----------------------------
returns = prices.pct_change().dropna()

print("Daily Returns:")
print(returns.head())

# Average daily return of each asset
mean_returns = returns.mean()

print("\nAverage Daily Returns:")
print(mean_returns)

# Covariance matrix
cov_matrix = returns.cov()

print("\nCovariance Matrix:")
print(cov_matrix)

# Annualized return and covariance
annual_returns = mean_returns * 252
annual_cov_matrix = cov_matrix * 252

print("\nAnnualized Returns:")
print(annual_returns)

print("\nAnnualized Covariance Matrix:")
print(annual_cov_matrix)

# Save outputs
prices.to_csv("prices.csv")
returns.to_csv("daily_returns.csv")
annual_returns.to_csv("annual_returns.csv")
annual_cov_matrix.to_csv("annual_cov_matrix.csv")

print("CSVs saved successfully.")

# -----------------------------
# Stage 4: Portfolio formulas
# -----------------------------

# Number of assets
num_assets = len(tickers)

# Equal-weight portfolio
weights = np.array([1 / num_assets] * num_assets)

print("\nPortfolio Weights:")
print(weights)
print("Sum of weights:", weights.sum())

# Portfolio expected annual return
portfolio_return = np.dot(weights, annual_returns)

print("\nPortfolio Expected Annual Return:")
print(portfolio_return)

# Portfolio annual volatility
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))

print("\nPortfolio Annual Volatility:")
print(portfolio_volatility)

print("\nPortfolio Summary:")
print(f"Expected Annual Return: {portfolio_return:.2%}")
print(f"Annual Volatility: {portfolio_volatility:.2%}")

# -----------------------------
# Stage 5: Simulate portfolios
# -----------------------------

num_portfolios = 5000
results = np.zeros((3, num_portfolios))
weights_record = []
risk_free_rate = 0.02

for i in range(num_portfolios):

   # Random weights (allow short selling)
    weights = np.random.randn(num_assets)
    weights /= np.sum(np.abs(weights))

    weights_record.append(weights)

  # Portfolio return
    portfolio_return = np.dot(weights, annual_returns)

    # Portfolio volatility
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(annual_cov_matrix, weights))
    )

    # Sharpe ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Store results
    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = sharpe_ratio

    print("\nRandom portfolios generated:", num_portfolios)
print("Example Sharpe ratio:", results[2, 0])

# -----------------------------
# Stage 6: Find optimal portfolio
# -----------------------------
max_sharpe_idx = np.argmax(results[2])

optimal_return = results[0, max_sharpe_idx]
optimal_volatility = results[1, max_sharpe_idx]
optimal_sharpe = results[2, max_sharpe_idx]

optimal_weights = weights_record[max_sharpe_idx]

print("\nOptimal Portfolio (Maximum Sharpe Ratio):")
print("Return:", optimal_return)
print("Volatility:", optimal_volatility)
print("Sharpe Ratio:", optimal_sharpe)

print("\nOptimal Weights (%):")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.2%}")

# -----------------------------
# Stage 7: Plot Efficient Frontier
# -----------------------------

# Find minimum volatility portfolio too
min_vol_idx = np.argmin(results[1])
min_vol_return = results[0, min_vol_idx]
min_vol_volatility = results[1, min_vol_idx]

plt.figure(figsize=(10, 6))

# Plot all simulated portfolios
plt.scatter(
    results[1],          # x-axis: volatility
    results[0],          # y-axis: return
    c=results[2],        # color by Sharpe ratio
    cmap='viridis',
    alpha=0.6
)

# Highlight maximum Sharpe portfolio
plt.scatter(
    optimal_volatility,
    optimal_return,
    color='red',
    marker='*',
    s=300,
    label='Maximum Sharpe Portfolio'
)

# Highlight minimum volatility portfolio
plt.scatter(
    min_vol_volatility,
    min_vol_return,
    color='blue',
    marker='*',
    s=300,
    label='Minimum Volatility Portfolio'
)

plt.title("Monte Carlo Efficient Frontier")
plt.xlabel("Annualized Volatility (%)")
plt.ylabel("Annualized Return (%)")
plt.colorbar(label="Sharpe Ratio")
plt.legend()
plt.show()

