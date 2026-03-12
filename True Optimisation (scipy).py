
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
# Scipy Optimization Version
# -----------------------------

risk_free_rate = 0.02
num_assets = len(tickers)

# Portfolio return function
def portfolio_return(weights, annual_returns):
    return np.dot(weights, annual_returns)

# Portfolio volatility function
def portfolio_volatility(weights, annual_cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))

# Negative Sharpe ratio (because scipy minimizes)
def negative_sharpe_ratio(weights, annual_returns, annual_cov_matrix, risk_free_rate):
    p_return = portfolio_return(weights, annual_returns)
    p_volatility = portfolio_volatility(weights, annual_cov_matrix)
    return -((p_return - risk_free_rate) / p_volatility)

# Initial guess
initial_weights = np.array(num_assets * [1.0 / num_assets])

# Constraint
constraints = [
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},     # fully invested
    {'type': 'ineq', 'fun': lambda weights: 1.5 - np.sum(np.abs(weights))}  # leverage limit
]

# Long + short bounds
bounds = tuple((-1, 1) for _ in range(num_assets))

# Run optimizer
optimized_result = minimize(
    negative_sharpe_ratio,
    initial_weights,
    args=(annual_returns, annual_cov_matrix, risk_free_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Extract results
optimal_weights = optimized_result.x
optimal_return = portfolio_return(optimal_weights, annual_returns)
optimal_volatility = portfolio_volatility(optimal_weights, annual_cov_matrix)
optimal_sharpe = (optimal_return - risk_free_rate) / optimal_volatility

# Print results
print("\nScipy Optimized Portfolio (Maximum Sharpe Ratio)")
print("------------------------------------------------")
print("Return:", f"{optimal_return:.2%}")
print("Volatility:", f"{optimal_volatility:.2%}")
print("Sharpe Ratio:", round(optimal_sharpe, 4))

print("\nOptimal Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.2%}")

# Efficient frontier using optimization

target_returns = np.linspace(0.05, 0.35, 50)

frontier_volatility = []
frontier_returns = []

def minimize_volatility(weights, annual_cov_matrix):
    return portfolio_volatility(weights, annual_cov_matrix)

for target_return in target_returns:

    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'eq', 'fun': lambda weights, tr=target_return: portfolio_return(weights, annual_returns) - tr},
        {'type': 'ineq', 'fun': lambda weights: 1.5 - np.sum(np.abs(weights))}
    ]

    result = minimize(
        minimize_volatility,
        initial_weights,
        args=(annual_cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        frontier_volatility.append(result.fun)
        frontier_returns.append(target_return)

# -----------------------------
# Minimum variance portfolio
# -----------------------------
min_vol_result = minimize(
    minimize_volatility,
    initial_weights,
    args=(annual_cov_matrix,),
    method='SLSQP',
    bounds=bounds,
    constraints=[
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'ineq', 'fun': lambda weights: 1.5 - np.sum(np.abs(weights))}
    ]
)

min_vol_weights = min_vol_result.x
min_vol_return = portfolio_return(min_vol_weights, annual_returns)
min_vol_volatility = portfolio_volatility(min_vol_weights, annual_cov_matrix)

# -----------------------------
# Plot everything
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(frontier_volatility, frontier_returns, 'b--', linewidth=2, label='Efficient Frontier')

# Maximum Sharpe portfolio
plt.scatter(
    optimal_volatility,
    optimal_return,
    color='red',
    marker='*',
    s=300,
    label='Maximum Sharpe Portfolio'
)

# Minimum variance portfolio
plt.scatter(
    min_vol_volatility,
    min_vol_return,
    color='green',
    marker='o',
    s=200,
    label='Minimum Variance Portfolio'
)

plt.xlabel("Annualized Volatility")
plt.ylabel("Annualized Return")
plt.title("Efficient Frontier (Optimization)")
plt.legend()
plt.grid(True)
plt.show()
