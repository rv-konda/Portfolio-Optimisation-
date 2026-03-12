# Portfolio Optimization in Python
Monte Carlo Simulation and SciPy-Based Markowitz Optimization


**OBJECTIVE** - Implement the Markowitz Portfolio Theory using (i) Monte Carlo simulation and (ii) Optimisation using SciPy. 
**TARGET ASSETS** - The portfolio consists of the following assets: Eli Lilly (LLY), Freeport-McMoRan (FCX), Apple (AAPL), Prologis Inc (PLD), JPMorgan (JPM), SPDR Gold Trust (GLD) & United States Brent Oil Fund (BNO). 
_Rationale:_ Choose a diversified portfolio to observe the comovements of these assets while holding a preference for gold (safety), copper (growth) and brent oil (test a short strategy). 

**TIME PERIOD** - 2019 - 2024 (approx. 1250 trading days)
This time period captures several economic regimes making it apt for recognising different patterns. 
2019 : late cycle expansion
2020 : COVID crash + extreme volatility 
2021 : Stimulus driven bull market
2022 : Inflation shock + Aggressive rate hikes
2023 - 2024 : Tighter monetary policy + geopolitical shocks 

**_THE EFFICIENT PORTFOLIO PROBLEM_**
The classical Markowitz Mean-Variance problem optimisation is: _Determining the lowest risk portfolio that achieves a specific return._
The code sets a return range of 5% - 35% for the chosen assets.
The **Monte Carlo Simulation** generates 5000 random portfolio with different weights (sum to 1) to study their price data. The code further calculated their annualised return and covariance matrix. 
*The code initially uses an equal weight portfolio. 
*While generating random weights in the simulation, the code also allows the possibility for short selling (as inferred by a negative weight).

The code then uses the generated data to plot an efficient fontier of the portfolios that generate that particular return at the least level of volatility. 

**_THE SHARPE RATIO PROBLEM_**
The Sharpe ratio determines which portfolio gives the highest return per unit of risk. 
The code uses minimise (-Sharpe ratio) to find the portfolio with the highest Sharpe ratio. 

We use the Sharpe ratio to determine which portfolio on the genrated efficient fontier has the highst return per unit of risk. 

In the **optimisation version** the code solves for the most efficient portfolio mathematically without requiring the simulation process. 
This approach is more robust than the Monte Carlo simulation because the optimiser systematically searches for the optimal weight distribution rather than randomly sampling portfolios.

**RESULTS**
**_Equal Weights ( 0.14285714 each)_**
Expected Annual Return: 25.83%
Annual Volatility: 20.41%

**_Optimal weights using the Monte Carlo simulation_**
Optimal Portfolio (Maximum Sharpe Ratio):
Return: 26.94%
Volatility: 17.21%
Sharpe Ratio: 1.450

Optimal Weights (%):
LLY: 26.04%
FCX: 7.84%
AAPL: 0.23%
PLD: 6.73%
JPM: 8.41%
GLD: 39.63%
BNO: -11.11%

**_Optimal Portfolio using SciPy_**
Return: 27.62%
Volatility: 16.03%
Sharpe Ratio: 1.598

Optimal Weights:
LLY: 28.16%
FCX: 1.52%
AAPL: 0.48%
PLD: 45.66%
JPM: 11.53%
GLD: 30.72%
BNO: -18.07%

**FURTHER IMPROVEMENTS**
-> Use a more narrow target return range. 
-> Backtest the optimal portfolios with out of sample data to check if they are relevant with later developments. 



