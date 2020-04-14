# Portfolio Rebalancing Optimization
A stock price simulation that optimizes the rolling net sharpe ratio of a portfolio. The model supports several heuristic optimization algorithms.

## 1. Features
#### Sandbox
`main_rebalancing.py` is a little sandbox that let's you play around with the parameters. The underlining data is from the german stock market (`germanstocks.csv`). The data is being bootstrapped to ensure a different outcome for every simulation. The objective function is the effective (net) sharpe ratio _after transaction costs_. 
#### Experimental Setup
`experiment_rebalancing.py` is an experimental design to try several hyperparameters. Multi processing is included. The script writes a `.csv` and evaluates several performance measures.
#### Model Parameters
Currently the implementation support several parameters.
- `strategy` is the algorithm that optimizes the objective function which is the effective (or net) sharpe ratio. Possible values are:
    1. [Differential Evolution](https://link.springer.com/article/10.1023/A:1008202821328) Alorithm on the portfolio weights `x`
    2. Differential Evolution Algorithm on the delta of portfolio weights `dx`
    3. A simple neighborhood search
    4. [Sequential Least Squares Programming](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html) from scipy
    5. Equally weightd portfolio strategy
- `perfectForesight` determines if the covariance matrix is estimated based on the true _data generating process_ (`True`) or on past observations (`False`).
- `rS` is the interest (daily) rate of the risk-free asset.
- `w0` is the starting budget.
- `T_prior` is the rolling time horizon used for the estimation of the covariance matrix.
- `T_invst` is the investment horizon. 
- `BL` is the block length for the bootstrap.
- `cvar` is the amount of variable transaction costs.
 
#### Examples
Example of a three year portfolio simulation based on Sequential Least Squares Programming optimization. The plot shows:
( a ) Stock market prices
( b ) Portfolio weights over time
( c ) portfolio balance the dynamic portfolio (net/gross) and a portfolio without rebalancing.
( d ) Rolling sharpe ratio w.r.t. to marturity
![example_SLSQP](https://github.com/gregorLen/PortfolioRebalancingOptimization/blob/master/img/example_SLSQP.png?raw=true)

Example of an "equally weighted portfolio"-strategy:
![example_EW](https://github.com/gregorLen/PortfolioRebalancingOptimization/blob/master/img/example_ew.png?raw=true)


## 2. Dependencies
- numpy 
- matplotlib 
- pandas 
- scipy

To install requirements,  `cd` to the directory of the repository and run `pip install -r requirements.txt`. A virtual environment is recommended. 

## 3. TO DO 
- other evolutionary optimization algorithms, i.e. [Particle Swarm Optimization](https://www.sciencedirect.com/topics/engineering/particle-swarm-optimization)
- other objective functions 
---
## 4. Contact
I am very thankful for any kind of feedback. Also, if you have questions, please contact gregor.lenhard@unibas.ch .