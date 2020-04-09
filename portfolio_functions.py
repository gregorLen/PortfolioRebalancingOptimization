"""
Thesis "Heuristic Portfolio Rebalancing Optimization Under Transaction Cost"
Gregor Lenhard - Spring 2019
CODE: Portfolio functions
"""
import standard_functions as func
import optimizer_functions as opt
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# reabalance portfolio optimization
# =============================================================================
def rebal_SR(strategy, rSim, w0, T_invest, T_prior, cvar, rS,
             plot_res=False, perfectForesight=False, E=None, V=None):
    """
    Performs a portfolio rebalancing optimization as decribed in the paper.
    """
    nAssets = np.shape(rSim)[1]
    optimize_function = opt.pick_optimizer(strategy)
    PF = create_PF(T_invest, nAssets)
    
    # initial setup
    if perfectForesight == False:
        E, V = calc_E_V(rSim[:T_prior])
      
    if strategy == 2 or strategy == 3: # dx optimizers need initial solution:
        init_x = opt.rebalSR_DE(E, V, T_invest, PF['X'][0], cvar,rS).T 
        PF['X'][0] = optimize_function(E, V, T_invest, init_x, cvar, rS).T
    else:       
        PF['X'][0] = optimize_function(E,V,T_invest,PF['X'][0],0,rS).T

    x_noRebal = PF['X'][0]
    PF['C'][0] = trans_cost_abs(PF['X'][0], cvar, w0)
    PF['w'][0] = w0 - PF['C'][0]
    PF['SR'][0] = sharpeRatio_x(E,V,T_invest, PF['X'][0],0, rS)
    PF['SR_alt'][0] = PF['SR'][0]
    PF['SR_noRebal'][0] = PF['SR'][0]
    
    
    # over time iteration --> t = 0:T_invest
    for t in range(T_invest):
        # preparation
        t_left = T_invest-t    
        
        if perfectForesight == False: # emprical: calc E and V every time
            E, V = calc_E_V(rSim[t:T_prior+t])
        
        # record performance
        PF['X'][t+1] = PF['X'][t] * np.exp(rSim[T_prior+t])
        PF['w'][t+1] = np.ones(nAssets)*PF['w'][t] @ PF['X'][t+1]
        PF['X'][t+1] = updateWeights(PF['X'][t+1])
        
        x_noRebal = updateWeights(x_noRebal * np.exp(rSim[T_prior+t]))
        PF['SR_noRebal'][t+1] = sharpeRatio_x(E,V,t_left,x_noRebal,0,rS)
         
        # find alternative (Optimization)
        x_alt = optimize_function(E, V, t_left, PF['X'][t+1], cvar, rS).T
        dx = x_alt - PF['X'][t+1]
        c_rel = trans_cost_rel(dx, cvar)

        # compare alternative        
        if strategy == 5:
            # for the SLSQP_ic: 0 cost considered --> normal SR
            sr_alt = sharpeRatio_x(E,V, t_left,x_alt.T,0, rS)
        else:
            # for all other cases: SR_eff
            sr_alt = sharpeRatio_x(E,V, t_left,x_alt.T,c_rel, rS)
        sr_cur = sharpeRatio_x(E,V,t_left,PF['X'][t+1],0,rS)
        PF['SR_alt'][t+1] = sr_alt
        delta = sr_alt - sr_cur
        
        # alternative better --> rebalance
        if delta > 0 or strategy == 9: 
                # --> always rebalance in strategy 9 (balanced PF)
            # update alternative
            PF['trade'][t+1] += 1
            PF['X'][t+1] = x_alt
            PF['C'][t+1] = trans_cost_abs(dx, cvar, PF['w'][t+1])
            PF['w'][t+1] = PF['w'][t+1] - PF['C'][t]
            if plot_res == True:
                print(f"rebalanced in period t = {t}")
            PF['SR'][t+1] = sr_alt
            
        else:
            PF['SR'][t+1] = sr_cur

    # evaluate Portfolio
    PF = summarize_PF(PF, strategy, cvar, perfectForesight, rSim, T_prior, T_invest)
    
    # plot results    
    if plot_res == True:
        plot_PF(PF, rSim, T_invest, T_prior)
        
    return PF

# =============================================================================
# Sharpe Ratio of portfolio weights (ex ante)
# =============================================================================
def sharpeRatio_x(E, V, T, x, c_rel=0, rS=0):
    """
    returns the ex ante Sharpe Ratio of portfolio weights and given returns.
    ARGUMENTS:
        E = expected returns
        V = variance-covariance matrix
        x = vector of portfolio weights 
        T = time horizon
        c_rel = relative cost of the portfolio (optional)
        rS = risk-free return (optional)
    RETURNS:
        sr = sharpe ratio
    """
    amort = 250

    if T > amort:
        E_eff = x.T @ E  - rS - c_rel/amort
    else:
        E_eff = x.T @ E - rS - c_rel/T
        
    V = x.T @ V @ x
    STD = np.sqrt(V)
    sr = E_eff/STD  * np.sqrt(T)
        
    return sr

# =============================================================================
# calculates E vector and V matrix
# =============================================================================
def calc_E_V(r):
    """
    returns means (E) and vaciance-covariance matrix (V) from 
    market returns (r)
    """
    E = r.mean(axis=0)
    E.shape = (len(E), 1)
    V = np.cov(r, rowvar=False)  
    
    return E, V
# =============================================================================
# update weights function
# =============================================================================
def updateWeights(X0):
    """
    Updates portfolio weights such that sum is 1.
    ARGUMENTS:
        X0 = a vector OR a matrix of portfolio weights
    RETURNS:
        X = a vector OR a matrix of porflio weights with (row)sum = 1
    """
    X0[X0<0] = 0
    X = X0/np.sum(X0, axis=0)   
    
    return X.T

# =============================================================================
# calculates absolute cost 
# =============================================================================
def trans_cost_abs(dx, cvar=0, w=100):
    """
    returns the absolute cost of a rebalancing transaction based on the 
    delta of portfolio weights.
    ARGUMENTS:
        dx = delta vector of portfolio weights
        cvar = rate of variable transaction cost
        w = wealth
    """

    var = np.sum(np.abs(dx)) * cvar * w

    return var

# =============================================================================
# calculates relative cost 
# =============================================================================
def trans_cost_rel(dx, cvar=0):
    """
    returns the relative transaction cost of a rebalancing transaction
    based on the delta of portfolio weights.
    ARGUMENTS:
        dx = delta vector of portfolio weights
        cvar = rate of variable transaction cost      
    """
    cost = np.sum(np.abs(dx)) * cvar 
    
    return cost

# =============================================================================
# add ris free asset
# =============================================================================
def add_rS(rS, r, E, V):
    """
    manually adds a risk-free asset to the expected retun vector, the variance- 
    covariance matrix and the return matrix.
    ARGMUMENTS:
        rS = risk-free rate of return
        r = return matrix
        E = expected return vector
        V = variance-covariancematrix
    """
    r = np.insert(r, len(E), rS, axis=1)
    E = np.vstack((E, rS))
    V = np.insert(np.insert(V,len(V),0, axis=0),len(V),0, axis=1)
    
    return r, E, V
        
# =============================================================================
# Simulate a Portfolio 
# =============================================================================
def simPortfolio(x, r, w0=100):
    """
    Simulates a Portfolio time series based on two inputs
    ARGUMENTS:
        x = initial portfolio weights
        r = simulated returns 
    RETURNS:
        yP = time series of a portfolio
    """
    #x.shape = (1,len(x))
    X = x * np.exp(np.cumsum(np.insert(r, 0, 0, axis=0), axis = 0))
    yP = w0 * np.sum(X, axis=1)
           
    return yP

# =============================================================================
# Create an empty portfolio
# =============================================================================
def create_PF(T, nAssets):
    """
    returns an empty portfolio.
    ARGUMENTS:
        T = time horizon of the portfolio
        nAssets = number of assets
    RETUNRS:
        PF = Portfolio including weight matrix (X), wealth vector(w), cost 
        vector (C), sharpe ratio vector (SR), alternative-sharpe ratio (SR_alt),
        no-rebal sharpe ratio (SR_noRebal)
    """
    PF = dict(
        X = np.zeros((T+1, nAssets)),      # portfolio weight matrix
        w = np.zeros((T+1, 1)),            # investor wealth vector
        C = np.zeros((T+1, 1)),            # cost vector    
        SR = np.zeros((T+1, 1)),           # sharpe ratio of the portfolio
        SR_alt = np.zeros((T+1, 1)),        
        SR_noRebal = np.zeros((T+1, 1)),
        trade = np.zeros((T+1, 1))
        )
        
    return PF

# =============================================================================
# plot portfolio results
# =============================================================================
def plot_PF(PF, rSim, T_invest, T_prior):
    """
    Plots results of a Portfolio simulation. 
    ARGUMENTS:
        PF = portfolio
        rSim = simulated returns of the market
        T_invest = investment horizon
        T_prio = lead time
    """
    x_vals = np.arange(len(PF['X']))
    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (10,15))
    ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()
    ax1.grid(); ax3.grid(); ax4.grid()
    
    ax1.plot(100 * np.exp(np.cumsum(np.insert(rSim[T_prior:], 0, 0, axis=0),
                                    axis = 0)), linewidth = 0.8)
    ax1.set_title("Stock Market")
    #ax1.set_yscale('log')
    
    ax2.stackplot(x_vals, PF['X'].T)
    ax2.set_title("Portfolio Weights")
    xcoords = np.where(PF['C'] > 0)[0]
    for xc in xcoords:
        ax2.axvline(x=xc, color='k', linestyle='-', linewidth = .3)
    
    ax3.plot(x_vals, np.hstack([PF['w'], PF['w_gross'], PF['w_noRebal']]))
    ax3.set_title("Porfolio Performance")
    ax3.legend(["Rebalancing (Net)","Rebalancing (gross)","No Rebalancing"])
    ax3.axhline(y=PF['w'][0], linewidth=1, color = "k", linestyle = ":")
       
    ax4.plot(x_vals, np.hstack([PF['SR'], PF['SR_alt'], PF['SR_noRebal']]))
    ax4.set_title(f"SR rebal = {np.round_(PF['sr_Rebal'],2)} // SR noRebal = {np.round_(PF['sr_noRebal'],2)}")
    ax4.legend(["SR Portfolio","SR alternative", "SR noRebal"])
    
    plt.show()
    
    print("Sharpe Ratio of the rebalanced portfolio is",
          f"{np.round_(PF['sr_Rebal'],2)}")
    print("Sharpe Ratio of the unrebalanced portfolio is",
          f"{np.round_(PF['sr_noRebal'],2)}")
    print("Return of the portfolio is",
          f"{100*(PF['w'][-1]/PF['w'][0]-1)/(len(PF['w'])/250)} % p.a.")
    print(f"{PF['trades']} trades")


# =============================================================================
# Summarize Portfolio (at the end)
# =============================================================================
def summarize_PF(PF, strategy, cvar, perfectForesight, rSim, T_prior, T_invest):
    PF['w_gross'] = PF['w'] + np.cumsum(PF['C'], axis=0)
    PF['sr_Rebal'], PF['mu_Rebal'], PF['sigma_Rebal'] = func.sharpeRatio_ts(PF['w'])
    PF['w_noRebal'] = simPortfolio(PF['X'][0], rSim[T_prior:], PF['w'][0])  
                        # start portfolio without rebal
    PF['w_noRebal'].shape = (T_invest+1,1)   
    PF['sr_noRebal'], PF['mu_noRebal'], PF['sigma_noRebal'] = func.sharpeRatio_ts(
                                                                PF['w_noRebal'])
    PF['strategy'] = strategy
    PF['cvar'] = cvar
    PF['perfFore'] = perfectForesight
    PF['final_w'] = float(PF['w'][-1])
    PF['final_w_noRebal'] = float(PF['w_noRebal'][-1])
    PF['cost'] = np.sum(PF['C'])
    PF['trades'] = np.sum(PF['trade'] > 0)
    PF['gini'] = giniX(PF['X'])
    PF['hhi'] = hhiX(PF['X'])
    
    return PF

# =============================================================================
# Gini Coefficient of Portfolio weights
# =============================================================================
def giniX(X):
    gin = []
    for i in range(len(X)):
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(X[i], X[i])).mean()
        # Relative mean absolute difference
        rmad = mad/np.mean(X[i])
        # Gini coefficient
        gin.append(0.5 * rmad)
     
    return np.mean(gin) 

def hhiX(X):
    hhi_vals = []
    for i in range(len(X)):
        n = len(X[i])
        hhi = sum(X[i]**2)
        norm_hhi = (hhi-1/n)/(1-1/n) 
        hhi_vals.append(norm_hhi)
    return np.mean(norm_hhi)
