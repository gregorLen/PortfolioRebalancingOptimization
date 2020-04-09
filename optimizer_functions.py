# -*- coding: utf-8 -*-
"""
Thesis "Heuristic Portfolio Rebalancing Optimization Under Transaction Cost"
Gregor Lenhard - Spring 2019
CODE: Portfolio Optimizers
"""
import numpy as np
import standard_functions as func
import portfolio_functions as pf
from scipy import optimize
# =============================================================================
# Pick optimizer to find portfolio weights
# =============================================================================
def pick_optimizer(i):
    """
    picks the method to optimize portfolio weights
    """
    return {1: rebalSR_DE,
            2: rebalSR_DE_dx,
            3: rebalSR_NH,
            4: rebalSR_SLSQP,
            5: rebalSR_SLSQP_ignore_cost,
            9: rebal_EW,
            }[i]

# =============================================================================
# SLSQP_ic
# =============================================================================
    """
    Performs an optimization via Sequential Least Squares 
    Programming (SLSQP). Objective function is the Sharpe ratio (fSR(x)).
    """
def rebalSR_SLSQP_ignore_cost(E,V,T=None,x=None, cvar=None, rS=0):
     ### SCIPY PACKAGE
    n = len(E)
    
    # objective function
    def fSR(x):
        EP_eff = x.T @ E - rS 
        VP = np.sqrt(x.T @ V @ x)
        sr = -EP_eff/VP 
        return sr
    
    # non-negativity constraint
    non_neg = []
    for i in range(n):
        non_neg.append((0,None))
    non_neg = tuple(non_neg)
    
    
    x_init = np.ones(n)/n
    # sum(x) == 1
    cons = ({'type': 'eq', 'fun': lambda x:  np.dot(x, np.ones(n))-1})
    
    # solve problem
    res = optimize.minimize(fSR, x_init, method='SLSQP', 
                            bounds=non_neg, constraints=cons, 
                            options={'disp':False})
    xOpt = res.x
    xOpt.shape = (n,1)  
    
    return xOpt
    
# =============================================================================
# SLSQP
# =============================================================================
def rebalSR_SLSQP(E,V,T=None,x=None, cvar=None, rS=0):
    """
    Performs an optimization via Sequential Least Squares 
    Programming (SLSQP). Objective function is the effective Sharpe ratio 
    (fSR(x)).
    """
    x_cur = np.copy(x)
    n = len(E)
    if T > 250:
        amort = 250
    else:
        amort = T
    
    # objective function
    def fSR(x):
        c_rel = np.sum(np.abs(x-x_cur)) * cvar
        EP_eff = x.T @ E - rS - c_rel/amort
        VP = np.sqrt(x.T @ V @ x)
        sr = -EP_eff/VP 
        return sr
    
    # non-negativity constraint
    non_neg = []
    for i in range(n):
        non_neg.append((0,None))
    non_neg = tuple(non_neg)
    
    
    x_init = np.ones(n)/n
    # sum(x) == 1
    cons = ({'type': 'eq', 'fun': lambda x:  np.dot(x, np.ones(n))-1})
    
    # solve problem
    res = optimize.minimize(fSR, x_init, method='SLSQP', 
                            bounds=non_neg, constraints=cons, 
                            options={'disp':False})
    xOpt = res.x
    xOpt.shape = (n,1)
    
    return xOpt

# =============================================================================
# balanced portfolio
# =============================================================================
def rebal_EW(E,V,T=None,x=None, cvar=None, rS=0):
    """
    This function calculates balanced portfolio of equal portfolio weights
    """
    n = len(E)
    xOpt = np.ones((n,1))/n
    
    xOpt.shape = (n,1)
    
    return xOpt

# =============================================================================
# DE portfolio rebalancing optimization - Support functions
# =============================================================================
def eval_population(E,V,T,x,xPop,cvar=0,rS=0):
    """
    Support function for the Differcial Evolution Algorithm. Evaluates the 
    population.    
    """
    amort = 250
    popSize = np.shape(xPop)[0]
    srPop = np.zeros((popSize))
      
    dxPop = xPop-x

    c_rel = np.sum(np.abs(dxPop), axis=1) * cvar
    c_rel.shape = (popSize,1)
    
    if T > amort:
        EP = xPop @ E - rS - c_rel/amort
    else:
        EP = xPop @ E - rS - c_rel/T
    
    EP.shape = (popSize) 
    VP= np.sqrt(np.diag(xPop @ V @ xPop.T))
    
    srPop = EP/VP * np.sqrt(T)
        
    return srPop

# =============================================================================
# DE_x portfolio rebalancing optimization SR
# =============================================================================
def rebalSR_DE(E,V,T,x=0, cvar=0, rS=0):
    """
    Performs an optimization based on the Differential Evolution algorithm
    with respect to the portfolio vector x. Objective function is the 
    effective Sharpe ratio.
    """
    # set parameters
    popSize = 20
    nG = 200   # number of generations
    F = 0.8
    CR = 0.3
    D = len(E)   # dimensions
    # =========================================================================
    # randomly initialize population
    xPop = pf.updateWeights(np.random.rand(popSize, D).T)    
    # evaluate initial population   
    srPop = eval_population(E,V,T,x,xPop,cvar,rS) 
    # find current optimum   
    xOpt = np.copy(xPop[np.argmax(srPop),:])
    srOpt = max(srPop)
    
    # iteration over generations nG
    for gen in range(nG):
        # produce offsprings
        p1 = np.random.permutation(popSize)
        p2 = np.random.permutation(popSize)
        p3 = np.random.permutation(popSize)
        xCross = xPop[p1,:] + F * (xPop[p2,:] -  xPop[p3,:])
        crossover = np.random.rand(popSize,D) <= CR
        
        xN = np.copy(xPop)
        xN[crossover] = xCross[crossover] 
        xN = pf.updateWeights(xN.T)
        
        # evaluate offsprings
        srN = eval_population(E,V,T,x,xN,cvar,rS)
        
        # find new population
        repl = srN > srPop
        xPop[repl,:] = np.copy(xN[repl,:])
        srPop[repl] = np.copy(srN[repl])
        # check for new global optimum
        srEl = max(srPop)
        if srEl > srOpt:
            srOpt = np.copy(srEl)
            xOpt = np.copy(xPop[np.argmax(srPop),:])
                
    return xOpt



# =============================================================================
# DE_dx portfolio rebalancing optimization
# =============================================================================
def rebalSR_DE_dx(E,V,T,x=None,cvar=0,rS=0):
    """
    Performs an optimization based on the Differential Evolution algorithm
    with respect to the change in portfolio weights dx. Objective function is 
    the effective Sharpe ratio.
    """
    # set parameters
    popSize = 20
    nG = 150
    F = 0.9
    CR = 0.3
    D = len(E)   # dimensions
    # =========================================================================
    # initialize
    X = np.zeros((popSize, D)) + x
    dxPop = np.random.rand(popSize,D) * 0.05 - 0.025
    srPop = np.zeros((popSize))
    xPop = pf.updateWeights((X + dxPop).T)
    
    # evaluate initial populaiton
    srPop = eval_population(E,V,T,x,xPop,cvar,rS)
   
    # evaluation over generations
    for gen in range(nG):
        # produce offsprings
        p1 = np.random.permutation(popSize)
        p2 = np.random.permutation(popSize)
        p3 = np.random.permutation(popSize)
        dxCross = dxPop[p1,:] + F * (dxPop[p2,:] - dxPop[p3,:])        
        
        crossover = np.random.rand(popSize,D) <= CR
        dxN = np.copy(dxPop)
        dxN[crossover] = dxCross[crossover]  
        xN = pf.updateWeights((X + dxN).T)

        
        # evaluate offsprings
        srN = eval_population(E,V,T,x,xN,cvar,rS)           
            
        # find new population
        repl = srN > srPop
        dxPop[repl,:] = np.copy(dxN[repl,:])
        xPop[repl,:] = np.copy(xN[repl,:])
        srPop[repl] = np.copy(srN[repl])
    
    # return optimal new portfolio weights    
    dxOpt = np.copy(dxPop[np.argmax(srPop),:])
    xOpt = pf.updateWeights(x + dxOpt)
    
    xOpt.shape = (D,1)

    return xOpt

# =============================================================================
# DE portfolio rebalancing optimization
# =============================================================================
def rebalSR_NH(E,V,T,x=None,cvar=None,rS=0):
    """
    Performs an optimization based on a Neighborhoodsearch (NH).
    with respect to the change in portfolio weights vector dx. Objective 
    function is the effective Sharpe ratio.
    """
    # set parameters    
    tau = 0.3         #threshold
    popSize = 20   # population size
    trade_ratios = [1,  # SELL
                    1,  # BUY
                    1]  # HOLD
    
    # initialize
    D = len(E)                # dimensions
    nR = int(np.sqrt(D) * 10)   # number of rounds
    trade_decision = np.floor(np.cumsum(trade_ratios/np.sum(trade_ratios)) 
                                                        	* D).astype(int)
    sell_elements = np.arange(0, trade_decision[0])
    buy_elements = np.arange(trade_decision[0], trade_decision[1])
    nSell = len(sell_elements)
    nBuy = len(buy_elements)
    elem = np.arange(D)
    
    # initial population = current best solution
    xPop = np.zeros((popSize, D)) + x
    srPop = eval_population(E,V,T,x,xPop,cvar,rS)
    xOpt = np.copy(x)
    srOpt = max(srPop)
    xN = np.copy(xPop)
    # =========================================================================
    for R in range(nR):
        threshold = tau - (R/nR)*tau  # threshold decreases for each round
        for i, solution in enumerate(xN):
            # randomly select buy and sell stocks
            np.random.shuffle(elem)
            i_sell = elem[sell_elements]
            i_buy = elem[buy_elements]
            #i_hold = elem[hold_elements]
            
            # perform trades    
            sell_trade = solution[i_sell] * np.random.rand(nSell)
            trade_amount = sum(sell_trade)
            buy_trade = pf.updateWeights(np.random.rand(nBuy)) * trade_amount
            solution[i_sell] -= sell_trade
            solution[i_buy] += buy_trade
            
        # evaluate offspring
        srN = eval_population(E,V,T,x,xN,cvar,rS)
        
        # find new population
        delta = srN - srPop
        repl = delta < threshold 
        #replace solutions
        xPop[repl,:] = np.copy(xN[repl,:])
        srPop[repl] = np.copy(srN[repl])
        # check for new global optimum
        srEl = max(srPop)
        if srEl > srOpt:
            srOpt = np.copy(srEl)
            xOpt = np.copy(xPop[np.argmax(srPop),:])
    xOpt.shape = (D,1)

    return xOpt