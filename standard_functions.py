#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis "Heuristic Portfolio Rebalancing Optimization Under Transaction Cost"
Gregor Lenhard - Spring 2019
CODE: Support Functions
"""
import numpy as np
import pandas as pd
# =============================================================================
#  Geometric Brownian Motion
# =============================================================================
def GBM(S0=100, mu=0.05, sigma=0.2, T=1, M=250, nP=1, dataframe=False):
    """
    Geometric Brownian Motion
    S0 = Starting Value 
    mu = anualized return 
    sigma = volatility
    T = sampling period in years 
    M = nuber of periods
    dataframe = True if you motion shall be stored in a pd.DataFrame
    """
    dt = T/M
    r = np.random.randn(M, nP) * sigma * np.sqrt(dt) + (mu-(np.power(sigma,2))/2)*dt
    S = S0 * np.exp(np.cumsum(np.insert(r, 0, 0, axis=0), axis = 0))
    
    if dataframe==True:
        S = pd.DataFrame(S)
    
    return S , r

# =============================================================================
#  Geometric Brownian Motion
# =============================================================================
def GBM_PF(r, T=100, dataframe=False):
    """
    Geometric Brownian Motion Porfolio returns based on an input portfolio
    of assets.
    INPUT
        r = returns of assets
        T = length of simulation
        dataframe = True if you motion shall be stored in a pd.DataFrame
    OUTPUT
        rSim = simulated returns
    """
    mu = r.mean(axis=0)
    sigma = np.cov(r, rowvar=False)
    
    nAssets=r.shape[1]
    
    rSim = np.random.randn(T, nAssets) @ np.linalg.cholesky(sigma).T + mu
    
    if dataframe==True:
        rSim = pd.DataFrame(rSim)
        
    mu.shape = ((nAssets, 1))
    
    return rSim , mu, sigma#, SSim
# =============================================================================
#  Bootstrap 
# =============================================================================
def bootstrap_r(r, n=100, BL=10):
    """
    bootstrapPrice() takes a time series of returns as input and returns 
    bootstrapped prices and returns.
    
    ARGUMENTS
        r = Series or DataFrame of returns
        N = length of sample 
        BL = block length
    RETURNS
        ySim = bootstrapped time series of prices
        rSim = bootstrapped time series of returns
    """
    # r = pd.DataFrame(r)
    N = np.shape(r)[0]
    length = n/BL   
    i_resample = np.array(np.array([np.floor(np.random.rand(int(length))*(N-BL))]).T 
                          + np.array(range(BL))).astype(int)
    i_resample.shape = (n)
    rSim = r[i_resample]       
    ySim = 100*np.exp(np.cumsum(np.insert(rSim, 0, 0, axis=0), axis=0))
    
    return ySim, rSim

# =============================================================================
#  Bootstrap Price
# =============================================================================
def bootstrap_y(y, n=100, BL=10):
    """
    bootstrapPrice() takes a time series of asset prices as input and returns 
    bootstrapped Prices and Returns.
    
    ARGUMENTS
        y = Series or DataFrame of Asset Prices
        n = length of simulated 
        BL = block length
    RETURNS
        ySim = bootstrapped time series of prices
        rSim = bootstrapped time series of returns
    """
    # y = pd.DataFrame(y)
    r = np.diff(np.log(y), axis=0)
    
    ySim, rSim = bootstrap_r(r, n, BL)
    
    return ySim, rSim

# =============================================================================
# Sharpe Ratio of a time series (ex post)
# =============================================================================
def sharpeRatio_ts(ts, T=None, c_rel=0, amort=None, rS=0):
    """
    Calculates the ex ante SharpeRatio of a daily time series.
    Costs are being distributed over time (T)
    """
    rS = rS / 252
    r = np.diff(np.log(ts), axis=0)
  
    if T == None:
        T = len(ts)
    if amort == None:
        amort = T
    
    E = r.mean()
    
    if T > amort:
        E_eff = E - c_rel/amort
    else:
        E_eff = E - c_rel/T
    
    STD = r.std()
    sr = (E_eff-rS)/STD  * np.sqrt(T)
    
    return sr, E, STD