"""
MA: "Heuristic Portfolio Rebalancing Optimization Under Transaction Cost"
Gregor Lenhard - Spring 2019
CODE: Main Code
"""
import os
os.chdir(os.path.dirname(os.path.realpath('__file__')))
import numpy as np
import pandas as pd
import standard_functions as func
import portfolio_functions as pf
import time
# =============================================================================
# == Data
setting = {'plot_data': True,
           'data': "germanstocks.csv",
           'i_stock' : np.arange(25),
           'i_date': np.arange(0,3739)           
           }
# == Parameters
para = {'strategy' : 1,
    # 1 = DEx; 2 = DEdx; 3 = NH; 4 = SLSQP, 5 = SLSQP_ignore_cost, 9 = EW
        'perfectForesight' : False,
        'rS' : 0.01/250,
        'w0' : 50000,
        'T_prior' : 1250,
        'T_invest' : 750,
        'BL' : 10,
        # cost
        'cvar' : 0.02
        }
# == Random Seed
# =============================================================================
# simulate data
np.random.seed(1)
df = pd.read_csv(setting['data'], sep=";", index_col=0)
y = df.iloc[setting['i_date'], setting['i_stock']]
r = np.diff(np.log(y), axis=0)
ySim, rSim = func.bootstrap_y(y, n=para['T_invest'] + para['T_prior'], 
                              BL=para['BL'])   
rSim, mu, sigma = func.GBM_PF(rSim, T=para['T_invest'] + para['T_prior'])
# =============================================================================
# Perform Portfolio Rebalancing strategy
start = time.time()
PF = pf.rebal_SR(para['strategy'], rSim,para['w0'], para['T_invest'],
                 para['T_prior'], para['cvar'],  para['rS'],
                 setting['plot_data'], para['perfectForesight'], mu, sigma)
end = time.time()
print(f"Done! Time needed: {np.round(end-start,2)} sec")
