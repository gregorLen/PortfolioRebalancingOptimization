"""
MA: "Heuristic Portfolio Rebalancing Optimization Under Transaction Cost"
Gregor Lenhard - Spring 2019
CODE: Experiment
"""
import os
os.chdir(os.path.dirname(os.path.realpath('__file__')))
import numpy as np
import pandas as pd
import standard_functions as func
import portfolio_functions as pf
import multiprocessing as mp
import time
# =============================================================================
# == Data
setting = {'plot_data': False,
           'data': "germanstocks.csv",
           'i_stock' : np.arange(25),
           'i_date': np.arange(3739)}
# == Parameters
para = {'strategy' :  [1,2,3,4,5,9],
    # 1 = DEx; 2 = DEdx; 3 = NH; 4 = SLSQP, 5=SLSQP_ignore_cost, 9 = EW
        'perfectForesight' : [True, False],
        'rS' : 0.01/250,
        'w0' : 50000,
        'T_prior' : 1250,
        'T_invest' : 750,
        'BL' : 10,
        'cvar' : [0, 0.001, 0.005, 0.01, 0.02, 0.05]}
# == Experimental Setting
Experiment = np.arange(100)

Results = pd.DataFrame(columns=['Experiment','cvar','perfForesight', 'strategy',
                                'sr_Rebal','sr_noRebal','wealth','cost',
                                'wealth_noRebal','trades','gini','hhi','mu',
                                'sigma'])
## load data
df = pd.read_csv(setting['data'], sep=";", index_col=0)
y = df.iloc[setting['i_date'], setting['i_stock']]
r = np.diff(np.log(y), axis=0)
# =============================================================================
# Experimental support functions
# =============================================================================
# == write experiment to csv
def write_results():
    if os.path.isfile('Results.csv'):
        with open('Results.csv', 'a',newline='') as csvFile:
            Results.to_csv(csvFile, header=False, index=False)
            
    else:
        with open('Results.csv', 'a',newline='') as csvFile:
            Results.to_csv(csvFile, header=True, index=False)
# == Log experimental results in Dataframe "Results"
def log_result(PF):
    global Results
    Results = Results.append({'Experiment': exp,
                'cvar': PF['cvar'],
                'perfForesight' : PF['perfFore'],
                'strategy': PF['strategy'],
                'sr_Rebal': PF['sr_Rebal'],
                'sr_noRebal': PF['sr_noRebal'],
                'wealth': PF['final_w'],
                'cost': PF['cost'],
                'wealth_noRebal': PF['final_w_noRebal'],
                'trades': PF['trades'],
                'gini': PF['gini'],
                'hhi': PF['hhi'],
                'mu': PF['mu_Rebal'],
                'sigma': PF['sigma_Rebal']
                } , ignore_index=True)
# == run multiple setting simultan
def run_multi_process(strategy, perfectForesight, c_vals,exp):
    pool = mp.Pool()
    for cvar in c_vals:
        pool.apply_async(pf.rebal_SR,
                         args = (strategy, rSim,para['w0'], 
                                 para['T_invest'], para['T_prior'], cvar, 
                                 para['rS'], 
                                 setting['plot_data'], perfectForesight,
                                 mu, sigma, ),
                         callback = log_result)
    pool.close()
    pool.join()
    print(f"Experimental Setting No. {exp}. Strategy {strategy},", 
          f"foresight=={perfectForesight}. Done.")
# =============================================================================
# RUN EXPERIMENT    
# =============================================================================
start = time.time()

for exp in Experiment:
    np.random.seed(exp)
    # simulate data
    ySim, rSim = func.bootstrap_y(y, n=para['T_invest'] + para['T_prior'],
                                  BL=para['BL'])   
    rSim, mu, sigma = func.GBM_PF(rSim, T=para['T_invest'] + para['T_prior'])
    # =========================================================================
    for strategy in para['strategy']:
        for perfectForesight in para['perfectForesight']:
            if __name__ == '__main__':
                run_multi_process(strategy, perfectForesight, para['cvar'],exp)
    
    # write df to csv
    write_results()
    # delete results from df
    Results = Results.iloc[0:0]
    
end = time.time()
print(f"Done! Time needed: {np.round((end-start)/60,2)} mins")

