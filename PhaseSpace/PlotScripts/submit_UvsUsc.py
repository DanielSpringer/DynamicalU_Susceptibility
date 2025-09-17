#%%
import os
import sys
import numpy as np
import run_tools as rt

# sys.argv[1]
config = {
    "partition": ["hclm12"],
    "Steps": "1",
    "Nmeas": "1e5",
    "Ncorr": "1e2",
    "beta": 30,
    "CurrentFile": "oo_6LD",
    "readold": -1,
    "MeasSusz": 1,
    "Uw": 1,
    "StatSteps": 0,
    "N4wf": 20,
    "N4wb": 0,
    "J": 0,
    "V": 0,
    "w0": 4,
    "set_lam": -1
}



if config["set_lam"] != -1:
    ### SETTINGS lam = 4
    if config["set_lam"] == 4:
        U_st_min = 2.0
        U_st_max = 10.0
        U_st_steps = 11
        U_st_grid = np.linspace(U_st_min, U_st_max, U_st_steps)

        U_sc_min = 0.0
        U_sc_max = 2.0
        U_sc_steps = 11
        U_sc_grid = np.linspace(U_sc_min, U_sc_max, U_sc_steps)

    if config["set_lam"] == 20:
        U_st_min = 2.0
        U_st_max = 10.0
        U_st_steps = 11
        U_st_grid = np.linspace(U_st_min, U_st_max, U_st_steps)

        U_sc_min = 0.0
        U_sc_max = 2.0
        U_sc_steps = 11
        U_sc_grid = np.linspace(U_sc_min, U_sc_max, U_sc_steps)
        
    rt.fixed_lambda(config, U_st_grid, U_sc_grid, config["set_lam"])
    
if config["w0"] != -1:
    ### SETTINGS w0 = 4
    if config["w0"] == 4.0:
        U_st_min = 0.0
        U_st_max = 16.0
        U_st_steps = int((U_st_max-U_st_min)/0.2)+1
        U_st_grid = np.linspace(U_st_min, U_st_max, U_st_steps)

        U_sc_min = 0.0
        U_sc_max = 3.6
        U_sc_steps = int((U_sc_max-U_sc_min)/0.2)+1
        U_sc_grid = np.linspace(U_sc_min, U_sc_max, U_sc_steps)

    ### SETTINGS w0 = 2
    if config["w0"] == 2.0:
        U_st_min = 0.0
        U_st_max = 16.0
        U_st_steps = int((U_st_max-U_st_min)/0.2)+1
        U_st_grid = np.linspace(U_st_min, U_st_max, U_st_steps)

        U_sc_min = 0.0
        U_sc_max = 3.6
        U_sc_steps = int((U_sc_max-U_sc_min)/0.2)+1
        U_sc_grid = np.linspace(U_sc_min, U_sc_max, U_sc_steps)

    ### SETTINGS w0 = 1 
    if config["w0"] == 1.0:
        U_st_min = 0.2
        U_st_max = 1.8
        U_st_steps = 9
        U_st_grid = np.linspace(U_st_min, U_st_max, U_st_steps)

        U_sc_min = 0.0
        U_sc_max = 1.8
        U_sc_steps = 11
        U_sc_grid = np.linspace(U_sc_min, U_sc_max, U_sc_steps)

    rt.fixed_omega(config, U_st_grid, U_sc_grid, config["w0"])


if config["w0"] == -1 and config["set_lam"] == -1:
    U_sc = 0.0
    U_st = 10.0
    w0_min = 0.01
    w0_max = 20
    w0_steps = 11
    w0_grid = np.linspace(w0_min, w0_max, w0_steps)
    
    rt.fixed_UscUst(config, U_sc, U_st, w0_grid, w0_min, w0_max)
    




# if set_lam > 0:

# if w0 > 0:
#     rt.fixed_omega(config, U_st_grid, U_sc_grid, w0)

#%%
# config = {
#     "Steps": "30",
#     "Nmeas": "1e6",
#     "Ncorr": "3e2",
#     "beta": 20,
#     "CurrentFile": "oo_1",
#     "readold": -1,
#     "MeasSusz": 1,
#     "Uw": 1,
#     "StatSteps": 0,
#     "N4wf": 20,
#     "N4wb": 0
# }

# for k, v in config.items():
#     globals()[k]=v
        
# #%%
# print(Steps)            
