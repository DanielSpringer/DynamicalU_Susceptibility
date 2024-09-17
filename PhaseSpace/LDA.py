#%%
import h5py
import numpy as np 
import matplotlib.pyplot as plt
# from plotly.subplots import make_subplots

PATH = "/home/daniel/Data/DynamicalU_Susceptibility/PhaseSpace/Ust_vs_Usc/beta20/3dCubic/testDOS/oo_1-2024-04-18-Thu-15-35-38.hdf5"

with h5py.File(PATH, "r") as f:
    w = np.array(f[".axes"]["w-dos"])
    dos = np.array(f["start"]["lda-dos"]['value'])

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(18,4))
ax[0].plot(w,dos[0,0,:])
