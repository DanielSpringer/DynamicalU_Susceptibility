#%%
import numpy as np 
import matplotlib.pyplot as plt
import h5py

foldername = [
  "U4_w0_0c2_lam_0c6/",
]
filename = [
  "UbUs_1b-2023-08-01-Tue-12-57-57.hdf5",
]

for n in range(len(foldername)):
    with h5py.File(foldername[n]+filename[n], "r") as f:
        iw = np.array(f['.axes']['iw'][:])
        tau = np.array(f['.axes']['tau'][:])
        siw = np.array(f['dmft-005']['ineq-001']['siw']['value'])
    plt.plot(iw[1000:1130], siw[0,0,1000:1130].imag, label=foldername[n][:-1])
    plt.plot(iw[1000:1130], siw[0,1,1000:1130].imag, label=foldername[n][:-1])
    plt.legend()

# %%
def pol_Z_fit(iw, gamma, Z):
    sigma = -gamma + iw - 1/Z*iw
    return sigma

#%%
print(siw[0,1])
# %%
