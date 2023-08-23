#%%
# import plotly.io as pio
# import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import h5py

#%%
foldername = [
  "U4_w0_0c2_lam_0c6/",
  "U4_w0_0c2_lam_0c65/",
  ]
filename = [
  "UbUs_1b-2023-08-01-Tue-12-57-57.hdf5",
  "UbUs_1b-2023-08-01-Tue-11-33-41.hdf5",
  ]
sztau_filename = "sztau.dat"
ntau11_filename = "ntau_11.dat"
ntau12_filename = "ntau_12.dat"

for n in range(len(foldername)):
    with h5py.File(foldername[n]+filename[n], "r") as f:
        iw = np.array(f['.axes']['iw'][:])
        tau = np.array(f['.axes']['tau'][:])
        # gtau = np.array(f['dmft-last']['ineq-001']['gtau']['value'])
        # giw = np.array(f['dmft-last']['ineq-001']['giw']['value'])
        # siw = np.array(f['dmft-last']['ineq-001']['siw']['value'])
        gtau = np.array(f['dmft-005']['ineq-001']['gtau']['value'])
        giw = np.array(f['dmft-005']['ineq-001']['giw']['value'])
        siw = np.array(f['dmft-005']['ineq-001']['siw']['value'])
    plt.figure(1)
    plt.plot(tau, gtau[0,0], label=foldername[n][:-1])
    plt.figure(3)
    plt.plot(iw[1000:1130], giw[0,0,1000:1130].imag, label=foldername[n][:-1])
    plt.figure(4)
    plt.plot(iw[1000:1130], siw[0,0,1000:1130].imag, label=foldername[n][:-1])
    plt.legend()

#%%
for n in range(len(foldername)):
    sztau = np.loadtxt(foldername[n]+sztau_filename, usecols=[1,2,3])
    print(foldername[n], sztau.shape)
    plt.figure(2)
    plt.plot(tau[:120], sztau[:120,2], label=foldername[n][:-1])
    # plt.plot(tau, sztau[:-1,2], label=foldername[n][:-1])
    plt.title("Sztau")
    plt.yscale("log")  
    plt.legend()


#%%
for n in range(len(foldername)):
    ntau11 = np.loadtxt(foldername[n]+ntau11_filename, usecols=[5,6,7])
    ntau11 = np.array(ntau11)
    ntau11 = ntau11[:,2] -0.25
    
    print(foldername[n], ntau11.shape)
    plt.figure(2)
    plt.plot(tau[:120], ntau11[:120], label=foldername[n][:-1])
    # plt.plot(tau, sztau[:-1,2], label=foldername[n][:-1])
    plt.title("Ntau")
    plt.yscale("log")  
    plt.legend()

# %%
for n in range(len(foldername)):
    ntau12 = np.loadtxt(foldername[n]+ntau12_filename, usecols=[5,6,7])
    print(foldername[n], ntau12.shape)
    plt.figure(2)
    plt.plot(tau[:120], ntau12[:120,2], label=foldername[n][:-1])
    # plt.plot(tau, sztau[:-1,2], label=foldername[n][:-1])
    plt.title("Ntau")
    plt.yscale("log")  
    plt.legend()

# %%
