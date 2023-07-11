#%%
# import plotly.io as pio
# import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import h5py

#%%
foldername = [
  "Ubare_2.00/",
  "Ubare_2.00_Uscreened_0.8_wp_5e-6_g2_3e-6/",
  "Ubare_2.00_Uscreened_0.8_wp_0.005_g2_0.003/",
  "Ubare_2.00_Uscreened_0.8_wp_0.05_g2_0.03/",
  "Ubare_2.00_Uscreened_0.8_wp_0.5_g2_0.3/",
  "Ubare_2.00_Uscreened_0.8_wp_1.0_g2_0.6/",
  "Ubare_2.00_Uscreened_0.8_wp_5.0_g2_3.0/",
  "Ubare_0.8/"
  ]
filename = [
  "UbUs_1-2023-06-21-Wed-18-35-14.hdf5",
  "UbUs_1-2023-07-10-Mon-20-37-28.hdf5",
  "UbUs_1-2023-07-09-Sun-06-36-22.hdf5",
  "UbUs_1-2023-07-08-Sat-10-43-21.hdf5",
  "UbUs_1-2023-06-21-Wed-19-14-57.hdf5",
  "UbUs_1-2023-06-23-Fri-04-51-35.hdf5",
  "UbUs_1-2023-06-23-Fri-05-56-33.hdf5",
  "UbUs_1-2023-06-21-Wed-21-47-18.hdf5"
  ]
sztau_filename = "sztau.dat"

for n in range(len(foldername)):
    with h5py.File(foldername[n]+filename[n], "r") as f:
        iw = np.array(f['.axes']['iw'][:])
        tau = np.array(f['.axes']['tau'][:])
        gtau = np.array(f['dmft-last']['ineq-001']['gtau']['value'])
        giw = np.array(f['dmft-last']['ineq-001']['giw']['value'])
        siw = np.array(f['dmft-last']['ineq-001']['siw']['value'])
    plt.figure(1)
    plt.plot(tau, gtau[0,0], label=foldername[n][:-1])
    plt.figure(3)
    plt.plot(iw[1000:1030], giw[0,0,1000:1030].imag, label=foldername[n][:-1])
    plt.figure(4)
    plt.plot(iw[1000:1030], siw[0,0,1000:1030].imag, label=foldername[n][:-1])
    plt.legend()

for n in range(len(foldername)):
    sztau = np.loadtxt(foldername[n]+sztau_filename, usecols=[1,2,3])
    print(foldername[n], sztau.shape)
    plt.figure(2)
    plt.plot(tau, sztau[:-1,2], label=foldername[n][:-1])
    plt.legend()


# %%
