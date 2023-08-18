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


# %% Z-FACTOR FITTING! Double check the actual definition of the z-factor. Ale Notes???
from scipy.optimize import curve_fit

# G = 1 / [ (iw + igamma + iwalpha) + (mu - ek - ReSigma(0) )]
# G = 1 / [ (iw(1+alpha) + igamma) + (mu - ek - ReSigma(0) )]
# Z = 1/(1+alpha)
# G = 1 / [ (iw/Z + igamma) + (mu - ek - ReSigma(w=0) )]
# G = Z / [ (iw + Z*igamma) + Z*(mu - ek - ReSigma(w=0) )]
# ZB = exp(−λ2/ω2).
# Sigma = ReSigma(w=0) - igamma - iwalpha

def fit_ImSigma(iw, gamma, alpha):
    ImSigma = -gamma - alpha * iw
    return ImSigma

def fit_ImSigma_2(iw, gamma, alpha, beta):
    ImSigma = -gamma - alpha * iw - beta * iw**2
    return ImSigma

n = 1003
iw_limit = iw[1000:n].real
siw_limit = (siw[0,0,1000:n]).imag

#%%
popt, pcov = curve_fit(fit_ImSigma, iw_limit, siw_limit)
fit_sigma = fit_ImSigma(iw_limit, popt[0], popt[1])
popt2, pcov2 = curve_fit(fit_ImSigma_2, iw_limit, siw_limit)
fit_sigma_2 = fit_ImSigma_2(iw_limit, popt2[0], popt2[1], popt2[2])

plt.figure(21)
plt.plot(iw_limit, siw_limit)
plt.plot(iw_limit, fit_sigma)
plt.plot(iw_limit, fit_sigma_2)

print(1/(1+popt[1]))
print(1/(1+popt2[1]))
# %%
