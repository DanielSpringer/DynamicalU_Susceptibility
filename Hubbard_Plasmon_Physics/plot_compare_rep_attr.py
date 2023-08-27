#%%
# import plotly.io as pio
# import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit

def fit_ImSigma(iw, gamma, alpha):
    ImSigma = -gamma - alpha * iw
    return ImSigma

def fit_ImSigma_2(iw, gamma, alpha, beta):
    ImSigma = -gamma - alpha * iw - beta * iw**2
    return ImSigma

#%% STATIC ATTRACTIVE
foldername = [
  "static/Ubare4/",
  "static/Ubare-4/",
  ]
filename = [
  "Bethe-CHI-2021-02-15-Mon-05-44-52.hdf5",
  "HubbPlas-2023-08-26-Sat-21-48-05.hdf5",
  ]
figname = [
    "Ubare4", 
    "Ubare-4", 
]
savename = [
  "Ubare4.dat",
  "Ubare-4.dat",
]

sztau_filename = "sztau.dat"
ntau11_filename = "ntau_11.dat"
ntau12_filename = "ntau_12.dat"


for n in range(len(foldername)):
    with h5py.File(foldername[n]+filename[n], "r") as f:
        iw = np.array(f['.axes']['iw'][:])
        # print(foldername[n], iw.shape)
        tau = np.array(f['.axes']['tau'][:])
        gtau = np.array(f['dmft-last']['ineq-001']['gtau']['value'])
        giw = np.array(f['dmft-last']['ineq-001']['giw']['value'])
        siw = np.array(f['dmft-last']['ineq-001']['siw']['value'])
    iw0 = int(iw.shape[0]/2)
    nn = iw0 + 5
    iwmax = iw0 + 50
    iw_limit = iw[iw0:nn].real
    siw_limit = (siw[0,0,iw0:nn]).imag
    # popt, pcov = curve_fit(fit_ImSigma, iw_limit, siw_limit)
    # fit_sigma = fit_ImSigma(iw_limit, popt[0], popt[1])
    popt2, pcov2 = curve_fit(fit_ImSigma_2, iw_limit, siw_limit)
    fit_sigma_2 = fit_ImSigma_2(iw_limit, popt2[0], popt2[1], popt2[2])
    plt.figure(1)
    plt.plot(tau, gtau[0,0], label=f"{figname[n]}, Z: {round(1/(1+popt2[1]), 2)}")
    plt.title("gtau")
    plt.legend()
    plt.figure(3)
    plt.plot(iw[iw0:iwmax], giw[0,0,iw0:iwmax].imag, label=f"{figname[n]}, Z: {round(1/(1+popt2[1]), 2)}")
    plt.title("giw")
    plt.legend()
    plt.figure(4)
    plt.plot(iw[iw0:iwmax], siw[0,0,iw0:iwmax].imag, label=f"{figname[n]}, Z: {round(1/(1+popt2[1]), 2)}")
    plt.title("siw")
    plt.legend()
    # print(1/(1+popt2[1]))

    data = np.zeros((gtau.shape[2],3))
    data[:,0] = tau.transpose()
    data[:,1] = -gtau[0,1].transpose()
    data[:,2] = 1e-3
    np.savetxt(foldername[n]+savename[n], data)

#%%
for n in range(len(foldername)):
    sztau = np.loadtxt(foldername[n]+sztau_filename, usecols=[1,2,3])
    print(foldername[n], sztau.shape)
    plt.figure(2)
    plt.plot(tau, sztau[:-1,2], label=foldername[n][:-1])
plt.legend()
plt.xlabel(r'$\tau$')
plt.title("Sztau")

#%%
for n in range(len(foldername)):
    ntau11 = np.loadtxt(foldername[n]+ntau11_filename, usecols=[5,6,7])
    ntau11 = np.array(ntau11)
    ntau11 = ntau11[:,2] -0.25
    plt.figure(2)
    plt.plot(tau[:120], ntau11[:120], label=f"Ntau11 {foldername[n][:-1]}")

for n in range(len(foldername)):
    ntau12 = np.loadtxt(foldername[n]+ntau12_filename, usecols=[5,6,7])
    plt.figure(2)
    plt.plot(tau[:120], ntau12[:120,2], label=f"Ntau12 {foldername[n][:-1]}")

plt.title("Ntau")
plt.yscale("log")  
plt.xlabel(r'$\tau$')
plt.legend()


# %%
window = 490
plt.figure(4)
for folder, name in zip(foldername, figname):
    PATH = folder+"maxent.out.maxspec.dat"
    data = np.loadtxt(PATH)
    wc = int(data[:,0].shape[0]/2)
    wmax = wc+window
    wmin = wc-window
    plt.plot(data[wmin:wmax,0], data[wmin:wmax,1], label=f"{name}")
    plt.legend()

