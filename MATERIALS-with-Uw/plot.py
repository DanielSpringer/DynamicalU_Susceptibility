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

#%%
foldername = [
  "SrVO3/",
  "SrMoO3/",
  ]
filename = [
  "SrVO3_2-2023-09-09-Sat-21-25-50.hdf5",
  "SrMoO3_1-2023-09-07-Thu-18-56-48.hdf5"
  ]
figname = [
    "SrVO3", 
    "SrMoO3", 
]
savename = [
  "SrVO3.dat",
  "SrMoO3.dat",
]
double_occ = [
    0.00591403, 
    0.0056532, 
    0.00594925, 
    0.00719721, 
    0.00692405, 
    0.00723876
]

sztau_filename = "sztau.dat"
ntau11_filename = "ntau1111.dat"
ntau12_filename = "ntau1112.dat"

siws = []
popt2s = []
iwmaxs = []
iw0s = []
iws = []
### Setting different noise levels for the two compounds...
# for n in range(0,len(foldername)-1):
for n in range(0,len(foldername)):
    with h5py.File(foldername[n]+filename[n], "r") as f:
        iw = np.array(f['.axes']['iw'][:])
        print(foldername[n])
        tau = np.array(f['.axes']['tau'][:])
        gtau = np.array(f['dmft-last']['ineq-001']['gtau']['value'])
        giw = np.array(f['dmft-last']['ineq-001']['giw']['value'])
        siw = np.array(f['dmft-last']['ineq-001']['siw']['value'])
        siws.append(siw)
        iws.append(iw)
    iw0 = int(iw.shape[0]/2)
    nn = iw0 + 5
    iwmax = iw0 + 50
    iw0s.append(iw0)
    iwmaxs.append(iwmax)
    iw_limit = iw[iw0:nn].real
    siw_limit = (siw[0,0,iw0:nn]).imag
    # popt, pcov = curve_fit(fit_ImSigma, iw_limit, siw_limit)
    # fit_sigma = fit_ImSigma(iw_limit, popt[0], popt[1])
    popt2, pcov2 = curve_fit(fit_ImSigma_2, iw_limit, siw_limit)
    popt2s.append(popt2)
    fit_sigma_2 = fit_ImSigma_2(iw_limit, popt2[0], popt2[1], popt2[2])
    plt.figure(1)
    plt.plot(tau, gtau[0,0], label=f"{figname[n]}, Z: {round(1/(1+popt2[1]), 2)}")
    plt.plot(tau, gtau[1,0], label=f"{figname[n]}, Z: {round(1/(1+popt2[1]), 2)}")
    plt.plot(tau, gtau[2,0], label=f"{figname[n]}, Z: {round(1/(1+popt2[1]), 2)}")
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
    data[:,2] = 9e-5
    np.savetxt(foldername[n]+savename[n], data)

#%%
for n in range(len(foldername)):
    sztau = np.loadtxt(foldername[n]+sztau_filename, usecols=[1,2,3])
    plt.figure(2)
    plt.plot(tau, sztau[:-1,2], label=f"{foldername[n][:-1]} sum: {np.trapz(x=tau, y=sztau[:-1,2])}")
    # plt.plot(tau, sztau[:-1,2], label=f"{foldername[n][:-1]} differential: {np.gradient(tau,sztau[:-1,2])[0]}")
plt.legend()
plt.xlabel(r'$\tau$')
plt.title("Sztau")

#%%
plt.figure(211)
for n in range(len(foldername)):
    ntau11 = np.loadtxt(foldername[n]+ntau11_filename, usecols=[5,6,7])
    ntau11 = np.array(ntau11)
    ntau11 = ntau11[:,2] - double_occ[3*n]
    plt.plot(tau[:120], ntau11[:120], label=f"Ntau11 {foldername[n][:-1]}")
plt.title(f"Ntau11 - double_occ")
plt.yscale("log")  
plt.xlabel(r'$\tau$')
plt.legend()

plt.figure(3)
for n in range(len(foldername)):
    ntau12 = np.loadtxt(foldername[n]+ntau12_filename, usecols=[5,6,7])
    plt.plot(tau[:120], ntau12[:120,2], label=f"Ntau12 {foldername[n][:-1]}")
plt.title("Ntau12")
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



# %% SPEZIFISCHE PLOTS
plt.figure(4)
plt.plot(iws[0][iw0s[0]:iwmaxs[0]], siws[0][0,0,iw0s[0]:iwmaxs[0]].imag, label=f"{figname[0]}, Z: {round(1/(1+popt2s[0][1]), 2)}")
plt.plot(iws[6][iw0s[6]:iwmaxs[6]], siws[6][0,0,iw0s[6]:iwmaxs[6]].imag, label=f"{figname[6]}, Z: {round(1/(1+popt2s[6][1]), 2)}")
plt.title("siw")
plt.legend()

#%%
plt.figure(5)
PATH = foldername[0]+"maxent.out.maxspec.dat"
data = np.loadtxt(PATH)
wc = int(data[:,0].shape[0]/2)
wmax = wc+window
wmin = wc-window
plt.plot(data[wmin:wmax,0], data[wmin:wmax,1], label=f"{figname[0]}")
PATH = foldername[6]+"maxent.out.maxspec.dat"
data = np.loadtxt(PATH)
wc = int(data[:,0].shape[0]/2)
wmax = wc+window
wmin = wc-window
plt.plot(data[wmin:wmax,0], data[wmin:wmax,1], label=f"{figname[6]}")
plt.legend()

#%%
plt.figure(2)
sztau = np.loadtxt(foldername[0]+sztau_filename, usecols=[1,2,3])
plt.plot(tau, sztau[:-1,2], label=foldername[0][:-1])
sztau = np.loadtxt(foldername[6]+sztau_filename, usecols=[1,2,3])
plt.plot(tau, sztau[:-1,2], label=foldername[6][:-1])
plt.legend()
plt.xlabel(r'$\tau$')
plt.title("Sztau")

plt.figure(211)
ntau11 = np.loadtxt(foldername[0]+ntau11_filename, usecols=[5,6,7])
ntau11 = np.array(ntau11)
ntau11 = ntau11[:,2] -0.25
plt.plot(tau[:120], ntau11[:120], label=f"Ntau11 {foldername[0][:-1]}")
ntau11 = np.loadtxt(foldername[6]+ntau11_filename, usecols=[5,6,7])
ntau11 = np.array(ntau11)
ntau11 = ntau11[:,2] -0.25
plt.plot(tau[:120], ntau11[:120], label=f"Ntau11 {foldername[6][:-1]}")

plt.title("Ntau11 - 0.25")
plt.yscale("log")  
plt.xlabel(r'$\tau$')
plt.legend()

plt.figure(3)
ntau12 = np.loadtxt(foldername[0]+ntau12_filename, usecols=[5,6,7])
plt.plot(tau[:120], ntau12[:120,2], label=f"Ntau12 {foldername[6][:-1]}")
ntau12 = np.loadtxt(foldername[6]+ntau12_filename, usecols=[5,6,7])
plt.plot(tau[:120], ntau12[:120,2], label=f"Ntau12 {foldername[6][:-1]}")
plt.title("Ntau12")
plt.yscale("log")  
plt.xlabel(r'$\tau$')
plt.legend()

# %%
