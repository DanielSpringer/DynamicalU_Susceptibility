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
  "static/",
#   "dynamical/Ubare5_Usc3/w0c5_ls0c5/",
#   "dynamical/Ubare5_Usc3/w1_ls1c0/",
  "dynamical/Ubare5_Usc3/w2_ls2c0/",
#   "dynamical/Ubare8_Usc2/w2c0_ls6c0/",
  "dynamical/Ubare8_Usc2/w4c0_ls12c0/",
  "dynamical/Ubare8_Usc3/w6_ls15c0/",
  ]
filename = [
  "Bethe-CHI-2021-02-15-Mon-05-44-52.hdf5",
#   "HubbPlas-2023-08-18-Fri-08-47-18.hdf5",
#   "HubbPlas-2023-08-18-Fri-06-36-53.hdf5",
  "HubbPlas-2023-08-18-Fri-06-22-25.hdf5",
#   "HubbPlas-2023-08-20-Sun-08-17-46.hdf5",
  "HubbPlas-2023-08-20-Sun-11-10-02.hdf5",
  "HubbPlas-2023-08-20-Sun-00-29-55.hdf5",
  ]
figname = [
    "static", 
    # "$U_b=5, U_s=3, \omega=0.5, \lambda=0.5$",
    # "$U_b=5, U_s=3, \omega=1.0, \lambda=1.0$",
    "$U_b=5, U_s=3, \omega=2.0, \lambda=2.0$",
    # "$U_b=8, U_s=2, \omega=2.0, \lambda=6.0$"
    "$U_b=8, U_s=2, \omega=4.0, \lambda=12.0$",
    "$U_b=8, U_s=3, \omega=6.0, \lambda=15.0$"
]
savename = [
  "Ubare4.dat",
#   "dynamical/Ubare5_Usc3/w0c5_ls0c5/",
#   "dynamical/Ubare5_Usc3/w1_ls1c0/",
  "Ubare5_Usc3_w2_ls2c0.dat",
#   "dynamical/Ubare8_Usc2/w2c0_ls6c0/",
  "Ubare8_Usc2_w4c0_ls12c0.dat",
  "Ubare8_Usc3_w6_ls15c0.dat",

]

sztau_filename = "sztau.dat"

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

asdasd

# for n in range(len(foldername)):
#     sztau = np.loadtxt(foldername[n]+sztau_filename, usecols=[1,2,3])
#     print(foldername[n], sztau.shape)
#     plt.figure(2)
#     plt.plot(tau, sztau[:-1,2], label=foldername[n][:-1])
#     plt.legend()

#%%
window = 490
plt.figure(4)
for n in range(4):
    PATH = foldername[n]+"maxent.out.maxspec.dat"
    data = np.loadtxt(PATH)
    wc = int(data[:,0].shape[0]/2)
    wmax = wc+window
    wmin = wc-window
    plt.plot(data[wmin:wmax,0], data[wmin:wmax,1], label=f"{figname[n]}")
    # plt.legend()
