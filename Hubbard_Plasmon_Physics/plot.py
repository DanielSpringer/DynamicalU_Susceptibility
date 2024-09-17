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
  "static/Ubare4/",
#   "dynamical/Ubare5_Usc3/w0c5_ls0c5/",
#   "dynamical/Ubare5_Usc3/w1_ls1c0/",
  "dynamical/Ubare5_Usc3/w2_ls2c0/",
  "dynamical/Ubare7_Usc0/w2c0_ls7c0/",
#   "dynamical/Ubare8_Usc2/w2c0_ls6c0/",
  "dynamical/Ubare8_Usc2/w4c0_ls12c0/",
  "dynamical/Ubare8_Usc3/w6_ls15c0/",
  # "dynamical/Ubare10_Usc2/w5c0_ls20c0/",
  "dynamical/Ubare10_Usc2/w6c0_ls24c0/",
  "dynamical/Ubare17_Usc1/w8c0_ls64/",
  # "dynamical/Ubare17_Usc1/w12c0_ls96/",
  # "dynamical/Ubare8_Usc1/w1c0_ls3c5/",
  "dynamical/Ubare8_Usc1/w2c0_ls7c0/",
  "dynamical/Ubare10_Usc2/w6c0_ls24c0/",
#   "dynamical/Ubare8c5_Usc1/w2c0_ls7c5/",
#   "dynamical/Ubare9c0_Usc1/w2c0_ls8c0/"
  "dynamical/Ubare7c5_Usc1/w2c0_ls6c5/",
  "dynamical/Ubare7_Usc1/w2c0_ls6c0/",
  "dynamical/Ubare16_Usc0/w15c0_ls120/",
  "dynamical/Ubare16_Usc0/w5c0_ls40/",
  "static/Ubare2/",
  "static/Ubare2c5/",
#   "static/Ubare3/",
  ]
filename = [
  "Bethe-CHI-2021-02-15-Mon-05-44-52.hdf5",
#   "HubbPlas-2023-08-18-Fri-08-47-18.hdf5",
#   "HubbPlas-2023-08-18-Fri-06-36-53.hdf5",
  "HubbPlas-2023-08-18-Fri-06-22-25.hdf5",
  "HubbPlas-2023-08-27-Sun-17-34-51.hdf5",
#   "HubbPlas-2023-08-20-Sun-08-17-46.hdf5",
  "HubbPlas-2023-08-20-Sun-11-10-02.hdf5",
  "HubbPlas-2023-08-20-Sun-00-29-55.hdf5",
  # "HubbPlas-2023-08-23-Wed-08-31-46.hdf5",
  "HubbPlas-2023-08-23-Wed-09-38-36.hdf5",
  "HubbPlas-2023-08-23-Wed-07-16-28.hdf5",
  # "HubbPlas-2023-08-23-Wed-09-20-24.hdf5",
  # "HubbPlas-2023-08-26-Sat-18-47-02.hdf5",
  "HubbPlas-2023-08-26-Sat-19-21-07.hdf5",
  "HubbPlas-2023-08-23-Wed-09-38-36.hdf5",
#   "HubbPlas-2023-09-30-Sat-19-45-35.hdf5",
#   "HubbPlas-2023-09-30-Sat-15-50-51.hdf5",
  "HubbPlas-2023-10-01-Sun-06-26-18.hdf5",
  "HubbPlas-2023-08-27-Sun-09-19-55.hdf5",
  "HubbPlas_2-2023-10-13-Fri-19-12-19.hdf5",
  "HubbPlas-2023-10-13-Fri-09-19-14.hdf5",
  "HubbPlas-2023-08-26-Sat-20-43-55.hdf5",
  "HubbPlas-2023-10-13-Fri-21-37-17.hdf5",
#   "HubbPlas-2023-10-13-Fri-21-32-13.hdf5"
  ]
figname = [
    "static U4", 
    # "$U_b=5, U_s=3, \omega=0.5, \lambda=0.5$",
    # "$U_b=5, U_s=3, \omega=1.0, \lambda=1.0$",
    "$U_b=5, U_s=3, \omega=2.0, \lambda=2.0$",
    "$U_b=7, U_s=0, \omega=2.0, \lambda=7.0$",
    # "$U_b=8, U_s=2, \omega=2.0, \lambda=6.0$"
    "$U_b=8, U_s=2, \omega=4.0, \lambda=12.0$",
    "$U_b=8, U_s=3, \omega=6.0, \lambda=15.0$",
    # "$U_b=10, U_s=2, \omega=5.0, \lambda=20.0$",
    "$U_b=10, U_s=2, \omega=6.0, \lambda=24.0$",
    "$U_b=17, U_s=1, \omega=8.0, \lambda=64.0$",
    # "$U_b=17, U_s=1, \omega=12.0, \lambda=96.0$",
    # "$U_b=8, U_s=1, \omega=1.0, \lambda=3.5$",
    "$U_b=8, U_s=1, \omega=2.0, \lambda=7.0$",
    "$U_b=10, U_s=2, \omega=6.0, \lambda=24.0$",
    # "$U_b=8.5, U_s=1, \omega=2.0, \lambda=7.5$",
    # "$U_b=9, U_s=1, \omega=2.0, \lambda=8.0$",
    "$U_b=7.5, U_s=1, \omega=2.0, \lambda=6.5$",
    "$U_b=7.0, U_s=1, \omega=2.0, \lambda=6.0$",
    "$U_b=16.0, U_s=0, \omega=15.0, \lambda=120.0$",
    "$U_b=16.0, U_s=0, \omega=5.0, \lambda=40.0$",
    "static U2", 
    "static U2c5", 
    "static U3", 

]
savename = [
  "Ubare4.dat",
#   "dynamical/Ubare5_Usc3/w0c5_ls0c5/",
#   "dynamical/Ubare5_Usc3/w1_ls1c0/",
  "Ubare5_Usc3_w2_ls2c0.dat",
  "Ubare7_Usc0_w2_ls7c0.dat",
#   "dynamical/Ubare8_Usc2/w2c0_ls6c0/",
  "Ubare8_Usc2_w4c0_ls12c0.dat",
  "Ubare8_Usc3_w6_ls15c0.dat",
  # "Ubare10_Usc2_w5_ls20c0.dat",
  "Ubare10_Usc2_w6_ls24c0.dat",
  "Ubare17_Usc1_w8_ls64c0.dat",
  # "Ubare17_Usc1_w12_ls96c0.dat",
  # "Ubare8_Usc1_w1c0_ls3c5.dat",
  "Ubare8_Usc1_w2c0_ls7c0.dat",
  "Ubare10_Usc2_w6_ls24c0.dat",
#   "Ubare8c5_Usc1_w2c0_ls7c5.dat",
#   "Ubare9_Usc1_w2c0_ls8c0.dat",
  "Ubare7c5_Usc1_w2c0_ls6c5.dat",
  "Ubare7c0_Usc1_w2c0_ls6c0.dat",
  "Ubare16c0_Usc0_w15c0_ls120c0.dat",
  "Ubare16c0_Usc0_w5c0_ls40c0.dat",
  "Ubare2.dat",
  "Ubare2c5.dat",
  "Ubare3.dat",
]

sztau_filename = "sztau.dat"
ntau11_filename = "ntau_11.dat"
ntau12_filename = "ntau_12.dat"

siws = []
popt2s = []
iwmaxs = []
iw0s = []
iws = []
for n in range(len(foldername)):
    print(foldername[n]+filename[n])
    with h5py.File(foldername[n]+filename[n], "r") as f:
        iw = np.array(f['.axes']['iw'][:])
        # print(foldername[n])
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
    plt.title("gtau")
    plt.legend()
    plt.figure(3)
    plt.plot(iw[iw0:iwmax], giw[0,0,iw0:iwmax].imag, label=f"{figname[n]}, Z: {round(1/(1+popt2[1]), 2)}")
    plt.title("giw")
    plt.legend()
    plt.figure(4)
    plt.plot(iw[iw0:iwmax], siw[0,0,iw0:iwmax].imag, label=f"{figname[n]}, Z: {round(1/(1+popt2[1]), 2)}")
    plt.title("siw")
    # plt.legend()
    # print(1/(1+popt2[1]))

    data = np.zeros((gtau.shape[2],3))
    data[:,0] = tau.transpose()
    data[:,1] = -gtau[0,1].transpose()
    data[:,2] = 1e-3
    np.savetxt(foldername[n]+savename[n], data)

#%%
for n in range(len(foldername)):
    sztau = np.loadtxt(foldername[n]+sztau_filename, usecols=[1,2,3])
    plt.figure(2)
    plt.plot(tau, sztau[:-1,2], label=f"{foldername[n][:-1]} sum: {np.trapz(x=tau, y=sztau[:-1,2])}")
plt.legend()
plt.xlabel(r'$\tau$')
plt.title("Sztau")
# sum(sztau[:-1,2])*50/1000
#%%
plt.figure(211)
for n in range(len(foldername)):
    ntau11 = np.loadtxt(foldername[n]+ntau11_filename, usecols=[5,6,7])
    ntau11 = np.array(ntau11)
    ntau11 = ntau11[:,2] -0.25
    plt.plot(tau[:120], ntau11[:120], label=f"Ntau11 {foldername[n][:-1]}")
plt.title("Ntau11 - 0.25")
plt.yscale("log")  
plt.xlabel(r'$\tau$')
# plt.legend()

plt.figure(3)
for n in range(len(foldername)):
    ntau12 = np.loadtxt(foldername[n]+ntau12_filename, usecols=[5,6,7])
    plt.plot(tau[:120], ntau12[:120,2], label=f"Ntau12 {foldername[n][:-1]}")
plt.title("Ntau12")
plt.yscale("log")  
plt.xlabel(r'$\tau$')
# plt.legend()


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
    # plt.legend()

#%%
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

s = 0
e = -1
marker_dict = dict(
        size=1.3,
        color='red',
        colorscale='Viridis',
    )
line_dict = dict(
        color='darkblue',
        width=2
    )

fig = go.Figure()
step = 0
for folder, name in zip(foldername, figname):
    PATH = folder+"maxent.out.maxspec.dat"
    data = np.loadtxt(PATH)
    wc = int(data[:,0].shape[0]/2)
    wmax = wc+window
    wmin = wc-window
    x = np.zeros(data[wmin:wmax,0].shape[0]) + step
    fig.add_trace(
        go.Scatter3d(
        x=x, y=data[wmin:wmax,0], z=data[wmin:wmax,1],
        marker=marker_dict,
        line=line_dict,
        name=folder
    ))
    step += 0.3



fig.update_layout(
    width=800,
    height=700,
    autosize=False,
    scene=dict(
        camera=dict(
            up=dict(
                x=0,
                y=0,
                z=1
            ),
            eye=dict(
                x=0,
                y=1.0707,
                z=1,
            )
        ),
        aspectratio = dict( x=1, y=1, z=0.7 ),
        aspectmode = 'manual'
    ),
)

fig.show()


# %% SPEZIFISCHE PLOTS
plt.figure(4)
plt.plot(iws[0][iw0s[0]:iwmaxs[0]], siws[0][0,0,iw0s[0]:iwmaxs[0]].imag, label=f"{figname[0]}, Z: {round(1/(1+popt2s[0][1]), 2)}")
plt.plot(iws[2][iw0s[2]:iwmaxs[2]], siws[2][0,0,iw0s[2]:iwmaxs[2]].imag, label=f"{figname[2]}, Z: {round(1/(1+popt2s[2][1]), 2)}")
# plt.plot(iws[7][iw0s[7]:iwmaxs[7]], siws[7][0,0,iw0s[7]:iwmaxs[7]].imag, label=f"{figname[7]}, Z: {round(1/(1+popt2s[7][1]), 2)}")
# plt.plot(iws[9][iw0s[9]:iwmaxs[9]], siws[9][0,0,iw0s[9]:iwmaxs[9]].imag, label=f"{figname[9]}, Z: {round(1/(1+popt2s[9][1]), 2)}")
plt.plot(iws[10][iw0s[10]:iwmaxs[10]], siws[10][0,0,iw0s[10]:iwmaxs[10]].imag, label=f"{figname[10]}, Z: {round(1/(1+popt2s[10][1]), 2)}")
# plt.plot(iws[6][iw0s[6]:iwmaxs[6]], siws[6][0,0,iw0s[6]:iwmaxs[6]].imag, label=f"{figname[6]}, Z: {round(1/(1+popt2s[6][1]), 2)}")
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
PATH = foldername[2]+"maxent.out.maxspec.dat"
data = np.loadtxt(PATH)
wc = int(data[:,0].shape[0]/2)
wmax = wc+window
wmin = wc-window
plt.plot(data[wmin:wmax,0], data[wmin:wmax,1], label=f"{figname[2]}")
PATH = foldername[10]+"maxent.out.maxspec.dat"
data = np.loadtxt(PATH)
wc = int(data[:,0].shape[0]/2)
wmax = wc+window
wmin = wc-window
plt.plot(data[wmin:wmax,0], data[wmin:wmax,1], label=f"{figname[10]}")
plt.legend()

#%%
plt.figure(51)
PATH = foldername[11]+"maxent.out.maxspec.dat"
data = np.loadtxt(PATH)
wc = int(data[:,0].shape[0]/2)
window = 495
wmax = wc+window
wmin = wc-window
plt.plot(data[wmin:wmax,0], data[wmin:wmax,1], label=f"{figname[11]}")
PATH = foldername[14]+"maxent.out.maxspec.dat"
data = np.loadtxt(PATH)
wc = int(data[:,0].shape[0]/2)
window = 500
wmax = wc+window
wmin = wc-window
plt.plot(data[wmin:wmax,0], data[wmin:wmax,1], label=f"{figname[14]}")
# PATH = foldername[12]+"maxent.out.maxspec.dat"
# data = np.loadtxt(PATH)
# wc = int(data[:,0].shape[0]/2)
# wmax = wc+window
# wmin = wc-window
# plt.plot(data[wmin:wmax,0], data[wmin:wmax,1], label=f"{figname[12]}")
plt.legend()

#%%
plt.figure(2)
sztau = np.loadtxt(foldername[0]+sztau_filename, usecols=[1,2,3])
plt.plot(tau, sztau[:-1,2], label=f"{foldername[0][:-1]} differential: {np.gradient(tau,sztau[:-1,2])[0]}")
sztau = np.loadtxt(foldername[2]+sztau_filename, usecols=[1,2,3])
plt.plot(tau, sztau[:-1,2], label=f"{foldername[2][:-1]} differential: {np.gradient(tau,sztau[:-1,2])[0]}")
# sztau = np.loadtxt(foldername[7]+sztau_filename, usecols=[1,2,3])
# plt.plot(tau, sztau[:-1,2], label=f"{foldername[7][:-1]} differential: {np.gradient(tau,sztau[:-1,2])[0]}")
# sztau = np.loadtxt(foldername[9]+sztau_filename, usecols=[1,2,3])
# plt.plot(tau, sztau[:-1,2], label=f"{foldername[9][:-1]} differential: {np.gradient(tau,sztau[:-1,2])[0]}")
sztau = np.loadtxt(foldername[10]+sztau_filename, usecols=[1,2,3])
plt.plot(tau, sztau[:-1,2], label=f"{foldername[10][:-1]} differential: {np.gradient(tau,sztau[:-1,2])[0]}")
plt.legend()
plt.xlabel(r'$\tau$')
plt.title("Sztau")

plt.figure(211)
ntau11 = np.loadtxt(foldername[0]+ntau11_filename, usecols=[5,6,7])
ntau11 = np.array(ntau11)
ntau11 = ntau11[:,2] -0.25
plt.plot(tau[:120], ntau11[:120], label=f"Ntau11 {foldername[0][:-1]}")
ntau11 = np.loadtxt(foldername[2]+ntau11_filename, usecols=[5,6,7])
ntau11 = np.array(ntau11)
ntau11 = ntau11[:,2] -0.25
plt.plot(tau[:120], ntau11[:120], label=f"Ntau11 {foldername[2][:-1]}")
# ntau11 = np.loadtxt(foldername[7]+ntau11_filename, usecols=[5,6,7])
# ntau11 = np.array(ntau11)
# ntau11 = ntau11[:,2] -0.25
# plt.plot(tau[:120], ntau11[:120], label=f"Ntau11 {foldername[7][:-1]}")
# ntau11 = np.loadtxt(foldername[9]+ntau11_filename, usecols=[5,6,7])
# ntau11 = np.array(ntau11)
# ntau11 = ntau11[:,2] -0.25
# plt.plot(tau[:120], ntau11[:120], label=f"Ntau11 {foldername[9][:-1]}")
ntau11 = np.loadtxt(foldername[10]+ntau11_filename, usecols=[5,6,7])
ntau11 = np.array(ntau11)
ntau11 = ntau11[:,2] -0.25
plt.plot(tau[:120], ntau11[:120], label=f"Ntau11 {foldername[10][:-1]}")
plt.title("Ntau11 - 0.25")
plt.yscale("log")  
plt.xlabel(r'$\tau$')
plt.legend()

plt.figure(3)
ntau12 = np.loadtxt(foldername[0]+ntau12_filename, usecols=[5,6,7])
plt.plot(tau[:120], ntau12[:120,2], label=f"Ntau12 {foldername[0][:-1]}")
ntau12 = np.loadtxt(foldername[2]+ntau12_filename, usecols=[5,6,7])
plt.plot(tau[:120], ntau12[:120,2], label=f"Ntau12 {foldername[2][:-1]}")
# ntau12 = np.loadtxt(foldername[7]+ntau12_filename, usecols=[5,6,7])
# plt.plot(tau[:120], ntau12[:120,2], label=f"Ntau12 {foldername[7][:-1]}")
# ntau12 = np.loadtxt(foldername[9]+ntau12_filename, usecols=[5,6,7])
# plt.plot(tau[:120], ntau12[:120,2], label=f"Ntau12 {foldername[9][:-1]}")
ntau12 = np.loadtxt(foldername[10]+ntau12_filename, usecols=[5,6,7])
plt.plot(tau[:120], ntau12[:120,2], label=f"Ntau12 {foldername[10][:-1]}")
plt.title("Ntau12")
plt.yscale("log")  
plt.xlabel(r'$\tau$')
plt.legend()

# %%
lam = 6
wp = 2
np.exp(-lam**2/wp**2)

#%%
x = np.linspace(0,1,10)
y = x**2
np.gradient(y,x)
# %%


#%%
plt.figure(2)
sztau = np.loadtxt(foldername[11]+sztau_filename, usecols=[1,2,3])
plt.plot(tau, sztau[:-1,2], label=f"{foldername[11][:-1]} differential: {np.gradient(tau,sztau[:-1,2])[11]}")
sztau = np.loadtxt(foldername[14]+sztau_filename, usecols=[1,2,3])
plt.plot(tau, sztau[:-1,2], label=f"{foldername[14][:-1]} differential: {np.gradient(tau,sztau[:-1,2])[14]}")
plt.legend()
plt.xlabel(r'$\tau$')
plt.title("Sztau")


#%%
plt.figure(211)
ntau11 = np.loadtxt(foldername[11]+ntau11_filename, usecols=[5,6,7])
ntau11 = np.array(ntau11)
ntau11 = ntau11[:,2] -0.25
plt.plot(tau[:120], ntau11[:120], label=f"Ntau11 {foldername[11][:-1]}")
ntau11 = np.loadtxt(foldername[14]+ntau11_filename, usecols=[5,6,7])
ntau11 = np.array(ntau11)
ntau11 = ntau11[:,2] -0.25
plt.plot(tau[:120], ntau11[:120], label=f"Ntau11 {foldername[14][:-1]}")
plt.title("Ntau11 - 0.25")
plt.yscale("log")  
plt.xlabel(r'$\tau$')
plt.legend()

plt.figure(3)
ntau12 = np.loadtxt(foldername[11]+ntau12_filename, usecols=[5,6,7])
plt.plot(tau[:120], ntau12[:120,2], label=f"Ntau12 {foldername[11][:-1]}")
ntau12 = np.loadtxt(foldername[14]+ntau12_filename, usecols=[5,6,7])
plt.plot(tau[:120], ntau12[:120,2], label=f"Ntau12 {foldername[14][:-1]}")
plt.title("Ntau12")
plt.yscale("log")  
plt.xlabel(r'$\tau$')
plt.legend()
