#%%
# 
import h5py
import numpy as np
from scipy.optimize import curve_fit
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import os
import re
from decimal import Decimal
import ana_cont.continuation as cont
import time

print(" OK " )

def fit_ImSigma_4(iw, gamma, alpha, beta, delta, delta2):
    ImSigma = -gamma - alpha * iw - beta * iw**2 - delta * iw**3 - delta2 * iw**4
    return ImSigma

def fit_ImSigma_3(iw, gamma, alpha, beta, delta):
    ImSigma = -gamma - alpha * iw - beta * iw**2 - delta * iw**3
    return ImSigma

def fit_ImSigma_2(iw, gamma, alpha, beta):
    ImSigma = -gamma - alpha * iw - beta * iw**2
    return ImSigma

def fit_ImSigma_1(iw, gamma, alpha):
    ImSigma = gamma - alpha * iw
    return ImSigma

# frequencies = ["lam_100"]
# frequencies = ["w0_2"]
frequencies = ["w0_4"]
data_np = {}
data_np[frequencies[0]] = {}
# data_np["w0_2"] = {}
data = {}
# data["w0_4"] = {}
data[frequencies[0]] = {}
Ulim = 2380
beta = 30

for freq in frequencies:
    # PATH = "/mnt/data/BACKUP/daniel/Data/DynamicalU_Susceptibility/PhaseSpace/Ust_vs_Usc/beta30/"+freq
    PATH = f"/mnt/scratch/daniel/Data/DynamicalU_Susceptibility/PhaseSpace/Ust_vs_Usc/beta{beta}/"+freq
    # /mnt/scratch/daniel/Data/DynamicalU_Susceptibility/PhaseSpace/Ust_vs_Usc/beta30/w0_4
    # print(PATH)
    U_st = []
    U_sc = []
    ivmaxs = []
    iv0s = []
    popt2s = []
    fitpoints = 3

    for subdir, dirs, files in os.walk(PATH):
        for file in files:
            # print(subdir)
            if frequencies[0][0] == "w":
                # print(subdir)
                # print(re.findall(r'\d+', subdir))
                U1 = re.findall(r'\d+', subdir)[3]
                U2 = re.findall(r'\d+', subdir)[4]
                # print(U1, U2)
                # U1 = re.findall(r'\d+', subdir)[5]
                # U2 = re.findall(r'\d+', subdir)[6]
            if frequencies[0][0] == "l":
                U1 = re.findall(r'\d+', subdir)[3]
                U2 = re.findall(r'\d+', subdir)[4]

            # print(U1, U2)
            if int(U2) < Ulim:
                # print(U1, U2)
                if int(U1) not in U_st and int(U1) != 919:
                    U_st.append(int(U1))
                if int(U2) not in U_sc:
                    U_sc.append(int(U2))

                if int(U1) != 919:
                    if str(U1) not in data[freq].keys():
                        data[freq][str(U1)] = {}
                    if str(U2) not in data[freq][str(U1)].keys():
                        data[freq][str(U1)][str(U2)] = {}

                    data[freq][str(U1)][str(U2)]["iv"] = 0
                    data[freq][str(U1)][str(U2)]["mu"] = 0
                    data[freq][str(U1)][str(U2)]["siv"] = 0
                    data[freq][str(U1)][str(U2)]["gtau"] = 0
                    data[freq][str(U1)][str(U2)]["Z"] = 0
                    data[freq][str(U1)][str(U2)]["scat"] = 0
                    data[freq][str(U1)][str(U2)]["alpha"] = 0
                    data[freq][str(U1)][str(U2)]["doubleocc"] = 0
                    data[freq][str(U1)][str(U2)]["szsz"] = 0
                    data[freq][str(U1)][str(U2)]["exists"] = 0
            
    for subdir, dirs, files in os.walk(PATH):
        # print(files)
        for file in files:
            if frequencies[0][0] == "w":
                U1 = re.findall(r'\d+', subdir)[3]
                U2 = re.findall(r'\d+', subdir)[4]
            if frequencies[0][0] == "l":
                U1 = re.findall(r'\d+', subdir)[3]
                U2 = re.findall(r'\d+', subdir)[4]

            if int(U2) < Ulim and int(U1) < 2220 and int(U1) != 919:
                if beta == 30:
                    if int(U1)<1250:
                        if freq == "w0_4": lpt = "oo_2"
                        if freq == "w0_2": lpt = "oo_2"
                    if int(U1)>1250:
                        if freq == "w0_4": lpt = "oo_3"
                        if freq == "w0_2": lpt = "oo_3"
                if beta == 20:
                    if int(U1)<1250:
                        if freq == "w0_4": lpt = "oo_2"
                        if freq == "w0_2": lpt = "oo_4"
                    if int(U1)>1250:
                        if freq == "w0_4": lpt = "oo_3"
                        if freq == "w0_2": lpt = "oo_3"
                lpt = "oo_5"

                if lpt in file:# or "oo_3" in file:
                    filename = subdir+'/'+file
                    # print(U1,U2,filename)
                    with h5py.File(filename, "r") as f:
                        if 'dmft-last' in f.keys():
                            # print(U1,U2, file)
                            data[freq][str(U1)][str(U2)]["exists"] = 1
                            iv = np.array(f['.axes']['iw'][:])
                            tau = np.array(f['.axes']['tau'][:])
                            mu = np.array(f['dmft-last']['ineq-001']['muimp']['value'])
                            siv = np.array(f['dmft-last']['ineq-001']['siw']['value'])
                            giv = np.array(f['dmft-last']['ineq-001']['giw']['value'])
                            gtau = np.array(f['dmft-last']['ineq-001']['gtau']['value'])
                            # doubleocc = np.array(f['dmft-last']['ineq-001']['ntau-n0']['value'][0,0,0,1,0])
                            doubleocc = np.array(f['dmft-last']['ineq-001']['occ']['value'][0,0,0,1])
                            # sztausz0 = np.array(f['dmft-last']['ineq-001']['sztau-sz0']['value'][0,0,0,1,0])
                            # print(np.array(f['dmft-last']['ineq-001']['ntau-n0']['value'][0,0,0,1,0]))
                            szsz = np.array(np.sum ( f['dmft-last']['ineq-001']['ntau-n0']['value'][0,0,0,0,:] + f['dmft-last']['ineq-001']['ntau-n0']['value'][0,1,0,1,:] - f['dmft-last']['ineq-001']['ntau-n0']['value'][0,0,0,1,:] - f['dmft-last']['ineq-001']['ntau-n0']['value'][0,1,0,0,:] )*beta/1000 )
                            szszb2 = np.array( ( f['dmft-last']['ineq-001']['ntau-n0']['value'][0,0,0,0,500] + f['dmft-last']['ineq-001']['ntau-n0']['value'][0,1,0,1,500] - f['dmft-last']['ineq-001']['ntau-n0']['value'][0,0,0,1,500] - f['dmft-last']['ineq-001']['ntau-n0']['value'][0,1,0,0,500] ) )
                            nch = np.array(np.sum ( f['dmft-last']['ineq-001']['ntau-n0']['value'][0,0,0,0,:] + f['dmft-last']['ineq-001']['ntau-n0']['value'][0,1,0,1,:] + f['dmft-last']['ineq-001']['ntau-n0']['value'][0,0,0,1,:] + f['dmft-last']['ineq-001']['ntau-n0']['value'][0,1,0,0,:] )*beta/1000 )
                            # szszsz = np.array( ( f['dmft-last']['ineq-001']['ntau-n0']['value'][0,0,0,0,:] + f['dmft-last']['ineq-001']['ntau-n0']['value'][0,1,0,1,:] - f['dmft-last']['ineq-001']['ntau-n0']['value'][0,0,0,1,:] - f['dmft-last']['ineq-001']['ntau-n0']['value'][0,1,0,0,:] ) )
                            # ntaunntau = np.array(f['dmft-last']['ineq-001']['ntau-n0']['value'])
                            iv0 = int(iv.shape[0]/2)
                            nn = iv0 + fitpoints
                            ivmax = iv0 + 10
                            iv_limit = iv[iv0:nn].real
                            siv_limit = (siv[0,0,iv0:nn]).imag

                #             popt2, pcov2 = curve_fit(fit_ImSigma_1, iv_limit, siv_limit)
                #             popt2, pcov2 = curve_fit(fit_ImSigma_2, iv_limit, siv_limit)

                            popt2, pcov2 = curve_fit(fit_ImSigma_1, iv_limit, siv_limit)
                            popt2s.append(popt2)

                #             fit_sigma_2 = fit_ImSigma_1(iv_limit, popt2[0], popt2[1])
                #             fit_sigma_2 = fit_ImSigma_2(iv_limit, popt2[0], popt2[1], popt2[2])
                            # fit_sigma_2 = fit_ImSigma_4(iv_limit, popt2[0], popt2[1], popt2[2], popt2[3], popt2[4])

                    data[freq][str(U1)][str(U2)]["mu"] = mu
                    data[freq][str(U1)][str(U2)]["iv"] = iv
                    data[freq][str(U1)][str(U2)]["siv"] = siv
                    data[freq][str(U1)][str(U2)]["giv"] = giv
                    data[freq][str(U1)][str(U2)]["gtau"] = gtau
                    # data[freq][str(U1)][str(U2)]["fit_siv"] = fit_sigma_2
                    data[freq][str(U1)][str(U2)]["Z"] = round(1/(1+popt2[1]), 2)
                    data[freq][str(U1)][str(U2)]["scat"] = round(popt2[0])
                    data[freq][str(U1)][str(U2)]["alpha"] = popt2[1]
                    data[freq][str(U1)][str(U2)]["doubleocc"] = doubleocc
                    data[freq][str(U1)][str(U2)]["nch"] = nch
                    data[freq][str(U1)][str(U2)]["szsz"] = szsz
                    data[freq][str(U1)][str(U2)]["szszb2"] = szszb2
                    data[freq][str(U1)][str(U2)]["gtaubetahalf"] = gtau[0,0,500]
                    
                    # if int(U2)>200:
                    #     print(U1,U2,data[freq][str(U1)][str(U2)]["Z"])
                        

    f = h5py.File(filename, "r")

    U_sc.sort()
    U_st.sort()

    data_np[freq]["Z"] = np.zeros((len(U_st), len(U_sc),3))
    data_np[freq]["scat"] = np.zeros((len(U_st), len(U_sc),3))
    data_np[freq]["alpha"] = np.zeros((len(U_st), len(U_sc),3))
    data_np[freq]["siv0"] = np.zeros((len(U_st), len(U_sc),3), dtype=complex)
    data_np[freq]["siv_diff"] = np.zeros((len(U_st), len(U_sc),3), dtype=complex)
    data_np[freq]["gtaubetahalf"] = np.zeros((len(U_st), len(U_sc),3))
    data_np[freq]["doubleocc"] = np.zeros((len(U_st), len(U_sc),3))
    data_np[freq]["nch"] = np.zeros((len(U_st), len(U_sc),3))
    data_np[freq]["szsz"] = np.zeros((len(U_st), len(U_sc),3))
    data_np[freq]["siv"] = np.zeros((len(U_st), len(U_sc),2000), dtype=complex)
    data_np[freq]["giv"] = np.zeros((len(U_st), len(U_sc),2000), dtype=complex)
    data_np[freq]["mu"] = np.zeros((len(U_st), len(U_sc),1), dtype=complex)

#     print(U_st)
#     print(U_sc)
    k = 0
    for u in U_st:
        if str(u) in data[freq].keys():
            l = 0
            for v in U_sc:
                if str(v) in data[freq][str(u)].keys():
                    if data[freq][str(u)][str(v)]["exists"]==1:
                        # print(u,v)
                        data_np[freq]["Z"][k,l,0] = u
                        data_np[freq]["Z"][k,l,1] = v
                        data_np[freq]["Z"][k,l,2] = data[freq][str(u)][str(v)]["Z"]
        
                        data_np[freq]["scat"][k,l,0] = u
                        data_np[freq]["scat"][k,l,1] = v
                        data_np[freq]["scat"][k,l,2] = data[freq][str(u)][str(v)]["scat"]
        
                        data_np[freq]["alpha"][k,l,0] = u
                        data_np[freq]["alpha"][k,l,1] = v
                        data_np[freq]["alpha"][k,l,2] = data[freq][str(u)][str(v)]["alpha"]
        
                        data_np[freq]["siv0"][k,l,0] = u
                        data_np[freq]["siv0"][k,l,1] = v
                        data_np[freq]["siv0"][k,l,2] = data[freq][str(u)][str(v)]["siv"][0,0,1000]
        
                        data_np[freq]["siv_diff"][k,l,0] = u
                        data_np[freq]["siv_diff"][k,l,1] = v
                        data_np[freq]["siv_diff"][k,l,2] = data[freq][str(u)][str(v)]["siv"][0,0,1001] - data[freq][str(u)][str(v)]["siv"][0,0,1000]
        
                        data_np[freq]["gtaubetahalf"][k,l,0] = u
                        data_np[freq]["gtaubetahalf"][k,l,1] = v
                        data_np[freq]["gtaubetahalf"][k,l,2] = data[freq][str(u)][str(v)]["gtau"][0,0,500]
        
                        data_np[freq]["doubleocc"][k,l,0] = u
                        data_np[freq]["doubleocc"][k,l,1] = v
                        data_np[freq]["doubleocc"][k,l,2] = data[freq][str(u)][str(v)]["doubleocc"]
        
                        data_np[freq]["nch"][k,l,0] = u
                        data_np[freq]["nch"][k,l,1] = v
                        data_np[freq]["nch"][k,l,2] = data[freq][str(u)][str(v)]["nch"]
        
                        data_np[freq]["szsz"][k,l,0] = u
                        data_np[freq]["szsz"][k,l,1] = v
                        data_np[freq]["szsz"][k,l,2] = data[freq][str(u)][str(v)]["szsz"]
                    
                        data_np[freq]["siv"][k,l,:] = data[freq][str(u)][str(v)]["siv"][0,0,:]
                        data_np[freq]["giv"][k,l,:] = data[freq][str(u)][str(v)]["giv"][0,0,:]
                        data_np[freq]["mu"][k,l] = data[freq][str(u)][str(v)]["mu"][0,0]

                    else:
                        # print(u,v, " NO ")
                        data_np[freq]["Z"][k,l,0] = u
                        data_np[freq]["Z"][k,l,1] = v
                        data_np[freq]["Z"][k,l,2] = -0.5

                        data_np[freq]["scat"][k,l,0] = u
                        data_np[freq]["scat"][k,l,1] = v
                        data_np[freq]["scat"][k,l,2] = -0.5
        
                        data_np[freq]["alpha"][k,l,0] = u
                        data_np[freq]["alpha"][k,l,1] = v
                        data_np[freq]["alpha"][k,l,2] = -0.5
        
                        data_np[freq]["siv0"][k,l,0] = u
                        data_np[freq]["siv0"][k,l,1] = v
                        data_np[freq]["siv0"][k,l,2] = -0.5
        
                        data_np[freq]["siv_diff"][k,l,0] = u
                        data_np[freq]["siv_diff"][k,l,1] = v
                        data_np[freq]["siv_diff"][k,l,2] = -0.5
        
                        data_np[freq]["gtaubetahalf"][k,l,0] = u
                        data_np[freq]["gtaubetahalf"][k,l,1] = v
                        data_np[freq]["gtaubetahalf"][k,l,2] = -0.5
        
                        data_np[freq]["doubleocc"][k,l,0] = u
                        data_np[freq]["doubleocc"][k,l,1] = v
                        data_np[freq]["doubleocc"][k,l,2] = 0
        
                        data_np[freq]["nch"][k,l,0] = u
                        data_np[freq]["nch"][k,l,1] = v
                        data_np[freq]["nch"][k,l,2] = 0
        
                        data_np[freq]["szsz"][k,l,0] = u
                        data_np[freq]["szsz"][k,l,1] = v
                        data_np[freq]["szsz"][k,l,2] = 0
                l+=1
        k+=1
    data_np[freq]["Z"][:,:,2][data_np[freq]["Z"][:,:,2]<0] = 0.0
    data_np[freq]["Z"][:,:,2][data_np[freq]["Z"][:,:,2]>1] = 0.0
    
#%%
isozete = []
isozete_idx = []
isoaim = 0.65
tol = 0.05

l = 0
c = [-1,-1,-1]
for v in U_sc:
    m = 100
    k = 0
    for u in U_st:
        # if np.absolute(data_np[frequencies[0]]["Z"][k,l,2]-isoaim) < tol:
        if np.absolute(data_np[frequencies[0]]["Z"][k,l,2]-isoaim) < m:
            # print(k,l, data_np[frequencies[0]]["Z"][k,l,2])
            m = np.absolute(data_np[frequencies[0]]["Z"][k,l,2]-isoaim)
            c = [u,v,data_np[frequencies[0]]["Z"][k,l,2]]
            idx = [k,l]
        k+=1
    if m < 100:
        isozete.append(c)
        isozete_idx.append(np.array(idx))
    l+=1

# Refine
isozete_ref = []
isozete_idx_ref = []
for z, idx in zip(isozete, isozete_idx):
    if abs(isoaim - z[2])<tol:
        isozete_ref.append(z)
        isozete_idx_ref.append(idx)

# Z = 0.65
if isoaim == 0.65:
    del isozete_idx[-1:]
    del isozete_ref[-1:]
# Z = 0.7
if isoaim == 0.7:
    del isozete_idx[-3:]
    del isozete_ref[-3:]

isozete_ref = np.array(isozete_ref, dtype=int)
isozete_idx_ref = np.array(isozete_idx_ref, dtype=int)
isozete_ordered = np.array([isozete_ref[:,1], isozete_ref[:,0] - isozete_ref[:,1]])

args = np.argsort(isozete_ordered[0,:])
isozete_ordered = isozete_ordered[:,args]

args = np.argsort(isozete_idx_ref[0,:])
isozete_idx_ref = isozete_idx_ref[:,args]

print(isozete_idx_ref)
print(len(isozete_idx_ref))

# %%
def read_hk(PATH):

    with open(PATH, 'r', encoding='latin1') as f:
        lines = f.readlines()

    Nk, Natoms, Nbands = map(int, lines[0].strip().split())

    if Natoms != 1 or Nbands != 1:
        raise ValueError("This parser assumes scalar Hamiltonians (Natoms = 1, Nbands = 1)")

    kpts = []
    hks = []

    for i in range(Nk):
        k_line = lines[1 + 2*i].strip()
        h_line = lines[2 + 2*i].strip()

        k = list(map(float, k_line.split()))
        h_real, h_imag = map(float, h_line.split())
        h = complex(h_real, h_imag)

        kpts.append(k)
        hks.append(h)

    return np.array(kpts), np.array(hks)


PATH = "/mnt/scratch/daniel/Data/DynamicalU_Susceptibility/PhaseSpace/Ust_vs_Usc/beta30/ham_path"
k, hk = read_hk(PATH)
print(hk.shape)

# %%capture

niw = 300    # number of Matsubara frequencies
iw_max = niw
wn = np.pi/beta * (2.*np.arange(iw_max) + 1.)
iv = data[freq][str(U_st[0])][str(U_sc[0])]["iv"][1000:1000+iw_max]
nw =  2000   # number of real frequencies
wmax = 6
w = np.linspace(-wmax, wmax, num=nw, endpoint=True)                        

noise_amplitude = 2e-3
err = np.ones_like(wn)*noise_amplitude

siv_spectra = []
sivs = []
siws = []
giw = []

print(f" Total samples for this Z={isoaim} value: ", len(isozete_ref))
t0 = time.perf_counter()
for n,U in enumerate(isozete_ref):#.transpose():
    # if n==0:
        print(U)
        siw = data[freq][str(U[0])][str(U[1])]["siv"][0,0,1000:]
        mu = data[freq][str(U[0])][str(U[1])]["mu"]
        Z = data[freq][str(U[0])][str(U[1])]["Z"]
        plt.figure(0, figsize=(8,4))
        plt.plot(siw.imag[:niw])

        # define the problem
        model = np.ones_like(w)  # flat default model
        model /= np.trapz(model, w)  # normalization, not strictly necessary

        # the problem is defined by: imaginary grid, real grid, data, kernel type
        probl = cont.AnalyticContinuationProblem(im_axis=wn,  # imaginary grid
                                                re_axis=w,  # real grid
                                                im_data=1j * siw.imag[:niw],  # data
                                                kernel_mode='freq_fermionic')  # kernel type
        
        # most basic way to solve the problem
        sol, _ = probl.solve(method='maxent_svd',  # Maxent solver that works with singular value decomp.
                            optimizer='newton',  # newton root finding for the optimization problem
                            alpha_determination='chi2kink',
                            model=model,  # default model
                            stdev=err[:niw],  # standard deviation of the data
                            alpha_start=1e12,  # largest alpha, starting value
                            interactive=False)
        
        sol,_ = probl.solve(method='maxent_svd', alpha_determination='chi2kink', optimizer='newton', stdev=err, model=model)
        sw = cont.GreensFunction(spectrum=sol.A_opt, wgrid=w, kind='fermionic').kkt() + mu[0,0]
        siv_spectra.append(sol.A_opt)
        sivs.append(siw)
        siws.append(sw)
        # giw.append(1 / (w[:,None] - hk[None,:] + mu[0,0] - sw[:,None]))
        giw = (1 / (w[:,None] - hk[None,:] + mu[0,0] - sw[:,None]))

        np.savez_compressed(f"saves_Z0c65_d/spectra_U1_{isozete_ref[n,0]}_U2_{isozete_ref[n,1]}_nw{nw}_wmax{wmax}_niw{niw}_noise_amplitude{noise_amplitude}.npz", w=w, siws=siw, giw=giw)
        print("Saved:, ", f"saves_Z0c65_d/spectra_U1_{isozete_ref[n,0]}_U2_{isozete_ref[n,1]}_nw{nw}_wmax{wmax}_niw{niw}_noise_amplitude{noise_amplitude}.npz")
        # break

dt = time.perf_counter() - t0
print(f"solve took {dt:.3f} s", flush=True)
print( " DONE ")

# %%
