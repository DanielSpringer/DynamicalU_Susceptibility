'''
The two scripts below produce the relevant U11.dat files to be read by w2dynamics for the settings:
Ubare = 10
Usc = 1
w0 = 4
g**2 -> calculated

'''


#%% REAL FREQUENCY VERSION
# import numpy as np
import matplotlib.pyplot as plt

w0 = 4
sigma = 0.5
nw = 100
v = np.linspace(0.5,5,nw)
# peak = 1/(2*np.pi*sigma**2) * 1j*np.exp(-(v-w0_idx)**2/(2*sigma**2))
peak = -1/(2*np.pi*sigma**2) * np.exp(-(v-w0)**2/(2*sigma**2))
data = np.array([v,peak]).transpose()

dw = v[1] - v[0]
shift_U = dw/np.pi * 2 * np.sum( peak / v )
U_shift_aim = -9
g2 = U_shift_aim/shift_U
print(g2)
peak *= g2
shift_U = dw/np.pi * 2 * np.sum( peak / v )
print(shift_U)
plt.plot(v,peak.real)

f = open("U11.dat", "w")
f.write(str(nw) + "\n")
for n,m in enumerate(data):
    if peak[n].real > abs(1e-20):
        peak[n] = -1e-19
    f.write(str(v[n].real) + '  ' + str(peak[n]) + "\n")
f.close()

#%% MATSUBARA FREQUENCY VERSION
import numpy as np
import matplotlib.pyplot as plt

w0 = 4
nw = 1001
niv = np.linspace(0,1000,nw, dtype=int)
dv = 2*np.pi/(30)
iv = niv*dv

U_stat = 10
U_sc = 1
g2 = w0*(U_stat - U_sc) / 2
Usc = U_stat - 2*g2/w0
print(Usc)

Uiv = U_stat + 2*g2*w0 / (-iv**2 - w0**2)
plt.plot(iv,Uiv)

data = np.array([iv,Uiv]).transpose()
f = open("U11.dat", "w")
f.write(str(nw) + "\n")
for n,m in enumerate(data):
    f.write(str(iv[n]) + '  ' + str(Uiv[n]) + "\n")
f.close()

# %%
