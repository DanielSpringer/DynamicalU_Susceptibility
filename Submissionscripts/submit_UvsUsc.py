#%%
import os
import sys
import numpy as np
#os.system('pwd')

# sys.argv[1]
Steps="30"
Nmeas="5e4"
Ncorr="1e2"
beta=20
CurrentFile="oo_1"
readold=0
MeasSusz=0
Uw=1
StatSteps=0
N4wf=20
N4wb=0

U_st_min = 4.0
U_st_max = 10.0
U_st_steps = 7
U_st_grid = np.linspace(U_st_min, U_st_max, U_st_steps)

J = 0
V = 0
w0 = 4.0
prompt = "mkdir -p w0_" + str(int(w0))
os.system(prompt)
os.chdir("w0_"+str(int(w0)))

U_sc_min = 0.0
U_sc_max = 2.0
U_sc_steps = 3
U_sc_grid = np.linspace(U_sc_min, U_sc_max, U_sc_steps)


for ust in U_st_grid:
    for usc in U_sc_grid:
        lam = w0 * (ust - usc) / 2
        mu = ust / 2

        PATH = os.getcwd() #os.system('pwd')

        prompt = "mkdir -p " + str(int(ust*100))
        os.system(prompt)
        os.chdir(str(int(ust*100)))

        prompt = "mkdir -p " + str(int(usc*100))
        os.system(prompt)
        os.chdir(str(int(usc*100)))

        os.system("cp ../../../Create* .")
        # os.system("cp ../../../hk* .")
        os.system("cp ../../../hclm* .")
        # os.system("rm oo_2* Uw_PhaseSpace* ctqmc*")

        files = os.listdir('.')
        # print(files)
        fileold = "NONE"
        for f in files:
            if "oo_1" in f:
                fileold = f
            if "oo_2" in f:
                fileold = f
            if "oo_3" in f:
                fileold = f

        prompt = "./Create_Uw " + str(w0) + " " + str(lam)
        os.system(prompt)
        prompt = "./Create_Parametersin " + str(ust) + " " + str(mu) + " " + Steps + " " + Nmeas + " " + Ncorr + " " + fileold + " " + str(beta) + " " + str(J) + " " + str(V) + " " + CurrentFile + " " + str(readold) + " " + str(MeasSusz) + " " + str(Uw) + " " + str(StatSteps) + " " + str(N4wf)
        os.system(prompt)
        os.system("qsub hclm.sh")
        os.chdir("../../")
        # break


#%%
import numpy as np
U_st_min = 2.0
U_st_max = 10.0
U_st_steps = 11
U_st_grid = np.linspace(U_st_min, U_st_max, U_st_steps)

U_sc_min = 0.0
U_sc_max = 2.0
U_sc_steps = 11
U_sc_grid = np.linspace(U_sc_min, U_sc_max, U_sc_steps)

print(U_st_grid)
print(U_sc_grid)