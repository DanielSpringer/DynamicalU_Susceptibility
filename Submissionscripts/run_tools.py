import os
import sys
import numpy as np


def fixed_UscUst(config, U_sc, U_st, w0_grid, w0_min, w0_max):
    
    for k, v in config.items():
        globals()[k]=v
    
    # prompt = "mkdir -p UscUst_"+str(int(U_sc))+"_"+str(int(U_st))
    # os.system(prompt)
    # os.chdir("UscUst_"+str(int(U_sc))+"_"+str(int(U_st)))
    prompt = "mkdir -p w0lam_"+str(int(round(w0_min,2)*100))+"_"+str(int(round(w0_max,2)*100))
    os.system(prompt)
    os.chdir("w0lam_"+str(int(round(w0_min,2)*100))+"_"+str(int(round(w0_max,2)*100)) )

    for w0 in w0_grid:
        lam = w0 * (U_st - U_sc) / 2
        if w0 < 1:
            print(round(w0,2), round(lam,2))
        if w0 > 1:
            print(int(w0), round(lam))
        mu = U_st / 2

        PATH = os.getcwd() #os.system('pwd')

        prompt = "mkdir -p w0_" + str(int(round(w0,2)*100)) + "_lam_" + str(int(round(lam,2)*100))
        os.system(prompt)
        os.chdir("w0_" + str(int(round(w0,2)*100)) + "_lam_" + str(int(round(lam,2)*100)) )


        os.system("cp ../../Create* .")
        os.system("cp ../../ham .")
        os.system("cp ../../hclm.slrm .")
        # # os.system("rm slurm*")
        # # os.system("rm oo_1*")
        # # os.system("rm vsc*")
        

        files = os.listdir('.')
        # print(files)
        fileold = "NONE"
        for f in files:
            # if "oo_1" in f:
            #     cmd = "mv "+f+" oo_0a.hdf5"
            #     os.system(cmd)
                # cmd = ". ~/Xinstalls/w2dynamics/w2dynamics/hgrep oo_0a.hdf5 siw : 1 1 1 0"
                # os.system(cmd)
                
            if "oo_0a" in f:
                fileold = f
            if "oo_0b" in f:
                fileold = f
            if "oo_2" in f:
                fileold = f
            if "oo_3" in f:
                fileold = f

        prompt = "./Create_Uw " + str(w0) + " " + str(lam)
        os.system(prompt)
        prompt = "./Create_Parametersin " + str(U_st) + " " + str(mu) + " " + Steps + " " + Nmeas + " " + Ncorr + " " + fileold + " " + str(beta) + " " + str(J) + " " + str(V) + " " + CurrentFile + " " + str(readold) + " " + str(MeasSusz) + " " + str(Uw) + " " + str(StatSteps) + " " + str(N4wf)
        os.system(prompt)
        os.system("sbatch hclm.slrm")

        os.chdir("../")


def fixed_omega(config, U_st_grid, U_sc_grid, w0):
    
    for k, v in config.items():
        globals()[k]=v
    
    prompt = "mkdir -p w0_" + str(int(w0))
    os.system(prompt)
    os.chdir("w0_"+str(int(w0)))

    for ust in U_st_grid:
        for usc in U_sc_grid:
            print(ust, usc)
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
            os.system("cp ../../../ham .")
            os.system("cp ../../../hclm.slrm .")
            # os.system("rm slurm*")
            # os.system("rm oo_2*")
            # os.system("rm vsc*")
            # os.system("rm *")
                        
            # # # os.system("rm oo_1-2024-04-26-Fri*")

            files = os.listdir('.')
            # print(files)
            fileold = "NONE"
            for f in files:
                # if "oo_1" in f:
                #     cmd = "mv "+f+" oo_0a.hdf5"
                #     os.system(cmd)
                    # cmd = ". ~/Xinstalls/w2dynamics/w2dynamics/hgrep oo_0a.hdf5 siw : 1 1 1 0"
                    # os.system(cmd)
                    
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
            # os.system("sbatch hclm.slrm")

            os.chdir("../../")

            # prompt = "rm -r " + str(int(usc*100))
            # os.system("pwd")
            # os.system(prompt)
            # os.chdir("../")


def fixed_lambda(config, U_st_grid, U_sc_grid, set_lam):
    
    for k, v in config.items():
        globals()[k]=v
    
    prompt = "mkdir -p lam_" + str(int(set_lam))
    os.system(prompt)
    os.chdir("lam_"+str(int(set_lam)))

    for ust in U_st_grid:
        for usc in U_sc_grid:
            print(ust, usc)
            # lam = w0 * (ust - usc) / 2
            if (ust - usc) > 0:
                lam = set_lam
                w0 = 2 * lam / (ust - usc)
                Uw = 1
            else:
                print("  NO SCREENING  ")
                Uw = 0
                w0 = 1
                lam = 1
            mu = ust / 2

            PATH = os.getcwd() #os.system('pwd')

            prompt = "mkdir -p " + str(int(ust*100))
            os.system(prompt)
            os.chdir(str(int(ust*100)))

            prompt = "mkdir -p " + str(int(usc*100))
            os.system(prompt)
            os.chdir(str(int(usc*100)))

            os.system("cp ../../../Create* .")
            os.system("cp ../../../ham .")
            os.system("cp ../../../hclm.slrm .")
            # os.system("rm slurm*")
            # os.system("rm oo_1*")
            # os.system("rm vsc*")
            
            # # # os.system("rm oo_1-2024-04-26-Fri*")

            files = os.listdir('.')
            # print(files)
            fileold = "NONE"
            for f in files:
                # if "oo_1" in f:
                #     cmd = "mv "+f+" oo_0a.hdf5"
                #     os.system(cmd)
                    # cmd = ". ~/Xinstalls/w2dynamics/w2dynamics/hgrep oo_0a.hdf5 siw : 1 1 1 0"
                    # os.system(cmd)
                    
                if "oo_0a" in f:
                    fileold = f
                if "oo_0b" in f:
                    fileold = f
                if "oo_2" in f:
                    fileold = f
                if "oo_3" in f:
                    fileold = f

            prompt = "./Create_Uw " + str(w0) + " " + str(lam)
            os.system(prompt)
            prompt = "./Create_Parametersin " + str(ust) + " " + str(mu) + " " + Steps + " " + Nmeas + " " + Ncorr + " " + fileold + " " + str(beta) + " " + str(J) + " " + str(V) + " " + CurrentFile + " " + str(readold) + " " + str(MeasSusz) + " " + str(Uw) + " " + str(StatSteps) + " " + str(N4wf)
            os.system(prompt)
            os.system("sbatch hclm.slrm")

            os.chdir("../../")
