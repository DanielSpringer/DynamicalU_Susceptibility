import os
import sys
import numpy as np
import time
import re
import glob
import os

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
            if (ust-usc>=0) and (ust-usc<=12.40):
                print(round(ust,1), round(usc,1), round(ust-usc,1))
                lam = w0 * (ust - usc) / 2
                mu = ust / 2

                PATH = os.getcwd() #os.system('pwd')

                prompt = "mkdir -p " + str(int(ust*100))
                os.system(prompt)
                os.chdir(str(int(ust*100)))

                prompt = "mkdir -p " + str(int(usc*100))
                os.system(prompt)
                os.chdir(str(int(usc*100)))

                # os.system("rm *slrm")
                os.system("cp ../../../Create* .")
                os.system("cp ../../../ham .")
                os.system("cp ../../../*slrm .")
                # os.system("rm slurm*")
                

                files = os.listdir('.')
                # print(files)
                fileold = "NONE"

                pattern = re.compile(r'^oo_(\d+)\b')
                best = None
                best_num = -1
                for s in files:
                    m = pattern.match(s)
                    if not m:
                        continue
                    n = int(m.group(1))
                    if n > best_num:
                        best_num = n
                        fileold = s

                prompt = "./Create_Uw " + str(w0) + " " + str(lam)
                os.system(prompt)
                prompt = "./Create_Parametersin " + str(ust) + " " + str(mu) + " " + Steps + " " + Nmeas + " " + Ncorr + " " + fileold + " " + str(beta) + " " + str(J) + " " + str(V) + " " + CurrentFile + " " + str(readold) + " " + str(MeasSusz) + " " + str(Uw) + " " + str(StatSteps) + " " + str(N4wf)
                os.system(prompt)

                # time.sleep(0.7)
                
                pattern = "oo_4*"

                # find all filesystem entries matching oo_1*
                matches = glob.glob(pattern)
                # keep only regular files (optional, remove os.path.isfile if you don’t care)
                pexist = [f for f in matches if os.path.isfile(f)]
                if not pexist:
                    # print("No files starting with oo_1 found")
                    # os.system("rm slurm*")
                    if "0512" in config["partition"]:
                        os.system("sbatch vsc_0512.slrm")
                    if "0512_20" in config["partition"]:
                        os.system("sbatch vsc_0512_20.slrm")
                    if "1024" in config["partition"]:
                        os.system("sbatch vsc_1024.slrm")
                    if "2048" in config["partition"]:
                        os.system("sbatch vsc_2048.slrm")
                    if "dev5" in config["partition"]:
                        os.system("sbatch vsc_dev.slrm")

                    if "hclm" in config["partition"]:
                        os.system("sbatch hclm.slrm")
                    if "hclm12" in config["partition"]:
                        os.system("sbatch hclm12.slrm")


#                pattern = "oo_3*"
#                # find all filesystem entries matching oo_1*
#                matches = glob.glob(pattern)
#                # keep only regular files (optional, remove os.path.isfile if you don’t care)
#                pexist = [f for f in matches if os.path.isfile(f)]
#                if pexist:
#                    # print("No files starting with oo_1 found")
#                    # os.system("rm slurm*")
#                    if "0512" in config["partition"]:
#                        os.system("sbatch vsc_0512.slrm")
#                    if "0512_20" in config["partition"]:
#                        os.system("sbatch vsc_0512_20.slrm")
#                    if "1024" in config["partition"]:
#                        os.system("sbatch vsc_1024.slrm")
#                    if "2048" in config["partition"]:
#                        os.system("sbatch vsc_2048.slrm")
#                    if "dev5" in config["partition"]:
#                        os.system("sbatch vsc_dev.slrm")
#                    if "hclm" in config["partition"]:
#                        os.system("sbatch hclm.slrm")
#                    if "hclm12" in config["partition"]:
#                        os.system("sbatch hclm12.slrm")

                os.chdir("../../")


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
            os.system("cp ../../../*slrm .")
            # os.system("rm vsc*")

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
            
            time.sleep(0.2)
            if "0512" in config["partition"]:
                os.system("sbatch w2dyn_zen0512.slrm")
            if "0512_20" in config["partition"]:
                os.system("sbatch w2dyn_zen0512_20.slrm")
            if "1024" in config["partition"]:
                os.system("sbatch w2dyn_zen1024.slrm")
            if "1024_10" in config["partition"]:
                os.system("sbatch w2dyn_zen1024_10.slrm")
            if "2048" in config["partition"]:
                os.system("sbatch w2dyn_zen2048.slrm")
            if "2048_20" in config["partition"]:
                os.system("sbatch w2dyn_zen2048_20.slrm")
            if "2048_10" in config["partition"]:                                                                                                                                                   
                os.system("sbatch w2dyn_zen2048_10.slrm")
            if "A100" in config["partition"]:
                os.system("sbatch w2dyn_A100.slrm")
            if "dev5" in config["partition"]:
                os.system("sbatch w2dyn_dev.slrm")

            if "hclm" in config["partition"]:
                os.system("sbatch hclm.slrm")

            os.chdir("../../")
