import sys
import os

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from function_definitions import *
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as spalin
import numpy as np #USE DTYPE np.cdouble FOR COMPLEX THINGS
from scipy import linalg
import matplotlib.pyplot as plt
import time

plt.rcParams['text.usetex'] = True
rng = np.random.default_rng(2128971964) #WHO YOU GONNA CALL?

folder_name = 'SpinSpinEDvsTypicality'
data_dir = get_data_directory(current_directory,folder_name)

L = 8
g_list = [0.2]#[0,0.2]
D1_list = [1.0]
D2_list = [0]
num_runs = 256

t_max = 1#20
t_step = 0.1
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)
M_projectors,M_dimensions = magnetization_projectors(L,return_dimensions=True)

spinx_list,spiny_list,spinz_list = gen_spin_operators(L)
middle_spin_operator = spinz_list[L//2]

for Delta_1 in D1_list:
    for Delta_2 in D2_list:
        for g in g_list:
            H = construct_HN_Ham(L,g = g,Delta_1 = Delta_1,Delta_2 = Delta_2)

            print("L=%i,Delta_1=%.2f,Delta_2=%.2f,g = %.2f" % (L,Delta_1,Delta_2,g))
            op_1 = middle_spin_operator
            op_2 = op_1
            print("Doing ED")
            t0 = time.time()
            ED_results = ED_correlator(op_1,op_2,t,[H],L,M_projectors)
            t1 = time.time()
            time_taken = t1 - t0
            print("ED took time %.3f" % time_taken)
            ED_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2fED.npy' % (L,g,Delta_1,Delta_2))
            np.save(ED_data_filename,ED_results)
            
            #256 L=10 spin-spin typicality results are already available in the folder RRVsandIPR