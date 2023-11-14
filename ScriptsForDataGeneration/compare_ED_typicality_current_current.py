#THE g=0.2 CURRCURR TYPICALITY TOOK A LITTLE OVER 7 HOURS WITH OLD CODE
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

folder_name = 'CurrentCurrentEDvsTypicality'
data_dir = get_data_directory(current_directory,folder_name)

L = 10
g_list = [0.2]#[0,0.2]
D1_list = [1.0]
D2_list = [0]
num_runs = 1#256

t_max = 20
t_step = 0.1
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)

M_projectors,M_dimensions = magnetization_projectors(L,return_dimensions=True)
j_op = gen_current_operator(L)

#LOOK OVER THIS

for Delta_1 in D1_list:
    for Delta_2 in D2_list:
        for g in g_list:
            H = construct_HN_Ham(L,g = g,Delta_1 = Delta_1,Delta_2 = Delta_2)

            print("L=%i,Delta_1=%.2f,Delta_2=%.2f,g = %.2f" % (L,Delta_1,Delta_2,g))
            op_1 = j_op
            op_2 = op_1
            print("Doing ED")
            t0 = time.time()
            ED_results = ED_correlator(op_1,op_2,t,H,L,M_projectors)
            t1 = time.time()
            time_taken = t1 - t0
            print("ED took time %.3f" % time_taken)
            ED_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2fED.npy' % (L,g,Delta_1,Delta_2))
            np.save(ED_data_filename,ED_results)
            
            print("Diagonalizing op_1 for typicality")
            t0 = time.time()
            op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list = [],[],[]
            num_M_sectors = len(M_dimensions)
            for M_sector_index in range(num_M_sectors):
                M_projector = M_projectors[M_sector_index]
                M_sector_op_1 = M_projector@op_1@np.conj(M_projector.T)
                unique_evals,op_1_projectors,op_1_sector_dimensions = diagonalize_operator(M_sector_op_1)
                op_1_evals_list.append(unique_evals)
                op_1_projectors_list.append(op_1_projectors)
                op_1_sector_dimensions_list.append(op_1_sector_dimensions)
            t1 = time.time()
            time_taken = t1 - t0
            print("Diagonalizing op_1 for typicality took time %.3f" % time_taken)
            print("Doing typicality")
            t0 = time.time()
            
            correlations_all_runs = np.zeros((num_runs,num_times))
            for run in range(num_runs):
                print("run: %i" % run)
                correlations_all_runs[run] = typicality_correlator(op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list,[op_2],M_projectors,M_dimensions,t,H,L,rng)
            t1 = time.time()
            time_per_run = (t1 - t0)/num_runs
            print("time per run is %.3f" % time_per_run)
            typicality_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data.npy' % (L,g,Delta_1,Delta_2,type_of_correlation,num_runs))
            np.save(typicality_data_filename,correlations_all_runs)