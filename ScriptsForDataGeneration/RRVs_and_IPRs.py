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

folder_name = 'RRVsAndIPRs'
data_dir = get_data_directory(current_directory,folder_name)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--g",type=float)
args = parser.parse_args()
g = args.g

L_list = [10,12,14]#[10,12,14]
D1_list = [1.5]
D2_list = [0]
num_runs = 200

t_max=20
t_step=0.1
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)

for L in L_list:
    M_projectors,M_dimensions = magnetization_projectors(L,return_dimensions=True)
    zero_M_projector = M_projectors[L//2] #The L//2th sector is the M = 0 sector
    zero_M_sector_dimension = M_dimensions[L//2]
    for Delta_1 in D1_list:
        for Delta_2 in D2_list:
            print("L=%i,Delta_1=%.2f,Delta_2=%.2f,g = %.2f" % (L,Delta_1,Delta_2,g))
            H = construct_HN_Ham(L,g = g,Delta_1 = Delta_1,Delta_2 = Delta_2)
            
            spinx_list,spiny_list,spinz_list = gen_spin_operators(L)
            op_1 = spinz_list[L//2]
            op_2 = op_1
            
            t0 = time.time()
            op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list = diagonalize_spinz_operator_within_M_sectors(M_projectors,M_dimensions,op_1)
            
            H_within_M0_sector = zero_M_projector@H@np.conj(zero_M_projector.T)
            
            t0 = time.time()
            evals,matrix_of_right_eigenvectors = scipy.linalg.eig(H_within_M0_sector.toarray())
            matrix_of_left_eigenvectors = np.conj(np.linalg.inv(matrix_of_right_eigenvectors).T)
            t1 = time.time()
            time_taken = t1 - t0
            print("Time taken to get left and right eigenvectors: %.3f" % time_taken)
            
            correlations_all_runs = np.zeros((num_runs,num_times))
            IPRs_all_runs = np.zeros((num_runs,num_times))
            t0 = time.time()
            eigen_IPR = []
            for alpha in range(len(M_dimensions)):
                if alpha == L//2:
                    eigen_IPR.append(True)
                else:
                    eigen_IPR.append(False)
            for run in range(num_runs):
                print("run: %i" % run)
                correlations,IPRs = typicality_correlator(op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list,[op_2],M_projectors,M_dimensions,t,H,L,rng,
                    eigen_IPR=eigen_IPR,matrices_of_left_eigenvectors={L//2: matrix_of_left_eigenvectors})
                correlations_all_runs[run] = correlations
                IPRs_all_runs[run] = IPRs
            t1 = time.time()
            time_per_run_per_tstep = (t1 - t0)/(num_runs*num_times)
            print("Typicality time per run per time step: %.3f" % time_per_run_per_tstep)
            correlations_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data.npy' % (L,g,Delta_1,Delta_2,num_runs))
            np.save(correlations_data_filename,correlations_all_runs)
            IPRs_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data_eigen_IPR.npy' % (L,g,Delta_1,Delta_2,num_runs))
            np.save(IPRs_data_filename,IPRs_all_runs)