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

GB_number = 2128971964 #WHO YOU GONNA CALL?

plt.rcParams['text.usetex'] = True

folder_name = 'RandomPotentialSpinSpinTransport'
data_dir = get_data_directory(current_directory,folder_name)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--D1",type=float)
parser.add_argument("--W",type=float)
parser.add_argument("--L",type=int)
parser.add_argument("--SS",type=int)

args = parser.parse_args()
Delta_1 = args.D1
W = args.W
L = args.L #Jonas said 16 or 18 would be good
seed_start = args.SS
if seed_start > -1:
    seed = int(str(seed_start)+str(GB_number))
else:
    seed = GB_number

rng = np.random.default_rng(seed) 

t_max = 50
t_step = 0.2
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)
nonHermitian_num_runs = 1000
Hermitian_num_runs = 2 #We should need a lot fewer runs in the Hermitian case

M_projectors,M_dimensions = magnetization_projectors(L,return_dimensions=True)
spinx_list,spiny_list,spinz_list = gen_spin_operators(L)

op_1 = spinz_list[L//2]
op_2_list = spinz_list
op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list = diagonalize_spinz_operator_within_M_sectors(M_projectors,M_dimensions,op_1)

print("Delta_1=%.2f" % Delta_1)
print("W=%.2f" % W)
if W > 0.0001:
    num_runs = nonHermitian_num_runs
else:
    num_runs = Hermitian_num_runs
transport_results = np.zeros((num_runs,len(op_2_list),num_times))
t0 = time.time()
for run in range(num_runs): #average over the initial states and the random potentials at the same time, since the two integrals involved in averaging should commute
    print(run)
    if run > 0 and run % 50 == 0:
        if run > 50:
            old_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2fSS=%i,%iruns_all_data.npy' % (L,Delta_1,W,seed_start,run-50))
            if os.path.exists(old_filename): #from https://www.w3schools.com/python/python_file_remove.asp
                os.remove(old_filename)
        
        spin_spin_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2fSS=%i,%iruns_all_data.npy' % (L,Delta_1,W,seed_start,run))
        np.save(spin_spin_filename,transport_results)
    print("D1=%.2fW=%.2f, run: %i" % (Delta_1,W,run))
    H = construct_random_imaginary_potential_Ham(L,Delta_1,W,rng)
    transport_results[run] = typicality_correlator(op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list,op_2_list,M_projectors,M_dimensions,t,H,L,rng)
        
t1 = time.time()

spin_spin_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2fSS=%i,%iruns_all_data.npy' % (L,Delta_1,W,seed_start,num_runs))

np.save(spin_spin_filename,transport_results)
old_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2fSS=%i,%iruns_all_data.npy' % (L,Delta_1,W,seed_start,num_runs-50))
if os.path.exists(old_filename): #from https://www.w3schools.com/python/python_file_remove.asp
    os.remove(old_filename)

time_per_run_per_tstep = (t1 - t0)/(num_runs*num_times)
print("L=%iD1=%.2fW=%.2f Time per run per time step: %.3f" % (L,Delta_1,W,time_per_run_per_tstep))