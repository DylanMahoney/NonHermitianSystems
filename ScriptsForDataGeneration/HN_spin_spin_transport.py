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

folder_name = 'HNSpinSpinTransport'
data_dir = get_data_directory(current_directory,folder_name)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--L",type=int)
parser.add_argument("--g",type=float)
parser.add_argument("--D1",type=float)
parser.add_argument("--D2",type=float)
args = parser.parse_args()
Delta_1 = args.D1
Delta_2 = args.D2
L = args.L
g = args.g

t_max = 30
t_step = 0.2
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)

#L = 22 needs more than 8GB of RAM; 12GB works for a single run
num_runs = 6

print("L = %i" % L)
t0 = time.time()
M_projectors,M_dimensions = magnetization_projectors(L,return_dimensions=True)
spinx_list,spiny_list,spinz_list = gen_spin_operators(L)
t1 = time.time()
time_taken = t1 - t0
print("Got list of spin operators and M sector projectors in time %.3f" % time_taken)
op_1 = spinz_list[L//2]
op_2_list = spinz_list
op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list = diagonalize_spinz_operator_within_M_sectors(M_projectors,M_dimensions,op_1)

print("g=%.2f,Delta_2=%.2f" % (g,Delta_2))
H = construct_HN_Ham(L,g = g,Delta_1 = Delta_1,Delta_2 = Delta_2,spin_operators_list=[spinx_list,spiny_list,spinz_list])
print("Got Hamiltonian")

t0 = time.time()
transport_results = np.zeros((num_runs,len(op_2_list),num_times))
for run in range(num_runs):
    print("run: %i" % run)
    transport_results[run] = typicality_correlator(op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list,op_2_list,M_projectors,M_dimensions,t,H,L,rng)
t1 = time.time()
time_per_run_per_timestep = (t1 - t0)/(num_runs*num_times)
print("Time per run per timestep: %.3f" % time_per_run_per_timestep)

spin_spin_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data.npy' % (L,g,Delta_1,Delta_2,num_runs))
np.save(spin_spin_filename,transport_results)
#for L = 22, about 64 seconds per run per time step