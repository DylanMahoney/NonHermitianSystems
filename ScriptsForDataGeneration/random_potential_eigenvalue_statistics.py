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

folder_name = 'RandomPotentialEigenvalueStatistics'
data_dir = get_data_directory(current_directory,folder_name)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--W",type=float)
parser.add_argument("--M",type=int)
args = parser.parse_args()
W = args.W
M = args.M

L=14 #16 or 18 would be nice #for HN: [18,20,22] #L = 18 only takes a few minutes, L = 20 takes about 9 minutes, L = 22 takes about 6 hours +- 1
Delta_1 = 1
num_runs = 100 #Jonas said this was a reasonable number to do

M_projectors,M_dimensions = magnetization_projectors(L,return_dimensions=True)
projector = M_projectors[L//2 + M] #The L//2th sector is the M = 0 sector
dimension = M_dimensions[L//2 + M]
H_without_W_stuff = construct_HN_Ham(L,Delta_1 = Delta_1)

spinx_list,spiny_list,spinz_list = gen_spin_operators(L)
z_all_runs = np.zeros((num_runs,dimension),dtype=np.cdouble)
evals_all_runs = np.zeros((num_runs,dimension),dtype=np.cdouble)
print("L = %i, W=%.2f" % (L,W))
t0 = time.time()
for run in range(num_runs):
    local_potentials = 1j*np.random.uniform(low=-W,high=W,size=L)
    local_potential_term = gen_op_total([local_potentials[r]*spinz_list[r] for r in range(L)])
    H = H_without_W_stuff + local_potential_term

    z,evals = get_zs_within_sector(H,projector)
    z_all_runs[run] = z
    evals_all_runs[run] = evals
z_filename = os.path.join(data_dir,'L=%iW=%.2f,M=%i,%iruns_zs.npy' % (L,W,M,num_runs))
np.save(z_filename,z_all_runs)
evals_filename = os.path.join(data_dir,'L=%iW=%.2f,M=%i,%iruns_evals.npy' % (L,W,M,num_runs))
np.save(evals_filename,evals_all_runs)
t1 = time.time()
time_taken = t1 - t0
print("Time taken to get z's and evals: %.4f" % time_taken)