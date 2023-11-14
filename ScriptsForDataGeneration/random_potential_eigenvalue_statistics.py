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

L=14#16 or 18 would be nice #for HN: [18,20,22] #L = 18 only takes a few minutes, L = 20 takes about 9 minutes, L = 22 takes about 6 hours +- 1
Delta_1 = 1
W_list = [0.5,1]#[0.5,1] #We already have data for W = 0 from the HN
num_runs = 100#100 #Jonas said this was a reasonable number to do

M_projectors,M_dimensions = magnetization_projectors(L,return_dimensions=True)
projector = M_projectors[L//2] #The L//2th sector is the M = 0 sector
dimension = M_dimensions[L//2]
H_without_W_stuff = construct_HN_Ham(L,Delta_1 = Delta_1)

spinx_list,spiny_list,spinz_list = gen_spin_operators(L)
for W in W_list:
    z_all_runs = np.zeros((num_runs,dimension),dtype=np.cdouble)
    print("L = %i, W=%.2f" % (L,W))
    t0 = time.time()
    for run in range(num_runs):
        local_potentials = 1j*np.random.uniform(low=-W,high=W,size=L)
        local_potential_term = gen_op_total([local_potentials[r]*spinz_list[r] for r in range(L)])
        H = H_without_W_stuff + local_potential_term
        z = get_zs_within_sector(H,projector)
        z_all_runs[run] = z
    z_filename = os.path.join(data_dir,'L=%iW=%.2f,%iruns_zs.npy' % (L,W,num_runs))
    np.save(z_filename,z_all_runs)
    t1 = time.time()
    time_taken = t1 - t0
    print("Time taken to get z's: %.4f" % time_taken)