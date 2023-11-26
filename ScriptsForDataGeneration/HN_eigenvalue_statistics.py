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

folder_name = 'HNEigenvalueStatistics'
data_dir = get_data_directory(current_directory,folder_name)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--g",type=float)
parser.add_argument("--L",type=int)
parser.add_argument("--D1",type=float)
parser.add_argument("--D2",type=float)
args = parser.parse_args()
g = args.g
L = args.L
Delta_1 = args.D1
Delta_2 = args.D2
#OLD CODE: for a single choice of parameters, L = 18 only takes a few minutes, L = 20 takes about 9 minutes, L = 22 takes about 6 hours +- 1
#From slurm, I think the L = 22 actually only takes about 3 hours


m,k = 1,1
print("L = %i, g=%.2f, Delta_1 = %.2f, Delta_2=%.2f Started" % (L,g,Delta_1,Delta_2))
t0 = time.time()
H = construct_HN_Ham(L,g = g,Delta_1 = Delta_1,Delta_2 = Delta_2)
projector = momentum_projectors_within_M_sector_list(L,m=m)[k]
z,evals = get_zs_within_sector(H,projector)
z_filename = os.path.join(data_dir,'L=%i,g=%.2f,D1=%.2f,D2=%.2f_zs.npy' % (L,g,Delta_1,Delta_2))
np.save(z_filename,z)
evals_filename = os.path.join(data_dir,'L=%i,g=%.2f,D1=%.2f,D2=%.2f_evals.npy' % (L,g,Delta_1,Delta_2))
np.save(evals_filename,evals)
t1 = time.time()
time_taken = t1 - t0
print("L = %i, g=%.2f, Delta_1 = %.2f, Delta_2=%.2f Time taken to get z's: %.4f" % (L,g,Delta_1,Delta_2,time_taken))