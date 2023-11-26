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
import pickle

plt.rcParams['text.usetex'] = True
rng = np.random.default_rng(2128971964) #WHO YOU GONNA CALL?

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--L",type=int)
args = parser.parse_args()
L = args.L

folder_name = 'DiagonalizedJ'
data_dir = os.path.join(get_data_directory(current_directory,folder_name),'L=%i'%L)
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

t0 = time.time()
M_projectors,M_dimensions = magnetization_projectors(L,return_dimensions=True)
op_1 = gen_current_operator(L)

op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list = [],[],[]
num_M_sectors = len(M_dimensions)
for M_sector_index in range(num_M_sectors):
    M_projector = M_projectors[M_sector_index]
    M_sector_op_1 = M_projector@op_1@np.conj(M_projector.T)
    unique_evals,op_1_projectors,op_1_sector_dimensions = diagonalize_operator(M_sector_op_1)
    op_1_evals_list.append(unique_evals)
    op_1_projectors_list.append(op_1_projectors)
    op_1_sector_dimensions_list.append(op_1_sector_dimensions)

data_to_save = [op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list]
filename = os.path.join(data_dir,'diagonalized_j.pickle')

with open(filename,'wb') as f:
    pickle.dump(data_to_save, f, pickle.HIGHEST_PROTOCOL)
    
t1 = time.time()
time_taken = t1 - t0
print("For L = %i, diagonalizing j_op within M sectors took time %.3f" % (L,time_taken))