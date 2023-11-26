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

folder_name = 'CurrentCurrentEDvsTypicality'
data_dir = get_data_directory(current_directory,folder_name)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--g",type=float)
args = parser.parse_args()
g = args.g

L = 10
D1_list = [1.0]
D2_list = [0]
num_runs = 200

t_max = 20
t_step = 0.1
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)

M_projectors,M_dimensions = magnetization_projectors(L,return_dimensions=True)
j_op = gen_current_operator(L)

pickle_filename = os.path.join(os.path.join(get_data_directory(current_directory,'DiagonalizedJ'),'L=%i'%L),'diagonalized_j.pickle')

with open(pickle_filename,'rb') as f:
    data_saved = pickle.load(f)

op_1_evals_list = data_saved[0]
op_1_projectors_list = data_saved[1]
op_1_sector_dimensions_list = data_saved[2]

for Delta_1 in D1_list:
    for Delta_2 in D2_list:
        H = construct_HN_Ham(L,g = g,Delta_1 = Delta_1,Delta_2 = Delta_2)

        print("L=%i,Delta_1=%.2f,Delta_2=%.2f,g = %.2f" % (L,Delta_1,Delta_2,g))
        op_1 = j_op
        op_2 = op_1
        print("Doing ED")
        t0 = time.time()
        ED_results = ED_correlator(op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list,op_1,op_2,t,H,L,M_projectors)
        t1 = time.time()
        time_taken_per_time_step = (t1 - t0)/num_times
        print("ED took time %.3f per time step" % time_taken_per_time_step)
        ED_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2fED.npy' % (L,g,Delta_1,Delta_2))
        np.save(ED_data_filename,ED_results)
        
        print("Doing typicality")
        t0 = time.time()
        
        correlations_all_runs = np.zeros((num_runs,num_times))
        for run in range(num_runs):
            print("run: %i" % run)
            correlations_all_runs[run] = typicality_correlator(op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list,[op_2],M_projectors,M_dimensions,t,H,L,rng)
        t1 = time.time()
        time_per_run_per_timestep = (t1 - t0)/(num_runs*num_times)
        print("time per run per time step is %.3f" % time_per_run_per_timestep)
        typicality_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data.npy' % (L,g,Delta_1,Delta_2,num_runs))
        np.save(typicality_data_filename,correlations_all_runs)