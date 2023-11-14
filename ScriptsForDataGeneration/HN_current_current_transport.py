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

folder_name = 'HNCurrentCurrentTransport'
data_dir = get_data_directory(current_directory,folder_name)

t_max = 50
t_step = 0.5
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)
L_list = [8,10,12]#bring up to 14 or so
g_list = [0,0.1,0.2]#[0,0.1,0.2]
Delta_1 = 1.5
Delta_2_list = [0,1.5]#[0,1.5]

for L in L_list:
    M_projectors,M_dimensions = magnetization_projectors(L,return_dimensions=True)
    j_op = gen_current_operator(L)
    H_list = []
    H_index_dictionary = {}
    for g in g_list:
        for Delta_2 in Delta_2_list:
            H = construct_HN_Ham(L,g = g,Delta_1 = Delta_1,Delta_2 = Delta_2)
            H_index_dictionary[(g,Delta_2)] = len(H_list)
            H_list.append(H)
            
    t0 = time.time()
    current_current_results = ED_correlator(j_op,j_op,t,H_list,L,M_projectors)
    t1 = time.time()
    time_taken = (t1 - t0)/(num_times*len(H_list))
    print("Doing current-current ED took %.3f per Hamiltonian per time_step" % time_taken)
    #testing with just a single Hamiltonian and only 2 time steps, this will be an overestimate, for L = 10 this is 1.591, for L = 12 this is 105
    for g in g_list:
        for Delta_2 in Delta_2_list:
            
            current_current_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2fcurrent_current.npy' % (L,g,Delta_1,Delta_2))
            np.save(current_current_filename,current_current_results[H_index_dictionary[(g,Delta_2)]])
