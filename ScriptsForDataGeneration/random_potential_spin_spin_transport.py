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

folder_name = 'RandomPotentialSpinSpinTransport'
data_dir = get_data_directory(current_directory,folder_name)

t_max = 20#20 or 30
t_step = 0.2 #INCREASED FROM 0.1 TO SAVE TIME
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)
L_list = [18]#Jonas said 16 or 18 would be good
Delta_1_list = [1,1.5]#[1,1.5]
Delta_2 = 0
nonHermitian_num_runs = 200#INCREASE THIS to at least 50, ideally 100-200
Hermitian_num_runs = 2 #We should need a lot fewer runs in the Hermitian case
W_list = [0,0.5,1]#[0,0.5,1] #MAYBE CUT ONE OF THESE THREE VALUES LATER

for L in L_list:
    M_projectors,M_dimensions = magnetization_projectors(L,return_dimensions=True)
    spinx_list,spiny_list,spinz_list = gen_spin_operators(L)
    
    op_1 = spinz_list[L//2]
    op_2_list = spinz_list
    op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list = diagonalize_spinz_operator_within_M_sectors(M_projectors,M_dimensions,op_1)
    for Delta_1 in Delta_1_list:

        print("Delta_1=%.2f" % Delta_1)
        H_without_W_stuff = construct_HN_Ham(L,Delta_1 = Delta_1)
        for W in W_list:
            print("W=%.2f" % W)
            if W > 0.0001:
                num_runs = nonHermitian_num_runs
            else:
                num_runs = Hermitian_num_runs
            transport_results = np.zeros((num_runs,len(op_2_list),num_times))
            t0 = time.time()
            for run in range(num_runs): #average over the initial states and the random potentials at the same time, since the two integrals involved in averaging should commute
                print("run: %i" % run)
                local_potentials = 1j*np.random.uniform(low=-W,high=W,size=L)
                local_potential_term = gen_op_total([local_potentials[r]*spinz_list[r] for r in range(L)])
                H = H_without_W_stuff + local_potential_term
                transport_results[run] = typicality_correlator(op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list,op_2_list,M_projectors,M_dimensions,t,H,L,rng)
            t1 = time.time()
            time_per_run_per_tstep = (t1 - t0)/(num_runs*num_times)
            print("Time per run per time step: %.3f" % time_per_run_per_tstep)
            
            spin_spin_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2f,%iruns_all_data.npy' % (L,Delta_1,W,num_runs))
            np.save(spin_spin_filename,transport_results)
            #the below comments are for the H-N model
            #time_per_run t(10) = 10, t(12) = 14, t(14) = 22, t(16) = 42, t(18) = 133, t(20) = 627
            #L = 20 needs more than 4GB (8GB did the trick)