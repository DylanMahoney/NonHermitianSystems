#THIS ONE IS FRESH
import sys
import os
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as spalin
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from function_definitions import *
rng = np.random.default_rng(2128971964) #WHO YOU GONNA CALL?

import time
import os
import imageio
import argparse
parser = argparse.ArgumentParser()
from multiprocessing import Pool
import warnings

folder_name = 'RRVsAndIPRs'
data_dir = get_data_directory(current_directory,folder_name)

parser.add_argument("--L",type=int)
parser.add_argument("--g",type=float)
parser.add_argument("--D1",type=float)
args = parser.parse_args()
Delta_1 = args.D1
Delta_2 = 0
L = args.L
g = args.g
num_runs = 150

t_max=20
t_step=0.1
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)

M_projectors,M_dimensions = magnetization_projectors(L,return_dimensions=True)
zero_M_projector = M_projectors[L//2] #The L//2th sector is the M = 0 sector
zero_M_sector_dimension = M_dimensions[L//2]
num_M_sectors = len(M_dimensions)

s0_list, sx_list,sy_list,sz_list = gen_s0sxsysz(L)
op_1 = 0.5*sz_list[L//2]
op_2 = op_1

t0 = time.time()
#op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list = [],[],[]
op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list = diagonalize_spinz_operator_within_M_sectors(M_projectors,M_dimensions,op_1) #I think I can trust this code

#for M_sector_index in range(num_M_sectors):
#    M_projector = M_projectors[M_sector_index]
#    M_sector_op_1 = M_projector@op_1@np.conj(M_projector.T)
#    unique_evals,op_1_projectors,op_1_sector_dimensions = diagonalize_operator(M_sector_op_1)
#    op_1_evals_list.append(unique_evals)
#    op_1_projectors_list.append(op_1_projectors)
#    op_1_sector_dimensions_list.append(op_1_sector_dimensions)

t1 = time.time()
time_taken = t1 - t0
print("Diagonalizing op_1 for typicality took time %.3f" % time_taken)
print("L=%i,Delta_1=%.2f,Delta_2=%.2f,g = %.2f" % (L,Delta_1,Delta_2,g))
H = construct_HN_Ham(L,g = g,Delta_1 = Delta_1,Delta_2 = Delta_2)

H_within_M0_sector = zero_M_projector@H@np.conj(zero_M_projector.T)

t0 = time.time()
evals, matrix_of_left_eigenvectors, matrix_of_right_eigenvectors = scipy.linalg.eig(H_within_M0_sector.toarray(),left=True)
print(np.max(np.abs(np.diag(np.conj(matrix_of_right_eigenvectors.T)@matrix_of_right_eigenvectors) - np.ones(zero_M_sector_dimension))))
print("^Checking if each right eigenvector is normalized^")
#count how many eigenvalues have an imaginary part > 1e-10
print(np.where(np.abs(np.imag(evals)) > 1e-10)[0].size)
print("^number of eigenvalues with |imaginary part| > 1e-10^")
L_matrix = np.conj(matrix_of_left_eigenvectors.T) #goes from left eigenvectors being columns to being rows
R_matrix = matrix_of_right_eigenvectors
#For biorthogonality, follow approach from this blog post: https://joshuagoings.com/2015/04/03/biorthogonalizing-left-and-right-eigenvectors-the-easy-lazy-way/
matrix_of_left_eigenvectors,matrix_of_right_eigenvectors = biorthogonalize(L_matrix,R_matrix)
matrix_of_left_eigenvectors = np.conj(matrix_of_left_eigenvectors.T) #goes from left eigenvectors being rows to being columns

t1 = time.time()
time_taken = t1 - t0
print("Time taken to get left and right eigenvectors: %.3f" % time_taken)

check_that_left_and_right_eigenvectors_are_eigenvectors(H_within_M0_sector,evals,matrix_of_left_eigenvectors,matrix_of_right_eigenvectors)
check_that_eigenvectors_are_biorthogonal(matrix_of_left_eigenvectors,matrix_of_right_eigenvectors)

print(np.max(np.abs(np.diag(np.conj(matrix_of_left_eigenvectors.T)@matrix_of_left_eigenvectors) - np.ones(zero_M_sector_dimension))))
print("^Checking if each left eigenvector is normalized^") #THIS TEST WAS PASSED!
#HYPOTHESIS: THE BIORTHOGONALIZING WAS UNNECESSARY AND WE CAN JUST TAKE MATRIX INVERSE OF MATRIX OF RIGHT EIGENVECTORS
#difference_between_two_methods = np.linalg.inv(matrix_of_right_eigenvectors) - np.conj(matrix_of_left_eigenvectors.T)
#distance_between_the_two = np.sum(np.abs(difference_between_two_methods)**2)
#print(distance_between_the_two)
#Feb 27 2024: I think the above hypothesis is wrong

correlations_all_runs = np.zeros((num_runs,num_times))
#Method RL: normalize such that right eigenvectors are each normalized, then inner product with left
#Method RR: normalize such that right eigenvectors are each normalized, then inner product with right
#Method LR: normalize such that left eigenvectors are each normalized, then inner product with right
#Method LL: normalize such that left eigenvectors are each normalized, then inner product with left

#The left eigenvectors are already normalized:
LR = np.copy(matrix_of_right_eigenvectors)
LL = np.copy(matrix_of_left_eigenvectors)

#Now let's normalize the right eigenvectors
for i in range(zero_M_sector_dimension):
    renormalization_factor = np.linalg.norm(matrix_of_right_eigenvectors[:,i])
    matrix_of_right_eigenvectors[:,i] = matrix_of_right_eigenvectors[:,i]/renormalization_factor
    matrix_of_left_eigenvectors[:,i] = matrix_of_left_eigenvectors[:,i]*renormalization_factor
check_that_eigenvectors_are_biorthogonal(matrix_of_left_eigenvectors,matrix_of_right_eigenvectors)
print("^Hopefully biorthogonality has been maintained^")
RL = np.copy(matrix_of_left_eigenvectors)
RR = np.copy(matrix_of_right_eigenvectors)
IPRs_all_runs = np.zeros((num_runs,4,num_times))
anomalous_norms_all_runs = np.zeros((num_runs,4,num_times))
entropies_all_runs = np.zeros((num_runs,4,num_times))
t0 = time.time()
for run in range(num_runs):
    print("run: %i" % run)
    #correlations,IPRs = spin_spin_typicality_single_sector_eigen_IPR(t,H_within_sector,matrix_of_left_eigenvectors,matrix_of_right_eigenvectors,sector_dimension,middle_spin_projector_within_sector,middle_spin_operator_within_sector)
    correlations,anomalous_norms,IPRs,entropies = typicality_correlator_with_IPRs(op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list,[op_2],M_projectors,M_dimensions,t,H,L,rng,[RL,RR,LR,LL],verbose=True)
    correlations_all_runs[run] = correlations
    IPRs_all_runs[run] = IPRs
    anomalous_norms_all_runs[run] = anomalous_norms #I FORGOT THE [RUN] AT FIRST
    entropies_all_runs[run] = entropies #I FORGOT THE [RUN] AT FIRST
t1 = time.time()
time_per_run = (t1 - t0)/num_runs
print("Typicality time per run: %.3f" % time_per_run)
correlations_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data.npy' % (L,g,Delta_1,Delta_2,num_runs))
np.save(correlations_data_filename,correlations_all_runs)
IPRs_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data_IPR4methods.npy' % (L,g,Delta_1,Delta_2,num_runs))
np.save(IPRs_data_filename,IPRs_all_runs)
anomalous_norms_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data_anomalous_norms4methods.npy' % (L,g,Delta_1,Delta_2,num_runs))
np.save(anomalous_norms_data_filename,anomalous_norms_all_runs)
entropies_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data_entropies4methods.npy' % (L,g,Delta_1,Delta_2,num_runs))
np.save(entropies_data_filename,entropies_all_runs)