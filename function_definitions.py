import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as spalin
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import os
import time

def gen_s0sxsysz(L): #THIS FUNCTION COPIED FROM PHYSICS 470 COURSE MATERIALS
    sx = sparse.csr_matrix([[0., 1.],[1., 0.]]) 
    sy = sparse.csr_matrix([[0.,-1j],[1j,0.]]) 
    sz = sparse.csr_matrix([[1., 0],[0, -1.]])
    s0_list =[]
    sx_list = [] 
    sy_list = [] 
    sz_list = []
    I = sparse.eye(2**L, format='csr', dtype='complex')
    for i_site in range(L):
        if i_site==0: 
            X=sx 
            Y=sy 
            Z=sz 
        else: 
            X= sparse.csr_matrix(np.eye(2)) 
            Y= sparse.csr_matrix(np.eye(2)) 
            Z= sparse.csr_matrix(np.eye(2))
            
        for j_site in range(1,L): 
            if j_site==i_site: 
                X=sparse.kron(X,sx, 'csr')
                Y=sparse.kron(Y,sy, 'csr') 
                Z=sparse.kron(Z,sz, 'csr') 
            else: 
                X=sparse.kron(X,np.eye(2),'csr') 
                Y=sparse.kron(Y,np.eye(2),'csr') 
                Z=sparse.kron(Z,np.eye(2),'csr')
        sx_list.append(X)
        sy_list.append(Y) 
        sz_list.append(Z)
        s0_list.append(I)

    return s0_list, sx_list,sy_list,sz_list

def gen_spin_operators(L):
    s0_list, sx_list,sy_list,sz_list = gen_s0sxsysz(L)
    spinx_list = [0.5*sx for sx in sx_list]
    spiny_list = [0.5*sy for sy in sy_list]
    spinz_list = [0.5*sz for sz in sz_list]
    return spinx_list,spiny_list,spinz_list

def gen_spinplus_spinminus(spinx_list,spiny_list):
    L = len(spinx_list)
    spinplus_list = []
    spinminus_list = []
    for i in range(L):
        spinx = spinx_list[i]
        spiny = spiny_list[i]
        spinplus_list.append(spinx + 1j*spiny)
        spinminus_list.append(spinx - 1j*spiny)
    return spinplus_list,spinminus_list

def gen_interaction_kdist(op_list, op_list2=[],k=1, bc='obc'): #TAKEN FROM PHYSICS 470 COURSE MATERIALS
    #  returns the interaction \sum_i O_i O_{i+k} 
    L= len(op_list)

    if op_list2 ==[]:
        op_list2=op_list
    H = sparse.csr_matrix(op_list[0].shape)
    Lmax = L if bc == 'pbc' else L-k
    for i in range(Lmax):
        H = H+ op_list[i]*op_list2[np.mod(i+k,L)]
    return H

def gen_current_operator(L):
    #s0_list, sx_list,sy_list,sz_list = gen_s0sxsysz(L)
    spinx_list,spiny_list,spinz_list = gen_spin_operators(L)
    spinplus_list,spinminus_list = gen_spinplus_spinminus(spinx_list,spiny_list)
    #splus_list,sminus_list = gen_splus_sminus(sx_list,sy_list)
    j_op = gen_interaction_kdist(spinplus_list, spinminus_list,k=1, bc='pbc')
    j_op = j_op - gen_interaction_kdist(spinminus_list, spinplus_list,k=1, bc='pbc')
    j_op = (1j/2)*j_op
    return j_op

def construct_HN_Ham(L,bc = 'pbc',g = 0,Delta_1 = 1,Delta_2 = 0):
    spinx_list,spiny_list,spinz_list = gen_spin_operators(L)
    spinplus_list,spinminus_list = gen_spinplus_spinminus(spinx_list,spiny_list)
    
    H = 0.5*np.exp(g)*gen_interaction_kdist(spinplus_list, spinminus_list,k=1, bc=bc)
    H = H + 0.5*np.exp(-g)*gen_interaction_kdist(spinminus_list, spinplus_list,k=1, bc=bc)
    
    H = H + Delta_1*gen_interaction_kdist(spinz_list,k=1,bc=bc)
    H = H + Delta_2*gen_interaction_kdist(spinz_list,k=2,bc=bc)
    return H

def gen_op_total(op_list): #THIS FUNCTION STOLEN FROM PHYSICS 470
    L = len(op_list)
    tot = op_list[0]
    for i in range(1,L): 
        tot = tot + op_list[i] 
    return tot

#The possible eigenvalues are -L,-L+2,-L+4, ... , +L-2, +L
def magnetization_projectors(L,return_dimensions=False,return_indices=False): #I'M NOT SURE IF I WROTE THIS OR TOOK IT FROM PHYSICS 470
    s0,sx,sy,sz = gen_s0sxsysz(L)
    S = gen_op_total(sz)
    diags = S.diagonal()
    projectors = []
    dimensions = []
    indices_list = []
    evals = np.arange(-L,L+2,2,dtype=int)
    for eval in evals:
        indices = np.where(diags == eval)[0]
        dimension = indices.size
        P = sparse.csr_matrix((np.ones(dimension),(indices,np.arange(dimension,dtype=int))),shape=(2**L,dimension))
        projectors.append(P.T)
        dimensions.append(dimension)
    if return_dimensions:
        return projectors,dimensions
    else:
        return projectors

def gen_sym_full(ind, L, S): #THIS FUNCTION IS STOLEN FROM PHYSICS 470 EXCEPT THAT I DELETED THE PARITY SYMMETRY STUFF AND PUT IT THROUGH A CODE BEAUTIFIER, SINCE THE INDENTATION WAS INCONSISTENT
    tol = 0.1 / L
    SSHS = int(np.round(2 * S + 1, 1))
    base = SSHS

    # I am constructing values, i-indices, and j-indices.
    # These allow the construction of sparse projectors.
    ktval = [[] for i in range(L)]
    ktjind = [[] for i in range(L)]
    ktiind = [[] for i in range(L)]
    kcounter = [0 for i in range(L)]

    temind = 0
    counter = 0
    for n in ind:
        # print('n is' + f'{n}')
        b = np.base_repr(n, base=SSHS).zfill(L)
        ind = [n]
        for d in range(1, L + 1):
            # Constructing translated product states
            indx = int(b[d:] + b[:d], SSHS)
            ind.append(indx)

            # Some product states are translations of another.
            # The following if statement eliminates redundancies by selecting
            # a "representative" product state before proceeding.
            if indx < n:
                break
            if indx == n:
                for k in range(0, L):
                    tmp = k * d / L
                    if np.abs(np.round(tmp) - tmp) < tol:
                        # ^Ensure that the momentum phases can "wrap around" between site L-1 and 0.

                        # The above found the indices corresponding to translations of the
                        # representative product state. Below, I find the indices of the spatially-reflected
                        # product states found above, useful for the k=0 and k=pi sectors. Similarly to above, I have
                        # simple checks to select representatives.

                        # If k is not 0 or pi, we can just use the translated states found earlier.
                        ktjind[k].append(ind[:-1])
                        ktiind[k].append([kcounter[k]] * (d))
                        kcounter[k] = kcounter[k] + 1
                        ktval[k].append(
                            (np.exp(2 * np.pi * 1j * k / L) ** np.arange(d))
                            / np.sqrt(d)
                        )
                break

    # Now constructing sparse matrices using the sets of (value, i, j) from above.
    # If a given symmetry sector would have 0 eigenvectors in it, I return a row matrix filled
    # with 0s in its place to avoid exceptions.
    slist = []
    for k in range(L):
        try:
            tem = sparse.csr_matrix(
                (
                    np.concatenate(ktval[k]),
                    (np.concatenate(ktiind[k]), np.concatenate(ktjind[k])),
                ),
                shape=(kcounter[k], SSHS**L),
            )
        except:
            tem = sparse.csr_matrix((1, SSHS**L))
        slist.append(tem)

    return slist

def momentum_projectors_within_M_sector_list(L,m=1): #still projects from the whole Hilbert space, into subspaces with M = m and lattice momentum = k
    s0,sx,sy,sz = gen_s0sxsysz(L)
    S = gen_op_total(sz)
    diags = S.diagonal()
    indices = np.where(diags == 2*m)[0]
    return gen_sym_full(indices,L,1/2) #list, starting with k=0, then k=1,etc.
    
def gen_random_state(rng,L,gave_dimension = False,normalize = True):
    #If gave_dimension is false, will assume it gave the number of lattice sites
    if gave_dimension:
        dimension = L
    else:
        dimension = 2**L
    state = rng.standard_normal(dimension) + 1j*rng.standard_normal(dimension)
    if normalize:
        state = state/np.linalg.norm(state)
    return state
    
def solve_Schrodinger_equation(H,psi_0,t_max):
    def fun(t,psi):
        return -1j*H@psi
    output = scipy.integrate.solve_ivp(fun,(0,t_max),psi_0)
    return output["y"][:,-1]

def biorthogonalize(L,R): #https://joshuagoings.com/2015/04/03/biorthogonalizing-left-and-right-eigenvectors-the-easy-lazy-way/
    M = L@R
    M_L,M_U = scipy.linalg.lu(M,permute_l=True)
    return np.linalg.inv(M_L)@L,R@np.linalg.inv(M_U)

#I know it's sort of redundant to have both of the following two functions, but I subjectively prefer it this way
def get_data_directory(current_directory,folder_name): #Data goes in the folder 'RawData' which which is in the parent directory of the working directory
    parent_directory = os.path.dirname(current_directory)
    data_dir = os.path.join(os.path.join(parent_directory,'RawData'),folder_name)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    return data_dir
def get_fig_directory(current_directory,folder_name): #Figures go in the directory 'Figures' which is in the parent directory of the working directory
    parent_directory = os.path.dirname(current_directory)
    fig_dir = os.path.join(os.path.join(parent_directory,'Figures'),folder_name)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    return fig_dir
    
    
def ED_correlator(op_1,op_2,t,H_list,L,M_projectors):
    num_times = t.size
    num_M_sectors = len(M_projectors)
    results_all_sectors = np.zeros((len(H_list),num_times))
    
    #IN GENERAL, FOR TRACING SPARSE MATRICES, USE csr_matrix.trace(offset=0) INSTEAD OF NP.TRACE
    for M_sector_index,M_projector in enumerate(M_projectors):
        M_sector_H_list = [M_projector@H@np.conj(M_projector.T) for H in H_list]
        M_sector_op_2 = M_projector@op_2@np.conj(M_projector.T)
        M_sector_op_1 = M_projector@op_1@np.conj(M_projector.T)
        
        M_sector_dimension = M_sector_H_list[0].shape[0]
        print("M sector dimension: %i" % M_sector_dimension)
        print("Diagonalizing op_1 within sector")
        t0 = time.time()
        unique_evals,op_1_projectors,op_1_sector_dimensions = diagonalize_operator(M_sector_op_1)
        op_1_sector_dimensions = np.asarray(op_1_sector_dimensions)

        t1 = time.time()
        time_taken = t1 - t0
        print("Time taken to diagonalize op_1 within sector: %.3f" %time_taken)
        print("number of op_1 eigenvalues in this M sector: %i" % len(unique_evals))
        
        for h,M_sector_H in enumerate(M_sector_H_list):
            print("Hamiltonian #%i:" % h)
            evals,U = np.linalg.eig(M_sector_H.toarray())
            D = np.diag(evals)
            U_inv = np.linalg.inv(U)
            U_dag = np.conj(U.T)
            U_dag_inv = np.linalg.inv(U_dag)
            t1 = time.time()
            time_taken = t1 - t0
            print("Time taken to diagonalize M_sector_H: %.3f" % time_taken)
            t0 = time.time()
            
            Delta_t = t[1] - t[0] #THIS CODE ASSUMES T VALUES ARE EVENLY SPACED
            matrix_for_t_evolution = U@np.diag(np.exp(-1j*evals*Delta_t))@U_inv
            conjugate_matrix_for_t_evolution = np.conj(matrix_for_t_evolution.T)

            for n,eigenvalue in enumerate(unique_evals):
                density_matrix = op_1_projectors[n]/np.trace(op_1_projectors[n])
            
                for i,t_value in enumerate(t):
                    if i>0:
                        density_matrix = matrix_for_t_evolution@density_matrix@np.conj(matrix_for_t_evolution.T)
                        density_matrix = density_matrix/np.trace(density_matrix)
                    results_all_sectors[h,i] += (eigenvalue*op_1_sector_dimensions[n]/(2**L))*np.trace(density_matrix@M_sector_op_2)
            
            t1 = time.time()
            time_taken = t1 - t0
            print("Time taken for time evolution with this Hamiltonian in this M sector: %.4f" % time_taken)
                
                
    return results_all_sectors
    
def diagonalize_operator(operator,tol=1e-8):  #As long as the largest numerical error difference between identical eigenvalues is smaller than the smallest genuine gap between eigenvalues,
    #there exists some tolerance that will yield the correct decomposition with no duplicates in unique_evals
    #tolerance is for determining when two eigenvalues are the "same" in the presence of numerical error
    Hilbert_space_dimension = operator.shape[0]
    evals,evecs = np.linalg.eigh(operator.toarray())
    unique_evals = np.zeros(0)
    projectors = []
    sector_dimensions = []
    
    for n in range(Hilbert_space_dimension):
        nth_eval = evals[n]
        nth_evec = evecs[:,n]
        if n==0:
            unique_evals = np.append(unique_evals,nth_eval)
            projectors.append(np.outer(nth_evec,np.conj(nth_evec)))
            sector_dimensions.append(1)
        else:
            if np.min(np.abs(unique_evals - nth_eval)) > tol:
                
                unique_evals = np.append(unique_evals,nth_eval)
                projectors.append(np.outer(nth_evec,np.conj(nth_evec)))
                sector_dimensions.append(1)
            else:
                sector_index = np.argmin(np.abs(unique_evals - nth_eval))
                projectors[sector_index] = projectors[sector_index] + np.outer(nth_evec,np.conj(nth_evec))
                sector_dimensions[sector_index] += 1
    return unique_evals,projectors,sector_dimensions

def diagonalize_spinz_operator_within_M_sectors(M_projectors,M_dimensions,spinz_op):
    op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list = [],[],[]
    num_M_sectors = len(M_dimensions)
    for M_sector_index in range(num_M_sectors):
        M_projector = M_projectors[M_sector_index]
        M_dimension = M_dimensions[M_sector_index]
        M_sector_op_1 = M_projector@spinz_op@np.conj(M_projector.T)
        
        
        if M_sector_index == 0: #spins all down
            unique_evals = [-0.5]
            op_1_projectors = [sparse.eye(M_dimension)]
            op_1_sector_dimensions = [M_dimension]
        elif M_sector_index == num_M_sectors - 1: #spins all up
            unique_evals = [0.5]
            op_1_projectors = [sparse.eye(M_dimension)]
            op_1_sector_dimensions = [M_dimension]
        else:
            unique_evals = [-0.5,0.5]
            spin_down_projector,spin_up_projector =-1*M_sector_op_1 + 0.5*sparse.eye(M_dimension),M_sector_op_1 + 0.5*sparse.eye(M_dimension)
            spin_down_projector.data,spin_up_projector.data = np.round(spin_down_projector.data),np.round(spin_up_projector.data)
            op_1_projectors = [spin_down_projector,spin_up_projector]
            op_1_sector_dimensions = [np.where(spin_down_projector.diagonal() == 1)[0].size,np.where(spin_up_projector.diagonal() == 1)[0].size]
        op_1_evals_list.append(unique_evals)
        op_1_projectors_list.append(op_1_projectors)
        op_1_sector_dimensions_list.append(op_1_sector_dimensions)
    return op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list
    
def typicality_correlator(op_1_evals_list,op_1_projectors_list,op_1_sector_dimensions_list,op_2_list,M_projectors,M_dimensions,t,H,L,rng,
eigen_IPR=None,matrices_of_left_eigenvectors=None):
    if eigen_IPR == None:
        eigen_IPR = [False] * len(M_dimensions)
    num_times = t.size
    num_M_sectors = len(M_dimensions)

    results_all_sectors = np.zeros((len(op_2_list),num_times))
    if max(eigen_IPR):
        IPRs = np.zeros(num_times)
        IPR_Hilbert_space_dimension = sum([dim for M_sector_index,dim in enumerate(M_dimensions) if eigen_IPR[M_sector_index]])
    #print(M_dimensions)
    #print(eigen_IPR)
    #print(IPR_Hilbert_space_dimension)
                
    for M_sector_index in range(num_M_sectors):
        M_projector = M_projectors[M_sector_index]
        M_dimension = M_dimensions[M_sector_index]

        M_sector_H =M_projector@H@np.conj(M_projector.T)
        
        op_1_evals = op_1_evals_list[M_sector_index]
        op_1_projectors = op_1_projectors_list[M_sector_index]
        op_1_sector_dimensions = op_1_sector_dimensions_list[M_sector_index]
        num_op_1_sectors = len(op_1_evals)
        
        M_sector_op_2_list = [M_projector@op_2@np.conj(M_projector.T) for op_2 in op_2_list]
        
        if eigen_IPR[M_sector_index]:
            matrix_of_left_eigenvectors = matrices_of_left_eigenvectors[M_sector_index]
        
        M_sector_state = gen_random_state(rng,M_dimension,gave_dimension=True,normalize=False)
        
        for op_1_sector_index in range(num_op_1_sectors):
            op_1_eval = op_1_evals[op_1_sector_index]
            op_1_projector = op_1_projectors[op_1_sector_index]
            op_1_dimension = op_1_sector_dimensions[op_1_sector_index]
            #print("Op_1 sector dimension within M sector: %i" % op_1_dimension)
            
            state = op_1_projector@M_sector_state
            state = state/np.linalg.norm(state)
            for i,t_value in enumerate(t):
                if i > 0:
                    Delta_t = t[i] - t[i-1]
                    state = solve_Schrodinger_equation(M_sector_H,state,Delta_t)
                state = state/np.linalg.norm(state)
                for j,M_sector_op_2 in enumerate(M_sector_op_2_list):
                    EV = np.conj(state)@M_sector_op_2@state
                    results_all_sectors[j,i] += op_1_eval*op_1_dimension*EV/(2**L)
                if eigen_IPR[M_sector_index]:
                    state_in_eigenbasis = np.conj(matrix_of_left_eigenvectors.T)@state #normalize state (it's already normalized) then find its coefficients in the eigenbasis
                    IPR_one_state = np.sum(np.abs(state_in_eigenbasis)**4)
                    IPRs[i] += op_1_dimension*IPR_one_state/IPR_Hilbert_space_dimension
    if max(eigen_IPR):
        return results_all_sectors,IPRs
    return results_all_sectors

def evals_to_zs(evals):
    z = np.zeros(evals.size,dtype=complex)
    for k,lam in enumerate(evals):
        other_evals = np.delete(evals,k)
        distances = np.abs(other_evals - lam)
        NN_index = np.argmin(distances)
        NN = other_evals[NN_index]
        
        other_evals = np.delete(other_evals,NN_index)
        distances = np.abs(other_evals - lam)
        NNN_index = np.argmin(distances)
        
        NNN = other_evals[NNN_index]
        z[k] = (NN - lam)/(NNN - lam)
    return z

def get_zs_within_sector(H,projector):
    
    sector_H = projector@H@np.conj(projector.T)
    print("Sector dimension: %i" % sector_H.shape[0])

    evals = np.linalg.eigvals(sector_H.toarray())
    z = evals_to_zs(evals)
    return z

def get_biortho_evecs(H):
    evals, matrix_of_left_eigenvectors, matrix_of_right_eigenvectors = scipy.linalg.eig(H, left=True)
    L_matrix = np.conj(matrix_of_left_eigenvectors.T) #goes from left eigenvectors being columns to being rows
    R_matrix = matrix_of_right_eigenvectors
    #For biorthogonality, follow approach from this blog post: https://joshuagoings.com/2015/04/03/biorthogonalizing-left-and-right-eigenvectors-the-easy-lazy-way/
    matrix_of_left_eigenvectors,matrix_of_right_eigenvectors = biorthogonalize(L_matrix,R_matrix)
    matrix_of_left_eigenvectors = np.conj(matrix_of_left_eigenvectors.T) #goes from left eigenvectors being rows to being columns
    return matrix_of_left_eigenvectors,matrix_of_right_eigenvectors