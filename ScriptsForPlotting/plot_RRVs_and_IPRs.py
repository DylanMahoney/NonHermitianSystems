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
import matplotlib.transforms as mtransforms
import time

plt.rcParams['text.usetex'] = True
rng = np.random.default_rng(2128971964) #WHO YOU GONNA CALL?
plt.rc('text', usetex=True)
plt.rcParams.update({
    "text.usetex": True,
"font.family": "Computer Modern",
'font.size': 12
})
plt.rc('text.latex', preamble=r'\usepackage{amsmath,braket}')
plt.rcParams['figure.constrained_layout.use'] = True

folder_name = 'RRVsAndIPRs'
fig_dir = get_fig_directory(current_directory,folder_name)
data_dir = get_data_directory(current_directory,folder_name)

color_list = get_list_of_colors_I_like(3)

L_list = [10,12,14]#[10,12,14]
g = 0.2#g_list = [0,0.2]
D1_list = [1,1.5]
Delta_2 = 0
num_runs = 150

t_max=20
t_step=0.1
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)
title_list = ['right normalized, right i.p.','right normalized, left i.p.','left normalized, right i.p.','left normalized, left i.p.']
for eigen_method_index in range(4):
    #Method RL: normalize such that right eigenvectors are each normalized, then inner product with left
    #Method RR: normalize such that right eigenvectors are each normalized, then inner product with right
    #Method LR: normalize such that left eigenvectors are each normalized, then inner product with right
    #Method LL: normalize such that left eigenvectors are each normalized, then inner product with left
    fig,axs = plt.subplots(4,len(D1_list),sharex='col')
    plt.style.use('Solarize_Light2')
    
    for i,Delta_1 in enumerate(D1_list):
        for j,L in enumerate(L_list):
            color = color_list[j]
            IPR_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data_IPR4methods.npy' % (L,g,Delta_1,Delta_2,num_runs))
            IPR_all_runs = np.load(IPR_data_filename)[:,eigen_method_index,:]
            IPR = IPR_all_runs[-1,:] #Let's just show data for the last run because that's all I saved
            #STUPID ERROR
            #IPR = np.mean(IPR_all_runs,axis=0)
            axs[0,i].plot(t,IPR,color=color,label='L = %i' % L)
            axs[0,i].set_ylabel("IPR")
            
            anomalous_norms_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data_anomalous_norms4methods.npy' % (L,g,Delta_1,Delta_2,num_runs))
            #anomalous_norms_all_runs = np.load(anomalous_norms_data_filename)[:,eigen_method_index,:]
            #anomalous_norms = np.mean(anomalous_norms_all_runs,axis=0)
            anomalous_norms_all_runs = np.load(anomalous_norms_data_filename)[eigen_method_index,:]
            anomalous_norms = anomalous_norms_all_runs
            axs[1,i].plot(t,anomalous_norms,color=color,label='L = %i' % L)
            axs[1,i].set_ylabel("\"norm\"")
            
            normalized_IPR = IPR/anomalous_norms
            axs[2,i].plot(t,normalized_IPR,color=color,label='L = %i' % L)
            axs[2,i].set_ylabel("normalized IPR")
            
            entropies_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data_entropies4methods.npy' % (L,g,Delta_1,Delta_2,num_runs))
            #entropies_all_runs = np.load(entropies_data_filename)[:,eigen_method_index,:]
            #entropies = np.mean(entropies_all_runs,axis=0)
            entropies_all_runs = np.load(entropies_data_filename)[eigen_method_index,:]
            entropies = entropies_all_runs
            axs[3,i].plot(t,entropies,color=color,label='L = %i' % L)
            axs[3,i].set_ylabel("entropy")
    
    for ax in axs.flatten():
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
    fig.suptitle(title_list[eigen_method_index])
    filename = os.path.join(fig_dir,'many_statistics_method%i' % eigen_method_index)
    fig.savefig(filename,bbox_inches='tight')
    plt.close(fig)

exit()
    

Delta_1 = 1.5 #change to 1.5
Delta_2 = 0
num_runs = 10#200

t_max=20
t_step=0.1
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)

fig,axs = plt.subplots(3,2,sharex='col',sharey='row')
plt.style.use('Solarize_Light2')

for j,g in enumerate(g_list):
            relative_variance_list = []
            for i,L in enumerate(L_list):
                color = color_list[i]
                correlations_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data.npy' % (L,g,Delta_1,Delta_2,num_runs))
                all_runs_data = np.load(correlations_data_filename)
                central_site_all_runs = all_runs_data
                average_value = np.mean(central_site_all_runs,axis=0)
                variance_of_runs = np.var(central_site_all_runs,axis=0)
                relative_variance_of_runs = variance_of_runs/(average_value**2)
                relative_variance_list.append(relative_variance_of_runs)
                std_of_runs = np.sqrt(variance_of_runs)
                run_sem = std_of_runs/np.sqrt(num_runs)
                
                axs[0,j].plot(t,average_value,color=color,label='L = %i' % L) #+- 2 standard errors of the mean
                axs[0,j].fill_between(t,average_value - 2*run_sem,average_value+2*run_sem,color=color) #https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html
                #alpha = 0.2 isn't working
                print("removed another alpha thingy")
                axs[1,j].plot(t[1:],relative_variance_of_runs[1:],label='L=%i' % L,color=color)
                
                
                IPRs_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data_eigen_IPR.npy' % (L,g,Delta_1,Delta_2,num_runs))
                print("L=%i" % L)
                print(np.load(IPRs_data_filename).shape)
                IPR_all_runs = np.load(IPRs_data_filename)
                print(IPR_all_runs[0,:])
                print(IPR_all_runs[5,:])
                IPR = np.mean(np.load(IPRs_data_filename),axis=0)
                IPR_std = np.std(np.load(IPRs_data_filename),axis=0)
                IPR_sem = IPR_std/np.sqrt(num_runs)
                #print(IPR_sem)
            
                axs[2,j].plot(t,IPR,label='L=%i' % L,color=color)
                #TESTING FILL_BETWEEN
                #axs[2,j].fill_between(t,IPR - 2*IPR_sem, IPR + 2*IPR_sem,alpha=0.75,color=color)
                axs[2,j].fill_between(t,IPR - 2*IPR_sem, IPR + 2*IPR_sem,alpha=1,color=color)
                print("I did the fill_between test")
                #axs[2].axhline(y=1/(2**L/2),label='L=%i random state' % L,color=color,linestyle='--') #random state with middle spin up
                #Since the eigenbasis is not generally orthonormal, I'm not sure that the above line is actually the expected IPR for a random state in the eigen basis.
                
                
            axs[0,j].set_xscale('log')
            axs[0,j].set_yscale('log')
            axs[0,0].set_ylabel(r"$\overline{C(0,t)}$",fontsize=14)
            
            axs[1,0].set_ylabel("$R(t)$",fontsize=14)
            axs[1,j].set_yscale('log')
            axs[2,j].set_yscale('log')
            axs[2,0].set_ylabel("$\overline{I_{\ket{\psi,M=0}(t)}}$",fontsize=14)
            
            
            axs[2,j].set_xlabel(r'time $t$',fontsize=14)
            axs[0,0].legend(markerfirst=False,frameon=False)
            #axs[1].legend(markerfirst=False,frameon=False)
            #axs[2].legend(markerfirst=False,frameon=False)
            

add_letter_labels(fig,axs,110,144,[r'$g=0$',r'$g=0.2$',r'$g=0$',r'$g=0.2$',r'$g=0$', r'$g=0.2$'],white_labels=False)
filename = os.path.join(fig_dir,'RRVs_and_IPRs_normalized_right_eigenvecs.pdf')
fig.set_size_inches(figure_width, figure_width/2)
fig.savefig(filename,dpi=120)
plt.close(fig)