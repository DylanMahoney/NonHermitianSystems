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
    "font.family": "Computer Modern"
})
plt.rc('text.latex', preamble=r'\usepackage{amsmath,braket}')
plt.rcParams['figure.constrained_layout.use'] = True

folder_name = 'RRVsAndIPRs'
fig_dir = get_fig_directory(current_directory,folder_name)
data_dir = get_data_directory(current_directory,folder_name)

figure_width = 12
color_list = get_list_of_colors_I_like(3)

L_list = [10,12,14]
g_list = [0,0.2]
Delta_1 = 1.5
Delta_2 = 0
num_runs = 200

t_max=20
t_step=0.1
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)

fig,axs = plt.subplots(3,2,sharex='col',sharey='row')

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
                axs[0,j].fill_between(t,average_value - 2*run_sem,average_value+2*run_sem,color=color,alpha=0.2) #https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html
                
                axs[1,j].plot(t[1:],relative_variance_of_runs[1:],label='L=%i' % L,color=color)
                
                
                IPRs_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data_eigen_IPR.npy' % (L,g,Delta_1,Delta_2,num_runs))
                IPR = np.mean(np.load(IPRs_data_filename),axis=0)
            
                axs[2,j].plot(t,IPR,label='L=%i' % L,color=color)
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
filename = os.path.join(fig_dir,'RRVs_and_IPRs.png')
fig.set_size_inches(figure_width, figure_width/2)
fig.savefig(filename,dpi=120)
plt.close(fig)