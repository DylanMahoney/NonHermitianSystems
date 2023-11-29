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

folder_name = 'EDvsTypicality'
fig_dir = get_fig_directory(current_directory,folder_name)
spin_spin_folder_name = 'SpinSpinEDvsTypicality'
spin_data_dir = get_data_directory(current_directory,spin_spin_folder_name)
current_current_folder_name = 'CurrentCurrentEDvsTypicality'
current_data_dir = get_data_directory(current_directory,current_current_folder_name)

spin_spin_typic_folder_name = 'RRVsAndIPRs'
spin_typic_data_dir = get_data_directory(current_directory,spin_spin_typic_folder_name)

figure_width = 12
color_list = get_list_of_colors_I_like(3)



#NOW LET'S COMPARE TYPICALITY TO ED FOR SPIN-SPIN AND CURRENT-CURRENT
L = 10
g_list = [0,0.2]
Delta_1 = 1.5
Delta_2 = 0
num_runs = 200
bc = 'pbc'
type_of_correlation_list = ['SpinSpin','CurrCurr']
correlator_notation_dict = {'SpinSpin':r"$C_{ss}(0,t)$",'CurrCurr':r"$\mathcal{J}(t)$"}
correlation_folder_name_dict = {'SpinSpin':spin_data_dir,'CurrCurr':current_data_dir}

t_max = 20
t_step = 0.1
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)
fig,axs=plt.subplots(2,2,sharex='col',sharey='row')
#color_list = plt.get_cmap('viridis')(np.flip(np.linspace(0,0.8,3)))
for i,g in enumerate(g_list):
    for j,type_of_correlation in enumerate(type_of_correlation_list):
        data_dir = correlation_folder_name_dict[type_of_correlation]
        if type_of_correlation == 'CurrCurr':
            typicality_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data.npy' % (L,g,Delta_1,Delta_2,num_runs))
            all_runs_data = np.load(typicality_data_filename)
        else:
            typicality_data_filename = os.path.join(spin_typic_data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data.npy' % (L,g,Delta_1,Delta_2,num_runs))
            all_runs_data = np.load(typicality_data_filename)
        
        #Let's just do the same-site correlation function to reduce the runtime of the simulation
        average_value = np.mean(all_runs_data,axis=0)
        run_std = np.std(all_runs_data,axis=0)
        run_sem = run_std/np.sqrt(num_runs)
        first_run = all_runs_data[0,:]
        
        ED_data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2fED.npy' % (L,g,Delta_1,Delta_2))
        ED_data = np.load(ED_data_filename)
        #sometimes ED_data has shape (201,) but other times it has shape (1,201)
        if ED_data.shape[0] == 1:
            ED_data = ED_data[0,:]
        
        
        bright_color = color_list[0]
        axs[j,i].plot(t,ED_data,color='k',label='ED result')
        axs[j,i].plot(t,average_value,color=bright_color,linestyle=(0,(2.5,2.5)),label='typicality average') #+- 3 SEM = run std / sqrt(num_runs)
        axs[j,i].fill_between(t,average_value - 2*run_sem,average_value+2*run_sem,color=bright_color,alpha=0.2) #https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html
        axs[j,i].plot(t,first_run,color=bright_color,linestyle=':',label='first run')
        
        axs[j,i].set_ylim(top=1.5,bottom=0.015)
        
        axs[j,i].set_xscale('log')
        axs[j,i].set_yscale('log')
        
        #ax.set_xlabel("time $t$",fontsize=14)
        #ax.set_ylabel(correlator_notation_dict[type_of_correlation],fontsize=14)
        #ax.set_title(r"$g = %.2f$" % g,fontsize=16)

axs[1,0].set_xlabel("time $t$",fontsize=14)
axs[1,1].set_xlabel("time $t$",fontsize=14)

axs[0,0].set_ylabel(r"$C_{ss}(0,t)$",fontsize=14)
axs[1,0].set_ylabel(r"$\mathcal{J}(t)$",fontsize=14)

axs[0,0].legend(markerfirst=False,frameon=False)

add_letter_labels(fig,axs,110,144,[r'$g=0$',r'$g=0.2$',r'$g=0$',r'$g=0.2$'],white_labels=False)

filename = os.path.join(fig_dir,'typicality_vs_ED.png')
fig.set_size_inches(figure_width,figure_width/2)
fig.savefig(filename,dpi=120)
plt.close(fig)