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

folder_name = 'EDvsTypicality'
fig_dir = get_fig_directory(current_directory,folder_name)
spin_spin_folder_name = 'SpinSpinEDvsTypicality'
spin_data_dir = get_data_directory(current_directory,spin_spin_folder_name)

spin_spin_typic_folder_name = 'RRVsAndIPRs'
spin_typic_data_dir = get_data_directory(current_directory,spin_spin_typic_folder_name)

L = 10
g_list = [0,0.2]
Delta_1 = 1.5
Delta_2 = 0
num_runs = 200
bc = 'pbc'
use_this_string_later = r"$C(0,t)$"

t_max = 20
t_step = 0.1
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)
fig, axs = plt.subplots(1,2,sharey=True)
plt.style.use('Solarize_Light2')
alphabet = ['a','b','c','d']

for i,g in enumerate(g_list):
    typicality_data_filename = os.path.join(spin_typic_data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data.npy' % (L,g,Delta_1,Delta_2,num_runs))
    all_runs_data = np.load(typicality_data_filename)
    average_value = np.mean(all_runs_data,axis=0)
    first_run = all_runs_data[0,:]
    
    ED_data_filename = os.path.join(spin_data_dir,'L=%ig=%.2fD1=%.2fD2=%.2fED.npy' % (L,g,Delta_1,Delta_2))
    ED_data = np.load(ED_data_filename)
    #sometimes ED_data has shape (201,) but other times it has shape (1,201)
    if ED_data.shape[0] == 1:
        ED_data = ED_data[0,:]
    axs[i].plot(t[::5],ED_data[::5],'.',markersize=10,color='black',label='ED')
    axs[i].plot(t,average_value,label='$\overline{C_{\psi}(0,t)}$, $200$ runs')
    axs[i].plot(t,first_run,'--',label='$C_{\psi}(0,t)$')
    
    axs[i].set_xlabel(r'{\Large time $t$}')
    
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')
    
    axs[i].set_ylim(top=0.3,bottom=0.01)
    axs[i].set_xlim(right=30,left=0.4)
    axs[i].tick_params(which='both',direction='in')

axs[0].text(6,0.18,r'{\Large\textbf{(a)} $g = 0$}')
axs[1].text(6,0.18,r'{\Large\textbf{(b)} $g = 0.2$}')
axs[0].set_ylabel(r'\Large $C(0,t)$')
axs[0].legend(frameon=False,loc='lower left')

x_left, x_right = axs[0].get_xlim()
y_low, y_high = axs[0].get_ylim()
ratio = 0.0085
for ax in axs:
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

filename = os.path.join(fig_dir,'typicality_vs_ED_spin_spin.pdf')
fig.savefig(filename,bbox_inches='tight')
plt.close(fig)
