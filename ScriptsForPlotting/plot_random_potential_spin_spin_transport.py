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

folder_name = 'RandomPotentialSpinSpinTransport'
fig_dir = get_fig_directory(current_directory,folder_name)
data_dir = get_data_directory(current_directory,folder_name)

figure_width = 12

#FIRST LET'S PLOT THE SAME-SITE CORRELATION AS A FUNCTION OF TIME
fig,axs = plt.subplots(1,2,sharey=True)
fig.set_size_inches(figure_width, figure_width/2)

first_index = 5 #Later we might want to not show early times

Delta_1_list = [1,1.5]
W_list = [0,0.5]#[0,0.5,1]
nonHermitian_num_runs = 200#INCREASE THIS to at least 50, ideally 100-200
Hermitian_num_runs = 2 #We should need a lot fewer runs in the Hermitian case

color_list = get_list_of_colors_I_like(len(W_list))

L = 18
t_max = 20#20 or 30
t_step = 0.2 #INCREASED FROM 0.1 TO SAVE TIME
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)

for i,Delta_1 in enumerate(Delta_1_list):
    #Do stuff to the ith Axis object
    for j,W in enumerate(W_list):
        if W < 0.0001:
            num_runs = Hermitian_num_runs
        else:
            num_runs = nonHermitian_num_runs
        data_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2f,%iruns_all_data.npy' % (L,Delta_1,W,num_runs))
        data = np.load(data_filename)
        if W < 0.0001:
            axs[i].plot(t[first_index:],data[0,L//2,first_index:],label="W = %.1f" % W, color = color_list[j])
            for k in range(1,Hermitian_num_runs):
                axs[i].plot(t[first_index:],data[k,L//2,first_index:],color=color_list[j])
        else:
            same_site_data = data[:,L//2,:]
            average_data = np.mean(same_site_data,axis=0)
            std = np.std(same_site_data,axis=0)
            sem = std/np.sqrt(num_runs)
            axs[i].plot(t[first_index:],average_data[first_index:],label="W=%.1f" % W,color=color_list[j])
            axs[i].fill_between(t[first_index:],(average_data - 2*sem)[first_index:], (average_data + 2*sem)[first_index:],alpha=0.2,color=color_list[j]) #2 SEM'S

axs[0].set_ylabel("$C_{ss}(t)$",fontsize=14)
for ax in axs.flatten():
    ax.set_xlabel("$t$")
    ax.set_xscale('log')
    ax.set_yscale('log')

axs[0].legend(markerfirst=False,frameon=False)
labels = [r'\textbf{(a): $\Delta_1 = %.1f$}' % Delta_1_list[0],r'\textbf{(b): $\Delta_1 = %.1f$}' % Delta_1_list[1],r'\textbf{(c)}',r'\textbf{(d)}',r'\textbf{(e)}',r'\textbf{(f)}']
for i,ax in enumerate(axs):
    trans = mtransforms.ScaledTranslation(10/72, -30/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, labels[i], transform=ax.transAxes + trans,
            fontsize='large', verticalalignment='top',fontweight='bold') #removed fontfamily = 'serif' #removed bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0)
filename = os.path.join(fig_dir,'random_potential_spin_spin_same_site_no_W=1_ran_out_of_time.png')

fig.savefig(filename,dpi=120)
plt.close(fig)