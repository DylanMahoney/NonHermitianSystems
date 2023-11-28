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

folder_name = 'HNSpinSpinTransport'
fig_dir = get_fig_directory(current_directory,folder_name)
data_dir = get_data_directory(current_directory,folder_name)

figure_width = 12

#FIRST LET'S PLOT THE SAME-SITE CORRELATION AS A FUNCTION OF TIME
fig,axs = plt.subplots(1,2,sharey=True)

t_max = 30 #increased from 20
t_step = 0.2 #increased from 0.1
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)
L = 24 #I also computed L=18 so that could be shown as a shadow
g_list = [0,0.1,0.2]
Delta_1 = 1.5
Delta_2_list = [0,1.5]
num_runs = 2

first_time = 0.8 #We may not want to show early times when nothing interesting is happening
first_index = int(first_time/t_step)

color_list = get_list_of_colors_I_like(len(g_list))
finite_size_eq = 1/(4*L)
hydrodynamics_start_time_list = [5,5] #DETERMINED VIA EYEBALLS, AND IN RESPONSE TO THE SLOPES LOOKING FUNNY
hydro_start_indices = [int(t/t_step) for t in hydrodynamics_start_time_list]


for i,Delta_2 in enumerate(Delta_2_list):
    #Do stuff to the ith Axis object
    for j,g in enumerate(g_list):
        data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data.npy' % (L,g,Delta_1,Delta_2,num_runs))
        data = np.load(data_filename)
        same_site_data = data[:,L//2,:]
        first_run = same_site_data[0,:]
        second_run = same_site_data[1,:]
        #Let's cut off everything after one of the runs hits 1/(4*L) for visual cleanliness and typicality reliability
        min_of_two_runs = np.minimum(first_run,second_run)
        last_value = t.size
        if np.min(min_of_two_runs) <= finite_size_eq:
            last_value = np.argmax(min_of_two_runs <= finite_size_eq) #First time it hits or drops below eq
        
        #Let's fit a power law to the average of the two runs, starting at hydro_start_index
        #TODO: put this into a function so that I don't copy-paste a bunch of code for the other spin-spin transport figure
        #TODO: verify that this does what it's supposed to while under less time pressure
        hydro_start_index = hydro_start_indices[i]
        avg_run = np.mean(same_site_data,axis=0)
        starting_value = avg_run[hydro_start_index]
        t0 = t[hydro_start_index]
        def power_law_decay(t,alpha): # = const (t - t0)^{- \alpha}
            return starting_value*t0**alpha*t**(-alpha)
        popt,pcov = scipy.optimize.curve_fit(power_law_decay,t[hydro_start_index:last_value],avg_run[hydro_start_index:last_value],p0=0.66)
        optimal_alpha = popt[0]
        axs[i].plot(t[hydro_start_index:last_value],power_law_decay(t[hydro_start_index:last_value],optimal_alpha),linestyle='--',label='slope %.4f' % optimal_alpha,color=color_list[j])
        axs[i].axvline(x=t0,color='k',linestyle=':',label=r'$t = %i$' % t0)
        axs[i].plot(t[first_index:last_value+1],first_run[first_index:last_value+1],label="g = %.1f" % g,color=color_list[j])
        axs[i].plot(t[first_index:last_value+1],second_run[first_index:last_value+1],color=color_list[j])
    axs[i].axhline(y=finite_size_eq,color='k',label=r'$\frac{1}{4L}$')

axs[0].set_ylabel("$C_{ss}(t)$",fontsize=14)
for ax in axs.flatten():
    ax.set_xlabel("$t$")
    ax.set_xscale('log')
    ax.set_yscale('log')

axs[0].legend(markerfirst=False,frameon=False)
axs[1].legend(markerfirst=False,frameon=False)
add_letter_labels(fig,axs,124,72,[r'$\Delta_2 = 0$',r'$\Delta_2 = 1.5$'],white_labels=False)
filename = os.path.join(fig_dir,'HN_spin_spin_same_site_L=%i.png'%L)

fig.set_size_inches(figure_width,figure_width/2)
fig.savefig(filename,dpi=120)
plt.close(fig)