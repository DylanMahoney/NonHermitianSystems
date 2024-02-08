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

folder_name = 'RandomPotentialSpinSpinTransport'
fig_dir = get_fig_directory(current_directory,folder_name)
data_dir = get_data_directory(current_directory,folder_name)


#FIRST LET'S PLOT THE SAME-SITE CORRELATION AS A FUNCTION OF TIME
#LET'S NOT INCLUDE DELTA_1 = 1.5
fig,ax = plt.subplots()
plt.style.use('Solarize_Light2')

Delta_1_list = [1]#[1,1.5]
W_list = [0,0.5,1]#[0,0.5,1]
nonHermitian_num_runs = 1000
Hermitian_num_runs = 2 #We should need a lot fewer runs in the Hermitian case

color_list = get_list_of_colors_I_like(len(W_list))

L = 18
t_max = 30
t_step = 0.2
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)

#FOR NOW I'M COMMENTING OUT THE POWER LAW FITS

first_time = 1
first_index = int(first_time/t_step) #Later we might want to not show early times
finite_size_eq = 1/(4*L)
#hydrodynamics_start_time_list = [9,9] #DETERMINED VIA EYEBALLS, AND IN RESPONSE TO THE SLOPES LOOKING FUNNY
#hydro_start_indices = [int(t/t_step) for t in hydrodynamics_start_time_list]

for i,Delta_1 in enumerate(Delta_1_list):
    #Do stuff to the ith Axis object
    for j,W in enumerate(W_list):
        if W < 0.0001:
            num_runs = Hermitian_num_runs
        else:
            num_runs = nonHermitian_num_runs
        data_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2f,%iruns_all_data.npy' % (L,Delta_1,W,num_runs))
        data = np.load(data_filename)
        same_site_data = data[:,L//2,:]
        #Let's cut off everything after the run average hits 1/(4*L) for visual cleanliness and typicality reliability
        run_average = np.mean(same_site_data,axis=0)
        last_value = t.size
        if np.min(run_average) <= finite_size_eq:
            last_value = np.argmax(run_average <= finite_size_eq) #First time it hits or drops below eq
        if j == 0:
            ax.axhline(y=finite_size_eq,color='k',ls='--',label=r'$\frac{1}{4L}$')
        if W < 0.0001:
            ax.plot(t[first_index:last_value],same_site_data[0,first_index:last_value],label=r"$W=%.1f$" % W, color = color_list[j])
            for k in range(1,Hermitian_num_runs):
                ax.plot(t[first_index:last_value],same_site_data[k,first_index:last_value],color=color_list[j])
        else:
            average_data = np.mean(same_site_data,axis=0)
            std = np.std(same_site_data,axis=0)
            sem = std/np.sqrt(num_runs)
            ax.plot(t[first_index:last_value],average_data[first_index:last_value],label=r"$W=%.1f$" % W,color=color_list[j])
            ax.fill_between(t[first_index:last_value],(average_data - 2*sem)[first_index:last_value], (average_data + 2*sem)[first_index:last_value],alpha=0.2,color=color_list[j]) #2 SEM'S
        
        #NOW DO POWER LAW FIT
        #I SHOULD PROBABLY PUT THIS IN ITS OWN FUNCTION INSTEAD OF COPY-PASTING
        #hydro_start_index = hydro_start_indices[i]
        #starting_value = run_average[hydro_start_index]
        #t0 = t[hydro_start_index]
        #def power_law_decay(t,alpha): # = const (t - t0)^{- \alpha}
        #    return starting_value*t0**alpha*t**(-alpha)
        #popt,pcov = scipy.optimize.curve_fit(power_law_decay,t[hydro_start_index:last_value],run_average[hydro_start_index:last_value],p0=0.66)
        #optimal_alpha = popt[0]
        #axs[i].plot(t[hydro_start_index:last_value],power_law_decay(t[hydro_start_index:last_value],optimal_alpha),linestyle='--',label='slope %.4f' % optimal_alpha,color=color_list[j])

ax.set_ylabel("$\mathcal{C}(0,t)$")
ax.set_xlabel("$t$")
ax.set_xscale('log')
ax.set_yscale('log')

ax.legend(markerfirst=False,frameon=False)
#axs[1].legend(markerfirst=False,frameon=False)
#add_letter_labels(fig,axs,72,30,[r'$\Delta_1 = 1.0$',r'$\Delta_1 = 1.5$'],white_labels=False)

#labels = [r'\textbf{(a): $\Delta_1 = %.1f$}' % Delta_1_list[0],r'\textbf{(b): $\Delta_1 = %.1f$}' % Delta_1_list[1],r'\textbf{(c)}',r'\textbf{(d)}',r'\textbf{(e)}',r'\textbf{(f)}']
#for i,ax in enumerate(axs):
#    trans = mtransforms.ScaledTranslation(10/72, -30/72, fig.dpi_scale_trans)
#    ax.text(0.0, 1.0, labels[i], transform=ax.transAxes + trans,
#            fontsize='large', verticalalignment='top',fontweight='bold') #removed fontfamily = 'serif' #removed bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0)
filename = os.path.join(fig_dir,'random_potential_spin_spin_same_site_%iruns.pdf' % nonHermitian_num_runs)

fig.savefig(filename,bbox_inches='tight')
plt.close(fig)