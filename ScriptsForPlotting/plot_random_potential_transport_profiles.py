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
W_list = [0,1]#[0,0.5,1,5]
num_seed_strings = 30
nonHermitian_num_runs_per_SS = 1000
Hermitian_num_runs = 2 #We should need a lot fewer runs in the Hermitian case

color_list = get_list_of_colors_I_like(len(W_list))

L=18
t_max = 50
t_step = 0.2
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)

#FOR NOW I'M COMMENTING OUT THE POWER LAW FITS

first_time = 1
first_index = int(first_time/t_step) #Later we might want to not show early times
slicing_time = 10
#hydrodynamics_start_time_list = [9,9] #DETERMINED VIA EYEBALLS, AND IN RESPONSE TO THE SLOPES LOOKING FUNNY
#hydro_start_indices = [int(t/t_step) for t in hydrodynamics_start_time_list]

#questionable_data = np.load(os.path.join(data_dir,'L=18D1=1.00W=0.50SS=3,550runs_all_data.npy'))

#SHOULD WE MENTION IN THE PAPER THAT LOOKING AT DIFFERENT W'S WOULD BE A NICE THING TO DO / SOMETHING THAT CONFUSED US?
marker_list = ['.','x']
finite_size_eq = 1/(4*L)
for i,Delta_1 in enumerate(Delta_1_list):
    #Do stuff to the ith Axis object
    for j,W in enumerate(W_list):
        print("W=%.2f" % W)
        if W < 0.0001:
            num_runs = Hermitian_num_runs
            total_num_runs = num_runs
            data_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2f,%iruns_all_data.npy' % (L,Delta_1,W,num_runs))
            data = np.load(data_filename)
        else:
            total_num_runs = 0
            num_runs_per_SS = nonHermitian_num_runs_per_SS
            for seed_start in range(num_seed_strings):
                print(seed_start)
                for num_runs in range(nonHermitian_num_runs_per_SS,0,-50): #try to grab as many runs as exist for this seed string, so start with higher
                    data_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2fSS=%i,%iruns_all_data.npy' % (L,Delta_1,W,seed_start,num_runs))
                    if os.path.exists(data_filename): #We've found the max number of runs that have been saved
                        
                        data_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2fSS=%i,%iruns_all_data.npy' % (L,Delta_1,W,seed_start,num_runs))
                        print(data_filename)
                        break
                if seed_start == 0:
                    data = np.load(data_filename)
                    #FOR L = 18, STILL HAVE THE ZERO ISSUE ON FEB 16 2024!
                    if L == L:
                        should_all_be_pos = data[:,L//2,0]
                        beginning_of_zeros = np.argmax(should_all_be_pos < 0.001)
                        data = data[0:beginning_of_zeros,:,:]
                    total_num_runs += data.shape[0]
                else:
                    data_to_append = np.load(data_filename)
                    if L == L:
                        should_all_be_pos = data_to_append[:,L//2,0]
                        beginning_of_zeros = np.argmax(should_all_be_pos < 0.001)
                        data_to_append = data_to_append[0:beginning_of_zeros,:,:]
                    total_num_runs += data_to_append.shape[0]
                    
                    print(data_to_append[0,7,50])
                    print("^To check that different seeds are really different^")
                    #print(data_to_append[:,L//2,0])
                    data = np.append(data,data_to_append,axis=0)
        time_slice = int(slicing_time/t_step)
        time_slice_data = data[:,:,time_slice]
        ax.plot(np.arange(L),np.mean(time_slice_data,axis=0),marker_list[j],label=r'$W=%.1f$' % W,color=color_list[j],markersize=12,mew=2)
        total_magnetization = np.sum(data,axis=1)
        print(np.max(np.abs(0.25 - total_magnetization)))
        print("^max deviation from M=0.25^")
        
        print("W=%.2f, total number of runs used is %i" % (W,total_num_runs))
ax.set_ylabel(r"$\mathcal{C}(r,t=%i)$" % slicing_time)
ax.set_xlabel("$L$")
ax.set_xticks(np.arange(0,18,3),np.arange(0,18,3))
ax.set_yscale('log')

ax.legend(markerfirst=False,frameon=False)

#axs[1].legend(markerfirst=False,frameon=False)
#add_letter_labels(fig,axs,72,30,[r'$\Delta_1 = 1.0$',r'$\Delta_1 = 1.5$'],white_labels=False)

#labels = [r'\textbf{(a): $\Delta_1 = %.1f$}' % Delta_1_list[0],r'\textbf{(b): $\Delta_1 = %.1f$}' % Delta_1_list[1],r'\textbf{(c)}',r'\textbf{(d)}',r'\textbf{(e)}',r'\textbf{(f)}']
#for i,ax in enumerate(axs):
#    trans = mtransforms.ScaledTranslation(10/72, -30/72, fig.dpi_scale_trans)
#    ax.text(0.0, 1.0, labels[i], transform=ax.transAxes + trans,
#            fontsize='large', verticalalignment='top',fontweight='bold') #removed fontfamily = 'serif' #removed bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0)
filename = os.path.join(fig_dir,'random_potential_spin_spin_same_site_t=%i_%iruns.pdf' % (slicing_time,total_num_runs))

fig.savefig(filename,bbox_inches='tight')
plt.close(fig)