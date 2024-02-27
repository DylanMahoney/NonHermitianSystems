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

folder_name = 'HNSpinSpinTransport'
fig_dir = get_fig_directory(current_directory,folder_name)
data_dir = get_data_directory(current_directory,folder_name)

#FIRST LET'S PLOT THE SAME-SITE CORRELATION AS A FUNCTION OF TIME
landscape = False
if landscape:
    fig,axs = plt.subplots(2,1,sharex=True)
else:
    fig,axs = plt.subplots(1,2,sharey=True)
plt.style.use('Solarize_Light2')

t_max = 30 #increased from 20
t_step = 0.2 #increased from 0.1
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)
L = 24 #I also computed L=18 so that could be shown as a shadow
g_list = [0,0.1,0.15,0.2]
Delta_1 = 1.5
Delta_2_list = [0,1.5]
num_runs = 6
#CHANGE TO USE ALL 6 RUNS

first_time = 1.0 #We may not want to show early times when nothing interesting is happening
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
        avg_run = np.mean(same_site_data,axis=0)
        std_run = np.std(same_site_data,axis=0)
        sem = std_run/np.sqrt(num_runs)
        #Let's cut off everything after the run average hits 1/(4*L) for visual cleanliness and typicality reliability and finite size effects
        last_value = t.size
        if np.min(avg_run) <= finite_size_eq:
            last_value = np.argmax(avg_run <= finite_size_eq) #First time it hits or drops below eq
        
        #Let's fit a power law to the average of the two runs, starting at hydro_start_index
        #TODO: put this into a function so that I don't copy-paste a bunch of code for the other spin-spin transport figure
        #TODO: verify that this does what it's supposed to while under less time pressure
        if i == 1:
            hydro_start_index = hydro_start_indices[i]
            
            starting_value = avg_run[hydro_start_index]
            t0 = t[hydro_start_index]
            sigma = np.cov(same_site_data[:,hydro_start_index:last_value],rowvar=False,bias = True) #bias = True to be consistent in 1/sqrt(N) convention
            #Due to numerical error, sigma may not be positive definite
            print(np.min(np.linalg.eigvals(sigma)))
            if np.min(np.linalg.eigvals(sigma)) < 0:
                sigma = sigma - np.real(np.min(np.linalg.eigvals(sigma))*np.eye(sigma.shape[0])) + 1e-20*np.eye(sigma.shape[0])
                print("Shifted eigenvalues a tiny bit")
            print(np.min(np.linalg.eigvals(sigma)))
            #Due to numerical error, sigma may not be Hermitian
            sigma = (sigma + np.conj(sigma.T))/2
            #Ignore cross-variances
            sigma = std_run[hydro_start_index:last_value]
            print("sigma = standard deviations")
            def power_law_decay(t,alpha): # = const (t - t0)^{- \alpha}
                return starting_value*t0**alpha*t**(-alpha)
            popt,pcov = scipy.optimize.curve_fit(power_law_decay,t[hydro_start_index:last_value],avg_run[hydro_start_index:last_value],p0=1,sigma=sigma)
            optimal_alpha = popt[0]
            print("D2=%.2f, g=%.2f, optimal_alpha=%.2f" % (Delta_2,g,optimal_alpha))
            axs[i].plot(t[hydro_start_index:last_value],power_law_decay(t[hydro_start_index:last_value],optimal_alpha),linestyle='--',color=color_list[j])
            #ADD TEXT CONTAINING OPTIMAL ALPHA
            (x0,y0,width,height) = axs[i].get_position().bounds
            if j == 0:
                axs[i].text(x=15,y=0.03,s=r"{\Large $\propto t^{-%.2f}$}" % optimal_alpha,color=color_list[j],ha='center',va='center',rotation_mode='anchor',rotation=np.arctan(-optimal_alpha*height/width)*180/np.pi)
            #if j == 1 and i == 0:
            #    axs[i].text(x=24,y=0.012,s=r"$\propto t^{-%.2f}$" % optimal_alpha,color=color_list[j])
            if j == 1:# and i == 1:
                axs[i].text(x=20,y=0.018,s=r"{\Large $\propto t^{-%.2f}$}" % optimal_alpha,color=color_list[j],ha='center',va='center',rotation_mode='anchor',rotation=np.arctan(-optimal_alpha*height/width)*180/np.pi)
            if j == 2:
                axs[i].text(x=13,y=1.69e-2,s=r"{\Large $\propto t^{-%.2f}$}" % optimal_alpha,color=color_list[j],ha='center',va='center',rotation_mode='anchor',rotation=np.arctan(-optimal_alpha*height/width)*180/np.pi)
            if j == 3:
                axs[i].text(x=6,y=0.016,s=r"{\Large $\propto t^{-%.2f}$}" % optimal_alpha,color=color_list[j],ha='center',va='center',rotation_mode='anchor',rotation=np.arctan(-optimal_alpha*height/width)*180/np.pi)
        axs[i].plot(t[first_index:last_value+1],avg_run[first_index:last_value+1],label="g = %.2f" % g,color=color_list[j])
        axs[i].fill_between(t[first_index:last_value+1],(avg_run - 2*sem)[first_index:last_value+1], (avg_run + 2*sem)[first_index:last_value+1],alpha=0.75,color=color_list[j])
    print(finite_size_eq)
    axs[i].axhline(y=finite_size_eq,color='k',label=r'$\frac{1}{4L}$',ls='--')

axs[0].set_ylabel(r"\Large $C(0,t)$")
if landscape:
    axs[1].set_ylabel(r"\Large $C(0,t)$")
axs[1].set_xlabel("time $t$")
if not landscape:
    axs[0].set_xlabel("time $t$")
    axs[0].yaxis.set_label_coords(x=-0.05,y=0.5)
for ax in axs.flatten():
    ax.set_xscale('log')
    ax.set_yscale('log')
#for ax in axs.flatten():
#    x_left, x_right = ax.get_xlim()
#    y_low, y_high = ax.get_ylim()
#    ratio = 0.0085
#    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

axs[0].legend(markerfirst=False,frameon=False)
#axs[1].legend(markerfirst=False,frameon=False)
axs[0].text(x=0.9,y=8e-3,s=r"{\Large\textbf{(a)} $\Delta_2 = 0$}")
axs[1].text(x=0.9,y=8e-3,s=r"{\Large\textbf{(b)} $\Delta_2 = 1.5$}")
#add_letter_labels(fig,axs,124,72,[r'$\Delta_2 = 0$',r'$\Delta_2 = 1.5$'],white_labels=False)
filename = os.path.join(fig_dir,'fig5.pdf')
fig.savefig(filename,bbox_inches='tight')
plt.close(fig)