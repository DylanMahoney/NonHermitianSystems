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

L = 24
t_max = 30 #increased from 20
t_step = 0.2 #increased from 0.1
g_list = [0,0.1,0.2]
Delta_1 = 1.5
Delta_2_list = [0,1.5]
num_runs = 2
fig,axs = plt.subplots(1,2,sharey=True)

time_snapshot = 6 #This was determined by looking at the same-site correlation plot and fiddling with it until the profiles looked nice
snapshot_index = int(time_snapshot/t_step)
color_list = get_list_of_colors_I_like(len(g_list))

markers_list = ['o','x',10]
r = np.arange(L)
end_cut = 2 #the ends have some outlier points that we don't need to include.
r_ticks = np.arange(end_cut,L-end_cut,step=4,dtype=int)

for i,Delta_2 in enumerate(Delta_2_list):
    for j,g in reversed(list(enumerate(g_list))):
        data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data.npy' % (L,g,Delta_1,Delta_2,num_runs))
        data = np.load(data_filename)
        first_run = data[0,:,snapshot_index]
        second_run = data[1,:,snapshot_index]
        label = "$g = %.2f$" % g
        axs[i].scatter(r[end_cut:L-end_cut],first_run[end_cut:L-end_cut],marker=markers_list[j],label=label,color=color_list[j])
        axs[i].scatter(r[end_cut:L-end_cut],second_run[end_cut:L-end_cut],marker=markers_list[j],color=color_list[j])
        #NOW LET'S ADD GAUSSIAN FITS
        #LET'S EXCLUDE THE SITES VISUALLY IDENTIFIED AS OUTLIERS/SITES WHEN TYPICALITY DIDN'T WORK WELL
        average_data = np.mean(data[:,:,snapshot_index],axis=0)
        def gaussian(r,a,b):
            return a*np.exp(-b*r**2)
        popt,pcov = scipy.optimize.curve_fit(gaussian,r[end_cut:L-end_cut] - L//2,average_data[end_cut:L-end_cut],p0=(average_data[L//2],0.1)) #the 0.1 estimate for b was determined via eyeballs, pencil, and paper
        aopt = popt[0]
        bopt = popt[1]
        axs[i].plot(r[end_cut:L-end_cut],gaussian(r[end_cut:L-end_cut] - L//2,aopt,bopt),color=color_list[j],label='Gaussian fit')

axs[0].set_ylabel("$C(r,t=%i)$" % time_snapshot,fontsize=14)

for ax in axs.flatten():
    ax.set_xlabel("$r$")
    ax.set_xticks(ticks =r_ticks,labels=r_ticks)
    ax.set_yscale('log')

axs[0].legend(markerfirst=False,frameon=False,loc='lower center')
add_letter_labels(fig,axs,60,36,[r'$\Delta_2 = 0$',r'$\Delta_2 = 1.5$'],white_labels=False)
filename = os.path.join(fig_dir,'correlation_profiles_L=%i.png' % L)

fig.set_size_inches(figure_width, figure_width/2)
fig.savefig(filename,dpi=120)
plt.close(fig)