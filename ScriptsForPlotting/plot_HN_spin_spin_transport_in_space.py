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

crowded = False
if crowded:
    g_list = [0,0.1,0.15,0.2]
    markers_list = ['o','x','^','s']
else:
    g_list = [0,0.1]
    markers_list = ['o','s']

L = 24
t_max = 30 #increased from 20
t_step = 0.2 #increased from 0.1
Delta_1 = 1.5
Delta_2_list = [0,1.5]
num_runs = 6
if crowded:
    fig,axs = plt.subplots(1,2,sharey=True)
else:
    fig,axs = plt.subplots(2,1,sharex=True)
plt.style.use('Solarize_Light2')

if crowded:
    time_slice = 6
else:
    time_slice = 9
#These were determined by looking at the same-site correlation plot and fiddling with it until the profiles looked nice
time_slice_index = int(time_slice/t_step)
color_list = get_list_of_colors_I_like(4)
if not crowded:
    color_list = [color_list[0],color_list[3]]
r = np.arange(L)
end_cut = 2 #maybe the ends have some outlier points that we don't need to include?
# we will remove 2*end_cut + 1 positions from the plots
r_ticks = np.arange(1+end_cut,L-end_cut,step=3,dtype=int)

for i,Delta_2 in enumerate(Delta_2_list):
    for j,g in enumerate(g_list):
        data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2f,%iruns_all_data.npy' % (L,g,Delta_1,Delta_2,num_runs))
        data = np.load(data_filename)
        average_data = np.mean(data[:,:,time_slice_index],axis=0)
        label = "$g = %.2f$" % g
        if crowded:
            axs[i].plot(r[1+end_cut:L-end_cut],average_data[1+end_cut:L-end_cut],marker=markers_list[j],label=label,color=color_list[j],markersize=7.5,mew=1.5,ls='')
        if not crowded:
            #Let's add some error bars
            sem = np.std(data[:,:,time_slice_index],axis=0)/np.sqrt(num_runs)
            axs[i].errorbar(r[1+end_cut:L-end_cut],average_data[1+end_cut:L-end_cut],yerr=2*sem[1+end_cut:L-end_cut],capsize=6,marker=markers_list[j],label=label,color=color_list[j],markersize=7.5,mew=1.5,ls='')
        #NOW LET'S ADD GAUSSIAN FITS
        #LET'S EXCLUDE THE SITES VISUALLY IDENTIFIED AS OUTLIERS/SITES WHEN TYPICALITY DIDN'T WORK WELL
        
        def gaussian(r,a,b):
            return a*np.exp(-b*r**2)
        sigma = np.cov(data[:,1+end_cut:L-end_cut,time_slice_index],rowvar=False,bias = True) #bias = True to be consistent in 1/sqrt(N) convention
        #Due to numerical error, sigma may not be positive definite
        print(np.min(np.linalg.eigvals(sigma)))
        if np.min(np.linalg.eigvals(sigma)) < 0:
            sigma = sigma - np.real(np.min(np.linalg.eigvals(sigma))*np.eye(sigma.shape[0])) + 1e-20*np.eye(sigma.shape[0])
            print("Shifted eigenvalues a tiny bit")
        print(np.min(np.linalg.eigvals(sigma)))
        #Due to numerical error, sigma may not be Hermitian
        sigma = (sigma + np.conj(sigma.T))/2
        print("sigma = standard deviations")
        sigma = np.std(data[:,1+end_cut:L-end_cut,time_slice_index],axis=0)
        popt,pcov = scipy.optimize.curve_fit(gaussian,r[1+end_cut:L-end_cut] - L//2,average_data[1+end_cut:L-end_cut],p0=(average_data[L//2],0.1),sigma=sigma) #the 0.1 estimate for b was determined via eyeballs, pencil, and paper
        aopt = popt[0]
        bopt = popt[1]
        axs[i].plot(r[1+end_cut:L-end_cut],gaussian(r[1+end_cut:L-end_cut] - L//2,aopt,bopt),color=color_list[j],ls='--',alpha=0.8)

axs[0].set_ylabel("$C(r,t=%i)$" % time_slice)
axs[0].set_yscale('log')
if not crowded:
    axs[1].set_ylabel("$C(r,t=%i)$" % time_slice)
axs[1].set_yscale('log')
axs[1].set_xlabel("$r$")
axs[1].set_xticks(ticks =r_ticks,labels=r_ticks - L//2)
if crowded:
    axs[0].set_xlabel("$r$")
    axs[0].set_xticks(ticks =r_ticks,labels=r_ticks - L//2)
y_max = axs[0].get_ylim()[1]
if crowded:
    axs[0].set_ylim(bottom=5e-5,top=y_max)
    axs[1].set_ylim(bottom=5e-5,top=y_max)

#if crowded:
#    axs[0].xaxis.set_label_coords(0.5, -0.02)
#    axs[1].xaxis.set_label_coords(0.5,-0.02)
#else:
    #axs[1].xaxis.set_label_coords(0.5, -0.05)

x_to_put_text = 2.5
if crowded:
    y_to_put_text = 5e-2
else:
    y_to_put_text = 3e-2
axs[0].text(x=x_to_put_text,y=y_to_put_text,s=r"{\Large\textbf{(a)} $\Delta_2 = 0$}",color='k',ha='left',va='top')
axs[1].text(x=x_to_put_text,y=y_to_put_text,s=r"{\Large\textbf{(b)} $\Delta_2 = 1.5$}",color='k',ha='left',va='top')
if crowded:
    axs[0].legend(markerfirst=False,frameon=False)
else:
    axs[0].legend(markerfirst=False,frameon=False,loc = 'upper right')
if not crowded:
    filename = os.path.join(fig_dir,'fig6.pdf')
else:
    filename = os.path.join(fig_dir,'fig6alt.pdf')

fig.savefig(filename,bbox_inches='tight')
plt.close(fig)