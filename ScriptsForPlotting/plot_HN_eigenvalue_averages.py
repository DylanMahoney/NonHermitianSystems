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

folder_name = 'HNEigenvalueStatistics'
fig_dir = get_fig_directory(current_directory,folder_name)
data_dir = get_data_directory(current_directory,folder_name)

figure_width = 12
color_list = get_list_of_colors_I_like(3)

r_prediction_dict = {'Poisson':2/3,'Ginibre':0.7381018}
cos_prediction_dict = {'Poisson':0,'Ginibre':-0.2405161}
linestyle_dict = {'Poisson':'-','Ginibre':'-.'}
#TAKEN FROM https://arxiv.org/abs/1910.12784

L_list = [16,18,20,22] #change to [18,20,22] after finishes running
Delta_1 = 1.5
g_list = [0.1,0.2]
Delta_2_list = [0,1.5]
fig,axs = plt.subplots(2,2,sharex='col',sharey='row')
color_list = get_list_of_colors_I_like(len(g_list))

for i,Delta_2 in enumerate(Delta_2_list):
    r_mean_array = np.zeros((len(g_list),len(L_list)))
    cos_mean_array = np.zeros((len(g_list),len(L_list)))
    for j,g in enumerate(g_list):
        for k,L in enumerate(L_list):
            zs_filename = os.path.join(data_dir,'L=%i,g=%.2f,D1=%.2f,D2=%.2f_zs.npy' % (L,g,Delta_1,Delta_2))
            z = np.load(zs_filename)
            r = np.abs(z)
            x = np.real(z)
            r_mean_array[j,k] = np.mean(np.abs(z))
            cos_mean_array[j,k] = np.mean(x/r)
    
    for j,g in enumerate(g_list):
        if i == 0:
            axs[0,i].scatter(L_list,r_mean_array[j,:],color=color_list[j],label='g=%.2f' % g,alpha=0.8)
        else:
            axs[0,i].scatter(L_list,r_mean_array[j,:],color=color_list[j],alpha=0.8)
        axs[1,i].scatter(L_list,cos_mean_array[j,:],color=color_list[j],alpha=0.8)
    axs[0,i].set_xticks(L_list)
    axs[1,i].set_xticks(L_list)
    for reference in ['Poisson','Ginibre']:
        if i ==0:
            axs[0,i].axhline(y=r_prediction_dict[reference],label=reference,c='k',linestyle=linestyle_dict[reference])
        else:
            axs[0,i].axhline(y=r_prediction_dict[reference],c='k',linestyle=linestyle_dict[reference])
        axs[1,i].axhline(y=cos_prediction_dict[reference],c='k',linestyle=linestyle_dict[reference])
    
    #axs[j,1].set_xlabel("L")

axs[0,0].set_ylabel(r"$\langle r \rangle$")
axs[1,0].set_ylabel(r"$\langle \cos(\theta) \rangle$")

axs[1,0].set_xlabel("L")
axs[1,1].set_xlabel("L")

axs[0,0].legend(markerfirst=False,frameon=False,ncol=2)

add_letter_labels(fig,axs,36,120,[r'$\Delta_2=0$',r'$\Delta_2=1.5$',r'$\Delta_2=0$',r'$\Delta_2=1.5$'],white_labels=False)


filename = os.path.join(fig_dir,'average_r_cos_theta.png')

fig.set_size_inches(figure_width, figure_width/2)

fig.savefig(filename,dpi=120)
plt.close(fig)
