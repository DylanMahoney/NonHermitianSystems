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

L = 18
g=0.2
Delta_1 = 1.5
Delta_2_list = [0,1.5]
fig,axs = plt.subplots(3,2)

for i,Delta_2 in enumerate(Delta_2_list):
    z_filename = os.path.join(data_dir,'L=%i,g=%.2f,D1=%.2f,D2=%.2f_zs.npy' % (L,g,Delta_1,Delta_2))
    z = np.load(z_filename)
    x = np.real(z)
    y = np.imag(z)
    r = np.abs(z)
    theta = np.angle(z)

    axs[0,i].hist2d(x,y,bins=40,range = [[-1.1, 1.1], [-1.1, 1.1]],density=True)
    axs[0,i].set_xlim(left=-1.1,right=1.1)
    axs[0,i].set_ylim(bottom=-1.1,top=1.1)
    
    axs[0,i].set_ylabel(r"$\text{Im}(z)$")
    axs[0,i].set_aspect('equal')
    
    axs[1,i].hist(r,bins=40,density=True,color=color_list[2])
    axs[1,i].set_xlim(left=0,right=1)

    axs[1,i].set_ylabel("P(r)")
    
    axs[2,i].hist(theta,bins=40,density=True,color=color_list[2])
    axs[2,i].set_xlim(left=-np.pi,right=np.pi)

    axs[2,i].set_xticks(ticks = [-np.pi,-np.pi/2,0,np.pi/2,np.pi],labels=[r"$-\pi$", r"$-\frac \pi 2$", r"$0$", r"$\frac \pi 2 $", r"$\pi$"])
    axs[2,i].set_ylabel(r"$P(\theta)$")

    axs[0,i].set_xlabel(r"$\text{Re}(z)$")
    axs[1,i].set_xlabel("r")
    axs[2,i].set_xlabel(r"$\theta$")
    
add_letter_labels(fig,axs,150,200,[r'$\Delta_2=0$',r'$\Delta_2=1.5$',r'$\Delta_2=0$',r'$\Delta_2=1.5$',r'$\Delta_2=0$',r'$\Delta_2=1.5$'],white_labels=True)

filename = os.path.join(fig_dir,'HN_eval_histg=%.2fL=%i.png' % (g,L))

fig.set_size_inches(figure_width/1.5, figure_width)

fig.savefig(filename,dpi=120)
plt.close(fig)