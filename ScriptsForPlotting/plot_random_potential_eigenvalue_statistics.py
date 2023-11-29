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

folder_name = 'RandomPotentialEigenvalueStatistics'
fig_dir = get_fig_directory(current_directory,folder_name)
data_dir = get_data_directory(current_directory,folder_name)

figure_width = 12
color_list = get_list_of_colors_I_like(3)

L = 14
W_list = [0.1,0.5,1] #Let's make several figures and then decide which one we want to use
M_list = [0,1]
num_runs = 100

for W in W_list:
    for M in M_list:
        fig,axs = plt.subplots(3,1)
        z_filename = os.path.join(data_dir,'L=%iW=%.2f,M=%i,%iruns_zs.npy' % (L,W,M,num_runs))
        zs_all_runs = np.load(z_filename)
        z = zs_all_runs.flatten()
        x = np.real(z)
        y = np.imag(z)
        r = np.abs(z)
        theta = np.angle(z)
        
        axs[0].hist2d(x,y,bins=80,range = [[-1.1, 1.1], [-1.1, 1.1]],density=True)
        axs[0].set_xlim(left=-1.1,right=1.1)
        axs[0].set_ylim(bottom=-1.1,top=1.1)
        
        axs[0].set_ylabel(r"$\text{Im}(z)$")
        axs[0].set_aspect('equal')
        
        axs[1].hist(r,bins=80,density=True,color=color_list[2])
        axs[1].set_xlim(left=0,right=1)
        
        axs[1].set_ylabel(r"$P(r)$")
        
        axs[2].hist(theta,bins=80,density=True,color=color_list[2])
        axs[2].set_xlim(left=-np.pi,right=np.pi)
        
        axs[2].set_xticks(ticks = [-np.pi,-np.pi/2,0,np.pi/2,np.pi],labels=[r"$-\pi$", r"$-\frac \pi 2$", r"$0$", r"$\frac \pi 2 $", r"$\pi$"])
        axs[2].set_ylabel(r"$P(\theta)$")
        
        axs[0].set_xlabel(r"$\text{Re}(z)$")
        axs[1].set_xlabel("r")
        axs[2].set_xlabel(r"$\theta$")
        
        add_letter_labels(fig,axs,110,144,['','',''],white_labels=True)
        
        filename = os.path.join(fig_dir,'random_potential_eval_stats_W=%.1fM=%i.png' % (W,M))
        
        fig.set_size_inches(figure_width/3, figure_width)
        
        fig.savefig(filename,dpi=120)
        plt.close(fig)
        #Also plot eigenvalues
        fig,axs = plt.subplots(3,1)
        evals_filename = os.path.join(data_dir,'L=%iW=%.2f,M=%i,%iruns_evals.npy' % (L,W,M,num_runs))
        evals_all_runs = np.load(evals_filename)
        evals = evals_all_runs.flatten()
        x = np.real(evals)
        y = np.imag(evals)
        
        axs[0].hist2d(x,y,bins=80,density=True)
        
        axs[0].set_ylabel(r"$\text{Im}(\lambda)$")
        
        axs[1].hist(x,bins=80,density=True,color=color_list[2])
        
        axs[1].set_ylabel(r"$P(\text{Re}(\lambda)$")
        
        axs[2].hist(y,bins=80,density=True,color=color_list[2])
        
        axs[2].set_ylabel(r"$P(\text{Im}(\lambda))$")
        
        axs[0].set_xlabel(r"$\text{Re}(\lambda)$")
        axs[1].set_xlabel(r"$\text{Re}(\lambda)$")
        axs[2].set_xlabel(r"$\text{Im}(\lambda)$")
        
        add_letter_labels(fig,axs,110,144,['','',''],white_labels=True)
        
        filename = os.path.join(fig_dir,'random_potential_plain_evals_W=%.1fM=%i.png' % (W,M))
        
        fig.set_size_inches(figure_width/3, figure_width)
        
        fig.savefig(filename,dpi=120)
        plt.close(fig)
