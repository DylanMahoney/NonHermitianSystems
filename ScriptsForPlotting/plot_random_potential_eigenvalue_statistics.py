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

folder_name = 'RandomPotentialEigenvalueStatistics'
fig_dir = get_fig_directory(current_directory,folder_name)
data_dir = get_data_directory(current_directory,folder_name)

L = 14
W_list = [0.1,0.5,1] #Let's make several figures and then decide which one we want to use
M_list = [0,1]
num_runs = 100

x_to_put_text = 0.95
y_to_put_text = 0.1
histogram_color = get_list_of_colors_I_like(2)[1]

for W in W_list:
    for M in M_list:
        #LEARN HOW TO USE GRIDSPEC TO CONTROL WHITE SPACE AND ASPECT RATIOS BETTER
        mosaic = """
            AABB
            AACC
            """
        fig = plt.figure()
        ax_dict = fig.subplot_mosaic(mosaic)
        #plt.style.use('Solarize_Light2')
        z_filename = os.path.join(data_dir,'L=%iW=%.2f,M=%i,%iruns_zs.npy' % (L,W,M,num_runs))
        zs_all_runs = np.load(z_filename)
        z = zs_all_runs.flatten()
        x = np.real(z)
        y = np.imag(z)
        r = np.abs(z)
        theta = np.angle(z)
        
        #COPIED FROM THE HN HISTOGRAMS, KIND OF MESSY
        ax_dict['A'].hist2d(x,y,bins=40,range = [[-1.1, 1.1], [-1.1, 1.1]],density=True)
        ax_dict['A'].set_xlim(left=-1.1,right=1.1)
        ax_dict['A'].set_ylim(bottom=-1.1,top=1.1)
        
        ax_dict['A'].set_xticks(ticks=[],labels=[])
        
        ax_dict['A'].set_ylabel(r"$\Im(z)$")
        ax_dict['A'].set_aspect(1)
        #ax_dict['A'].set_xlabel(r"$\Re(z)$")
        
        #ax_dict['A'].tick_params(which='both',direction='in')
        #ax_dict['A'].tick_params(left=False,bottom=False)
        #ax_dict['A'].set_aspect('equal')
        ax_dict['A'].text(x=-1.1+2.2*x_to_put_text,y=-1.1+2.2*y_to_put_text,s=r"{\Large\textbf{(a)}}",color='w',ha='right')
        
        ax_dict['B'].hist(r,bins=40,density=True,color=histogram_color)
        ax_dict['B'].set_xlim(left=0,right=1)
        ax_dict['B'].set_ylim(bottom=0,top=2.5)
        #ax_dict['B'].tick_params(left=False,bottom=False)
        #ax_dict['B'].set_xlabel(r"$\rho$")
        ax_dict['B'].yaxis.set_label_position("right")
        ax_dict['B'].yaxis.tick_right()
        ax_dict['B'].xaxis.set_label_position("top") #upper?
        ax_dict['B'].xaxis.tick_top()
        ax_dict['B'].set_xticks(ticks = [0,0.5,1],labels=[r"$0$", r"$\frac 1 2 $", r"$1$"])
        ax_dict['B'].set_ylabel(r"$P(\rho)$")
        ax_dict['B'].set_xlabel(r"$\rho$")
        #ax_dict['B'].set_yticks(ticks=[0,2],labels=[r"$0.0$",r"$2.0$"])
        #ax_dict['B'].set_aspect(0.5)
        x_left, x_right = ax_dict['B'].get_xlim()
        y_low, y_high = ax_dict['B'].get_ylim()
        ratio = 0.5
        ax_dict['B'].set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
        ax_dict['B'].text(x=0+1*x_to_put_text,y=0+2.5*y_to_put_text,s=r"{\Large\textbf{(b)}}",color='w',ha='right')
        ax_dict['B'].set_facecolor('white')
        
        ax_dict['C'].hist(theta,bins=40,density=True,color=histogram_color)
        ax_dict['C'].set_xlim(left=-np.pi,right=np.pi)
        ax_dict['C'].set_xticks(ticks = [-np.pi,-np.pi/2,0,np.pi/2,np.pi],labels=[r"$-\pi$", r"$-\frac \pi 2$", r"$0$", r"$\frac \pi 2 $", r"$\pi$"])
        ax_dict['C'].set_ylim(bottom=0,top=0.22)
        ax_dict['C'].set_yticks(ticks=[0,1/(2*np.pi)],labels=[r"$0$",r"$\frac{1}{2\pi}$"])
        #ax_dict['C'].tick_params(left=False,bottom=False)
        
        ax_dict['C'].yaxis.set_label_position("right")
        ax_dict['C'].yaxis.tick_right()
        ax_dict['C'].set_xticks(ticks=[-np.pi,0,np.pi],labels=[r"$-\pi$",r"$0$", r"$\pi$"])
        ax_dict['C'].set_xlabel(r"$\theta$")
        
        ax_dict['C'].set_ylabel(r"$P(\theta)$")
        x_left, x_right = ax_dict['C'].get_xlim()
        y_low, y_high = ax_dict['C'].get_ylim()
        ratio = 0.5
        ax_dict['C'].set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
        #ax_dict['C'].set_aspect(0.5)
        #ax_dict['C'].set_xlabel(r"$\theta$")
        ax_dict['C'].text(x=-np.pi+2*np.pi*x_to_put_text,y=0+0.22*y_to_put_text,s=r"{\Large\textbf{(c)}}",color='w',ha='right')
        
        ax_dict['C'].set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        filename = os.path.join(fig_dir,'random_potential_eval_stats_W=%.1fM=%i.pdf' % (W,M))
        fig.savefig(filename,bbox_inches='tight')
        plt.close(fig)
        
        #Also plot eigenvalues
exit()
#everything below the exit was previously within the loops
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
