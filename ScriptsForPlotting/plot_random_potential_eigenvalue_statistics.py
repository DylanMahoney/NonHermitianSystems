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
import matplotlib.gridspec
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
num_runs = 100
W = 1
M = 1

#LET'S CHANGE THE BELOW NUMBERS TO BE IN FIGURE COORDINATES FOR CONSISTENCY ACROSS THE 3 FIGURES
x_to_put_text = 0.03
y_to_put_text = 2*x_to_put_text
histogram_color = get_list_of_colors_I_like(2)[1]

#LEARN HOW TO USE GRIDSPEC TO CONTROL WHITE SPACE AND ASPECT RATIOS BETTER
total_width = 8 #measured in eigth-inches
total_height = 3 #measured in eigth-inches
text_height = 0.1 #measured in eigth-inches
left_width = 4.2
right_width = total_width - left_width

gs = matplotlib.gridspec.GridSpec(4,3, width_ratios=[text_height,(left_width-text_height),right_width], 
                                       height_ratios=[total_height/4 - text_height,text_height,total_height/4 - text_height,text_height])

fig = plt.figure()
taller = True
if taller:
    fig.set_size_inches(6.4,3.2)
ax_dict = {}
ax_dict['A'] = fig.add_subplot(gs[0:3,1])
ax_dict['B'] = fig.add_subplot(gs[0,2])
ax_dict['C'] = fig.add_subplot(gs[2,2])
#plt.style.use('Solarize_Light2')
#the above are proportions of the overall data range plotted
y_position_of_x_labels = -0.1
x_position_of_y_labels = -0.05
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

#ax_dict['A'].set_yticks(ticks=[-1,1],labels=[r"$-1$",r"$1$"])
#ax_dict['A'].set_ylabel(r"$\Im(z)$")
#ax_dict['A'].yaxis.set_label_coords(x_position_of_y_labels, 0.5)

ax_dict['A'].set_ylabel(r"$\Im(z)$")
ax_dict['A'].set_xlabel(r"$\Re(z)$")
ax_dict['A'].set_yticks(ticks=[-1,1],labels=[r"$-1$",r"$1$"])
ax_dict['A'].set_xticks(ticks=[-1,1],labels=[r"$-1$",r"$1$"])
#ax_dict['A'].set_xticks(ticks=[],labels=[])
ax_dict['A'].yaxis.set_label_coords(x_position_of_y_labels, 0.5)
ax_dict['A'].xaxis.set_label_coords(0.5, y_position_of_x_labels/4) #A and D are a different size from the others
#ax_dict['A'].xaxis.tick_top()
#ax_dict['A'].set_aspect(1)
#ax_dict['A'].set_xlabel(r"$\Re(z)$")

#ax_dict['A'].tick_params(which='both',direction='in')
#ax_dict['A'].tick_params(left=False,bottom=False)
ax_dict['A'].set_aspect('equal')
(x0,y0,width,height) = ax_dict['A'].get_position().bounds

ax_dict['A'].text(x=x0+width-x_to_put_text,y=y0+y_to_put_text,s=r"{\Large\textbf{(a)}}",color='w',ha='center',va='center',transform=fig.transFigure)

ax_dict['B'].hist(r,bins=40,density=True,color=histogram_color)
ax_dict['B'].set_xlim(left=0,right=1)
ax_dict['B'].set_ylim(bottom=0,top=2.5)
#ax_dict['B'].tick_params(left=False,bottom=False)
#ax_dict['B'].set_xlabel(r"$\rho$")
ax_dict['B'].yaxis.set_label_position("right")
ax_dict['B'].yaxis.tick_right()
ax_dict['B'].set_xticks(ticks=[0,1],labels=[r"0",r"1"])
ax_dict['B'].set_ylabel(r"$P(\varrho)$")
ax_dict['B'].set_yticks(ticks=[0,2],labels=[r"$0$",r"$2$"])
#ax_dict['B'].set_aspect(0.5)
x_left, x_right = ax_dict['B'].get_xlim()
y_low, y_high = ax_dict['B'].get_ylim()
ratio = (total_height/4 - text_height)/right_width #0.5
#ax_dict['B'].set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

(x0,y0,width,height) = ax_dict['B'].get_position().bounds

ax_dict['B'].text(x=x0+width-x_to_put_text,y=y0+y_to_put_text,s=r"{\Large\textbf{(b)}}",color='w',ha='center',va='center',transform=fig.transFigure)
ax_dict['B'].set_xlabel(r"$\varrho$")
ax_dict['B'].xaxis.set_label_coords(0.5, y_position_of_x_labels)
if taller:
    ax_dict['B'].yaxis.set_label_coords(1-x_position_of_y_labels/2,0.4)

ax_dict['C'].hist(theta,bins=40,density=True,color=histogram_color)
ax_dict['C'].set_xlim(left=-np.pi,right=np.pi)
ax_dict['C'].set_xticks(ticks = [-np.pi,np.pi],labels=[r"$-\pi$", r"$\pi$"])
ax_dict['C'].set_ylim(bottom=0,top=0.22)
ax_dict['C'].set_yticks(ticks=[0,1/(2*np.pi)],labels=[r"$0$",r"$\frac{1}{2\pi}$"])
#ax_dict['C'].tick_params(left=False,bottom=False)

ax_dict['C'].set_ylabel(r"$P(\theta)$")
ax_dict['C'].yaxis.set_label_position("right")
ax_dict['C'].yaxis.tick_right()
if taller:
    ax_dict['C'].yaxis.set_label_coords(1-x_position_of_y_labels/2,0.333)
#ax_dict['C'].set_xticks(ticks=[],labels=[])

ax_dict['C'].set_xlabel(r"$\theta$")
x_left, x_right = ax_dict['C'].get_xlim()
y_low, y_high = ax_dict['C'].get_ylim()
ratio = 0.5
#ax_dict['C'].set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
#ax_dict['C'].set_aspect(0.5)
#ax_dict['C'].set_xlabel(r"$\theta$")
(x0,y0,width,height) = ax_dict['C'].get_position().bounds

ax_dict['C'].text(x=x0+width-x_to_put_text,y=y0+y_to_put_text,s=r"{\Large\textbf{(c)}}",color='w',ha='center',va='center',transform=fig.transFigure)
ax_dict['C'].xaxis.set_label_coords(0.5, y_position_of_x_labels)

filename = os.path.join(fig_dir,'fig8.pdf') # % (W,M)
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
