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

#GO THROUGH AND MAKE BETTER WHILE NOT DISTRACTED

plt.rcParams['text.usetex'] = True
rng = np.random.default_rng(2128971964) #WHO YOU GONNA CALL?
plt.rc('text', usetex=True)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})
plt.rc('text.latex', preamble=r'\usepackage{amsmath,braket}')
plt.rcParams['figure.constrained_layout.use'] = True

folder_name = 'HNCurrentCurrentTransport'
fig_dir = get_fig_directory(current_directory,folder_name)
data_dir = get_data_directory(current_directory,folder_name)
figure_width = 12
color_list = get_list_of_colors_I_like(3)

t_max = 50
t_step = 0.5
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)
L_list = [8,10,12]#bring up to 14 or so
g = 0.2 #g = 0 and g=0.1 also available
Delta_1 = 1.5
Delta_2_list = [0,1.5]#[0,1.5]
fig,axs = plt.subplots(1,2,sharey=True)
fig.set_size_inches(figure_width, figure_width/2)
for i,Delta_2 in enumerate(Delta_2_list):
    for j,L in enumerate(L_list):
        print(L)
        print(g)
        data_filename = os.path.join(data_dir,'L=%ig=%.2fD1=%.2fD2=%.2fcurrent_current.npy' % (L,g,Delta_1,Delta_2))
        data = np.load(data_filename)
        if i==0:
            label = '$L = %i$' % L
        else:
            label= 'None'
        axs[i].plot(t,data/L,label=label,color=color_list[j])

axs[0].set_title("$\Delta_2 = 0$",fontsize=14)
axs[1].set_title("$\Delta_2 = 1$",fontsize=14)
axs[0].set_ylabel("${\cal J}(t)/L$",fontsize=14)

for ax in axs.flatten():
    ax.set_xlabel("$t$")
    ax.set_xscale('log')
    ax.set_yscale('log')

axs[0].legend(markerfirst=False,frameon=False)
labels = [r'\textbf{(a)}',r'\textbf{(b)}',r'\textbf{(c)}',r'\textbf{(d)}',r'\textbf{(e)}',r'\textbf{(f)}']
for i,ax in enumerate(axs):
    trans = mtransforms.ScaledTranslation(10/72, -10/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, labels[i], transform=ax.transAxes + trans,
            fontsize='large', verticalalignment='top',fontweight='bold') #removed fontfamily = 'serif' #removed bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0)
filename = os.path.join(fig_dir,'HN_current_current_transport.png')

fig.savefig(filename,dpi=120)
plt.close(fig)