#https://stackoverflow.com/questions/13369888/matplotlib-y-axis-label-on-right-side
#IS RIGHT-ALIGNING THE LETTERS REALLY THE RIGHT CHOICE?
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

folder_name = 'HNEigenvalueStatistics'
fig_dir = get_fig_directory(current_directory,folder_name)
data_dir = get_data_directory(current_directory,folder_name)

histogram_color = get_list_of_colors_I_like(2)[1]
r = np.random.uniform(0,1,500)
theta = np.random.uniform(0,2*np,pi,500)
x = r*np.cos(theta)
y = r*np.sin(theta)

mosaic = """
    AABB
    AACC
    DDEE
    DDFF
    
    """
fig = plt.figure()
ax_dict = fig.subplot_mosaic(mosaic)
plt.style.use('Solarize_Light2')

x_to_put_text = 0.95
y_to_put_text = 0.1
#the above are proportions of the overall data range plotted
y_position_of_x_labels = -0.1
x_position_of_y_labels = -0.05

noNNN_z_filename = os.path.join(data_dir,'L=%i,g=%.2f,D1=%.2f,D2=%.2f_zs.npy' % (L,g,Delta_1,0))
z = np.load(noNNN_z_filename)
x = np.real(z)
y = np.imag(z)
r = np.abs(z)
theta = np.angle(z)

ax_dict['A'].hist2d(x,y,bins=40,range = [[-1.1, 1.1], [-1.1, 1.1]],density=True)
ax_dict['A'].set_xlim(left=-1.1,right=1.1)
ax_dict['A'].set_ylim(bottom=-1.1,top=1.1)

#ax_dict['A'].set_yticks(ticks=[-1,1],labels=[r"$-1$",r"$1$"])
#ax_dict['A'].set_ylabel(r"$\Im(z)$")
#ax_dict['A'].yaxis.set_label_coords(x_position_of_y_labels, 0.5)

ax_dict['A'].set_ylabel(r"$\Im(z)$")
#ax_dict['A'].set_xlabel(r"$\Re(z)$")
ax_dict['A'].set_yticks(ticks=[-1,1],labels=[r"$-1$",r"$1$"])
#ax_dict['A'].set_xticks(ticks=[-1,1],labels=[r"$-1$",r"$1$"])
ax_dict['A'].set_xticks(ticks=[],labels=[])
ax_dict['A'].yaxis.set_label_coords(x_position_of_y_labels, 0.5)
ax_dict['A'].xaxis.set_label_coords(0.5, y_position_of_x_labels/2) #A and D are a different size from the others
#ax_dict['A'].xaxis.tick_top()
#ax_dict['A'].set_aspect(1)
#ax_dict['A'].set_xlabel(r"$\Re(z)$")

#ax_dict['A'].tick_params(which='both',direction='in')
#ax_dict['A'].tick_params(left=False,bottom=False)
ax_dict['A'].set_aspect('equal')
ax_dict['A'].text(x=-1.1+2.2*x_to_put_text,y=-1.1+2.2*y_to_put_text,s=r"{\Large\textbf{(a)} $\Delta_2 = 0$}",color='w',ha='right')

ax_dict['B'].hist(r,bins=40,density=True,color=histogram_color)
ax_dict['B'].set_xlim(left=0,right=1)
ax_dict['B'].set_ylim(bottom=0,top=2.5)
#ax_dict['B'].tick_params(left=False,bottom=False)
#ax_dict['B'].set_xlabel(r"$\rho$")
ax_dict['B'].yaxis.set_label_position("right")
ax_dict['B'].yaxis.tick_right()
ax_dict['B'].set_xticks(ticks=[0,1],labels=[r"0",r"1"])
ax_dict['B'].set_ylabel(r"$P(\varrho)$")
#ax_dict['B'].set_yticks(ticks=[0,2],labels=[r"$0.0$",r"$2.0$"])
#ax_dict['B'].set_aspect(0.5)
x_left, x_right = ax_dict['B'].get_xlim()
y_low, y_high = ax_dict['B'].get_ylim()
ratio = 0.5
#ax_dict['B'].set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
ax_dict['B'].text(x=0+1*x_to_put_text,y=0+2.5*y_to_put_text,s=r"{\Large\textbf{(b)} $\Delta_2 = 0$}",color='w',ha='right')
ax_dict['B'].set_xlabel(r"$\varrho$")
ax_dict['B'].xaxis.set_label_coords(0.5, y_position_of_x_labels)

ax_dict['C'].hist(theta,bins=40,density=True,color=histogram_color)
ax_dict['C'].set_xlim(left=-np.pi,right=np.pi)
ax_dict['C'].set_xticks(ticks = [-np.pi,np.pi],labels=[r"$-\pi$", r"$\pi$"])
ax_dict['C'].set_ylim(bottom=0,top=0.22)
ax_dict['C'].set_yticks(ticks=[0,1/(2*np.pi)],labels=[r"$0$",r"$\frac{1}{2\pi}$"])
#ax_dict['C'].tick_params(left=False,bottom=False)

ax_dict['C'].yaxis.set_label_position("right")
ax_dict['C'].yaxis.tick_right()
#ax_dict['C'].set_xticks(ticks=[],labels=[])

ax_dict['C'].set_ylabel(r"$P(\theta)$")
ax_dict['C'].set_xlabel(r"$\theta$")
x_left, x_right = ax_dict['C'].get_xlim()
y_low, y_high = ax_dict['C'].get_ylim()
ratio = 0.5
#ax_dict['C'].set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
#ax_dict['C'].set_aspect(0.5)
#ax_dict['C'].set_xlabel(r"$\theta$")
ax_dict['C'].text(x=-np.pi+2*np.pi*x_to_put_text,y=0+0.22*y_to_put_text,s=r"{\Large\textbf{(c)} $\Delta_2 = 0$}",color='w',ha='right')
ax_dict['C'].xaxis.set_label_coords(0.5, y_position_of_x_labels)

NNN_z_filename = os.path.join(data_dir,'L=%i,g=%.2f,D1=%.2f,D2=%.2f_zs.npy' % (L,g,Delta_1,1.5))
z = np.load(NNN_z_filename)
x = np.real(z)
y = np.imag(z)
r = np.abs(z)
theta = np.angle(z)

ax_dict['D'].hist2d(x,y,bins=40,range = [[-1.1, 1.1], [-1.1, 1.1]],density=True)
ax_dict['D'].set_xlim(left=-1.1,right=1.1)
ax_dict['D'].set_ylim(bottom=-1.1,top=1.1)

ax_dict['D'].set_ylabel(r"$\Im(z)$")
#ax_dict['D'].set_xlabel(r"$\Re(z)$")
ax_dict['D'].set_yticks(ticks=[-1,1],labels=[r"$-1$",r"$1$"])
#ax_dict['D'].set_xticks(ticks=[-1,1],labels=[r"$-1$",r"$1$"])
ax_dict['D'].set_xticks(ticks=[],labels=[])
ax_dict['D'].yaxis.set_label_coords(x_position_of_y_labels, 0.5)
ax_dict['D'].xaxis.set_label_coords(0.5, y_position_of_x_labels/2) #A and D are a different size from the others
#ax_dict['D'].set_aspect(1)

ax_dict['D'].set_aspect('equal')
#ax_dict['D'].tick_params(left=False,bottom=False)
#ax_dict['D'].text(x=-1.1+2.2*x_to_put_text,y=-1.1+2.2*y_to_put_text,s=r"{\Large\textbf{(d)} $\Delta_2 = 1.5$}",color='w',ha='right')
ax_dict['D'].text(x=-1.1+2.2*x_to_put_text,y=-1.1+2.2*y_to_put_text,s=r"{\Large\textbf{(a)} $\Delta_2 = 0$}",color='w',ha='right')

ax_dict['E'].hist(r,bins=40,density=True,color=histogram_color)
ax_dict['E'].set_xlim(left=0,right=1)
ax_dict['E'].set_ylim(bottom=0,top=2.5)
#ax_dict['E'].tick_params(left=False,bottom=False)

ax_dict['E'].yaxis.set_label_position("right")
ax_dict['E'].yaxis.tick_right()
ax_dict['E'].set_xticks(ticks=[0,1],labels=[r"0",r"1"])
print("preferred P(rho) limits")
print(ax_dict['E'].get_ylim())

ax_dict['E'].set_ylabel(r"$P(\varrho)$")
x_left, x_right = ax_dict['E'].get_xlim()
y_low, y_high = ax_dict['E'].get_ylim()
ratio = 0.5
#ax_dict['E'].set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
#ax_dict['E'].set_aspect(0.5)
#ax_dict['E'].set_xlabel(r"$\rho$")
ax_dict['E'].text(x=0+1*x_to_put_text,y=0+2.5*y_to_put_text,s=r"{\Large\textbf{(e)} $\Delta_2 = 1.5$}",color='w',ha='right')
ax_dict['E'].set_xlabel(r"$\varrho$")
ax_dict['E'].xaxis.set_label_coords(0.5, y_position_of_x_labels)

ax_dict['F'].hist(theta,bins=40,density=True,color=histogram_color)
ax_dict['F'].set_xlim(left=-np.pi,right=np.pi)
#ax_dict['F'].set_xticks(ticks = [-np.pi,0,np.pi],labels=[r"$-\pi$ {\Large or} $0$", r"$0$ {\Large or} $\frac 1 2 $", r"$\pi$ {\Large or} $1$"])
ax_dict['F'].set_xticks(ticks = [-np.pi,np.pi],labels=[r"$-\pi$", r"$\pi$"])
print("preferred P(theta) limits")
print(ax_dict['F'].get_ylim())
ax_dict['F'].set_ylim(bottom=0,top=0.22)
#ax_dict['F'].tick_params(left=False,bottom=False)

ax_dict['F'].yaxis.set_label_position("right")
ax_dict['F'].yaxis.tick_right()

ax_dict['F'].set_ylabel(r"$P(\theta)$")
ax_dict['F'].set_yticks(ticks=[0,1/(2*np.pi)],labels=[r"$0$",r"$\frac{1}{2\pi}$"])
#ax_dict['F'].set_xlabel(r"$\theta$ {\Large or} $\varrho$")
ax_dict['F'].set_xlabel(r"$\theta$")
ax_dict['F'].xaxis.set_label_coords(0.5, y_position_of_x_labels)
x_left, x_right = ax_dict['F'].get_xlim()
y_low, y_high = ax_dict['F'].get_ylim()
ratio = 0.5
#ax_dict['F'].set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
#ax_dict['F'].set_aspect(0.5)
ax_dict['F'].text(x=-np.pi+2*np.pi*x_to_put_text,y=0+0.22*y_to_put_text,s=r"{\Large\textbf{(f)} $\Delta_2 = 1.5$}",color='w',ha='right')

filename = os.path.join(fig_dir,'HN_eval_hists.pdf')
fig.savefig(filename,bbox_inches='tight')
plt.close(fig)

exit()
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

    axs[1,i].set_ylabel(r"$P(\rho)$")
    
    axs[2,i].hist(theta,bins=40,density=True,color=color_list[2])
    axs[2,i].set_xlim(left=-np.pi,right=np.pi)

    axs[2,i].set_xticks(ticks = [-np.pi,-np.pi/2,0,np.pi/2,np.pi],labels=[r"$-\pi$", r"$-\frac \pi 2$", r"$0$", r"$\frac \pi 2 $", r"$\pi$"])
    axs[2,i].set_ylabel(r"$P(\theta)$")

    axs[0,i].set_xlabel(r"$\text{Re}(z)$")
    axs[1,i].set_xlabel(r"$\rho$")
    axs[2,i].set_xlabel(r"$\theta$")
    
add_letter_labels(fig,axs,155,207.5,[r'$\Delta_2=0$',r'$\Delta_2=1.5$',r'$\Delta_2=0$',r'$\Delta_2=1.5$',r'$\Delta_2=0$',r'$\Delta_2=1.5$'],white_labels=True)

filename = os.path.join(fig_dir,'HN_eval_histg=%.2fL=%i.png' % (g,L))

fig.set_size_inches(figure_width/1.5, figure_width)

fig.savefig(filename,dpi=120)
plt.close(fig)