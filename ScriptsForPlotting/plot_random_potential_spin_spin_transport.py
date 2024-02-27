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

folder_name = 'RandomPotentialSpinSpinTransport'
fig_dir = get_fig_directory(current_directory,folder_name)
data_dir = get_data_directory(current_directory,folder_name)


#FIRST LET'S PLOT THE SAME-SITE CORRELATION AS A FUNCTION OF TIME
#LET'S NOT INCLUDE DELTA_1 = 1.5
fig,axs = plt.subplots(2,1)
plt.style.use('Solarize_Light2')

Delta_1_list = [1]#[1,1.5]
W_list = [0,1]#[0,0.5,1,5]
num_seed_strings = 30
nonHermitian_num_runs_per_SS = 1000
Hermitian_num_runs = 2 #We should need a lot fewer runs in the Hermitian case

color_list = get_list_of_colors_I_like(len(W_list))

small_L = 16
large_L = 18
t_max = 50
t_step = 0.2
num_times = int(t_max/t_step)+1
t = np.linspace(0,t_max,num_times)
Delta_1 = 1

#FOR NOW I'M COMMENTING OUT THE POWER LAW FITS
num_xs = 20

first_time = 4
time_slice = 10
time_slice_index = int(time_slice/t_step)
first_index = int(first_time/t_step) #Later we might want to not show early times
hydro_start_time = 9
hydro_start_index = int(hydro_start_time/t_step)

#SHOULD WE MENTION IN THE PAPER THAT LOOKING AT DIFFERENT W'S WOULD BE A NICE THING TO DO / SOMETHING THAT CONFUSED US?
alternate_version = False
if alternate_version:
    L_list = [small_L,large_L]
else:
    L_list = [large_L]
for L in L_list: #COULD ALSO ADD BACK IN SMALL_L
    finite_size_eq = 1/(4*L)
    for j,W in enumerate(W_list):
        print("W=%.2f" % W)
        if alternate_version:
            LWlabel = r'$L=%i$; $W=%.1f$' % (L,W)
        else:
            LWlabel = r'$W=%i$' % W
        if W < 0.0001:
            num_runs = Hermitian_num_runs
            total_num_runs = num_runs
            if L == small_L:
                data_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2fSS=-1,%iruns_all_data.npy' % (L,Delta_1,W,num_runs))
            else:
                data_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2f,%iruns_all_data.npy' % (L,Delta_1,W,num_runs))
            data = np.load(data_filename)
        else:
            total_num_runs = 0
            num_runs_per_SS = nonHermitian_num_runs_per_SS
            for seed_start in range(num_seed_strings):
                print(seed_start)
                for num_runs in range(nonHermitian_num_runs_per_SS,0,-50): #try to grab as many runs as exist for this seed string, so start with higher
                    data_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2fSS=%i,%iruns_all_data.npy' % (L,Delta_1,W,seed_start,num_runs))
                    if os.path.exists(data_filename): #We've found the max number of runs that have been saved
                        
                        data_filename = os.path.join(data_dir,'L=%iD1=%.2fW=%.2fSS=%i,%iruns_all_data.npy' % (L,Delta_1,W,seed_start,num_runs))
                        print(data_filename)
                        break
                if seed_start == 0:
                    data = np.load(data_filename)
                    #FOR L = 18, STILL HAVE THE ZERO ISSUE ON FEB 16 2024!
                    if L == large_L:
                        should_all_be_pos = data[:,L//2,0]
                        beginning_of_zeros = np.argmax(should_all_be_pos < 0.001)
                        data = data[0:beginning_of_zeros,:,:]
                    total_num_runs += data.shape[0]
                else:
                    data_to_append = np.load(data_filename)
                    if L == large_L:
                        should_all_be_pos = data_to_append[:,L//2,0]
                        beginning_of_zeros = np.argmax(should_all_be_pos < 0.001)
                        data_to_append = data_to_append[0:beginning_of_zeros,:,:]
                    total_num_runs += data_to_append.shape[0]
                    
                    print(data_to_append[0,7,50])
                    print("^To check that different seeds are really different^")
                    #print(data_to_append[:,L//2,0])
                    data = np.append(data,data_to_append,axis=0)
        total_magnetization = np.sum(data,axis=1)
        print(np.max(np.abs(0.25 - total_magnetization)))
        print("^max deviation from M=0.25^")
        #NOW LET'S DO SAME-SITE DATA
        same_site_data = data[:,L//2,:]
        #Let's cut off everything after the run average hits 1/(4*L) for visual cleanliness and typicality reliability
        run_average = np.mean(same_site_data,axis=0)
        last_value = t.size
        if np.min(run_average) <= finite_size_eq:
            last_value = np.argmax(run_average <= finite_size_eq) #First time it hits or drops below eq
        if j == 0:
            if alternate_version:
                if L == small_L:
                    axs[0].axhline(y=finite_size_eq,color='k',ls=':',label=r'$\frac{1}{4(%i)}$' % L)
                else:
                    axs[0].axhline(y=finite_size_eq,color='k',ls='--',label=r'$\frac{1}{4(%i)}$' % L)
            else:
                axs[0].axhline(y=finite_size_eq,color='k',ls='--',label=r'$\frac{1}{4L}$')
        if W < 0.0001:
            if L == small_L:
                print(t.shape)
                print(same_site_data.shape)
                
                axs[0].plot(t[np.logspace(np.log10(first_index),np.log10(last_value),num_xs,dtype=int)],np.mean(same_site_data[:,np.logspace(np.log10(first_index),np.log10(last_value),num_xs,dtype=int)],axis=0),'x',label=r'$L=%i$ \& $W=%.1f$' % (L,W),color=color_list[j],markersize=12,mew=2)
            else:
                axs[0].plot(t[first_index:last_value],same_site_data[0,first_index:last_value],label=LWlabel, color = color_list[j])
                for k in range(1,Hermitian_num_runs):
                    axs[0].plot(t[first_index:last_value],same_site_data[k,first_index:last_value],color=color_list[j])
        else:
            if  L == small_L:
                axs[0].plot(t[np.logspace(np.log10(first_index),np.log10(last_value),num_xs,dtype=int)],run_average[np.logspace(np.log10(first_index),np.log10(last_value),num_xs,dtype=int)],'x',label=r'$L=%i$ \& $W=%.1f$' % (L,W),color=color_list[j],markersize=12,mew=2)
            else:
                std = np.std(same_site_data,axis=0)
                sem = std/np.sqrt(total_num_runs)
                axs[0].plot(t[first_index:last_value],run_average[first_index:last_value],label=LWlabel,color=color_list[j])
                axs[0].fill_between(t[first_index:last_value],(run_average - 2*sem)[first_index:last_value], (run_average + 2*sem)[first_index:last_value],alpha=0.75,color=color_list[j]) #2 SEM'S
        print("W=%.2f, total number of runs used is %i" % (W,total_num_runs))
        #NOW DO POWER LAW FIT
        #I SHOULD PROBABLY PUT THIS IN ITS OWN FUNCTION INSTEAD OF COPY-PASTING
        if L == large_L and j == 1:
            starting_value = run_average[hydro_start_index]
            t0 = t[hydro_start_index]
            def power_law_decay(t,alpha): # = const (t - t0)^{- \alpha}
                return starting_value*t0**alpha*t**(-alpha)
            #sigma = np.cov(same_site_data[:,hydro_start_index:last_value],rowvar=False,bias = True)#covariance matrix of {C(0,t_i) for each t_i}. bias=True to be consistent with 1/sqrt(N) convention used elsewhere in paper
            #print(sigma.shape)
            #print(sem.shape)
            #print(sem[hydro_start_index:last_value].shape)
            sigma = std[hydro_start_index:last_value] #DON'T USE CROSS-CORRELATIONS BETWEEN DIFFERENT TIMES BECAUSE IT RESULTS IN REALLY WEIRD RESULTS
            print("sigma = standard_deviations")
            #print(np.sqrt(np.diag(sigma))/np.sqrt(total_num_runs) - sem[hydro_start_index:last_value])
            #print("^should be all 0's after square rooting^")
            popt,pcov = scipy.optimize.curve_fit(power_law_decay,t[hydro_start_index:last_value],run_average[hydro_start_index:last_value],p0=0.66,sigma=sigma)
            optimal_alpha = popt[0]
            print(optimal_alpha)
            axs[0].plot(t[hydro_start_index:last_value],power_law_decay(t[hydro_start_index:last_value],optimal_alpha),linestyle='--',color=color_list[j])
            #if j == 0:
            #    ax.text(x=20,y=0.015,s=r"$\propto t^{-%.2f}$" % optimal_alpha,color=color_list[j])
            if j == 1:
                (x0,y0,width,height) = axs[0].get_position().bounds
                
                axs[0].text(x=20,y=0.025,s=r"{\Large $\propto t^{-%.2f}$}" % optimal_alpha,color=color_list[j],ha='center',va='center',rotation_mode='anchor',rotation=np.arctan(-optimal_alpha*height/width)*180/np.pi)
                #CHANGE ROTATION TO TAKE INTO ACCOUNT ASPECT RATIO
            #axs[i].plot(t[hydro_start_index:last_value],power_law_decay(t[hydro_start_index:last_value],optimal_alpha),linestyle='--',label='slope %.4f' % optimal_alpha,color=color_list[j])
            #if total_num_runs > max_num_runs:
            #    max_num_runs = total_num_runs
        #Now let's do the t=10 snapshot
        marker_list = ['.','x']
        if L == large_L:
            time_slice_data = data[:,:,time_slice_index]
            axs[1].plot(np.arange(L),np.mean(time_slice_data,axis=0),marker_list[j],label=r'$W=%i$' % W,color=color_list[j],markersize=10,mew=2)
axs[1].set_ylabel(r"$C(r,%i)$" % time_slice)
#axs[1].yaxis.set_label_position("right")
#axs[1].yaxis.tick_right()
axs[1].set_xlabel("$r$")
axs[1].set_xticks(np.arange(0,18,3),np.arange(0,18,3) - L//2)
axs[1].set_yscale('log')
axs[1].legend(markerfirst=False,frameon=False)

axs[0].set_ylabel("$C(0,t)$")
axs[0].set_xlabel("time $t$")
#axs[0].xaxis.set_label_position("top")
#axs[0].xaxis.tick_top()
axs[0].xaxis.set_label_coords(0.5, -0.05)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].legend(markerfirst=False,frameon=False)

x_to_put_letter = 0.03
y_to_put_letter = 0.15
(y_min,y_max) = axs[0].get_ylim()
(x_min,x_max) = axs[0].get_xlim()
print(x_min,x_max)
axs[0].text(x=10**(np.log10(x_min) + (np.log10(x_max) - np.log10(x_min))*x_to_put_letter),y=10**(np.log10(y_max) - (np.log10(y_max) - np.log10(y_min))*y_to_put_letter),s=r"{\Large\textbf{(a)}}",ha='center',va='center')
(y_min,y_max) = axs[1].get_ylim()
(x_min,x_max) = axs[1].get_xlim()
print(x_min,x_max)
axs[1].text(x=x_min + (x_max - x_min)*x_to_put_letter,y=10**(np.log10(y_max) - (np.log10(y_max) - np.log10(y_min))*y_to_put_letter),s=r"{\Large\textbf{(b)}}",ha='center',va='center')
#add_letter_labels(fig,axs,72,30,[r'$\Delta_1 = 1.0$',r'$\Delta_1 = 1.5$'],white_labels=False)

#labels = [r'\textbf{(a): $\Delta_1 = %.1f$}' % Delta_1_list[0],r'\textbf{(b): $\Delta_1 = %.1f$}' % Delta_1_list[1],r'\textbf{(c)}',r'\textbf{(d)}',r'\textbf{(e)}',r'\textbf{(f)}']
#for i,ax in enumerate(axs):
#    trans = mtransforms.ScaledTranslation(10/72, -30/72, fig.dpi_scale_trans)
#    ax.text(0.0, 1.0, labels[i], transform=ax.transAxes + trans,
#            fontsize='large', verticalalignment='top',fontweight='bold') #removed fontfamily = 'serif' #removed bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0)
if alternate_version:
    filename = os.path.join(fig_dir,'fig7alt.pdf')
else:
    filename = os.path.join(fig_dir,'fig7.pdf')

fig.savefig(filename,bbox_inches='tight')
plt.close(fig)