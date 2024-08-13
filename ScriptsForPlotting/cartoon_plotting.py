#!/usr/bin/env python
# coding: utf-8

# In[82]:


import sys
import os

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

color_list = get_list_of_colors_I_like(3)

t_max=500
t_step=0.01
num_times = int(t_max/t_step)+1
t = np.logspace(-2,np.log10(t_max),num_times)
fig,ax = plt.subplots(1,1,sharex='col',sharey='row')
early_time_constant = 0.1
early_time_frequency = 4

def infiniteC(t):
    return 0.01*(1+ np.exp(-t/early_time_constant)*np.cos(np.log(t)*early_time_frequency))/np.sqrt(t)
tau = 15 #rough amount of time taken to transition from agreeing with infinite to finite_size_limit
finite_limit = 0.04/np.sqrt(t_max)
#Now let's construct a smooth transition function using np.piecewise so that it can be vectorized
def f(x):
    return np.piecewise(x,[x<=0,x>0],[0,lambda x: np.exp(-1/x)])
def sigma(a,b,x):
    return f(x-a)/(f(x-a) + f(b-x))

T_index = np.where(infiniteC(t)<finite_limit)[0][0] #time when the transition to finite-size limit is over
T = t[T_index]
print(T)
print("just printed T")

def finiteC(t): #only from times before T
    #the exponential decay to the finite value is exponential *after* applying logarithmic axes
    starting_slope = (np.log10(infiniteC(T-tau))-np.log10(infiniteC(T-tau-t_step)))/(np.log10(T -tau) - np.log10(T -tau - t_step))
    A = np.log10(infiniteC(T-tau)) - np.log10(finite_limit)
    B = -starting_slope/A
    #exponential is finite_limit + A e^(-B*(t-(T-tau)))
    
    return np.piecewise(t,[t<T-tau,t>=T-tau],[lambda t: infiniteC(t),lambda t: 10**(np.log10(finite_limit) + A*np.exp(-B*(np.log10(t)-np.log10(T-tau))))])
    #return (1-sigma(T-tau,T,t))*infiniteC(t) + sigma(T-tau,T,t)*finite_limit
    #return np.exp(-tau/(T - t))*infiniteC(t) + (1 - np.exp(-tau/(T - t)))*finite_limit

color = 'k'
ax.plot(t,infiniteC(t),color=color,label='Infinite system size')

color = color_list[0]
ax.plot(t,finiteC(t),color=color,ls='--',label='Finite system size')

color=color_list[1]
ax.plot(t,0.01/np.sqrt(t),ls=':',label='Power-law fit')


ax.set_ylabel(r"\Large $C(0,t)$")
ax.set_xlabel(r"time $t$")

ax.legend(frameon=False)

ax.set_xscale('log')
ax.set_yscale('log')

start_time = np.min(t)
first_transition_time = 3e-1
second_transition_time = 2e1
end_time = np.max(t)

early_height = 7e-3
intermediate_height = 3e-2
late_height = early_height

#code adapted from ChatGPT:
def annotation_with_bars(text,text_y,t_left_limit,t_right_limit):
    # Add text annotation
    text_x = 10**((np.log10(t_left_limit)+np.log10(t_right_limit))/2)  # Coordinates for the text
    ax.text(text_x, text_y, text, ha='center')

    # Add the symbol |-----| under the text
    symbol_x = text_x  # x-coordinate for the symbol
    symbol_y = 0.85*text_y  # y-coordinate for the symbol (slightly below the text)

    # Draw the vertical bars
    ax.plot([t_left_limit, t_left_limit], [symbol_y*0.9, symbol_y/0.9], color='black')  # |
    ax.plot([t_right_limit, t_right_limit], [symbol_y*0.9, symbol_y/0.9], color='black')  # |

    # Draw the horizontal line
    ax.plot([t_left_limit, t_right_limit], [symbol_y, symbol_y], color='black')  # -----

annotation_with_bars('Early times',early_height,start_time,first_transition_time)
annotation_with_bars('Intermediate times',intermediate_height,first_transition_time,second_transition_time)
annotation_with_bars('Late times',late_height,second_transition_time,end_time)
fig.set_size_inches(6.4,4.8)

filename = 'fig10.pdf'
fig.savefig(filename,bbox_inches='tight')
plt.close(fig)

