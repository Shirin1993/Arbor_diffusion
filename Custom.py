##### import sys
import pylab as pl
#from scitools import *
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
from scipy import linalg
import random as rnd
import itertools as itertools
import os
import matplotlib.pylab as plt 
import pickle
import matplotlib
import sklearn.metrics

gammaP = 90 *1000 # for units
gammaD = 0.01 *1000
thP = .00011*1e6
thD = .00005*1e6

I0 = 5.5*1e-12  # synaptic input current in [A]  
taudec =  1*1e-3  # synaptic input current decay time in [s]


pi = np.pi #3.14
numberOfBursts = 2
dly = 0
Deff = 220 * 1e-12 #diffusion coefficient of calcium in dendrite in [m^2/s ]

timetotal = 70*1e-3  # Total time of simulation in [s] 70
dt = .01 * 1e-3  # s   #Timestep duration in [s] 
time = int(timetotal / dt)  # Number of steps

tau = 100 *1e-3  # Decay time of dendrite (Inactivation Time constant ) in [s]
tauspine = 100*1e-3  # Decay time of spine (Inactivation Time constant ) in [s]

delta_t = str(dt*1000)

def ttoms(milisec, dtt):
    '''
    convert ms to simulation time
    currenttimestep : simulation timestep
    dtt : timestep in [s]
    '''
    return int(milisec / (dtt * 1e3))

input_time = ttoms(1, dt)  # first current input time
dts = ttoms(.1, dt)  # time interval between single spikes [timestep]
delay = ttoms(dly, dt)  # delays between 2 spines inputs [timestep]
interval = ttoms(9.0, dt)  # delays between 2 bursts [timestep]

numberofspike0 = 1
numberofspike1 = 1

N_spine = 4  # number of spines
N_dend = 320  # number of dendritic segments
L = 80 * 1e-6  # 10 um     #Total length of dendritic branch in [m]
dx = L / (N_dend)  # Length of each dendritic segments in [m]

steps=time
xtime=np.linspace(0,(steps*dt*1e3),steps)

spine_list = [160-2, 160, 160+2, 160+2+2+2]

# Ca Parameters
Z = 2  # Calcium valance

Faraday = 96485.309  # Faraday constant [C/mol]
y = 0.11  # Fraction of current carried by calcium through the receptor

# Geometrical parameters
radius = 1 * 1e-6  # dendritic radius in [m]
spineradius = 1 * 1e-6  # #spine radius in [m]  0.26-1.10 /2  mean 0.68 /2 = 0.34
spine_len =  1 * 1e-6 # To be similar to Arbor
# rate of Ca influx from dend to spine neck (small hole theory) (Biess et al. 2011)
Vs = pi*spineradius*spineradius*spine_len#(4 / 3) * np.pi * spineradius ** 3  # Spine volume 
# synaptic weight parameters

w0 = 0.5  # initial synaptic weight of spine 1
w1 = 0.5  # initial synaptic weight of spine 1
w2 = 0.5
w3 = 0.5
adj = np.zeros(N_dend)  # dednd adjacent list (dend-spine connection)
adj[spine_list] = 1


##########################################################



B = []  # needed current convolution
for it in range(time):
    B.append(I0 * np.exp(-it * dt / taudec))

B_0 = []  # needed current convolution
I0B_0 = 1*I0
for it in range(time):
    B_0.append(I0B_0 * np.exp(-it * dt / taudec))


train = np.zeros((N_spine, time))  # current I(mA) list for spine


for d in range(numberOfBursts):
    for i in range(numberofspike0):
        train[0, d * interval + input_time + i * dts] = 1
    for i in range(numberofspike1):
        train[2, d * interval + input_time + i * 2*dts] = 1

# current convolution
xs = np.zeros((N_spine, time))
for spn in [0]:#range(N_spine):
    cc = np.convolve(train[spn, :], B)
    xs[spn, :] = cc[0:int((len(cc) + 1) / 2)]
for spn in [2]:#range(N_spine):
    cc = np.convolve(train[spn, :], B_0)
    xs[spn, :] = cc[0:int((len(cc) + 1) / 2)]

# creating matrix for differential eq. and defining parameters
alpha = Deff * dt / dx ** 2 
beta = Deff * dt / dx ** 2 

F = Deff * dt / dx ** 2
gamma = dt * y / (Z * Vs * Faraday)


##______________________________________________

dec = dt / tau
decspine = dt / tauspine

sincm = 1
if N_dend == 1:
    sincm = 0
A = np.zeros((N_dend + N_spine, N_dend + N_spine))
for i in range(N_spine + 1, N_dend + N_spine - 1):
    A[i, i] = 1 + 2 * F + (adj[i - N_spine]) * alpha + dec
A[N_spine, N_spine] = 1 + (sincm * F) + (adj[0]) * alpha  # first dnd
A[N_dend + N_spine - 1, N_dend + N_spine - 1] = 1 + (sincm * F) + (adj[N_dend - 1]) * alpha
for i in range(N_spine, N_dend + N_spine - 1):
    A[i, i + 1] = -F
for i in range(N_spine + 1, N_dend + N_spine):
    A[i, i - 1] = -F
for i in range(N_spine):
    A[i, i] = 1 + beta + decspine
for i in range(N_spine):
    A[i, N_spine + spine_list[i]] = - alpha
for i in range(N_spine):
    A[N_spine + spine_list[i], i] = - beta

b = np.zeros((N_spine + N_dend, time))
U = np.zeros((N_spine + N_dend, time))

U[0:N_spine, 0] = 0  
U[N_spine:N_spine + N_dend, 0] = 0


def theta(ar):  # Heaviside func for calcium thresholding
    listt = np.zeros(len(ar))
    for values in range(len(ar)):
        if ar[values] >= 0:
            listt[values] = 1
        if ar[values] < 0:
            listt[values] = 0
    return listt


weight = np.zeros((N_spine, time))
weight[0, 0] = w0
weight[1, 0] = w1
weight[2, 0] = w2
weight[3, 0] = w3

# Solve Backward Euler for calcium diffusion eq.
for t in range(time - 1):
    b[0:N_spine, t] = gamma * xs[:, t] * weight[:, t]
    C = U[:, t] + b[:, t]
    X = linalg.solve(A, C)
    weight[:, t + 1] = weight[:, t] + dt * (gammaP * (1 - weight[:, t]) * theta(1e6 * U[0:N_spine, t] - thP) - gammaD * weight[:, t] * theta(1e6 * U[0:N_spine, t] - thD))

    U[:, t + 1] = np.asarray(X).flatten()

for i in range(4):
    data_to_save = np.copy( 1e-3*1e6*U[i, :]) 
    filename = f'PCaS{i+1}_{delta_t}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(data_to_save, file)
    print(f"Data saved to {filename} using pickle.")

for i in range(4):
    data_to_savew = np.copy( weight[i, :]) 
    filenamew = f'PWS{i+1}_{delta_t}.pkl'
    with open(filenamew, 'wb') as filew:
        pickle.dump(data_to_savew, filew)
    print(f"Data saved to {filenamew} using pickle.")


for i in [159+N_spine,161+N_spine,164+N_spine]:
    data_to_save = np.copy( 1e-3*1e6*U[i, :]) # in uM # the U is in the unit of mol/m^3 . which is equal to 0.001 M . or 1 mM
    filename = f'PCaD{i}_{delta_t}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(data_to_save, file)



#______________________________________________
# Plots :


delta_t = 0.01
markersizes = 3



coefunit = 1e3
NumSpines = 4
fig, axes = plt.subplots(NumSpines, sharex=True,figsize=(10, 8), dpi=180)
#Arbor
yk = []
ykl = []
xk = np.arange(0,8000,1000)
for i in range(8):
    yk.append((np.round( i*1e-3*thD,2)))  
    ykl.append(("%.2f" % round(i*1e-3*thD, 2)))  
    
# xtik = np.round(xk,3)
for i in range(4):
    axes[i].plot([1e-3*thD]*len(S0))
    axes[i].plot([1e-3*thP]*len(S0))
    axes[i].set_yticks(yk,ykl)
    axes[i].set_xticks(xk,xk/100)

#Python 

delta_tp = str(dt*1000)
# Load all data from the single pickle file
filename = f'PCaS1_{delta_tp}.pkl'
with open(filename, 'rb') as file:
    loaded_all_data = pickle.load(file)

# Assign each piece of data to a variable
US0 = loaded_all_data
filename = f'PCaS2_{delta_tp}.pkl'
with open(filename, 'rb') as file:
    loaded_all_data = pickle.load(file)
    
US1 = loaded_all_data
filename = f'PCaS3_{delta_tp}.pkl'
with open(filename, 'rb') as file:
    loaded_all_data = pickle.load(file)
    
US2 = loaded_all_data
filename = f'PCaS4_{delta_tp}.pkl'
with open(filename, 'rb') as file:
    loaded_all_data = pickle.load(file)
US3 = loaded_all_data

line_color = '#008111'
axes[0].plot(US0,'.', markersize = markersizes,color=line_color,alpha = 0.5, label= f'Custom {np.round(US0.max(),5)} \n final  {np.round(US0[-1],3)}')#color=plt.cm.terrain(f * vmax))  # , color='tab:orange')
axes[1].plot(US1, '.',markersize = markersizes,color=line_color,alpha = 0.5, label= f'Custom {np.round(US1.max(),5)} \n final  {np.round(US1[-1],3)}')#color=plt.cm.terrain(f * vmax))  # , color='tab:orange')
axes[2].plot(US2, '.',markersize = markersizes,color=line_color,alpha = 0.5, label= f'Custom {np.round(US2.max(),5)} \n final  {np.round(US2[-1],3)}')#color=plt.cm.terrain(f * vmax))  # , color='tab:orange')
axes[3].plot(US3, '.',markersize = markersizes,color=line_color,alpha = 0.5, label= f'Custom {np.round(US3.max(),5)} \n final  {np.round(US3[-1],3)}')#color=plt.cm.terrain(f * vmax))  # , color='tab:orange')

axes[int(NumSpines/2)].set_ylabel("ca concentration (uM)")
axes[0].set_title("spine0")
axes[1].set_title("spine1")
axes[2].set_title("spine2")
axes[3].set_title("spine3")
axes[3].set_xlabel("t [ms]")
plt.subplots_adjust(hspace=0.9)  # Increase the space between plots
for i in range(NumSpines):
    axes[i].legend(loc ='upper right')

ylilmmax = 0.35
ylilmmin = 0
axes[0].set_ylim(ylilmmin,ylilmmax)
axes[2].set_ylim(ylilmmin,ylilmmax)
axes[1].set_ylim(ylilmmin,ylilmmax)
axes[3].set_ylim(ylilmmin,ylilmmax)


#Weights

delta_t = 0.01

coefunit = 1
NumSpines = 4
fig, axes = plt.subplots(NumSpines, sharex=True,figsize=(10, 8), dpi=180)

#Python 

delta_tp = str(dt*1000)
# Load all data from the single pickle file
filename = f'PWS1_{delta_tp}.pkl'
with open(filename, 'rb') as file:
    loaded_all_data = pickle.load(file)

# Assign each piece of data to a variable
WPS0 = loaded_all_data
filename = f'PWS2_{delta_tp}.pkl'
with open(filename, 'rb') as file:
    loaded_all_data = pickle.load(file)
    
WPS1 = loaded_all_data
filename = f'PWS3_{delta_tp}.pkl'
with open(filename, 'rb') as file:
    loaded_all_data = pickle.load(file)
    
WPS2 = loaded_all_data
filename = f'PWS4_{delta_tp}.pkl'
with open(filename, 'rb') as file:
    loaded_all_data = pickle.load(file)
WPS3 = loaded_all_data

line_color = '#008111'
axes[0].plot(WPS0,'-',markersize = 1, color=line_color,alpha = 0.5, label= f'Python {np.round(WPS0.max(),5)} \n final  {np.round(WPS0[-1],3)}')#color=plt.cm.terrain(f * vmax))  # , color='tab:orange')
axes[1].plot(WPS1, '-',markersize = 1,color=line_color,alpha = 0.5, label= f'Python {np.round(WPS1.max(),5)} \n final  {np.round(WPS1[-1],3)}')#color=plt.cm.terrain(f * vmax))  # , color='tab:orange')
axes[2].plot(WPS2, '-',markersize = 1,color=line_color,alpha = 0.5, label= f'Python {np.round(WPS2.max(),5)} \n final  {np.round(WPS2[-1],3)}')#color=plt.cm.terrain(f * vmax))  # , color='tab:orange')
axes[3].plot(WPS3, '-',markersize = 1,color=line_color,alpha = 0.5, label= f'Python {np.round(WPS3.max(),5)} \n final  {np.round(WPS3[-1],3)}')#color=plt.cm.terrain(f * vmax))  # , color='tab:orange')

axes[int(NumSpines/2)].set_ylabel("ca concentration (uM)")
axes[0].set_title("spine0")
axes[1].set_title("spine1")
axes[2].set_title("spine2")
axes[3].set_title("spine3")
axes[3].set_xlabel("t step")
plt.subplots_adjust(hspace=0.9)  # Increase the space between plots
for i in range(NumSpines):
    axes[i].legend(loc ='upper right')


ylilmmax = 1.2
ylilmmin = .3
axes[0].set_ylim(ylilmmin,ylilmmax)
axes[2].set_ylim(ylilmmin,ylilmmax)
# ylilmmax = 25

axes[1].set_ylim(ylilmmin,ylilmmax)
axes[3].set_ylim(ylilmmin,ylilmmax)
for i in range(4):
    axes[i].plot([0.5]*len(WPS0), '--',c='gray')


# Dendrite:
#Dendrite


delta_t = 0.01
markersizes = 3



coefunit = 1e3
NumDend = 3
fig, axes = plt.subplots(3, sharex=True,figsize=(10, 6), dpi=180)

yk = []
ykl = []

xk = np.arange(0,8000,1000)
for i in range(8):
    yk.append((np.round( i*1e-3*thD,2)))  
    ykl.append(("%.2f" % round(i*1e-3*thD, 2)))  
    
    
for i in range(NumDend):

    axes[i].set_yticks(yk,ykl)
    axes[i].set_xticks(xk,xk/100)
    
    

#Python 

dendID =   [159+N_spine,161+N_spine,164+N_spine]
delta_tp = str(dt*1000)

# Load all data from the single pickle file
filename = f'PCaD{dendID[0]}_{delta_tp}.pkl'
with open(filename, 'rb') as file:
    loaded_all_data = pickle.load(file)

# Assign each piece of data to a variable
UD0 = loaded_all_data
filename = f'PCaD{dendID[1]}_{delta_tp}.pkl'
with open(filename, 'rb') as file:
    loaded_all_data = pickle.load(file)
    
UD1 = loaded_all_data
filename = f'PCaD{dendID[2]}_{delta_tp}.pkl'
with open(filename, 'rb') as file:
    loaded_all_data = pickle.load(file)
    
UD2 = loaded_all_data


line_color = '#008111'
axes[0].plot(UD0,'.', markersize = markersizes,color=line_color,alpha = 0.5, label= f'Python {np.round(UD0.max(),5)} \n final  {np.round(UD0[-1],3)}')#color=plt.cm.terrain(f * vmax))  # , color='tab:orange')
axes[1].plot(UD1, '.',markersize = markersizes,color=line_color,alpha = 0.5, label= f'Python {np.round(UD1.max(),5)} \n final  {np.round(UD1[-1],3)}')#color=plt.cm.terrain(f * vmax))  # , color='tab:orange')
axes[2].plot(UD2, '.',markersize = markersizes,color=line_color,alpha = 0.5, label= f'Python {np.round(UD2.max(),5)} \n final  {np.round(UD2[-1],3)}')#color=plt.cm.terrain(f * vmax))  # , color='tab:orange')

axes[1].set_ylabel("ca concentration (uM)")
axes[0].set_title("Dend0")
axes[1].set_title("Dend1")
axes[2].set_title("Dend2")
axes[2].set_xlabel("t [ms]")
plt.subplots_adjust(hspace=0.9)  # Increase the space between plots
for i in range(len(dendID)):
    axes[i].legend(loc ='upper right')


ylilmmax = 0.35
ylilmmin = 0
axes[0].set_ylim(ylilmmin,ylilmmax)
axes[2].set_ylim(ylilmmin,ylilmmax)

axes[1].set_ylim(ylilmmin,ylilmmax)
axes[2].set_ylim(ylilmmin,ylilmmax)

