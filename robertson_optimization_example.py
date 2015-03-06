# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# We start by importing the calls for basic functions we use with the modeling. Numpy is for controlling data and matplotlib is for graphics

# <codecell>

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# <markdowncell>

# Next we will import the model and the odesolver from pysb

# <codecell>

from pysb.examples.robertson import model
from pysb.integrate import odesolve
import pysb

# <markdowncell>

# H. H. Robertson, The solution of a set of reaction rate equations, in Numerical
# Analysis: An Introduction, J. Walsh, ed., Academic Press, 1966, pp. 178-182.

# <markdowncell>

# $
# A \rightarrow B \\
# 2B \rightarrow B + C   \\
# B + C \rightarrow A + C  \\$

# <markdowncell>

# We will integrate the model for 40 seconds.

# <codecell>

t = np.linspace(0, 40,100)
obs_names = ['A_total', 'C_total']

# <markdowncell>

# Here we run solve the ode. We pass it the model, time, and any extra arguments ( here we provide r tolerace nad a tolerance)

# <codecell>

solver = pysb.integrate.Solver(model, t, integrator='lsoda',rtol=1e-8, atol=1e-8)

# <codecell>

solver.run()

# <codecell>

def normalize(trajectories):
    """Rescale a matrix of model trajectories to 0-1"""
    ymin = trajectories.min(0)
    ymax = trajectories.max(0)
    return (trajectories - ymin) / (ymax - ymin)

def extract_records(recarray, names):
    """Convert a record-type array and list of names into a float array"""
    return np.vstack([recarray[name] for name in names]).T

# <codecell>

yobs= extract_records(solver.yobs, obs_names)
norm_data = normalize(yobs)

# <codecell>

plt.plot(t,norm_data)
plt.legend(['A_Total','C_Total'], loc = 0)

# <markdowncell>

# We are going to make some noisy data for optimization.

# <codecell>

noisy_data_A = yobs[:,0] + np.random.uniform(-0.02,0.02,np.shape(yobs[:,0]))
norm_noisy_data_A = normalize(noisy_data_A)
noisy_data_C = yobs[:,1] + np.random.uniform(-.01,.01,np.shape(yobs[:,1]))
norm_noisy_data_C = normalize(noisy_data_C)
ydata_norm = np.column_stack((norm_noisy_data_A,norm_noisy_data_C))

# <codecell>

plt.plot(t,norm_noisy_data_A)
plt.plot(t,norm_noisy_data_C)
plt.plot(t,norm_data)
plt.legend(['A_total_noisy','C_total_noisy','A_total', 'B_total', 'C_total'], loc=0)

# <codecell>

param_values = np.array([p.value for p in model.parameters])
rate_mask = np.array([p in rate_params for p in model.parameters])
nominal_values = np.array([p.value for p in model.parameters])
xnomial = np.log10(nominal_values[rate_mask])

def display(x=None):
    if x == None:
        solver.run(param_values)
        
    else:
        Y=np.copy(x)
        param_values[rate_mask] = 10 ** Y
        solver.run(param_values)
    ysim_array = extract_records(solver.yobs, obs_names)
    ysim_norm = normalize(ysim_array)
    plt.figure(figsize=(8,6),dpi=200)
    plt.plot(t,ysim_norm[:,0],label='A')
    plt.plot(t,ysim_norm[:,1],label='C')
    plt.plot(t,norm_noisy_data_A,label='Noisy A')
    plt.plot(t,norm_noisy_data_C,label='Noisy C')
    if x ==None:
        print ''
    else:
        plt.plot(t,norm_data,label=['Ideal'])
    plt.legend(loc=0)
    plt.ylabel('concentration')
    plt.xlabel('time (s)')
    plt.show()

# <codecell>

display()

# <codecell>

from scipy.optimize import minimize

# <codecell>

rate_params = model.parameters_rules()
param_values = np.array([p.value for p in model.parameters])
rate_mask = np.array([p in rate_params for p in model.parameters])
# Build a boolean mask for those params against the entire param list
k_ids = [p.value for p in model.parameters_rules()]

# <codecell>

nominal_values = np.array([p.value for p in model.parameters])
xnominal = np.log10(nominal_values[rate_mask])
bounds_radius = 1
lb = xnominal - bounds_radius
ub = xnominal + bounds_radius

# <codecell>

print xnominal
start_position = xnominal +2*np.random.uniform(-1,1,size = np.shape(xnominal))
print start_position
display(start_position)

# <codecell>

def obj_function(params):
    # Apply hard bounds
    if np.any((params < lb) | (params> ub)):
        #print "bounds-check failed"
        return 1000
    params_tmp = np.copy(params)
    param_values[rate_mask] = 10 ** params_tmp
    solver.run(param_values)
    ysim_array = extract_records(solver.yobs, obs_names)
    ysim_norm = normalize(ysim_array)
    err = np.sum((ydata_norm - ysim_norm) ** 2 )
    if np.isnan(err):
        return 1000
    #print err
    return err

# <markdowncell>

# There are many existing optimization algorithms written in Python (remember, we don't want to reinvent the wheel). Scipy is a general package that includes many methods. We will demonstrate the basic minimization with the Nelder-mead algorithm.
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
# 

# <codecell>

results = minimize(obj_function,start_position,method='Nelder-mead',options={'xtol': 1e-8, 'disp': True})

# <codecell>

print results
best = np.reshape(results['x'],np.shape(xnominal))

# <codecell>

display(start_position)
display(best)

# <codecell>


