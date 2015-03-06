# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import operator
import random
import scipy.optimize
import numpy
import pylab
from deap import base
from deap import creator
from deap import tools
import pysb.integrate
import pysb.util
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import os
import inspect
import multiprocessing
import multiprocessing as mp
from earm.lopez_embedded import model

# <codecell>

obs_names = ['mBid', 'aSmac', 'cPARP']
data_filename = '/home/pinojc/git/earm/earm/experimental_data.npy'
ydata_norm = numpy.load(data_filename)
exp_var = 0.2
tspan = np.linspace(0,5.5 * 3600,len(ydata_norm)*10)  # 5.5 hours, in seconds

# <codecell>

rate_params = model.parameters_rules()
param_values = np.array([p.value for p in model.parameters])
rate_mask = np.array([p in rate_params for p in model.parameters])
# Build a boolean mask for those params against the entire param list
k_ids = [p.value for p in model.parameters_rules()]

# <codecell>

solver = pysb.integrate.Solver(model, tspan, integrator='vode',rtol=1e-8, atol=1e-8)

# <codecell>

def likelihood(x):

    Y=np.copy(x)
    param_values[rate_mask] = 10 ** Y
    solver.run(param_values)
    ysim_array = extract_records(solver.yobs, obs_names)
    ysim_norm = normalize(ysim_array)
    err = numpy.sum((ydata_norm - ysim_norm[::10]) ** 2 / (2 * exp_var ** 2))
    return err,

# <markdowncell>

# The experimental data is normalized from [0,1] so we define a function that will normalized our simulated trajectories. We also define a function to extract the obserables from the simulations.

# <codecell>

def normalize(trajectories):
    """Rescale a matrix of model trajectories to 0-1"""
    ymin = trajectories.min(0)
    ymax = trajectories.max(0)
    return (trajectories - ymin) / (ymax - ymin)

def extract_records(recarray, names):
    """Convert a record-type array and list of names into a float array"""
    return numpy.vstack([recarray[name] for name in names]).T

# <markdowncell>

# To start the optimization we will use values that were used to create the model. We will search parameter space in log space and set upper and lower bounds to 1 order of magnitude up and down.

# <codecell>

nominal_values = np.array([p.value for p in model.parameters])
xnominal = np.log10(nominal_values[rate_mask])
bounds_radius = 1
lb = xnominal - bounds_radius
ub = xnominal + bounds_radius

# <markdowncell>

# To use the particle swarm algorithm we need to first generate a population and define an update function. Below is a basic example of the general PSO modified from Deap.

# <codecell>


def generate(size, speedmin, speedmax):
    
    random_pertubation = (random.uniform(-1., 1.) for _ in range(size))
    start_position = map(operator.add,np.log10(k_ids),random_pertubation)
    part = creator.Particle(start_position)
    part.speed = [random.uniform(speedmin, speedmax) for _ in range(size)]
    part.smin = speedmin
    part.smax = speedmax
    return part

def updateParticle(particle_position, best, phi1, phi2):

    u1 = (random.uniform(0, phi1) for _ in range(len(particle_position)))
    u2 = (random.uniform(0, phi2) for _ in range(len(particle_position)))
    v_u1 = map(operator.mul, u1, map(operator.sub, particle_position.best, particle_position))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, particle_position))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(particle_position.speed):
        if speed < particle_position.smin:
            particle_position.speed[i] = particle_position.smin
        elif speed > particle_position.smax:
            particle_position.speed[i] =  part.smax
    part[:] = list(map(operator.add, particle_position, particle_position.speed))
    for i, pos in enumerate(particle_position):
        if pos < lb[i]:
            part[i] = lb[i]
        elif pos > ub[i]:
            part[i] =  ub[i]

# <markdowncell>

# Now we use Deap to create the optimization protocol. 

# <codecell>

toolbox = base.Toolbox()
creator.create("FitnessMin", base.Fitness,weights=(-1.00,))
creator.create("Particle", list, fitness=creator.FitnessMin, \
    speed=list,smin=list, smax=list, best=None)
toolbox.register("particle", generate, size=np.shape(rate_params)[0],\
                 speedmin=-.1,speedmax=.1)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2, phi2=2)
toolbox.register("evaluate", likelihood)

# <markdowncell>

# Deap also has tools to keep track of some statistics.

# <codecell>

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)
logbook = tools.Logbook()
logbook.header = ["gen", "evals"] + stats.fields

# <markdowncell>

# Here we create a function to display a given parameter set to the experimental data.

# <codecell>

def display(x):
    Y=np.copy(x)
    param_values = np.array([p.value for p in model.parameters])
    rate_mask = np.array([p in rate_params for p in model.parameters])
    param_values[rate_mask] = 10 ** Y
    solver.run(param_values)
    ysim_array = extract_records(solver.yobs, obs_names)
    ysim_norm = normalize(ysim_array)
    count=1
    plt.figure(figsize=(8,6),dpi=200)
    for j,obs_name in enumerate(obs_names):
      plt.subplot(3,1,count)
      plt.plot(solver.tspan,ysim_norm[:,j])
      plt.plot(solver.tspan[::10],ydata_norm[:,j],'-x')
      plt.title(str(obs_name))
      count+=1
    plt.ylabel('concentration')
    plt.xlabel('time (s)')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

# <markdowncell>

# This example uses parallel computing so we must tell all processers the information we need which is a list of the parameters for each iteration and a dictonary to hold the results. This way we send the largest part of data to each processor once. This way we can send the processor a key and its places the result into a dictonary. This can be done others but this is a just a quick example.

# <codecell>

def init(sample,dictionary):
    global Sample
    global Dictionary
    Sample,Dictionary = sample,dictionary

# <codecell>

def OBJ(block):
    #print block
    obj_values[block]=likelihood(sample[block])

# <markdowncell>

# Finally, the main call of the function. 

# <codecell>

if __name__ == '__main__':
    print "Model with rate parameters from literature"
    display(xnominal)
    GEN = 100
    num_particles = 50
    pop = toolbox.population(n=num_particles)
    best = creator.Particle(xnominal)
    best.fitness.values = likelihood(xnominal)
    best_values =[]
    evals = []
    print logbook.header
    for g in range(1,GEN+1):
        m = mp.Manager()
        obj_values = m.dict()
        sample = []
        for p in pop:
            sample.append(p)
        p = mp.Pool(4,initializer = init, initargs=(sample,obj_values))
        allblocks =range(len(pop))
        p.imap_unordered(OBJ,allblocks)
        p.close()
        p.join()
        count=0
        
        for part in pop:
            part.fitness.values = obj_values[count]
            count+=1
            if g == 1:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            elif part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)
 
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream),best.fitness.values
        best_values.append(best.fitness.values)
        evals.append(g*num_particles)
    
    
    display(best)
    plt.semilogy(evals,best_values)

# <codecell>


