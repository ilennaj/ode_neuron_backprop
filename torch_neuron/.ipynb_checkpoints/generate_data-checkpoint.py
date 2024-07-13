# dataset.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import optim

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pandas as pd
import scipy.signal
import scipy.stats
import csv
import pickle
import networkx as nx

from scipy.optimize import curve_fit
from pathlib import Path

from torch_neuron.training import *
from torch_neuron.architectures import *
from torch_neuron.morphology import *

from torchdiffeq import odeint_adjoint as odeint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialize_model_stim(params,
                          hyperparams,
                          adj_mat, 
                          input_current_array=None,
                          manual_dist=None,
                          neurons=2,
                         ):
    # Initialize model to generate simulated target data
    
    model = Layer_2_3_stim(params,
                          hyperparams,
                          adj_mat, 
                          manual_dist=hyperparams['manual_dist'],
                          neurons=neurons,
                          input_current_array=input_current_array,
                          method='dopri5',
                          ax_weight=hyperparams['ax_weight'],
                          constant=True,
                          distribution=hyperparams['distribution'],
                          mu=hyperparams['mu'],
                          sigma=hyperparams['sigma'],
                          input_type='manual',
                          ).to(device)
    return(model)

def generate_step_current(hyperparams=None, nodes=5, start=None, stop=None, node_currents=None, method='randomsample',
                          step_indices=None):
    # Set step current based on a method - "variablestep" defaults to the else condition
    
    if hyperparams == None:
        hyperparams = set_hyperparams()
    
    tt = torch.linspace(float(hyperparams['sim_params']['t0']), float(hyperparams['sim_params']['T']), \
                        int((float(hyperparams['sim_params']['T']) - float(hyperparams['sim_params']['t0']))/hyperparams['sim_params']['dt']))[1:-1]
    
    # Initialize start and stop of step currents if None
    if start == None:
        start = np.ones(nodes)
    if stop == None:
        stop = np.ones(nodes)
    
    # Create a mask
    mask = torch.zeros((nodes, len(tt)))
    
    for node in np.arange(nodes):
        mask[node, int(start[node]):-int(stop[node])] = 1
       

    if method=='randomsample':
    
        # Initialize node currents if None
        if node_currents == None:
            node_currents = torch.zeros((nodes,1))

        # If node currents 1 dimensional, add a dimension
        if len(node_currents.shape) == 1:
            node_currents = node_currents.unsqueeze(1)

        # Make input current array 
        input_currents = node_currents.repeat(1, len(tt))
        
         # Apply start and end mask
        input_currents = node_currents * mask
        
    else: #variable step current
        
        if step_indices == None:
            step_indices = torch.zeros((hyperparams['lin_nodes'],1))
        
        # Convert index list to indices for changepoints
        changepoints = torch.zeros(step_indices.size()[0],step_indices.size()[1]+2)
        changepoints[:, 1:-1] = step_indices
        changepoints[:, -1] = -1


        cp_indices = changepoints.to(int)
        
        # Initialize input currents (nodes x timesteps)
        input_currents = torch.ones(hyperparams['lin_nodes'],len(tt))

        for i in np.arange(changepoints.size()[0]): # per node
            for j in np.arange(changepoints.size()[1]-1): # per index
                
                input_currents[i,cp_indices[i,j]:cp_indices[i,j+1]] = node_currents[i,j]


    input_currents = input_currents*mask
   
    
    # Return input current array
    return input_currents


    


def initialize_currents_stim(hyperparams=None, 
                             num_samples=100, 
                             i_min=0, 
                             i_max=20, 
                             method='randomsample',
                            ):
    # Initialize the currents - else condition is is "variablestep" current
    
    # Initialize hyperparameters
    if hyperparams == None:
        hyperparams = set_hyperparams_stim()
        
    hazard_rate = hyperparams['hazard_rate']
    
    if hyperparams['stim_locations'] == None:
        stim_locations = np.arange(hyperparams['lin_nodes'])
    else:
        stim_locations = hyperparams['stim_locations']
    
    # Initialize Stimulation Mask
    stim_mask = torch.zeros((hyperparams['lin_nodes']))
    
    if isinstance(stim_locations, list):
        stim_locations = np.array(stim_locations)
    # Only use locations within the neuron    stim_locations = stim_locations[stim_locations < hyperparams['lin_nodes']]
    # Make stim_mask
    stim_mask[stim_locations] = 1
    
    # Initialize node_currents array with random current values
    node_currents_array = torch.zeros((num_samples, hyperparams['lin_nodes']))
    
    # Initialize input_currents array (sample x nodes x time)
    timesteps = int(hyperparams['sim_params']['T']/hyperparams['sim_params']['dt'])-2
    input_currents_array = torch.zeros((num_samples, hyperparams['lin_nodes'], timesteps))
    
    if method == 'randomsample':
        # Randomly sample from range of currents for each node_current setting
        for sample in range(num_samples):
            node_currents_array[sample] = torch.Tensor(random.choices(range(i_min, i_max), weights=None, k=hyperparams['lin_nodes']))

        # Produce set of input_currents
        for sample, node_currents in enumerate(node_currents_array):
            input_currents_array[sample] = generate_step_current(hyperparams, 
                                                                 nodes=hyperparams['lin_nodes'],
                                                                 node_currents=node_currents,
                                                                 method=method)
    else:

        # Number Indices when the current changes
        step_changes = int(timesteps*hazard_rate)

        # Initialize node_currents array with random current values
        node_currents_array = torch.zeros((num_samples, hyperparams['lin_nodes'], step_changes+1))

        # Initialize input_currents array (sample x nodes x time)

        input_currents_array = torch.zeros((num_samples, hyperparams['lin_nodes'], timesteps))

        # Randomly initialize indices when step current changes        
        step_indices = torch.multinomial(torch.arange(0,timesteps).to(torch.float), num_samples=step_changes, replacement=False)
        step_indices, _ = torch.sort(step_indices)

        # Randomly sample from range of currents for each node_current setting
        for sample in range(num_samples):
            for changepoint in range(step_changes+1):
                node_currents_array[sample, :,changepoint] = torch.Tensor(random.choices(range(i_min, i_max), weights=None, k=hyperparams['lin_nodes']))

        for sample, node_currents in enumerate(node_currents_array):
            step_indices_array = torch.zeros((hyperparams['lin_nodes'], step_changes))
            for node in np.arange(hyperparams['lin_nodes']):
                step_indices = torch.multinomial(torch.arange(0,timesteps).to(torch.float), num_samples=step_changes, replacement=False)
                step_indices, _ = torch.sort(step_indices)
                step_indices_array[node] = step_indices
            input_currents_array[sample] = generate_step_current(hyperparams, 
                                                                 nodes=hyperparams['lin_nodes'],
                                                                 node_currents=node_currents,
                                                                 method=method,
                                                                 step_indices=step_indices_array)
    
    # Add noise before mask    
    if hyperparams['noise_flag'] == True:
        print('Adding smooth noise to input')
        input_currents_array = smooth_noise_stim(input_currents_array, hyperparams, plot=False)
    # Apply mask to input currents array
    for i in np.arange(len(input_currents_array)):
        input_currents_array[i] =  torch.Tensor(stim_mask).unsqueeze(1) * input_currents_array[i] 
        
    return input_currents_array

def generate_current_voltage_stim(
                            hyperparams=None,
                            num_samples=100,
                            i_min=0,
                            i_max=20,
                            method='randomsample',
                            manual_dist=None,
                            ):
    # Initialize the currents - else condition is is "variablestep" current


    # Initialize hyperparams if None
    if hyperparams == None:
        hyperparams = set_hyperparams_stim()
        
    set_seed(hyperparams['seed_target'])
    
    # Initialize location for saved files
    
    Path(f'./results/{str(hyperparams["date"])}/').mkdir(parents=True, exist_ok=True)
    PATH = f'./results/{str(hyperparams["date"])}/model_{hyperparams["model_type"]}_comp_{hyperparams["lin_nodes"]}/seed_target_{hyperparams["seed_target"]}/'
    Path(PATH).mkdir(parents=True, exist_ok=True)
    
    settings = f'i_min_{str(i_min)}_i_max_{str(i_max)}_method_{method}'
    
    # Load model parameters
    params = load_default_params(hyperparams['model_type'])

    # Take adj_mat as input later
    adj_mat = Lin_mat(hyperparams['lin_nodes'])

    # Initialize currents
    input_currents = initialize_currents_stim(hyperparams=hyperparams, 
                                   num_samples=num_samples,
                                   i_min=i_min,
                                   i_max=i_max,
                                   method=method,
                                  )
#     print(hyperparams['noise_flag'])
#     if hyperparams['noise_flag'] == True:
#         print('Adding smooth noise to input')
#         input_currents = smooth_noise_stim(input_currents, hyperparams, plot=False)

    # Save dictionary of currents samples
    stim_loc = hyperparams['stim_locations']
    if stim_loc == None:
            stim_id = 'all'
    else:
        stim_id = '_'.join(map(str, stim_loc))
    
    current_file = f'{PATH}current_traces_{settings}_dist_{hyperparams["distribution"]}_{hyperparams["mu"]}_{hyperparams["sigma"]}_{hyperparams["seed"]}_{stim_id}'
    
    if hyperparams['noise_flag'] == True:
        noise_file_str = f'_noise_sigma_{hyperparams["input_noise_sigma"]}_noise_seed_{hyperparams["input_noise_seed"]}'
        current_file = current_file + noise_file_str

    torch.save(input_currents, current_file)

    
    # Initialize model
    model = initialize_model_stim(params,
                                  hyperparams,
                                  adj_mat, 
                                  input_current_array=input_currents,
                                  neurons=num_samples,
                                  manual_dist=hyperparams['manual_dist'],
                                )
        
    # Run all current settings in parallel
        
    model.input_current_array = input_currents.to(device)
    start = time.perf_counter()
    print('Starting Simulation...')
    tt, solution, _ = model.simulate()
    solution = [soln.permute(1,2,0,3) for soln in solution]
    end = time.perf_counter()
    print('Duration %.2f s' % (end-start))
    
    voltage_traces = solution[0].cpu()
    
    voltage_file = f'{PATH}voltage_traces_{settings}_dist_{hyperparams["distribution"]}_{hyperparams["mu"]}_{hyperparams["sigma"]}_{hyperparams["seed"]}_{stim_id}'
    
    if hyperparams['noise_flag'] == True:
        noise_file_str = f'_noise_sigma_{hyperparams["input_noise_sigma"]}_noise_seed_{hyperparams["input_noise_seed"]}'
        voltage_file = voltage_file + noise_file_str
    
    torch.save(voltage_traces, voltage_file)


# Make noise smoother with gaussian kernel
def smooth_noise_stim(curr, hyperparams=None, plot=False):
    if hyperparams == None:
        hyperparams = set_hyperparams_stim()
    # Add smoothed noise to currents
        
    # Add noise to current
    # Set seed
    seed = set_seed(hyperparams['input_noise_seed'])
    # If no noise, smooth only
    if hyperparams['input_noise_sigma'] != 0:
        noise = torch.normal(mean=hyperparams['input_noise_mu'], std=hyperparams['input_noise_sigma'], size=curr.shape)
        curr = curr + noise
    
    # Produce gaussian kernel
    steps = int(hyperparams['kernel_width']/hyperparams['sim_params']['dt'])
    center = 0
    width = np.sqrt(hyperparams['variance'])
    x_range = np.linspace(center - 3*width, center + 3*width, steps)
    gauss = scipy.stats.norm.pdf(x_range, center, width)

    # Normalize and prep kernel
    kernel = torch.Tensor(gauss)
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.repeat(1,1,1)
    
    # Smooth noisy current
    curr_smooth = torch.zeros_like(curr)
    for i in range(curr.shape[1]):
        c_trace_smooth = F.conv1d(curr[:,i,:].unsqueeze(1), kernel, padding=int(len(gauss)/2)).squeeze()
        #shrink the convolved trace to be the original size of the input current
        if c_trace_smooth.shape[-1] > curr_smooth.shape[-1]:
            while c_trace_smooth.shape[-1] > curr_smooth.shape[-1]:
#                 print(c_trace_smooth.shape[-1], curr_smooth.shape[-1])
                c_trace_smooth = c_trace_smooth[:,:-1]
        
        curr_smooth[:,i,:] = c_trace_smooth
            
    if plot:
        fig, ax = plt.subplots(3,4, figsize=(12, 6), facecolor='white')
        fig.tight_layout(h_pad=1, w_pad=1)
        for i, j in enumerate(np.arange(curr.shape[0])[10:14]):
            ax[0,i].plot(x_range, gauss)
            ax[1,i].plot(curr[j,0])
            ax[2,i].plot(curr_smooth[j,0])
            ax[0,i].set_ylabel('gaussian kernel')
            ax[1,i].set_ylabel('no noise')
            ax[2,i].set_ylabel(f'noise')
        plt.show()
    
    return(curr_smooth)


# Noisy current
def noisy_current_range(sim_params, mus, sigma=10):
    # Make noisy current
    
    T = sim_params['T']
    dt = sim_params['dt']
    curr = torch.normal(mus[0], sigma, size=(1, int(T/dt)))
#     print(curr.shape)
    for i, mu in enumerate(mus[1:]):
        curr = torch.cat((curr, torch.normal(mu, sigma, size=(1, int(T/dt)))), 0)  
#     print(curr.shape)
    return(curr)