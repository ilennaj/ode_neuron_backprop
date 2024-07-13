# Functions for visualizing data

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

import ipdb
from pathlib import Path
import os
import glob

from torch_neuron.generate_data import *
from torch_neuron.prep_data import *

def models_from_entry(data_lookup, entry_ix=0, date=10000000, method='variablestep', hazard_rate=0.05, hyperparams=None):

    # Initialize hyperparameters
    if hyperparams == None:
        hyperparams = set_hyperparams_stim()
    
    lr = 0.5 ############# Possible bug if changing around lr values #########
    params = load_default_params()
    batch_size=100
    
    # Current parameters #### Possible bug if changing kernel values
    kernel_width = 1
    variance = 0.2
    input_noise_mu = 0
    
    # Load parameter trajectories
    # loss_curve = torch.load(data_lookup.loc[entry_ix]['loss_filename_id'])
    g_Na_curve = torch.load(data_lookup.loc[entry_ix]['g_Na_filename_id'])
    g_K_curve = torch.load(data_lookup.loc[entry_ix]['g_K_filename_id'])

#     print(g_Na_curve[0,:]/100, g_Na_curve[-1,:]/100)
#     print(g_K_curve[0,:]/45, g_K_curve[-1,:]/45)

    scaling = np.array([[100],[45]])

    g_init = (torch.cat((g_Na_curve[0,:].reshape(1,-1),g_K_curve[0,:].reshape(1,-1)),dim=0)/scaling).float()
    g_final = (torch.cat((g_Na_curve[-1,:].reshape(1,-1),g_K_curve[-1,:].reshape(1,-1)),dim=0)/scaling).float()

#     print(g_init)
#     print(g_final)

    center = data_lookup.loc[entry_ix]['mu']
    perturb = data_lookup.loc[entry_ix]['sigma']
    lin_nodes = data_lookup.loc[entry_ix]['lin_nodes']
    seed = data_lookup.loc[entry_ix]['seed']

    stim_locations = data_lookup.loc[entry_ix]['stim_locations']
    rec_locations = data_lookup.loc[entry_ix]['rec_locations']
    
    noise_sigma = data_lookup.loc[entry_ix]['noise_sigma']
    noise_flag = data_lookup.loc[entry_ix]['flag']
    
    # VARIABLE TARGET CONDITIONS
    mu_target = center
    sigma_target = perturb
    lin_nodes_target = lin_nodes
#     distribution_target = 'manual'


    # STABLE MODEL INITIALIZATION
    lin_nodes_model = lin_nodes
#     distribution_model = 'linear'
    mu_model = 1.0
    sigma_model = 0.0
    
    distribution_target = 'manual'
    distribution_model = 'manual'

    constant = True
    override = True
    

    
    hyperparams_target = hyperparams
    hyperparams_model = set_hyperparams_stim(date=date,
                                             seed=seed,
                                             lin_nodes=lin_nodes_model,
                                             distribution=distribution_model,
                                             mu=mu_model,
                                             sigma=sigma_model,
                                             method=method, 
                                             seed_target=seed,
                                             hazard_rate=hazard_rate,
                                             stim_locations=stim_locations,
                                             rec_locations=rec_locations,
                                             kernel_width=kernel_width,
                                             variance=variance,
                                             noise_flag=noise_flag,
                                             input_noise_mu=input_noise_mu,
                                             input_noise_sigma=noise_sigma,
                                             input_noise_seed=seed,
                                            )

    # Set Manual Distibution and adjust hyperparameters accordingly

    #### Set model hyperparams with init or final manual distribution of parameters
    manual_dist_model_init = g_init
    manual_dist_model_final = g_final
    manual_dist_target = init_random_model_params(hyperparams=hyperparams_target,
                                                  center=mu_target,
                                                  perturb=sigma_target,
                                                  manual_seed=seed
                                                  )

    # Make new hyperparams for different init and final models
    hyperparams_model_init = hyperparams_model.copy()
    hyperparams_model_final = hyperparams_model.copy()

    # Assign init and final distributions
    hyperparams_model_init['manual_dist'] = manual_dist_model_init
    hyperparams_model_final['manual_dist'] = manual_dist_model_final
    hyperparams_target['manual_dist'] = manual_dist_target

    # Initialize adjacency matrix
    adj_mat_model = Lin_mat(hyperparams_model_init['lin_nodes'])
    
    # Generate input currents, add noise if noiseflag == true
    input_currents = initialize_currents_stim_lim(
                                   data_lookup, 
                                   entry_ix,
                                   hyperparams=hyperparams_target, 
                                   num_samples=hyperparams_target['num_samples'],
                                   i_min=hyperparams_target['i_min'],
                                   i_max=hyperparams_target['i_max'],
                                   method=hyperparams_target['method'],
                                  )

    #######SAVE CURRENTS##########
    
    PATH = f'./results/{date}/'
    PATH_target = f'targ_nodes_{lin_nodes_target}_dist_{distribution_target}_mu_{mu_target}_sigma_{sigma_target}/'
    PATH_seed = f'seed_{seed}/'
    PATH_lr = f'lr_{lr}/'
    PATH_complete = PATH+PATH_target+PATH_seed+PATH_lr

    stim_loc = hyperparams_model['stim_locations']
    if stim_loc == None:
            stim_id = 'all'
    else:
        stim_id = '_'.join(map(str, stim_loc))

    rec_loc = hyperparams_model['rec_locations']
    if rec_loc == None:
            rec_id = 'all'
    else:
        rec_id = '_'.join(map(str, rec_loc))


    file_name_1 = f'nodes_{hyperparams_model["lin_nodes"]}_dist_{hyperparams_model["distribution"]}_'
    file_name_2 = f'mu_{hyperparams_model["mu"]}_sigma_{hyperparams_model["sigma"]}_'
    file_name_3 = f'stim_{stim_id}_rec_{rec_id}'
    file_name_4 = f'_noise_{noise_sigma}_flag_{noise_flag}.pth'
    file_name_complete = file_name_1 + file_name_2 + file_name_3 + file_name_4


    Path(PATH).mkdir(parents=True, exist_ok=True)
    Path(PATH_complete).mkdir(parents=True, exist_ok=True)

    torch.save(input_currents, f'{PATH_complete}current_trace_{file_name_complete}')
    
    #### INITIALIZE MODELS ####
    model_init = Layer_2_3_stim(params,
                              hyperparams_model_init,
                              adj_mat_model, 
                              manual_dist=manual_dist_model_init,
                              neurons=batch_size,
                              input_current_array=input_currents,
                              method='dopri5',
                              ax_weight=hyperparams_model_init['ax_weight'],
                              constant=constant,
                              override=override,
                              distribution=distribution_model,
                              mu = hyperparams_model_init['mu'],
                              sigma = hyperparams_model_init['sigma'],
                              input_type='manual',
                              ).to(device)

    model_final = Layer_2_3_stim(params,
                              hyperparams_model_final,
                              adj_mat_model, 
                              manual_dist=manual_dist_model_final,
                              neurons=batch_size,
                              input_current_array=input_currents,
                              method='dopri5',
                              ax_weight=hyperparams_model_final['ax_weight'],
                              constant=constant,
                              override=override,
                              distribution=distribution_model,
                              mu = hyperparams_model_final['mu'],
                              sigma = hyperparams_model_final['sigma'],
                              input_type='manual',
                              ).to(device)

    model_target = Layer_2_3_stim(params,
                              hyperparams_target,
                              adj_mat_model, 
                              manual_dist=manual_dist_target,
                              neurons=batch_size,
                              input_current_array=input_currents,
                              method='dopri5',
                              ax_weight=hyperparams_target['ax_weight'],
                              constant=constant,
                              override=override,
                              distribution=distribution_target,
                              mu = hyperparams_target['mu'],
                              sigma = hyperparams_target['sigma'],
                              input_type='manual',
                              ).to(device)
    return(model_init, model_final, model_target)


def simulate_and_save_voltage(model, data_lookup, entry_ix, label=None, hyperparams_target=None):
    
    lr = 0.5 
    
    if hyperparams_target==None:
        hyperparams_target = set_hyperparams_stim()
        
    date = hyperparams_target['date']
        
    lin_nodes_target = hyperparams_target['lin_nodes']
    distribution_target = hyperparams_target['distribution']
    mu_target = hyperparams_target['mu']
    sigma_target = hyperparams_target['sigma']
    
    stim_loc = data_lookup.loc[entry_ix]['stim_locations']
    rec_loc = data_lookup.loc[entry_ix]['rec_locations']
    
    noise_sigma = data_lookup.loc[entry_ix]['noise_sigma']
    noise_flag = data_lookup.loc[entry_ix]['flag']

    
    if label == None:
        label = 'unlabeled'
    start = time.perf_counter()
    print('Starting Simulation...')
    tt, solution, _ = model.simulate()
    solution = [soln.permute(1,2,0,3) for soln in solution]
    end = time.perf_counter()
    print('Duration %.2f s' % (end-start))

    voltage_traces = solution[0].cpu()

    PATH = f'./results/{date}/'
    PATH_target = f'targ_nodes_{lin_nodes_target}_dist_{distribution_target}_mu_{mu_target}_sigma_{sigma_target}/'
    PATH_seed = f'seed_{model.hyperparams["seed"]}/'
    PATH_lr = f'lr_{lr}/'
    PATH_complete = PATH+PATH_target+PATH_seed+PATH_lr

    if stim_loc == None:
            stim_id = 'all'
    else:
        stim_id = '_'.join(map(str, stim_loc))

    if rec_loc == None:
            rec_id = 'all'
    else:
        rec_id = '_'.join(map(str, rec_loc))


    file_name_1 = f'{label}_nodes_{model.hyperparams["lin_nodes"]}_dist_{model.hyperparams["distribution"]}_'
    file_name_2 = f'mu_{model.hyperparams["mu"]}_sigma_{model.hyperparams["sigma"]}_'
    file_name_3 = f'stim_{stim_id}_rec_{rec_id}'
    file_name_4 = f'_noise_{noise_sigma}_flag_{noise_flag}.pth'
    file_name_complete = file_name_1 + file_name_2 + file_name_3 + file_name_4

    Path(PATH).mkdir(parents=True, exist_ok=True)
    Path(PATH_complete).mkdir(parents=True, exist_ok=True)
    torch.save(voltage_traces, f'{PATH_complete}voltage_trace_{file_name_complete}')
    
def generate_all_voltage_traces(data_lookup, entry_ix=0, method='variablestep', hazard_rate=0.05, hyperparams=None):
    
    if hyperparams==None:
        hyperparams = set_hyperparams_stim()
        
    date = hyperparams['date']
        
    # Produce models
    model_init, model_final, model_target = models_from_entry(data_lookup, entry_ix=entry_ix, date=date, method=method, hazard_rate=hazard_rate, hyperparams=hyperparams)
    
    
    simulate_and_save_voltage(model_init, data_lookup, entry_ix, label='init', hyperparams_target=hyperparams)
    simulate_and_save_voltage(model_final, data_lookup, entry_ix, label='final', hyperparams_target=hyperparams)
    simulate_and_save_voltage(model_target, data_lookup, entry_ix, label='target', hyperparams_target=hyperparams)
    
def load_voltage_traces_by_seed(data_lookup, entry_ix, hyperparams_target=None, noise_sigma=0, noise_flag=False):
    
    
    if hyperparams_target==None:
        hyperparams_target = set_hyperparams_stim()
        
    date = hyperparams_target['date']
    lin_nodes_target = hyperparams_target['lin_nodes']
    distribution_target = hyperparams_target['distribution']
    mu_target = hyperparams_target['mu']
    sigma_target = hyperparams_target['sigma']

    
    seed = data_lookup.loc[entry_ix]['seed']
    lr = 0.5 ############# Possible bug if changing around lr values #########
    
    stim_loc = data_lookup.loc[entry_ix]['stim_locations']
    rec_loc = data_lookup.loc[entry_ix]['rec_locations']

    PATH = f'./results/{date}/'
    PATH_target = f'targ_nodes_{lin_nodes_target}_dist_{distribution_target}_mu_{mu_target}_sigma_{sigma_target}/'
    PATH_seed = f'seed_{seed}/'
    PATH_lr = f'lr_{lr}/'
    PATH_complete = PATH+PATH_target+PATH_seed+PATH_lr
    
    # IDs for query
    
    if stim_loc == None:
            stim_id = 'all'
    else:
        stim_id = '_'.join(map(str, stim_loc))

    if rec_loc == None:
            rec_id = 'all'
    else:
        rec_id = '_'.join(map(str, rec_loc))

    # Directory path to search
    directory_path = PATH_complete

    # Search query (part of the file name you want to match)
    search_query = 'voltage_trace'
    stim_rec =  f'stim_{stim_id}_rec_{rec_id}'
    noise = f'_noise_{noise_sigma}_flag_{noise_flag}'

    # Construct the search pattern using the query
    search_pattern = os.path.join(directory_path, '*' + search_query + '*' + stim_rec + noise + '*')

    # Use glob to find files matching the pattern
    matching_files = glob.glob(search_pattern)

    # Print the matching file names
    v_traces = []
    print("Matching Files:")
    for file_path in matching_files:
        print(file_path)
        v_traces.append([file_path, torch.load(file_path)])
    
    return(v_traces)

def load_current_traces_by_seed(data_lookup, entry_ix, hyperparams_target=None, noise_sigma=0, noise_flag=False):
    
    if hyperparams_target==None:
        hyperparams_target = set_hyperparams_stim()
    
    date = hyperparams_target['date']
    lin_nodes_target = hyperparams_target['lin_nodes']
    distribution_target = hyperparams_target['distribution']
    mu_target = hyperparams_target['mu']
    sigma_target = hyperparams_target['sigma']
    
    seed = data_lookup.loc[entry_ix]['seed']
    lr = 0.5 
    
    stim_loc = data_lookup.loc[entry_ix]['stim_locations']
    rec_loc = data_lookup.loc[entry_ix]['rec_locations']

    PATH = f'./results/{date}/'
    PATH_target = f'targ_nodes_{lin_nodes_target}_dist_{distribution_target}_mu_{mu_target}_sigma_{sigma_target}/'
    PATH_seed = f'seed_{seed}/'
    PATH_lr = f'lr_{lr}/'
    PATH_complete = PATH+PATH_target+PATH_seed+PATH_lr

    # Directory path to search
    directory_path = PATH_complete
    
    # IDs for query
    
    if stim_loc == None:
            stim_id = 'all'
    else:
        stim_id = '_'.join(map(str, stim_loc))

    if rec_loc == None:
            rec_id = 'all'
    else:
        rec_id = '_'.join(map(str, rec_loc))

    # Search query (part of the file name you want to match)
    search_query = 'current_trace'
    stim_rec =  f'stim_{stim_id}_rec_{rec_id}'
    noise = f'_noise_{noise_sigma}_flag_{noise_flag}'

    # Construct the search pattern using the query
    search_pattern = os.path.join(directory_path, '*' + search_query + '*' + stim_rec + noise + '*')
    # Use glob to find files matching the pattern
    matching_files = glob.glob(search_pattern)

    # Print the matching file names
    c_traces = []
    print("Matching Files:")
    for file_path in matching_files:
        print(file_path)
        c_traces.append([file_path, torch.load(file_path)])

    return(c_traces)


def initialize_currents_stim_lim(data_lookup, 
                                 entry_ix,
                                 hyperparams=None, 
                                 num_samples=100, 
                                 i_min=0, 
                                 i_max=20, 
                                 method='randomsample',
                            ):
    
    # Initialize hyperparameters
    if hyperparams == None:
        hyperparams = set_hyperparams_stim()
        
    hazard_rate = hyperparams['hazard_rate']

    stim_locations = data_lookup.loc[entry_ix]['stim_locations']
    noise_flag = data_lookup.loc[entry_ix]['flag']
    noise_sigma = data_lookup.loc[entry_ix]['noise_sigma']
    
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

        # Number Indices when the curent changes
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
    if noise_flag == True:
        print('Adding smooth noise to input')
        input_currents_array = smooth_noise_stim_lim(input_currents_array, hyperparams, noise_sigma, plot=False)
    # Apply mask to input currents array
    for i in np.arange(len(input_currents_array)):
        input_currents_array[i] =  torch.Tensor(stim_mask).unsqueeze(1) * input_currents_array[i] 
        
    return input_currents_array

def smooth_noise_stim_lim(curr, hyperparams=None, noise_sigma=0, plot=False):
    if hyperparams == None:
        hyperparams = set_hyperparams_stim()
        
    # Add noise to current
    # Set seed
    seed = set_seed(hyperparams['input_noise_seed'])
    # If no noise, smooth only
    if noise_sigma != 0:
        noise = torch.normal(mean=hyperparams['input_noise_mu'], std=noise_sigma, size=curr.shape)
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
