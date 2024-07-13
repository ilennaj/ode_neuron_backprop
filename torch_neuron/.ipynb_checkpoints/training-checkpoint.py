import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from torch import optim
import pandas as pd
from torchdiffeq import odeint_adjoint as odeint
import time
from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping
from mpl_toolkits import axes_grid1

import ipdb
from tqdm import tqdm

from torch_neuron.architectures import *

import random
from pathlib import Path


def train_neurons_stim(model, 
                     train_loader,
                     hyperparams_target,
                     hyperparams_model,
                     epochs=1, 
                     objective='mse', 
                     lr=0.01,  
                     patience=20, 
                     epoch_thresh=100,
                     save_data=True,
                     delta=0.01,
                     verbose=False,
                      ):
    '''
    Trains ODE neuron model, multicompartment with tensorboard functionality
    Inputs: model, trainloader, hyperparams_target, hyperparams_model, epochs,
            objective, lr, patience, epoch_thresh, save_data, delta, verbose
    Outputs: model
    '''
    # Start timing duration of training
    start, end = time.perf_counter(), time.perf_counter()
    
    
    # Initialize loss function and optimizer
    if objective == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # to track training loss and accuracy as model trains
    loss_curve = []
    g_Na_curve = []
    g_K_curve = []

    # Get initial parametres
    initial_params = model.g_Na.clone(), model.g_K.clone()

    #Interpret stimulation and recording locations for tensorboard
    stim_loc = hyperparams_target['stim_locations']
    if stim_loc == None:
            stim_id = 'all'
    else:
        stim_id = '_'.join(map(str, stim_loc))
        
    rec_loc = hyperparams_model['rec_locations']
    if rec_loc == None:
            rec_id = 'all'
    else:
        rec_id = '_'.join(map(str, rec_loc))
    
    # initialize writer for Tensorboard
    tb = SummaryWriter(log_dir='runs/'+time.strftime("%Y%m%d-%H%M%S")+'_stim_'+stim_id+'_rec_'+rec_id+'_'+str(hyperparams_model['seed']))
    
    # Initialize early stopping object
    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=verbose)
    
    # Get recording mask
    rec_mask = get_rec_mask(hyperparams_model).cuda()
    
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        ######################    
        # train the model    #
        ######################
        running_loss = 0.0

        model.train()
        
        for trial, data in enumerate(train_loader):
            curr = data['curr'].cuda()
            volt = data['volt'].cuda()


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            model.input_current_array = curr

            tt, solution, _ = model.simulate()
            print('Forward complete')
            intermed = time.perf_counter()
            print('Duration: %.2f' % (intermed-end))

            # Record from all compartments
            predicted_trace = solution[0].permute(1,2,0,3)
            
            
            # Determine Losses - mask non-recorded sites
            loss = criterion(predicted_trace*rec_mask, volt*rec_mask)
            
            # Backward step
            loss.backward()
            print('Backward complete')
            end = time.perf_counter()
            print('Duration: %.2f' % (end-intermed))

            # Optimize
            optimizer.step()

            end = time.perf_counter()
            print('Epoch: %d | Trial: %d | Duration: %.2f' % (epoch, trial, end-start))

            # If nans found in loss, stop training
            if torch.sum(torch.isnan(loss)) > 0 or torch.sum(torch.isnan(predicted_trace)) > 0:
                print('loss is nan, now testing')
                break

            running_loss += loss.item()

        # Save loss and parameter trajectories
        loss_curve.append(running_loss)
        g_Na_curve.append(model.g_Na.detach().cpu().numpy().squeeze())
        g_K_curve.append(model.g_K.detach().cpu().numpy().squeeze())

        # Save values in tensorboard
        tb.add_scalar('Total Loss', running_loss, epoch)
        
        tb.add_histogram('g_Na_dist', model.g_Na, global_step=epoch, bins='tensorflow')
        tb.add_histogram('g_K_dist', model.g_K, global_step=epoch, bins='tensorflow')

        # Visualize output in tensorboard
        if epoch % 5 == 0: #Every 5 epochs
           # Record figure comparing voltage traces
            print('ADDING FIGURE! Epoch %d'% epoch)
            tb.add_figure('Prediction vs Actual', plot_pred_vs_target_stim(curr, tt, predicted_trace, volt), global_step=epoch)
            if model.distribution == 'manual':
                tb.add_figure('dist_comparison', plot_dist_comparison_manual(initial_params, model, model.manual_dist, hyperparams_target), global_step=epoch)
            elif model.distribution == 'linear':
                tb.add_figure('dist_comparison', plot_dist_comparison_linear(initial_params, model, hyperparams_target), global_step=epoch)

        # Begin testing for early stopping
        if epoch > epoch_thresh:
            early_stopping(loss, model)
            # early_stopping.save_checkpoint(loss, model)

        # Stop training when early stop conditions are met
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Save all data
        if save_data == True:
            
            PATH = f'./results/{hyperparams_model["date"]}/'
            PATH_target = f'targ_nodes_{hyperparams_target["lin_nodes"]}_dist_{hyperparams_target["distribution"]}_mu_{hyperparams_target["mu"]}_sigma_{hyperparams_target["sigma"]}/'
            PATH_seed = f'seed_{hyperparams_model["seed"]}/'
            PATH_lr = f'lr_{lr}/'
            PATH_complete = PATH+PATH_target+PATH_seed+PATH_lr
            Path(PATH_complete).mkdir(parents=True, exist_ok=True)
                    
            settings = f'i_min_{hyperparams_model["i_min"]}_i_max_{hyperparams_model["i_max"]}_method_{hyperparams_model["method"]}'
            
            stim_loc = hyperparams_target['stim_locations']
            if stim_loc == None:
                    stim_id = 'all'
            else:
                stim_id = '_'.join(map(str, stim_loc))
                
            rec_loc = hyperparams_model['rec_locations']
            if rec_loc == None:
                    rec_id = 'all'
            else:
                rec_id = '_'.join(map(str, rec_loc))
            
            
            filename_id = f'ax_{hyperparams_model["ax_weight"]}_{settings}_dist_{hyperparams_model["distribution"]}_{hyperparams_model["mu"]}_{hyperparams_model["sigma"]}_stim_{stim_id}_rec_{rec_id}'
            
            if hyperparams_target['noise_flag'] == True:
                noise_file_str = f'_noise_sigma_{hyperparams_target["input_noise_sigma"]}_noise_seed_{hyperparams_target["input_noise_seed"]}'
                filename_id = filename_id + noise_file_str
            
            loss_file = f'loss_{filename_id}'
            g_Na_file = f'g_Na_{filename_id}'
            g_K_file = f'g_K_{filename_id}'
            
            torch.save(torch.Tensor(loss_curve), PATH_complete+loss_file)
            torch.save(torch.Tensor(np.array(g_Na_curve)), PATH_complete+g_Na_file)
            torch.save(torch.Tensor(np.array(g_K_curve)), PATH_complete+g_K_file)
    print('Finished Training, %d epochs' % (epoch+1))

    # Save metadata for tensorboard
    tb.add_hparams({'lr': lr, 
                    'batch_size': model.neurons, 
                    'duration': model.T,
                    'training_distribution': model.distribution,
                    'mu': model.mu,
                    'sigma': model.sigma,
                    'mu_t': hyperparams_target['mu'],
                    'sigma_t': hyperparams_target['sigma'],
                    'seed': hyperparams_model['seed'],
                    'date': hyperparams_model['date'],
                   }, 
                    {'Final Loss' : loss.item(),
                     'final epoch': epoch,
                     'Avg Epoch Duration': (end-start)/(epoch+1)
                    },
                    run_name=time.strftime("%Y%m%d-%H%M%S"),
                  )

    if model.distribution == 'manual':
        tb.add_figure('dist_comparison', plot_dist_comparison_manual(initial_params, model, model.manual_dist, hyperparams_target), global_step=epoch)
    elif model.distribution == 'linear':
        tb.add_figure('dist_comparison', plot_dist_comparison_linear(initial_params, model, hyperparams_target), global_step=epoch)
    

    
    tb.flush()
    tb.close()
    
    return(model)

def plot_dist_comparison(init_params, model):
    # Tensorboard visualization function - visualize parameters over training
    nodes_range = torch.arange(model.nodes)
    fig, ax = plt.subplots(1,2, figsize=(6,4), facecolor='white')
    fig.tight_layout(w_pad=2)
    ax[0].scatter(nodes_range, init_params[0].detach().cpu().numpy(), marker='.')
    ax[0].scatter(nodes_range, model.g_Na.detach().cpu().numpy(), marker='x')
    ax[1].scatter(nodes_range, init_params[1].detach().cpu().numpy(), marker='.')
    ax[1].scatter(nodes_range, model.g_K.detach().cpu().numpy(), marker='x')
    ax[0].set_xlabel('Compartment Number')
    ax[1].set_xlabel('Compartment Number')
    ax[0].set_ylabel('Parameter Value')
    ax[0].set_title('g_Na')
    ax[1].set_title('g_K')
    mu_gt = model.hyperparams['mu']
    sigma_gt = model.hyperparams['sigma']
    ax[0].plot(torch.linspace(mu_gt-sigma_gt, mu_gt+sigma_gt, model.nodes))
    ax[1].plot(torch.linspace(mu_gt-sigma_gt, mu_gt+sigma_gt, model.nodes))

    ax[1].legend(['gt','init', 'final'])
    

#     plt.close(fig)
    return(fig)

def plot_dist_comparison_linear(init_params, model, hyperparams_target):
    # Tensorboard visualization function - visualize parameters over training
    nodes_range = torch.arange(model.nodes)
    fig, ax = plt.subplots(1,2, figsize=(6,4), facecolor='white')
    fig.tight_layout(w_pad=2)
    ax[0].scatter(nodes_range, init_params[0].detach().cpu().numpy(), marker='.')
    ax[0].scatter(nodes_range, model.g_Na.detach().cpu().numpy(), marker='x')
    ax[1].scatter(nodes_range, init_params[1].detach().cpu().numpy(), marker='.')
    ax[1].scatter(nodes_range, model.g_K.detach().cpu().numpy(), marker='x')
    ax[0].set_xlabel('Compartment Number')
    ax[1].set_xlabel('Compartment Number')
    ax[0].set_ylabel('Parameter Value')
    ax[0].set_title('g_Na')
    ax[1].set_title('g_K')
    if hyperparams_target['distribution'] == 'linear':
        mu_gt = hyperparams_target['mu']
        sigma_gt = hyperparams_target['sigma']
        ax[0].plot(torch.linspace(mu_gt-sigma_gt, mu_gt+sigma_gt, model.nodes)*100) # g_Na
        ax[1].plot(torch.linspace(mu_gt-sigma_gt, mu_gt+sigma_gt, model.nodes)*45) # g_K
    elif hyperparams_target['distribution'] == 'manual':
        manual_dist = hyperparams_target['manual_dist']
        ax[0].scatter(nodes_range, manual_dist[0]*100, marker='o')
        ax[1].scatter(nodes_range, manual_dist[1]*45, marker='o')
            
    ax[1].legend(['gt','init', 'final'])
    
    ax[0].set_ylim(0, 200)
    ax[1].set_ylim(0, 200)

#     plt.close(fig)
    return(fig)

def plot_dist_comparison_manual(init_params, model, manual_dist):
    # Tensorboard visualization function - visualize parameters over training
    nodes_range = torch.arange(model.nodes)
    fig, ax = plt.subplots(1,2, figsize=(6,4), facecolor='white')
    fig.tight_layout(w_pad=2)
    ax[0].scatter(nodes_range, init_params[0].detach().cpu().numpy(), marker='.')
    ax[0].scatter(nodes_range, model.g_Na.detach().cpu().numpy(), marker='x')
    ax[1].scatter(nodes_range, init_params[1].detach().cpu().numpy(), marker='.')
    ax[1].scatter(nodes_range, model.g_K.detach().cpu().numpy(), marker='x')
    ax[0].set_xlabel('Compartment Number')
    ax[1].set_xlabel('Compartment Number')
    ax[0].set_ylabel('Parameter Value')
    ax[0].set_title('g_Na')
    ax[1].set_title('g_K')
    mu_gt = hyperparams_target['mu']
    sigma_gt = hyperparams_target['sigma']
    ax[0].plot(manual_dist*200)
    ax[1].plot(manual_dist*200)

    ax[1].legend(['gt','init', 'final'])
    

#     plt.close(fig)
    return(fig)

def set_seed(seed):
    # Set seed
    torch.manual_seed(seed)  # Set seed for PyTorch
    np.random.seed(seed)  # Set seed for NumPy
    random.seed(seed)  # Set seed for Python's random module
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def plot_pred_vs_target_stim(input_current, ttt, predicted_trace, target_trace):
    # Tensorboard visualization function - visualize voltage outputs
    fig, ax = plt.subplots(1,4, figsize=(12,4), facecolor='white')
    ax = ax.ravel()
    fig.tight_layout()
    for i in range(4):
        ax[i].set_title('Current: '+str([int(c.item()) for c in torch.max(input_current[i], dim=1)[0]]))
        for j in range(len(predicted_trace[0])):
            ax[i].plot(ttt.cpu().detach().numpy(), predicted_trace[i,j].cpu().detach().numpy())
        for j in range(len(predicted_trace[0])):
            ax[i].plot(ttt.cpu().detach().numpy(), target_trace[i,j].cpu().detach().numpy(), 
                       color='black', alpha=0.4, linestyle='dashed')
            
    node_labels = [f'node: {n}' for n in range(len(predicted_trace[0]))]
    ax[0].legend(node_labels+['target'])
    fig.show()
    
    return(fig)

def get_rec_mask(hyperparams):
    # Mask for limiting recording sites when calculating loss
    if hyperparams['rec_locations'] == None:
        rec_locations = np.arange(hyperparams['lin_nodes'])
    else:
        rec_locations = hyperparams['rec_locations']
    
    # Initialize Stimulation Mask
    rec_mask = torch.zeros((hyperparams['lin_nodes']))
    
    if isinstance(rec_locations, list):
        rec_locations = np.array(rec_locations)
    # Only use locations within the neuron
    rec_locations = rec_locations[rec_locations < hyperparams['lin_nodes']]
    # Make stim_mask
    rec_mask[rec_locations] = 1
    
    rec_mask = torch.Tensor(rec_mask).reshape(1,-1,1,1)
    
    return(rec_mask)