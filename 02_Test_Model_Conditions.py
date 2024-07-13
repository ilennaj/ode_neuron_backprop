import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import optim

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import datetime
import pandas as pd
import scipy.signal
import scipy.stats
import csv
import pickle
import networkx as nx
import pandas as pd

from scipy.optimize import curve_fit
from pathlib import Path

from torch_neuron.training import *
from torch_neuron.architectures import *
from torch_neuron.morphology import *
from torch_neuron.dataset import *
from torch_neuron.model_drion import *
from torch_neuron.generate_data import *
from torch_neuron.prep_data import *

from torchdiffeq import odeint_adjoint as odeint
# from torchdiffeq import odeint

from torch.utils.tensorboard import SummaryWriter
import PIL
from pytorchtools import EarlyStopping
from tqdm import tqdm
import random

import contextlib
import io
import sys

SEEDS = [20230619, 20230620, 20230621, 20230622, 20230623,
         20230624, 20230625, 20230626, 20230627, 20230628]

adj_matrix = np.load('./results/refs/gidon_morph.npy')

seed = SEEDS[0]
# DATE = 20231016
# DATE = 20231031
# DATE = 20240430
DATE = 20240624


center = 1.0
perturb = 0.3
lin_nodes_target = len(adj_matrix)
distribution_target = 'manual'

# Smoothing settings
# input_noise_sigmas_flags = [[0, False], [0, True], [5, True], [10, True], [15, True]]
input_noise_sigmas_flags = [[0, False]]


input_noise_mu = 0
noise_flags = True
input_noise_seed = SEEDS[0]

# set_seed(seed)

# Smoothing settings
kernel_width=1
variance=0.2

# input_dist_list = [9,12,15,18]
input_dist_list = [18]

stim_locations_list = []
rec_locations_list = []
for input_dist in input_dist_list:
    stim_locations_list.append(np.arange(0,len(adj_matrix) - 1, input_dist).tolist())
    rec_locations_list.append(None)
# stim_locations_list = [np.arange(0,len(adj_matrix) - 1, input_dist).tolist(), for input_dist in input_dist_list]
# rec_locations_list = list(np.arange(0, len(adj_matrix)))
# rec_locations_list = np.choice(np.arange(0, len(adj_matrix)), int(len_adj_matrix)/input_dist)
# rec_locations_list = [None]

stim_rec_locations_list = []

for i, stim_loc in enumerate(stim_locations_list):
    for j, rec_loc in enumerate(rec_locations_list):
        stim_rec_locations_list.append([stim_loc, rec_loc])

stim_rec_locations_list = zip(stim_locations_list, rec_locations_list)
stim_rec_locations_list = [a for a in stim_rec_locations_list]

stim_rec_locations_list



conditions = {}
count = 0

for stim_rec_locations in stim_rec_locations_list:
        for sigmas_flags in input_noise_sigmas_flags:
            for seed in np.array(SEEDS)[np.array([0,1,3])]:
            
                conditions[count] = [
                                        stim_rec_locations[0],
                                        stim_rec_locations[1],
                                        sigmas_flags[0],
                                        sigmas_flags[1],
                                        center,
                                        perturb,
                                        lin_nodes_target,
                                        distribution_target,
                                        seed,
                                        
                                    ]

                count += 1
conditions = pd.DataFrame(conditions).transpose()
conditions.columns = ['stim_locations','rec_locations','noise_sigma','flag','mu','sigma','lin_nodes','distribution','seed']
conditions


# Initialize StringIO objects to store the output
captured_output = io.StringIO()
captured_error = io.StringIO()

with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_error):
    # Loop over seeds
    # for seed in SEEDS[:5]:
    # for seed in SEEDS[:1]:
    for condition_num, condition in conditions.iterrows():
        
        print('Condition: ', condition_num)
        stim_locations = condition['stim_locations']
        
        rec_locations = condition['rec_locations']
        input_noise_sigma = condition['sigma']
        noise_flag = condition['flag']
        seed = condition['seed']
        
        # Initialize Run Settings
        date = DATE
        # seed = SEEDS[0]
        lr = 0.5
        batch_size = 100 
        epochs = 200
    
        patience = 300
        delta = 0.001
        epoch_thresh = 0

        i_max = 10

        ax_weight = 0.01

        

        

        method = 'variablestep'
        hazard_rate = 0.05

        override=True
        constant=False

        center = 1.0
        perturb = 0.3

        # Initialize target model to be homogeneous LMN

        # for condition_num, condition in conditions.iterrows():


        # VARIABLE TARGET CONDITIONS
        mu_target = center
        sigma_target = perturb
        lin_nodes_target = len(adj_matrix)
        distribution_target = 'manual'


        # STABLE MODEL INITIALIZATION
        lin_nodes_model = len(adj_matrix)
        distribution_model = 'linear'
        mu_model = 1.0
        sigma_model = 0.0


        #     print(f'Condition {condition_num}: mu={condition["mu"]} | sigma={condition["sigma"]} | {lin_nodes_target} | {distribution_target}')

        # Load params

        params = load_default_params()



        ## Change hyperparameter settings
        hyperparams_target = set_hyperparams_stim(date=date,
                                                    seed=seed,
                                                    lin_nodes=lin_nodes_target,
                                                    distribution=distribution_target,
                                                    mu=mu_target,
                                                    sigma=sigma_target,
                                                    method=method,
                                                    seed_target=seed,
                                                    hazard_rate=hazard_rate,
                                                    kernel_width=kernel_width,
                                                    variance=variance,
                                                    noise_flag=noise_flag,
                                                    input_noise_mu=input_noise_mu,
                                                    input_noise_sigma=input_noise_sigma,
                                                    input_noise_seed=input_noise_seed,
                                                    stim_locations=stim_locations,
                                                    rec_locations=rec_locations,
                                                    ax_weight=ax_weight,
                                                    i_min=0,
                                                    i_max=i_max,
                                                    )
        hyperparams_model = set_hyperparams_stim(date=date,
                                                seed=seed,
                                                lin_nodes=lin_nodes_model,
                                                distribution=distribution_model,
                                                mu=mu_model,
                                                sigma=sigma_model,
                                                method=method, 
                                                seed_target=seed,
                                                hazard_rate=hazard_rate,
                                                kernel_width=kernel_width,
                                                variance=variance,
                                                noise_flag=noise_flag,
                                                input_noise_mu=input_noise_mu,
                                                input_noise_sigma=input_noise_sigma,
                                                input_noise_seed=input_noise_seed,
                                                stim_locations=stim_locations,
                                                rec_locations=rec_locations,
                                                ax_weight=ax_weight,
                                                i_min=0,
                                                i_max=i_max,
                                                )

        # Set Manual Distibution and adjust hyperparameters accordingly

        manual_dist_model = None
        manual_dist_target = init_random_model_params(hyperparams=hyperparams_target,
                                                    center=mu_target,
                                                    perturb=sigma_target,
                                                    manual_seed=seed
                                                    )
        hyperparams_target['manual_dist'] = manual_dist_target

        # Initialize adjacency matrix
        # adj_mat_model = Lin_mat(hyperparams_model['lin_nodes'])
        adj_mat_model = torch.tensor(adj_matrix, dtype=torch.float32)


        # Initialize dataloader using target hyperparams and conditions
        dataset = NeuronTraceDataset_stim(hyperparams=hyperparams_target,
                                        manual_dist=manual_dist_target,
                                        )
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=1)

        # Set seed to be the same for every condition
        set_seed(seed)

        # Initialize model
        model = Layer_2_3_stim(params,
                        hyperparams_model,
                        adj_mat_model, 
                        manual_dist=None,
                        neurons=batch_size,
                        input_current_array=None,
                        method='dopri5',
                        ax_weight=hyperparams_model['ax_weight'],
                        constant=constant,
                        override=override,
                        distribution=distribution_model,
                        mu = hyperparams_model['mu'],
                        sigma = hyperparams_model['sigma'],
                        input_type='manual',
                        ).to(device)

        ##### LOAD TRAINED MODEL

        # f = '/n/home12/ijones/projects/diffeq/results/20240423/targ_nodes_184_dist_manual_mu_1.0_sigma_0.3/seed_20230619/lr_0.5/model_nodes_184_dist_linear_mu_1.0_sigma_0.0_20230619.pth'
        # model_trained = torch.load(f)
        # model.load_state_dict(model_trained)
    

        #####



        # # Paralellize model if multiple GPUs are available
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     model = torch.nn.DataParallel(model)

        # # Assign model to device
        # model.to(device)

        # Start Training Loop
        try: 
            model = train_neurons_stim(model, 
                                        dataloader,
                                        hyperparams_target, # For generating ground truth parameter plot and file saving
                                        hyperparams_model,
                                        epochs=epochs, 
                                        objective='mse', 
                                        lr=lr, 
                                        patience=patience, 
                                        epoch_thresh=epoch_thresh,
                                        save_data=True,
                                        delta=delta,
                                        verbose=True,
                                        )
        except AssertionError:
            print(f"Assertion Error with condition #{condition_num}")

        except RuntimeError as e:
            # Write the error message to stderr
            print(str(e), file=sys.stderr)

        except Exception as e:
            # Write the error message to stderr
            print(str(e), file=sys.stderr)
        #     continue
        finally:
            try:
                PATH = f'./results/{date}/'
                PATH_target = f'targ_nodes_{lin_nodes_target}_dist_{distribution_target}_mu_{mu_target}_sigma_{sigma_target}/'
                PATH_seed = f'seed_{seed}/'
                PATH_lr = f'lr_{lr}/'
                file_name_model = f'model_nodes_{lin_nodes_model}_dist_{distribution_model}_mu_{mu_model}_sigma_{sigma_model}_{seed}'
                
                if hyperparams_target['noise_flag'] == True:
                    noise_file_str = f'_noise_sigma_{hyperparams_target["input_noise_sigma"]}_noise_seed_{hyperparams_target["input_noise_seed"]}.pth'
                    file_name_model = file_name_model + noise_file_str + '.pth'
                else:
                    file_name_model = file_name_model + '.pth'


                Path(PATH).mkdir(parents=True, exist_ok=True)
                Path(PATH+PATH_target+PATH_seed+PATH_lr).mkdir(parents=True, exist_ok=True)

                torch.save(model.state_dict(), PATH+PATH_target+PATH_seed+PATH_lr+file_name_model)
            except Exception as e:
                print(str(e), file=sys.stderr)

            # Get the captured output and error
            captured_output_str = captured_output.getvalue()
            captured_error_str = captured_error.getvalue()

            # Save the captured output to a file
            with open(PATH+PATH_target+PATH_seed+PATH_lr+'output'+str(condition_num)+'.log', 'w') as f:
                f.write(captured_output_str)

            # Save the captured error to a file
            with open(PATH+PATH_target+PATH_seed+PATH_lr+'error'+str(condition_num)+'.log', 'w') as f:
                f.write(captured_error_str)