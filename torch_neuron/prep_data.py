


import torch
import numpy as np
import pickle
from pathlib import Path
from torch_neuron.generate_data import *



def save_hyperparameters_stim(**kwargs):
    date = kwargs['date']
    seed = kwargs['seed']
    mu_range = kwargs['mu_range']
    sigma_range = kwargs['sigma_range']
    model_type = kwargs['model_type']
    lin_nodes = kwargs['lin_nodes']
    ax_weight = kwargs['ax_weight']
    distribution = kwargs['distribution']
    mu = kwargs['mu']
    sigma = kwargs['sigma']
    
    i_min = kwargs['i_min']
    i_max = kwargs['i_max']
    method = kwargs['method']
    
    
    Path(f'./results/{str(date)}/').mkdir(parents=True, exist_ok=True)
    PATH = f'./results/{str(date)}/model_{model_type}_comp_{lin_nodes}/'
    Path(PATH).mkdir(parents=True, exist_ok=True)
    
    settings = f'i_min_{str(i_min)}_i_max_{str(i_max)}_method_{method}'
    
    hyperparam_file = f'hyperparams_ax_{str(ax_weight)}_{settings}_dist_{distribution}_{mu}_{sigma}_{seed}'
    
    if Path(f'{PATH}{hyperparam_file}').is_file():
        with open(f'{PATH}{hyperparam_file}.pickle', 'rb') as handle:
            hyperparams = pickle.load(handle)
        return(hyperparams)
    else:
        hyperparams = kwargs
        with open(f'{PATH}{hyperparam_file}.pickle', 'wb') as handle:
            pickle.dump(hyperparams, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return(hyperparams)
    

def get_default_hyperparameters_stim():
    date = 10000000
    seed = 10000000
    model_type='hh'
    lin_nodes=6
    manual_inj_d=True
    inj_pos_d=[1,5]

    ax_weight=0.5

    mu_num=5
    sigma_num=1
    mu_range=(10,50)
    sigma_range=(0,0)

    kernel_width=1
    variance=1
    sim_params = load_default_sim_params()
    
    distribution='linear'
    mu=0.5
    sigma=0.0
    
    num_samples=100
    i_min=0
    i_max=20
    method='randomsample'
    
    hyperparams = save_hyperparameters_stim(
                                     date=date,
                                     seed=seed,
                                     sim_params=sim_params,
                                     model_type=model_type,
                                     lin_nodes=lin_nodes,
                                     manual_inj_d=manual_inj_d,
                                     inj_pos_d=inj_pos_d,
                                     ax_weight=ax_weight,
                                     mu_num=mu_num,
                                     sigma_num=sigma_num, 
                                     mu_range=mu_range,
                                     sigma_range=sigma_range,
                                     kernel_width=kernel_width,
                                     variance=variance,
                                     distribution=distribution,
                                     mu=mu,
                                     sigma=sigma,
                                     num_samples=num_samples,
                                     i_min=i_min,
                                     i_max=i_max,
                                     method=method,
                                    )
    return(hyperparams)

def set_hyperparams_stim(date=10000000,
                         seed_target=10000000,
                         seed=10000000,
                         lin_nodes=6,
                         inj_pos_d=[1,5],
                         ax_weight=0.5,
                         sim_ms=5,
                         dt=0.1,
                         mu_num=20,
                         mu_range=(5, 25),
                         mu=0.5,
                         sigma=0.0,
                         distribution='linear',
                         num_samples=100,
                         i_min=0,
                         i_max=20,
                         method='randomsample',
                         hazard_rate=0.05,
                         manual_dist=None,
                         stim_locations=None,
                         rec_locations=None,
                         kernel_width=1,
                         variance=0.2,
                         noise_flag=False,
                         input_noise_mu=0.0,
                         input_noise_sigma=5.0,
                         input_noise_seed=10000000,
                   ):

    hyperparams = get_default_hyperparameters_stim()

    hyperparams['date'] = date
    hyperparams['seed_target'] = seed_target
    hyperparams['seed'] = seed
    hyperparams['lin_nodes'] = lin_nodes
    hyperparams['inj_pos_d'] = inj_pos_d
    hyperparams['ax_weight'] = ax_weight
    hyperparams['sim_params']['T'] = sim_ms
    hyperparams['sim_params']['stim_length'] = sim_ms
    hyperparams['sim_params']['Tstepinit'] = sim_ms
    hyperparams['sim_params']['Tstepfinal'] = sim_ms
    hyperparams['sim_params']['dt'] = dt
    hyperparams['mu_num'] = mu_num
    hyperparams['mu_range'] = mu_range

    hyperparams['mu'] = mu
    hyperparams['sigma'] = sigma
    hyperparams['distribution'] = distribution
    
    hyperparams['num_samples'] = num_samples
    hyperparams['i_min'] = i_min
    hyperparams['i_max'] = i_max
    hyperparams['method'] = method
    hyperparams['hazard_rate'] = hazard_rate
    hyperparams['manual_dist'] = manual_dist
    hyperparams['stim_locations'] = stim_locations
    hyperparams['rec_locations'] = rec_locations
    
    # Smoothing parameters
    hyperparams['kernel_width'] = kernel_width
    hyperparams['variance'] = variance
    
    # Noise parameters
    hyperparams['noise_flag'] = noise_flag
    hyperparams['input_noise_mu'] = input_noise_mu
    hyperparams['input_noise_sigma'] = input_noise_sigma
    hyperparams['input_noise_seed'] = input_noise_seed
    
    save_hyperparameters_stim(**hyperparams)
    
    return hyperparams

# Load data
def load_current_voltage_stim(hyperparams=None,):
    # load current and voltage traces

    if hyperparams == None:
        hyperparams = set_hyperparams_stim()
        
    stim_loc = hyperparams['stim_locations']
    if stim_loc == None:
            stim_id = 'all'
    else:
        stim_id = '_'.join(map(str, stim_loc))
        
    
    PATH = f'./results/{str(hyperparams["date"])}/model_{hyperparams["model_type"]}_comp_{hyperparams["lin_nodes"]}/seed_target_{hyperparams["seed_target"]}/'
    settings = f'i_min_{str(hyperparams["i_min"])}_i_max_{str(hyperparams["i_max"])}_method_{hyperparams["method"]}'
    file_voltage = f'{PATH}voltage_traces_{settings}_dist_{hyperparams["distribution"]}_{hyperparams["mu"]}_{hyperparams["sigma"]}_{hyperparams["seed"]}_{stim_id}'
    file_current = f'{PATH}current_traces_{settings}_dist_{hyperparams["distribution"]}_{hyperparams["mu"]}_{hyperparams["sigma"]}_{hyperparams["seed"]}_{stim_id}'
    
    if hyperparams['noise_flag'] == True:
        noise_file_str = f'_noise_sigma_{hyperparams["input_noise_sigma"]}_noise_seed_{hyperparams["input_noise_seed"]}'
        file_voltage = file_voltage + noise_file_str
        file_current = file_current + noise_file_str
    
    success = True
    try:
        voltage_traces = torch.load(file_voltage)
    except(FileNotFoundError):
        print('Voltage traces do not exist')
        voltage_traces = None
        success = False
    try:
        current_traces = torch.load(file_current)
    except(FileNotFoundError):
        print('Current traces do not exist')
        success = False
        current_traces = None
    return(current_traces, voltage_traces, success)

def load_data_stim(hyperparams=None, manual_dist=None):
    #Generate current and voltage traces and then load them
    
    if hyperparams == None:
        hyperparams = set_hyperparams_stim()
    
    num_samples = hyperparams['num_samples']
    i_min = hyperparams['i_min']
    i_max = hyperparams['i_max']
    method = hyperparams['method']
    
        
    generate_current_voltage_stim(hyperparams=hyperparams,
                                  num_samples=num_samples,
                                  i_min=i_min,
                                  i_max=i_max,
                                  method=method,
                                  manual_dist=manual_dist,
                            )
    

    curr_traces, volt_traces, success = load_current_voltage_stim(hyperparams)
    return(curr_traces, volt_traces, success)
            
#     if not success == True:
#         print('Failed load data')
#         return(curr_traces, volt_traces, success)
#     else:
#         return(curr_traces, volt_traces, success)
    
class NeuronTraceDataset_stim(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, 
                 hyperparams=None, 
                 manual_dist=None, 
                 transform=None,
                 seed=None,
                ):
        """
        Args:
            hyperparams (dict): hyperparameters
            manual_dist (str): setting for uniform random parameters
            transform : function to transform the sample output
            seed : for replicability of target data
            
        """
        if hyperparams==None:
            hyperparams = set_hyperparams_stim()
        
        if seed==None:
            seed = hyperparams['seed_target']
        
        curr, volt, _ = load_data_stim(hyperparams, manual_dist)
        
        self.hyperparams = hyperparams
        self.curr = curr
        self.volt = volt
        

    def __len__(self):
        return len(self.curr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        curr_ix = self.curr[idx]
        volt_ix = self.volt[idx]
        
        sample = {'curr': curr_ix, 'volt': volt_ix}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
def init_random_model_params(hyperparams=None,
                             center=1,
                             perturb=0.3,
                             manual_seed=None,
                             lin_nodes=None,
                        ):
    # To set a uniform random sample of parameters
    
    if hyperparams == None:
        hyperparams = set_hyperparams_stim()
        
    if manual_seed == None:
        manual_seed = hyperparams['seed_target']

        
    if lin_nodes == None:
        lin_nodes = hyperparams['lin_nodes']
    
    set_seed(manual_seed)
    
    alpha = center - perturb
    beta = center + perturb
    dist = torch.FloatTensor(2,hyperparams['lin_nodes']).uniform_(alpha, beta)
    return dist