import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
import pandas as pd
from torchdiffeq import odeint_adjoint as odeint

import pickle

import ipdb

from torch_neuron.morphology import *

class Layer_2_3_stim(nn.Module):
    '''
    Hodgkin Huxley single compartment model.
    '''
    def __init__(self, 
                 params,
                 hyperparams,
                 adj_mat, 
                 manual_dist=None,
                 neurons=1, 
                 input_current_array=None,
                 parallel_current_settings=True,
                 method='dopri5', 
                 ax_weight=0.5, 
                 constant=False, 
                 override=False,
                 atol=1e-8, 
                 rtol=1e-8, 
                 distribution='homogeneous', 
                 sigma=0.5, 
                 mu=-0.4, 
                 input_type='manual', 
                 device="cuda:0",
                 verbose=False):
        super(Layer_2_3_stim, self).__init__()

        self.params = params # model settings
        self.hyperparams = hyperparams # model settings
        self.manual_dist = manual_dist # distribution of parameters
        self.constant = constant # flag to make parameters frozen - no learning
        self.neurons = neurons # number of identical models to receive batch inputs
        self.parallel_current_settings = parallel_current_settings
        self.override = override # Set parameter values to a given setting from hyperparams
        self.ax_weight = ax_weight # axial conductance between compartments
        self.device = device #gpu
        
        sim_params = hyperparams['sim_params'] # simulation parameters
        
        # Initialize adjacency matrix
        self.adj_mat = adj_mat.repeat(self.neurons,1,1).to(self.device)
        self.nodes = self.adj_mat.size()[1]

        # Initialize axial conductance matrix
        self.ax_mat = torch.ones((self.nodes, self.nodes)).repeat(self.neurons,1,1).to(self.device) * ax_weight
        
        # Initialize identity matrix
        self.diag_mat = torch.eye(self.nodes).repeat(self.neurons,1,1).to(self.device)
        
        # Initialize how params are distributed
        self.distribution = distribution
        self.mu = mu
        self.sigma = sigma
        
        
        self.single_vec = torch.ones((1,1))
        self.tensor_mat = torch.ones((self.neurons, self.nodes, 1))
        
        print(f'constant = {constant} | distribution = {distribution}')
        
        if self.constant == True:
            # Set constant parameters
            if distribution == 'homogeneous':
                print(distribution)
                self.g_Na = torch.ones((1,1)).to(self.device) * params['g_Na']
                self.g_K = torch.ones((1,1)).to(self.device) * params['g_K']
                
            if distribution == 'heterogeneous':
                print(distribution)
                self.g_Na = torch.ones((1, self.nodes, 1)).to(self.device) * params['g_Na']
                self.g_K = torch.ones((1, self.nodes, 1)).to(self.device) * params['g_K']
                
            elif distribution == 'manual':
                param_dist_Na = self.manual_dist[0].reshape(1,self.nodes,1)
                param_dist_K = self.manual_dist[1].reshape(1,self.nodes,1)
                self.g_Na = param_dist_Na.to(self.device) * params['g_Na']
                self.g_K = param_dist_K.to(self.device) * params['g_K']
            
            elif distribution == 'linear':
                param_range = torch.linspace(self.mu-self.sigma, self.mu+self.sigma, self.nodes).reshape(1, self.nodes, 1)
                
                self.g_Na = param_range.to(self.device) * params['g_Na']
                self.g_K = param_range.to(self.device) * params['g_K']
                
            self.g_leak = torch.ones((self.neurons, self.nodes, 1)).to(self.device) * params['g_leak']
            self.E_leak = torch.ones((self.neurons, self.nodes, 1)).to(self.device) * params['E_leak']
            self.E_K = torch.ones((self.neurons, self.nodes, 1)).to(self.device) * params['E_K']
            self.E_Na = torch.ones((self.neurons, self.nodes, 1)).to(self.device) * params['E_Na']
            self.Cm = torch.ones((self.neurons, self.nodes, 1)).to(self.device) * params['Cm']
            self.V_Shift = torch.ones((self.neurons, self.nodes, 1)).to(self.device) * params['V_Shift']
        else:
            # set learnable parameters
            self.g_leak = torch.nn.Parameter(torch.ones((self.neurons, self.nodes, 1)).to(self.device) * params['g_leak'], requires_grad=False)
            self.E_leak = torch.nn.Parameter(torch.ones((self.neurons, self.nodes, 1)).to(self.device) * params['E_leak'], requires_grad=False)
            self.E_K = torch.nn.Parameter(torch.ones((self.neurons, self.nodes, 1)).to(self.device) * params['E_K'], requires_grad=False)
            self.E_Na = torch.nn.Parameter(torch.ones((self.neurons, self.nodes, 1)).to(self.device) * params['E_Na'], requires_grad=False)
            self.Cm = torch.nn.Parameter(torch.ones((self.neurons, self.nodes, 1)).to(self.device) * params['Cm'], requires_grad=False)
            self.V_Shift = torch.nn.Parameter(torch.ones((self.neurons, self.nodes, 1)).to(self.device) * params['V_Shift'], requires_grad=False)
        
            if self.distribution == 'homogeneous': 
                print(self.distribution)
                self.init_homogen(params)
            elif self.distribution == 'heterogeneous':
                print(self.distribution)
                self.init_heterogen(params)
            elif self.distribution == 'manual':
                print(self.distribution)
                self.init_manual(params)
            else:
                print(self.distribution)
                self.init_linear(params)
                
        if verbose:
            fig, ax = plt.subplots(2,1, figsize=(6,4), facecolor='white')
            ax[0].scatter(np.arange(0,self.nodes), self.g_Na.squeeze().squeeze().detach().cpu().numpy())
            ax[1].hist(self.g_Na.squeeze().detach().cpu().numpy(), bins = self.nodes)

            ax[0].set_ylim(self.g_Na.squeeze().squeeze().detach().cpu().min()-0.1,self.g_Na.squeeze().squeeze().detach().cpu().max()+0.1)
            ax[1].set_xlim(self.g_Na.squeeze().squeeze().detach().cpu().min()-0.1,self.g_Na.squeeze().squeeze().detach().cpu().max()+0.1)
            fig.show()
        
        self.V0 = torch.ones((self.neurons, self.nodes, 1)).to(self.device) * sim_params['V0']
        self.t0 = sim_params['t0']
        self.T = sim_params['T']
        self.dt = sim_params['dt']
        self.delay = sim_params['delay']
        self.stim_length = sim_params['stim_length']
        self.tt = torch.linspace(float(self.t0), float(self.T), \
                                 int((float(self.T) - float(self.t0))/self.dt))[1:-1].to(self.device)
        
        if self.parallel_current_settings:
            if input_current_array == None:
                self.input_current_array = torch.zeros((self.neurons, self.nodes, len(self.tt))).to(self.device)
            else:
                self.input_current_array = input_current_array.to(self.device) 
            
        else:
            if input_current_array == None:
                self.input_current_array = torch.zeros((self.nodes, len(self.tt))).to(self.device)
            else:
                self.input_current_array = input_current_array.to(self.device)     
        
        self.Q = 1
        
        self.method = method
        self.atol = atol
        self.rtol = rtol
        
        

        
    # Parameter Initialization Methods
    
    def init_homogen(self, params):
        
        if self.override == True:
            self.g_Na = Parameter(self.single_vec* params['g_Na'], requires_grad=True)
            self.g_K = Parameter(self.single_vec * params['g_K'], requires_grad=True)
            
            self.register_parameter('g_Na', self.g_Na)
            self.register_parameter('g_K', self.g_K)
        else:
            # Initialize learnable parameters
            self.register_parameter('g_Na', Parameter(torch.ones((1,1))))
            self.register_parameter('g_K', Parameter(torch.ones((1,1))))

            # Set initial leanable parameter values
            torch.nn.init.uniform_(self.g_Na, a=params['g_Na']/200-abs(self.sigma), b=params['g_Na']/200+abs(self.sigma)) * 200
            torch.nn.init.uniform_(self.g_K, a=params['g_K']/200-abs(self.sigma), b=params['g_K']/200+abs(self.sigma)) * 200
            
    
    def init_heterogen(self, params):
        
        if self.override == True:
            self.g_Na = Parameter(torch.ones((1, self.nodes, 1)) * params['g_Na'], requires_grad=True)
            self.g_K = Parameter(torch.ones((1, self.nodes, 1)) * params['g_K'], requires_grad=True)
            
            self.register_parameter('g_Na', self.g_Na)
            self.register_parameter('g_K', self.g_K)
        else:
            # Initialize learnable parameters
            self.register_parameter('g_Na', Parameter(torch.ones((1, self.nodes, 1))))
            self.register_parameter('g_K', Parameter(torch.ones((1, self.nodes, 1))))

            # Set initial learnable parameter values
            torch.nn.init.uniform_(self.g_Na, a=params['g_Na']/200-abs(self.sigma), b=params['g_Na']/200+abs(self.sigma)) * 200
            torch.nn.init.uniform_(self.g_K, a=params['g_K']/200-abs(self.sigma), b=params['g_K']/200+abs(self.sigma)) * 200
            
    def init_linear(self, params):
        
        param_range = torch.linspace(self.mu-self.sigma, self.mu+self.sigma, self.nodes).reshape(1, self.nodes, 1)
        
        if self.override == True:
            self.g_Na = Parameter(param_range * torch.ones((1, self.nodes, 1)) * params['g_Na'], requires_grad=True)
            self.g_K = Parameter(param_range * torch.ones((1, self.nodes, 1)) * params['g_K'], requires_grad=True)
            
            self.register_parameter('g_Na', self.g_Na)
            self.register_parameter('g_K', self.g_K)
            
            
        else:
            # Initialize learnable parameters
            self.register_parameter('g_Na', Parameter((param_range * torch.ones((1, self.nodes, 1))) * 200))
            self.register_parameter('g_K', Parameter((param_range * torch.ones((1, self.nodes, 1))) * 200))
            
            
    def init_manual(self, params):
        
        param_dist = self.manual_dist.reshape(1,self.nodes,1)
        
        if self.override == True:
            self.g_Na = Parameter(param_dist * torch.ones((1, self.nodes, 1)) * params['g_Na'], requires_grad=True)
            self.g_K = Parameter(param_dist * torch.ones((1, self.nodes, 1)) * params['g_K'], requires_grad=True)
            
            self.register_parameter('g_Na', self.g_Na)
            self.register_parameter('g_K', self.g_K)
        else:
            # Initialize learnable parameters
            self.register_parameter('g_Na', Parameter((param_dist * torch.ones((1, self.nodes, 1))) * 200))
            self.register_parameter('g_K', Parameter((param_dist * torch.ones((1, self.nodes, 1))) * 200))

    
    
    # Membrane currents (in uA/cm^2)
    
    #  Fast inactivating Sodium Current
    def I_Na(self, V, m, h): 
        if self.constant == True:
            return self.g_Na * m**2 * h * (V - self.E_Na)
        elif self.override == True:
            return self.g_Na * m**2 * h * (V - self.E_Na)
        else:
            return torch.abs(self.g_Na) * m**2 * h * (V - self.E_Na) #Scaling g_Na to be between 0 and 200 # Changed to abs()
    #  Fast inactivating Potassium Current
    def I_K(self, V, m): 
        if self.constant == True:
            return self.g_K * m    * (V - self.E_K)
        elif self.override == True:
            return self.g_K * m    * (V - self.E_K)
        else:
            return torch.abs(self.g_K) * m    * (V - self.E_K) #Scaling g_K to be between 0 and 200 # Changed to abs()
    #  Leak current
    def I_leak(self, V):     return self.g_leak           * (V - self.E_leak)
    
    def forward(self, t, state):
        # Differential equation function
        V, m_Na, m_K, h_Na = state
        
        V = V + self.V_Shift

        dV = (self.Iinj(t) 
              - self.I_leak(V) 
              - self.I_Na(V, m_Na, h_Na)
              - self.I_K(V, m_K) 
              + self.Iadj(V)
              ) / self.Cm

        #Activations
        dm_Na = (self.m_inf_Na(V) - m_Na) / self.m_tau_Na(V)
        dm_K = 2*(self.m_inf_K(V) - m_K) / self.m_tau_K(V)
        #Inactivations
        dh_Na = (self.h_inf_Na(V) - h_Na) / self.h_tau_Na(V)
        

        # Return x value after eval one step
        return(dV, dm_Na, dm_K, dh_Na)


    # Activations
    # Sodium Activation
    def m_alpha_Na(self, V): return(1.841*torch.log(0.102*torch.exp(0.17*V) + 1)) # Softplus threshold
    def m_beta_Na(self, V): return(2.010*torch.log(266.087*torch.exp(-0.139*V) + 1)) # Softplus threshold
    def m_tau_Na(self, V): return(1 / (self.m_alpha_Na(V) + self.m_beta_Na(V) + 1e-8) / self.Q)
    def m_inf_Na(self, V): return(self.m_alpha_Na(V) / (self.m_alpha_Na(V) + self.m_beta_Na(V) + 1e-8))
    
    # Potassium Activation
    def m_alpha_K(self, V): return(0.114*torch.log(0.007*torch.exp(0.140*V) + 1)) # Softplus threshold
    def m_beta_K(self, V): return(0.25 * torch.exp((20 - V) / 40))
    def m_tau_K(self, V): return(1 / (self.m_alpha_K(V) + self.m_beta_K(V) + 1e-8) / self.Q)
    def m_inf_K(self, V): return(self.m_alpha_K(V) / (self.m_alpha_K(V) + self.m_beta_K(V) + 1e-8))
    
    # Inactivations
    # Potassium Inactivation
    def h_alpha_Na(self, V): return(0.128 * torch.exp((17 - V) / 18))
    def h_beta_Na(self, V): return(4 / (1 + torch.exp((40 - V) / 5)))
    def h_tau_Na(self, V): return(1 / (self.h_alpha_Na(V) + self.h_beta_Na(V) + 1e-8) / self.Q)
    def h_inf_Na(self, V): return(self.h_alpha_Na(V) / (self.h_alpha_Na(V) + self.h_beta_Na(V) + 1e-8))


    # Get initial state
    def get_initial_state(self):
        state = (self.V0, 
                 self.m_inf_Na(self.V0),
                 self.m_inf_K(self.V0),
                 self.h_inf_Na(self.V0)
                )
        
        return(self.t0, state)

    # Injected, external current, I_ext
    def Iinj(self, t):
        # determine index for lookup
        ix_t = int(t/self.dt)
        if self.parallel_current_settings:
            if ix_t >= len(self.input_current_array[0,0]):
                ix_t = len(self.input_current_array[0,0])-1
        else:
            if ix_t >= len(self.input_current_array[0]):
                ix_t = len(self.input_current_array[0])-1

        # Lookup input current and return
        if self.parallel_current_settings:
            self.injection = self.input_current_array[:, :, ix_t].unsqueeze(-1)
        else:
            self.injection = self.input_current_array[:, ix_t].unsqueeze(-1)

        return(self.injection)
            

    # Current from adjacent compartments
    def Iadj(self, V):
        #Make V diff Matrix
        V_mat = V.permute(0,2,1).repeat(1,self.nodes,1)
        V_diff = V_mat.permute(0,2,1) - V_mat
        # Calculate current changes from adjacent compartments (values fall on the diagonal)
        adj_current = torch.bmm((self.ax_mat*self.adj_mat), (V_diff*self.adj_mat))
        adj_current = (adj_current * self.diag_mat).sum(dim=1).unsqueeze(2)
        
        return(adj_current)

    def simulate(self, atol=1e-8, rtol=1e-8, method='dopri5'):        
        t0, state = self.get_initial_state()
        solution = odeint(self, state, self.tt, atol=self.atol, rtol=self.rtol, method=self.method)
        
        # Hinge loss calculation - NOT USED IN OPTIMIZATION given training loop setup
        if self.override:
            loss_model = self.hinge_criterion(self.g_Na/200) + self.hinge_criterion(self.g_K/200)
        else:
            loss_model = self.hinge_criterion(self.g_Na) + self.hinge_criterion(self.g_K)
        
        return(self.tt, solution, loss_model)
    
    def hinge_criterion(self, activity):
        
        # Specify targets with same size as layer
        hinge_loss = Hinge_loss(margin=0, device=self.device)
        target = - torch.ones_like(activity)
        return(hinge_loss(activity-1, target) + hinge_loss(0-activity, target))

class Hinge_loss(nn.Module):
    def __init__(self, margin = 1, reduction='mean', device='cuda:0'):
        super(Hinge_loss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.device = device
    
    def forward(self, y, target):
        if self.reduction == 'sum':
            to_max = torch.cat((torch.zeros_like(target).to(self.device).unsqueeze(2), (self.margin - y*target).unsqueeze(2)), dim=2)
            maxxed, _ = torch.max(to_max, dim=2)
            return(torch.mean(maxxed, dim=1))
        elif self.reduction == 'mean':
            to_max = torch.cat((torch.zeros_like(target).to(self.device).unsqueeze(2), (self.margin - y*target).unsqueeze(2)), dim=2)
            maxxed, _ = torch.max(to_max, dim=2)
            return(torch.mean(maxxed, dim=1))
        else:
            to_max = torch.cat((torch.zeros_like(target).to(self.device).unsqueeze(2), (self.margin - y*target).unsqueeze(2)), dim=2)
            maxxed, _ = torch.max(to_max, dim=2)
            return(maxxed)



def load_default_params(model_type='hh'):
    if model_type == 'hh':
        with open('./results/refs/params.pickle', 'rb') as handle:
            params = pickle.load(handle)
        return(params)
    if model_type == 'drion':
        with open('./results/refs/params_drion.pkl', 'rb') as handle:
            params = pickle.load(handle)
        return(params)

def load_default_sim_params():
    with open('./results/refs/sim_params.pkl', 'rb') as handle:
            sim_params = pickle.load(handle)
    return(sim_params)
    
def make_gaussian_params(mu, sigma, samples, scale):
    x = np.linspace(0, samples, samples)
    y = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    y = y * scale
    y = torch.Tensor(y)
    return x, y