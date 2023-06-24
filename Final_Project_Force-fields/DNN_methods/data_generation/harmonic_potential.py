import numpy as np
from scipy.constants import Boltzmann as kB
from numpy.random import randn as gauss
from numpy.random import rand as uniform
        
### follows the parameters used in DeepCalib

### Physical parameters 
R = 1e-7                                # Radius of the Brownian particle [m]
eta = 0.001                             # Viscosity of the medium [kg m^-1 s^-1]
T = 300                                 # Temperature [K]
k0 = 10                                 # Reference stiffness [fN \mu m ^-1]
gamma0 = 6 * np.pi * eta * R            # Reference friction coefficient [kg s^-1]

### Simulation parameters
N = 1000                   # Number of samples of the trajectory
Dt = 1e-2                  # Timestep 
oversampling = 5           # Simulation oversampling
offset = 1000              # Number of equilibration timesteps
batch_size = 32            # Number of trajectories

### Define functions to scale and rescale inputs
scale_inputs = lambda x: x * 1e+6                    # Scales input trajectory to order 1
rescale_inputs = lambda scaled_x: scaled_x * 1e-6    # Rescales input trajectory to physical units

### Define function to scale and rescale targets
scale_targets = lambda k: np.log(k / k0)                               # Scales targets to order 1
rescale_targets = lambda scaled_k: np.exp(scaled_k) * k0               # Inverse of targets_scaling

def simulate_trajectory(batch_size=batch_size, 
                    T=T,
                    k0=k0,
                    gamma0=gamma0,
                    N=N, 
                    Dt=Dt, 
                    oversampling=oversampling, 
                    offset=offset):

    ### Randomize trajectory parameters
    k = k0 * (10**(uniform(batch_size) * 3 - 1.5))     # Generates random stiffness values that are uniformly distributed in log scale
    gamma = gamma0 * (uniform(batch_size) * .1 + .95)   # Marginal randomization of friction coefficient to tolarate small changes

    ### Simulate
    dt = Dt / oversampling                 # time step of the simulation
    x = np.zeros((batch_size, N))          # initialization of the x array
    D = kB * T / gamma                     # diffusion coefficient
    C1 = -k *1e-9 / gamma * dt             # drift coefficient of the Langevin equation
    C3 = np.sqrt(2 * D * dt)               # random walk coefficient of the Langevin equation
    X = x[:, 0]
    n = 0
    
    for t in range(offset):                      # Offset (for some prerun before running)
        X = X + C1 * X + C3 * gauss(batch_size)
        
    for t in range(N * oversampling):            # Simulation                
        X = X + C1 * X + C3 * gauss(batch_size)
        if t % oversampling == 0:                # We save every oversampling^th values 
            x[:, n] = X 
            n += 1
            
    inputs = scale_inputs(x)
    inputs_real = x
    targets = scale_targets(k)
    target_real = k

    return inputs, inputs_real, targets, target_real