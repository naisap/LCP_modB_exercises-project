import numpy as np
from scipy.constants import Boltzmann as kB
from numpy.random import randn as gauss
from numpy.random import rand as uniform

### follows the parameters used in DeepCalib

#### 1D

### Physical parameters 
R = 1e-7                                # Radius of the Brownian particle [m]
eta = 0.001                             # Viscosity of the medium [kg m^-1 s^-1]
T = 300                                 # Temperature [K]
L0 = 2e-6                               # Reference distance from middle to one minimum [m]
H0 = kB*300                             # Barrier height [Joule]
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
scale_targets = lambda L, H: [L/L0 -1, np.log(H / H0)]                        # Scales targets to order 1
rescale_targets = lambda scaled_L, scaled_H: [(1 + scaled_L)*L0*1e6, 
                                            np.exp(scaled_H) * H0/kB/300] # Inverse of targets_scaling

def simulate_trajectory(batch_size=batch_size, 
                        T=T,
                        H0=H0,
                        L0=L0,
                        gamma0=gamma0,
                        N=N, 
                        Dt=Dt, 
                        oversampling=oversampling, 
                        offset=offset):#, 
                        #scale_inputs=scale_inputs, 
                        #scale_targets=scale_targets):
    
    ### Randomize trajectory parameters
    L = L0 * (uniform(batch_size)+.5) 
    H = H0 * 10**(uniform(batch_size)*1.75 - .75)       # Generates random values for computing the stiffness
    gamma = gamma0 * (uniform(batch_size) * .1 + .95)   # Marginal randomization of friction coefficient to tolarate small changes

    ### Simulate
    dt = Dt / oversampling                 # time step of the simulation
    x = np.zeros((batch_size, N))          # initialization of the x array
    k0 = 4*H/L**2 
    k1 = 4*H/L**4
    D = kB * T / gamma                     # diffusion coefficient
    C1 = +k0 / gamma * dt
    C2 = -k1 / gamma * dt                  # drift coefficient of the Langevin equation
    C3 = np.sqrt(2 * D * dt)               # random walk coefficient of the Langevin equation
    X = x[:, 0]
    n = 0
    
    for t in range(offset):                      # Offset (for some prerun before running)
        X = X + C1 * X + C2 * X**3 + C3 * gauss(batch_size)
        #X = X - (2 * X**3) + (12 * X**2) - (18 * X) + 3 + (C3 * gauss(batch_size))
        
    for t in range(N * oversampling):            # Simulation                
        X = X + C1 * X + C2 * X**3 + C3 * gauss(batch_size)
        #X = X - (2 * X**3) + (12 * X**2) - (18 * X) + 3 + (C3 * gauss(batch_size))
        if t % oversampling == 0:                # We save every oversampling^th values 
            x[:, n] = X 
            n += 1
            
    inputs = scale_inputs(x)
    inputs_real = x
    targets = np.swapaxes(scale_targets(*[L, H]),0,1)
    target_real = np.swapaxes([L*1e6, H/kB/300],0,1)

    return inputs, inputs_real, targets, target_real 