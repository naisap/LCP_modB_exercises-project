import numpy as np
from scipy.constants import Boltzmann as kB
from numpy.random import randn as gauss
from numpy.random import rand as uniform

## Physical parameters
sigma0 = 10                               
rho0 = 28                             
beta0 = 2.667
eta = 0.001                             # Viscosity of the medium [kg m^-1 s^-1]
R = 1e-7                                # Radius of the Brownian particle [m]
T = 300
gamma0 = 6 * np.pi * eta * R            # Reference friction coefficient [kg s^-1]

### Simulation parameters
N = 3000                   # Number of samples of the trajectory
Dt = 1e-2                  # Timestep 
oversampling = 5           # Simulation oversampling
offset = 1000              # Number of equilibration timesteps
batch_size = 32            # Number of trajectories

### Define functions to scale and rescale inputs
scale_inputs = lambda x, y, z: [x * 1e+6, y * 1e+6, z * 1e+6]        # Scales input trajectory to order 1
rescale_inputs = lambda scaled_x, scaled_y, scaled_z: [scaled_x * 1e-6,
                            scaled_y * 1e-6, scaled_z * 1e-6]     # Rescales input trajectory to physical units

### Define function to scale and rescale targets
scale_targets = lambda sigma, rho, beta: [sigma, rho, beta]                        # Scales targets to order 1
rescale_targets = lambda scaled_sigma, scaled_rho, scaled_beta: [scaled_sigma, 
                                                                 scaled_rho, 
                                                                 scaled_beta] # Inverse of targets_scaling

def simulate_trajectory(batch_size=batch_size, 
                sigma0=sigma0,
                rho0=rho0,
                beta0=beta0,
                T=T,
                N=N, 
                Dt=Dt, 
                oversampling=oversampling, 
                offset=offset):

    ### Randomize trajectory parameters
    sigma = sigma0 * (uniform(batch_size)+.5)
    rho = rho0 * (uniform(batch_size)+.5)
    beta = beta0 * (uniform(batch_size)+.5)
    gamma = gamma0 * (uniform(batch_size) * .1 + .95)   # Marginal randomization of friction coefficient to tolarate small changes

    ### Simulate
    dt = Dt / oversampling                 # time step of the simulation
    x = np.zeros((batch_size, N))          # initialization of the x array
    y = np.zeros((batch_size, N))          # initialization of the y array
    z = np.zeros((batch_size, N))          # initialization of the z array
    D = kB * T / gamma                     # diffusion coefficient
    C1 = sigma #* dt                            # sigma
    C2 = rho #* dt                              # rho
    C3 = beta #* dt                             # beta
    C4 = np.sqrt(2 * D * dt)               # random walk coefficient of the Langevin equation
    X = x[:, 0]
    Y = y[:, 0]
    Z = z[:, 0]
    n = 0

    for t in range(offset):                      # Offset (for some prerun before running)
        X = X + C1*(Y-X)*dt + C4*gauss(batch_size)
        Y = Y + (C2*X-Y-X*Z)*dt + C4*gauss(batch_size)
        Z = Z + (X*Y - C3*Z)*dt + C4*gauss(batch_size)

    for t in range(N * oversampling):            # Simulation 
        X = X + C1*(Y-X)*dt + C4*gauss(batch_size)
        Y = Y + (C2*X-Y-X*Z)*dt + C4*gauss(batch_size)
        Z = Z + (X*Y - C3*Z)*dt + C4*gauss(batch_size) 
            
        if t % oversampling == 0:                # We save every oversampling^th values 
            x[:, n] = X 
            y[:, n] = Y
            z[:, n] = Z
            n += 1
            
    inputs = np.swapaxes([x, y, z],0,1)
    inputs_real = np.swapaxes([x, y, z],0,1)
    targets = np.swapaxes([sigma, rho, beta],0,1)
    target_reals = np.swapaxes([sigma, rho, beta],0,1)

    return inputs, inputs_real, targets, target_reals