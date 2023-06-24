# Modifications are made on top of the original DeepCalib code.

def train_deep_learning_network(
    network,
    simulate_trajectory,
    sample_sizes = (32, 128, 512),#(32, 128, 512, 2048),
    iteration_numbers = (3001, 2001, 1001),#(1001, 2001, 3001),#(3001, 2001, 1001, 101),
    verbose=.1):
    """Train a deep learning network.
    
    Input:
    network: deep learning network
    simulate_trajectory: trajectory generator function
    sample_sizes: sizes of the batches of trajectories used in the training [tuple of positive integers]
    iteration_numbers: numbers of batches used in the training [tuple of positive integers]
    verbose: frequency of the update messages [number between 0 and 1]
        
    Output:
    training_history: dictionary with training history
    """  
    
    import numpy as np
    from time import time
     
    training_history = {}
    training_history['Sample Size'] = []
    training_history['Iteration Number'] = []
    training_history['Iteration Time'] = []
    training_history['MSE'] = []
    training_history['MAE'] = []
    
    for sample_size, iteration_number in zip(sample_sizes, iteration_numbers):
        for iteration in range(iteration_number):
            
            # measure initial time for iteration
            initial_time = time()

            # generate trajectories and targets
            network_blocksize = network.get_layer(index=0).get_config()['batch_input_shape'][1:][1]                        
            number_of_outputs = network.get_layer(index=-1).get_config()['units']
            output_shape = (sample_size, number_of_outputs)
            targets = np.zeros(output_shape)
            
            
            batch_size = sample_size
            trajectory, _ , target, target_real = simulate_trajectory(batch_size)
            #trajectory = trajectory.scaled_values
            trajectory_dimensions = [sample_size, round(trajectory.size/network_blocksize/sample_size), network_blocksize]
            trajectories = np.array(trajectory).reshape(trajectory_dimensions)
            targets = target#.scaled_values
                
                

            # training
            history = network.fit(trajectories,
                                targets,
                                epochs=1, 
                                batch_size=sample_size,
                                verbose=False)
                        
            # measure elapsed time during iteration
            iteration_time = time() - initial_time

            # record training history
            mse = history.history['mse'][0]
            mae = history.history['mae'][0]
                        
            training_history['Sample Size'].append(sample_size)
            training_history['Iteration Number'].append(iteration)
            training_history['Iteration Time'].append(iteration_time)
            training_history['MSE'].append(mse)
            training_history['MAE'].append(mae)

            if not(iteration%int(verbose**-1)):
                print('Sample size %6d   iteration number %6d   MSE %10.4f   MAE %10.4f   Time %10f ms' % (sample_size, iteration + 1, mse, mae, iteration_time * 1000))
                
    return training_history


def plot_learning_performance(training_history, number_of_timesteps_for_average = 100, figsize=(20,20)):
    """Plot the learning performance of the deep learning network.
    
    Input:
    training_history: dictionary with training history, typically obtained from train_deep_learning_network()
    number_of_timesteps_for_average: length of the average [positive integer number]
    figsize: figure size [list of two positive numbers]
        
    Output: none
    """    

    import matplotlib.pyplot as plt
    from numpy import convolve, ones
    
    plt.figure(figsize=figsize)

    plt.subplot(5, 1, 1)
    plt.semilogy(training_history['MSE'], 'k')
    plt.semilogy(convolve(training_history['MSE'], ones(number_of_timesteps_for_average) / number_of_timesteps_for_average, mode='valid'), 'r')
    plt.ylabel('MSE', fontsize=24)
    plt.xlabel('Epochs', fontsize=24)

    plt.subplot(5, 1, 2)
    plt.semilogy(training_history['MAE'], 'k')
    plt.semilogy(convolve(training_history['MAE'], ones(number_of_timesteps_for_average) / number_of_timesteps_for_average, mode='valid'), 'r')
    plt.ylabel('MAE', fontsize=24)
    plt.xlabel('Epochs', fontsize=24)
    plt.show()