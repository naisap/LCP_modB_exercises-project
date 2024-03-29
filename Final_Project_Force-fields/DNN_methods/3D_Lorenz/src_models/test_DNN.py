# Modifications are made on top of the original DeepCalib code.

def predict(network, trajectory):
    """ Predict parameters of the force field from the trajectory using the deep learnign network.
    
    Inputs:
    network: deep learning network
    image: trajectroy [numpy array of real numbers]
    
    Output:
    predicted_targets: predicted parameters of the calibrated force field [1D numpy array containing outputs]
    """
    
    from numpy import reshape
    
    network_blocksize = network.get_layer(index=0).get_config()['batch_input_shape'][1:][1]
    predicted_targets = network.predict(reshape(trajectory, [1,round(trajectory.size/network_blocksize),network_blocksize]))   
        
    return predicted_targets


def test_performance(simulate_trajectory, network, rescale_targets, number_of_predictions_to_show=100):#, dt = 1e-1):

    
    network_blocksize = network.get_layer(index=0).get_config()['batch_input_shape'][1:][1]


    predictions_scaled = []
    predictions_physical = []

    batch_size = number_of_predictions_to_show
    trajectory, trajectory_real, targets, targets_real = simulate_trajectory(batch_size)
    targets_physical = list(targets_real)#targets.values)
    targets_scaled = list(targets)#.scaled_values)
    #trajectory = trajectory.scaled_values
    trajectory_dimensions = [number_of_predictions_to_show, round(trajectory.size/network_blocksize/number_of_predictions_to_show) , network_blocksize]
    trajectories = np.array(trajectory).reshape(trajectory_dimensions)
       

    for i in range(number_of_predictions_to_show):
        predictions = predict(network, trajectories[i])


        predictions_scaled.append(predictions[0])
        predictions_physical.append(rescale_targets(*predictions[0]))

    number_of_outputs = network.get_layer(index=-1).get_config()['units']    

    targets_physical = np.array(targets_physical).transpose()
    targets_scaled = np.array(targets_scaled).transpose()
    predictions_scaled = np.array(predictions_scaled).transpose()
    predictions_physical = np.array(predictions_physical).transpose()

    # Do not show results at the edges of the training range 

    if number_of_outputs>1:

        ind = np.isfinite(targets_scaled[0])
        for target_number in range(number_of_outputs):
            target_max = .9 * np.max(targets_scaled[target_number]) + .1 * np.min(targets_scaled[target_number])
            target_min = .1 * np.max(targets_scaled[target_number]) + .9 * np.min(targets_scaled[target_number])
            ind = np.logical_and(ind, targets_scaled[target_number] < target_max)
            ind = np.logical_and(ind, targets_scaled[target_number] > target_min)
    else:
        target_max = .9 * np.max(targets_scaled) + .1 * np.min(targets_scaled)
        target_min = .1 * np.max(targets_scaled) + .9 * np.min(targets_scaled)
        ind = np.logical_and(targets_scaled < target_max, targets_scaled > target_min)

    return targets_scaled, targets_physical, predictions_scaled, predictions_physical


def plot_test_performance(targets_scaled, targets_physical, predictions_scaled, predictions_physical, network):
    
    import matplotlib.pyplot as plt
    import numpy as np
    #from . import predict

    number_of_outputs = network.get_layer(index=-1).get_config()['units']    
    
    if number_of_outputs>1:

        for target_number in range(number_of_outputs):
            plt.figure(figsize=(20, 10))

            plt.subplot(121)
            plt.plot(targets_scaled[target_number],
                     predictions_scaled[target_number],
                     '.')
            #plt.xlabel(targets.scalings[target_number], fontsize=18)
            #plt.ylabel('Predicted ' + targets.scalings[target_number], fontsize=18)
            plt.axis('square')
            plt.title('Prediction performance in scaled units', fontsize=18)

            plt.subplot(122)
            plt.plot(targets_physical[target_number],
                     predictions_physical[target_number],
                    '.')
            #plt.xlabel(targets.names[target_number], fontsize=18)
            #plt.ylabel('Predicted ' + targets.names[target_number], fontsize=18)
            plt.axis('square')
            plt.title('Prediction performance in real units', fontsize=18)


    else: 
        plt.figure(figsize=(20, 10))

        plt.subplot(121)
        plt.plot(targets_scaled[ind],
                 predictions_scaled.transpose()[ind],
                 '.')
        #plt.xlabel(targets.scalings[0], fontsize=18)
        #plt.ylabel('Predicted ' + targets.scalings[0], fontsize=18)
        plt.axis('square')
        plt.title('Prediction performance in scaled units', fontsize=18)

        plt.subplot(122)
        plt.plot(targets_physical[ind],
                 predictions_physical.transpose()[ind],
                '.')
        #plt.xlabel(targets.names[0], fontsize=18)
        #plt.ylabel('Predicted ' + targets.names[0], fontsize=18)
        plt.axis('square')
        plt.title('Prediction performance in real units', fontsize=18)