# Modifications are made on top of the original DeepCalib code.

import matplotlib.pyplot as plt
import numpy as np
from numpy import reshape

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


#def test_performance(simulate_trajectory, network, rescale_targets, number_of_predictions_to_show=100):#, dt = 1e-1):
def plot_test_performance(simulate_trajectory, network, rescale_targets, number_of_predictions_to_show=100):
    
    import numpy as np
    import matplotlib.pyplot as plt

    network_blocksize = network.get_layer(index=0).get_config()['batch_input_shape'][1:][1]


    predictions_scaled = []
    predictions_physical = []

    batch_size = number_of_predictions_to_show
    trajectory, _ , targets, targets_real = simulate_trajectory(batch_size)
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
    print(number_of_outputs) 

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
        print(ind)
        plt.subplot(121)
        plt.plot(targets_scaled[ind],
                 predictions_scaled.transpose()[ind], '.')
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





def plot_test_performance2(predictions_list, networks_list, trajectory_name, network_names=['GRU','LSTM','GRU with attention', 'LSTM with attention', 'RNN with attention']):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    #plt.figure(figsize=(20, 10))

    #fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(24, 12))

    #plt.subplot(121)
    #plot_list = [[121,122],[121,122]]

    plot_colors = ['red','orange','blue','green','grey']
    plot_markers = ['s','p','*','h','H']

    if trajectory_name == 'double-well':
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(24, 12))

        label_x = [['L/L0 -1','Distance[$\mu$m]'], ['log(H/H0)', 'Barrier Height [$k_B$T]'] ]
        label_y = [['Predicted L/L0 -1','Predicted Distance[$\mu$m]'], ['Predicted log(H/H0)', 'Predicted Barrier Height [$k_B$T]']]
    
    elif trajectory_name == 'harmonic':
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))

        label_x = ['log(k/k0)', 'k [fN/$\mu$m]']
        label_y = ['Predicted log(k/k0)', 'Predicted k [fN/$\mu$m]']
    
    elif trajectory_name == 'lorenz_63':
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(24, 12))

        label_x = ['$\sigma$','{}'.format(r"$\rho$"),'{}'.format(r"$\beta$")]
        label_y = ['Predicted $\sigma$','Predicted {}'.format(r"$\rho$"), 'Predicted {}'.format(r"$\beta$")]


    i=0
    for prediction, network in zip(predictions_list, networks_list): 

        #print(np.shape(prediction))
        
        targets_scaled = prediction[0]
        targets_physical = prediction[1]
        predictions_scaled = prediction[2]
        predictions_physical = prediction[3]

        number_of_outputs = network.get_layer(index=-1).get_config()['units']   
    
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
    

        # plots
        if number_of_outputs>1:

            for target_number in range(number_of_outputs):

                if trajectory_name == 'lorenz_63':

                    ## 1st row

                    axs[target_number].plot(targets_physical[target_number],
                            predictions_physical[target_number],
                            '.', color=plot_colors[i], marker=plot_markers[i], alpha=0.1)

                    #find line of best fit
                    res = stats.linregress(targets_physical[target_number], predictions_physical[target_number])
                    axs[target_number].plot(targets_physical[target_number], res.intercept + res.slope*targets_physical[target_number], label = network_names[i]+': {}'.format('- $R^2$: {}'.format(round(res.rvalue**2,5))), color=plot_colors[i])

                    axs[target_number].set_xlabel(label_x[target_number], fontsize=18)
                    axs[target_number].set_ylabel(label_y[target_number], fontsize=18)

                    axs[0].set_title('Prediction performance in real units', fontsize=18)

                    #axs[2,0].legend()
                    axs[target_number].legend()

                else:

                    ## 1st row
                    axs[target_number,0].plot(targets_scaled[target_number],
                            predictions_scaled[target_number],
                            '.', color=plot_colors[i], marker=plot_markers[i], alpha=0.1)

                    #find line of best fit
                    #a, b = np.polyfit(targets_scaled[target_number],
                    #        predictions_scaled[target_number], 1)
                    #add line of best fit to plot
                    #axs[target_number,0].plot(targets_scaled[target_number], a*targets_scaled[target_number]+b, label = network_names[i]+': {}'.format(np.mean(a*targets_scaled[target_number]+b)), color=plot_colors[i])

                    res = stats.linregress(targets_scaled[target_number], predictions_scaled[target_number])
                    axs[target_number,0].plot(targets_scaled[target_number], res.intercept + res.slope*targets_scaled[target_number], label = network_names[i], color=plot_colors[i])

                    axs[0,0].set_xlabel(label_x[0][0], fontsize=18)
                    axs[0,0].set_ylabel(label_y[0][0], fontsize=18)
                    axs[0,1].set_xlabel(label_x[0][1], fontsize=18)
                    axs[0,1].set_ylabel(label_y[0][1], fontsize=18)

                    #axs[0,0].axis('square')
                    axs[0,0].set_title('Prediction performance in scaled units', fontsize=18)
                    #axs[1,0].axis('square')
                    axs[1,0].set_title('Prediction performance in scaled units', fontsize=18)


                    ## 2nd row
                    #plt.subplot(grid[1])
                    axs[target_number,1].plot(targets_physical[target_number],
                            predictions_physical[target_number],
                            '.', color=plot_colors[i], marker=plot_markers[i], alpha=0.1)

                    #find line of best fit
                    #a, b = np.polyfit(targets_physical[target_number],
                    #        predictions_physical[target_number], 1)
                    #add line of best fit to plot
                    #axs[target_number,1].plot(targets_physical[target_number], a*targets_physical[target_number]+b, label = network_names[i]+': {}'.format(np.mean(a*targets_physical[target_number]+b)), color=plot_colors[i])

                    res = stats.linregress(targets_physical[target_number],predictions_physical[target_number])
                    axs[target_number,1].plot(targets_physical[target_number], res.intercept + res.slope*targets_physical[target_number], label = network_names[i]+': {}'.format('- $R^2$: {}'.format(round(res.rvalue**2,5))), color=plot_colors[i])

                    axs[1,0].set_xlabel(label_x[1][0], fontsize=18)
                    axs[1,0].set_ylabel(label_y[1][0], fontsize=18)
                    axs[1,1].set_xlabel(label_x[1][1], fontsize=18)
                    axs[1,1].set_ylabel(label_y[1][1], fontsize=18)

                    #axs[0,1].axis('square')
                    axs[0,1].set_title('Prediction performance in real units', fontsize=18)
                    #axs[1,1].axis('square')
                    axs[1,1].set_title('Prediction performance in real units', fontsize=18)

                    axs[0,0].legend()
                    axs[0,1].legend()
                    axs[1,0].legend()
                    axs[1,1].legend()

        else: 
            ## 1st
            axs[0].plot(targets_scaled[ind], predictions_scaled.transpose()[ind], '.', color=plot_colors[i], marker=plot_markers[i], alpha=0.1)

            res = stats.linregress(targets_scaled[ind], predictions_scaled.transpose()[ind,0])
            axs[0].plot(targets_scaled[ind], res.intercept + res.slope*targets_scaled[ind], label = network_names[i], color=plot_colors[i])

            axs[0].set_ylabel(label_y[0], fontsize=18)
            axs[0].set_xlabel(label_x[0], fontsize=18)

            axs[0].set_title('Prediction performance in scaled units', fontsize=18)

            ## 2nd
            axs[1].plot(targets_physical[ind], predictions_physical.transpose()[ind], '.', color=plot_colors[i], marker=plot_markers[i], alpha=0.1)
            
            res = stats.linregress(targets_physical[ind], predictions_physical.transpose()[ind])
            axs[1].plot(targets_physical[ind], res.intercept + res.slope*targets_physical[ind], label = network_names[i]+': {}'.format('- $R^2$: {}'.format(round(res.rvalue**2,5))), color=plot_colors[i])


            axs[1].set_ylabel(label_y[1], fontsize=18)
            axs[1].set_xlabel(label_x[1], fontsize=18)

            axs[1].set_title('Prediction performance in real units', fontsize=18)

            axs[0].legend()
            axs[1].legend()

        i += 1 #next network from network_list

    return plt.show()