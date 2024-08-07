import numpy as np
import logging
from matplotlib import pyplot as plt

def plotWeights(inputs, weight_arr, drgnfly, bias_arr): 
    color_arr = ['#FF0000', '#800000', '#808000', '#00FF00', '#008000', '#00FFFF', '#008080', '#0000FF', '#000080', '#FF00FF', '#800080', '#FFA07A', '#FF4500', '#FFD700', '#ADFF2F', '#32CD32', '#20B2AA', '#00CED1', '#1E90FF']

    plt.figure(figsize=(8, 6)) 
    for i in range(drgnfly.numInputNodes):
        if i > len(color_arr):
            plt.scatter(inputs.X[:,i], weight_arr[:,i], c='m', label='Input #' + str(i))
            plt.plot(np.unique(inputs.X[:,i]), np.poly1d(np.polyfit(inputs.X[:,i], weight_arr[:,i], 2))(np.unique(inputs.X[:,i])), c='m',lw=1)

            # plt.scatter(inputs.X[:,i], bias_arr[:], c='k', label='Bias for Input #' + str(i), marker='^')
            # plt.plot(np.unique(inputs.X[:,i]), np.poly1d(np.polyfit(inputs.X[:,i], bias_arr[:], 2))(np.unique(inputs.X[:,i])), c='k', lw=1, linestyle='--')

        else:
            plt.scatter(inputs.X[:,i], weight_arr[:,i], c=color_arr[i], label='Input #' + str(i))
            plt.plot(np.unique(inputs.X[:,i]), np.poly1d(np.polyfit(inputs.X[:,i], weight_arr[:,i], 2))(np.unique(inputs.X[:,i])), c=color_arr[i],lw=1)

            # plt.scatter(inputs.X[:,i], bias_arr[:], c=color_arr[i], label='Bias for Input #' + str(i), marker='^')
            # plt.plot(np.unique(inputs.X[:,i]), np.poly1d(np.polyfit(inputs.X[:,i], bias_arr[:], 2))(np.unique(inputs.X[:,i])), c=color_arr[i], lw=1, linestyle='--')

        """ Bias Values are spesific to a set of inputs """
    if drgnfly.numInputNodes == 1:
        plt.scatter(inputs.X[:,0], bias_arr[:], c='k', label='Bias')
        plt.plot(np.unique(inputs.X[:,0]), np.poly1d(np.polyfit(inputs.X[:,0], bias_arr[:], 2))(np.unique(inputs.X[:,0])), c='k', lw=1, linestyle='--')
    else:
        logging.warning('Bias plot will not be generated if there is more than one input value.')
    plt.legend()
    plt.title('Paramater Weights vs Inputs')
    plt.xlabel('Input Value')
    plt.ylabel('Weight')
    plt.grid()
    plt.show()


    


def plotLoss(loss):
    epochs = np.arange(1, len(loss) + 1)
    plt.figure(figsize=(8, 6)) 
    plt.plot(epochs, loss, 'c')
    plt.title('Loss vs. Epochs', color='white')
    plt.xlabel('Epochs', color='white')
    plt.ylabel('Loss', color='white')
    plt.grid(color='white')
    plt.gca().spines['top'].set_color('white') 
    plt.gca().spines['right'].set_color('white')  
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white') 
    plt.gca().set_facecolor('black') 
    plt.gcf().set_facecolor('black')  
    plt.xticks(color='white')
    plt.yticks(color='white') 
    plt.tick_params(axis='both', colors='white')
    plt.show()
